import sys, os

# Adjasting the visibility scope of the script to reach the torch_pruning package.
# Get the directory of the current file (__file__ is the path to benchmark.py)
current_dir = os.path.dirname(os.path.realpath(__file__))
# Go up two levels to the grandparent directory
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))
print("Directory Scope during runtime: ", grandparent_dir)

# Add the grandparent directory to sys.path to find torch_pruning
sys.path.append(grandparent_dir)

from functools import partial
import argparse
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import engine.utils as utils
import torch_pruning as tp
#import registry

VARS_NAME_OUT = ('u','v','w')

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "prune", "test"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--yp", type=str, default="yp15", choices=['yp15', 'yp30', 'yp50', 'yp100'])
parser.add_argument("--ret", type=str, default="180", choices=['180', '550'])
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--total-epochs", type=int, default=25)
parser.add_argument("--lr-decay-milestones", default="10,20", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--restore", type=str, default=None)
parser.add_argument('--output-dir', default='./model_compression/benchmarks/run', help='path where to save')

# For prepruning
parser.add_argument("--preprune", action="store_true", default=False)

# For pruning
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--speed-up", type=float, default=2)
parser.add_argument("--max-sparsity", type=float, default=1.0)
parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
parser.add_argument("--reg", type=float, default=5e-4)
parser.add_argument("--delta_reg", type=float, default=1e-4, help='for growing regularization')
parser.add_argument("--weight-decay", type=float, default=5e-4)

parser.add_argument("--ovrl-exp-logger", action="store_true", default=False)

# For Alternate pruning
parser.add_argument("--alternate-pruning", action="store_true", default=False)
parser.add_argument("--al-lr", default=0.01, type=float, help="learning rate for alternate pruning training")

# For Hybrid Pruning
parser.add_argument("--hybrid-pruning", action="store_true", default=False)
parser.add_argument("--hb-exp-ratio", default=0.5, type=float, help="the weight of the expressiveness in the pruning decisions")
parser.add_argument("--hb-imp-ratio", default=0.5, type=float, help="the weight of the importance metric in the pruning decisions")

# General
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=False)
parser.add_argument("--sl-total-epochs", type=int, default=5, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
parser.add_argument("--sl-restore", type=str, default=None)
parser.add_argument("--iterative-steps", default=400, type=int)
parser.add_argument("--scheduler", type=str, default="MultiStep")

# For pre-training 
parser.add_argument("--reset-trained-weights", action="store_true", default=False)

args = parser.parse_args()

class NumpyDictDataset(Dataset):
    def __init__(self, data_dict):
        self.features = data_dict['features']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'labels': self.labels[idx]}
    
def merge_datasets_from_folder(folder_path):
    merged_data = {'features': [], 'labels': []}
    
    for filename in os.listdir(folder_path):
        print("Loading data from: ", filename)
        if filename.endswith('.pkl'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as handle:
                data_dict = pickle.load(handle)
                merged_data['features'].extend(data_dict['features'])
                merged_data['labels'].extend(data_dict['labels'])
            print(f"Current Samples no.{len(merged_data['labels'])}")
    
    print(f"Datasets have been merged. Samples no.{len(merged_data['labels'])}")
    return merged_data

def save_output_image(output, sample, save_dir, yp, type_m='Baseline'):
    

    output_tensor = torch.cat(output, dim=1)
    print("Predicted Fields shape: ", output_tensor.shape)

    # Convert output and original sample to numpy arrays
    output_np = output_tensor.cpu().detach().numpy()
    sample_np = sample['labels'].cpu().detach().numpy()
    print("Labels Fields shape: ", sample_np.shape)
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Plot and save the 3D output image
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(output_np[0][0], output_np[0][1], output_np[0][2])
    ax.set_title('Predicted Fields (Outputs)')
    plt.savefig(os.path.join(save_dir, 'output_fields_'+type_m+'_'+str(yp)+'.png'))
    plt.close(fig)
    
    # Plot and save the original sample image
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(sample_np[0][0], sample_np[0][1], sample_np[0][2])
    ax.set_title('Original Fields (Labels)')
    plt.savefig(os.path.join(save_dir, 'original_fields_'+type_m+'_'+str(yp)+'.png'))
    plt.close(fig)

def select_random_sample(data_loader):
    """
    Selects a random sample from the given DataLoader.

    Args:
    - data_loader: torch.utils.data.DataLoader object containing the dataset.

    Returns:
    - sample: A random sample from the DataLoader.
    """
    try:
        # Convert data_loader to an iterable
        data_iter = iter(data_loader)

        # Get a random index within the range of the dataset size
        random_index = random.randint(0, len(data_loader.dataset) - 1)

        # Move to the random index in the DataLoader iterator
        for _ in range(random_index + 1):
            sample = next(data_iter)

    except StopIteration:
        # If the iterator reaches the end of the dataset, reset it
        data_iter = iter(data_loader)
        sample = next(data_iter)

    return sample

def progressive_pruning(pruner, model, speed_up, example_inputs):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        print(f"Current Step {pruner.current_step}: {current_speed_up}")
        #print("Pruning History: ", pruner.pruning_history())
        if pruner.current_step == pruner.iterative_steps:
            break
        #print(current_speed_up)
    return current_speed_up

def progressive_pruning_expressivity_logger(pruner, model, speed_up, example_inputs, activation_batch, test_loader, device):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        current_expressivity = tp.utils.overall_expressiveness(model, activation_batch)
        #print(f"Current Step {pruner.current_step}: {current_speed_up}")
        #print(f"Current Expressiveness: {current_expressivity}")
        # TODO: ADD ACCURACY CALCULATOR
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info(
            "Pruning Iteration {:d}, Acc={:.4f}, Val Loss={:.4f}, Expressivity={:.4f}, Current Speed-up={:.4f}".format(
                pruner.current_step, acc, val_loss, current_expressivity, current_speed_up
            )
        )
        #print("Pruning History: ", pruner.pruning_history())
        if pruner.current_step == pruner.iterative_steps:
            break
        #print(current_speed_up)
    return current_speed_up

def progressive_alternate_pruning_expressivity_logger(pruner, model, speed_up, example_inputs, activation_batch, train_loader, test_loader, device):
    model.eval()
    base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    lr_schedule = tp.utils.schedulers.alterante_cosine_lr_schedule(init_lr=args.al_lr, min_lr=args.al_lr/10, total_steps=args.iterative_steps)
    while current_speed_up < speed_up:
        pruner.step(interactive=False)
        one_step_train_model(model=model, train_loader=train_loader, lr=lr_schedule[pruner.current_step-1], device=device)
        pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
        current_expressivity = tp.utils.overall_expressiveness(model, activation_batch)
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info(
            "Pruning Iteration {:d}, Acc={:.4f}, Val Loss={:.4f}, Expressivity={:.4f}, Current Speed-up={:.4f}, Pruned Ops={:.2f} M, Current LR={:.4f}".format(
                pruner.current_step, acc, val_loss, current_expressivity, current_speed_up, pruned_ops / 1e6, lr_schedule[pruner.current_step-1],
            )
        )
        #print("Pruning History: ", pruner.pruning_history())
        if pruner.current_step == pruner.iterative_steps:
            break
        #print(current_speed_up)
    return current_speed_up

def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()

def inference_latency_eval(model, input_size, device=None):
    model.to(device)
    dummy_input = torch.randn(input_size, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    return mean_syn, std_syn

def inference_throughput_eval(model, input_size, device=None):
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    optimal_batch_size = 16 * 4 # CIFAR10 RUNS OUT OF MEMORY AT 2048*32
    model.to(device)
    dummy_input = torch.randn(optimal_batch_size, input_size[1], input_size[2], input_size[3], dtype=torch.float).to(
        device)

    repetitions = 20 # 100
    total_time = 0
    with torch.no_grad(), tqdm(total=repetitions, desc='Inference Time') as pbar:
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
            pbar.update(1)

    throughput = (repetitions * optimal_batch_size) / total_time
    return throughput

def one_step_train_model(
    model,
    train_loader,
    lr,
    device=None,        
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model for 1 epoch
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

def train_regression_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    lr_decay_milestones,
    lr_decay_gamma=0.1,
    save_as=None,
    scheduler="MultiStep",
    weight_decay=1e-5,
    device=None,
    save_state_dict_only=True,
    pruner=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    if scheduler == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_decay_gamma
        )
    elif scheduler == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0001 
        )

    model.to(device)
    best_mse = float('inf')
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # TODO: ADD PER DIMENSION EVALUATION
    
        for i, batch in enumerate(train_loader):
            data, target = batch['features'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            out = model(data.float())
            #print("Type of out: ", type(out))
            #print("Type of out item: ", type(out[0]))
            #print("Out Item: ", out[0].shape)
            
            # Convert list of tensor outputs to a single numpy array
            out = torch.cat(out, dim=1) #.detach().cpu().numpy()
            
            # Adjust target tensor if necessary
            #target = target[:, 0, :, :].unsqueeze(1)  # Assuming you only need the first output from the target
            #print("Target Shape: ", target.shape)
            #print("Output Shape: ", out.shape)
            loss = criterion(out, target.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, Iter {i}/{len(train_loader)}, Loss={loss.item():.4f}, LR={optimizer.param_groups[0]['lr']:.4f}"
                )

        model.eval()
        val_loss, val_loss_comps = evaluate_regression(model, test_loader, device)
        print(
            f"Epoch {epoch}/{epochs}, Validation Loss={val_loss:.4f}, {VARS_NAME_OUT[0]}={val_loss_comps[0]:.4f},  {VARS_NAME_OUT[1]}={val_loss_comps[1]:.4f}, {VARS_NAME_OUT[2]}={val_loss_comps[2]:.4f}, LR={optimizer.param_groups[0]['lr']:.4f}"
        )

        if val_loss < best_mse:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "prune":
                print("SAVE AS: ", save_as)
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}_{}_reg.pth".format(args.ret, args.yp, args.model.split('/')[-1]) )
                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}_{}.pth".format(args.ret, args.yp, args.model.split('/')[-1]) )
                torch.save(model.state_dict(), save_as)
            best_mse = val_loss
            if save_as is not None:
                torch.save(model.state_dict(), save_as)

        scheduler.step()
    print("Model was saved at: ", save_as)
    print("Best Validation MSE:", best_mse)

    return save_as

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def evaluate_regression(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    losses = [0.0, 0.0, 0.0]  # For storing losses for each output dimension
    with torch.no_grad():
        for batch in dataloader:
            data, target = batch['features'].to(device), batch['labels'].to(device)
            output = model(data.float())
            output_cat = torch.cat(output, dim=1)
            loss = criterion(output_cat, target.float())  # Assuming target is continuous
            total_loss += loss.item() * data.size(0)
            
            # Compute loss for each output dimension
            for i in range(len(output)):
                loss_i = criterion(output[i], target[:, i].float().unsqueeze(1))
                losses[i] += loss_i.item() * data.size(0)
                
    mean_loss = total_loss / len(dataloader.dataset)
    mean_losses = [loss / len(dataloader.dataset) for loss in losses]
    return mean_loss, mean_losses

def get_pruner(model, example_inputs, activation_batch):
    args.sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        if args.hybrid_pruning:
            pruner_entry = partial(tp.pruner.HybridMagnitudePruner, global_pruning=args.global_pruning)
        else:
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        if args.hybrid_pruning:
            pruner_entry = partial(tp.pruner.HybridMagnitudePruner, global_pruning=args.global_pruning)
        else:
            pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        if args.hybrid_pruning:
            pruner_entry = partial(tp.pruner.HybridBNScalePruner, global_pruning=args.global_pruning)
        else:
            pruner_entry = partial(tp.pruner.BNScalePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        imp = tp.importance.BNScaleImportance()
        if args.hybrid_pruning:
            args.sparsity_learning = False
            pruner_entry = partial(tp.pruner.HybridBNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
        else:
            args.sparsity_learning = True
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_slim":
        imp = tp.importance.BNScaleImportance()
        if args.hybrid_pruning:
            args.sparsity_learning = False
            pruner_entry = partial(tp.pruner.HybridBNScalePruner, reg=args.reg, global_pruning=args.global_pruning, group_lasso=True)
        else:
            args.sparsity_learning = True
            pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning, group_lasso=True)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        if args.hybrid_pruning:
            pruner_entry = partial(tp.pruner.HybridGroupNormPruner, global_pruning=args.global_pruning)
        else:
            pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
        if args.hybrid_pruning:
            args.sparsity_learning = False
            pruner_entry = partial(tp.pruner.HybridGroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
        else:
            args.sparsity_learning = True
            pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "growing_reg":
        imp = tp.importance.GroupNormImportance(p=2)
        if args.hybrid_pruning:
            args.sparsity_learning = False
            pruner_entry = partial(tp.pruner.HybridGrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
        else:
            args.sparsity_learning = True
            pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
    #Our Methods

    # TODO: BUILD EXPRESSIVE PRUNER
    # 1. Adjust the compatibility to also recieve an activation batch in order to calculate the activations map in each iteration
    # 2. Extent Expressive Pruner to recieve the activations map and calculate the expressivity of all elements (begin with kernels expressivity)
    # 3. Make sure that the ExpressiveImportance and the ExpressivePruner are alligned and communicate well
    # 4. Refactor the main functionalities of the ExpressivePruner included in the step() operation, where they define the actions (local_pruning and global_pruning) taken at each pruning iteration.
    # 5. Refactor the optimization loop at main.py based on my todo_notes of the last meeting. 

    elif args.method == "expressiveness":
        args.sparsity_learning = False
        if args.model == 'resnet56':
            imp = tp.importance.ExpressiveImportance(target_types=[nn.modules.conv._ConvNd, nn.Linear])
        else:
            imp = tp.importance.ExpressiveImportance()
        pruner_entry = partial(tp.pruner.ExpressivePruner, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    ignored_layers = []
    ch_sparsity_dict = {}
    # ignore output layers
    # TODO: FIX!
    """
    for m in model.modules():
        print(m)
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
            ignored_layers.append(m)
    """
    # Expressiveness Module is built upon pruneinator which requires an additional argument "activation_batch" during initialization, so in order to not change metapruner base class, 
    # we divide the init calls.

    if args.hybrid_pruning:
        if args.model == 'resnet56':
            expressiveness=tp.importance.ExpressiveImportance(target_types=[nn.modules.conv._ConvNd, nn.Linear])
        else: 
            expressiveness=tp.importance.ExpressiveImportance()
        pruner = pruner_entry(
                model,
                example_inputs,
                activation_batch=activation_batch,
                importance=imp,
                expressiveness=expressiveness,
                iterative_steps=args.iterative_steps,
                ch_sparsity=1.0,
                ch_sparsity_dict=ch_sparsity_dict,
                max_ch_sparsity=args.max_sparsity,
                ignored_layers=ignored_layers,
                unwrapped_parameters=unwrapped_parameters,
                exp_component=args.hb_exp_ratio,
                imp_component=args.hb_imp_ratio,
            )
    else:
        if args.method == "expressiveness":

        # Here we fix iterative_steps=200 to prune the model progressively with small steps 
        # until the required speed up is achieved.
            pruner = pruner_entry(
                model,
                example_inputs,
                activation_batch=activation_batch,
                importance=imp,
                iterative_steps=args.iterative_steps,
                ch_sparsity=1.0,
                ch_sparsity_dict=ch_sparsity_dict,
                max_ch_sparsity=args.max_sparsity,
                ignored_layers=ignored_layers,
                unwrapped_parameters=unwrapped_parameters,
            )
        else: 
            # Here we fix iterative_steps=200 to prune the model progressively with small steps 
            # until the required speed up is achieved.
            pruner = pruner_entry(
                model,
                example_inputs,
                importance=imp,
                iterative_steps=args.iterative_steps,
                ch_sparsity=1.0,
                ch_sparsity_dict=ch_sparsity_dict,
                max_ch_sparsity=args.max_sparsity,
                ignored_layers=ignored_layers,
                unwrapped_parameters=unwrapped_parameters,
            )

    return pruner


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Logger
    if args.mode == "prune":
        prefix = 'global' if args.global_pruning else 'local'
        model_name = args.model.split('/')[-1].split('.p')[0]
        logger_name = "{}-{}-{}".format(prefix, args.method, model_name)
        args.logger_name = logger_name
        if args.preprune:
            args.output_dir = os.path.join(args.output_dir, args.mode, "preprune")
        elif args.hybrid_pruning:
            if args.preprune:
                args.output_dir = os.path.join(args.output_dir, args.mode, "preprune")
            else:
                args.output_dir = os.path.join(args.output_dir, args.mode, "hybrid_pruning")
        else:
            args.output_dir = os.path.join(args.output_dir, args.mode)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "pretrain":
        args.output_dir = os.path.join(args.output_dir, args.mode)
        logger_name = "{}-{}-{}-{}".format(args.mode, args.ret, args.yp, args.model.split('/')[-1])
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "test":
        log_file = None
        logger_name = "{}-{}-{}-{}".format(args.mode, args.ret, args.yp, args.model.split('/')[-1]) # ADDED LINE
    args.logger = utils.get_logger(logger_name, output=log_file)

    # Define Device 
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.model)
    print(model)
    # Model & Dataset
    #num_classes, train_dst, val_dst, input_size = registry.get_dataset(
    #    args.dataset, data_root="data"
    #)
    #args.num_classes = num_classes
    #model = registry.get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset)
    # Unpickle Dataset

    with open(args.dataset, 'rb') as handle:
        train_dataset = pickle.load(handle)

    train_dst = NumpyDictDataset(train_dataset)
    val_dst = NumpyDictDataset(train_dataset)

    # Load Data 
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    
    # FIX TEST LOADER TO PARSE THE CORRECT DATA
    #test_loader = torch.utils.data.DataLoader(
        #val_dst, batch_size=args.batch_size, num_workers=4
    #)

    #del train_dst, val_dst

    random_sample = select_random_sample(train_loader)
    random_sample_input = random_sample['features'].to(args.device)
    print(random_sample['features'].shape)
    print(random_sample['labels'].shape)
    print(type(random_sample))
    
    fields_save_dir = '../fields_comparison'

    results_df = pd.DataFrame(columns=["compression_ratio",
                                        "speed_up",                     
                                        "Params (M)", 
                                        "Pruned Params (M)", 
                                        "FLOPs (M)", 
                                        "Pruned FLOPs (M)", 
                                        "Overall MSE", 
                                        "Pruned Overall MSE",
                                        "Components MSE",
                                        "Pruned Components MSE",
                                        "Inference Mean",
                                        "Pruned Inference Mean",
                                        "Inference STD",
                                        "Pruned Inference STD",
                                        "Throughput",
                                        "Pruned Throughput"])
    
    print("Pytorch Version:", torch.__version__)

    args.image_extraction_sample = True

    for k, v in utils.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    if args.restore is not None:
        loaded = torch.load(args.restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))
    model = model.to(args.device)

    output = model(random_sample_input.float())

    if args.image_extraction_sample:
        save_output_image(output, random_sample, fields_save_dir, yp = args.yp)

    # Set Activation Batch
    activation_batch = None
    if args.method == "expressiveness" or args.ovrl_exp_logger or args.hybrid_pruning:
        # TODO: CHANGE BACK TO activation_batch_old
        #activation_batch_old=next(iter(train_loader))[0].to(args.device) # index 0 is used to only retrieve the inputs (x) and not the labels (y).
        #print(activation_batch_old.shape)
        # Shape of the tensor
        tensor_shape = (32, 3, 32, 32)
        # Generate the random tensor
        # Load the tensors
        #loaded_activation_batch = torch.load('/home/angel/Meta-Pruning/Meta-Pruning/benchmarks/activation_batches/cifar10_batch_6.pth')
        loaded_activation_batch = torch.load('/home/angel/Meta-Pruning/Meta-Pruning/benchmarks/activation_batches/cifar10_batch.pth')
        activation_batch = loaded_activation_batch['data'].to(args.device) 
        batch_labels_loaded = loaded_activation_batch['labels']
        # TODO: REMOVE 
        #activation_batch = torch.rand(tensor_shape).to(args.device)
        #entropy_old = utils.calculate_entropy(activation_batch_old.cpu())
        entropy = utils.calculate_entropy(activation_batch.cpu())
        #print(f"Random Entropy: {entropy_old}")
        print(f"Fixed Entropy: {entropy}")


    # TODO: Investigate if it is more optimal to have predefined batches of a certain paradigm
    # for example one batch that includes one sample from each class, instead of a random batch generator
    # How to do: Add an arg and make an if-elif-else statement


    ######################################################
    # Training / Pruning / Testing
    #example_inputs = train_dst[0][0].unsqueeze(0).to(args.device)
    #example_inputs = train_dst[0]['features']
    example_inputs = torch.tensor(train_dst[:1]['features'], dtype=torch.float32).to(args.device)
    input_size = torch.tensor(train_dst[:1]['features'], dtype=torch.float32).shape
    print("Input Size: ", input_size)
    
    if args.mode == "pretrain":
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))

        # Reset Model Weights # 
        if args.reset_trained_weights: 
            model.apply(weight_reset)

        train_regression_model(
            model=model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=train_loader, # TODO: CHANGE TO TEST LOADER
            scheduler=args.scheduler
        )
    elif args.mode == "prune":
        #flop_ratios_list = [1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 5.50, 6, 7, 8, 9, 10, 15, 20]
        #flop_ratios_list = [1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00]
        flop_ratios_list = [25.00]


        for i, compression_ratio in enumerate(flop_ratios_list):

            args.speed_up = compression_ratio
            model = torch.load(args.model)
            model = model.to(args.device)
            args.logger.info("Iteration {}/{}: Target Compression Ratio {}".format(i+1, len(flop_ratios_list), compression_ratio))
            pruner = get_pruner(model, example_inputs=example_inputs, activation_batch=activation_batch)

            # 0. Sparsity Learning
            if args.sparsity_learning:
                reg_pth = "reg_{}_{}_{}_{}.pth".format(args.ret, args.yp, args.model.split('/')[-1], args.reg)
                reg_pth = os.path.join( os.path.join(args.output_dir, reg_pth) )
                if not args.sl_restore:
                    args.logger.info("Regularizing...")
                    train_regression_model(
                        model,
                        train_loader=train_loader,
                        test_loader=train_loader, # TODO: CHANGE TO TEST LOADER
                        epochs=args.sl_total_epochs,
                        lr=args.sl_lr,
                        lr_decay_milestones=args.sl_lr_decay_milestones,
                        lr_decay_gamma=args.lr_decay_gamma,
                        pruner=pruner,
                        save_state_dict_only=True,
                        save_as = reg_pth,
                    )
                args.logger.info("Loading the sparse model from {}...".format(reg_pth))
                model.load_state_dict( torch.load( reg_pth, map_location=args.device) )
        
            # 1. Pruning
            model.eval()
            ori_ops, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            ori_mse, ori_mse_components = evaluate_regression(model, train_loader, device=args.device) # TODO: CHANGE TO TEST LOADER
            mean_sys, std_sys = inference_latency_eval(model, input_size, device=args.device)
            args.logger.info("Latency Mean: {:.4f} Latency STD: {:.4f}".format(mean_sys, std_sys))
            #throughput = inference_throughput_eval(model, input_size, device=args.device)
            #args.logger.info("Throughput: {:.4f} (Samples per Second)\n".format(throughput))
            args.logger.info("Pruning...")
            if args.alternate_pruning:
                args.logger.info("Alternate Pruning {}...".format(args.alternate_pruning))
                args.logger.info("Expressivity Logger {}...".format(args.ovrl_exp_logger))
                progressive_alternate_pruning_expressivity_logger(pruner, model, speed_up=args.speed_up, example_inputs=example_inputs, activation_batch=activation_batch, train_loader=train_loader, test_loader=train_loader, device=args.device) # TODO: CHANGE TO TEST LOADER
            else:
                args.logger.info("Alternate Pruning {}...".format(args.alternate_pruning))
                if args.ovrl_exp_logger:
                    args.logger.info("Expressivity Logger {}...".format(args.ovrl_exp_logger))
                    progressive_pruning_expressivity_logger(pruner, model, speed_up=args.speed_up, example_inputs=example_inputs, activation_batch=activation_batch, test_loader=train_loader, device=args.device) # TODO: CHANGE TO TEST LOADER
                else: 
                    args.logger.info("Expressivity Logger {}...".format(args.ovrl_exp_logger))
                    progressive_pruning(pruner, model, speed_up=args.speed_up, example_inputs=example_inputs)
            del pruner # remove reference
            args.logger.info(model)
            pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
            pruned_mse, pruned_mse_components = evaluate_regression(model, train_loader, device=args.device) # TODO: CHANGE TO TEST LOADER
            
            args.logger.info(
                "Params: {:.2f} M => {:.2f} M ({:.2f}, {:.2f}X%)".format(
                    ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100, ori_size / pruned_size,
                )
            )
            args.logger.info(
                "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                    ori_ops / 1e6,
                    pruned_ops / 1e6,
                    pruned_ops / ori_ops * 100,
                    ori_ops / pruned_ops,
                )
            )
            args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_mse, pruned_mse))
            args.logger.info("Acc: {} => {}".format(ori_mse_components, pruned_mse_components))
            #args.logger.info(
                #"Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
            #)
            pruned_mean_sys, pruned_std_sys = inference_latency_eval(model, input_size, device=args.device)
            args.logger.info("Latency Mean: {:.4f} Latency STD: {:.4f}".format(pruned_mean_sys, pruned_std_sys))
            
            # 2. Finetuning
            args.logger.info("Finetuning...")
            print("Finetuning Device: ", args.device)
            finetuned_model_pth = train_regression_model(
                model,
                epochs=args.total_epochs,
                lr=args.lr,
                lr_decay_milestones=args.lr_decay_milestones,
                train_loader=train_loader,
                test_loader=train_loader, # TODO: CHANGE TO TEST LOADER
                device=args.device,
                save_state_dict_only=True,
                scheduler=args.scheduler
            )

            model.load_state_dict( torch.load( finetuned_model_pth, map_location=args.device) )
            print(model)

            # 3. Testing
            model.eval()
            finetuned_mse, finetuned_mse_components = evaluate_regression(model, train_loader, device=args.device) # TODO: CHANGE TO TEST LOADER
            #pruned_throughput = inference_throughput_eval(model, input_size, device=args.device)
            #args.logger.info("Throughput: {:.4f} (Samples per Second)\n".format(pruned_throughput))

            new_row = {
                    "compression_ratio": (ori_size / pruned_size),
                    "speed_up": (ori_ops / pruned_ops),  
                    "Params (M)": (ori_size / 1e6), 
                    "Pruned Params (M)": (pruned_size / 1e6), 
                    "FLOPs (M)": (ori_ops / 1e6), 
                    "Pruned FLOPs (M)": (pruned_ops / 1e6), 
                    "Overall MSE": ori_mse, 
                    "Pruned Overall MSE": pruned_mse,
                    "Finetuned Overall MSE": finetuned_mse,
                    "Components MSE": ori_mse_components,
                    "Pruned Components MSE": pruned_mse_components,
                    "Finetuned Components MSE": finetuned_mse_components,
                    "Inference Mean": mean_sys,
                    "Pruned Inference Mean": pruned_mean_sys,
                    "Inference STD": std_sys,
                    "Pruned Inference STD": pruned_std_sys,
                    #"Throughput": throughput,
                    #"Pruned Throughput": pruned_throughput

                }
            args.logger.info(new_row)
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
            # Save performance DataFrame as CSV
            scope = 'global' if args.global_pruning else 'local'
            results_df.to_csv('./model_compression/Benchmark_Results/'+args.method+'_'+args.ret+'_'+args.yp+'_'+str(args.total_epochs)+'_'+scope+'.csv', index=False)
            
            pruned_output = model(random_sample_input.float())

            if args.image_extraction_sample:
                save_output_image(pruned_output, random_sample, fields_save_dir, yp = args.yp, type_m='Pruned')
            del model
        
        args.logger.info(results_df)

    elif args.mode == "test":
        model.eval()
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        test_mse, test_components_mse = evaluate_regression(model, train_loader, device=args.device) # TODO: CHANGE TO TEST LOADER
        args.logger.info("Overall MSE: {:.4f}".format(test_mse))
        args.logger.info("Components MSE: {}".format(test_components_mse))
        mean_sys, std_sys = inference_latency_eval(model, input_size, device=args.device)
        args.logger.info("Latency Mean: {:.4f} Latency STD: {:.4f}".format(mean_sys, std_sys))
        print(input_size[1:])
        throughput = inference_throughput_eval(model, input_size, device=args.device)
        args.logger.info("Throughput: {:.4f} (Samples per Second)\n".format(throughput))

if __name__ == "__main__":
    main()
