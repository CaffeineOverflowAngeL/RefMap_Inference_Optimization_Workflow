import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tqdm import tqdm
import numpy as np
import pickle
import tensorflow as tf
import tf2onnx
import argparse
import engine.utils as utils
import onnx
import onnxruntime as ort
import torch

## Legacy KTH import dependencies
sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')

# Training utils
from training_utils import get_model, load_trained_model, get_epoch_times
from tfrecord_utils import get_dataset

#%% Configuration import
import config
#################################

from onnx2pytorch import ConvertModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

from evaluate_utils import get_mse_per_dim, get_mse_stats_per_dim

VARS_NAME_OUT = ('u','v','w')
N_VARS_OUT = 3

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--mode", type=str, required=True, choices=["onx", "pt", "pb"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--model-type", type=str, default='h5', choices=['hdf5', 'h5'])
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--debug-mode", action="store_true", default=False)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument('--output-dir', default='../converted_models/converter_logs', help='path where to save')

args = parser.parse_args()

def get_batches(features, batch_size):
    for start in range(0, len(features), batch_size):
        end = start + batch_size
        yield features[start:end]

def evaluate_onnx(onnx_path, dataset, batch_size):
    sess = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    
    input_name = sess.get_inputs()[0].name
    output_names = [output.name for output in sess.get_outputs()]
    print("Model Output Names:", output_names)
    
    all_predictions = []  # This will store the concatenated predictions for all batches
    total_batches = (len(dataset['features']) + batch_size - 1) // batch_size

    for batch_features in tqdm(get_batches(dataset['features'], batch_size), total=total_batches, desc="Generating ONNX outputs", mininterval=0.3):
        batch_features = batch_features.astype(np.float32)
        
        # Run prediction
        preds_onnx = sess.run(output_names, {input_name: batch_features})
        
        # Since each output is of shape (batch_size, 1, 192, 192), stack them along axis 1 to maintain the channel dimension correctly
        # This should correct the issue by ensuring we're not squeezing out necessary dimensions or concatenating incorrectly
        batch_preds = np.concatenate(preds_onnx, axis=1)  # Concatenate along the channel axis

        # Add this batch's predictions to the accumulator
        all_predictions.append(batch_preds)
    
    # Concatenate all batch outputs along the batch size dimension to get the final shape
    final_predictions = np.concatenate(all_predictions, axis=0)

    return final_predictions

def evaluate_pytorch(model_path, dataset, batch_size):
    # Load the model
    model = torch.load(model_path)
    # Check for GPU availability and move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    # Assuming dataset['features'] is a numpy array; convert it to a PyTorch tensor and move to the same device
    features = torch.tensor(dataset['features']).float().to(device)
    # If labels are needed for some reason, ensure they are also moved to the device
    # But here we're focusing on generating predictions, so labels might not be necessary
    
    # Create a DataLoader for the dataset
    # No need to move labels to device if we're only generating predictions
    loader = DataLoader(TensorDataset(features), batch_size=batch_size)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating PyTorch outputs", mininterval=0.3):
            feature = batch[0]
            pred_pytorch = model(feature)
            # Handle model output depending on its type (Tensor or list of Tensors)
            if isinstance(pred_pytorch, torch.Tensor):
                pred_pytorch = pred_pytorch.to('cpu').numpy()
            else:  # Assuming it's a list of Tensors
                pred_pytorch = [pred.to('cpu').numpy() for pred in pred_pytorch]
                # You may need to further process pred_pytorch if you need a specific shape or combination of these tensors
                # For simplicity, we're assuming you want to concatenate them along a new axis
                pred_pytorch = np.concatenate(pred_pytorch, axis=1)  # Adjust axis as necessary
            
            predictions.append(pred_pytorch)
    
    # Concatenate all batch outputs along the first axis
    predictions = np.concatenate(predictions, axis=0)

    return predictions

def check_mse_deviation(mse_onnx, mse_pytorch, threshold=1.0):
    """
    Checks if the deviation between MSE values of ONNX and PyTorch exceeds a given threshold.
    
    Parameters:
    - mse_onnx: dict, MSE values for ONNX predictions per dimension.
    - mse_pytorch: dict, MSE values for PyTorch predictions per dimension.
    - threshold: float, maximum allowed deviation percentage.
    
    Exits with failure if deviation exceeds the threshold for any dimension.
    """
    for key in mse_onnx:
        mse_onnx_val = mse_onnx[key]
        mse_pytorch_val = mse_pytorch[key]
        
        # Calculate the relative deviation in percentage
        deviation = abs(mse_onnx_val - mse_pytorch_val) / max(mse_onnx_val, mse_pytorch_val) * 100
        
        # Check if deviation exceeds the threshold
        if deviation > threshold:
            # Logging the deviation for debugging purposes
            print(f"Deviation for {key} exceeds threshold: {deviation:.2f}% > {threshold}%. ONNX MSE: {mse_onnx_val}, PyTorch MSE: {mse_pytorch_val}")
            exit("Failed due to excessive deviation in MSE values between ONNX and PyTorch predictions.")
        else: 
            print(f"Deviation for {key}: {deviation:.2f}% < {threshold}%. ONNX MSE: {mse_onnx_val}, PyTorch MSE: {mse_pytorch_val}")
def main():

    # Setup Log File 
    logger_name = "{}-{}-{}".format('model', args.mode, args.model.split('/')[-1])
    log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    args.logger = utils.get_logger(logger_name, output=log_file)

    # Defining saving format path
    args.onnx_save_path = args.model.split('/')[-1][:-3] + str('.onnx')
    args.pt_save_path = args.model.split('/')[-1][:-3] + str('.pth')
    args.pb_save_path = args.model.split('/')[-1][:-3]

    for k, v in utils.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    if args.model_type == 'h5':
        loaded_model = tf.keras.models.load_model(args.model)
    elif args.model_type == 'hdf5':
        ## Initialize Loading format ##
        os.environ['MODEL_CNN'] = 'WallRecon'
        prb_def = os.environ.get('MODEL_CNN', None)
        pred_path = None

        if not prb_def:
            print('"MODEL_CNN" enviroment variable not defined ("WallRecon" or "OuterRecon"), default value "WallRecon" is used')
            app = config.WallRecon
            prb_def = 'WallRecon'
        elif prb_def == 'WallRecon':
            app = config.WallRecon
        elif prb_def == 'OuterRecon':
            app = config.OuterRecon
        else:
            raise ValueError('"MODEL_CNN" enviroment variable must be defined either as "WallRecon" or "OuterRecon"')

        print(app)

        # =============================================================================
        #   IMPLEMENTATION WARNINGS
        # =============================================================================

        # Data augmentation not implemented in this model for now
        app.DATA_AUG = False
        # Transfer learning not implemented in this model for now
        app.TRANSFER_LEARNING = False

        app.BATCH_SIZE = args.batch_size

        on_GPU = app.ON_GPU
        n_gpus = app.N_GPU

        distributed_training = on_GPU == True and n_gpus>1

        ## Initialize model_config
        dataset_test, X_test, n_samples_tot, model_config, tfr_files_test = \
        get_dataset(prb_def, app, timestamp=app.TIMESTAMP, 
                    train=False, distributed_training=distributed_training,  tfr_files_bool=True)
        # In case of 'hdf5' format, please specify the exact path of the model on config.py file. 
        # Preparation for saving the results ------------------------------------------
        pred_path = app.CUR_PATH+'/.predictions/'
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
        # Pred Path :)
        pred_path = pred_path+model_config['name']+'/'
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)

        model_config['pred_path'] = pred_path

        loaded_model = load_trained_model(model_config)
    
        args.logger.info("Loading model: {} from {}".format(model_config['name'], args.model))
    else: 
        raise NotImplementedError("The Tensorflow Model type you have inserted is not supported. Current implementations are h5 and hdf5.")

    print(loaded_model.summary())

    # Convert the model to ONNX format
    onnx_model, _ = tf2onnx.convert.from_keras(loaded_model)

    #print("ONNX MODEL: ", onnx_model)
    onnx_PATH = '/home/angel/RefMap_Workflow/fcn_turbolence_prediction/converted_models/onnnx' + args.onnx_save_path
    onnx.save(onnx_model, onnx_PATH)

    if args.mode == 'pt':

        # Convert ONNX model to PyTorch
        pytorch_model = ConvertModel(onnx_model, batch_dim=args.batch_size, experimental=True)
        pt_PATH = '../converted_models/pt/' + args.pt_save_path
        torch.save(pytorch_model, pt_PATH)

        print(pytorch_model)

    if args.mode == 'pb':
        pb_PATH = '../converted_models/pb/' + args.pb_save_path
        loaded_model.save(pb_PATH)

    # Evaluate the converted models
    with open(args.dataset, 'rb') as handle:
        train_dataset = pickle.load(handle)
    
    if args.debug_mode:
        train_dataset['features'] = train_dataset['features'][:128]
        train_dataset['labels'] = train_dataset['labels'][:128]

    args.logger.info("Loaded dataset's shape is {}...".format(train_dataset['features'].shape))

    # After converting models
    predictions_onnx = evaluate_onnx(onnx_PATH, train_dataset, args.batch_size)

    #### Output Shapes Equality ##########
    args.logger.info("Labels shape is {}...".format(train_dataset['labels'].shape))
    args.logger.info("ONNX predictions shape is: {}...".format(predictions_onnx.shape))
    if args.mode == 'pt':
        predictions_pytorch = evaluate_pytorch(pt_PATH, train_dataset, args.batch_size)
        args.logger.info("Pytorch predictions shape is: {}...".format(predictions_pytorch.shape))

        # Assert that all shapes are equal
        assert train_dataset['labels'].shape == predictions_onnx.shape == predictions_pytorch.shape, "Shapes are not equal: Labels ({}), ONNX ({}), PyTorch ({})".format(train_dataset['labels'].shape, predictions_onnx.shape, predictions_pytorch.shape)

        ######################################

        ### Per output dimension deviation ###

        # MSE per dimension ONNX
        overall_mse_per_dimension_onnx = get_mse_per_dim(N_VARS_OUT, predictions_onnx, train_dataset['labels'], VARS_NAME_OUT)
        args.logger.info("ONNX format MSE evaluation for each output dimension is: {}".format(overall_mse_per_dimension_onnx))

        # MSE per dimension Pytorch
        overall_mse_per_dimension_pt = get_mse_per_dim(N_VARS_OUT, predictions_pytorch, train_dataset['labels'], VARS_NAME_OUT)
        args.logger.info("Pytorch format MSE evaluation for each output dimension is: {}".format(overall_mse_per_dimension_pt))

        # Check for deviations and potentially exit if deviation exceeds X%
        check_mse_deviation(overall_mse_per_dimension_onnx, overall_mse_per_dimension_pt)
        ######################################

        stats = get_mse_stats_per_dim(N_VARS_OUT, predictions_onnx, train_dataset['labels'], VARS_NAME_OUT)
        #args.logger.info("MSE stats per dimension:", stats)
        print(stats)

if __name__ == "__main__":
    main()
