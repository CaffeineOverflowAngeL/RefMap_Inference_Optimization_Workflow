import numpy as np
import torch
import torch.nn as nn
import time
import gc
from tqdm import tqdm

def set_activations_hook(name, activations, sigmoid=False):
  def hook(model, input, output):
    if sigmoid:
      activations[name] = nn.Sigmoid()(output.detach())
    else:
      activations[name] = output.detach()

  return hook  # Call the hook function

def get_module_of_name(model, module_name):

  module = None
  for name, module in model.named_modules():
    if name == module_name: 
      return module
  
  if module == None:
    print("No module was found with this name.")

  return module

@torch.no_grad()
def get_activations_map(model, type_of_modules, activation_batch, sigmoid_boolean=False, remove_hooks=True):

  activations = {}
  handlers = []
  model.eval()

  for name, module in model.named_modules():
      # Select the type of modules to register the hook
    if any(isinstance(module, module_type) for module_type in type_of_modules):
      activations[name] = None
      # Debugging statements
      #print("Name: ", module)
      #print(set_activations_hook(name, activations, sigmoid=False))
      # Register the hooks
      handler = module.register_forward_hook(set_activations_hook(name, activations, sigmoid=sigmoid_boolean))
      handlers.append(handler)

  model(activation_batch)

  if remove_hooks:
    for handle in handlers:
      handle.remove()

  return activations

def hamming_distance_vector_old(x, y, device, normalize=True):
  """
  Computes the Hamming distance between two tensors of the same shape.
  """
  assert x.shape == y.shape, "x and y must have the same shape"

    #print("x: ",x)
    #print("y: ",y)
  #print(f"\t Insinde Hamming, inputs: {x}, {y}")
  # TODO: This formulation is according to relu activation, our module should be compatible to any given activation function
  x = (x > 0).float()  # convert all positive values to 1
  y = (y > 0).float()  # convert all positive values to 1

#  assert (x < 1).any() and (y < 1).any(), "There are values greater than 1."
  #print(f"\t Insinde Hamming, inputs relu norm: {x}, {y}")
    #print("x: ",x)
    #print("y: ",y)

    # Compute the element-wise XOR
  xor = torch.bitwise_xor(x.int(), y.int())
  #print(f"\t XOR: {xor}")

    # Distance per unit 
  dist = xor.view(x.size(0), -1).sum(1)
  #print("DIST SHAPE:", dist.shape)
  #print(print(f"\t DIST: {dist}"))

    # Normalize
  if normalize: 
    # Define Neurons per filter 
    #neurons_count_check = float(xor.shape[1]*xor.shape[2])
    neurons_count = float(xor.size(1) * xor.size(2))
    #print(neurons_count_check, neurons_count)

    # Convert cuda tensors to cpu tensors

    #if device.type == "cuda" and device.index == 0:
    if device < 0:
      dist = dist.cpu()

      # Apply the function to each element of the tensor
      #print(dist.float().apply_(lambda x: ((neurons_count) - x)/neurons_count))
      #normalized_dist = dist.float().apply_(lambda x: ((neurons_count) - x)/neurons_count) # Similarity - Non Expressivity
      device_dist = torch.device("cuda:"+str(device) if torch.cuda.is_available() and device>=0 else "cpu")
      normalized_dist = dist.float().apply_(lambda x: (x)/neurons_count).to(device_dist) # Expressivity
      #print(print(f"\t normalized_dist: {normalized_dist}"))
    else: # if > 0 tensors are located on the gpu

      # Calculate 'normalized_dist' using PyTorch operations
      normalized_dist = dist / neurons_count
      #print(print(f"\t normalized_dist_cpu: {normalized_dist}"))

    #print("Nomralized")
    #print(print(f"\t NORMALIZED DIST: {normalized_dist}"))
    return normalized_dist

    # Compute the number of ones (i.e., set bits) in the XOR tensor
  return dist

def hamming_distance_batch(batch_activations, device, normalize=True):
  """
  Computes the Hamming distance for a batch.
  """
  st = time.time()
  batch_hd_layer_activation_scores = {}

  #print(batch_activations)
  for layer_name, layer_batch_activations in batch_activations.items():
    #print(f"Layer: {layer_name}, Activation shape: {layer_batch_activations.shape}")
    batch_size = layer_batch_activations.shape[0]

    hamming_distance_map = np.zeros((batch_size, batch_size), dtype=object)
    #hamming_distance_map = [[0]*batch_size]*batch_size # Creating a 2D batch_size x batch_size array
    for i in range(batch_size):
      for j in range(batch_size):
        hamming_distance_map[i][j] = hamming_distance_vector_old(layer_batch_activations[i], layer_batch_activations[j], device=device, normalize=normalize)
        #print(hamming_distance_map[i][j])

    batch_hd_layer_activation_scores[layer_name] = hamming_distance_map
    #print(hamming_distance_map.shape)

  et = time.time()
  # get the execution time
  elapsed_time = et - st
  #print('Hamming Batch Execution time:', elapsed_time, 'seconds')
  return batch_hd_layer_activation_scores

def kernel_expressivity_scores(batch_hd_layer_activation_scores, device):
  st = time.time()
  batch_hd_layer_kernels_scores = {} 
  device_scores = torch.device("cuda:"+str(device) if torch.cuda.is_available() and device>=0 else "cpu")

  for layer_name, layer_batch_activation_scores in batch_hd_layer_activation_scores.items():
    #print(f"Layer: {layer_name}, Activation shape: {layer_batch_activation_scores.shape}")
    #print(layer_batch_activation_scores[0][0])
    kernels_amount = layer_batch_activation_scores[0][0].shape[0]
    batch_size = layer_batch_activation_scores.shape[0]
    #print(f"Layer: {layer_name}, Kernels Ammount: {kernels_amount}, Batch Size: {batch_size}")
    #print(kernels_amount)
    # Compute the mean vector
    if device < 0:
      mean_vector = np.zeros(kernels_amount)
    else: 
      # Initialize 'mean_vector' as a CUDA tensor filled with zeros
      mean_vector = torch.zeros(kernels_amount, device=device_scores)
      print(type(layer_batch_activation_scores))
      print("Device scores: ", device_scores)
      """
      if isinstance(layer_batch_activation_scores, np.ndarray):
        # Check the data type of the NumPy array elements
        if layer_batch_activation_scores.dtype == np.object:
            # Convert the array elements to a supported type, such as float32
            layer_batch_activation_scores = layer_batch_activation_scores.astype(np.float32)
        layer_batch_activation_scores = torch.tensor(layer_batch_activation_scores)
      """
      #layer_batch_activation_scores = layer_batch_activation_scores.to(device_scores)

    for i in range(batch_size):
      for j in range(batch_size):
          if i >= j: 
            continue
          else: 
            if device < 0:
              mean_vector = np.add(mean_vector, layer_batch_activation_scores[i][j])
            else:
              mean_vector += layer_batch_activation_scores[i][j]
            #print("Added: ", layer_batch_activation_scores[i][j][0])
            #print(mean_vector[0])
          

    mean_vector /= (((batch_size * batch_size)-batch_size)/2)
    #print("Diaireths: ", ((batch_size * batch_size)-batch_size)/2)
    #print("Final: ", mean_vector)
    
    # Assign the kernel scores (of shape (kernels, 1)) for the given layer
    batch_hd_layer_kernels_scores[layer_name] = mean_vector

  et = time.time()
  # get the execution time
  elapsed_time = et - st
  #print('Kernels Expressivity Execution time:', elapsed_time, 'seconds')

  return batch_hd_layer_kernels_scores

def kernel_expressivity_scores_mem_opt_old(layer_batch_activation_scores, device):
  device_scores = torch.device("cuda:"+str(device) if torch.cuda.is_available() and device>=0 else "cpu")

  kernels_amount = layer_batch_activation_scores.shape[2]
  batch_size = layer_batch_activation_scores.shape[0]
  # Compute the mean vector
  if device < 0:
    mean_vector = np.zeros(kernels_amount)
  else: 
    # Initialize 'mean_vector' as a CUDA tensor filled with zeros
    mean_vector = torch.zeros(kernels_amount, device=device_scores)
    layer_batch_activation_scores = layer_batch_activation_scores.to(device_scores)

  for i in tqdm(range(batch_size), desc="Expressivity Map"):
    for j in range(batch_size):
        if i >= j: 
          continue
        else: 
          if device < 0:
            mean_vector = np.add(mean_vector, layer_batch_activation_scores[i][j])
          else:
            mean_vector += layer_batch_activation_scores[i][j]
          #print("Added: ", layer_batch_activation_scores[i][j][0])
          #print(mean_vector[0])
        

  mean_vector /= (((batch_size * batch_size)-batch_size)/2)

  return mean_vector

def kernel_expressivity_scores_mem_opt(layer_batch_activation_scores, device):
    device_scores = torch.device("cuda:" + str(device) if torch.cuda.is_available() and device >= 0 else "cpu")
    kernels_amount = layer_batch_activation_scores.shape[2]
    batch_size = layer_batch_activation_scores.shape[0]
    
    # Transfer to the appropriate device
    layer_batch_activation_scores = layer_batch_activation_scores.to(device_scores)
    
    # Initialize 'mean_vector' as a tensor filled with zeros on the desired device
    mean_vector = torch.zeros(kernels_amount, device=device_scores)

    combinations_count = 0  # Count of valid (i, j) pairs
    for i in tqdm(range(batch_size), desc="Expressivity Map"):
        for j in range(i + 1, batch_size):  # Start j from i+1 to avoid redundant combinations and self-pairs
            mean_vector += layer_batch_activation_scores[i][j]
            combinations_count += 1

    mean_vector /= combinations_count

    return mean_vector

def hamming_distance_matrix_2(batch, normalize=True):
    # Convert all positive values to 1, non-positive to 0
    binarized_batch = (batch > 0).float()

    assert (binarized_batch <= 1).all(), "There are values greater than 1."

    # Compute the XOR for all pairs and then count the differing bits for each channel
    xor_result = torch.bitwise_xor(binarized_batch.unsqueeze(1).int(), binarized_batch.unsqueeze(0).int())
    hamming_dist = xor_result.sum(-1).sum(-1)  # summing over the last two dimensions (8x8)
    
    if normalize:
        normalization_factor = float(batch.size(-1) * batch.size(-2))
        hamming_dist = hamming_dist / normalization_factor

    return hamming_dist

def hamming_distance_batch_mem_opt(batch_activations, device, normalize=True, sub_batch_size=128):
    batch_hd_layer_activation_scores = {}

    for layer_name, layer_batch_activations in batch_activations.items():
        
        num_samples = layer_batch_activations.size(0)
        filter_dim = layer_batch_activations.size(1)  # Assuming the filter dimension is the third dimension
        
        # Initialize the matrix with an extra filter dimension
        final_distance_matrix = torch.zeros(num_samples, num_samples, filter_dim, device=device)

        # Iterate over sub-batches for the "rows" of the distance matrix
        #for i in range(0, num_samples, sub_batch_size):
        for i in tqdm(range(0, num_samples, sub_batch_size), desc="Processing:  "+layer_name):
            sub_batch_i = layer_batch_activations[i:i+sub_batch_size]
            # Iterate over sub-batches for the "columns" of the distance matrix
            for j in range(0, num_samples, sub_batch_size):
                sub_batch_j = layer_batch_activations[j:j+sub_batch_size]

                # Compute pairwise hamming distances for sub-batches
                sub_distance = compute_sub_batch_distance_mem_opt(sub_batch_i, sub_batch_j)

                # Assign computed distances into the final matrix
                final_distance_matrix[i:i+sub_batch_size, j:j+sub_batch_size, :] = sub_distance[:sub_batch_i.size(0), :sub_batch_j.size(0), :]

        # Normalize if required
        if normalize:
            normalization_factor = float(layer_batch_activations.size(-1) * layer_batch_activations.size(-2))
            final_distance_matrix = final_distance_matrix / normalization_factor

        batch_hd_layer_activation_scores[layer_name] = final_distance_matrix

    return batch_hd_layer_activation_scores

def compute_sub_batch_distance_mem_opt(sub_batch_i, sub_batch_j):
    sub_batch_i, sub_batch_j = sub_batch_i.to("cuda"), sub_batch_j.to("cuda")
    # Convert all positive values to 1, non-positive to 0 for both sub-batches
    binarized_i = (sub_batch_i > 0).float()
    binarized_j = (sub_batch_j > 0).float()

    assert (binarized_i <= 1).all(), "There are values greater than 1 in sub_batch_i."
    assert (binarized_j <= 1).all(), "There are values greater than 1 in sub_batch_j."

    # Compute the XOR for all pairs in sub_batch_i against all in sub_batch_j and then count the differing bits for each channel
    xor_result = torch.bitwise_xor(binarized_i.unsqueeze(1).int(), binarized_j.unsqueeze(0).int())
    hamming_dist = xor_result.sum(-1).sum(-1)  # summing over the last two dimensions

    del xor_result, sub_batch_i, sub_batch_j # Once merged, you may not need this list anymore
    gc.collect()
    hamming_dist.to("cpu")

    return hamming_dist

def hamming_distance_batch_2(batch_activations, device, normalize=True):

    batch_hd_layer_activation_scores = {}

    for layer_name, layer_batch_activations in batch_activations.items():
        #print(f"Layer: {layer_name}, Activation shape: {layer_batch_activations.shape}")

        # Compute pairwise distances using the optimized function
        distances = hamming_distance_matrix_2(layer_batch_activations)

        batch_hd_layer_activation_scores[layer_name] = distances # .cpu().numpy()


    return batch_hd_layer_activation_scores

def kernel_expressivity_scores_2(batch_hd_layer_activation_scores, device):

    batch_hd_layer_kernels_scores = {}
    device_scores = torch.device("cuda:" + str(device) if torch.cuda.is_available() and device >= 0 else "cpu")

    for layer_name, layer_batch_activation_scores in batch_hd_layer_activation_scores.items():
        # Convert to PyTorch tensor
        scores_tensor = torch.tensor(layer_batch_activation_scores, device=device_scores)
        
        # The number of combinations excluding same instance pairs is: (n*(n-1))/2
        denominator = (scores_tensor.size(0) * (scores_tensor.size(0) - 1)) / 2

        # Mask for the lower triangular excluding diagonal
        lower_tri_mask = torch.tril(torch.ones_like(scores_tensor), diagonal=-1)

        # Sum the entire tensor using the mask, then divide by the denominator
        mean_vector = (scores_tensor * lower_tri_mask).sum(dim=(0, 1)) / denominator
        
        batch_hd_layer_kernels_scores[layer_name] = mean_vector # .cpu().numpy()

    #print(batch_hd_layer_kernels_scores.keys())
    #print(batch_hd_layer_kernels_scores['conv1'].shape)
    #exit()
    return batch_hd_layer_kernels_scores


def overall_hamming_distance(batch_activations, normalize=True):
    total_score = 0
    num_layers = len(batch_activations)
    
    for _, layer_batch_activations in batch_activations.items():
        # Compute pairwise Hamming distances for the current layer
        distances = hamming_distance_matrix_2(layer_batch_activations, normalize)

        # Average the distances over the entire batch
        avg_distance = distances.mean()

        # Accumulate the average distance
        total_score += avg_distance
    
    # Compute the overall normalized Hamming distance by averaging over all layers
    overall_score = total_score / num_layers
    
    return overall_score
