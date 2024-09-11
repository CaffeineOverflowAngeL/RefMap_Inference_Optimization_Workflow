import numpy as np
import logging
import os, sys
import subprocess
import re

from termcolor import colored

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)

        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        
        return prefix + " " + log
    
def get_logger(name='train', output=None, color=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # STDOUT
    stdout_handler = logging.StreamHandler( stream=sys.stdout )
    stdout_handler.setLevel( logging.DEBUG )

    plain_formatter = logging.Formatter( 
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S" )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S")
    else:
        formatter = plain_formatter
    stdout_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)

    # FILE
    if output is not None:
        if output.endswith('.txt') or output.endswith('.log'):
            os.makedirs(os.path.dirname(output), exist_ok=True)
            filename = output
        else:
            os.makedirs(output, exist_ok=True)
            filename = os.path.join(output, "log.txt")
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(plain_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger

def flatten_dict(dic):
    flattned = dict()
    def _flatten(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                if prefix is None:
                    _flatten( k, v )
                else:
                    _flatten( prefix+'/%s'%k, v )
            else:
                if prefix is None:
                    flattned[k] = v
                else:
                    flattned[ prefix+'/%s'%k ] = v
        
    _flatten(None, dic)
    return flattned

def get_gpu_memory():
    # Run the nvidia-smi command
    # Query the GPU memory and name
    try:
        nvidia_smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during nvidia-smi execution: {e}")
        return []

    gpu_memory_map = []
    # Process each line
    for line in nvidia_smi_output.strip().split('\n'):
        gpu_id, total_memory, used_memory = line.split(', ')
        gpu_memory_map.append((int(gpu_id), int(used_memory), int(total_memory)))

    return gpu_memory_map

def get_dir_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# Function to calculate MSE
def calculate_mse(arr1, arr2):
    return ((arr1 - arr2) ** 2).mean()

def error_formula(pred, true):
    print(pred.shape)
    print(np.mean(np.abs(pred[0]-true[0])/true[0]))
    error_rate = np.mean(np.abs(pred-true)/true)
    print(error_rate[0])
    print()
    return np.mean(np.abs(pred-true))

def calculate_error(y_pred, y_test):
    """
    Calculate the non-dimensional error in the root mean square (RMS) of the streamwise velocity component.

    Parameters:
    y_pred (np.array): The predicted RMS values.
    y_test (np.array): The reference RMS values from DNS.

    Returns:
    float: The non-dimensional error.
    """
    # Calculate the RMS error
    rms_error = np.sqrt(np.mean((y_pred - y_test) ** 2))

    # Calculate the non-dimensional error (normalized by the RMS of y_test)
    non_dim_error = rms_error / np.sqrt(np.mean(y_test ** 2))

    return non_dim_error

def calculate_error_from_rms(y_pred_rms, y_true_rms):
    """
    Calculate the non-dimensional error using the RMS values of predictions and true values.

    Parameters:
    y_pred_rms (np.array): The predicted RMS values.
    y_true_rms (np.array): The reference RMS values from DNS.

    Returns:
    float: The non-dimensional error.
    """
    # Calculate the absolute difference in RMS values
    absolute_difference = np.abs(y_pred_rms - y_true_rms)

    # Calculate the non-dimensional error (normalized by the RMS of y_true)
    non_dim_error = absolute_difference / y_true_rms

    return non_dim_error


def evaluate_output_fields(fields_pred, fields_true, vars_out_names):
    error_rates = {}
    for idx, var_name in enumerate(vars_out_names):
        var_error_rate = calculate_error_from_rms(fields_pred[:, idx, :, :], fields_true[:, idx, :, :])
        error_rates[var_name] = var_error_rate
    return error_rates

def count_zero_weights(model, verbose=True):
    # Initialize counters
    zero_weights_count = 0
    total_weights_count = 0

    # Iterate through each layer
    for layer in model.layers:
        # Get the weights for the layer
        weights = layer.get_weights()
        
        # Iterate through each set of weights (e.g., kernel and bias)
        for weight_matrix in weights:
            # Count the total number of weights
            total_weights_count += np.size(weight_matrix)

            # Count the zeros in this weight matrix
            zero_weights_count += np.count_nonzero(weight_matrix == 0)

    if verbose:
        # Print the results
        print(f"Total number of weights in the model: {total_weights_count}")
        print(f"Number of zero weights in the model: {zero_weights_count}")
    
    return zero_weights_count, total_weights_count

