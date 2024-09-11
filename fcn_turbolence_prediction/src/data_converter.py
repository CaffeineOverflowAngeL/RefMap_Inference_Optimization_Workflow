#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import tensorflow as tf
import sys 
import pickle

sys.path.insert(0, '../conf')
sys.path.insert(0, '../models')

# Training utils
from training_utils import get_model
from tfrecord_utils import get_dataset

#%% Configuration import
import config

prb_def = os.environ.get('MODEL_CNN', None)

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

os.environ["CUDA_VISIBLE_DEVICES"]=str(app.WHICH_GPU_TRAIN);
#os.environ["CUDA_VISIBLE_DEVICES"]="";

# =============================================================================
#   IMPLEMENTATION WARNINGS
# =============================================================================

# Data augmentation not implemented in this model for now
app.DATA_AUG = False
# Transfer learning not implemented in this model for now
app.TRANSFER_LEARNING = False

#%% Hardware detection and parallelization strategy
physical_devices = tf.config.list_physical_devices('GPU')
availale_GPUs = len(physical_devices) 
print('Using TensorFlow version:', tf.__version__, ', GPU:', availale_GPUs)
#print(keras.__version__)

if physical_devices:
  try:
    for gpu in physical_devices:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

on_GPU = app.ON_GPU
n_gpus = app.N_GPU

distributed_training = on_GPU == True and n_gpus>1

#%% Dataset and ANN model

tstamp = int(time.time())

dataset_train, dataset_val, n_samp_train, n_samp_valid, model_config, tfr_files_train = \
        get_dataset(prb_def, app, timestamp=tstamp, 
                    train=True, distributed_training=distributed_training, tfr_files_bool=True)

CNN_model, callbacks = get_model(model_config)

print('')
print('# ====================================================================')
print('#     Summary of the options for the model                            ')
print('# ====================================================================')
print('')
print(f"Model name: {model_config['name']}")
print(f'Number of samples for training: {int(n_samp_train)}')
print(f'Number of samples for validation: {int(n_samp_valid)}')
print(f'Total number of samples: {int(n_samp_train+n_samp_valid)}')
print(f"Batch size: {model_config['batch_size']}")
print('')
print(f'Data augmentation: {app.DATA_AUG} (not implemented in this model)')
print(f'Initial distribution of parameters: {app.INIT}')
if app.INIT == 'random':
    print('')
    print('')
if app.INIT == 'model':
    print(f'    Timestamp: {app.INIT_MODEL[-10]}')
    print(f'    Transfer learning: {app.TRANSFER_LEARNING} (not implemented in this model)')
print(f'Prediction of fluctuation only: {app.FLUCTUATIONS_PRED}')
print(f'y- and z-output scaling with the ratio of RMS values : {app.SCALE_OUTPUT}')
print(f'Normalized input: {app.NORMALIZE_INPUT}')
print('')
print('# ====================================================================')

print("Type of Dataset: ", type(dataset_train))

# Initialize lists to store the numpy arrays of features and labels
#features = []
#labels = []

# Iterate through the dataset
size = app.N_SAMPLES_TRAIN_TOTAL
print("Size of dataset: ", size)

"""
for idx, (feature, label) in enumerate(dataset_train):
    if idx%100==0:
        print("Iteration: ", idx)
    # Convert each element to numpy using the `.numpy()` method
    #print(feature.numpy().shape)
    #print((np.array(label)).shape)
    features.append(feature.numpy())
    labels.append(np.array(label))

    # Convert the lists of numpy arrays to numpy arrays
    features_array = np.array(features)
    labels_array = np.array(labels)  
"""

# Initialize variables to hold the features and labels. Start with them as None.
features = None
labels = None

itr = iter(dataset_train)
itrVal = iter(dataset_val)

X_train = np.ndarray((n_samp_train,app.N_VARS_IN,
                    model_config['nx_']+model_config['pad'],
                    model_config['nz_']+model_config['pad']),
                    dtype='float')
Y_train = np.ndarray((n_samp_train,app.N_VARS_OUT,
                    model_config['nx_'],
                    model_config['nz_']),
                    dtype='float') 
ii = 0
print(X_train.shape)
for i in range(n_samp_train):
    features, label = next(itr)
    #print(features.shape)
    #print(np.squeeze(np.array(label), axis=2).shape)

    label = np.squeeze(np.array(label), axis=2)
    # Swap the first two axes
    label = np.swapaxes(label, 0, 1)

    X_train[i] = features
    Y_train[i] = label
    """
    if app.N_VARS_OUT == 1 :
        Y_train[i,0] = next(itr)
    elif app.N_VARS_OUT == 2 :
        (Y_train[i,0], Y_train[i,1]) = next(itr)
    else:
        (Y_train[i,0], Y_train[i,1], Y_train[i,2]) = next(itr)
    """
    ii += 1
    # print(i+1)

print(f'Iterated over {ii} test samples')
print('')
print("Final Shapes: ")
print(X_train.shape)
print(Y_train.shape)
"""
for idx, (feature, label) in enumerate(dataset_train):
    if idx % 100 == 0:
        print("Iteration: ", idx)

    # Convert the current feature and label to numpy arrays
    current_feature = feature.numpy()  # Assuming feature supports .numpy()
    current_label = np.array(label)

    # Reshape or expand dimensions if necessary to ensure compatibility for concatenation
    current_feature = np.expand_dims(current_feature, axis=0)
    current_label = np.expand_dims(current_label, axis=0)

    # Concatenate the current data with the previous ones
    if features is None:
        features = current_feature
        labels = current_label
    else:
        features = np.concatenate((features, current_feature), axis=0)
        labels = np.concatenate((labels, current_label), axis=1)


print("Features:", features.shape)
print("Labels:", labels.shape)
"""

# Combine both arrays into a dictionary
data_to_save = {'features': X_train, 'labels': Y_train}

filename = './numpy_data/Ret180_192x192x65_dt135_yp50_file000_001.pkl'
print(tfr_files_train)
print("Saving data to ", filename)
with open(filename,'wb') as f:
    pickle.dump(data_to_save, f)


"""
tf.keras.models.save_model(
    CNN_model,
    save_path+model_config['name'],
    overwrite=True,
    include_optimizer=True,
    save_format='h5'
)
"""
