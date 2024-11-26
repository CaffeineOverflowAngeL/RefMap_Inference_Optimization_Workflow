#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class WallRecon:
    # Problem definition
    N_VARS_IN = 3
    N_VARS_OUT = 3
    N_PLANES_OUT = 3
    VARS_NAME_IN = ('u','w','p')
    VARS_NAME_OUT = ('u','v','w')
    
    NORMALIZE_INPUT = True
    SCALE_OUTPUT = True
    
    FLUCTUATIONS_PRED = True
    RELU_THRESHOLD = -1.0
    
    TRAIN_YP = 0
    TARGET_YP = (15)
    
    # Hardware definition
    ON_GPU = True
    N_GPU = 1 
    WHICH_GPU_TRAIN = 0 
    WHICH_GPU_TEST = 0
    # Input for 'CUDA_VISIBLE_DEVICES'

    # Dataset definition
    CUR_PATH = 'fcn_turbolence_prediction/src' # Default
    DS_PATH = 'fcn_turbolence_prediction/sample_data/yp30/Train' # Provide the absolute PATH
    DS_PATH_TEST = 'fcn_turbolence_prediction/sample_data/yp15/Test' # Provide the absolute PATH


    N_DATASETS = 1
    N_SAMPLES_TRAIN_TOTAL = 4400*N_DATASETS
    N_SAMPLES_TRAIN = (4400 #, 4400, 4400,
                       #4440, 4400, 4400, 4440, 4400, 4400, 
                       )

    INTERV_TRAIN = 3

    N_SAMPLES_TEST = (4200) #*8
    INTERV_TEST = 1 

    TIMESTAMP = '1588580899'
    FROM_CKPT = True
    CKPT = 50
    
    # Prediction that has to be post-processed
    PRED_NB = 0
    
    DATA_AUG = False   # Data augmentation
    
    # Network definition
    NET_MODEL = 1
    # See README for network descriptions
    
    # Model-specific options (syntax: [option]_[net_model])
    PAD_1 = 'wrap' 
    
    # Training options
    INIT = 'random' # default value: 'random'
    INIT_MODEL = '../src/.logs/WallReconfluct1TF2_3NormIn-3ScaledOut_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1588580899'
    TRANSFER_LEARNING = False
    N_TRAINABLE_LAYERS = 3
    
    N_EPOCHS = 2
    BATCH_SIZE = 1
    VAL_SPLIT = 0.2 #TODO: Change back to 0.2
    
    INIT_LR = 0.001
    LR_DROP = 0.5
    LR_EPDROP = 20.0
    
    # Callbacks specifications
    """
    The histogram_freq argument controls how often to log histogram visualizations. 
    If histogram_freq is set to 0 (or is not set), histograms won't be computed. 
    Set this to a positive integer to log histograms.
    """
    TB_HIST_FREQ = 1
    """
    The ckpt_freq is set to periodically save the model as the training progresses. 
    To apply on benchmark enable checkpoint when loading the model -> get_model(model_config_test, checkpoint = True)
    Default values for checkpointing is False.
    """
    CKPT_FREQ = 1000
    
class OuterRecon:
    # Problem definition
    N_VARS_IN = 3
    N_VARS_OUT = 3
    VARS_NAME = ('u','v','w')
    
    TRAIN_YP = 80
    TARGET_YP = 15
    
    # Hardware definition
    ON_GPU = True
    N_GPU = 1 
    WHICH_GPU_TRAIN = 0
    WHICH_GPU_PRED = 1
    # Input for 'CUDA_VISIBLE_DEVICES'
    
    # Dataset definition
    #CUR_PATH = '/storage/Train'
    #DS_PATH = '/storage/Test'
    
    N_DATASETS = 1 # fixed for now!
    N_SAMPLES_TRAIN = 1500
    INTERV_TRAIN = 1
    N_SAMPLES_TEST = 1500
    INTERV_TEST = 1
    
    # Network definition
    NET_VERSION = 0
    # See README for network descriptions
    
    # Model-specific options (syntax: [option]_[net_model])
    PAD_1 = 'wrap'       
    
    N_EPOCHS = 3
    BATCH_SIZE = 8
    VAL_SPLIT = 0.2
    
    INIT_LR = 0.01
    LR_DROP = 0.5
    LR_EPDROP = 50.0
    
    # Callbacks specifications
    TB_HIST_FREQ = 10
    CKPT_FREQ = 10

    # ADDED
    TIMESTAMP = '1562346575'
    # Dataset definition
    CUR_PATH = '../storage/FieldRecon'
    DS_PATH = '../storage/Train'
    DS_PATH_TEST = '../storage/Test'
    NET_MODEL = 1
    FLUCTUATIONS_PRED = False
    NORMALIZE_INPUT = False
    SCALE_OUTPUT = False
    INIT = 'random' # default value: 'random'
     
