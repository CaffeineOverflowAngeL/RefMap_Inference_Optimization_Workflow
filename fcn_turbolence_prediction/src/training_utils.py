#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import time
import math

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler 

#%% Configuration import
import config

prb_def = os.environ.get('MODEL_CNN', None)

if not prb_def:
    # raise ValueError('"MODEL_CNN" enviroment variable must be defined ("WallRecon" or "OuterRecon")')
    # print('"MODEL_CNN" enviroment variable not defined ("WallRecon" or "OuterRecon"), default value "WallRecon" is used')
    app = config.WallRecon
    prb_def = 'WallRecon'
elif prb_def == 'WallRecon':
    app = config.WallRecon
elif prb_def == 'OuterRecon':
    app = config.OuterRecon
else:
    raise ValueError('"MODEL_CNN" enviroment variable must be defined either as "WallRecon" or "OuterRecon"')

#%% Training utils
class ProfilerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, start_epoch, stop_epoch, steps_per_epoch, logger):
        super().__init__()
        self.log_dir = log_dir
        self.start_epoch = start_epoch
        self.stop_epoch = stop_epoch
        self.steps_per_epoch = steps_per_epoch
        self.logger = logger
        self.current_epoch = 0  # Initialize the current_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if epoch == self.start_epoch:
            tf.profiler.experimental.start(self.log_dir)
            #print(f"Started profiling at epoch {epoch}")
            self.logger.info("Started profiling at epoch {}".format(epoch))

    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.stop_epoch:
            tf.profiler.experimental.stop()
            #print(f"Stopped profiling at epoch {epoch}")
            self.logger.info("Stopped profiling at epoch {}".format(epoch))

    def on_train_batch_end(self, batch, logs=None):
        # Use self.current_epoch to access the current epoch
        if self.start_epoch <= self.current_epoch < self.stop_epoch:
            #self.logger.info("Batch number: {}".format(batch))
            if batch == int(self.steps_per_epoch):
                tf.profiler.experimental.stop()
                self.logger.info("Stopped profiling after {} steps in epoch {}".format(self.steps_per_epoch, self.current_epoch))
                #print(f"Stopped profiling after {self.steps_per_epoch} steps in epoch {self.current_epoch}")

# Credit to Martin Holub for the Class definition
class SubTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(SubTensorBoard, self).__init__(*args, **kwargs)

    """
    def lr_getter(self):
        # Get vals
        decay = self.model.optimizer.decay
        lr = self.model.optimizer.learning_rate
        iters = self.model.optimizer.iterations # only this should not be const
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2
        # calculate
        lr = lr * (1. / (1. + decay * K.cast(iters, K.dtype(decay))))
        t = K.cast(iters, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(beta_2, t)) / (1. - K.pow(beta_1, t)))
        return np.float32(K.eval(lr_t))
    """

    def lr_getter(self):
        lr = self.model.optimizer.learning_rate
        return np.float32(K.eval(lr))

    def on_epoch_end(self, episode, logs = {}):
        logs.update({'learning_rate': self.lr_getter()})
        super(SubTensorBoard, self).on_epoch_end(episode, logs)
        
# Credit to Marcin Możejko for the Callback definition
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
def get_epoch_times(callbacks):
    for callback in callbacks:
        if isinstance(callback, TimeHistory):  # Check if the callback is an instance of TimeHistory
            epoch_times = callback.times  # Access the times attribute
            break
    return epoch_times

def step_decay(epoch):
   epochs_drop = app.LR_EPDROP
   initial_lrate = app.INIT_LR
   drop = app.LR_DROP
   lrate = initial_lrate * math.pow(drop,  
           math.floor((epoch)/epochs_drop))
   return lrate

#%% FCN model

from fcn import cnn_model, thres_relu

def get_model(model_config, logger=None, weights_checkpoint=False):
    # Callbacks
    """
    steps_per_epoch = math.ceil((app.N_SAMPLES_TRAIN_TOTAL / app.BATCH_SIZE)*(1-app.VAL_SPLIT))
    logger.info("Defined steps per epoch for profiler: {}".format(steps_per_epoch))
    profiler_callback = ProfilerCallback(log_dir='.logs_new/{}'.format(model_config['name']), 
                                         start_epoch=1, 
                                         stop_epoch=int(app.N_EPOCHS)-1, 
                                         steps_per_epoch=steps_per_epoch,
                                         logger=logger)
    
    
    tensorboard = SubTensorBoard(
        log_dir='.logs_new/{}'.format(model_config['name']),
        histogram_freq=app.TB_HIST_FREQ,
        profile_batch='1,1000',
    )
    """
    
    weights_checkpoint = ModelCheckpoint(
        '.logs_new/'+model_config['name']+'/model.ckpt.{epoch:04d}.hdf5', 
        verbose=1, save_freq=app.CKPT_FREQ)
    
    lrate = LearningRateScheduler(step_decay)
    time_callback = TimeHistory()
    
    #callbacks = [tensorboard, checkpoint, lrate, time_callback]
    if weights_checkpoint:
        callbacks = [weights_checkpoint, lrate, time_callback] # TODO: FIX profiler_callback
    else:
        callbacks = [lrate, time_callback] # TODO: FIX profiler_callback
    
    init_lr = model_config['init_lr']
    
    if model_config['distributed_training']:
       print('Compiling and training the model for multiple GPU') 
       if app.INIT == 'model':
           init_model = tf.keras.models.load_model(model_config['model_path'])
           init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
       
       with model_config['strategy'].scope():
           CNN_model, losses = cnn_model(
               input_shape=model_config['input_shape'],
               padding=model_config['padding'],
               pad_out=model_config['pad_out'],
               pred_fluct = app.FLUCTUATIONS_PRED)
           
           if app.INIT == 'model':
               print('Weights of the model initialized with another trained model')
               # init_model = tf.keras.models.load_model(model_path)
               # init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
               CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
               os.remove('/tmp/model_weights-CNN_keras_model.h5')
    
               # A smaller learning rate is used in this case
               init_lr = init_lr/2
           
           CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(
                             learning_rate=init_lr))
       
    else:
       CNN_model, losses = cnn_model(
               input_shape=model_config['input_shape'],
               padding=model_config['padding'],
               pad_out=model_config['pad_out'],
               pred_fluct = app.FLUCTUATIONS_PRED) 
       # Initialization of the model for transfer learning, if required
       if app.INIT == 'model':
           print('Weights of the model initialized with another trained model')
           # TODO: check if this condition is still valid for the models that were
           # added later
    #       if int(model_path[-67]) != app.NET_MODEL:
    #           raise ValueError('The model for initialization is different from the model to be initialized')
               
           init_model = tf.keras.models.load_model(model_config['model_path'])
           init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
           CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
           os.remove('/tmp/model_weights-CNN_keras_model.h5')
           
           # A smaller learning rate is used in this case
           init_lr = init_lr/2
           
           # TODO: Modify this implementation of transfer learning to account for cropping layers
           # if app.TRANSFER_LEARNING == True:
           #     lyrs = CNN_model.layers
           #     n_lyrs = len(lyrs)
           #     for i_l in range(n_lyrs):
           #         print(CNN_model.layers[i_l].name, CNN_model.layers[i_l].trainable)
           #         if i_l <= n_lyrs - (2+3*(app.N_TRAINABLE_LAYERS-1)) - 1:  # Every layer has 3 sublayers (conv+batch_norm+activ), except the last one (no batch_norm)
           #             CNN_model.layers[i_l].trainable = False
           #         print(CNN_model.layers[i_l].name, CNN_model.layers[i_l].trainable)
    
       elif app.INIT == 'random':
           print('Weights of the model initialized from random distributions')
    
       print('Compiling and training the model for single GPU')
       CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))
    
        
    print(CNN_model.summary())
        
    return CNN_model, callbacks

def load_trained_model(model_config):
    pred_path = model_config['pred_path']
    init_lr = model_config['init_lr']
    
    if app.FROM_CKPT == True:
        model_path = app.CUR_PATH+'/.logs/'+model_config['name']+'/'
        ckpt = app.CKPT
        init_model = tf.keras.models.load_model(
                model_path+f'model.ckpt.{ckpt:04d}.hdf5',
                custom_objects={"thres_relu": layers.Activation(thres_relu)}
                )
        print('[MODEL LOADING]')
        print('Loading model '+str(app.NET_MODEL)+' from checkpoint '+str(ckpt))    
        pred_path = pred_path+f'ckpt_{ckpt:04d}/'
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
    else:
        model_path = app.CUR_PATH+'/.saved_models/'
        init_model = tf.keras.models.load_model(
                model_path+model_config['name'],
                custom_objects={"thres_relu": layers.Activation(thres_relu)}
                # custom_objects={"thres_relu": thres_relu}
                )
        print('[MODEL LOADING]')
        print('Loading model '+str(app.NET_MODEL)+' from saved model')
        pred_path = pred_path+'saved_model/'
        if not os.path.exists(pred_path):
            os.mkdir(pred_path)
    
    # If distributed training is used, we need to load only the weights
    if model_config['distributed_training']:
    
       print('Compiling and training the model for multiple GPU')
    
       with model_config['strategy'].scope():
    
           CNN_model, losses = cnn_model(
               input_shape=model_config['input_shape'],
               padding=model_config['padding'],
               pad_out=model_config['pad_out'],
               pred_fluct = app.FLUCTUATIONS_PRED)
    
           CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))
    
               
           init_model.save_weights('/tmp/model_weights-CNN_keras_model.h5')
           CNN_model.load_weights('/tmp/model_weights-CNN_keras_model.h5')
           os.remove('/tmp/model_weights-CNN_keras_model.h5')
           del init_model
               
    
    else:
        CNN_model = init_model
    
        CNN_model.compile(loss='mse',
                         optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr))
    
            
    print(CNN_model.summary())
    
    return CNN_model
