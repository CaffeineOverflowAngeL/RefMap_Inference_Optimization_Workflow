import os
import copy
import argparse
import time
import sys
import numpy as np

######## Configuring the Path Setup ########
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to the system path
sys.path.append(parent_dir)
#############################################

from keras.models import load_model
from statistics import mean
from fcn_turbolence_prediction.conf import config
from fcn_turbolence_prediction.src.tfrecord_utils import get_dataset
from fcn_turbolence_prediction.src.evaluate_utils import get_mse_per_dim, get_mse_stats_per_dim

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['MODEL_CNN'] = "WallRecon"

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

VARS_NAME_OUT = ('u','v','w')

def load_raw_turbolence_prediction_setup(prb_def, verbose=0):
    
    # Parse application info based on the selected task, i.e. WallRecon or OuterRecon.
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

    # Setting application values #
    # Data augmentation not implemented in this model for now
    app.DATA_AUG = False
    # Transfer learning not implemented in this model for now
    app.TRANSFER_LEARNING = False
    ###############################

    # Configuring Distributed Execution #
    # Hardware detection and parallelization strategy
    physical_devices = tf.config.list_physical_devices('GPU')
    availale_GPUs = len(physical_devices) 
    print('Using TensorFlow version:', tf.__version__, ', GPU:', availale_GPUs)
    #print(tf.keras.__version__)

    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    on_GPU = app.ON_GPU
    n_gpus = app.N_GPU

    distributed_training = on_GPU == True and n_gpus>1
    ########################################

    # Loading Turbolence Data #
    dataset_test, X_test, n_samples_tot, model_config = \
    get_dataset(prb_def, app, timestamp=app.TIMESTAMP, 
                train=False, distributed_training=distributed_training)
    ###########################

    if verbose:
        print('')
        print('# ====================================================================')
        print('#     Summary of the options for the model                            ')
        print('# ====================================================================')
        print('')

        print(f'Number of samples for training: {int(n_samples_tot)}')
        # print(f'Number of samples for validation: {int(n_samp_valid)}')
        print(f'Total number of samples: {n_samples_tot}')
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
        # print(f'File for input statistics: {app.AVG_INPUTS_FILE}')

        print('')
        print('# ====================================================================')
    
    # Iterating over ground truth datasets #
    itr = iter(dataset_test)
    itrX = iter(X_test)

    X_test_data = np.ndarray((n_samples_tot,app.N_VARS_IN,
                        model_config['nx_']+model_config['pad'],
                        model_config['nz_']+model_config['pad']),
                        dtype='float')
    Y_test_data = np.ndarray((n_samples_tot,app.N_VARS_OUT,
                        model_config['nx_'],
                        model_config['nz_']),
                        dtype='float') 
    ii = 0
    for i in range(n_samples_tot):
        X_test_data[i] = next(itrX)
        if app.N_VARS_OUT == 1 :
            Y_test_data[i,0] = next(itr)
        elif app.N_VARS_OUT == 2 :
            (Y_test_data[i,0], Y_test_data[i,1]) = next(itr)
        else:
            (Y_test_data[i,0], Y_test_data[i,1], Y_test_data[i,2]) = next(itr)
        ii += 1
        # print(i+1)
    if verbose:
        print(f'Iterated over {ii} samples')
        print('')
    ############################################

    return X_test_data, Y_test_data, model_config, app

def load_trained_model(model_config, app):
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
    else:
        model_path = app.CUR_PATH+'/.saved_models/'
        init_model = tf.keras.models.load_model(
                model_path+model_config['name'],
                custom_objects={"thres_relu": layers.Activation(thres_relu)}
                # custom_objects={"thres_relu": thres_relu}
                )
        print('[MODEL LOADING]')
        print('Loading model '+str(app.NET_MODEL)+' from saved model')
    
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

def cnn_model(input_shape,padding,pad_out, pred_fluct):

    if pred_fluct == None:
     pred_fluct = app.FLUCTUATIONS_PRED
    
    input_data = layers.Input(shape=input_shape, name='input_data')
    # ------------------------------------------------------------------
    cnv_1 = layers.Conv2D(64, (5, 5), padding=padding,
                                data_format='channels_first')(input_data)
    bch_1 = layers.BatchNormalization(axis=1)(cnv_1)
    act_1 = layers.Activation('relu')(bch_1)
    # ------------------------------------------------------------------
    cnv_2 = layers.Conv2D(128, (3, 3), padding=padding,
                                data_format='channels_first')(act_1)
    bch_2 = layers.BatchNormalization(axis=1)(cnv_2)
    act_2 = layers.Activation('relu')(bch_2)
    # ------------------------------------------------------------------
    cnv_3 = layers.Conv2D(256, (3, 3), padding=padding,
                                data_format='channels_first')(act_2)
    bch_3 = layers.BatchNormalization(axis=1)(cnv_3)
    act_3 = layers.Activation('relu')(bch_3)
    # ------------------------------------------------------------------
    cnv_4 = layers.Conv2D(256, (3, 3), padding=padding,
                                data_format='channels_first')(act_3)
    bch_4 = layers.BatchNormalization(axis=1)(cnv_4)
    act_4 = layers.Activation('relu')(bch_4)
    # ------------------------------------------------------------------
    cnv_5 = layers.Conv2D(128, (3, 3), padding=padding,
                                data_format='channels_first')(act_4)
    bch_5 = layers.BatchNormalization(axis=1)(cnv_5)
    act_5 = layers.Activation('relu')(bch_5)
    # ------------------------------------------------------------------
    # Different branches for different components
    
    # Branch 1
    cnv_b1 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
    if pred_fluct == True:
        act_b1 = layers.Activation(thres_relu)(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)
    else:        
        act_b1 = layers.Activation('relu')(cnv_b1)
        output_b1 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b1')(act_b1)
        
    losses = {'output_b1':'mse'}
    
    if app.N_VARS_OUT == 2:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
        if pred_fluct == True:
            act_b2 = layers.Activation(thres_relu)(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:        
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        
        outputs_model = [output_b1, output_b2]
        
        losses['output_b2']='mse'
    
    elif app.N_VARS_OUT == 3:
        # Branch 2
        cnv_b2 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
        if pred_fluct == True:
            act_b2 = layers.Activation(thres_relu)(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
        else:        
            act_b2 = layers.Activation('relu')(cnv_b2)
            output_b2 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b2')(act_b2)
    
        losses['output_b2']='mse'
        
        # Branch 3    
        cnv_b3 = layers.Conv2D(1, (3, 3), padding=padding,
                                data_format='channels_first')(act_5)
        if pred_fluct == True:
            act_b3 = layers.Activation(thres_relu)(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)
        else:        
            act_b3 = layers.Activation('relu')(cnv_b3)
            output_b3 = layers.Cropping2D(cropping=((int(pad_out/2), int(pad_out/2)),
                                                (int(pad_out/2), int(pad_out/2))),
                                      data_format='channels_first',name='output_b3')(act_b3)   
    
        outputs_model = [output_b1, output_b2, output_b3]
        
        losses['output_b3']='mse'
    
    else:
        outputs_model = output_b1
    
    CNN_model = tf.keras.models.Model(inputs=input_data, outputs=outputs_model)
    return CNN_model, losses
    

# Final ReLu function for fluctuations
def thres_relu(x):
   return tf.keras.activations.relu(x, threshold=app.RELU_THRESHOLD)

if __name__ == "__main__":

    INFERENCE_STEPS = 3000
    WARMUP_STEPS = 2000

    parser = argparse.ArgumentParser()

    feature_parser = parser.add_mutually_exclusive_group(required=True)

    feature_parser.add_argument('--use_native_tensorflow', dest="use_openxla", help="help", action='store_false')
    feature_parser.add_argument('--use_xla_model', dest="use_openxla", action='store_true')

    parser.add_argument("--restore-format", type=str, choices=["h5", "hdf5"], default="h5")
    parser.add_argument("--restore-path", type=str, required=False)

    parser.add_argument('--precision', dest="precision", type=str, default="fp16", choices=['int8', 'fp16', 'fp32'], help='Precision')
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=4, help='Batch size')
    parser.add_argument('--batch_size_inference', type=int, default=1, help='Batch size for inference latency evaluation')
    parser.add_argument('--n_vars_out', type=int, default=3)

    args = parser.parse_args()

    
    print("\n=========================================")
    print("Inference using: {} ...".format(
        "XLA" if args.use_openxla else "Native TensorFlow")
    )
    print("Batch size:", args.batch_size)
    if args.use_openxla:
        print("Precision: ", args.precision)
    print("=========================================\n")
    time.sleep(2)

    X_test_data, Y_test_data, model_config, app = load_raw_turbolence_prediction_setup(prb_def=os.environ.get('MODEL_CNN', None))
        
    # Load Model 
    if args.restore_format == 'h5':
        CNN_model = load_model(args.restore_path)

        print("Loading model: {} from {}".format(model_config['name'], args.restore_path))
    elif args.restore_format == 'hdf5':
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import TensorBoard
        from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler 
        CNN_model = load_trained_model(model_config, app)
    else: 
        raise NotImplementedError("Model must be specified in cli.")
    
    if args.use_openxla:
        # Enable XLA JIT compilation
        #tf.config.optimizer.set_jit(True)
        xla_fn = tf.function(CNN_model, jit_compile=True)
        xla_fn_full = tf.function(CNN_model.predict, jit_compile=True)
    
    # Select a random batch #
    n_samples_tot = X_test_data.shape[0]  # Total number of samples in the original dataset
    n_samples_to_draw = args.batch_size_inference  # Number of samples you want to draw, adjust as needed

    # Generate random indices
    random_indices = np.random.choice(n_samples_tot, n_samples_to_draw, replace=False)

    # Sample from the original array using the random indices
    X_sampled = X_test_data[random_indices]

    # Random batch shape
    print(X_sampled.shape)
    #############################
    

    if args.use_openxla:
        try:
            step_times = list()
            for step in range(1, INFERENCE_STEPS + 1):
                if step % 100 == 0:
                    print("Processing step: %04d ..." % step)
                start_t = time.perf_counter()
                xla_fn(X_sampled)
                """
                # Convert predictions to numpy arrays if they are not already
                a = np.array(a)
                b = np.array(b)
                approx_equality = np.allclose(a, b, atol=1e-1)
                print(f"Approximate equality: {approx_equality}")
                """
                #inferred_class = tf.math.argmax(probs).numpy()
                step_time = time.perf_counter() - start_t
                if step >= WARMUP_STEPS:
                    step_times.append(step_time)
        except tf.errors.OutOfRangeError:
            pass

        print("\n=========================================")
        print("Inference using: XLA Optimized Graph...")
        avg_step_time = mean(step_times)
        print("\nAverage step time: %.1f msec" % (avg_step_time * 1e3))
        print("Average throughput: %d samples/sec" % (
            args.batch_size_inference / avg_step_time
        ))
        print("=========================================\n")
    else: 
        try:
            step_times = list()
            for step in range(1, INFERENCE_STEPS + 1):
                if step % 100 == 0:
                    print("Processing step: %04d ..." % step)
                start_t = time.perf_counter()
                a = CNN_model.predict(X_sampled, verbose=0)
                """
                # Convert predictions to numpy arrays if they are not already
                a = np.array(a)
                b = np.array(b)
                approx_equality = np.allclose(a, b, atol=1e-1)
                print(f"Approximate equality: {approx_equality}")
                """
                #inferred_class = tf.math.argmax(probs).numpy()
                step_time = time.perf_counter() - start_t
                if step >= WARMUP_STEPS:
                    step_times.append(step_time)
        except tf.errors.OutOfRangeError:
            pass

        print("\n=========================================")
        print("Inference using: Default Compiler Graph ...")
        avg_step_time = mean(step_times)
        print("\nAverage step time: %.1f msec" % (avg_step_time * 1e3))
        print("Average throughput: %d samples/sec" % (
            args.batch_size_inference / avg_step_time
        ))
        print("=========================================\n")

    # Evaluating predictive performance # 
    # Testing

    if args.use_openxla:
        Y_pred_xla = np.ndarray((n_samples_tot,args.n_vars_out,
                            model_config['nx_'],model_config['nz_']),dtype='float') 
        
        b = []
        for start_idx in range(0, n_samples_tot):
            end_idx = min(start_idx + args.batch_size_inference, n_samples_tot)
            batch = X_sampled[start_idx:end_idx]
            batch_prediction = xla_fn(batch)
            b.append(np.array(batch_prediction[0]))

        overall_mse_per_dimension_xla = get_mse_per_dim(args.n_vars_out, Y_pred_xla, Y_test_data, VARS_NAME_OUT)

        print("MSE with XLA: ", overall_mse_per_dimension_xla)
    else: 
        Y_pred = np.ndarray((n_samples_tot,args.n_vars_out,
                            model_config['nx_'],model_config['nz_']),dtype='float')  

        if args.n_vars_out == 2:
            start_time=time.time()
            (Y_pred[:,0,np.newaxis], Y_pred[:,1,np.newaxis]) = CNN_model.predict(
                X_test_data, batch_size=args.batch_size)
            end_time=time.time()
        if args.n_vars_out == 3:
            (Y_pred[:,0,np.newaxis], Y_pred[:,1,np.newaxis], Y_pred[:,2,np.newaxis]) = \
                CNN_model.predict(X_test_data, batch_size=args.batch_size)
            
        if app.SCALE_OUTPUT == True:
            u_rms = model_config['rms'][0]\
                    [model_config['ypos_Ret'][str(app.TARGET_YP)]]
            
            for i in range(app.N_VARS_OUT):
                print('Rescale back component '+str(i)) 
                Y_pred[:,i] *= model_config['rms'][i]\
                    [model_config['ypos_Ret'][str(app.TARGET_YP)]]/\
                    u_rms
                Y_test_data[:,i] *= model_config['rms'][i]\
                    [model_config['ypos_Ret'][str(app.TARGET_YP)]]/\
                    u_rms

        #if pred_fluct == True:
        #    for i in range(app.N_VARS_OUT):
        #        print('Adding back mean of the component '+str(i))
        #        Y_pred[:,i] = Y_pred[:,i] + avgs[i][ypos_Ret[str(target_yp)]]
        #        Y_test_data[:,i] = Y_test_data[:,i] + avgs[i][ypos_Ret[str(target_yp)]]
            
        overall_mse_per_dimension = get_mse_per_dim(args.n_vars_out, Y_pred, Y_test_data, VARS_NAME_OUT)

        print("MSE default: ", overall_mse_per_dimension)
    
    