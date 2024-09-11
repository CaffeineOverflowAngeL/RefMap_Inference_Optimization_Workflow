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

from statistics import mean
from fcn_turbolence_prediction.conf import config
from fcn_turbolence_prediction.src.tfrecord_utils import get_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['MODEL_CNN'] = "WallRecon"

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

SAVEDMODEL_PATH = './RefMap_Workflow/fcn_turbolence_prediction/converted_models/pb/model.ckpt.0050.h'

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

    return X_test_data, Y_test_data
    
def load_with_converter(path, precision, batch_size):
    """Loads a saved model using a TF-TRT converter, and returns the converter
    """

    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    if precision == 'int8':
        precision_mode = trt.TrtPrecisionMode.INT8
    elif precision == 'fp16':
        precision_mode = trt.TrtPrecisionMode.FP16
    else:
        precision_mode = trt.TrtPrecisionMode.FP32

    params = params._replace(
        precision_mode=precision_mode,
        max_workspace_size_bytes=2 << 32,  # 8,589,934,592 bytes
        maximum_cached_engines=100,
        minimum_segment_size=3,
        allow_build_at_runtime=True
    )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path,
        conversion_params=params,
    )

    return converter


if __name__ == "__main__":

    INFERENCE_STEPS = 1000
    WARMUP_STEPS = 100

    parser = argparse.ArgumentParser()

    feature_parser = parser.add_mutually_exclusive_group(required=True)

    feature_parser.add_argument('--use_native_tensorflow', dest="use_tftrt", help="help", action='store_false')
    feature_parser.add_argument('--use_tftrt_model', dest="use_tftrt", action='store_true')

    parser.add_argument('--precision', dest="precision", type=str, default="fp16", choices=['int8', 'fp16', 'fp32'], help='Precision')
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=4, help='Batch size')

    args = parser.parse_args()

    
    print("\n=========================================")
    print("Inference using: {} ...".format(
        "TF-TRT" if args.use_tftrt else "Native TensorFlow")
    )
    print("Batch size:", args.batch_size)
    if args.use_tftrt:
        print("Precision: ", args.precision)
    print("=========================================\n")
    time.sleep(2)

    X_test_data, Y_test_data = load_raw_turbolence_prediction_setup(prb_def=os.environ.get('MODEL_CNN', None))
    
    def dataloader_fn(X_test_data, Y_test_data, batch_size):

         # Convert input data to tf.float32
        X_test_data = tf.cast(X_test_data, tf.float32)
        Y_test_data = tf.cast(Y_test_data, tf.float32)

        # Create TensorFlow datasets
        X_test = tf.data.Dataset.from_tensor_slices(X_test_data)
        Y_test = tf.data.Dataset.from_tensor_slices(Y_test_data)
        
        # Combine datasets into a single dataset
        turbolence_dataset = tf.data.Dataset.zip((X_test, Y_test))

        #print(test_dataset)

        ds = turbolence_dataset.cache().repeat().batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    if args.use_tftrt:
        converter = load_with_converter(
            os.path.join(SAVEDMODEL_PATH),
            precision=args.precision,
            batch_size=args.batch_size
        )
        if args.precision == 'int8':
            num_calibration_batches = 2
            calibration_data = dataloader_fn(X_test_data, Y_test_data, args.batch_size)
            calibration_data = calibration_data.take(num_calibration_batches)
            def calibration_input_fn():
                for x, y in calibration_data:
                    yield (x, )
            xx = converter.convert(calibration_input_fn=calibration_input_fn)
        else:
            # fp16 or fp32
            xx = converter.convert()

        converter.save(
            os.path.join(SAVEDMODEL_PATH, "converted")
        )

        root = tf.saved_model.load(os.path.join(SAVEDMODEL_PATH, "converted"))
    else:
        root = tf.saved_model.load(SAVEDMODEL_PATH)

    infer = root.signatures['serving_default']
    output_tensorname = list(infer.structured_outputs.keys())[0]

    ds = dataloader_fn(
        X_test_data,
        Y_test_data,
        batch_size=args.batch_size
    )
    iterator = iter(ds)
    features, labels = iterator.get_next()

    print(ds)

    try:
        step_times = list()
        for step in range(1, INFERENCE_STEPS + 1):
            if step % 100 == 0:
                print("Processing step: %04d ..." % step)
            start_t = time.perf_counter()
            #print("Root Signatures: ", root.signatures)
            #print("Infer structured input signature: ", infer.structured_input_signature)
            #print("Infer structured outputs: ", infer.structured_outputs)
            #print("Features shape: ", features.shape)
            probs = infer(features)[output_tensorname]
            #inferred_class = tf.math.argmax(probs).numpy()
            step_time = time.perf_counter() - start_t
            if step >= WARMUP_STEPS:
                step_times.append(step_time)
    except tf.errors.OutOfRangeError:
        pass

    avg_step_time = mean(step_times)
    print("\nAverage step time: %.1f msec" % (avg_step_time * 1e3))
    print("Average throughput: %d samples/sec" % (
        args.batch_size / avg_step_time
    ))
