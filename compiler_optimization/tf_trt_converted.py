import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

PATH = './RefMap_Workflow/fcn_turbolence_prediction/converted_models/pb/model.ckpt.0050.h'
SAVEDMODEL_PATH = 'trt_converted_models/015'

# Convert the SavedModel
converter = trt.TrtGraphConverterV2(input_saved_model_dir=PATH)
converter.convert()

# Save the converted model
converter.save(SAVEDMODEL_PATH)