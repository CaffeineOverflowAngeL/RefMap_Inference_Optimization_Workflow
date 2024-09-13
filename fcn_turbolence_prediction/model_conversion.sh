#!/bin/bash
echo "Convert TF saved models to onnx and Pytorch formats." 

# Execution Samples

#python3 model_converter.py --mode pt --model ... --dataset src/numpy_data/... --debug-mode

#python3 model_converter.py --mode pt --model ... --dataset src/numpy_data/... --model-type hdf5 --debug-mode 

# For all baseline models

echo "ret180 - yp15"
python3 model_converter.py --mode pb --model ... --dataset ./fcn_turbolence_prediction/numpy_data/yp15/Ret180_192x192x65_dt135_yp15_file000.pkl --debug-mode

echo "ret180 - yp30"
python3 model_converter.py --mode pt --model ... --dataset ... --debug-mode

echo "ret180 - yp50"
python3 model_converter.py --mode pt --model ... --dataset ... --debug-mode

echo "ret180 - yp100"
python3 model_converter.py --mode pt --model ... --dataset ... --debug-mode

python3 model_converter.py --mode pb --model ... --dataset ... --debug-mode