#!/bin/bash
echo "Convert TF saved models to onnx and Pytorch formats." 

# Execution Samples

#python3 model_converter.py --mode pt --model /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/.baseline_models/WallRecon1TF2_3In-3Out_0100_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707519212.h5 --dataset /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/numpy_data/yp100/Ret180_192x192x65_dt135_yp100_file000.pkl --debug-mode

#python3 model_converter.py --mode pt --model /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/storage/KTH_model_format/WallReconfluct1TF2_3NormIn-3ScaledOut_0100_192x192_Ret180_lr0.001_decay20drop0.5_relu-1588941081 --dataset /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/numpy_data/yp100/Ret180_192x192x65_dt135_yp100_file000.pkl --model-type hdf5 --debug-mode 

# For all baseline models

echo "ret180 - yp15"
python3 model_converter.py --mode pb --model /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/.baseline_models/WallRecon1TF2_3In-3Out_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707483599.h5 --dataset /home/angel/RefMap_Workflow/fcn_turbolence_prediction/numpy_data/yp15/Ret180_192x192x65_dt135_yp15_file000.pkl --debug-mode

echo "ret180 - yp30"
python3 model_converter.py --mode pt --model  /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/.baseline_models/WallRecon1TF2_3In-3Out_030_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707444383.h5 --dataset /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/numpy_data/yp30/Ret180_192x192x65_dt135_yp30_file000.pkl --debug-mode

echo "ret180 - yp50"
python3 model_converter.py --mode pt --model  /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/.baseline_models/WallRecon1TF2_3In-3Out_050_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707434638.h5 --dataset /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/numpy_data/yp50/Ret180_192x192x65_dt135_yp50_file000.pkl --debug-mode

echo "ret180 - yp100"
python3 model_converter.py --mode pt --model /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/.baseline_models/WallRecon1TF2_3In-3Out_0100_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707519212.h5 --dataset /mnt/d/RefMap/FCN-turbulence-predictions-from-wall-quantities/src/numpy_data/yp100/Ret180_192x192x65_dt135_yp100_file000.pkl --debug-mode

python3 model_converter.py --mode pb --model /home/angel/RefMap_Workflow/fcn_turbolence_prediction/src/.logs/WallReconfluct1TF2_3NormIn-3ScaledOut_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1588580899/model.ckpt.0050.hdf5 --dataset /home/angel/RefMap_Workflow/fcn_turbolence_prediction/numpy_data/yp15/Ret180_192x192x65_dt135_yp15_file000.pkl --debug-mode