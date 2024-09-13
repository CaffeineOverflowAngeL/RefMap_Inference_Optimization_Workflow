#!/bin/bash
echo "Prune FCN Turbolence models sample" 

# Test a Model # 
python3 model_compression/benchmarks/benchmark.py --model ./fcn_turbolence_prediction/converted_models/pt/WallRecon1TF2_3In-3Out_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707483599.pth --mode test --dataset ./fcn_turbolence_prediction/numpy_data/yp15/Ret180_192x192x65_dt135_yp15_file000.pkl

# Pre-train a Model # 
python3 model_compression/benchmarks/benchmark.py --model ./fcn_turbolence_prediction/converted_models/pt/WallRecon1TF2_3In-3Out_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707483599.pth --mode pretrain --dataset ./fcn_turbolence_prediction/numpy_data/yp15/Ret180_192x192x65_dt135_yp15_file000.pkl --reset-trained-weigths

# Prune a Model # 

# If you want to select global scope for the pruning decision also include --global-pruning in the commands (default setting has been set to local scope)#

python3 model_compression/benchmarks/benchmark.py --model ./fcn_turbolence_prediction/converted_models/pt/WallRecon1TF2_3In-3Out_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707483599.pth --mode prune --dataset ./fcn_turbolence_prediction/numpy_data/yp15/Ret180_192x192x65_dt135_yp15_file000.pkl --method l1  --lr 0.001 --yp yp15 --global-pruning

# Some methods, e.g. group_sl, growing_reg, and slim are better fitted with specialized regularization prior to pruning (--reg lr)
python3 model_compression/benchmarks/benchmark.py --model ./fcn_turbolence_prediction/converted_models/pt/WallRecon1TF2_3In-3Out_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707483599.pth --mode prune --dataset ./fcn_turbolence_prediction/numpy_data/yp15/Ret180_192x192x65_dt135_yp15_file000.pkl --method group_sl --lr 0.001 --yp yp15 --global-pruning --reg 5e-4 --sl-lr 0.0001
--method slim --speed-up 2.11 --global-pruning --reg 1e-5