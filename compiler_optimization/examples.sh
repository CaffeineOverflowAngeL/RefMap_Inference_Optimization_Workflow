#!/bin/bash
echo "Inference Optimization Examples." 

########## yp 15 ##########

# Baseline # 

python3 xla_inference.py --use_native_tensorflow --restore-path ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707483599.h5

# XLA Optimised # 

python3 xla_inference.py --use_xla_model --restore-path ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_015_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707483599.h5

########## yp 30 ##########

# Baseline #

python3 xla_inference.py --use_native_tensorflow --restore-path ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_030_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707444383.h5

# XLA Optimised # 

python3 xla_inference.py --use_xla_model --restore-path  ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_030_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707444383.h5

########## yp 50 ##########

# Baseline #

python3 xla_inference.py --use_native_tensorflow --restore-path ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_050_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707434638.h5

# XLA Optimised # 

python3 xla_inference.py --use_xla_model --restore-path  ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_050_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707434638.h5

########## yp 100 ##########

# Baseline #

python3 xla_inference.py --use_native_tensorflow --restore-path ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_0100_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707519212.h5

# XLA Optimised # 

python3 xla_inference.py --use_xla_model --restore-path  ./RefMap_Workflow/fcn_turbolence_prediction/converted_models/.baseline_models/WallRecon1TF2_3In-3Out_0100_192x192_Ret180_lr0.001_decay20drop0.5_relu-1707519212.h5