#!/bin/bash
echo "Inference Optimization Example for the midterm demonstration." 

########## yp 15 ##########

# Baseline # 

python3 compiler_optimization/xla_inference.py --use_native_tensorflow --restore-format hdf5

# XLA Optimised # 

python3 compiler_optimization/xla_inference.py --use_xla_model --restore-format hdf5