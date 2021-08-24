#!/bin/bash

# Input variables
DATA_DIR=/scratch/project_2002147/rytkysan/BoneEnhance/Data
SNAP=../../Workdir/snapshots/2021_05_05_10_55_23_2D_perceptual_tv_1176_seed10
#SNAP=../../Workdir/snapshots/2021_03_04_10_11_34_1_3D_mse_tv_1176
PRED_PATH=$4/predictions_oof_wacv  # Save images to local scratch
EVAL_PATH=$6/evaluation_oof_wacv  # Metrics to scratch
declare -i BS=64
declare -i SAMPLE_ID=1
declare -i STEP=3

# 1 = Slurm task ID
# 2 = Random seed
# 3 = Number of threads
# 4 = Data directory
# 5 = Snap directory
echo ${PRED_PATH}
# Run inference
python -m inference_evaluate_oof --bs ${BS} --snap_id $1 --data_location $4 --snapshots $5 --eval_dir ${EVAL_PATH} --save_dir ${PRED_PATH}
