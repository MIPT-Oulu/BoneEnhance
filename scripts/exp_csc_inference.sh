#!/bin/bash

# Input variables
DATA_DIR=/scratch/project_2002147/rytkysan/BoneEnhance/Data
SNAP=../../Workdir/snapshots/2021_05_05_10_55_23_2D_perceptual_tv_1176_seed10
#SNAP=../../Workdir/snapshots/2021_03_04_10_11_34_1_3D_mse_tv_1176
declare -i BS=64
declare -i SAMPLE_ID=3
declare -i STEP=3

PRED_PATH=$4/predictions_oof_wacv  # Save images to local scratch

# 1 = Slurm task ID
# 2 = Random seed
# 3 = Number of threads
# 4 = Data directory
# 5 = Snap directory

# Run inference
python -m inference_cluster --dataset_root $4/Clinical_data --save_dir $4/Results --bs ${BS} --snapshot $5 --sample_id ${SAMPLE_ID} --step ${STEP} --snap_id $1
