#!/bin/bash

# Input variables
DATA_DIR=/scratch/project_2002147/rytkysan/BoneEnhance/Data
SNAP=../../Workdir/snapshots/2021_03_03_11_52_07_1_3D_mse_tv_1176_HR
#SNAP=../../Workdir/snapshots/2021_03_04_10_11_34_1_3D_mse_tv_1176
declare -i BS=64
declare -i SAMPLE_ID=1
declare -i STEP=3

# Run inference
python -m inference_cluster --dataset_root ${DATA_DIR}/Clinical_data --save_dir ${DATA_DIR}/Results --bs ${BS} --snapshot ${SNAP} --sample_id ${SAMPLE_ID} --step ${STEP}
