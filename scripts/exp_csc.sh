#!/bin/bash
DATA_DIR=/scratch/project_2002147/rytkysan/data
python -m scripts.thickness_analysis_EP --batch_id ${SLURM_ARRAY_TASK_ID} \
                                        --masks ${DATA_DIR}/Predictions_FPN_Resnet18\
                                        --th_maps ${DATA_DIR}/Thickness_cluster