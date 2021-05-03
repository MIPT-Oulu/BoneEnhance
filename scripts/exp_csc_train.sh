#!/bin/bash

# 1 = Slurm task ID
# 2 = Random seed
conda info --envs
module list
# Run training
python -m train_cluster --exp_idx $1 --seed $2
