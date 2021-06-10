#!/bin/bash

# 1 = Slurm task ID
# 2 = Random seed
# 3 = Number of threads
# 4 = Data directory
conda info --envs
module list
# Run training
python -m train_cluster --exp_idx $1 --seed $2 --num_threads $3 --data_location $4
