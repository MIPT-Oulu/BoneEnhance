#!/bin/bash

# 1 = Slurm task ID
# 2 = Random seed

# Run training
python -m train_cluster --exp_idx $1 --seed $2
