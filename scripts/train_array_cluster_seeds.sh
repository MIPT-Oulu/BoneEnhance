#!/bin/bash
# Slurm variables
#SBATCH --job-name=BoneEnhance_train_seeds
#SBATCH --account=project_2002147
#SBATCH --mail-type=END #Send email when job is finished
#SBATCH --partition=gpu
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-8

# Set up environment
export SCRATCH=/scratch/project_2002147/rytkysan

# Conda environment
. ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate bone-enhance-env

# Random seeds
declare -a SEEDS=(10 20 30 40 50)
# Number of CPUs (match above)
declare -i NUM_THREADS=36
# Data path
DATA_PATH=../../Data

for SEED in "${SEEDS[@]}"
do
  echo "Start the job for seed ${SEED}..."
  srun ./exp_csc_train.sh ${SLURM_ARRAY_TASK_ID} ${SEED} ${NUM_THREADS} ${DATA_PATH}
  echo "Done the job!"
done