#!/bin/bash
# Slurm variables
#SBATCH --job-name=BoneEnhance_train_seeds
#SBATCH --account=project_2002147
#SBATCH --mail-type=END #Send email when job is finished
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=256G  # Running inference on large stacks requires at least 128GB memory
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-3

# Set up environment
export SCRATCH=/scratch/project_2002147/rytkysan

# Conda environment
. ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate bone-enhance-env

echo "Start the job..."
declare -i SEED=42  # Random seed
declare -i NUM_THREADS=10
# Data path
DATA_PATH=${SCRATCH}/BoneEnhance/Data
SNAP_PATH=${SCRATCH}/BoneEnhance/Workdir/dental_experiments

srun ./exp_csc_inference.sh ${SLURM_ARRAY_TASK_ID} ${SEED} ${NUM_THREADS} ${DATA_PATH} ${SNAP_PATH}
echo "Done the job!"