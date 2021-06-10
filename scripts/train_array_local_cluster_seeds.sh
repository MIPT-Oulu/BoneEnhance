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
#SBATCH --gres=gpu:v100:1,nvme:10
#SBATCH --array=1-6

# Set up environment
export SCRATCH=/scratch/project_2002147/rytkysan

# Conda environment
. ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate bone-enhance-env

# Copy data to local scratch
ARCHIVED_DATA=/scratch/project_2002147/rytkysan/BoneEnhance/Data
DATA_PATH=${LOCAL_SCRATCH}/Data
# Create data folder
mkdir -p ${DATA_PATH}
# Move data
rsync --inplace ${ARCHIVED_DATA}/target_1176_HR_2D.tar.gz ${DATA_PATH}
rsync --inplace ${ARCHIVED_DATA}/target_1176_HR.tar.gz ${DATA_PATH}
rsync --inplace ${ARCHIVED_DATA}/input_1176_HR_ds.tar.gz ${DATA_PATH}
# Extract
tar -xf ${DATA_PATH}/target_1176_HR_2D.tar.gz -C ${DATA_PATH}
tar -xf ${DATA_PATH}/target_1176_HR.tar.gz -C ${DATA_PATH}
tar -xf ${DATA_PATH}/input_1176_HR_ds.tar.gz -C ${DATA_PATH}

# Random seeds
declare -a SEEDS=(30 40 50 10 20)
# Number of CPUs (match above)
declare -i NUM_THREADS=36

for SEED in "${SEEDS[@]}"
do
  echo "Start the job for seed ${SEED}..."
  srun ./exp_csc_train.sh ${SLURM_ARRAY_TASK_ID} ${SEED} ${NUM_THREADS} ${DATA_PATH}
  echo "Done the job!"
done