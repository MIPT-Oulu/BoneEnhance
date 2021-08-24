#!/bin/bash
# Slurm variables
#SBATCH --job-name=BoneEnhance_train_seeds
#SBATCH --account=project_2002147
#SBATCH --mail-type=END #Send email when job is finished
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:v100:1,nvme:50
#SBATCH --array=1-15

# Set up environment
export SCRATCH=/scratch/project_2002147/rytkysan

# Conda environment
. ${SCRATCH}/miniconda3/etc/profile.d/conda.sh
conda activate bone-enhance-env

# Copy data to local scratch
ARCHIVED_DATA=/scratch/project_2002147/rytkysan/BoneEnhance/Data
DATA_PATH=${LOCAL_SCRATCH}/Data
SNAP_PATH=${SCRATCH}/BoneEnhance/Workdir/wacv_experiments_new
SAVE_PATH=${SCRATCH}/BoneEnhance/Data
# Create data folder
mkdir -p ${DATA_PATH}
# Move data
rsync --inplace ${ARCHIVED_DATA}/target_1176_HR_2D.tar.gz ${DATA_PATH}
rsync --inplace ${ARCHIVED_DATA}/target_1176_HR.tar.gz ${DATA_PATH}
rsync --inplace ${ARCHIVED_DATA}/input_1176_HR.tar.gz ${DATA_PATH}
rsync --inplace ${ARCHIVED_DATA}/input_1176_HR_ds.tar.gz ${DATA_PATH}
rsync --inplace ${ARCHIVED_DATA}/input_1176_HR_2D.tar.gz ${DATA_PATH}
# Extract
tar -xf ${DATA_PATH}/target_1176_HR_2D.tar.gz -C ${DATA_PATH}
tar -xf ${DATA_PATH}/target_1176_HR.tar.gz -C ${DATA_PATH}
tar -xf ${DATA_PATH}/input_1176_HR.tar.gz -C ${DATA_PATH}
tar -xf ${DATA_PATH}/input_1176_HR_ds.tar.gz -C ${DATA_PATH}
tar -xf ${DATA_PATH}/input_1176_HR_2D.tar.gz -C ${DATA_PATH}

# Number of CPUs (match above)
declare -i NUM_THREADS=36
declare -i SEED=42  # Random seed

echo "Start the job..."
srun ./exp_csc_inference_oof.sh ${SLURM_ARRAY_TASK_ID} ${SEED} ${NUM_THREADS} ${DATA_PATH} ${SNAP_PATH} ${SAVE_PATH}
echo "Done the job!"
