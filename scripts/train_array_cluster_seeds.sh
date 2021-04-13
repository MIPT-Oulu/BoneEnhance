#!/bin/bash
# Slurm variables
#SBATCH --job-name=BoneEnhance_train_seeds
#SBATCH --account=project_2002147
#SBATCH --mail-type=END #Send email when job is finished
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:2
#SBATCH --array=1-9

# Set up environment
export PROJAPPL=/projappl/project_2002147
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projappl/project_2002147/miniconda3/lib
csc-workspaces set project_2002147

# Reduce amount of modules to focus use on the virtual Conda environment
module purge

# Paths
ENV_FILE=/scratch/project_2002147/rytkysan/BoneEnhance/BoneEnhance/environment.yml

# Conda environment
source /projappl/project_2002147/miniconda3/etc/profile.d/conda.sh
conda activate bone-enhance-env
conda info --envs
#conda env create -f ${ENV_FILE}

declare -a SEEDS=(10 20 30 40 50 60 70 80)

for SEED in "${SEEDS[@]}"
do
  echo "Start the job for seed ${SEED}..."
  srun ./exp_csc_train.sh ${SLURM_ARRAY_TASK_ID} ${SEED}
  echo "Done the job!"
done