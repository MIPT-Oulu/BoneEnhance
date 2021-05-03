#!/bin/bash
# Slurm variables
#SBATCH --job-name=BoneEnhance_train
#SBATCH --account=project_2002147
#SBATCH --mail-type=END #Send email when job is finished
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:1

# Set up environment
#export PROJAPPL=/projappl/project_2002147
export SCRATCH=/scratch/project_2002147/rytkysan
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projappl/project_2002147/miniconda3/lib
#csc-workspaces set project_2002147

#module load gcc/8.3.0 cuda/10.1.168
#module load pytorch/1.7

# Conda environment
. ${SCRATCH}/miniconda3/etc/profile.d/conda.sh

#pip install hydra-core --user
#pip install -e ../../../solt/. --user
#pip install -e ../../../Collagen/. --user
#pip install -e .. --user
#pip install opencv-python --user

# Paths
#ENV_FILE=/scratch/project_2002147/rytkysan/BoneEnhance/BoneEnhance/environment.yml


#. ${PROJAPPL}/miniconda3/etc/profile.d/conda.sh
conda activate bone-enhance-env
#conda env create -f ${ENV_FILE} -prefix ${SCRATCH}/envs/bone-enhance-env
#. ./create_env.sh

echo "Start the job..."
declare -i SEED=42  # Random seed

for VALUE in {1..11}
do
  sbatch ./exp_csc_train.sh $VALUE ${SEED}
done

echo "Done the job!"