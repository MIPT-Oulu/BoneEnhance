#!/bin/bash
#SBATCH --job-name=BoneEnhance
#SBATCH --account=project_2002147
#SBATCH --mail-type=BEGIN,END #Send email when job starts
#SBATCH --partition=gputest
#SBATCH --time=0:15:00
#SBATCH --ntasks=3
#SBATCH --mem=350G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100:2

# Set up environment
export PROJAPPL=/projappl/project_2002147
csc-workspaces set project_2002147
module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.7
module load bioconda

# Paths
#ARCHIVED_DATA=/scratch/project_2002147/data.tar.gz
#DATA_DIR=${LOCAL_SCRATCH}/data
DATA_DIR=../../data
ENV_FILE=/scratch/project_2002147/rytkysan/BoneEnhance/BoneEnhance/requirements.txt

# Conda environment
source /projappl/project_2002147/miniconda3/etc/profile.d/conda.sh
conda activate bone-enhance-env
conda info --envs
#conda env create -f /scratch/project_2002147/rytkysan/BoneEnhance/BoneEnhance/environment.yml
echo $CONDA_EXE



#mkdir -p ${DATA_DIR}
#echo "Copying data to the node..."
#rsync ${ARCHIVED_DATA} ${DATA_DIR}
#echo "Done!"

#echo "Extracting the data..."
#tar -xzf ${DATA_DIR}/data.tar.gz -C ${DATA_DIR}
#echo "Done!"

echo "Start the job..."
#srun ./exp_csc_inference.sh ${SLURM_ARRAY_TASK_ID} ${DATA_DIR}/data
srun ./exp_csc_train.sh #${DATA_DIR} ${BS} ${SNAP} ${SAMPLE_ID}
echo "Done the job!"