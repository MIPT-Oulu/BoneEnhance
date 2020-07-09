#!/bin/bash
#SBATCH --job-name=RabbitThickness
#SBATCH --account=project_2002147
#SBATCH --partition=small
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=16000
#SBATCH --array=1-28

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.4
module load bioconda

#ARCHIVED_DATA=/scratch/project_2002147/data.tar.gz
#DATA_DIR=${LOCAL_SCRATCH}/data
DATA_DIR=../../data

#conda env create -n rabbit_ccs -f requirements_short.txt
#source activate rabbit_ccs

#mkdir -p ${DATA_DIR}
#echo "Copying data to the node..."
#rsync ${ARCHIVED_DATA} ${DATA_DIR}
#echo "Done!"

#echo "Extracting the data..."
#tar -xzf ${DATA_DIR}/data.tar.gz -C ${DATA_DIR}
#echo "Done!"

echo "Start the job..."
srun ./exp_csc.sh ${SLURM_ARRAY_TASK_ID} ${DATA_DIR}/data
echo "Done the job!"