#!/bin/bash
# Paths
ENV_FILE=${SCRATCH}/BoneEnhance/BoneEnhance/environment.yml
echo ${ENV_FILE}
# Conda environment
. /projappl/project_2002147/miniconda3/etc/profile.d/conda.sh
conda env create -f ${ENV_FILE} --prefix ${SCRATCH}/envs/bone-cluster-env
conda activate bone-cluster-env

cd ${SCRATCH} || exit

# Install Collagen
git clone https://github.com/MIPT-Oulu/Collagen.git
(
cd ./Collagen || exit
pip install -e .
cd ..
)
