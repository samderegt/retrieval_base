#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 01:00:00
#SBATCH -p genoa
#SBATCH --ntasks=125
#SBATCH --mem=336G

#SBATCH --job-name=no_bands_K_B_ret_11
#SBATCH --mail-type=ALL
#SBATCH --mail-user=regt@strw.leidenuniv.nl

# Activate the environment + load modules
source $HOME/activate_env

echo "Number of tasks $SLURM_NTASKS"
echo "Starting Python script"

config_file=config_fiducial_K_B
ret_script=retrieval_script.py

# Replace the config file
sed -i "s/import config as conf/import ${config_file} as conf/g" ${ret_script}

# Run the pre-processing
#python ${ret_script} -p
# Run the retrieval and evaluation
#mpiexec -np $SLURM_NTASKS --use-hwthread-cpus --bind-to none python ${ret_script} -r
mpiexec -np $SLURM_NTASKS --use-hwthread-cpus --bind-to none python ${ret_script} -e

# Revert to original config file
sed -i "s/import ${config_file} as conf/import config as conf/g" ${ret_script}

echo "Done"
