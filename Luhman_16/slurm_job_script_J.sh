#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 01:30:00
#SBATCH -p fat_genoa
#SBATCH --ntasks=48
#SBATCH --mem=1440G

#SBATCH --job-name=fiducial_J_B_ret_37_2column
#SBATCH --mail-type=ALL
#SBATCH --mail-user=regt@strw.leidenuniv.nl

# Activate the environment + load modules
source $HOME/activate_env

echo "Number of tasks $SLURM_NTASKS"
echo "Starting Python script"

#config_file=config_fiducial_J_A
config_file=config_fiducial_J_B
#config_file=config_fiducial_K_A
#config_file=config_fiducial_K_B

ret_script=retrieval_script.py

# Run the pre-processing
#python ${ret_script} ${config_file} -p
# Run the retrieval and evaluation
#mpiexec -np $SLURM_NTASKS --use-hwthread-cpus --bind-to none python ${ret_script} ${config_file} -r
mpiexec -np $SLURM_NTASKS --use-hwthread-cpus --bind-to none python ${ret_script} ${config_file} -e

echo "Done"
