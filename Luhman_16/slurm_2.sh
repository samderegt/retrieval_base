#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 13:00:00
#SBATCH -p genoa
#SBATCH --ntasks=192
#SBATCH --mem=336G

#SBATCH --job-name=freechem_K_B_ret_2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=regt@strw.leidenuniv.nl

# Activate the environment + load modules
source $HOME/activate_env
echo "Number of tasks $SLURM_NTASKS"

# Set the config file
config_file=config_freechem_K_B.py

# Run the pre-processing, retrieval and evaluation
python -u retrieval_script.py ${config_file} --setup
mpiexec --use-hwthread-cpus --bind-to none -np $SLURM_NTASKS python -u retrieval_script.py $config_file --run
python -u retrieval_script.py $config_file --evaluation

echo "Done"
