#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 24:00:00
#SBATCH -p fat_genoa
#SBATCH --ntasks=192
#SBATCH --mem=1440G

#SBATCH --job-name=all_gratings_eqchem_ret_1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=regt@strw.leidenuniv.nl

# Activate the environment + load modules
source $HOME/activate_env
echo "Number of tasks $SLURM_NTASKS"

# Set the config file
config_file=config_all_gratings_eqchem.py

# Run the pre-processing, retrieval and evaluation
python retrieval_script.py ${config_file} --setup
mpiexec --use-hwthread-cpus --bind-to none -np $SLURM_NTASKS python -u retrieval_script.py $config_file --run
python -u retrieval_script.py $config_file --evaluation

echo "Done"
