#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 03:15:00
#SBATCH -p fat_genoa
#SBATCH --ntasks=192
#SBATCH --mem=1440G

#SBATCH --job-name=Luhman_16B_spot_eq_band_ret_10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=regt@strw.leidenuniv.nl

# Activate the environment + load modules
source $HOME/activate_env

echo "Number of tasks $SLURM_NTASKS"
echo "Starting Python script"

config_file=config_fiducial_K_B

# Replace the config file
sed -i "s/import config as conf/import ${config_file} as conf/g" retrieval_script.py

# Run the pre-processing
python retrieval_script.py -p
# Run the retrieval and evaluation
mpiexec -np $SLURM_NTASKS --use-hwthread-cpus --bind-to none python retrieval_script.py -r
mpiexec -np $SLURM_NTASKS --use-hwthread-cpus --bind-to none python retrieval_script.py -e

# Revert to original config file
sed -i "s/import ${config_file} as conf/import config as conf/g" retrieval_script.py

echo "Done"
