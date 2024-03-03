#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 06:00:00
#SBATCH -p genoa
#SBATCH -n 65
#SBATCH --mem=336G

#SBATCH --job-name=chem_eq
#SBATCH --mail-type=ALL
#SBATCH --mail-user=regt@strw.leidenuniv.nl

echo "Current working directory:"
pwd

echo "HOME directory:"
echo $HOME

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load OpenBLAS/0.3.20-GCC-11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0
module load libarchive/3.6.1-GCCcore-11.3.0

# Activate virtual environment
source $HOME/retrieval_venv/bin/activate

# Export environment variables
export LD_LIBRARY_PATH=$HOME/retrieval_venv/MultiNest/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/retrieval_venv/SuiteSparse/lib:$LD_LIBRARY_PATH

export pRT_input_data_path=$HOME/retrieval_venv/pRT_input_data

echo "Number of tasks $SLURM_NTASKS"
echo "Starting Python script"

# --- chem-eq, Pquench ---------------------------------------------------------------
# Replace the config file and run pre-processing
sed -i 's/import config_DENIS as conf/import config_DENIS_chem_eq_Pquench as conf/g' retrieval.py
python retrieval.py --pre_processing

# Run the retrieval and evaluation
mpiexec -np $SLURM_NTASKS python retrieval.py --retrieval
python retrieval.py --evaluation

# Revert to original config file
sed -i 's/import config_DENIS_chem_eq_Pquench as conf/import config_DENIS as conf/g' retrieval.py

echo "Done"
