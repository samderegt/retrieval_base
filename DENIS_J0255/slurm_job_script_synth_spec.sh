#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 03:40:00
#SBATCH -p thin
#SBATCH -n 85
#SBATCH --mem=224G

#SBATCH --job-name=synthetic_new
#SBATCH --mail-type=ALL
#SBATCH --mail-user=regt@strw.leidenuniv.nl

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

# Run pre-processing and get the synthetic spectrum
sed -i 's/import config_DENIS as conf/import config_DENIS_synthetic_a as conf/g' retrieval.py
python retrieval.py --pre_processing --synthetic

# Replace the config file
sed -i 's/import config_DENIS_synthetic_a as conf/import config_DENIS_synthetic_b as conf/g' retrieval.py

# Run the retrieval
mpiexec -np $SLURM_NTASKS python retrieval.py --retrieval

sed -i 's/import config_DENIS_synthetic_b as conf/import config_DENIS_synthetic_a as conf/g' retrieval.py
python retrieval.py --evaluation

sed -i 's/import config_DENIS_synthetic_a as conf/import config_DENIS_synthetic_b as conf/g' retrieval.py
python retrieval.py --evaluation
sed -i 's/import config_DENIS_synthetic_b as conf/import config_DENIS as conf/g' retrieval.py

echo "Done"
