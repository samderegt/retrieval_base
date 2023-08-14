#!/bin/bash

# Set job requirements
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -t 17:00:00
#SBATCH -p thin
#SBATCH -n 32
#SBATCH --mem=56G

#SBATCH --job-name=eval
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

# Replace the config file and run evaluation
cp retrieval.py retrieval_nominal_2.py
sed -i 's/import config_DENIS as conf/import config_DENIS_nominal_2 as conf/g' retrieval_nominal_2.py

cp retrieval.py retrieval_wo_13CO.py
sed -i 's/import config_DENIS as conf/import config_DENIS_wo_13CO as conf/g' retrieval_wo_13CO.py

cp retrieval.py retrieval_wo_NH3.py
sed -i 's/import config_DENIS as conf/import config_DENIS_wo_NH3 as conf/g' retrieval_wo_NH3.py

cp retrieval.py retrieval_wo_CH4.py
sed -i 's/import config_DENIS as conf/import config_DENIS_wo_CH4 as conf/g' retrieval_wo_CH4.py

#cp retrieval.py retrieval_H2O_HITEMP.py
#sed -i 's/import config_DENIS as conf/import config_DENIS_H2O_HITEMP as conf/g' retrieval_H2O_HITEMP.py

#cp retrieval.py retrieval_chem_eq_P_quench.py
#sed -i 's/import config_DENIS as conf/import config_DENIS_chem_eq_P_quench as conf/g' retrieval_chem_eq_P_quench.py

#cp retrieval.py retrieval_chem_eq_wo_P_quench.py
#sed -i 's/import config_DENIS as conf/import config_DENIS_chem_eq_wo_P_quench as conf/g' retrieval_chem_eq_wo_P_quench.py

python retrieval_nominal_2.py --evaluation &
python retrieval_wo_13CO.py --evaluation &
python retrieval_wo_NH3.py --evaluation &
python retrieval_wo_CH4.py --evaluation &
wait
#python retrieval_H2O_HITEMP.py --evaluation
#python retrieval_chem_eq_P_quench.py --evaluation
#python retrieval_chem_eq_wo_P_quench.py --evaluation

rm -rf retrieval_*.py

echo "Done"
