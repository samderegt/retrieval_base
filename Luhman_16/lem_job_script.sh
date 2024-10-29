#!/bin/bash

output_file=logs/fiducial_J_AB_1column_vdW.out
#NTASKS=120
NTASKS=80

# Run the pre-processing, retrieval and evaluation
config_file=config_fiducial_J_B_1column
python -u retrieval_script.py $config_file -p &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file -r >> $output_file 2>&1
mpiexec --use-hwthread-cpus --bind-to none -np 20 python -u retrieval_script.py $config_file -e >> $output_file 2>&1

# Run the pre-processing, retrieval and evaluation
config_file=config_fiducial_J_A_1column
python -u retrieval_script.py $config_file -p >> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np 20 python -u retrieval_script.py $config_file -r >> $output_file 2>&1
mpiexec --use-hwthread-cpus --bind-to none -np 20 python -u retrieval_script.py $config_file -e >> $output_file 2>&1