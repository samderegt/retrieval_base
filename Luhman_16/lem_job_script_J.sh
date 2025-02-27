#!/bin/bash

NTASKS=60

# Run the pre-processing, retrieval and evaluation
output_file=logs/J_B_ret_62_parameterised_shift.out
config_file=config_fiducial_J_B_1column_parameterised_shift.py
python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1