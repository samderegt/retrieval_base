#!/bin/bash

output_file=logs/new_fiducial_K_A_ret_14.out
NTASKS=100

# Run the pre-processing, retrieval and evaluation
config_file=config_fiducial_K_A
python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1