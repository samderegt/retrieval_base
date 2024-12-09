#!/bin/bash

output_file=logs/test.out
NTASKS=50

# Run the pre-processing, retrieval and evaluation
config_file=config_fiducial_K_A_new
python -u retrieval_script_new.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script_new.py $config_file --run >> $output_file 2>&1
python -u retrieval_script_new.py $config_file --evaluation >> $output_file 2>&1