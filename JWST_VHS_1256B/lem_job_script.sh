#!/bin/bash

NTASKS=100

output_file=logs/g395h_ret_2.out
config_file=config_g395h.py

# Run the pre-processing, retrieval and evaluation
python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1