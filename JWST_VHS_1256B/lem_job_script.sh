#!/bin/bash

NTASKS=120

#output_file=logs/g140h_ret_1.out && config_file=config_g140h.py
#output_file=logs/g235h_ret_6.out && config_file=config_g235h.py
output_file=logs/g395h_ret_4.out && config_file=config_g395h_freechem.py

# Run the pre-processing, retrieval and evaluation
python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1