#!/bin/bash

NTASKS=100

# Run the pre-processing, retrieval and evaluation
#output_file=logs/K_A_freechem_ret_4.out
#config_file=config_freechem_K_A.py
#python -u retrieval_script.py $config_file --setup &> $output_file
#mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
#python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1

# Run the pre-processing, retrieval and evaluation
output_file=logs/K_B_freechem_ret_4.out
config_file=config_freechem_K_B.py
python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1