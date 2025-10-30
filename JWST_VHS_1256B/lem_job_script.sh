#!/bin/bash

NTASKS=50

# Run the pre-processing, retrieval and evaluation
output_file=logs/all_gratings_freechem_ret_3_1column.out && config_file=config_all_gratings_freechem.py
# python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1