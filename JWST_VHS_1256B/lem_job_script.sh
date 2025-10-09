#!/bin/bash

NTASKS=50

output_file=logs/freechem_ret_2.out && config_file=config_all_gratings_freechem.py

# Run the pre-processing, retrieval and evaluation
python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
# python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1
