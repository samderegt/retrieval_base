#!/bin/bash

NTASKS=100

output_file=logs/g395h_nrs2_freechem_ret_3.out && config_file=config_g395h_nrs2_freechem.py

# Run the pre-processing, retrieval and evaluation
python -u retrieval_script.py $config_file --setup &> $output_file
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1


# output_file=logs/g235h_nrs2_freechem_ret_1.out && config_file=config_g235h_nrs2_freechem.py

# # Run the pre-processing, retrieval and evaluation
# python -u retrieval_script.py $config_file --setup &> $output_file
# mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python -u retrieval_script.py $config_file --run >> $output_file 2>&1
# python -u retrieval_script.py $config_file --evaluation >> $output_file 2>&1