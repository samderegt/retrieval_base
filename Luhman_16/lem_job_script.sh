#!/bin/bash

export NTASKS=96

echo "Number of tasks $NTASKS"
echo "Starting Python script"

#source /net/lem/data1/regt/retrieval_venv/bin/activate.csh

# Replace the config file and run pre-processing
sed -i 's/import config as conf/import config_fiducial_K_B as conf/g' retrieval_script.py
#python retrieval_script.py -p

# Run the retrieval and evaluation
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python retrieval_script.py -r
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python retrieval_script.py -e

# Revert to original config file
sed -i 's/import config_fiducial_K_B as conf/import config as conf/g' retrieval_script.py

echo "Done"
