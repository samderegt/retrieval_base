#!/bin/bash

# Number of mpi-processes to start, 
# check that this number is un-occupied with htop
export NTASKS=85

# Settings you want to use
config_file=config

echo "Number of tasks $NTASKS"
echo "Starting Python script"

# Make sure to have activated virtualenv in terminal 
# where this script is run
#source /net/lem/data1/regt/retrieval_venv/bin/activate.csh

# Replace the config file and run pre-processing
sed -i "s/import config as conf/import ${config_file} as conf/g" retrieval_script.py
python retrieval_script.py -p

# Run the retrieval and evaluation
mpiexec --use-hwthread-cpus --bind-to none -np $NTASKS python retrieval_script.py -r
mpiexec --use-hwthread-cpus --bind-to none -np 20 python retrieval_script.py -e

# Revert to original config file
sed -i "s/import ${config_file} as conf/import config as conf/g" retrieval_script.py

echo "Done"

# To save the terminal output, run this script as:
# sh lem_job_script.sh >& logs/some_name.out &

# Terminal output can be checked via e.g.:
# tail -200 logs/some_name.out