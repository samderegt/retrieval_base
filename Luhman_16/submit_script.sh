config_file=config_fiducial_J_B
ret_script=retrieval_script.py

# Activate the environment + load modules
source $HOME/activate_env

# Replace the config file
sed -i "s/import config as conf/import ${config_file} as conf/g" ${ret_script}

# Run the pre-processing
python ${ret_script} -p

sbatch slurm_job_script.sh