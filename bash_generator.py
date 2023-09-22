import os
import json

# Directory paths
BASE_DIR = '.'
BASE_DIR_IN_TARGET = '/home/milad97/projects/def-chdesa/milad97/safdg'

BASH_DIR = os.path.join(BASE_DIR, 'bash', 'resnet-50', 'domainnet' )
CONFIG_DIR = os.path.join(BASE_DIR, 'configs', 'resnet-50', 'domainnet')
CONFIG_DIR_IN_TARGET = os.path.join(BASE_DIR_IN_TARGET, 'configs',  'resnet-50' ,'domainnet')

# Bash script template
BASH_TEMPLATE = '''#!/bin/bash
#SBATCH --account=def-chdesa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=1-16:00

nvidia-smi

mkdir $SLURM_TMPDIR/data
cp /home/milad97/scratch/datasets/domain.tar.gz $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/domain.tar.gz -C $SLURM_TMPDIR/data

module load python/3.10
module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load opencv/4.5.5

source /home/milad97/projects/def-chdesa/milad97/envs/safdg/bin/activate

config_path={}
data_dir=$SLURM_TMPDIR/data/domain_net

python main.py --config ${{config_path}} --data_dir ${{data_dir}}
'''

# Configuration template
CONFIG_TEMPLATE = {
  "domains": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
  "test_domain": None,
  "backbone": "resnet50",
  "batch_size": 32,
  "num_epochs": 50,
  "num_workers": 4,
  "reconstruction": True,
  "feature_stylization": True,
  "save_path": "/home/milad97/projects/def-chdesa/milad97/safdg/output/domainnet",
  "lmda_value": None,
  "p_value": None,
  "lr": 0.01
}

# Make directories
os.makedirs(BASH_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Values for loop
p_values = [0.1]
lmda_values = [0.2]
domains = CONFIG_TEMPLATE["domains"]

for test_domain in domains:
    for p_value in p_values:
        for lmda_value in lmda_values:
            # Update configuration template
            config = CONFIG_TEMPLATE.copy()
            config['p_value'] = p_value
            config['lmda_value'] = lmda_value
            config['test_domain'] = test_domain

            # Create specific directories for the test domain
            domain_config_dir = os.path.join(CONFIG_DIR, test_domain)
            domain_config_dir_in_target = os.path.join(CONFIG_DIR_IN_TARGET, test_domain)
            domain_bash_dir = os.path.join(BASH_DIR, test_domain)
            os.makedirs(domain_config_dir, exist_ok=True)
            os.makedirs(domain_bash_dir, exist_ok=True)

            # Save configuration to file
            config_filename = f'config_p_{p_value}_lmda_{lmda_value}.json'
            config_filepath = os.path.join(domain_config_dir, config_filename)
            config_filepath_in_target = os.path.join(domain_config_dir_in_target, config_filename)
            with open(config_filepath, 'w') as config_file:
                json.dump(config, config_file, indent=2)

            # Generate corresponding bash script
            bash_script = BASH_TEMPLATE.format(config_filepath_in_target)
            bash_filename = f'{test_domain}_bash_p_{p_value}_lmda_{lmda_value}.sh'
            bash_filepath = os.path.join(domain_bash_dir, bash_filename)
            with open(bash_filepath, 'w') as bash_file:
                bash_file.write(bash_script)

print("Files generated!")
