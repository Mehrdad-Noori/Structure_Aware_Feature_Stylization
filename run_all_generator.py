import os

# Directory paths
BASE_DIR = '.'
BASH_DIR = os.path.join(BASE_DIR, 'bash', 'resnet-50', 'domainnet', 'baseline')

# List of domains
domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

# Loop over each domain
for test_domain in domains:
    # Path for the domain-specific bash files
    domain_bash_dir = os.path.join(BASH_DIR, test_domain)

    # Bash script content
    bash_content = '''#!/bin/bash

# Loop over each bash script in the domain-specific directory and submit them
for script in `ls {}/*.sh`
do
    sbatch $script
done
    '''.format(domain_bash_dir)

    # Save the bash content to a new bash file
    bash_filename = f'run_{test_domain}.sh'
    bash_filepath = os.path.join(BASE_DIR, bash_filename)
    
    with open(bash_filepath, 'w') as bash_file:
        bash_file.write(bash_content)

print("Run bash scripts generated!")
