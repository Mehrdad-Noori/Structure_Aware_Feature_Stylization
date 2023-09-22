import os

# Base paths
bash_base_path = "./bash/cc"
output_base_path = "./output/domainnet"

# List of domain names
# Assuming domain names are subdirectories in the bash_base_path
domain_names = [d for d in os.listdir(bash_base_path) if os.path.isdir(os.path.join(bash_base_path, d))]

unfinished_jobs = []

# Iterate through each domain name
for domain_name in domain_names:
    domain_path = os.path.join(bash_base_path, domain_name)
    
    # Get all bash files in the domain directory
    bash_files = [f for f in os.listdir(domain_path) if f.endswith('.sh')]
    
    for bash_file in bash_files:
        # Extract p_value and lmda_value from the bash file name
        parts = bash_file.replace('.sh', '').split('_')
        p_value_index = parts.index('p') + 1
        lmda_value_index = parts.index('lmda') + 1
        
        p_value = parts[p_value_index]
        lmda_value = parts[lmda_value_index]
        
        # Check if summary.json exists for this job
        summary_path = os.path.join(output_base_path, domain_name, f"lmda_{lmda_value}_p_{p_value}", "summary.json")
        if not os.path.exists(summary_path):
            unfinished_jobs.append(os.path.join(domain_path, bash_file))

# Write all unfinished jobs to a bash file
with open("run_unfinished_jobs.sh", "w") as out_file:
    out_file.write("#!/bin/bash\n\n")
    for job in unfinished_jobs:
        out_file.write(f"sbatch {job}\n")

print(f"Unfinished jobs written to run_unfinished_jobs.sh")

