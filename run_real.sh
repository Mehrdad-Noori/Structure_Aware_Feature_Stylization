#!/bin/bash

# Loop over each bash script in the domain-specific directory and submit them
for script in `ls ./bash/cc/real/*.sh`
do
    sbatch $script
done
    