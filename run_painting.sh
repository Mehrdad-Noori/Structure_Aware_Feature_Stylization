#!/bin/bash

# Loop over each bash script in the domain-specific directory and submit them
for script in `ls ./bash/cc/painting/*.sh`
do
    sbatch $script
done
    