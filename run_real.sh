#!/bin/bash

# Loop over each bash script in the domain-specific directory and submit them
for script in `ls ./bash/resnet-50/domainnet/real/*.sh`
do
    sbatch $script
done
    