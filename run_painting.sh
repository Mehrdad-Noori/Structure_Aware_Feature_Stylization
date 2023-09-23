#!/bin/bash

# Loop over each bash script in the domain-specific directory and submit them
for script in `ls ./bash/resnet-50/domainnet/painting/*.sh`
do
    sbatch $script
done
    