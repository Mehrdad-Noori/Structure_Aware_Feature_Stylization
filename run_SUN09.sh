#!/bin/bash

# Loop over each bash script in the domain-specific directory and submit them
for script in `ls ./bash/resnet-50/vlcs/SUN09/*.sh`
do
    sbatch $script
done
    