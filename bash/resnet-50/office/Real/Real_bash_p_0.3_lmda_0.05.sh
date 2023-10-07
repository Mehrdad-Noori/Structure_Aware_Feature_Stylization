#!/bin/bash
#SBATCH --account=def-chdesa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=0-23:00

nvidia-smi

mkdir $SLURM_TMPDIR/data
cp /home/milad97/scratch/datasets/office2.tar.gz $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/office2.tar.gz -C $SLURM_TMPDIR/data

module load python/3.10
module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load opencv/4.5.5

source /home/milad97/projects/def-chdesa/milad97/envs/safdg/bin/activate

config_path=/home/milad97/projects/def-chdesa/milad97/safdg/configs/resnet-50/office/Real/config_p_0.3_lmda_0.05.json
data_dir=$SLURM_TMPDIR/data/office_home

python main.py --config ${config_path} --data_dir ${data_dir}
