#!/bin/bash
#SBATCH --account=def-chdesa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=0-20:00

nvidia-smi

mkdir $SLURM_TMPDIR/data
cp /home/milad97/scratch/datasets/domain.tar.gz $SLURM_TMPDIR
tar -xf $SLURM_TMPDIR/domain.tar.gz -C $SLURM_TMPDIR/data

module load python/3.10
module load StdEnv/2020 gcc/9.3.0 cuda/11.4
module load opencv/4.5.5

source /home/milad97/projects/def-chdesa/milad97/envs/safdg/bin/activate

config_path=/home/milad97/projects/def-chdesa/milad97/safdg/configs/domainnet/real/config_p_0.1_lmda_0.1.json
data_dir=$SLURM_TMPDIR/data/domain_net

python main.py --config ${config_path} --data_dir ${data_dir}
