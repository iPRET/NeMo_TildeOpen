#!/bin/bash
#SBATCH --account=jureap133 \
#SBATCH --partition=booster \
#SBATCH --nodes=256 \
#SBATCH --ntasks-per-node=1 \
#SBATCH --gres=gpu:4 \
#SBATCH --cpus-per-task=288 \
#SBATCH --time=3:00:00 \
#SBATCH --reservation=campaign \
#SBATCH --exclude=jpbo-055-37 \
#SBATCH --mem=0 \

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

ulimit -c 0

rm ./core.jpbo*

scontrol show hostname $SLURM_NODELIST > nodelist_$SLURM_JOB_ID.txt

srun distil_real_singularityscript_layers.sh
