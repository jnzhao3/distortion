#!/bin/bash
#SBATCH --job-name=aorl
#SBATCH --open-mode=append
#SBATCH -o /global/scratch/users/jenniferzhao/logs/%A_%a.out
#SBATCH -e /global/scratch/users/jenniferzhao/logs/%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A5000:1
#SBATCH --account=co_rail
#SBATCH --partition=savio4_gpu
#SBATCH --qos=rail_gpu4_high
#SBATCH --requeue
#SBATCH --array=1-6%32
#SBATCH --comment=sim32-vary-samples-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=6

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='python simulation_32.py --num-rounds=10 --ml-sampling-rounds=1000 --M=30 --N=30 --seed=1001 --num-samples=30000 --output-dir=results/sim32/samples_30000 --wandb-run-name=sim32_samples_30000'
  [2]='python simulation_32.py --num-rounds=10 --ml-sampling-rounds=1000 --M=30 --N=30 --seed=1001 --num-samples=100000 --output-dir=results/sim32/samples_100000 --wandb-run-name=sim32_samples_100000'
  [3]='python simulation_32.py --num-rounds=10 --ml-sampling-rounds=1000 --M=30 --N=30 --seed=1001 --num-samples=300000 --output-dir=results/sim32/samples_300000 --wandb-run-name=sim32_samples_300000'
  [4]='python simulation_32.py --num-rounds=10 --ml-sampling-rounds=1000 --M=30 --N=30 --seed=1001 --num-samples=1000000 --output-dir=results/sim32/samples_1000000 --wandb-run-name=sim32_samples_1000000'
  [5]='python simulation_32.py --num-rounds=10 --ml-sampling-rounds=1000 --M=30 --N=30 --seed=1001 --num-samples=3000000 --output-dir=results/sim32/samples_3000000 --wandb-run-name=sim32_samples_3000000'
  [6]='python simulation_32.py --num-rounds=10 --ml-sampling-rounds=1000 --M=30 --N=30 --seed=1001 --num-samples=10000000 --output-dir=results/sim32/samples_10000000 --wandb-run-name=sim32_samples_10000000'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            