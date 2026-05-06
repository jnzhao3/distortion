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
#SBATCH --qos=rail_gpu4_lowest
#SBATCH --requeue
#SBATCH --array=1-6%32
#SBATCH --comment=sim33-vary-beta-part1

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N=1
JOB_N=6

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  [1]='python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=100 --M=30 --N=30 --seed=1001 --sample-numbers=30000 300000 3000000 --beta=1.0 --output-dir=results/2026-05-05-sim33-01/beta_1.0 --wandb-run-name=2026-05-05-sim33-01_1.0'
  [2]='python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=100 --M=30 --N=30 --seed=1001 --sample-numbers=30000 300000 3000000 --beta=3.0 --output-dir=results/2026-05-05-sim33-01/beta_3.0 --wandb-run-name=2026-05-05-sim33-01_3.0'
  [3]='python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=100 --M=30 --N=30 --seed=1001 --sample-numbers=30000 300000 3000000 --beta=5.0 --output-dir=results/2026-05-05-sim33-01/beta_5.0 --wandb-run-name=2026-05-05-sim33-01_5.0'
  [4]='python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=100 --M=30 --N=30 --seed=1001 --sample-numbers=30000 300000 3000000 --beta=10.0 --output-dir=results/2026-05-05-sim33-01/beta_10.0 --wandb-run-name=2026-05-05-sim33-01_10.0'
  [5]='python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=100 --M=30 --N=30 --seed=1001 --sample-numbers=30000 300000 3000000 --beta=20.0 --output-dir=results/2026-05-05-sim33-01/beta_20.0 --wandb-run-name=2026-05-05-sim33-01_20.0'
  [6]='python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=100 --M=30 --N=30 --seed=1001 --sample-numbers=30000 300000 3000000 --beta=50.0 --output-dir=results/2026-05-05-sim33-01/beta_50.0 --wandb-run-name=2026-05-05-sim33-01_50.0'
)

parallel --delay 5s --linebuffer -j 1 {1} ::: "${commands[@]:$COM_ID_S:$PARALLEL_N}"
            