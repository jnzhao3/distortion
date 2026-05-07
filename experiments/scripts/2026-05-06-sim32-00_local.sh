#!/bin/bash

# List of scripts to run
scripts=(
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=30000 --output-dir=results/sim32/samples_30000 --wandb-run-name=sim32_samples_30000 --weight-base=1.1"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=100000 --output-dir=results/sim32/samples_100000 --wandb-run-name=sim32_samples_100000 --weight-base=1.1"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=300000 --output-dir=results/sim32/samples_300000 --wandb-run-name=sim32_samples_300000 --weight-base=1.1"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=1000000 --output-dir=results/sim32/samples_1000000 --wandb-run-name=sim32_samples_1000000 --weight-base=1.1"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=3000000 --output-dir=results/sim32/samples_3000000 --wandb-run-name=sim32_samples_3000000 --weight-base=1.1"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=10000000 --output-dir=results/sim32/samples_10000000 --wandb-run-name=sim32_samples_10000000 --weight-base=1.1"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=30000 --output-dir=results/sim32/samples_30000 --wandb-run-name=sim32_samples_30000 --weight-base=1.5"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=100000 --output-dir=results/sim32/samples_100000 --wandb-run-name=sim32_samples_100000 --weight-base=1.5"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=300000 --output-dir=results/sim32/samples_300000 --wandb-run-name=sim32_samples_300000 --weight-base=1.5"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=1000000 --output-dir=results/sim32/samples_1000000 --wandb-run-name=sim32_samples_1000000 --weight-base=1.5"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=3000000 --output-dir=results/sim32/samples_3000000 --wandb-run-name=sim32_samples_3000000 --weight-base=1.5"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=10000000 --output-dir=results/sim32/samples_10000000 --wandb-run-name=sim32_samples_10000000 --weight-base=1.5"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=30000 --output-dir=results/sim32/samples_30000 --wandb-run-name=sim32_samples_30000 --weight-base=2.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=100000 --output-dir=results/sim32/samples_100000 --wandb-run-name=sim32_samples_100000 --weight-base=2.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=300000 --output-dir=results/sim32/samples_300000 --wandb-run-name=sim32_samples_300000 --weight-base=2.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=1000000 --output-dir=results/sim32/samples_1000000 --wandb-run-name=sim32_samples_1000000 --weight-base=2.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=3000000 --output-dir=results/sim32/samples_3000000 --wandb-run-name=sim32_samples_3000000 --weight-base=2.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=10000000 --output-dir=results/sim32/samples_10000000 --wandb-run-name=sim32_samples_10000000 --weight-base=2.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=30000 --output-dir=results/sim32/samples_30000 --wandb-run-name=sim32_samples_30000 --weight-base=5.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=100000 --output-dir=results/sim32/samples_100000 --wandb-run-name=sim32_samples_100000 --weight-base=5.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=300000 --output-dir=results/sim32/samples_300000 --wandb-run-name=sim32_samples_300000 --weight-base=5.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=1000000 --output-dir=results/sim32/samples_1000000 --wandb-run-name=sim32_samples_1000000 --weight-base=5.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=3000000 --output-dir=results/sim32/samples_3000000 --wandb-run-name=sim32_samples_3000000 --weight-base=5.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=10000000 --output-dir=results/sim32/samples_10000000 --wandb-run-name=sim32_samples_10000000 --weight-base=5.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=30000 --output-dir=results/sim32/samples_30000 --wandb-run-name=sim32_samples_30000 --weight-base=10.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=100000 --output-dir=results/sim32/samples_100000 --wandb-run-name=sim32_samples_100000 --weight-base=10.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=300000 --output-dir=results/sim32/samples_300000 --wandb-run-name=sim32_samples_300000 --weight-base=10.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=1000000 --output-dir=results/sim32/samples_1000000 --wandb-run-name=sim32_samples_1000000 --weight-base=10.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=3000000 --output-dir=results/sim32/samples_3000000 --wandb-run-name=sim32_samples_3000000 --weight-base=10.0"
  "python simulation_32.py --num-rounds=10 --ml-sampling-rounds=10 --M=30 --N=30 --seed=1001 --num-samples=10000000 --output-dir=results/sim32/samples_10000000 --wandb-run-name=sim32_samples_10000000 --weight-base=10.0"
)

# List of available GPU IDs (modify as needed)
gpus=()

num_gpus=${#gpus[@]}
num_scripts=${#scripts[@]}

# Store PIDs of background jobs
pids=()

# Function to handle Ctrl+C
cleanup() {
  echo "Terminating all running processes..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null
  done
  wait
  exit 1
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Function to run scripts sequentially on a given GPU
run_on_gpu() {
  local gpu_id=$1
  shift
  local gpu_scripts=("$@")

  for script in "${gpu_scripts[@]}"; do
    echo "Running $script on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id eval "$script" &
    pids+=($!)  # Store PID of the process
    wait ${pids[-1]}  # Wait for the process to finish before moving to the next
  done
}

# Distribute scripts among GPUs
for ((i=0; i<num_gpus; i++)); do
  gpu_scripts=()

  # Assign every nth script to this GPU
  for ((j=i; j<num_scripts; j+=num_gpus)); do
    gpu_scripts+=("${scripts[j]}")
  done

  if [ ${#gpu_scripts[@]} -gt 0 ]; then
    run_on_gpu ${gpus[i]} "${gpu_scripts[@]}" &
    pids+=($!)  # Store PID of background process
  fi
done

wait  # Wait for all background jobs
echo "All scripts finished."
