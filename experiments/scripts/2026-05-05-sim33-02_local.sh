#!/bin/bash

# List of scripts to run
scripts=(
  "python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=2 --M=30 --N=30 --seed=1001 --sample-numbers 300000 3000000 --beta=1.0 --output-dir=results/2026-05-05-sim33-02/beta_1.0 --wandb-run-name=2026-05-05-sim33-02_1.0"
  "python /home/jennifer/distortion/simulation_33.py --num-rounds=10 --ml-sampling-rounds=2 --M=30 --N=30 --seed=1001 --sample-numbers 300000 3000000 --beta=3.0 --output-dir=results/2026-05-05-sim33-02/beta_3.0 --wandb-run-name=2026-05-05-sim33-02_3.0"
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
