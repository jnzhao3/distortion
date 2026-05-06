from typing import Iterable, Sequence


class LocalScriptGenerator:
    def __init__(
        self,
        prefix=("python main.py",),
        gpus: Sequence[int] = (0, 1, 2, 4, 5, 6, 7),
    ):
        self.prefix = list(prefix)
        self.commands = []
        self.gpus = list(gpus)

    def add_common_prefix(self, args):
        for key, value in args.items():
            self.prefix.append(f"--{key}={value}")

    def add_run(self, args):
        command_comps = []
        command_comps.extend(self.prefix)
        for key, value in args.items():
            command_comps.append(f"--{key}={value}")
        self.commands.append(" ".join(command_comps))

    @staticmethod
    def _escape_bash(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def generate_str(self):
        scripts_block = "\n".join(
            f'  "{self._escape_bash(command)}"'
            for command in self.commands
        )
        gpus_block = " ".join(str(gpu_id) for gpu_id in self.gpus)

        return f"""#!/bin/bash

# List of scripts to run
scripts=(
{scripts_block}
)

# List of available GPU IDs (modify as needed)
gpus=({gpus_block})

num_gpus=${{#gpus[@]}}
num_scripts=${{#scripts[@]}}

# Store PIDs of background jobs
pids=()

# Function to handle Ctrl+C
cleanup() {{
  echo "Terminating all running processes..."
  for pid in "${{pids[@]}}"; do
    kill "$pid" 2>/dev/null
  done
  wait
  exit 1
}}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Function to run scripts sequentially on a given GPU
run_on_gpu() {{
  local gpu_id=$1
  shift
  local gpu_scripts=("$@")

  for script in "${{gpu_scripts[@]}}"; do
    echo "Running $script on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id eval "$script" &
    pids+=($!)  # Store PID of the process
    wait ${{pids[-1]}}  # Wait for the process to finish before moving to the next
  done
}}

# Distribute scripts among GPUs
for ((i=0; i<num_gpus; i++)); do
  gpu_scripts=()

  # Assign every nth script to this GPU
  for ((j=i; j<num_scripts; j+=num_gpus)); do
    gpu_scripts+=("${{scripts[j]}}")
  done

  if [ ${{#gpu_scripts[@]}} -gt 0 ]; then
    run_on_gpu ${{gpus[i]}} "${{gpu_scripts[@]}}" &
    pids+=($!)  # Store PID of background process
  fi
done

wait  # Wait for all background jobs
echo "All scripts finished."
"""
