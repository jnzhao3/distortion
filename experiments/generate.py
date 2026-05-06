import os

LIMIT = 800

class SbatchGenerator:
    def __init__(self, prefix=("MUJOCO_GL=egl", "python main.py"), j=1, limit=32, comment="default", priority="lowest"):
        self.prefix = list(prefix)
        self.commands = []
        self.comment = comment
        self.j = j
        self.limit = limit
        self.priority = priority

    def add_common_prefix(self, args):
        for key, value in args.items():
            self.prefix.append(f"--{key}={value}")

    def add_run(self, args):
        command_comps = []
        command_comps.extend(self.prefix)
        for key, value in args.items():
            command_comps.append(f"--{key}={value}")
        self.commands.append(" ".join(command_comps))

    def generate_str(self):

        num_jobs = len(self.commands)

        num_scripts = (num_jobs - 1) // LIMIT + 1
        sbatch_str_list = []
        for script_index in range(num_scripts):
            commands = self.commands[script_index * LIMIT: (script_index + 1) * LIMIT]
            num_jobs_partial = len(commands)

            num_arr = (num_jobs_partial - 1) // self.j + 1
            # print("\n".join(self.commands))

            path = "~" + os.getcwd()[len(os.path.expanduser("~")):]

            d_str = "\n  ".join(
                [
                    "[{}]='{}'".format(i + 1, command)
                    for i, command in enumerate(commands)
                ]
            )

            sbatch_str = f"""#!/bin/bash
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
#SBATCH --qos=rail_gpu4_{self.priority}
#SBATCH --requeue
#SBATCH --array=1-{num_arr}%{self.limit}
#SBATCH --comment={self.comment}-part{script_index+1}

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N={self.j}
JOB_N={num_jobs_partial}

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))
# module load gnu-parallel
source ~/.bashrc
micromamba activate aorl


declare -a commands=(
  {d_str}
)

parallel --delay 5s --linebuffer -j {self.j} {{1}} ::: \"${{commands[@]:$COM_ID_S:$PARALLEL_N}}\"
            """
            sbatch_str_list.append(sbatch_str)
        
        return sbatch_str_list