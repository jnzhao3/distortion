"""
Generate experiment scripts for simulation_32.py, varying --num-samples.
Usage:
    python experiments/simulation_32_vary_samples.py --local   > run_local.sh
    python experiments/simulation_32_vary_samples.py --slurm   (writes slurm_*.sh)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from generate_local import LocalScriptGenerator
from generate import SbatchGenerator

# ── sweep ──────────────────────────────────────────────────────────────────
NUM_SAMPLES_SWEEP = [30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]

# ── shared constants ────────────────────────────────────────────────────────
COMMON = {
    "num-rounds":         10,
    "ml-sampling-rounds": 100,
    "M":                  30,
    "N":                  30,
    "seed":               1001,
    # "wandb-project":      "distortion-vary-samples",
}


def make_runs(generator):
    for n in NUM_SAMPLES_SWEEP:
        run_name = f"sim32_samples_{n}"
        generator.add_run({
            "num-samples":    n,
            "output-dir":     f"results/sim32/samples_{n}",
            "wandb-run-name": run_name,
        })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local",  action="store_true", help="Generate local bash script")
    p.add_argument("--slurm",  action="store_true", help="Generate SLURM sbatch scripts")
    args = p.parse_args()

    if not args.local and not args.slurm:
        p.print_help()
        sys.exit(1)

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "simulation_32.py")
    )
    prefix = ["python", script_path]
    for key, val in COMMON.items():
        prefix.append(f"--{key}={val}")

    scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    if args.local:
        gen = LocalScriptGenerator(prefix=prefix, gpus=[])
        make_runs(gen)
        fname = os.path.join(scripts_dir, "local_sim32_vary_samples.sh")
        with open(fname, "w") as f:
            f.write(gen.generate_str())
        print(f"Wrote {fname}")

    if args.slurm:
        gen = SbatchGenerator(
            prefix=prefix,
            j=1,
            limit=32,
            comment="sim32-vary-samples",
            priority="high",
        )
        make_runs(gen)
        scripts = gen.generate_str()
        for i, s in enumerate(scripts):
            fname = os.path.join(scripts_dir, f"slurm_sim32_vary_samples_part{i+1}.sh")
            with open(fname, "w") as f:
                f.write(s)
            print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
