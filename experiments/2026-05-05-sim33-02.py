"""
Generate experiment scripts for simulation_33.py, varying --beta.
Usage:
    python experiments/2026-05-05-vary-samples-33.py --local
    python experiments/2026-05-05-vary-samples-33.py --slurm
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from generate_local import LocalScriptGenerator
from generate import SbatchGenerator

# ── sweep ──────────────────────────────────────────────────────────────────
BETAS = [1.0, 3.0]

# ── shared constants ────────────────────────────────────────────────────────
NAME = '2026-05-05-sim33-02'
COMMON = {
    "num-rounds":          10,
    "ml-sampling-rounds":  2,
    "M":                   30,
    "N":                   30,
    "seed":                1001,
    "sample-numbers":      "300000 3000000",
    # "wandb-project":       "distortion-vary-samples-33",
}


def make_runs(generator):
    for beta in BETAS:
        generator.add_run({
            "beta":           beta,
            "output-dir":     f"results/{NAME}/beta_{beta}",
            "wandb-run-name": f"{NAME}_{beta}",
        })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local", action="store_true")
    p.add_argument("--slurm", action="store_true")
    args = p.parse_args()

    if not args.local and not args.slurm:
        p.print_help()
        sys.exit(1)

    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "simulation_33.py")
    )
    prefix = ["python", script_path]
    for key, val in COMMON.items():
        if isinstance(val, str) and ' ' in val:
            prefix.append(f"--{key}")
            prefix.extend(val.split())
        else:
            prefix.append(f"--{key}={val}")

    scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    if args.local:
        gen = LocalScriptGenerator(prefix=prefix, gpus=[])
        make_runs(gen)
        fname = os.path.join(scripts_dir, f"{NAME}_local.sh")
        with open(fname, "w") as f:
            f.write(gen.generate_str())
        print(f"Wrote {fname}")

    if args.slurm:
        gen = SbatchGenerator(
            prefix=prefix,
            j=1,
            limit=32,
            comment="sim33-vary-beta",
            priority="lowest",
        )
        make_runs(gen)
        for i, s in enumerate(gen.generate_str()):
            fname = os.path.join(scripts_dir, f"{NAME}_slurm.sh")
            with open(fname, "w") as f:
                f.write(s)
            print(f"Wrote {fname}")


if __name__ == "__main__":
    main()
