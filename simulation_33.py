import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import wandb
from datasets import load_dataset
from scipy.special import expit
from tqdm import tqdm

from utils_3 import PairwiseData, Population, leaderboard_dist
from utils_4 import (
    ij_from_pairwise,
    borda_ranking, borda_peeling_ranking,
    copeland_ranking, copeland_peeling_ranking,
    ml_argmax_ranking, ml_nonzero_ranking,
    make_ml_fn_from_ij_wins,
    expected_leaderboard_distortion, expected_leaderboard_distortion_w,
)
from utils_4 import _solve_maximal_lottery


def leaderboard_dist_w(ranking, true_ranking, avg_utils, w):
    ranking = np.asarray(ranking)
    true_ranking = np.asarray(true_ranking)
    ranking_utils = avg_utils[ranking]
    true_ranking_utils = avg_utils[true_ranking]
    denom_sum = (ranking_utils * w).sum()
    ratio = (true_ranking_utils * w).sum() / denom_sum if denom_sum > 0 else np.inf
    return ratio, None


def sampled_ranking_dist(candidates, ij_wins, tol=1e-12, rounds=10):
    lp_cache = {}
    dist = {}
    for _ in range(rounds):
        remaining = list(candidates)
        ranking = []
        while remaining:
            key = frozenset(remaining)
            if key not in lp_cache:
                rem = np.array(remaining)
                sub = ij_wins[np.ix_(rem, rem)]
                with np.errstate(invalid='ignore', divide='ignore'):
                    margins = np.where(sub + sub.T > 0, (sub - sub.T) / (sub + sub.T), 0.0)
                lp_cache[key] = _solve_maximal_lottery(margins, tol)
            lot = lp_cache[key]
            w_idx = np.random.choice(len(remaining), p=lot)
            ranking.append(remaining[w_idx])
            remaining.pop(w_idx)
        r = tuple(ranking)
        dist[r] = dist.get(r, 0.0) + 1.0 / rounds
    return dist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--beta',                 type=float, default=5.0)
    p.add_argument('--num-rounds',           type=int,   default=10)
    p.add_argument('--ml-sampling-rounds',   type=int,   default=100)
    p.add_argument('--M',                    type=int,   default=30)
    p.add_argument('--N',                    type=int,   default=30)
    p.add_argument('--seed',                 type=int,   default=1001)
    p.add_argument('--sample-numbers',       type=int,   nargs='+',
                   default=[30_000, 300_000, 3_000_000, 30_000_000])
    p.add_argument('--output-dir',           type=str,   default='results')
    p.add_argument('--wandb-project',        type=str,   default='distortion')
    p.add_argument('--wandb-run-name',       type=str,   default=None)
    return p.parse_args()


def make_plot(sample_numbers, distortions, title, method_style):
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, kw in method_style.items():
        means = np.array([np.mean(distortions[m][float(s)]) for s in sample_numbers])
        stds  = np.array([np.std( distortions[m][float(s)]) for s in sample_numbers])
        ax.plot(sample_numbers, means, **kw)
        ax.fill_between(sample_numbers, means - stds, means + stds, alpha=0.15, color=kw['color'])
    ax.set_xscale('log')
    ax.set_xlabel('num samples')
    ax.set_ylabel('distortion')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    np.random.seed(args.seed)

    slurm_info = {k: os.environ[k] for k in (
        'SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID',
    ) if k in os.environ}

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={**vars(args), **slurm_info},
    )

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # --- data ---
    print("Loading dataset...")
    ds = dict(load_dataset("lmarena-ai/arena-human-preference-140k"))
    pw = PairwiseData(ds, M=args.M, N=args.N)
    population = Population(pw)

    candidates = np.arange(pw.M)
    true_ranking = np.argsort(-population.avg_utilities)
    w = 1 / (1.1 ** np.arange(pw.M))

    empirical_pair_distribution = np.zeros((pw.M, pw.M))
    for i in range(len(pw.winners)):
        first  = max(pw.winners[i], pw.losers[i])
        second = min(pw.winners[i], pw.losers[i])
        empirical_pair_distribution[first, second] += 1.0
    empirical_pair_distribution /= empirical_pair_distribution.sum()
    flattened = empirical_pair_distribution.ravel()

    sample_numbers = args.sample_numbers
    methods = ['borda', 'borda_peeling', 'copeland', 'copeland_peeling',
               'ml_argmax', 'ml_nonzero', 'ml_sampling']
    sample_distortions  = {m: {float(s): [] for s in sample_numbers} for m in methods}
    supremum_distortions = {m: {float(s): [] for s in sample_numbers} for m in methods}

    # --- main loop ---
    for num_samples in tqdm(sample_numbers, desc='num_samples'):
        indices = np.random.choice(len(flattened), size=num_samples, p=flattened)
        coords = np.array(np.unravel_index(indices, empirical_pair_distribution.shape)).T
        model_As_fixed = coords[..., 0]
        model_Bs_fixed = coords[..., 1]
        u_diff = (population.population_utilities[:, model_As_fixed]
                  - population.population_utilities[:, model_Bs_fixed])
        p = (expit(args.beta * u_diff) * population.voter_distr[:, None]).sum(axis=0)

        for _ in tqdm(range(args.num_rounds), desc='round', leave=False):
            mask  = np.random.rand(num_samples) < p
            w_arr = np.where(mask, model_As_fixed, model_Bs_fixed)
            l_arr = np.where(mask, model_Bs_fixed, model_As_fixed)
            ij_wins = ij_from_pairwise(w_arr, l_arr, pw.M)

            for m, fn in [
                ('borda',            borda_ranking),
                ('borda_peeling',    borda_peeling_ranking),
                ('copeland',         copeland_ranking),
                ('copeland_peeling', copeland_peeling_ranking),
            ]:
                w_dist, _ = leaderboard_dist_w(fn(ij_wins), true_ranking, population.avg_utilities, w=w)
                dist,   _ = leaderboard_dist(fn(ij_wins), true_ranking, population.avg_utilities)
                sample_distortions[m][float(num_samples)].append(w_dist)
                supremum_distortions[m][float(num_samples)].append(dist)

            for m, ranking in [
                ('ml_argmax',  ml_argmax_ranking(ij_wins)),
                ('ml_nonzero', ml_nonzero_ranking(ij_wins)),
            ]:
                w_dist, _ = leaderboard_dist_w(ranking, true_ranking, population.avg_utilities, w=w)
                dist,   _ = leaderboard_dist(ranking, true_ranking, population.avg_utilities)
                sample_distortions[m][float(num_samples)].append(w_dist)
                supremum_distortions[m][float(num_samples)].append(dist)

            ranking_dist_sampled = sampled_ranking_dist(
                candidates, ij_wins, rounds=args.ml_sampling_rounds)
            dist   = expected_leaderboard_distortion(ranking_dist_sampled, true_ranking, population.avg_utilities)
            w_dist = expected_leaderboard_distortion_w(ranking_dist_sampled, true_ranking, population.avg_utilities, w=w)
            sample_distortions['ml_sampling'][float(num_samples)].append(w_dist)
            supremum_distortions['ml_sampling'][float(num_samples)].append(dist)

    # --- save data ---
    data = dict(
        sample_numbers=sample_numbers,
        sample_distortions=sample_distortions,
        supremum_distortions=supremum_distortions,
        config=vars(args),
    )
    data_path = os.path.join(output_dir, 'results.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved results to {data_path}")

    # --- plots ---
    method_style = {
        'borda':            dict(color='C0', marker='o', linestyle='-',  label='one-shot Borda'),
        'borda_peeling':    dict(color='C1', marker='s', linestyle='--', label='iterative Borda peeling'),
        'copeland':         dict(color='C2', marker='^', linestyle='-',  label='one-shot Copeland'),
        'copeland_peeling': dict(color='C3', marker='D', linestyle='--', label='iterative Copeland peeling'),
        'ml_argmax':        dict(color='C4', marker='P', linestyle='-',  label='iterative ML with argmax'),
        'ml_nonzero':       dict(color='C5', marker='*', linestyle='--', label='iterative ML with nonzero'),
        'ml_sampling':      dict(color='C6', marker='h', linestyle='-',  label='iterative ML sampling', linewidth=2),
    }

    title_suffix = f'(beta={args.beta}, M={pw.M}, {args.num_rounds} rounds)'

    plots = [
        (sample_distortions,  f'Fixed-weight distortion vs num samples  {title_suffix}', 'fixed_weight_distortion.png'),
        (supremum_distortions, f'Supremum distortion vs num samples  {title_suffix}',    'supremum_distortion.png'),
    ]

    wandb_log = {}
    for distortions, title, fname in plots:
        fig = make_plot(sample_numbers, distortions, title, method_style)
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        wandb_log[fname.replace('.png', '')] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(wandb_log)
    wandb.finish()


if __name__ == '__main__':
    main()
