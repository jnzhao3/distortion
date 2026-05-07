##============ FINITE SETTING, FIXED WEIGHT SETS ==========##


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
import shutil

from utils_3 import PairwiseData, Population
from utils_4 import (
    ij_from_pairwise,
    borda_ranking, borda_peeling_ranking,
    copeland_ranking, copeland_peeling_ranking,
    ml_argmax_ranking, ml_nonzero_ranking,
    make_ml_fn_from_ij_wins,
    expected_leaderboard_distortion, expected_leaderboard_distortion_w,
)
from utils_3 import leaderboard_dist
from scipy.optimize import linprog
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def leaderboard_dist_w(ranking, true_ranking, avg_utils, w):
    ranking = np.asarray(ranking)
    true_ranking = np.asarray(true_ranking)
    ranking_utils = avg_utils[ranking]
    true_ranking_utils = avg_utils[true_ranking]
    denom_sum = (ranking_utils * w).sum()
    ratio = (true_ranking_utils * w).sum() / denom_sum if denom_sum > 0 else np.inf
    return ratio, None


def sampled_ranking_dist(candidates, ij_wins, tol=1e-12, rounds=10):
    from utils_4 import _solve_maximal_lottery
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
    p.add_argument('--num-samples',         type=int,   default=30_000)
    p.add_argument('--num-rounds',          type=int,   default=10)
    p.add_argument('--ml-sampling-rounds',  type=int,   default=100)
    p.add_argument('--M',                   type=int,   default=30)
    p.add_argument('--N',                   type=int,   default=30)
    p.add_argument('--seed',                type=int,   default=1001)
    p.add_argument('--betas',               type=float, nargs='+',
                   default=[0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0,
                             25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
    p.add_argument('--weight-base',          type=float, default=1.1)
    p.add_argument('--output-dir',          type=str,   default='results')
    p.add_argument('--wandb-project',       type=str,   default='distortion')
    p.add_argument('--wandb-run-name',      type=str,   default=None)
    return p.parse_args()


def make_plot(betas, distortions, title, method_style, w=None, weight_base=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    all_uppers = []
    for m, kw in method_style.items():
        means = np.array([np.mean(distortions[m][float(b)]) for b in betas])
        stds  = np.array([np.std( distortions[m][float(b)]) for b in betas])
        ax.plot(betas, means, **kw)
        ax.fill_between(betas, means - stds, means + stds, alpha=0.15, color=kw['color'])
        all_uppers.append((means + stds).max())

    top = max(all_uppers) * 1.05 if all_uppers else 1.1
    ax.set_xlabel('β', fontstyle="italic")
    ax.set_ylabel('distortion')
    ax.set_ylim(bottom=1.0, top=top)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    if w is not None:
        inset = ax.inset_axes([0.72, 0.55, 0.25, 0.38])
        inset.bar(np.arange(len(w)), w, color='grey', alpha=0.7, width=0.8)
        base_str = f'{weight_base}' if weight_base is not None else ''
        inset.set_title(f'w = 1/{base_str}^k', fontsize=7)
        inset.set_xlabel('rank k', fontsize=6)
        inset.tick_params(labelsize=6)
        inset.set_xlim(-0.5, len(w) - 0.5)

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

    # --- data ---
    print("Loading dataset...")
    ds = dict(load_dataset("lmarena-ai/arena-human-preference-140k"))
    pw = PairwiseData(ds, M=args.M, N=args.N)
    population = Population(pw)

    betas = np.asarray(args.betas)
    true_ranking = np.argsort(-population.avg_utilities)
    candidates = np.arange(pw.M)
    w = 1 / (args.weight_base ** np.arange(pw.M))

    empirical_pair_distribution = np.zeros((pw.M, pw.M))
    for i in range(len(pw.winners)):
        first  = max(pw.winners[i], pw.losers[i])
        second = min(pw.winners[i], pw.losers[i])
        empirical_pair_distribution[first, second] += 1.0
    empirical_pair_distribution /= empirical_pair_distribution.sum()

    methods = ['borda', 'borda_peeling', 'copeland', 'copeland_peeling',
               'ml_argmax', 'ml_nonzero', 'ml_sampling']
    betas_distortions   = {m: {float(b): [] for b in betas} for m in methods}
    supremum_distortions = {m: {float(b): [] for b in betas} for m in methods}

    flattened = empirical_pair_distribution.ravel()
    indices = np.random.choice(len(flattened), size=args.num_samples, p=flattened)
    coords = np.array(np.unravel_index(indices, empirical_pair_distribution.shape)).T
    model_As_fixed = coords[..., 0]
    model_Bs_fixed = coords[..., 1]
    u_diff = (population.population_utilities[:, model_As_fixed]
              - population.population_utilities[:, model_Bs_fixed])

    # --- main loop ---
    for beta in tqdm(betas, desc='beta'):
        p = (expit(beta * u_diff) * population.voter_distr[:, None]).sum(axis=0)

        for _ in tqdm(range(args.num_rounds), desc='round', leave=False):
            mask  = np.random.rand(args.num_samples) < p
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
                betas_distortions[m][float(beta)].append(w_dist)
                supremum_distortions[m][float(beta)].append(dist)

            for m, ranking in [
                ('ml_argmax',  ml_argmax_ranking(ij_wins)),
                ('ml_nonzero', ml_nonzero_ranking(ij_wins)),
            ]:
                w_dist, _ = leaderboard_dist_w(ranking, true_ranking, population.avg_utilities, w=w)
                dist,   _ = leaderboard_dist(ranking, true_ranking, population.avg_utilities)
                betas_distortions[m][float(beta)].append(w_dist)
                supremum_distortions[m][float(beta)].append(dist)

            ranking_dist_sampled = sampled_ranking_dist(
                candidates, ij_wins, rounds=args.ml_sampling_rounds)
            dist   = expected_leaderboard_distortion(ranking_dist_sampled, true_ranking, population.avg_utilities)
            w_dist = expected_leaderboard_distortion_w(ranking_dist_sampled, true_ranking, population.avg_utilities, w=w)
            betas_distortions['ml_sampling'][float(beta)].append(w_dist)
            supremum_distortions['ml_sampling'][float(beta)].append(dist)

    # --- save data ---
    data = dict(
        betas=betas,
        betas_distortions=betas_distortions,
        supremum_distortions=supremum_distortions,
        config=vars(args),
    )
    data_path = os.path.join(output_dir, 'results.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved results to {data_path}")

    # --- plots ---
    # one-shot Borda; iterative Borda; one-shot Copeland; iterative Copeland; iterative ML with argmax; iterative ML with nonzero; iterative ML
    method_style = {
        'borda':            dict(color='C0', marker='o', linestyle='-',  label='one-shot Borda'),
        'borda_peeling':    dict(color='C1', marker='s', linestyle='--', label='iterative Borda'),
        'copeland':         dict(color='C2', marker='^', linestyle='-',  label='one-shot Copeland'),
        'copeland_peeling': dict(color='C3', marker='D', linestyle='--', label='iterative Copeland'),
        'ml_argmax':        dict(color='C4', marker='P', linestyle='-',  label='iterative ML with argmax'),
        'ml_nonzero':       dict(color='C5', marker='*', linestyle='--', label='iterative ML with nonzero'),
        'ml_sampling':      dict(color='C6', marker='h', linestyle='-',  label='iterative ML', linewidth=2),
    }

    title_suffix = f'(M={pw.M}, {args.num_samples} samples/round, {args.num_rounds} rounds)'
    small_betas = betas[betas <= 3.0]
    large_betas = betas[betas > 3.0]

    plots = [
        (betas,       betas_distortions, 'Fixed-weight distortion vs beta',             'fixed_weight_distortion.png'),
        (small_betas, betas_distortions, 'Fixed-weight distortion vs beta (small beta)', 'fixed_weight_distortion_small_beta.png'),
        (large_betas, betas_distortions, 'Fixed-weight distortion vs beta (large beta)', 'fixed_weight_distortion_large_beta.png'),
    ]

    wandb_log = {}
    for beta_subset, distortions, title, fname in plots:
        if len(beta_subset) == 0:
            continue
        fig = make_plot(beta_subset, distortions, f'{title}  {title_suffix}', method_style,
                        w=w, weight_base=args.weight_base)
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        wandb_log[fname.replace('.png', '')] = wandb.Image(fig)
        plt.close(fig)

    # --- large-beta grid ---
    grid_specs = [
        (betas_distortions, 'fixed-weight distortion', 'fw_large_grid.png'),
    ]
    for distortions, ylabel, fname in grid_specs:
        fig_grid, ax = plt.subplots(1, 1, figsize=(10, 5))
        all_uppers = []
        for m, kw in method_style.items():
            if len(large_betas) == 0:
                continue
            means = np.array([np.mean(distortions[m][float(b)]) for b in large_betas])
            stds  = np.array([np.std( distortions[m][float(b)]) for b in large_betas])
            ax.plot(large_betas, means, **kw)
            ax.fill_between(large_betas, means - stds, means + stds, alpha=0.15, color=kw['color'])
            all_uppers.append((means + stds).max())
        top = max(all_uppers) * 1.05 if all_uppers else 1.1
        ax.set_xlabel('β')
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=1.0, top=top)
        ax.set_title(f'{ylabel.capitalize()} vs β (large β)  {title_suffix}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

        inset = ax.inset_axes([0.72, 0.55, 0.25, 0.38])
        inset.bar(np.arange(len(w)), w, color='grey', alpha=0.7, width=0.8)
        inset.set_title(f'w = 1/{args.weight_base}^k', fontsize=7)
        inset.set_xlabel('rank k', fontsize=6)
        inset.tick_params(labelsize=6)
        inset.set_xlim(-0.5, len(w) - 0.5)

        plt.tight_layout()
        fig_grid.savefig(os.path.join(output_dir, fname), dpi=150)
        wandb_log[fname.replace('.png', '')] = wandb.Image(fig_grid)
        plt.close(fig_grid)

    wandb.log(wandb_log)
    wandb.finish()


if __name__ == '__main__':
    main()
