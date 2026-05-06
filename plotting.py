"""
Read results from results/sim32/samples_*/  and produce comparison plots.

Each sample-count directory may contain multiple timestamped runs; all are
pooled together so more runs = smaller error bars.

Output: plots/sim32/*.png
"""

import glob
import os
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = 'results/sim32'
PLOTS_DIR   = 'plots/sim32'

METHOD_STYLE = {
    'borda':            dict(color='C0', marker='o', linestyle='-',  label='one-shot Borda'),
    'borda_peeling':    dict(color='C1', marker='s', linestyle='--', label='iterative Borda'),
    'copeland':         dict(color='C2', marker='^', linestyle='-',  label='one-shot Copeland'),
    'copeland_peeling': dict(color='C3', marker='D', linestyle='--', label='iterative Copeland'),
    'ml_argmax':        dict(color='C4', marker='P', linestyle='-',  label='iterative ML argmax'),
    'ml_nonzero':       dict(color='C5', marker='*', linestyle='--', label='iterative ML nonzero'),
    'ml_sampling':      dict(color='C6', marker='h', linestyle='-',  label='iterative ML sampling', linewidth=2),
}


def load_all_results(results_dir):
    """
    Returns a dict: num_samples -> {'betas_distortions': ..., 'supremum_distortions': ..., 'betas': ...}
    Multiple runs for the same num_samples are pooled (lists concatenated).
    """
    pkl_files = glob.glob(os.path.join(results_dir, '**', 'results.pkl'), recursive=True)
    if not pkl_files:
        raise FileNotFoundError(f"No results.pkl found under {results_dir}")

    pooled = {}  # num_samples -> {method -> {beta -> [distortions]}}

    for path in sorted(pkl_files):
        with open(path, 'rb') as f:
            data = pickle.load(f)

        n = data['config']['num_samples']
        betas = np.asarray(data['betas'])

        if n not in pooled:
            pooled[n] = {
                'betas': betas,
                'betas_distortions':    defaultdict(lambda: defaultdict(list)),
                'supremum_distortions': defaultdict(lambda: defaultdict(list)),
            }

        for key in ('betas_distortions', 'supremum_distortions'):
            for method, beta_dict in data[key].items():
                for beta, vals in beta_dict.items():
                    pooled[n][key][method][float(beta)].extend(vals)

    return pooled


def make_distortion_plot(betas, distortions, title, method_style):
    fig, ax = plt.subplots(figsize=(10, 5))
    for m, kw in method_style.items():
        if m not in distortions:
            continue
        means = np.array([np.mean(distortions[m][float(b)]) for b in betas])
        stds  = np.array([np.std( distortions[m][float(b)]) for b in betas])
        ax.plot(betas, means, **kw)
        ax.fill_between(betas, means - stds, means + stds, alpha=0.15, color=kw['color'])
    ax.set_xlabel('β')
    ax.set_ylabel('distortion')
    ax.set_ylim(bottom=1.0, top=1.02)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    return fig


def make_samples_comparison_plot(pooled, betas_subset, distortion_key, method, title):
    """Plot distortion vs num_samples for a specific method and beta subset."""
    sample_numbers = sorted(pooled.keys())
    fig, ax = plt.subplots(figsize=(10, 5))

    for b in betas_subset:
        vals_per_n = []
        stds_per_n = []
        for n in sample_numbers:
            d = pooled[n][distortion_key][method].get(float(b), [])
            vals_per_n.append(np.mean(d) if d else np.nan)
            stds_per_n.append(np.std(d)  if d else np.nan)
        vals = np.array(vals_per_n)
        stds = np.array(stds_per_n)
        ax.plot(sample_numbers, vals, marker='o', label=f'β={b}')
        ax.fill_between(sample_numbers, vals - stds, vals + stds, alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel('num samples')
    ax.set_ylabel('distortion')
    ax.set_ylim(bottom=1.0, top=1.2)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"Loading results from {RESULTS_DIR} ...")
    pooled = load_all_results(RESULTS_DIR)
    sample_numbers = sorted(pooled.keys())
    print(f"Found {len(sample_numbers)} sample counts: {sample_numbers}")

    # ── per-num_samples: distortion vs beta plots ──────────────────────────
    for n in sample_numbers:
        betas = pooled[n]['betas']
        small_betas = betas[betas <= 3.0]
        large_betas = betas[betas >  3.0]
        suffix = f'(n={n:,})'

        per_run_plots = [
            (betas,       'betas_distortions',   f'Fixed-weight distortion vs β  {suffix}',             f'n{n}_fw_all.png'),
            (betas,       'supremum_distortions', f'Supremum distortion vs β  {suffix}',                 f'n{n}_sup_all.png'),
            (small_betas, 'betas_distortions',   f'Fixed-weight distortion vs β (small β)  {suffix}',   f'n{n}_fw_small.png'),
            (small_betas, 'supremum_distortions', f'Supremum distortion vs β (small β)  {suffix}',      f'n{n}_sup_small.png'),
            (large_betas, 'betas_distortions',   f'Fixed-weight distortion vs β (large β)  {suffix}',   f'n{n}_fw_large.png'),
            (large_betas, 'supremum_distortions', f'Supremum distortion vs β (large β)  {suffix}',      f'n{n}_sup_large.png'),
        ]

        for beta_subset, dist_key, title, fname in per_run_plots:
            if len(beta_subset) == 0:
                continue
            fig = make_distortion_plot(beta_subset, pooled[n][dist_key], title, METHOD_STYLE)
            fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
            plt.close(fig)
            print(f"  Saved {fname}")

    # ── combined grid: sup_large for all sample counts (2 rows × 3 cols) ────
    fig_grid, axes = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
    axes_flat = axes.flatten()

    for idx, n in enumerate(sample_numbers[:6]):
        ax = axes_flat[idx]
        betas = pooled[n]['betas']
        large_betas = betas[betas > 3.0]
        distortions = pooled[n]['supremum_distortions']

        for m, kw in METHOD_STYLE.items():
            if m not in distortions or len(large_betas) == 0:
                continue
            means = np.array([np.mean(distortions[m][float(b)]) for b in large_betas])
            stds  = np.array([np.std( distortions[m][float(b)]) for b in large_betas])
            ax.plot(large_betas, means, **kw)
            ax.fill_between(large_betas, means - stds, means + stds, alpha=0.15, color=kw['color'])

        ax.set_title(f'n={n:,}')
        ax.set_xlabel('β')
        ax.set_ylim(bottom=1.0, top=1.02)
        ax.grid(True, which='both', alpha=0.3)
        if idx % 3 == 0:
            ax.set_ylabel('supremum distortion')

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig_grid.legend(handles, labels, loc='lower center', ncol=4, fontsize=9,
                    bbox_to_anchor=(0.5, -0.05))
    fig_grid.suptitle('Supremum distortion vs β (large β) by sample count', y=1.01)
    plt.tight_layout()
    grid_path = os.path.join(PLOTS_DIR, 'sup_large_grid.png')
    fig_grid.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close(fig_grid)
    print(f"  Saved sup_large_grid.png")

    # ── across num_samples: distortion vs num_samples for selected betas ───
    all_betas = pooled[sample_numbers[0]]['betas']
    selected_betas = [b for b in [1.0, 3.0, 5.0, 10.0, 20.0, 50.0] if b in all_betas]

    for method, kw in METHOD_STYLE.items():
        for dist_key, dist_label in [
            ('betas_distortions',   'fixed-weight'),
            ('supremum_distortions', 'supremum'),
        ]:
            title = f'{kw["label"]} — {dist_label} distortion vs num samples'
            fname = f'samples_{method}_{dist_label.replace("-","_")}.png'
            fig = make_samples_comparison_plot(
                pooled, selected_betas, dist_key, method, title)
            fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
            plt.close(fig)
            print(f"  Saved {fname}")

    print(f"\nAll plots written to {PLOTS_DIR}/")


if __name__ == '__main__':
    main()
