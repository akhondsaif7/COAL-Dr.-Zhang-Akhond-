"""
Compare coverage, width, accuracy, and query rate across many d values, at
a SINGLE fixed delta=1 (established as the best operating point from the
earlier full delta sweeps -- coverage crosses ~90-95% while query rate
hasn't fully saturated, and accuracy is already at/near its plateau).

No delta sweep here -- this replaces the earlier multi-delta d-sweeps now
that delta itself is no longer being varied. Sigma is rescaled per-d
(sigma_for_d) to hold ||v_k|| ~ constant, so differences across d reflect
dimensionality/estimation effects, not a shrinking signal. Error bars are
mean +- std across seeds (seed is the independent unit -- see
d_sweep_rescaled_multiseed.py for why NOT to pool across test points).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from delta_sweep_filtered import sweep_deltas
from d_sweep_rescaled import sigma_for_d

SEEDS = tuple(range(1, 21))   # 20 seeds
DS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20)
DELTA = 1.0


def run_d_comparison(ds=DS, delta=DELTA, K=5, N=2000, seeds=SEEDS, out_suffix=''):
    T = int(0.8 * N)
    results = {}   # d -> dict(cov_mean, cov_std, width_mean, width_std, acc_mean, acc_std, q_mean, q_std)

    for d in ds:
        sigma = sigma_for_d(d)
        covs, widths, accs, qs = [], [], [], []
        for seed in seeds:
            res = sweep_deltas(deltas=(delta,), K=K, d=d, N=N, seed=seed, sigma=sigma)
            r = res[0]
            covs.append(r['coverage'])
            widths.append(r['width'])
            accs.append(r['acc'])
            qs.append(100 * r['queries'] / T)

        results[d] = dict(cov_mean=np.mean(covs), cov_std=np.std(covs),
                           width_mean=np.mean(widths), width_std=np.std(widths),
                           acc_mean=np.mean(accs), acc_std=np.std(accs),
                           q_mean=np.mean(qs), q_std=np.std(qs))
        r = results[d]
        print(f"d={d:>3} (sigma={sigma:.4f}): coverage={r['cov_mean']:.3f}+-{r['cov_std']:.3f}  "
              f"width={r['width_mean']:.3f}+-{r['width_std']:.3f}  "
              f"acc={r['acc_mean']:.3f}+-{r['acc_std']:.3f}  "
              f"queries={r['q_mean']:.1f}%+-{r['q_std']:.1f}%")

    ds_arr = list(ds)

    # --- Diagram 1: coverage ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.errorbar(ds_arr, [results[d]['cov_mean'] for d in ds_arr],
                 yerr=[results[d]['cov_std'] for d in ds_arr], fmt='o-', color='steelblue', capsize=3)
    ax1.axhline(0.95, color='gray', ls='--', lw=1, label='95% target')
    ax1.set_xlabel('d')
    ax1.set_ylabel('Coverage of true f*(x,a)')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'Coverage vs d (K={K}, delta={delta} fixed)\n(mean +- std, {len(seeds)} seeds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    out1 = f'd_comparison_coverage{out_suffix}.png'
    fig1.savefig(out1, dpi=150)
    print(f"\nSaved: {out1}")

    # --- Diagram 2: accuracy ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.errorbar(ds_arr, [results[d]['acc_mean'] for d in ds_arr],
                 yerr=[results[d]['acc_std'] for d in ds_arr], fmt='o-', color='darkorange', capsize=3)
    ax2.axhline(1 / K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    ax2.set_xlabel('d')
    ax2.set_ylabel('Test accuracy (vs realized noisy label)')
    ax2.set_title(f'Accuracy vs d (K={K}, delta={delta} fixed)\n(mean +- std, {len(seeds)} seeds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = f'd_comparison_accuracy{out_suffix}.png'
    fig2.savefig(out2, dpi=150)
    print(f"Saved: {out2}")

    # --- Diagram 3: width + query rate ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    axes3[0].errorbar(ds_arr, [results[d]['width_mean'] for d in ds_arr],
                       yerr=[results[d]['width_std'] for d in ds_arr], fmt='o-', color='seagreen', capsize=3)
    axes3[0].set_xlabel('d')
    axes3[0].set_ylabel('Mean interval width (UCB-LCB)')
    axes3[0].set_title(f'Interval width vs d (delta={delta} fixed)\n(mean +- std, {len(seeds)} seeds)')
    axes3[0].grid(True, alpha=0.3)

    axes3[1].errorbar(ds_arr, [results[d]['q_mean'] for d in ds_arr],
                       yerr=[results[d]['q_std'] for d in ds_arr], fmt='o-', color='crimson', capsize=3)
    axes3[1].axhline(100, color='gray', ls='--', lw=1)
    axes3[1].set_xlabel('d')
    axes3[1].set_ylabel('% of training examples queried')
    axes3[1].set_ylim(0, 105)
    axes3[1].set_title(f'Query rate vs d (delta={delta} fixed)\n(mean +- std, {len(seeds)} seeds)')
    axes3[1].grid(True, alpha=0.3)

    fig3.suptitle(f'Width and query rate vs d, K={K}, delta={delta} fixed ({len(seeds)} seeds)', fontsize=13)
    fig3.tight_layout(rect=[0, 0, 1, 0.92])
    out3 = f'd_comparison_width_query{out_suffix}.png'
    fig3.savefig(out3, dpi=150)
    print(f"Saved: {out3}")

    return results


if __name__ == '__main__':
    run_d_comparison()
