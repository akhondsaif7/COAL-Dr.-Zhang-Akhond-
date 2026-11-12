"""
Rescaled d-sweep (sigma rescaled per-d to hold ||v_k|| ~ constant), pooled
over 20 seeds. Produces three separate diagrams:
  - d_sweep_coverage_20seeds{suffix}.png  (coverage vs delta, by d)
  - d_sweep_accuracy_20seeds{suffix}.png  (accuracy vs delta, by d)
  - d_sweep_width_query_20seeds{suffix}.png  (width and query rate vs delta,
    by d, two panels on one figure)

Error bars are mean +- std computed across each seed's own (already
per-seed-aggregated) coverage/width/accuracy/query-rate number -- seed is
the actual independent unit of replication here, since each seed redraws
the whole dataset and retrains a fresh model from scratch. An earlier
version of this pooled every individual test point across seeds into one
big Wald confidence interval -- that was statistically wrong, since test
points within a single seed all come from the SAME fitted model and are
therefore correlated, not independent; pooling them inflated the effective
sample size and badly understated the true uncertainty. Averaging the 20
genuinely-independent per-seed numbers directly, as done here, is correct.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from delta_sweep_filtered import sweep_deltas
from d_sweep_rescaled import sigma_for_d

SEEDS = tuple(range(1, 21))   # 20 seeds


def run_d_sweep_20seeds(ds=(2, 4, 8), deltas=(0.5, 1, 2, 4, 8, 16, 32), K=5, N=2000, seeds=SEEDS,
                         suffix=''):
    all_results = {}

    for d in ds:
        sigma = sigma_for_d(d)
        print(f"\n--- d={d}  (sigma={sigma:.4f}) ---")
        per_seed = []
        for seed in seeds:
            res = sweep_deltas(deltas=deltas, K=K, d=d, N=N, seed=seed, sigma=sigma)
            per_seed.append(res)

        by_delta = {}
        for di, delta in enumerate(deltas):
            covs = [per_seed[s][di]['coverage'] for s in range(len(seeds))]
            widths = [per_seed[s][di]['width'] for s in range(len(seeds))]
            accs = [per_seed[s][di]['acc'] for s in range(len(seeds))]
            qs = [100 * per_seed[s][di]['queries'] / int(0.8 * N) for s in range(len(seeds))]

            by_delta[delta] = dict(cov_mean=np.mean(covs), cov_std=np.std(covs),
                                    width_mean=np.mean(widths), width_std=np.std(widths),
                                    acc_mean=np.mean(accs), acc_std=np.std(accs),
                                    q_mean=np.mean(qs), q_std=np.std(qs))
            print(f"  delta={delta:>5}: coverage={np.mean(covs):.3f}+-{np.std(covs):.3f}  "
                  f"width={np.mean(widths):.3f}+-{np.std(widths):.3f}  "
                  f"acc={np.mean(accs):.3f}+-{np.std(accs):.3f}  "
                  f"queries={np.mean(qs):.1f}%+-{np.std(qs):.1f}%")
        all_results[d] = by_delta

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ds)))

    def deltas_arr_of(d):
        return list(all_results[d].keys())

    # --- Diagram 1: coverage ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for d, color in zip(ds, colors):
        da = deltas_arr_of(d)
        vals = [all_results[d][dl]['cov_mean'] for dl in da]
        errs = [all_results[d][dl]['cov_std'] for dl in da]
        ax1.errorbar(da, vals, yerr=errs, fmt='o-', color=color, label=f'd={d}', capsize=3)
    ax1.axhline(0.95, color='gray', ls='--', lw=1, label='95% target')
    ax1.set_xscale('log')
    ax1.set_xlabel('delta (log scale)')
    ax1.set_ylabel('Coverage of true f*(x,a)')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'Coverage vs delta, by d\n(mean +- std, {len(seeds)} seeds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f'd_sweep_coverage_20seeds{suffix}.png', dpi=150)
    print(f"\nSaved: d_sweep_coverage_20seeds{suffix}.png")

    # --- Diagram 2: accuracy ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    for d, color in zip(ds, colors):
        da = deltas_arr_of(d)
        vals = [all_results[d][dl]['acc_mean'] for dl in da]
        errs = [all_results[d][dl]['acc_std'] for dl in da]
        ax2.errorbar(da, vals, yerr=errs, fmt='o-', color=color, label=f'd={d}', capsize=3)
    ax2.axhline(1 / K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    ax2.set_xscale('log')
    ax2.set_xlabel('delta (log scale)')
    ax2.set_ylabel('Test accuracy (vs realized noisy label)')
    ax2.set_title(f'Accuracy vs delta, by d\n(mean +- std, {len(seeds)} seeds)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f'd_sweep_accuracy_20seeds{suffix}.png', dpi=150)
    print(f"Saved: d_sweep_accuracy_20seeds{suffix}.png")

    # --- Diagram 3: width + query rate, two panels, one figure ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    for d, color in zip(ds, colors):
        da = deltas_arr_of(d)
        w_vals = [all_results[d][dl]['width_mean'] for dl in da]
        w_errs = [all_results[d][dl]['width_std'] for dl in da]
        q_vals = [all_results[d][dl]['q_mean'] for dl in da]
        q_errs = [all_results[d][dl]['q_std'] for dl in da]
        axes3[0].errorbar(da, w_vals, yerr=w_errs, fmt='o-', color=color, label=f'd={d}', capsize=3)
        axes3[1].errorbar(da, q_vals, yerr=q_errs, fmt='o-', color=color, label=f'd={d}', capsize=3)

    axes3[0].set_xscale('log')
    axes3[0].set_xlabel('delta (log scale)')
    axes3[0].set_ylabel('Mean interval width (UCB-LCB)')
    axes3[0].set_title(f'Interval width vs delta, by d\n(mean +- std, {len(seeds)} seeds)')
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    axes3[1].axhline(100, color='gray', ls='--', lw=1)
    axes3[1].set_xscale('log')
    axes3[1].set_xlabel('delta (log scale)')
    axes3[1].set_ylabel('% of training examples queried')
    axes3[1].set_ylim(0, 105)
    axes3[1].set_title(f'Query rate vs delta, by d\n(mean +- std, {len(seeds)} seeds)')
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    fig3.suptitle(f'Width and query rate vs delta, by d ({len(seeds)} seeds)', fontsize=13)
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    fig3.savefig(f'd_sweep_width_query_20seeds{suffix}.png', dpi=150)
    print(f"Saved: d_sweep_width_query_20seeds{suffix}.png")

    return all_results


if __name__ == '__main__':
    run_d_sweep_20seeds()
