"""
Rescaled d-sweep (sigma rescaled per-d to hold ||v_k|| ~ constant), averaged
over several seeds per d.

Error bars are mean +- std computed directly across each seed's own
(already-aggregated) coverage/width/accuracy/query-rate number -- seed is
the actual independent unit here (each seed redraws the whole dataset and
retrains a fresh model), so this is what the error bar should be computed
over. An earlier version of this script pooled every individual test point
across seeds and computed a Wald confidence interval from that pooled
count -- that was wrong, because test points within one seed all come from
the SAME fitted model and are therefore correlated, not independent; the
pooled approach treated 400 correlated points as 400 independent trials
and so drastically understated the true uncertainty. Averaging the 5 (or
20) independent per-seed numbers directly, as done here, is the correct
and much simpler fix.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from delta_sweep_filtered import sweep_deltas
from d_sweep_rescaled import sigma_for_d

SEEDS = (23, 7, 11, 42, 99)


def run_d_sweep_multiseed(ds=(2, 3, 4, 5, 6, 7, 8, 16, 20), deltas=(0.5, 1, 2, 4, 8, 16, 32),
                           K=5, N=2000, seeds=SEEDS):
    T = int(0.8 * N)
    all_results = {}   # d -> delta -> dict(cov_mean, cov_std, width_mean, width_std, acc_mean, acc_std, q_mean, q_std)

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
            qs = [100 * per_seed[s][di]['queries'] / T for s in range(len(seeds))]
            by_delta[delta] = dict(cov_mean=np.mean(covs), cov_std=np.std(covs),
                                    width_mean=np.mean(widths), width_std=np.std(widths),
                                    acc_mean=np.mean(accs), acc_std=np.std(accs),
                                    q_mean=np.mean(qs), q_std=np.std(qs))
            print(f"  delta={delta:>5}: coverage={np.mean(covs):.3f}+-{np.std(covs):.3f}  "
                  f"width={np.mean(widths):.3f}+-{np.std(widths):.3f}  "
                  f"acc={np.mean(accs):.3f}+-{np.std(accs):.3f}  "
                  f"queries={np.mean(qs):.1f}%+-{np.std(qs):.1f}%")
        all_results[d] = by_delta

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ds)))

    for d, color in zip(ds, colors):
        by_delta = all_results[d]
        deltas_arr = list(by_delta.keys())
        cov_arr = [by_delta[dl]['cov_mean'] for dl in deltas_arr]
        cov_err = [by_delta[dl]['cov_std'] for dl in deltas_arr]
        width_arr = [by_delta[dl]['width_mean'] for dl in deltas_arr]
        width_err = [by_delta[dl]['width_std'] for dl in deltas_arr]
        acc_arr = [by_delta[dl]['acc_mean'] for dl in deltas_arr]
        acc_err = [by_delta[dl]['acc_std'] for dl in deltas_arr]
        q_arr = [by_delta[dl]['q_mean'] for dl in deltas_arr]
        q_err = [by_delta[dl]['q_std'] for dl in deltas_arr]

        axes[0].errorbar(deltas_arr, cov_arr, yerr=cov_err, fmt='o-', color=color, label=f'd={d}', capsize=3)
        axes[1].errorbar(deltas_arr, width_arr, yerr=width_err, fmt='o-', color=color, label=f'd={d}', capsize=3)
        axes[2].errorbar(deltas_arr, acc_arr, yerr=acc_err, fmt='o-', color=color, label=f'd={d}', capsize=3)
        axes[3].errorbar(deltas_arr, q_arr, yerr=q_err, fmt='o-', color=color, label=f'd={d}', capsize=3)

    axes[0].axhline(0.95, color='gray', ls='--', lw=1)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('delta (log scale)')
    axes[0].set_ylabel('Coverage of true f*(x,a)')
    axes[0].set_title(f'Coverage vs delta, by d\n(mean +- std over {len(seeds)} seeds)')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale('log')
    axes[1].set_xlabel('delta (log scale)')
    axes[1].set_ylabel('Mean interval width (UCB-LCB)')
    axes[1].set_title(f'Interval width vs delta, by d\n(mean +- std over {len(seeds)} seeds)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xscale('log')
    axes[2].set_xlabel('delta (log scale)')
    axes[2].set_ylabel('Test accuracy (vs realized noisy label)')
    axes[2].set_title(f'Accuracy vs delta, by d\n(mean +- std over {len(seeds)} seeds)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].axhline(100, color='gray', ls='--', lw=1)
    axes[3].set_xscale('log')
    axes[3].set_xlabel('delta (log scale)')
    axes[3].set_ylabel('% of training examples queried')
    axes[3].set_title(f'Query rate vs delta, by d\n(mean +- std over {len(seeds)} seeds)')
    axes[3].set_ylim(0, 105)
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'Coverage-vs-selectivity across d (||v_k|| held constant), averaged over {len(seeds)} seeds',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = 'd_sweep_rescaled_multiseed.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    print(f"\n{'d':>4} {'acc @ delta=max':>18} {'queries @ delta=0.5':>22}")
    for d in ds:
        acc_last = all_results[d][deltas[-1]]
        q_first = all_results[d][deltas[0]]
        print(f"{d:>4} {acc_last['acc_mean']:.3f}+-{acc_last['acc_std']:.3f}   "
              f"{q_first['q_mean']:>6.1f}%+-{q_first['q_std']:.1f}%")

    return all_results


if __name__ == '__main__':
    run_d_sweep_multiseed()
