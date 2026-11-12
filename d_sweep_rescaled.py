"""
Same d-sweep as d_sweep.py, but with sigma rescaled per-d to hold ||v_k||
(and hence the true signal strength) roughly constant across the whole
sweep: sigma(d) = 0.02 * sqrt(20/d), pinned to the d=20 baseline
||v_k|| ~= sqrt(20)*0.02 ~= 0.089.

This isolates the effect of dimensionality/estimation-noise alone --
d_sweep.py (sigma=0.02 fixed) mixes that together with a shrinking signal
as d drops, which produced a non-monotonic accuracy peak at d=4. This
version asks: with signal strength held fixed, does accuracy/coverage/
selectivity still change with d, or was the earlier peak purely a signal-
strength artifact?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from delta_sweep_filtered import sweep_deltas

BASELINE_D = 20
BASELINE_SIGMA = 0.02


def sigma_for_d(d, baseline_d=BASELINE_D, baseline_sigma=BASELINE_SIGMA):
    """sigma(d) such that sqrt(d)*sigma(d) == sqrt(baseline_d)*baseline_sigma."""
    return baseline_sigma * np.sqrt(baseline_d / d)


def run_d_sweep_rescaled(ds=(2, 4, 8, 16, 20), deltas=(0.5, 1, 2, 4, 8, 16, 32), K=5, N=2000, seed=23):
    all_results = {}
    for d in ds:
        sigma = sigma_for_d(d)
        print(f"\n--- d={d}  (sigma={sigma:.4f}, target ||v_k||~={BASELINE_SIGMA*np.sqrt(BASELINE_D):.4f}) ---")
        all_results[d] = sweep_deltas(deltas=deltas, K=K, d=d, N=N, seed=seed, sigma=sigma)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ds)))

    for d, color in zip(ds, colors):
        res = all_results[d]
        deltas_arr = [r['delta'] for r in res]
        cov_arr = [r['coverage'] for r in res]
        acc_arr = [r['acc'] for r in res]
        T = int(0.8 * N)
        q_arr = [100 * r['queries'] / T for r in res]

        axes[0].plot(deltas_arr, cov_arr, 'o-', color=color, label=f'd={d}')
        axes[1].plot(deltas_arr, acc_arr, 'o-', color=color, label=f'd={d}')
        axes[2].plot(deltas_arr, q_arr, 'o-', color=color, label=f'd={d}')

    axes[0].axhline(0.95, color='gray', ls='--', lw=1)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('delta (log scale)')
    axes[0].set_ylabel('Coverage of true f*(x,a)')
    axes[0].set_title('Coverage vs delta, by d')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale('log')
    axes[1].set_xlabel('delta (log scale)')
    axes[1].set_ylabel('Test accuracy (vs realized noisy label)')
    axes[1].set_title('Accuracy vs delta, by d')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].axhline(100, color='gray', ls='--', lw=1)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('delta (log scale)')
    axes[2].set_ylabel('% of training examples queried')
    axes[2].set_title('Query rate vs delta, by d')
    axes[2].set_ylim(0, 105)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Coverage-vs-selectivity across d, with sigma rescaled to hold ||v_k|| ~ constant (isolating dimensionality)',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = 'd_sweep_rescaled.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return all_results


if __name__ == '__main__':
    run_d_sweep_rescaled()
