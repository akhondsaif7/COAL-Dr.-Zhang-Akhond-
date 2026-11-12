"""
How does the coverage-vs-selectivity tension change as d (feature
dimension) shrinks? Runs the delta sweep (cheat mode, filtered +
intercept-augmented data) at several values of d, holding K, sigma=0.02,
and everything else fixed, and compares.

NOTE: sigma (tilt scale) is held fixed at 0.02 here, matching the literal
proposal. Since typical ||v_k|| ~= sqrt(d)*sigma, shrinking d also shrinks
the true signal strength, not just the number of parameters being fit --
the two effects aren't separated in this sweep (see conversation).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from delta_sweep_filtered import sweep_deltas


def run_d_sweep(ds=(2, 4, 8, 16, 20), deltas=(0.5, 1, 2, 4, 8, 16, 32), K=5, N=2000, seed=23):
    all_results = {}
    for d in ds:
        print(f"\n--- d={d} ---")
        all_results[d] = sweep_deltas(deltas=deltas, K=K, d=d, N=N, seed=seed)

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

    fig.suptitle('Coverage-vs-selectivity tension across d (cheat mode, filtered + intercept-augmented, sigma=0.02 fixed)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = 'd_sweep.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return all_results


if __name__ == '__main__':
    run_d_sweep()
