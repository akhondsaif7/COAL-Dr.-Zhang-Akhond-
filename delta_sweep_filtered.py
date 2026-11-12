"""
Sweep cheat mode's delta (confidence-interval width hyperparameter) on the
FILTERED (rejection-sampled, exactly-linear) linear synthetic dataset, to
see whether some delta achieves good UCB/LCB coverage of the true f*(x,a)
without the interval becoming too wide to be practically useful (i.e.
without collapsing query-selectivity to ~100%, the same failure mode
rigorous mode hits via its own, much larger, provably-calibrated radius).

Features are intercept-augmented (x -> [x,1]) throughout, since that's a
separate, already-confirmed fix for a different bug (missing intercept) --
sweeping delta on top of that isolates delta's own effect cleanly.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_passive import PassiveCOAL
from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset


def sweep_deltas(deltas=(0.5, 1, 2, 4, 8, 16, 32), K=5, d=20, N=2000, seed=23, sigma=0.02):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed, sigma=sigma)
    print(f"Filtered dataset: {len(X)} points accepted ({n_rejected}/{n_drawn} rejected, "
          f"{100*n_rejected/n_drawn:.2f}%)")

    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)

    X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])
    d_feed = d + 1

    results = []
    for delta in deltas:
        coal_c = CheatCOAL(K=K, d=d_feed, reg=1.0, delta=delta, burn_in=20)
        for i in range(T):
            coal_c.observe(X_train_aug[i], y_train[i], active=True)
        correct = (coal_c.predict_batch(X_test_aug) == y_test)   # vs REALIZED noisy label, not Bayes-optimal
        acc = np.mean(correct)

        coal_c._fit_all()
        n_check = len(X_test)   # full test setup
        cov, widths = [], []
        for i in range(n_check):
            ucb, lcb = coal_c.get_ucb_lcb(X_test_aug[i])
            tp = true_probs(X_test[i])
            cov.append(np.all((lcb <= tp) & (tp <= ucb)))   # per-example: ALL K classes covered simultaneously
            for k in range(K):
                widths.append(ucb[k] - lcb[k])
        coverage = np.mean(cov)
        width = np.mean(widths)
        queries = coal_c.n_queries

        print(f"delta={delta:>6}: queries={queries:>5}/{T} ({100*queries/T:5.1f}%)  "
              f"coverage={coverage:.3f}  width={width:.3f}  acc={acc:.3f}")
        results.append(dict(delta=delta, queries=queries, coverage=coverage, width=width, acc=acc))

    # --- Plot: coverage, width, accuracy, query-rate vs delta (log x-axis) ---
    deltas_arr = [r['delta'] for r in results]
    cov_arr = [r['coverage'] for r in results]
    width_arr = [r['width'] for r in results]
    acc_arr = [r['acc'] for r in results]
    q_arr = [100 * r['queries'] / T for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].plot(deltas_arr, cov_arr, 'o-', color='seagreen', lw=2)
    axes[0].axhline(0.95, color='gray', ls='--', lw=1, label='95% target')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('delta (log scale)')
    axes[0].set_ylabel('Coverage of true f*(x,a)')
    axes[0].set_title('Coverage vs delta')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(deltas_arr, width_arr, 'o-', color='darkorange', lw=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('delta (log scale)')
    axes[1].set_ylabel('Mean interval width (UCB-LCB)')
    axes[1].set_title('Interval width vs delta')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(deltas_arr, acc_arr, 'o-', color='purple', lw=2)
    axes[2].axhline(1/K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    axes[2].set_xscale('log')
    axes[2].set_xlabel('delta (log scale)')
    axes[2].set_ylabel('Test accuracy (vs realized noisy label)')
    axes[2].set_title('Accuracy vs delta')
    axes[2].set_ylim(0, 1.0)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(deltas_arr, q_arr, 'o-', color='steelblue', lw=2)
    axes[3].axhline(100, color='gray', ls='--', lw=1, label='100% (passive)')
    axes[3].set_xscale('log')
    axes[3].set_xlabel('delta (log scale)')
    axes[3].set_ylabel('% of training examples queried')
    axes[3].set_title('Query rate vs delta')
    axes[3].set_ylim(0, 105)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle('Cheat-mode delta sweep on rejection-filtered (exactly-linear) data', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = 'delta_sweep_filtered.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return results


if __name__ == '__main__':
    sweep_deltas()
