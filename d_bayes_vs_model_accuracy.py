"""
For a few values of d, compare:
  - Bayes-optimal accuracy: argmax_k true_probs(x)[k] vs the ACTUAL realized
    (noisy) test label y_test. No model involved -- a property of the
    problem itself, the ceiling nothing can beat.
  - Trained (active-learning, cheat mode) model's accuracy: its own
    prediction vs that SAME realized y_test -- not vs the Bayes-optimal
    target, unlike every other accuracy figure in this codebase. This is
    the operationally honest number: how often would this model actually
    be right, if deployed.

Uses sigma rescaled per-d (sigma_for_d) to hold ||v_k|| ~ constant across
d, consistent with d_sweep_rescaled.py -- change to a fixed sigma=0.02 if
you want the unscaled comparison instead.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset
from d_sweep_rescaled import sigma_for_d


def compare_bayes_vs_model(ds=(2, 4, 8), delta=1.0, K=5, N=2000, seed=23):
    results = []
    for d in ds:
        sigma = sigma_for_d(d)
        X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed, sigma=sigma)
        split = int(0.8 * N)
        X_train, y_train = X[:split], y[:split]
        X_test,  y_test  = X[split:], y[split:]
        T = len(X_train)

        X_train_aug = np.hstack([X_train, np.ones((T, 1))])
        X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])

        true_best_test = np.argmax(np.array([true_probs(x) for x in X_test]), axis=1)
        bayes_acc = np.mean(true_best_test == y_test)   # Bayes-optimal vs REALIZED label

        coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=delta, burn_in=20)
        for i in range(T):
            coal.observe(X_train_aug[i], y_train[i], active=True)
        model_acc = np.mean(coal.predict_batch(X_test_aug) == y_test)   # model vs REALIZED label

        print(f"d={d:>2}  sigma={sigma:.4f}  queries={coal.n_queries}/{T} ({100*coal.n_queries/T:.1f}%)  "
              f"Bayes acc={bayes_acc:.3f}  Model acc={model_acc:.3f}")
        results.append(dict(d=d, sigma=sigma, bayes_acc=bayes_acc, model_acc=model_acc,
                             queries=coal.n_queries, T=T))

    # --- Diagram: grouped bars, Bayes vs Model accuracy, per d ---
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(ds))
    width = 0.35
    bayes_vals = [r['bayes_acc'] for r in results]
    model_vals = [r['model_acc'] for r in results]

    b1 = ax.bar(x - width/2, bayes_vals, width, label='Bayes-optimal (vs realized label)', color='steelblue')
    b2 = ax.bar(x + width/2, model_vals, width, label='Trained model (vs realized label)', color='darkorange')
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.axhline(1/K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels([f'd={d}' for d in ds])
    ax.set_ylabel('Accuracy vs actual (realized, noisy) test label')
    ax.set_ylim(0, 1.0)
    ax.set_title(f'Bayes-optimal vs trained-model accuracy, by d\n'
                 f'(cheat mode, delta={delta}, ||v_k|| held constant across d)')
    ax.legend()
    fig.tight_layout()
    out = 'd_bayes_vs_model_accuracy.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return results


if __name__ == '__main__':
    compare_bayes_vs_model()
