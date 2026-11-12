"""
For each K in the k_sweep, retrain cheat mode at its "best delta" (the
smallest delta that crosses 95% coverage in delta_sweep_filtered.py's
sweep), identify every post-burn-in training round where COAL chose to
SKIP (not query), and visualize the confidence interval it was looking at
when it made that decision, against the true probability for every class.

If more than 20 examples were skipped, randomly sample 20 to keep the
panel grid readable.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset

BEST_DELTA = {2: 2, 3: 2, 4: 1, 5: 2}   # first delta to cross 95% PER-EXAMPLE coverage, per delta_sweep_filtered.py


def visualize_skip_cis_for_k(K, delta, d=20, N=2000, seed=23, max_panels=20, pick_seed=11, out_path=None):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)
    X_train_aug = np.hstack([X_train, np.ones((T, 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])

    all_p_train = np.array([true_probs(x) for x in X_train])
    true_best_train = np.argmax(all_p_train, axis=1)

    coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=delta, burn_in=20)

    skipped = []   # list of dicts: idx, ucb, lcb, tp, correct, covered
    for i in range(T):
        all_fit = all(m.is_fit for m in coal.models)
        is_decision_point = all_fit and (coal.step >= coal.burn_in)
        ucb_lcb_now = coal._get_ucb_lcb(X_train_aug[i]) if is_decision_point else None

        queried = coal.observe(X_train_aug[i], y_train[i], active=True)

        if is_decision_point and not queried:
            ucb, lcb = ucb_lcb_now
            tp = all_p_train[i]
            pred = int(np.argmax((ucb + lcb) / 2.0))
            covered = bool(np.all((lcb <= tp) & (tp <= ucb)))
            skipped.append(dict(idx=i, ucb=ucb, lcb=lcb, tp=tp, covered=covered,
                                 correct=(pred == true_best_train[i])))

    n_skip = len(skipped)
    queries = coal.n_queries
    print(f"K={K}, delta={delta}: {n_skip} skipped out of {T} training rounds "
          f"({100*n_skip/T:.1f}%)")

    # --- Overall (test-set) metrics at this delta, for context ---
    coal._fit_all()
    true_best_test = np.argmax(np.array([true_probs(x) for x in X_test]), axis=1)
    test_acc = np.mean(coal.predict_batch(X_test_aug) == true_best_test)
    test_cov = []
    for i in range(len(X_test)):
        ucb, lcb = coal.get_ucb_lcb(X_test_aug[i])
        tp = true_probs(X_test[i])
        test_cov.append(np.all((lcb <= tp) & (tp <= ucb)))
    test_coverage = np.mean(test_cov)

    # --- Skip-population metrics ---
    skip_coverage = np.mean([s['covered'] for s in skipped]) if n_skip else float('nan')
    skip_acc = np.mean([s['correct'] for s in skipped]) if n_skip else float('nan')
    print(f"  Test-set: accuracy={test_acc:.3f}  per-example coverage={test_coverage:.3f}  "
          f"queries={queries}/{T} ({100*queries/T:.1f}%)")
    print(f"  Skipped population: per-example coverage={skip_coverage:.3f}  "
          f"inferred-label accuracy={skip_acc:.3f}")

    pick_rng = np.random.RandomState(pick_seed)
    if n_skip > max_panels:
        show = [skipped[j] for j in pick_rng.choice(n_skip, size=max_panels, replace=False)]
        subtitle_note = f"(randomly sampled {max_panels} of {n_skip} skipped)"
    else:
        show = skipped
        subtitle_note = f"(all {n_skip} skipped)"

    if n_skip == 0:
        print(f"  No skipped examples for K={K}, delta={delta} -- nothing to plot.")
        return dict(K=K, delta=delta, n_skip=0, test_acc=test_acc, test_coverage=test_coverage,
                    queries=queries, T=T)

    n_panels = len(show)
    ncols = 5
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    class_colors = plt.cm.tab10(np.linspace(0, 1, K))

    n_correct = 0
    for panel_i, s in enumerate(show):
        ax = axes[panel_i]
        ucb, lcb, tp = s['ucb'], s['lcb'], s['tp']
        n_correct += int(s['correct'])

        for k in range(K):
            color = class_colors[k]
            ax.plot([lcb[k], ucb[k]], [k, k], lw=6, color=color, alpha=0.9, solid_capstyle='butt')
            ax.scatter([tp[k]], [k], marker='|', color='black', s=160, zorder=5, linewidths=2)

        ax.set_yticks(range(K))
        ax.set_yticklabels([f'c{k}' for k in range(K)], fontsize=8)
        cov_tag = 'covered' if s['covered'] else 'not covered'
        ax.set_title(f"train idx {s['idx']}: {cov_tag}\n{'correct' if s['correct'] else 'WRONG'}",
                     fontsize=9, color='seagreen' if s['correct'] else 'crimson')
        ax.grid(True, axis='x', alpha=0.25)

    for j in range(n_panels, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"K={K}, best delta={delta}  |  skipped {n_skip}/{T} ({100*n_skip/T:.1f}%), "
                 f"shown {n_panels}, correct {n_correct}/{n_panels}\n"
                 f"skip-population: per-example coverage={skip_coverage:.0%}, "
                 f"inferred-label accuracy={skip_acc:.0%}  |  "
                 f"full test set: accuracy={test_acc:.0%}, per-example coverage={test_coverage:.0%}, "
                 f"queries={100*queries/T:.0f}%",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = out_path if out_path is not None else f'k_sweep_skip_ci_K{K}.png'
    fig.savefig(out, dpi=150)
    print(f"  Saved: {out}")

    return dict(K=K, delta=delta, n_skip=n_skip, n_shown=n_panels, n_correct=n_correct,
                skip_coverage=skip_coverage, skip_acc=skip_acc,
                test_acc=test_acc, test_coverage=test_coverage, queries=queries, T=T)


if __name__ == '__main__':
    results = []
    for K, delta in BEST_DELTA.items():
        results.append(visualize_skip_cis_for_k(K, delta))
    print("\nSummary:")
    for r in results:
        print(r)
