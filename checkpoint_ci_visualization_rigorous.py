"""
Same per-example CI diagnostic as checkpoint_ci_visualization.py, but for
RigorousCOAL (Algorithm 3, paper's eps_m schedule) instead of CheatCOAL --
to directly compare what "uncertainty quantification" looks like in the
non-cheating mode.

No delta-inflation overlay here (rigorous mode's radius isn't a free
hyperparameter -- it's fixed by Theorem 2 given T, K, d, delta_failure).
Instead this shows, side by side with cheat mode, just how much wider and
less discriminative the honest intervals are.
"""

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_rigorous import RigorousCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset


def visualize_checkpoint_rigorous(K=5, d=20, N=2000, seed=23, n_points=10, pick_seed=11,
                                   rigorous_delta=0.05, prec=0.01):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)

    X_train_aug = np.hstack([X_train, np.ones((T, 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])

    coal = RigorousCOAL(K=K, d=d + 1, reg=1.0, delta=0.5, burn_in=20)
    for i in range(T):
        coal.observe(X_train_aug[i], y_train[i], active=True, T=T, prec=prec, delta=rigorous_delta)
    coal._fit_all()

    per_example_covered = []
    all_ucb, all_lcb, all_tp = [], [], []
    for i in range(len(X_test)):
        ucb, lcb = coal.get_ucb_lcb_algorithm3(X_test_aug[i], T=T, prec=prec, delta=rigorous_delta)
        tp = true_probs(X_test[i])
        covered = np.all((lcb <= tp) & (tp <= ucb))
        per_example_covered.append(covered)
        all_ucb.append(ucb); all_lcb.append(lcb); all_tp.append(tp)
    per_example_covered = np.array(per_example_covered)
    all_ucb, all_lcb, all_tp = np.array(all_ucb), np.array(all_lcb), np.array(all_tp)

    # NOTE: the UCB/LCB midpoint is a poor proxy for "point estimate" in
    # rigorous mode -- BINSEARCH's HIGH/LOW bounds are independently targeted
    # at different sentinels (r=2, r=-1) and are NOT symmetric around theta.x
    # the way cheat mode's interval is by construction. Once intervals go
    # sentinel-wide, the midpoint drifts and stops tracking the model's
    # actual prediction. Report both to make this explicit.
    midpoints = (all_ucb + all_lcb) / 2.0
    spearman_per_point = [spearmanr(midpoints[i], all_tp[i]).correlation for i in range(len(X_test))]
    top1_match_midpoint = np.mean(np.argmax(midpoints, axis=1) == np.argmax(all_tp, axis=1))

    true_best_test = np.argmax(all_tp, axis=1)
    pred_theta = coal.predict_batch(X_test_aug)
    top1_match_theta = np.mean(pred_theta == true_best_test)

    width = np.mean(all_ucb - all_lcb)
    queries = coal.n_queries
    top1_match = top1_match_theta   # use the non-degenerate one for the diagram title

    print(f"Rigorous mode -- queries: {queries}/{T} ({100*queries/T:.1f}%)")
    print(f"Per-example coverage: {per_example_covered.mean():.3f}")
    print(f"Mean interval width: {width:.3f}")
    print(f"Mean Spearman rank correlation (UCB/LCB midpoint order vs true-prob order): "
          f"{np.nanmean(spearman_per_point):.3f}")
    print(f"Top-1 match via UCB/LCB midpoint (degenerate once sentinel-wide): {top1_match_midpoint:.3f}")
    print(f"Top-1 match via actual theta.x prediction (fair comparison to cheat mode): {top1_match_theta:.3f}")

    # --- Diagram: n_points random panels, all K classes ---
    pick_rng = np.random.RandomState(pick_seed)
    show_idx = pick_rng.choice(len(X_test), size=n_points, replace=False)

    ncols = 5
    nrows = int(np.ceil(n_points / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    class_colors = plt.cm.tab10(np.linspace(0, 1, K))

    for panel_i, i in enumerate(show_idx):
        ax = axes[panel_i]
        tp = all_tp[i]
        ucb, lcb = all_ucb[i], all_lcb[i]
        mid = midpoints[i]

        for k in range(K):
            color = class_colors[k]
            ax.plot([lcb[k], ucb[k]], [k, k], lw=6, color=color, alpha=0.9, solid_capstyle='butt')
            ax.scatter([tp[k]], [k], marker='|', color='black', s=160, zorder=5, linewidths=2)

        rank_ok = tuple(np.argsort(-mid)) == tuple(np.argsort(-tp))
        ax.set_yticks(range(K))
        ax.set_yticklabels([f'c{k}' for k in range(K)], fontsize=8)
        ax.set_title(f"pt {i}: {'COVERED' if per_example_covered[i] else 'not covered'}\n"
                     f"ranking {'MATCHES' if rank_ok else 'differs'}",
                     fontsize=9, color='seagreen' if per_example_covered[i] else 'crimson')
        ax.set_xlim(-1.3, 2.3)
        ax.grid(True, axis='x', alpha=0.25)

    for j in range(n_points, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Rigorous mode checkpoint CI diagnostics (Algorithm 3, paper eps_m schedule)\n"
                 f"coverage={per_example_covered.mean():.0%}, width={width:.2f}, queries={100*queries/T:.0f}%\n"
                 f"top-1 via theta.x={top1_match_theta:.0%} (fair vs cheat) | "
                 f"top-1 via UCB/LCB midpoint={top1_match_midpoint:.0%} (degenerate proxy)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = 'checkpoint_ci_visualization_rigorous.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return dict(coverage=per_example_covered.mean(), width=width, top1_match=top1_match,
                mean_spearman=float(np.nanmean(spearman_per_point)), queries=queries, T=T)


if __name__ == '__main__':
    visualize_checkpoint_rigorous()
