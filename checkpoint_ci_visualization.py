"""
For every training round where COAL chose to SKIP (not query) -- the "55
inferred labels" population from skip_accuracy_check.py -- visualize the
confidence interval it was actually looking at when it made that decision
(not a later checkpoint's interval), against the true probability for each
class. Uses TRAINING data, since skip/query decisions only ever happen on
training examples; test data is never fed through observe().

Tests two hypotheses:

(1) For a not-covered skipped point, would a modestly larger delta have
    covered it? (shown as a lighter overlay bar at an inflated delta,
    alongside the current-delta bar -- computed analytically, since a
    cheat-mode interval's half-width scales as sqrt(delta))
(2) Even when literal coverage fails, is the ORDER of the per-class point
    estimates still roughly consistent with the order of the true
    probabilities? (this is what would explain high label-inference
    accuracy despite imperfect coverage -- argmax only needs the RANKING
    to be right, not the absolute calibration)

"Per-example coverage" here means ALL K classes' true probabilities are
simultaneously inside their own CI at decision time -- a stricter notion
than the pooled per-(point,class) coverage rate reported elsewhere.
"""

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset


def visualize_checkpoint(delta=0.5, delta_inflate=4.0, K=5, d=20, N=2000, seed=23,
                          n_each=10, pick_seed=11, out_path=None):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    T = len(X_train)
    X_train_aug = np.hstack([X_train, np.ones((T, 1))])

    all_p_train = np.array([true_probs(x) for x in X_train])
    true_best_train = np.argmax(all_p_train, axis=1)

    # --- Train while recording the decision-time interval for every SKIPPED round ---
    coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=delta, burn_in=20)
    skipped = []   # dicts: idx, ucb, lcb, tp, correct
    n_decision_points = 0
    for i in range(T):
        all_fit = all(m.is_fit for m in coal.models)
        is_decision_point = all_fit and (coal.step >= coal.burn_in)
        ucb_lcb_now = coal._get_ucb_lcb(X_train_aug[i]) if is_decision_point else None

        queried = coal.observe(X_train_aug[i], y_train[i], active=True)

        if is_decision_point:
            n_decision_points += 1
            if not queried:
                ucb, lcb = ucb_lcb_now
                tp = all_p_train[i]
                covered = np.all((lcb <= tp) & (tp <= ucb))
                pred = int(np.argmax((ucb + lcb) / 2.0))
                skipped.append(dict(idx=i, ucb=ucb, lcb=lcb, tp=tp, covered=covered,
                                     correct=(pred == true_best_train[i])))

    n_skip = len(skipped)
    n_correct = sum(s['correct'] for s in skipped)
    per_example_coverage = np.mean([s['covered'] for s in skipped])
    print(f"Decision points: {n_decision_points}  (skipped={n_skip}, queried={n_decision_points - n_skip})")
    print(f"Per-example coverage on SKIPPED points (ALL {K} classes covered simultaneously): "
          f"{per_example_coverage:.3f} ({sum(s['covered'] for s in skipped)}/{n_skip})")
    print(f"Inferred label correct on SKIPPED points: {n_correct}/{n_skip} "
          f"({100*n_correct/n_skip:.1f}%)")

    # --- Ranking consistency, over the skipped population ---
    all_ucb = np.array([s['ucb'] for s in skipped])
    all_lcb = np.array([s['lcb'] for s in skipped])
    all_tp  = np.array([s['tp']  for s in skipped])
    midpoints = (all_ucb + all_lcb) / 2.0
    spearman_per_point = [spearmanr(midpoints[i], all_tp[i]).correlation for i in range(n_skip)]
    top1_match = np.mean(np.argmax(midpoints, axis=1) == np.argmax(all_tp, axis=1))
    full_rank_match = np.mean([tuple(np.argsort(-midpoints[i])) == tuple(np.argsort(-all_tp[i]))
                                for i in range(n_skip)])
    print(f"Mean Spearman rank correlation (point-estimate order vs true-prob order): "
          f"{np.nanmean(spearman_per_point):.3f}")
    print(f"Top-1 (argmax) match rate: {top1_match:.3f}")
    print(f"Full ranking match rate (all {K} classes in same order): {full_rank_match:.3f}")

    # --- Pick a mix of covered / not-covered SKIPPED points to visualize ---
    pick_rng = np.random.RandomState(pick_seed)
    covered_idx = [j for j, s in enumerate(skipped) if s['covered']]
    uncovered_idx = [j for j, s in enumerate(skipped) if not s['covered']]
    show_covered = pick_rng.choice(covered_idx, size=min(n_each, len(covered_idx)), replace=False) \
        if covered_idx else np.array([], dtype=int)
    show_uncovered = pick_rng.choice(uncovered_idx, size=min(n_each, len(uncovered_idx)), replace=False) \
        if uncovered_idx else np.array([], dtype=int)
    show_j = list(show_uncovered) + list(show_covered)   # uncovered first
    n_panels = len(show_j)

    # --- Diagram: one panel per skipped point, all K classes' CIs + true prob + inflated overlay ---
    ncols = 5
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)
    class_colors = plt.cm.tab10(np.linspace(0, 1, K))

    for panel_i, j in enumerate(show_j):
        s = skipped[j]
        ax = axes[panel_i]
        tp, ucb, lcb = s['tp'], s['ucb'], s['lcb']

        # inflated-delta interval (half-width scales as sqrt(delta))
        scale = np.sqrt(delta_inflate / delta)
        mid = (ucb + lcb) / 2.0
        half = (ucb - lcb) / 2.0
        ucb_inf, lcb_inf = mid + half * scale, mid - half * scale

        for k in range(K):
            color = class_colors[k]
            ax.plot([lcb_inf[k], ucb_inf[k]], [k, k], lw=10, color=color, alpha=0.18, solid_capstyle='butt')
            ax.plot([lcb[k], ucb[k]], [k, k], lw=6, color=color, alpha=0.9, solid_capstyle='butt')
            ax.scatter([tp[k]], [k], marker='|', color='black', s=160, zorder=5, linewidths=2)

        status = 'COVERED' if s['covered'] else 'NOT covered'
        rank_ok = tuple(np.argsort(-mid)) == tuple(np.argsort(-tp))
        ax.set_yticks(range(K))
        ax.set_yticklabels([f'c{k}' for k in range(K)], fontsize=8)
        ax.set_title(f"train idx {s['idx']}: {status}\n"
                     f"{'correct' if s['correct'] else 'WRONG'}, ranking {'MATCHES' if rank_ok else 'differs'}",
                     fontsize=9, color='seagreen' if s['covered'] else 'crimson')
        ax.grid(True, axis='x', alpha=0.25)

    for jj in range(n_panels, len(axes)):
        axes[jj].axis('off')

    fig.suptitle(f"Skipped-example CI diagnostics (cheat mode, delta={delta}; pale bar = delta={delta_inflate} inflated)\n"
                 f"skipped {n_skip}/{n_decision_points} decisions | per-example coverage={per_example_coverage:.0%} | "
                 f"inferred-label correct={n_correct}/{n_skip} | rank match={full_rank_match:.0%}",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = out_path if out_path is not None else 'checkpoint_ci_visualization.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return dict(n_skip=n_skip, n_correct=n_correct, per_example_coverage=per_example_coverage,
                top1_match=top1_match, full_rank_match=full_rank_match,
                mean_spearman=float(np.nanmean(spearman_per_point)))


if __name__ == '__main__':
    visualize_checkpoint()
