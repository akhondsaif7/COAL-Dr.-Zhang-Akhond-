"""
For every example COAL chose to SKIP (55 of them, in the standard
filtered+augmented cheat-mode run), show the actual confidence bands that
drove that decision: the estimated [LCB,UCB] for the class COAL kept as
"possibly optimal" (its prediction), and for the strongest remaining
competitor (highest UCB among the other classes) -- these are exactly the
two bands _should_query compares. Overlaid tick marks show the TRUE
probability of the predicted class and of the true Bayes-optimal class
(when they differ), so you can see directly whether the skip decision's
bands reflected reality or were an artifact of estimation noise.

Rows are sorted by true margin (ascending) and colored by whether the skip
was actually correct, so the smallest-margin / most dangerous skips are
grouped together at one end.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset
from difficulty import margins


def diagram_skip_bands(delta=0.5, K=5, d=20, N=2000, seed=23):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    T = len(X_train)
    X_train_aug = np.hstack([X_train, np.ones((T, 1))])

    all_p_train = np.array([true_probs(x) for x in X_train])
    margin_train = margins(all_p_train)
    true_best_train = np.argmax(all_p_train, axis=1)

    coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=delta, burn_in=20)
    rows = []

    for i in range(T):
        all_fit = all(m.is_fit for m in coal.models)
        had_prediction = all_fit and (coal.step >= coal.burn_in)
        pred = coal.predict(X_train_aug[i]) if all_fit else None
        ucb, lcb = (coal._get_ucb_lcb(X_train_aug[i]) if had_prediction else (None, None))

        queried = coal.observe(X_train_aug[i], y_train[i], active=True)

        is_burn_in_force = (i < coal.burn_in) or (not all_fit)
        if had_prediction and not is_burn_in_force and not queried:
            # The two bands _should_query actually compares: the kept class
            # (highest LCB) and the strongest remaining competitor (highest
            # UCB among the rest).
            best_k = np.argmax(lcb)
            rest = [k for k in range(K) if k != best_k]
            runner_k = rest[np.argmax(ucb[rest])]
            rows.append(dict(
                margin=margin_train[i], correct=(pred == true_best_train[i]),
                best_k=best_k, lcb_best=lcb[best_k], ucb_best=ucb[best_k],
                true_best_p=all_p_train[i, best_k],
                runner_k=runner_k, lcb_runner=lcb[runner_k], ucb_runner=ucb[runner_k],
                true_runner_p=all_p_train[i, runner_k],
                true_best_class=true_best_train[i],
                true_best_class_p=all_p_train[i, true_best_train[i]],
            ))

    rows.sort(key=lambda r: r['margin'])
    n = len(rows)
    print(f"{n} skipped examples  ({sum(r['correct'] for r in rows)} correct, "
          f"{sum(not r['correct'] for r in rows)} wrong)")

    fig, ax = plt.subplots(figsize=(10, max(6, 0.17 * n)))
    bar_h = 0.32

    for row_i, r in enumerate(rows):
        y0 = row_i
        # kept ("predicted") class band -- top sub-bar
        ax.plot([r['lcb_best'], r['ucb_best']], [y0 + bar_h, y0 + bar_h], lw=6,
                color='steelblue', solid_capstyle='butt', alpha=0.9)
        ax.scatter([r['true_best_p']], [y0 + bar_h], color='black', marker='|', s=90, zorder=5)

        # strongest competitor band -- bottom sub-bar
        ax.plot([r['lcb_runner'], r['ucb_runner']], [y0 - bar_h, y0 - bar_h], lw=6,
                color='darkorange', solid_capstyle='butt', alpha=0.9)
        ax.scatter([r['true_runner_p']], [y0 - bar_h], color='black', marker='|', s=90, zorder=5)

        # if the prediction was wrong, mark where the TRUE best class actually sat
        if not r['correct']:
            ax.scatter([r['true_best_class_p']], [y0], color='red', marker='*', s=70, zorder=6)

    ax.axvline(0, color='gray', lw=0.6, alpha=0.5)
    ylabels = [f"margin={r['margin']:.3f}  {'OK' if r['correct'] else 'WRONG'}" for r in rows]
    ax.set_yticks(range(n))
    ax.set_yticklabels(ylabels, fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color('seagreen' if r['correct'] else 'crimson')
        if not r['correct']:
            tick.set_fontweight('bold')

    ax.set_xlabel('Estimated probability (cheat-mode UCB/LCB)')
    ax.set_title(f"Confidence bands for all {n} skipped examples (sorted by true margin, ascending)\n"
                 "blue = kept/predicted class, orange = strongest competitor, "
                 "black tick = each band's true probability, red star = true best class (if prediction wrong)",
                 fontsize=10)
    ax.set_xlim(-0.3, 1.1)
    fig.tight_layout()
    out = 'skip_bands_diagram.png'
    fig.savefig(out, dpi=150)
    print(f"Saved: {out}")


if __name__ == '__main__':
    diagram_skip_bands()
