"""
When COAL decides to SKIP an example (not query it),
is that actually a safe decision? -- i.e. does the model's own
current prediction at that moment already match the true (Bayes-optimal)
label? If skip-accuracy is high (close to 1), skipping is justified: COAL
is confidently right. If it's low, COAL is skipping examples it's actually
wrong about, which would be a real problem for the active-learning logic
(skipping because it's overconfident, not because it's correct).

Compares skip-accuracy against query-accuracy (how often COAL's current
prediction is right on the examples it chooses to query anyway -- expected
to be lower, since those are exactly the ones it's uncertain about) and the
final overall test accuracy, as reference points.

Uses the rejection-filtered dataset + intercept augmentation (the two
already-confirmed fixes), cheat mode at delta=0.5.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset
from difficulty import margins


def check_skip_accuracy(delta=0.5, K=5, d=20, N=2000, seed=23):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)

    X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])
    d_feed = d + 1

    all_p_train = np.array([true_probs(x) for x in X_train])
    margin_train = margins(all_p_train)
    true_best_train = np.argmax(all_p_train, axis=1)
    true_best_test  = np.argmax(np.array([true_probs(x) for x in X_test]), axis=1)

    coal = CheatCOAL(K=K, d=d_feed, reg=1.0, delta=delta, burn_in=20)

    skip_correct, skip_total = 0, 0
    query_correct, query_total = 0, 0   # excludes forced burn-in queries
    skip_margin, query_margin = [], []   # difficulty (true margin) of each decision point

    for i in range(T):
        all_fit = all(m.is_fit for m in coal.models)
        had_prediction = all_fit and (coal.step >= coal.burn_in)  # a real post-burn-in decision point
        pred = coal.predict(X_train_aug[i]) if all_fit else None

        queried = coal.observe(X_train_aug[i], y_train[i], active=True)

        is_burn_in_force = (i < coal.burn_in) or (not all_fit)
        if had_prediction and not is_burn_in_force:
            correct = (pred == true_best_train[i])
            if queried:
                query_correct += int(correct)
                query_total += 1
                query_margin.append(margin_train[i])
            else:
                skip_correct += int(correct)
                skip_total += 1
                skip_margin.append(margin_train[i])

    skip_acc = skip_correct / skip_total if skip_total else float('nan')
    query_acc = query_correct / query_total if query_total else float('nan')
    final_test_acc = np.mean(coal.predict_batch(X_test_aug) == true_best_test)

    print(f"Post-burn-in decision points: {skip_total + query_total}  (skipped={skip_total}, queried={query_total})")
    print(f"Skip-accuracy  (correct when COAL chose to SKIP):  {skip_acc:.3f}  ({skip_correct}/{skip_total})")
    print(f"Query-accuracy (correct when COAL chose to QUERY): {query_acc:.3f}  ({query_correct}/{query_total})")
    print(f"Final held-out test accuracy:                      {final_test_acc:.3f}")

    # --- Diagrams ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = ['Skipped\nexamples', 'Queried\nexamples', 'Final test\naccuracy']
    values = [skip_acc, query_acc, final_test_acc]
    colors = ['seagreen', 'crimson', 'steelblue']
    bars = axes[0].bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[0].axhline(1/K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Fraction matching true (Bayes-optimal) label")
    axes[0].set_title(f"Is it safe for COAL to skip?\nn_skip={skip_total}, n_query={query_total}")
    axes[0].legend()

    # Why are skips safe? -- difficulty (true margin) of skipped vs queried points
    bins = np.linspace(min(min(skip_margin, default=0.0), min(query_margin, default=0.0)),
                        max(max(skip_margin, default=0.3), max(query_margin, default=0.3)), 25)
    axes[1].hist(query_margin, bins=bins, alpha=0.6, color='crimson', label=f'queried (n={query_total})')
    axes[1].hist(skip_margin, bins=bins, alpha=0.6, color='seagreen', label=f'skipped (n={skip_total})')
    axes[1].set_xlabel("True margin: max_i p(y=i|x) - 2nd max_i p(y=i|x), at decision time")
    axes[1].set_ylabel('Count')
    axes[1].set_title('Why skipping is safe:\nskipped points have a wide (unambiguous) margin')
    axes[1].legend()

    fig.suptitle(f"Active-learning skip sanity check (cheat mode, delta={delta})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = 'skip_accuracy_check.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    print(f"Mean margin -- skipped: {np.mean(skip_margin):.3f}   queried: {np.mean(query_margin):.3f}")

    return dict(skip_acc=skip_acc, query_acc=query_acc, final_test_acc=final_test_acc,
                skip_total=skip_total, query_total=query_total)


if __name__ == '__main__':
    check_skip_accuracy()
