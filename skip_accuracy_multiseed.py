"""
Stability check for the skip-accuracy sanity check: everything in
skip_accuracy_check.py / skip_bands_diagram.py was run on a single seed
(23). Since in the "wrong skip" population there was only 4 points out of 55,
conclusions drawn from it could easily be an artifact of that one seed's
particular draw. This reruns the same experiment across many independent
seeds and reports the distribution of the key numbers, the same way
clipping_rate_analysis.py did for the clipping rate.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset
from difficulty import margins


def run_seed(seed, delta=0.5, K=5, d=20, N=2000, burn_in=20):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)
    X_train_aug = np.hstack([X_train, np.ones((T, 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])

    all_p_train = np.array([true_probs(x) for x in X_train])
    margin_train = margins(all_p_train)
    true_best_train = np.argmax(all_p_train, axis=1)
    true_best_test  = np.argmax(np.array([true_probs(x) for x in X_test]), axis=1)

    coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=delta, burn_in=burn_in)
    skip_margins_correct, skip_margins_wrong = [], []
    skip_correct, skip_total = 0, 0
    query_correct, query_total = 0, 0

    for i in range(T):
        all_fit = all(m.is_fit for m in coal.models)
        had_prediction = all_fit and (coal.step >= coal.burn_in)
        pred = coal.predict(X_train_aug[i]) if all_fit else None

        queried = coal.observe(X_train_aug[i], y_train[i], active=True)

        is_burn_in_force = (i < coal.burn_in) or (not all_fit)
        if had_prediction and not is_burn_in_force:
            correct = (pred == true_best_train[i])
            if queried:
                query_correct += int(correct)
                query_total += 1
            else:
                skip_correct += int(correct)
                skip_total += 1
                (skip_margins_correct if correct else skip_margins_wrong).append(margin_train[i])

    final_test_acc = np.mean(coal.predict_batch(X_test_aug) == true_best_test)

    return dict(
        seed=seed,
        skip_total=skip_total, query_total=query_total,
        skip_acc=skip_correct / skip_total if skip_total else float('nan'),
        query_acc=query_correct / query_total if query_total else float('nan'),
        final_test_acc=final_test_acc,
        mean_margin_correct=np.mean(skip_margins_correct) if skip_margins_correct else float('nan'),
        mean_margin_wrong=np.mean(skip_margins_wrong) if skip_margins_wrong else float('nan'),
        n_wrong=len(skip_margins_wrong),
    )


def run_multiseed(n_seeds=20, delta=0.5):
    results = [run_seed(seed, delta=delta) for seed in range(n_seeds)]

    print(f"{'seed':>4} {'skip_tot':>9} {'skip_acc':>9} {'query_acc':>10} {'test_acc':>9} "
          f"{'n_wrong':>8} {'marg_ok':>8} {'marg_wrong':>11}")
    for r in results:
        print(f"{r['seed']:>4} {r['skip_total']:>9} {r['skip_acc']:>9.3f} {r['query_acc']:>10.3f} "
              f"{r['final_test_acc']:>9.3f} {r['n_wrong']:>8} {r['mean_margin_correct']:>8.3f} "
              f"{r['mean_margin_wrong']:>11.3f}")

    skip_accs = np.array([r['skip_acc'] for r in results])
    query_accs = np.array([r['query_acc'] for r in results])
    test_accs = np.array([r['final_test_acc'] for r in results])
    skip_totals = np.array([r['skip_total'] for r in results])
    n_wrongs = np.array([r['n_wrong'] for r in results])

    print(f"\nAcross {n_seeds} seeds:")
    print(f"  skip_acc:    mean={skip_accs.mean():.3f}  std={skip_accs.std():.3f}  "
          f"min={skip_accs.min():.3f}  max={skip_accs.max():.3f}")
    print(f"  query_acc:   mean={query_accs.mean():.3f}  std={query_accs.std():.3f}")
    print(f"  test_acc:    mean={test_accs.mean():.3f}  std={test_accs.std():.3f}")
    print(f"  skip_total:  mean={skip_totals.mean():.1f}  min={skip_totals.min()}  max={skip_totals.max()}")
    print(f"  n_wrong:     mean={n_wrongs.mean():.1f}  min={n_wrongs.min()}  max={n_wrongs.max()}")

    # margin gap only defined where a seed actually had >=1 wrong skip
    valid = [r for r in results if r['n_wrong'] > 0]
    gaps = np.array([r['mean_margin_correct'] - r['mean_margin_wrong'] for r in valid])
    print(f"\n  seeds with >=1 wrong skip: {len(valid)}/{n_seeds}")
    if len(gaps):
        print(f"  margin gap (correct-skip margin minus wrong-skip margin): "
              f"mean={gaps.mean():.3f}  std={gaps.std():.3f}  min={gaps.min():.3f}  max={gaps.max():.3f}  "
              f"(positive = correct skips have wider margins, as seed 23 showed)")

    # --- Diagrams ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].bar(range(n_seeds), skip_accs, color='seagreen')
    axes[0].axhline(skip_accs.mean(), color='crimson', ls='--', lw=1.5, label=f'mean={skip_accs.mean():.3f}')
    axes[0].set_xlabel('Seed')
    axes[0].set_ylabel('Skip-accuracy')
    axes[0].set_title('Skip-accuracy across seeds')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].bar(range(n_seeds), skip_totals, color='steelblue')
    axes[1].axhline(skip_totals.mean(), color='crimson', ls='--', lw=1.5, label=f'mean={skip_totals.mean():.1f}')
    axes[1].set_xlabel('Seed')
    axes[1].set_ylabel('# of skipped examples (out of 1600 train)')
    axes[1].set_title('Skip count across seeds')
    axes[1].legend()

    seeds_with_wrong = [r['seed'] for r in valid]
    gap_vals = [r['mean_margin_correct'] - r['mean_margin_wrong'] for r in valid]
    colors = ['seagreen' if g > 0 else 'crimson' for g in gap_vals]
    axes[2].bar(range(len(valid)), gap_vals, color=colors)
    axes[2].set_xticks(range(len(valid)))
    axes[2].set_xticklabels(seeds_with_wrong, fontsize=8)
    axes[2].axhline(0, color='gray', lw=0.8)
    axes[2].set_xlabel('Seed (only seeds with >=1 wrong skip)')
    axes[2].set_ylabel('Mean margin: correct skips - wrong skips')
    axes[2].set_title(f'Margin gap across seeds\n(positive = matches seed-23 pattern), n={len(valid)}/{n_seeds}')

    fig.suptitle(f'Stability of the skip-accuracy sanity check across {n_seeds} seeds (delta={delta})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = 'skip_accuracy_multiseed.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return results


if __name__ == '__main__':
    run_multiseed()
