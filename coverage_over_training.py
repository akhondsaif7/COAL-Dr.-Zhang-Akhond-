"""
Coverage verification, a follow-up to the
single-snapshot, per-(example,label)-averaged check used everywhere else
in this codebase:

  1. Checkpointed throughout training (10 checkpoints spread across the
     1600 training rounds), instead of only checked once at the end -- so
     we can see whether coverage is stable, improving, or degrading as
     more data arrives, not just its value at one final snapshot.

  2. Computed at the EXAMPLE level: an example is "covered" only if its
     confidence interval contains the true f*(x,i) for ALL K labels i
     simultaneously -- not averaged across (example, label) pairs. This is
     the decision-relevant notion of coverage, since COAL's query decision
     (_should_query) depends on all K intervals jointly for a given point.
     If per-class miscoverage is roughly independent across classes, joint
     (all-K) coverage should come in well below the per-class average --
     e.g. 0.934^5 =~ 0.71 if each class is independently covered 93.4% of
     the time.

Both metrics are reported side by side at every checkpoint, since they
answer different questions (per-class interval quality vs. whether the
decision-relevant joint event holds).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset
from difficulty import margins


def coverage_over_training(delta=0.5, K=5, d=20, N=2000, seed=23, n_checkpoints=10, n_check=None,
                            verbose=True, make_plot=True):
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset(K=K, d=d, N=N, seed=seed)
    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)

    X_train_aug = np.hstack([X_train, np.ones((T, 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])

    # Fixed held-out sample, reused at every checkpoint so the only thing
    # that changes between checkpoints is how much training data COAL has seen.
    # Defaults to the full test set
    if n_check is None or n_check >= len(X_test):
        n_check = len(X_test)
        idx = np.arange(len(X_test))
    else:
        pick_rng = np.random.RandomState(11)
        idx = pick_rng.choice(len(X_test), size=n_check, replace=False)
    true_p_checked = np.array([true_probs(X_test[i]) for i in idx])

    coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=delta, burn_in=20)

    checkpoint_every = T // n_checkpoints
    checkpoints = [checkpoint_every * c for c in range(1, n_checkpoints + 1)]
    checkpoints[-1] = T   # make sure the final checkpoint covers all of T

    results = []
    next_cp = 0
    for i in range(T):
        coal.observe(X_train_aug[i], y_train[i], active=True)

        if i + 1 == checkpoints[next_cp]:
            coal._fit_all()   # force a fresh fit exactly at this checkpoint, not up-to-9-queries stale
            per_class_cov = []
            joint_cov = []
            widths = []
            for j, test_i in enumerate(idx):
                ucb, lcb = coal.get_ucb_lcb(X_test_aug[test_i])
                tp = true_p_checked[j]
                covered_per_class = (lcb <= tp) & (tp <= ucb)   # shape (K,), bool
                per_class_cov.extend(covered_per_class.tolist())
                joint_cov.append(bool(covered_per_class.all()))
                widths.extend((ucb - lcb).tolist())

            results.append(dict(
                t=checkpoints[next_cp],
                n_queries=coal.n_queries,
                per_class_coverage=np.mean(per_class_cov),
                joint_coverage=np.mean(joint_cov),
                mean_width=np.mean(widths),
                joint_cov_per_point=np.array(joint_cov),   # one bool per test point, this checkpoint
            ))
            next_cp += 1
            if next_cp >= len(checkpoints):
                break

    if verbose:
        print(f"{'t':>6} {'n_query':>8} {'per-class cov':>14} {'joint (all-K) cov':>18} {'mean width':>11}")
        for r in results:
            print(f"{r['t']:>6} {r['n_queries']:>8} {r['per_class_coverage']:>14.3f} "
                  f"{r['joint_coverage']:>18.3f} {r['mean_width']:>11.3f}")

        # Naive independence estimate for comparison: per_class_cov^K
        final_per_class = results[-1]['per_class_coverage']
        naive_joint_estimate = final_per_class ** K
        print(f"\nFinal per-class coverage: {final_per_class:.3f}")
        print(f"Naive independence estimate of joint coverage (per_class^K): {naive_joint_estimate:.3f}")
        print(f"Actual measured joint coverage: {results[-1]['joint_coverage']:.3f}")

    if make_plot:
        # --- Diagram ---
        ts = [r['t'] for r in results]
        per_class = [r['per_class_coverage'] for r in results]
        joint = [r['joint_coverage'] for r in results]
        widths = [r['mean_width'] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        axes[0].plot(ts, per_class, 'o-', color='steelblue', lw=2, label='per-(example,label) coverage (old metric)')
        axes[0].plot(ts, joint, 'o-', color='crimson', lw=2, label='per-example joint (all-K) coverage (new metric)')
        axes[0].axhline(0.95, color='gray', ls='--', lw=1, label='95% target')
        axes[0].set_xlabel('Training round t')
        axes[0].set_ylabel('Coverage')
        axes[0].set_title(f'Coverage over training (delta={delta}, n_check={n_check})')
        axes[0].set_ylim(0, 1.05)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, widths, 'o-', color='darkorange', lw=2)
        axes[1].set_xlabel('Training round t')
        axes[1].set_ylabel('Mean interval width (UCB-LCB)')
        axes[1].set_title('Interval width over training')
        axes[1].grid(True, alpha=0.3)

        fig.suptitle('Coverage verification throughout training: per-class vs. joint (example-level)', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        out = 'coverage_over_training.png'
        fig.savefig(out, dpi=150)
        if verbose:
            print(f"\nSaved: {out}")

        # --- Per-test-point persistence: is coverage failure concentrated on
        # a stable subset of hard points, or scattered like noise across
        # different points at different checkpoints? For each test point,
        # count how many of the n_checkpoints it was jointly covered at.
        test_margins = margins(true_p_checked)
        grid = np.array([r['joint_cov_per_point'] for r in results])   # (n_checkpoints, n_check)
        times_covered = grid.sum(axis=0)   # per test point, 0..n_checkpoints

        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

        bins = np.arange(n_checkpoints + 2) - 0.5
        axes2[0].hist(times_covered, bins=bins, color='steelblue', edgecolor='white')
        axes2[0].set_xlabel(f'# of {n_checkpoints} checkpoints this test point was covered at')
        axes2[0].set_ylabel('# of test points')
        axes2[0].set_xticks(range(n_checkpoints + 1))
        axes2[0].set_title('Is coverage failure persistent or scattered?\n'
                            '(bimodal 0-vs-10 = persistent hard/easy points;\n'
                            'centered around middle = scattered, noise-like)')

        axes2[1].scatter(test_margins, times_covered, alpha=0.35, s=18, color='seagreen')
        # binned average trend line
        order = np.argsort(test_margins)
        bin_edges = np.quantile(test_margins, np.linspace(0, 1, 11))
        bin_centers, bin_means = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (test_margins >= lo) & (test_margins <= hi)
            if mask.sum() > 0:
                bin_centers.append(test_margins[mask].mean())
                bin_means.append(times_covered[mask].mean())
        axes2[1].plot(bin_centers, bin_means, 'o-', color='crimson', lw=2, label='binned mean (10 bins)')
        axes2[1].set_xlabel('True margin')
        axes2[1].set_ylabel(f'# of {n_checkpoints} checkpoints covered')
        axes2[1].set_title('Does true margin predict\nhow often a point is covered?')
        axes2[1].legend(fontsize=8)

        fig2.suptitle('Per-test-point coverage persistence across training', fontsize=13)
        fig2.tight_layout(rect=[0, 0, 1, 0.90])
        out2 = 'coverage_persistence.png'
        fig2.savefig(out2, dpi=150)
        if verbose:
            print(f"Saved: {out2}")

    return results


if __name__ == '__main__':
    coverage_over_training()
