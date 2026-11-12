"""
Stability check for coverage_over_training.py: that was run on a single
seed (23). Reruns it across many independent seeds and reports, at every
checkpoint, the distribution (mean/std) of per-class coverage, joint
(example-level) coverage, and interval width -- the same style of check
already done for the clipping rate and the skip-accuracy sanity check.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coverage_over_training import coverage_over_training


def run_multiseed(n_seeds=20, delta=0.5, n_checkpoints=10, n_check=None):
    all_results = []
    for seed in range(n_seeds):
        res = coverage_over_training(delta=delta, seed=seed, n_checkpoints=n_checkpoints,
                                      n_check=n_check, verbose=False, make_plot=False)
        all_results.append(res)
        print(f"seed {seed:>2}: final per-class={res[-1]['per_class_coverage']:.3f}  "
              f"final joint={res[-1]['joint_coverage']:.3f}  final width={res[-1]['mean_width']:.3f}")

    n_cp = len(all_results[0])
    ts = [all_results[0][c]['t'] for c in range(n_cp)]

    per_class_mat = np.array([[r[c]['per_class_coverage'] for r in all_results] for c in range(n_cp)])
    joint_mat = np.array([[r[c]['joint_coverage'] for r in all_results] for c in range(n_cp)])
    width_mat = np.array([[r[c]['mean_width'] for r in all_results] for c in range(n_cp)])

    per_class_mean, per_class_std = per_class_mat.mean(axis=1), per_class_mat.std(axis=1)
    joint_mean, joint_std = joint_mat.mean(axis=1), joint_mat.std(axis=1)
    width_mean, width_std = width_mat.mean(axis=1), width_mat.std(axis=1)

    print(f"\n{'t':>6} {'per-class (mean+-std)':>24} {'joint (mean+-std)':>22} {'width (mean+-std)':>20}")
    for c in range(n_cp):
        print(f"{ts[c]:>6} {per_class_mean[c]:>10.3f} +- {per_class_std[c]:<9.3f} "
              f"{joint_mean[c]:>8.3f} +- {joint_std[c]:<9.3f} "
              f"{width_mean[c]:>8.3f} +- {width_std[c]:<7.3f}")

    final_pc = per_class_mat[-1]
    final_j = joint_mat[-1]
    print(f"\nFinal checkpoint across {n_seeds} seeds:")
    print(f"  per-class coverage: mean={final_pc.mean():.3f}  std={final_pc.std():.3f}  "
          f"min={final_pc.min():.3f}  max={final_pc.max():.3f}")
    print(f"  joint coverage:     mean={final_j.mean():.3f}  std={final_j.std():.3f}  "
          f"min={final_j.min():.3f}  max={final_j.max():.3f}")
    naive_est = final_pc.mean() ** 5
    print(f"  naive independence estimate (mean per-class^K): {naive_est:.3f}  "
          f"vs actual mean joint: {final_j.mean():.3f}")

    # --- Diagrams ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].plot(ts, per_class_mean, 'o-', color='steelblue', lw=2, label='per-class (mean)')
    axes[0].fill_between(ts, per_class_mean - per_class_std, per_class_mean + per_class_std,
                          color='steelblue', alpha=0.2)
    axes[0].plot(ts, joint_mean, 'o-', color='crimson', lw=2, label='joint, example-level (mean)')
    axes[0].fill_between(ts, joint_mean - joint_std, joint_mean + joint_std, color='crimson', alpha=0.2)
    axes[0].axhline(0.95, color='gray', ls='--', lw=1, label='95% target')
    axes[0].set_xlabel('Training round t')
    axes[0].set_ylabel('Coverage')
    axes[0].set_title(f'Coverage over training\n(mean +- 1 std across {n_seeds} seeds)')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts, width_mean, 'o-', color='darkorange', lw=2)
    axes[1].fill_between(ts, width_mean - width_std, width_mean + width_std, color='darkorange', alpha=0.2)
    axes[1].set_xlabel('Training round t')
    axes[1].set_ylabel('Mean interval width')
    axes[1].set_title(f'Interval width over training\n(mean +- 1 std across {n_seeds} seeds)')
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(final_pc, final_j, color='seagreen')
    xs = np.linspace(final_pc.min(), final_pc.max(), 50)
    axes[2].plot(xs, xs ** 5, color='gray', ls='--', lw=1, label='naive independence (x^5)')
    axes[2].set_xlabel('Final per-class coverage')
    axes[2].set_ylabel('Final joint coverage')
    axes[2].set_title('Per-class vs joint coverage,\none point per seed')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Stability of coverage-over-training across {n_seeds} seeds (delta={delta})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = 'coverage_over_training_multiseed.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

    return all_results


if __name__ == '__main__':
    run_multiseed()
