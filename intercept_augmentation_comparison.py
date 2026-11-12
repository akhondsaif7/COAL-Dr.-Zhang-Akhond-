"""
Diagram: does augmenting x with a constant 1 (so the per-class regressors
can represent the true model's 1/K intercept) actually fix cheat-mode
UCB/LCB calibration, and at what cost (if any) to accuracy or query count?

Runs experiment_linear_synthetic() twice -- once with augment=False (the
original, realizability-broken setup) and once with augment=True (the fix)
-- on the identical underlying data, and plots the four relevant metrics
side by side: coverage, interval width, accuracy, and query count.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiment_linear_synthetic import experiment_linear_synthetic


def compare_augmentation():
    print("\n" + "="*60)
    print("Comparing: no intercept vs intercept-augmented features")
    print("="*60)

    res_no  = experiment_linear_synthetic(augment=False)
    res_yes = experiment_linear_synthetic(augment=True)

    labels = ['no intercept\n(raw x)', 'intercept-augmented\n([x, 1])']
    colors = ['crimson', 'seagreen']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # --- Coverage ---
    cov = [res_no['coverage'], res_yes['coverage']]
    axes[0].bar(labels, cov, color=colors)
    for i, v in enumerate(cov):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Coverage of true f*(x,a)')
    axes[0].set_title('Cheat-mode UCB/LCB coverage')
    axes[0].axhline(0.95, color='gray', ls='--', lw=1, label='95% target')
    axes[0].legend(fontsize=8)

    # --- Width ---
    width = [res_no['width'], res_yes['width']]
    axes[1].bar(labels, width, color=colors)
    for i, v in enumerate(width):
        axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1].set_ylabel('Mean interval width (UCB - LCB)')
    axes[1].set_title('Cheat-mode interval width\n(essentially unchanged)')

    # --- Accuracy ---
    acc = [res_no['acc_cheat'], res_yes['acc_cheat']]
    axes[2].bar(labels, acc, color=colors)
    for i, v in enumerate(acc):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[2].set_ylim(0, 1.0)
    axes[2].set_ylabel('Test accuracy (vs Bayes-optimal)')
    axes[2].set_title('Cheat-mode accuracy\n(essentially unchanged)')

    # --- Query count ---
    T = res_no['T']
    q = [res_no['queries_cheat'], res_yes['queries_cheat']]
    axes[3].bar(labels, q, color=colors)
    for i, v in enumerate(q):
        axes[3].text(i, v + T*0.01, f'{v}/{T}\n({100*v/T:.1f}%)', ha='center', fontweight='bold')
    axes[3].set_ylim(0, T*1.15)
    axes[3].set_ylabel('Training examples queried')
    axes[3].set_title('Cheat-mode query count\n(essentially unchanged)')

    fig.suptitle('Fixing the missing-intercept realizability bug in cheat mode\n'
                 'Same data, same seed -- only whether x is augmented with a constant 1 differs',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = 'intercept_augmentation_comparison.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    print(f"\nCoverage: {cov[0]:.1%} -> {cov[1]:.1%}   "
          f"(width {width[0]:.3f} -> {width[1]:.3f}, "
          f"accuracy {acc[0]:.3f} -> {acc[1]:.3f}, "
          f"queries {q[0]} -> {q[1]})")


if __name__ == '__main__':
    compare_augmentation()
