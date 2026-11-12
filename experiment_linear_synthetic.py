"""
Experiment -- Crammer-Gentile linear synthetic (known, correctly-specified f*)

Crammer & Gentile (ICML 2011), gamma=0 linear multiclass model:
    P(y=a|x) = u_a . [x, 1],   with sum_a u_a = [0,...,0, 1]
Built by centering K random weight vectors (sum_a v_a = 0 exactly) and
appending a fixed 1/K bias term, so class probabilities sum to exactly 1
by linear algebra alone -- no softmax anywhere.

No separability engineering -- x is drawn plain N(0,I), exactly as the
paper's own protocol describes ("Get instance x_t : ||x_t||=1"-style
i.i.d. draws, unnormalized here), with no distance-from-origin
manipulation to manufacture easy/ambiguous points.

Feature augmentation: the true model has a fixed intercept term (1/K) on
top of v_a.x, but IncrementalLinearRegression (coal_common.fast_fit) fits
theta=(reg*I+X'X)^-1 X'y with no bias term -- a linear-through-the-origin
fit that structurally cannot represent that intercept. Every feature vector
fed to COAL is therefore augmented with a constant 1 (x -> [x,1], d -> d+1),
giving the regressor's own weight vector a free entry to learn the intercept.

Trains two models on the same (augmented) data -- Passive and COAL-cheat --
and compares query count and test accuracy.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_passive import PassiveCOAL
from coal_cheat import CheatCOAL
from difficulty import margins


def experiment_linear_synthetic(augment=True):
    print("\n" + "="*60)
    print(f"Experiment: Crammer-Gentile linear synthetic (known f*)  [augment={augment}]")
    print("="*60)

    rng = np.random.RandomState(23)
    K, d, N = 5, 20, 2000

    v_raw = rng.randn(K, d) * 0.02
    v = v_raw - v_raw.mean(axis=0, keepdims=True)   # sum_a v_a = 0 exactly
    c = np.full(K, 1.0 / K)
    U = np.hstack([v, c[:, None]])                  # u_a = [v_a, 1/K]

    def true_probs(x):
        """Exact f*(x,.) -- sums to 1 by construction; clip+renormalise
        only in the rare case a component goes negative."""
        p = U.dot(np.concatenate([x, [1.0]]))
        if np.any(p < 0):
            p = np.clip(p, 0, None)
            p = p / p.sum()
        return p

    # Plain i.i.d. draws -- no near/far distance manipulation
    X = rng.randn(N, d)
    y = np.array([rng.choice(K, p=true_probs(X[i])) for i in range(N)])

    split = int(0.8 * N)
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)

    if augment:
        # Augment every feature vector with a constant 1, so the regressors
        # can represent the true model's intercept.
        X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
        X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])
        d_feed = d + 1
    else:
        X_train_aug, X_test_aug = X_train, X_test
        d_feed = d

    all_p_test = np.array([true_probs(X_test[i]) for i in range(len(X_test))])
    margin_test = margins(all_p_test)
    true_best_test = np.argmax(all_p_test, axis=1)

    print(f"Train: {len(X_train)}  Test: {len(X_test)}  K={K}  d={d} (fed as d={d_feed}"
          f"{', with intercept' if augment else ', no intercept'})")
    print(f"Test-set difficulty (margin = top - runner-up): "
          f"mean={margin_test.mean():.3f}  median={np.median(margin_test):.3f}  "
          f"min={margin_test.min():.3f}  max={margin_test.max():.3f}")

    # --- Train Passive and COAL-cheat on identical (augmented) data ---
    coal_p = PassiveCOAL(K=K, d=d_feed, reg=1.0, delta=0.5, burn_in=20)
    for i in range(len(X_train_aug)):
        coal_p.observe(X_train_aug[i], y_train[i])
    acc_p = np.mean(coal_p.predict_batch(X_test_aug) == true_best_test)

    coal_c = CheatCOAL(K=K, d=d_feed, reg=1.0, delta=0.5, burn_in=20)
    for i in range(len(X_train_aug)):
        coal_c.observe(X_train_aug[i], y_train[i], active=True)
    acc_c = np.mean(coal_c.predict_batch(X_test_aug) == true_best_test)

    print(f"\n{'Method':<12} {'Queries':>15} {'Accuracy':>10}")
    for name, coal, acc in [('Passive', coal_p, acc_p), ('Cheat', coal_c, acc_c)]:
        print(f"{name:<12} {coal.n_queries:>7}/{T:<6} {100*coal.n_queries/T:>5.1f}%   {acc:>8.3f}")

    # --- Coverage check: does cheat mode's UCB/LCB actually capture f*? ---
    # Uses the FULL test set, not a subsample -- cheat mode's get_ucb_lcb is a
    # single closed-form pred_range call per class (no bisection search like
    # rigorous mode), so there's no real compute cost to checking every point.
    coal_c._fit_all()
    n_check = len(X_test)
    cov, widths = [], []
    for i in range(n_check):
        ucb, lcb = coal_c.get_ucb_lcb(X_test_aug[i])
        tp = true_probs(X_test[i])
        cov.append(np.all((lcb <= tp) & (tp <= ucb)))   # per-example: ALL K classes covered simultaneously
        for k in range(K):
            widths.append(ucb[k] - lcb[k])
    print(f"\nCheat-mode coverage (n={n_check}, per-example -- ALL {K} classes covered simultaneously): "
          f"coverage={np.mean(cov):.3f}  width={np.mean(widths):.3f}")

    # --- Results plot: query count, accuracy ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    methods = ['Passive', 'COAL-cheat']
    colors  = ['steelblue', 'darkorange']

    counts = [coal_p.n_queries, coal_c.n_queries]
    axes[0].bar(methods, counts, color=colors)
    for i, cnt in enumerate(counts):
        axes[0].text(i, cnt + T*0.01, f'{cnt}\n({100*cnt/T:.1f}%)', ha='center', fontweight='bold')
    axes[0].set_ylabel('Training examples queried')
    axes[0].set_title(f'Query count (out of {T})')
    axes[0].set_ylim(0, T*1.18)

    accs = [acc_p, acc_c]
    axes[1].bar(methods, accs, color=colors)
    axes[1].axhline(1/K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    for i, a in enumerate(accs):
        axes[1].text(i, a + 0.02, f'{a:.3f}', ha='center', fontweight='bold')
    axes[1].set_ylabel('Test accuracy (argmax matches true best action)')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    tag = 'intercept-augmented' if augment else 'no intercept'
    out = 'linear_synthetic_results.png' if augment else 'linear_synthetic_results_no_augment.png'
    fig.suptitle(f'Crammer-Gentile linear synthetic -- Passive vs cheat ({tag})', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved: {out}")

    return dict(augment=augment, T=T,
                queries_passive=coal_p.n_queries, queries_cheat=coal_c.n_queries,
                acc_passive=acc_p, acc_cheat=acc_c,
                coverage=float(np.mean(cov)), width=float(np.mean(widths)))


if __name__ == '__main__':
    experiment_linear_synthetic()
