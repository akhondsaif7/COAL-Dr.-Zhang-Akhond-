"""
Illustrates what switching from clip+renormalize to reject+resample
actually changes: the difficulty distribution of the resulting dataset, and
the downstream accuracy/coverage effects, holding everything else (seed,
K, d, N, delta, intercept augmentation) fixed.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coal_passive import PassiveCOAL
from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset
from difficulty import margins


def generate_clipped_dataset(K=5, d=20, N=2000, seed=23):
    rng = np.random.RandomState(seed)
    v_raw = rng.randn(K, d) * 0.02
    v = v_raw - v_raw.mean(axis=0, keepdims=True)
    c = np.full(K, 1.0 / K)
    U = np.hstack([v, c[:, None]])

    def true_probs(x):
        p = U.dot(np.concatenate([x, [1.0]]))
        if np.any(p < 0):
            p = np.clip(p, 0, None)
            p = p / p.sum()
        return p

    X = rng.randn(N, d)
    y = np.array([rng.choice(K, p=true_probs(X[i])) for i in range(N)])
    return X, y, true_probs


def run_one(X, y, true_probs, K, d, delta=0.5, seed_split=23):
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]
    T = len(X_train)

    X_train_aug = np.hstack([X_train, np.ones((T, 1))])
    X_test_aug  = np.hstack([X_test,  np.ones((len(X_test), 1))])

    all_p_test = np.array([true_probs(x) for x in X_test])
    margin_test = margins(all_p_test)
    true_best_test = np.argmax(all_p_test, axis=1)

    # Bayes accuracy: the ceiling from the data-generating process itself --
    # how often the Bayes-optimal guess (true_best_test) matches the actually
    # REALIZED (noisy) label y_test. No model or training involved at all;
    # this is a property of the problem, not of any regressor or delta.
    bayes_acc = np.mean(true_best_test == y_test)

    coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=delta, burn_in=20)
    for i in range(T):
        coal.observe(X_train_aug[i], y_train[i], active=True)
    acc = np.mean(coal.predict_batch(X_test_aug) == true_best_test)
    queries = coal.n_queries

    coal._fit_all()
    n_check = len(X_test)   # full test set -- cheap for cheat mode (no bisection search)
    cov, widths = [], []
    for i in range(n_check):
        ucb, lcb = coal.get_ucb_lcb(X_test_aug[i])
        tp = true_probs(X_test[i])
        cov.append(np.all((lcb <= tp) & (tp <= ucb)))   # per-example: ALL K classes covered simultaneously
        for k in range(K):
            widths.append(ucb[k] - lcb[k])

    return dict(margin_test=margin_test, acc=acc, bayes_acc=bayes_acc, queries=queries, T=T,
                coverage=np.mean(cov), width=np.mean(widths))


def illustrate_filtering(K=5, d=20, N=2000, seed=23, delta=0.5):
    X_clip, y_clip, tp_clip = generate_clipped_dataset(K, d, N, seed)
    X_filt, y_filt, U_filt, tp_filt, n_rejected, n_drawn = generate_filtered_dataset(K, d, N, seed)

    res_clip = run_one(X_clip, y_clip, tp_clip, K, d, delta=delta)
    res_filt = run_one(X_filt, y_filt, tp_filt, K, d, delta=delta)

    print(f"Rejected {n_rejected}/{n_drawn} ({100*n_rejected/n_drawn:.2f}%) draws to build the filtered set")
    print(f"{'':12} {'queries':>12} {'accuracy':>10} {'coverage':>10} {'width':>8}")
    for name, r in [('clipped', res_clip), ('filtered', res_filt)]:
        print(f"{name:12} {r['queries']:>5}/{r['T']:<5} {r['acc']:>10.3f} {r['coverage']:>10.3f} {r['width']:>8.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # --- Difficulty distribution: margin (top - runner-up) on test set, clipped vs filtered ---
    axes[0].hist(res_clip['margin_test'], bins=25, alpha=0.6, color='crimson', label='clipped')
    axes[0].hist(res_filt['margin_test'], bins=25, alpha=0.6, color='seagreen', label='filtered')
    axes[0].set_xlabel("Test point's margin: max_i p(y=i|x) - 2nd max_i p(y=i|x)")
    axes[0].set_ylabel('Count')
    axes[0].set_title('Difficulty distribution (margin):\nclipped vs filtered')
    axes[0].legend()
    print(f"Mean margin -- clipped: {res_clip['margin_test'].mean():.4f}   "
          f"filtered: {res_filt['margin_test'].mean():.4f}")

    # --- Accuracy comparison ---
    labels = ['clipped', 'filtered']
    colors = ['crimson', 'seagreen']
    accs = [res_clip['acc'], res_filt['acc']]
    axes[1].bar(labels, accs, color=colors)
    for i, v in enumerate(accs):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1].axhline(1/K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel('Test accuracy (vs Bayes-optimal)')
    axes[1].set_title('Accuracy: clipped vs filtered\n(same delta, same seed)')
    axes[1].legend()

    # --- Coverage comparison ---
    covs = [res_clip['coverage'], res_filt['coverage']]
    axes[2].bar(labels, covs, color=colors)
    for i, v in enumerate(covs):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[2].axhline(0.95, color='gray', ls='--', lw=1, label='95% target')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel('Coverage of true f*(x,a)')
    axes[2].set_title('Coverage: clipped vs filtered\n(same delta, same seed)')
    axes[2].legend()

    # --- Bayes accuracy comparison: is the PROBLEM itself harder after filtering? ---
    bayes_accs = [res_clip['bayes_acc'], res_filt['bayes_acc']]
    axes[3].bar(labels, bayes_accs, color=colors)
    for i, v in enumerate(bayes_accs):
        axes[3].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[3].axhline(1/K, color='gray', ls='--', lw=1, label=f'Random ({1/K:.2f})')
    axes[3].set_ylim(0, 1.0)
    axes[3].set_ylabel('Bayes accuracy (true-best vs REALIZED noisy label)')
    axes[3].set_title('Bayes accuracy: clipped vs filtered\n(no model involved -- a property of the problem)')
    axes[3].legend()
    print(f"Bayes accuracy -- clipped: {res_clip['bayes_acc']:.3f}   filtered: {res_filt['bayes_acc']:.3f}")

    fig.suptitle(f'Clip+renormalize vs reject+resample (delta={delta}, intercept-augmented, seed={seed})', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = 'filtering_illustration.png'
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    illustrate_filtering()
