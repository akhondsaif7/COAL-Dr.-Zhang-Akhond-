"""
The ground-truth model P(y=a|x) = v_a.x + 1/K can occasionally produce a
negative "probability" for some class a (when the random tilt v_a.x swings
below the -1/K floor), which the generator patches by clipping to 0 and
renormalising. This script measures how often that patch actually fires,
across many independent re-generations of the dataset (different seeds),
since a single run's rate can be misleading (seeds vary from <1% to ~10%).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def analyze_clipping_rates(K=5, d=20, N=2000, n_runs=20, out_path='clipping_rates_20runs.png'):
    """
    Regenerate the linear synthetic dataset n_runs times (seeds 0..n_runs-1),
    and record, per run:
      - example-level rate: % of the N points with >=1 negative class component
      - component-level rate: % of the N*K individual class-probability values
        that came out negative
    Saves a two-panel bar chart and returns (clip_rates, component_clip_rates).
    """
    clip_rates = []
    component_clip_rates = []

    for seed in range(n_runs):
        rng = np.random.RandomState(seed)
        v_raw = rng.randn(K, d) * 0.02
        v = v_raw - v_raw.mean(axis=0, keepdims=True)
        c = np.full(K, 1.0 / K)
        U = np.hstack([v, c[:, None]])

        def raw_p(x):
            return U.dot(np.concatenate([x, [1.0]]))

        X = rng.randn(N, d)
        raw = np.array([raw_p(X[i]) for i in range(N)])
        any_neg = (raw < 0).any(axis=1)
        clip_rates.append(100 * any_neg.mean())
        component_clip_rates.append(100 * (raw < 0).sum() / (N * K))

    clip_rates = np.array(clip_rates)
    component_clip_rates = np.array(component_clip_rates)

    print("Per-run example-level clipping rate (%):")
    for seed, r in enumerate(clip_rates):
        print(f"  seed {seed:2d}: {r:.2f}%")
    print(f"\nMean: {clip_rates.mean():.2f}%  Std: {clip_rates.std():.2f}%  "
          f"Min: {clip_rates.min():.2f}%  Max: {clip_rates.max():.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(n_runs), clip_rates, color='steelblue')
    axes[0].axhline(clip_rates.mean(), color='crimson', ls='--', lw=1.5,
                     label=f'mean = {clip_rates.mean():.2f}%')
    axes[0].set_xlabel('Run (seed)')
    axes[0].set_ylabel('Examples with >=1 negative component (%)')
    axes[0].set_title(f'Example-level clipping rate across {n_runs} runs\n(N={N} points/run, K={K}, d={d})')
    axes[0].set_xticks(range(n_runs))
    axes[0].legend()
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(range(n_runs), component_clip_rates, color='darkorange')
    axes[1].axhline(component_clip_rates.mean(), color='crimson', ls='--', lw=1.5,
                     label=f'mean = {component_clip_rates.mean():.2f}%')
    axes[1].set_xlabel('Run (seed)')
    axes[1].set_ylabel('Negative components (%) out of N*K total')
    axes[1].set_title(f'Component-level clipping rate across {n_runs} runs\n({N}x{K}={N*K} probability values/run)')
    axes[1].set_xticks(range(n_runs))
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle('Clipping rate variability across 20 independent data-generation runs', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    return clip_rates, component_clip_rates


if __name__ == '__main__':
    analyze_clipping_rates()