"""
Final-checkpoint look at cheat-mode intervals: 10 jointly-covered vs. 10 not-covered
test points, per-class interval vs. true probability, plus how much delta must grow
to cover the misses and whether interval order matches true-probability order.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset

DELTA, K, d, N, SEED = 0.5, 5, 20, 2000, 23

X, y, U, true_probs, _, _ = generate_filtered_dataset(K=K, d=d, N=N, seed=SEED)
split = int(0.8 * N)
aug = lambda A: np.hstack([A, np.ones((len(A), 1))])
Xtr, ytr, Xte = aug(X[:split]), y[:split], aug(X[split:])
TP = np.array([true_probs(x) for x in X[split:]])

coal = CheatCOAL(K=K, d=d + 1, reg=1.0, delta=DELTA, burn_in=20)
for i in range(len(Xtr)):
    coal.observe(Xtr[i], ytr[i], active=True)
coal._fit_all()

n = len(Xte)
mid = np.zeros((n, K)); q = np.zeros((n, K))
for i in range(n):
    for k in range(K):
        m = coal.models[k]
        mid[i, k] = m.theta.dot(Xte[i])
        q[i, k] = Xte[i].dot(m.Ainv.dot(Xte[i]))   # half-width = sqrt(q * delta)
hw = np.sqrt(q * DELTA)
lcb, ucb = mid - hw, mid + hw
cov_k = (lcb <= TP) & (TP <= ucb)
joint = cov_k.all(axis=1)

# smallest delta covering every class of a point: need |tp-mid| <= sqrt(q*delta)
delta_needed = ((np.abs(TP - mid) / np.sqrt(q)) ** 2).max(axis=1)

tau = np.array([kendalltau(mid[i], TP[i])[0] for i in range(n)])
argmax_ok = mid.argmax(1) == TP.argmax(1)

print(f"queries={coal.n_queries}/{len(Xtr)}  test n={n}")
print(f"per-class coverage={cov_k.mean():.3f}  joint coverage={joint.mean():.3f}")
print(f"delta_needed over uncovered points: median={np.median(delta_needed[~joint]):.2f} "
      f"p75={np.percentile(delta_needed[~joint], 75):.2f} max={delta_needed[~joint].max():.2f}  (current delta={DELTA})")
for mult in [1, 1.5, 2, 3, 4, 6]:
    print(f"  delta x{mult:<3} ({DELTA*mult:.2f}): joint coverage={np.mean(delta_needed <= DELTA*mult):.3f}")
print(f"Kendall tau(mid, true p) per point: mean={tau.mean():.3f} (covered {tau[joint].mean():.3f}, "
      f"uncovered {tau[~joint].mean():.3f})")
print(f"argmax(mid)==argmax(true): all={argmax_ok.mean():.3f} covered={argmax_ok[joint].mean():.3f} "
      f"uncovered={argmax_ok[~joint].mean():.3f}")

rng = np.random.RandomState(5)
pick = lambda mask: rng.choice(np.where(mask)[0], size=min(10, mask.sum()), replace=False)
cov_idx, unc_idx = pick(joint), pick(~joint)

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
fig, axes = plt.subplots(4, 5, figsize=(20, 13))
for row, (idxs, label) in enumerate([(cov_idx, 'COVERED'), (unc_idx, 'NOT covered')]):
    for j, i in enumerate(idxs):
        ax = axes[row * 2 + j // 5, j % 5]
        for k in range(K):
            ok = cov_k[i, k]
            ax.plot([lcb[i, k], ucb[i, k]], [k, k], lw=7, color=colors[k], alpha=0.85 if ok else 0.4,
                    solid_capstyle='butt')
            ax.scatter([mid[i, k]], [k], marker='o', color='white', edgecolor='black', s=22, zorder=4)
            ax.scatter([TP[i, k]], [k], marker='D', color='crimson' if not ok else 'black', s=36, zorder=5)
        ax.set_yticks(range(K)); ax.set_yticklabels([f'c{k}' for k in range(K)], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{label}  pt {i}  d_needed={delta_needed[i]:.2f}  tau={tau[i]:.2f}", fontsize=9,
                     color='seagreen' if label == 'COVERED' else 'crimson')
        ax.grid(True, axis='x', alpha=0.25)
fig.suptitle(f"Final checkpoint (t={len(Xtr)}, delta={DELTA}): interval (bar), midpoint (white dot), true p (diamond; red = missed)\n"
             f"joint coverage {joint.mean():.1%}; delta x{np.ceil(np.percentile(delta_needed,95)/DELTA):.0f} would cover 95% of points",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('final_checkpoint_intervals.png', dpi=130)

fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.5))
ax2[0].hist(delta_needed, bins=40, color='steelblue')
ax2[0].axvline(DELTA, color='crimson', ls='--', label=f'current delta={DELTA}')
ax2[0].set_xlabel('smallest delta covering all K classes'); ax2[0].set_ylabel('# test points'); ax2[0].legend()
ax2[1].hist([tau[joint], tau[~joint]], bins=15, label=['covered', 'not covered'], color=['seagreen', 'crimson'])
ax2[1].set_xlabel('Kendall tau (interval midpoints vs true p)'); ax2[1].legend()
fig2.tight_layout(); fig2.savefig('final_checkpoint_delta_needed.png', dpi=130)
print("Saved final_checkpoint_intervals.png, final_checkpoint_delta_needed.png")
