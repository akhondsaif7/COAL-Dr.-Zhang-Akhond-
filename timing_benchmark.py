"""
Empirical timing benchmark: how long does a single CheatCOAL training pass
take at increasing T, holding d=40, K=10 fixed (matching Hsu's multi-class
MNIST setup: d=40 after PCA, K=10 digits)?

_fit_all() does a FULL batch refit (O(n*d^2)) every 10 queries, not an
incremental update -- so if querying stays near 100% (as earlier d-sweeps
showed happens once d/T is unfavorable), total cost is a sum of O(q*d^2)
refits at q=10,20,...,T, which is O(T^2 * d^2) overall: quadratic in T.
Measuring directly at a few T values (rather than assuming linear) to fit
the actual scaling and extrapolate to Hsu's T=60000.
"""
import time
import numpy as np

from coal_cheat import CheatCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset

K, d, DELTA = 10, 40, 1.0
Ts = [500, 1000, 2000, 4000]

print(f"K={K} d={d} delta={DELTA}")
results = []
for T in Ts:
    X, y, U, true_probs, n_rej, n_drawn = generate_filtered_dataset(K=K, d=d, N=T, seed=1, sigma=0.02)
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    d_feed = d + 1

    coal = CheatCOAL(K=K, d=d_feed, reg=1.0, delta=DELTA, burn_in=20)
    t0 = time.perf_counter()
    for i in range(T):
        coal.observe(X_aug[i], y[i], active=True)
    elapsed = time.perf_counter() - t0
    qrate = 100 * coal.n_queries / T
    results.append((T, elapsed, qrate))
    print(f"T={T:>6}: {elapsed:8.3f}s  (query rate {qrate:5.1f}%)")

# Fit elapsed ~ a * T^p via log-log regression on the last few points
Ts_arr = np.array([r[0] for r in results], dtype=float)
el_arr = np.array([r[1] for r in results], dtype=float)
logT, logE = np.log(Ts_arr), np.log(el_arr)
p, logA = np.polyfit(logT, logE, 1)
A = np.exp(logA)
print(f"\nFitted scaling: elapsed ~= {A:.3e} * T^{p:.2f}")

for T_target in [10000, 20000, 60000]:
    est = A * T_target ** p
    print(f"Extrapolated T={T_target}: {est:.1f}s ({est/60:.1f} min, {est/3600:.2f} hr)")
