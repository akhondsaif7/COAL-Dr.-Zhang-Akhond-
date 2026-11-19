"""
Empirical timing benchmark for RigorousCOAL (BINSEARCH), at K=10, d=40
(matching Hsu's multi-class MNIST setup), using SMALL T values -- each
query decision can trigger up to 2*K oracle calls (HIGH+LOW per class),
each of which can itself run up to 200 bisection iterations, so this is
expected to be much more expensive per step than CheatCOAL. Starting
small and extrapolating, same approach as the CheatCOAL benchmark.
"""
import time
import numpy as np

from coal_rigorous import RigorousCOAL
from experiment_linear_synthetic_filtered import generate_filtered_dataset

K, d, PREC, DELTA_CONF = 10, 40, 0.01, 0.05
Ts = [50, 100, 200, 400]
HORIZON = 60000   # pass the REAL target horizon T to the eps_m schedule, even
                   # though we only run a prefix of it here -- eps_m depends on
                   # the assumed total horizon, not just steps actually run

print(f"K={K} d={d} prec={PREC} delta(conf)={DELTA_CONF} horizon={HORIZON}")
results = []
for T in Ts:
    X, y, U, true_probs, n_rej, n_drawn = generate_filtered_dataset(K=K, d=d, N=T, seed=1, sigma=0.02)
    X_aug = np.hstack([X, np.ones((len(X), 1))])
    d_feed = d + 1

    coal = RigorousCOAL(K=K, d=d_feed, reg=1.0, burn_in=20)
    t0 = time.perf_counter()
    for i in range(T):
        coal.observe(X_aug[i], y[i], active=True, T=HORIZON, prec=PREC, delta=DELTA_CONF)
    elapsed = time.perf_counter() - t0
    qrate = 100 * coal.n_queries / T
    results.append((T, elapsed, qrate))
    print(f"T={T:>6}: {elapsed:8.3f}s  ({elapsed/T*1000:6.2f} ms/step, query rate {qrate:5.1f}%)")

Ts_arr = np.array([r[0] for r in results], dtype=float)
el_arr = np.array([r[1] for r in results], dtype=float)
logT, logE = np.log(Ts_arr), np.log(el_arr)
p, logA = np.polyfit(logT, logE, 1)
A = np.exp(logA)
print(f"\nFitted scaling: elapsed ~= {A:.3e} * T^{p:.2f}")

for T_target in [1000, 5000, 20000, 60000]:
    est = A * T_target ** p
    unit = "s"
    if est > 3600:
        print(f"Extrapolated T={T_target}: {est:.0f}s ({est/3600:.2f} hr)")
    else:
        print(f"Extrapolated T={T_target}: {est:.1f}s ({est/60:.1f} min)")
