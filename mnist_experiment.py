"""
Full-scale MNIST multiclass experiment, matching Hsu's dissertation setup
(Section 6.3.2): K=10 (all digits), T=60000 train / 10000 test, d=40 via
PCA. Compares PassiveCOAL (queries every label, Hsu's passive baseline)
against RigorousCOAL (Algorithm 3 / BINSEARCH -- the actual target of this
experiment, since this is the first time it's been run on real, not
synthetic, data).

CheatCOAL is deliberately NOT included here: its `delta` hyperparameter has
never been tuned/calibrated for K=10 (our BEST_DELTA table only covers
K=2..5), so a result from it here would be an arbitrary, uncalibrated
number rather than a meaningful comparison point.

No "coverage of true f*(x,a)" metric is reported -- unlike our synthetic
Crammer-Gentile data, real MNIST has no known ground-truth generative
probability function, so that metric (central to every synthetic
experiment so far) simply isn't computable here. Only accuracy (vs. the
real, essentially noiseless digit label), query rate, and mean interval
width are reported -- the same quantities Hsu reports (labeled error rate,
labeling rate).

Preprocessing: raw pixels scaled to [0,1], PCA to d=40 fit on the TRAINING
split only (then applied to test, avoiding leakage), followed by per-
component standardization (zero mean, unit variance) -- done so the fixed
reg=1.0 ridge regularization behaves comparably to how it was calibrated
against our O(1)-magnitude synthetic features, rather than being
implicitly near-zero relative to raw/unscaled PCA component magnitudes.
"""
import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from coal_passive import PassiveCOAL
from coal_rigorous import RigorousCOAL

K = 10
D = 40
T_TRAIN = 60000
T_TEST = 10000
PREC = 0.01
DELTA_CONF = 0.05

print("Fetching MNIST (cached after first run)...")
t0 = time.perf_counter()
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
print(f"  done in {time.perf_counter()-t0:.1f}s. Shape: {mnist.data.shape}")

X_all = mnist.data.astype(np.float64) / 255.0
y_all = mnist.target.astype(int)

# Canonical LeCun ordering: first 60000 = train, last 10000 = test (same split Hsu used)
X_train_raw, y_train = X_all[:T_TRAIN], y_all[:T_TRAIN]
X_test_raw,  y_test  = X_all[T_TRAIN:T_TRAIN + T_TEST], y_all[T_TRAIN:T_TRAIN + T_TEST]

print(f"Fitting PCA(d={D}) on training split only...")
pca = PCA(n_components=D, random_state=0)
X_train_pca = pca.fit_transform(X_train_raw)
X_test_pca = pca.transform(X_test_raw)
print(f"  explained variance ratio sum: {pca.explained_variance_ratio_.sum():.3f}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_pca)
X_test = scaler.transform(X_test_pca)

X_train_aug = np.hstack([X_train, np.ones((len(X_train), 1))])
X_test_aug = np.hstack([X_test, np.ones((len(X_test), 1))])
d_feed = D + 1

print(f"\nTrain: {X_train_aug.shape}, Test: {X_test_aug.shape}, K={K}\n")


def run_mode(name, coal, extra_kwargs):
    t0 = time.perf_counter()
    for i in range(T_TRAIN):
        coal.observe(X_train_aug[i], y_train[i], active=True, **extra_kwargs)
        if (i + 1) % 10000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{name}] {i+1}/{T_TRAIN} steps, {elapsed:.1f}s elapsed, "
                  f"{coal.n_queries} queried so far ({100*coal.n_queries/(i+1):.1f}%)")

    coal._fit_all()
    preds = coal.predict_batch(X_test_aug)
    acc = np.mean(preds == y_test)

    widths = []
    if hasattr(coal, 'get_ucb_lcb'):
        for i in range(0, T_TEST, 20):   # subsample test set for width (expensive to do all 10000)
            ucb, lcb = coal.get_ucb_lcb(X_test_aug[i])
            widths.extend((ucb - lcb).tolist())
    elif name == 'Rigorous':
        for i in range(0, T_TEST, 100):  # BINSEARCH is pricier per call -- sparser subsample
            ucb, lcb = coal.get_ucb_lcb_algorithm3(X_test_aug[i], T=T_TRAIN, prec=PREC, delta=DELTA_CONF)
            widths.extend((ucb - lcb).tolist())
    mean_width = np.mean(widths) if widths else float('nan')

    total_elapsed = time.perf_counter() - t0
    qrate = 100 * coal.n_queries / T_TRAIN
    print(f"\n[{name}] FINAL: elapsed={total_elapsed:.1f}s  queries={coal.n_queries}/{T_TRAIN} "
          f"({qrate:.1f}%)  test_acc={acc:.4f}  mean_width~={mean_width:.4f}\n")
    return dict(name=name, elapsed=total_elapsed, queries=coal.n_queries, qrate=qrate,
                acc=acc, mean_width=mean_width)


results = []

print("=" * 60)
print("Running PassiveCOAL (baseline: queries every label)...")
print("=" * 60)
passive = PassiveCOAL(K=K, d=d_feed, reg=1.0, burn_in=20)
results.append(run_mode('Passive', passive, {}))

print("=" * 60)
print("Running RigorousCOAL (Algorithm 3 / BINSEARCH)...")
print("=" * 60)
rigorous = RigorousCOAL(K=K, d=d_feed, reg=1.0, burn_in=20)
results.append(run_mode('Rigorous', rigorous, dict(T=T_TRAIN, prec=PREC, delta=DELTA_CONF)))

print("=" * 60)
print("SUMMARY")
print("=" * 60)
for r in results:
    print(f"{r['name']:>10}: elapsed={r['elapsed']:7.1f}s  queries={r['queries']:>6}/{T_TRAIN} "
          f"({r['qrate']:5.1f}%)  test_acc={r['acc']:.4f}  mean_width~={r['mean_width']:.4f}")
