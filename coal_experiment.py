"""
COAL (Cost-Overlapped Active Learning) experiment.

Compares:
  - Passive learning: query every example, train on all data
  - COAL:            query only when the regressor's UCB/LCB bands overlap
                     (i.e., there is genuine uncertainty about the best action)

Accuracy metric: fraction of test queries where argmax predicted reward = argmax true reward.
Plot: x = cumulative label queries, y = test accuracy.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

import settings
import Incremental

np.random.seed(0)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

K  = 8   # documents per query used as actions
D  = 47  # feature dimension

def load_npz(path, K):
    """Load and filter queries that have >= K documents."""
    npz  = np.load(path)
    docs = npz['docsPerQuery']
    mask = docs >= K
    feat = npz['features'][mask, :K, :].astype(np.float64)
    rew  = npz['relevances'][mask, :K].astype(np.float64)
    return feat, rew   # (N, K, D), (N, K)

feat_train, rew_train = load_npz(settings.BASE_DIR + "data/MQ2008_train.npz", K)
feat_test,  rew_test  = load_npz(settings.BASE_DIR + "data/MQ2008_val.npz",   K)

N_train = len(feat_train)
N_test  = len(feat_test)
print(f"Train: {N_train} queries | Test: {N_test} queries | K={K} | d={D}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_accuracy(model):
    """Fraction of test queries where predicted best action = true best action."""
    if not model.is_fit:
        return 0.0
    n_correct = 0
    for i in range(N_test):
        preds = model.predict(feat_test[i])   # shape (K,)
        if np.argmax(preds) == np.argmax(rew_test[i]):
            n_correct += 1
    return n_correct / N_test


def get_ucb_lcb(model, feat_i, delta):
    """
    Compute per-action UCB and LCB for a single context.

    feat_i : (K, D) feature matrix for this context
    delta  : confidence width parameter passed to pred_range

    Returns ucb, lcb  — both shape (K,)
    """
    ucb = np.empty(K)
    lcb = np.empty(K)
    for k in range(K):
        lcb[k], ucb[k] = model.pred_range(feat_i[k], delta)
    return ucb, lcb


def coal_should_query(ucb, lcb):
    """
    COAL query criterion (Section 4 of Krishnamurthy et al. 2019).

    The 'possibly-optimal' set contains every action k whose upper confidence
    bound is at least as large as the best lower confidence bound.  If this
    set has more than one member the algorithm is uncertain about the best
    action and should request the label.

    Formally:  query  iff  |{ k : UCB_k >= max_j LCB_j }| > 1
    """
    best_lcb          = np.max(lcb)
    n_possibly_optimal = np.sum(ucb >= best_lcb)
    return n_possibly_optimal > 1


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

BURN_IN    = 20    # always query for first BURN_IN examples (cold start)
REFIT_FREQ = 10    # refit regressor every N new queries (batch update)
EVAL_FREQ  = 10    # record test accuracy every N training steps
DELTA      = 0.5   # confidence parameter for pred_range

def run(active: bool):
    """
    Train a single linear regressor using either COAL (active=True)
    or passive learning (active=False).

    Returns
    -------
    query_counts : list[int]   cumulative queries at each evaluation checkpoint
    accuracies   : list[float] test accuracy at each checkpoint
    """
    model   = Incremental.IncrementalLinearRegression(reg=1.0)
    X_buf   = []
    y_buf   = []
    n_queries    = 0
    query_counts = []
    accuracies   = []

    for i in range(N_train):
        feat_i = feat_train[i]   # (K, D)
        rew_i  = rew_train[i]    # (K,)

        # --- decide whether to query ---
        if i < BURN_IN or not model.is_fit:
            do_query = True
        elif active:
            ucb, lcb = get_ucb_lcb(model, feat_i, DELTA)
            do_query = coal_should_query(ucb, lcb)
        else:
            do_query = True   # passive: always query

        if do_query:
            # Observe full reward vector for all K actions (full-information query)
            for k in range(K):
                X_buf.append(feat_i[k])
                y_buf.append(rew_i[k])
            n_queries += 1

            # Periodic batch refit
            if n_queries % REFIT_FREQ == 0:
                model.fit(np.array(X_buf), np.array(y_buf))

        # --- checkpoint ---
        if i % EVAL_FREQ == 0:
            query_counts.append(n_queries)
            accuracies.append(test_accuracy(model))

    label = "COAL" if active else "Passive"
    print(f"[{label}] queries used: {n_queries}/{N_train}  "
          f"({100*n_queries/N_train:.1f}%)  "
          f"final test acc: {accuracies[-1]:.3f}")
    return query_counts, accuracies


# ---------------------------------------------------------------------------
# Run and plot
# ---------------------------------------------------------------------------

print("\nRunning passive learning ...")
passive_q, passive_acc = run(active=False)

print("Running COAL (active learning) ...")
coal_q, coal_acc = run(active=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(passive_q, passive_acc, color='steelblue',  lw=2, label='Passive (always query)')
ax.plot(coal_q,    coal_acc,    color='darkorange', lw=2, label='COAL (active)')

ax.set_xlabel('Cumulative label queries', fontsize=13)
ax.set_ylabel('Test accuracy', fontsize=13)
ax.set_title('COAL vs Passive Learning\n(synthetic MQ2008, linear regressor)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = settings.BASE_DIR + 'coal_vs_passive.png'
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")
