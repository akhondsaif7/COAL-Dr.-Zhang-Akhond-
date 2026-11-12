"""
Variant of experiment_linear_synthetic that REJECTS (filters out) any x
producing a negative raw class probability, instead of clipping+renormalizing
it. See coal docs / conversation: clip+renormalize changes EVERY class's
probability for the ~1-10% of points it fires on (not just the negative
one), making the true label model non-linear on that subset. Rejection
instead only ever keeps x's where the model is genuinely, exactly linear --
this is also closer to the Crammer-Gentile paper's own framing, which states
the model is only "well defined" for x satisfying -alpha <= u_i.x for all i,
rather than prescribing a patch for violations.

This changes the covariate distribution slightly: x is now drawn from
N(0,I) CONDITIONED on producing an entirely valid probability vector (a
truncated Gaussian), not unconstrained N(0,I). Still i.i.d., still a
legitimate (if slightly different) distribution.
"""

import numpy as np


def generate_filtered_dataset(K=5, d=20, N=2000, seed=23, sigma=0.02):
    """
    Same tilt construction as experiment_linear_synthetic, but rejects
    (redraws) any x whose raw probability vector has a negative component,
    instead of clipping it.

    sigma: per-entry std of the tilt vectors v_k (before mean-centering).
    Typical ||v_k|| ~= sqrt(d)*sigma. Default 0.02 matches every prior
    experiment; pass a rescaled sigma to hold ||v_k|| fixed across a d-sweep.

    Returns (X, y, U, true_probs, n_rejected, n_drawn).
    """
    rng = np.random.RandomState(seed)
    v_raw = rng.randn(K, d) * sigma
    v = v_raw - v_raw.mean(axis=0, keepdims=True)
    c = np.full(K, 1.0 / K)
    U = np.hstack([v, c[:, None]])

    def true_probs(x):
        """Always a valid probability vector here -- no clipping needed,
        since only x's that already satisfy this are ever accepted."""
        return U.dot(np.concatenate([x, [1.0]]))

    X, y = [], []
    n_drawn = 0
    n_rejected = 0
    while len(X) < N:
        x = rng.randn(d)
        n_drawn += 1
        p = true_probs(x)
        if np.any(p < 0):
            n_rejected += 1
            continue
        X.append(x)
        y.append(rng.choice(K, p=p))

    return np.array(X), np.array(y), U, true_probs, n_rejected, n_drawn


if __name__ == '__main__':
    X, y, U, true_probs, n_rejected, n_drawn = generate_filtered_dataset()
    print(f"Drew {n_drawn} candidate points to accept {len(X)} "
          f"(rejected {n_rejected}, {100*n_rejected/n_drawn:.2f}%)")
    all_p = np.array([true_probs(x) for x in X])
    print(f"Min probability across accepted set: {all_p.min():.4f} (should be >= 0)")
    print(f"Sums all == 1: {np.allclose(all_p.sum(axis=1), 1.0)}")
