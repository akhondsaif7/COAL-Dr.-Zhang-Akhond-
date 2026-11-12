"""
Difficulty measure for a point x given its true class-probability vector p.

max_i p(y=i|x) alone doesn't say how hard x actually is to classify: a point
with max_p=0.3 and second-place at 0.05 is easy (big gap), while max_p=0.3
with second-place at 0.29 is maximally ambiguous (tiny gap) -- max_p can't
tell these apart. The quantity that actually governs whether the correct
argmax can be recovered from finite noisy data -- and whether COAL's
possibly-optimal set (UCB_k >= best LCB) collapses to one class -- is the
margin between the top two classes.
"""

import numpy as np


def margin(p):
    """Top-class probability minus runner-up, for a single probability vector p (shape (K,))."""
    s = np.sort(p)
    return s[-1] - s[-2]


def margins(all_p):
    """Vectorised margin() for a matrix of probability vectors, shape (N, K) -> (N,)."""
    s = np.sort(all_p, axis=1)
    return s[:, -1] - s[:, -2]
