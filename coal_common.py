"""
Shared COAL (Cost-Overlapped Active Learning) infrastructure for multiclass
classification.

Key difference from the ranking experiment:
  - All actions share the same context x  (not action-specific features)
  - We maintain K separate linear regressors, one per class
  - Regressor k predicts  f(x, k) ≈ P(y = k | x)
  - Querying one example reveals I(y==k) for ALL k simultaneously

This file holds only what's common to every query-decision strategy: the
per-class regressors, data buffers, fitting, and prediction. Each strategy
(passive / cheat / rigorous) lives in its own file and subclasses COALBase,
implementing just its own `_get_ucb_lcb`.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.linalg as la

import Incremental


def fast_fit(model, X, y):
    """
    Fit an IncrementalLinearRegression without forming the O(n^2) diagonal
    weight matrix.  All weights are implicitly 1.

    The original fit() does  W = np.diag(weights); A += X^T W X
    which is O(n^2 d).  X^T X is O(n d^2) and is equivalent when
    weights are uniform.
    """
    n, d = X.shape
    model.n      = n
    model.d      = d
    model.X      = X
    model.y      = y
    model.weights = np.ones(n)
    model.y_norm  = np.linalg.norm(y)

    model.A    = np.eye(d) * model.reg + X.T @ X   # O(n d^2)
    model.Ainv = la.pinv(model.A)
    model.b    = X.T @ y
    model.theta = model.Ainv @ model.b
    model.mse   = float(np.sum((X @ model.theta - y) ** 2))
    model.is_fit = True


class COALBase:
    """
    Shared COAL engine for multiclass classification.

    Parameters
    ----------
    K       : number of classes
    d       : feature dimension
    reg     : L2 regularisation for each per-class regressor
    delta   : confidence-interval width (meaning depends on subclass)
    burn_in : always query for the first `burn_in` examples

    Subclasses implement `_get_ucb_lcb(x, **kwargs)` to define how a query
    decision is made when `active=True`; everything else (model management,
    buffers, fitting, prediction) is shared.
    """

    def __init__(self, K, d, reg=1.0, delta=0.5, burn_in=20):
        self.K       = K
        self.d       = d
        self.reg     = reg
        self.delta   = delta
        self.burn_in = burn_in
        self._reset()

    def _reset(self):
        # One regressor per class
        self.models = [Incremental.IncrementalLinearRegression(reg=self.reg)
                       for _ in range(self.K)]
        # Separate data buffers per class
        self.X_bufs = [[] for _ in range(self.K)]
        self.y_bufs = [[] for _ in range(self.K)]
        self.n_queries  = 0
        self.step       = 0

    def _fit_all(self):
        for k in range(self.K):
            if len(self.X_bufs[k]) > 0:
                fast_fit(self.models[k],
                         np.array(self.X_bufs[k]),
                         np.array(self.y_bufs[k]))

    def _get_ucb_lcb(self, x, **kwargs):
        """Subclasses must override: return (ucb, lcb), each shape (K,)."""
        raise NotImplementedError("Subclasses must implement _get_ucb_lcb")

    def _should_query(self, ucb, lcb):
        """Query iff the possibly-optimal set has more than one member."""
        best_lcb = np.max(lcb)
        return np.sum(ucb >= best_lcb) > 1

    def predict(self, x):
        """Return predicted class (argmax of per-class point estimates)."""
        scores = np.array([
            self.models[k].predict(x) if self.models[k].is_fit else 0.0
            for k in range(self.K)
        ])
        return int(np.argmax(scores))

    def predict_batch(self, X):
        """Vectorised prediction for a matrix X of shape (N, d). Returns (N,) class indices."""
        N = len(X)
        scores = np.zeros((N, self.K))
        for k in range(self.K):
            if self.models[k].is_fit:
                scores[:, k] = self.models[k].predict(X)   # model.predict handles 2-D input
        return np.argmax(scores, axis=1)

    def observe(self, x, y_true, active=True, **ucb_kwargs):
        """
        Present one example.

        Parameters
        ----------
        x      : feature vector  (d,)
        y_true : true class index
        active : if False, always query (passive baseline)
        ucb_kwargs : forwarded to the subclass's _get_ucb_lcb (e.g. rigorous
                     mode needs T/prec/delta for the eps_m schedule)

        Returns
        -------
        queried : bool
        """
        self.step += 1
        all_fit = all(m.is_fit for m in self.models)

        if self.step <= self.burn_in or not all_fit:
            do_query = True
        elif active:
            ucb, lcb = self._get_ucb_lcb(x, **ucb_kwargs)
            do_query = self._should_query(ucb, lcb)
        else:
            do_query = True

        if do_query:
            for k in range(self.K):
                self.X_bufs[k].append(x)
                self.y_bufs[k].append(1.0 if y_true == k else 0.0)
            self.n_queries += 1

            # Refit every 10 queries
            if self.n_queries % 10 == 0:
                self._fit_all()

        return do_query

    def observe_reward_vector(self, x, r_vec, active=True, **ucb_kwargs):
        """
        Like observe(), but for independent per-action rewards r_vec (semibandit-
        style: r(a) in {0,1} for each action a, drawn independently) instead of a
        single one-hot true class. Used for reward-vector synthetic datasets,
        where E[r(a)|x] = (W*x+b)[a] per action -- there is no single "true label"
        and no requirement that rewards sum to 1 across actions.
        """
        self.step += 1
        all_fit = all(m.is_fit for m in self.models)

        if self.step <= self.burn_in or not all_fit:
            do_query = True
        elif active:
            ucb, lcb = self._get_ucb_lcb(x, **ucb_kwargs)
            do_query = self._should_query(ucb, lcb)
        else:
            do_query = True

        if do_query:
            for k in range(self.K):
                self.X_bufs[k].append(x)
                self.y_bufs[k].append(float(r_vec[k]))
            self.n_queries += 1
            if self.n_queries % 10 == 0:
                self._fit_all()

        return do_query
