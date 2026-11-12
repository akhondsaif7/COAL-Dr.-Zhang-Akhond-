"""
Cheat mode -- closed-form confidence interval (RegCB's self.cheat=True path).

Exact for the linear class: mid = theta.x, half-width = sqrt(x' Ainv x * delta).
Mathematically equivalent to LinUCB/OFUL for this model class (Foster et al.
2018, footnote 6) -- this is what oracle_cb's RegressorUCB uses by default.

`delta` here is a fixed, free hyperparameter -- it never grows with T or K,
unlike rigorous mode's Theorem-2-derived radius (coal_rigorous.py). That's
what makes it cheap and fast-shrinking, but not a provably-calibrated
confidence interval.
"""

import numpy as np

from coal_common import COALBase


class CheatCOAL(COALBase):

    def _get_ucb_lcb(self, x, **kwargs):
        ucb = np.empty(self.K)
        lcb = np.empty(self.K)
        for k in range(self.K):
            if self.models[k].is_fit:
                lcb[k], ucb[k] = self.models[k].pred_range(x, self.delta)
            else:
                lcb[k], ucb[k] = 0.0, 1.0   # uninformative before any data
        return ucb, lcb

    def get_ucb_lcb(self, x):
        """Convenience alias for external coverage-checking code."""
        return self._get_ucb_lcb(x)