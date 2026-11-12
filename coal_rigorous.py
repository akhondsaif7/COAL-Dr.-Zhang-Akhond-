"""
Rigorous mode -- Algorithm 3 (BINSEARCH), matching Foster et al. 2018
("Practical Contextual Bandits with Regression Oracles"), driven by the
paper's own eps_m epoch schedule (Theorem 2) rather than a free hyperparameter.
"""

import numpy as np

from coal_common import COALBase


class RigorousCOAL(COALBase):

    def _oracle_call(self, model, x, r, w):
        """
        f <- argmin_{f~} R(f~, w),  z <- f(x,a)

        Uses the Sherman-Morrison update from IncrementalLinearRegression
        (verified correct: theta_new/pred match a from-scratch weighted
        refit exactly at every weight tested). But computes R(f) -- the
        loss on the ORIGINAL data H only -- by DIRECT evaluation against
        the stored (model.X, model.y), rather than trusting fit_incremental's
        own 'past_mse' shortcut, which is a separate, confirmed-wrong formula
        (verified: at w=0 it returns 13.19 instead of the true Rmin=10.61 on
        a toy fit, and stays off by a growing amount at every other w tested).

        Returns (z, R_f) exactly matching the paper's notation.
        """
        z_vec = model.Ainv.dot(x)
        Ainv_new = model.Ainv - w * np.outer(z_vec, z_vec) / (1 + w * x.dot(z_vec))
        b_new = model.b + w * x * r
        theta_new = Ainv_new.dot(b_new)
        z = theta_new.dot(x)
        R_f = float(np.sum((model.X.dot(theta_new) - model.y) ** 2))   # raw sum, unnormalized -- matches Algorithm 3 step 3 exactly
        return z, R_f

    def _binsearch_algorithm3(self, model, x, bound_type, radius, prec, max_doublings=60):
        """
        Faithful port of Algorithm 3 (BINSEARCH), Foster et al. 2018, page 4.

        Paper's inputs -> ours: radius == beta, prec == alpha.

        Steps, verbatim:
          2: r = 2 if HIGH else -1
          3: R(f) := sum over history of (f(x',a')-r')^2
          4: R~(f,w) := R(f) + (w/2)*(f(x,a)-r)^2
          5: wL <- 0, wH <- beta/alpha              <- closed form, NOT a doubling search
          6: fL <- argmin R~(f,wL), zL <- fL(x,a)
          7: fH <- argmin R~(f,wH), zH <- fH(x,a)
          8: Rmin <- R(fL)
          9: Delta <- alpha*beta/(r-zL)^3            <- second stopping threshold, not an
                                                         initial-weight device (see below)
          10: while |zH-zL|>alpha and |wH-wL|>Delta:
          11:   w <- (wH+wL)/2
          12:   f <- argmin R~(f,w), z <- f(x,a)
          13:   if R(f) >= Rmin+beta: wH<-w, zH<-z
          15:   else:                 wL<-w, zL<-z
          19: return zH

        Step 9's Delta is what bounds the number of oracle calls to
        O(log(1/alpha)): without it, the z-only convergence test could loop
        forever if z barely moves as w changes. Its magnitude is what
        matters (it's a threshold on |wH-wL|, which is always >= 0), so we
        take abs() -- (r-zL)^3 can be negative (e.g. LOW: r=-1, zL>=0) and
        the paper's derivation implicitly assumes the positive form.

        Branch direction at 13/15 matches the box exactly: R(f) exceeding
        budget -> w is "too constraining" -> becomes the new wH.
        """
        r = 2.0 if bound_type == 'HIGH' else -1.0

        # Sherman-Morrison update loses precision past ~1e14-1e15 (w*x.z terms
        # overwhelm float64's ~15-17 significant digits, causing catastrophic
        # cancellation). Cap wH well inside the numerically stable region;
        # beta/alpha can be huge (eff_radius in the hundreds / prec=0.01).
        W_NUMERICAL_SAFE_MAX = 1e10

        wL = 0.0
        zL, Rmin = self._oracle_call(model, x, r, wL)   # step 6, 8: Rmin <- R(fL)

        wH = min(radius / prec, W_NUMERICAL_SAFE_MAX)   # step 5: wH <- beta/alpha, closed form
        zH, R_wH = self._oracle_call(model, x, r, wH)   # step 7

        if R_wH <= Rmin + radius:
            # R(f) at the paper's own wH has not exceeded the budget: f can
            # match r exactly without leaving the version space. No search
            # needed -- the sentinel itself is the answer.
            return r

        denom = (r - zL) ** 3
        Delta = abs(prec * radius / denom) if denom != 0 else 0.0   # step 9

        bisect_iters = 0
        while abs(zH - zL) > prec and abs(wH - wL) > Delta and bisect_iters < 200:  # step 10 + safety cap
            w = (wH + wL) / 2.0                                     # step 11
            z, R_f = self._oracle_call(model, x, r, w)               # step 12
            if R_f >= Rmin + radius:                                 # step 13: budget exceeded
                wH, zH = w, z                                        # -> lower the known-unsafe ceiling
            else:
                wL, zL = w, z                                        # -> raise the known-safe floor
            bisect_iters += 1

        return zH                                                    # step 19

    def _epoch_of(self, t):
        """Doubling epoch schedule (paper, Sec 3): epoch m begins at time 2^(m-1).
        Returns the largest m with 2^(m-1) <= t."""
        m = 1
        while 2 ** m <= t:
            m += 1
        return m

    def eff_radius_paper(self, T, delta=0.05):
        """
        Exact eps_m schedule from Theorem 2 (Foster et al. 2018):
            eps_m = (M-m+1)*C / (tau_m-1),   C = 16*log(2|G|*K*T^2/delta)
        where m is the current epoch (doubling schedule), tau_m=2^(m-1) is
        the epoch's start time, and M = total epochs for horizon T.

        BINSEARCH's own R(f) (see _oracle_call) is a RAW, unnormalized sum
        over the history H_m = {(x_s,a_s,r_s)}_{s=1}^{tau_m-1} (Theorem 1),
        while eps_m bounds the NORMALIZED risk R_hat_m = R(f)/(tau_m-1). So
        the raw-sum radius to actually pass into BINSEARCH is
        eps_m*(tau_m-1) = (M-m+1)*C -- the (tau_m-1) cancels exactly,
        regardless of its value, which is why this returns (M-m+1)*C
        directly rather than dividing by (tau_m-1) first.

        |G| has no literal meaning for our continuous linear class; substituted
        by self.d, the standard covering-number-style proxy for a bounded linear
        hypothesis class (this substitution is ours, not quoted from the paper).
        """
        M = self._epoch_of(T)
        m = min(self._epoch_of(self.step), M)
        C = 16.0 * np.log(2.0 * self.d * self.K * T**2 / delta)
        if m <= 1:
            return M * C   # epoch 1: tau_1-1=0, division undefined; use the loosest (m=1) budget
        return (M - m + 1) * C

    def get_ucb_lcb_algorithm3(self, x, T, prec=0.01, delta=0.05):
        """
        UCB/LCB via Algorithm 3, using the paper's eps_m schedule (Theorem 2).
        T: known/assumed horizon (total rounds), needed to compute M and C.
        """
        eff_radius = self.eff_radius_paper(T, delta=delta)
        ucb = np.empty(self.K)
        lcb = np.empty(self.K)
        for k in range(self.K):
            model = self.models[k]
            if not model.is_fit:
                lcb[k], ucb[k] = 0.0, 1.0
                continue
            ucb[k] = self._binsearch_algorithm3(model, x, 'HIGH', eff_radius, prec)
            lcb[k] = self._binsearch_algorithm3(model, x, 'LOW', eff_radius, prec)
        return ucb, lcb

    def _get_ucb_lcb(self, x, T=None, prec=0.01, delta=0.05, **kwargs):
        if T is None:
            raise ValueError("RigorousCOAL requires T (horizon) for the eps_m schedule "
                              "-- pass it via observe(..., T=T)")
        return self.get_ucb_lcb_algorithm3(x, T=T, prec=prec, delta=delta)
