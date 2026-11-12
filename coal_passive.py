"""
Passive mode: always query, regardless of the model's own uncertainty.

This is the baseline every active-learning strategy (cheat, rigorous) is
compared against -- it never computes a UCB/LCB at all.
"""

from coal_common import COALBase


class PassiveCOAL(COALBase):
    """Always queries. `active` is accepted for interface compatibility but ignored."""

    def observe(self, x, y_true, active=True, **kwargs):
        return super().observe(x, y_true, active=False)

    def observe_reward_vector(self, x, r_vec, active=True, **kwargs):
        return super().observe_reward_vector(x, r_vec, active=False)
