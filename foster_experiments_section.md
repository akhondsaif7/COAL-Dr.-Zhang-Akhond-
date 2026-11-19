# Foster et al. 2018 — Section 5 ("Experiments") Summary

Source: Foster, Agarwal, Dudík, Luo, Schapire, "Practical Contextual Bandits with
Regression Oracles," ICML 2018 (arXiv:1803.01088). BINSEARCH/Algorithm 3 is
deliberately excluded — see separate conversation notes for that topic.

## Datasets (18 total)

- **Full-reward (10):** MSLR-WEB30k, Yahoo! LTR Challenge V2.0, and 8 UCI
  classification sets. Bandit feedback simulated by hiding rewards for
  actions not chosen; multiclass structure comes directly from the labels.
- **Noisy UCI variants (8):** same 8 UCI sets, rewards instead redrawn from a
  noisy reward-matrix model — added because the plain classification-derived
  rewards always carry clean multiclass structure.
- **Replicates:** several per dataset (permuted order; regenerated rewards
  for the noisy variants), shared identically across every method compared.

## Methods compared

- **Own methods:** RegCB-Elimination (Algorithm 1, randomizes among
  plausibly-best actions) and RegCB-Optimistic (Algorithm 2, always takes the
  single highest-upper-bound action; run with no warm start in the
  experiments).
- **Baselines:** ε-Greedy, ILOVETOCONBANDITS (ILTCB, minimax-optimal),
  Bootstrap (Dimakopoulou et al.; oracle-compatible but no theoretical
  guarantee).
- LinUCB is mathematically equivalent to RegCB-Optimistic under a linear
  oracle, so it is not plotted as a separate method.

## Regression oracles

- ℓ2-regularized linear functions.
- Gradient-boosted depth-5 regression trees (GB5): tree-ensemble structure
  frozen at the start of each epoch, only leaf values re-optimized within an
  epoch (a computational speed-up, unrelated to BINSEARCH).
- A "supervised" (full-information, trained once on everything) reference
  baseline is also shown.

## Protocol

- Online, one example at a time; train/holdout split. Performance reported
  as **validation reward** on held-out data the algorithm never trains on
  (not regret).
- Each method tuned over 8 exponentially-spaced hyperparameter values × 5
  repetitions. RegCB reports its single best fixed value; baselines get the
  pointwise maximum across their own tuning grid (a deliberately generous
  handicap in the baselines' favor).
- Epoch lengths grow as `2^(i/2)`.
- **Parameter-tuning subsection, Section 5:** the theoretical `β_m` schedule
  (Theorems 2–4) is *not* used in the experiments. Direct quote: "we set
  `β_m = β` for all `m` and tune `β`" — a single flat constant, swept the
  same way as every baseline's own hyperparameter.

## Results

- **Figure 1** (validation performance vs. rounds, e.g. `letter`,
  `letter+N`, `yahoo`): both RegCB variants beat all baselines on
  unmodified UCI data; on the noisy variants and `yahoo`, ILTCB and
  Bootstrap close the gap substantially, Bootstrap slightly ahead.
- **Figure 2** (aggregate normalized relative loss across datasets, solid
  = GB5 oracle, dashed = linear oracle): with GB5, RegCB always beats
  ε-Greedy/ILTCB, sometimes loses to Bootstrap, stays within ~20% of the
  best method; Elimination edges out Optimistic in aggregate. With a
  **linear** oracle, RegCB dominates every baseline, including Bootstrap.
- **Figure 3** ("Results: Confidence Width" subsection): disagreement-set
  size and the optimistic action's confidence width, both plotted against
  round count `t` on a log scale, sliding-window-averaged (window 20).
  RegCB's width shrinks with fitted log-log slopes of about `−0.34`
  (`letter`), `−0.51` (`letter+N`), `−0.52` (`yahoo`); Bootstrap's width on
  `letter` barely shrinks, which the authors tie to its weaker reward
  there.
