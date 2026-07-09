# What didn't work — negative results

An honest log of approaches we built and then discarded (or shelved), with the
evidence, so we don't relitigate them. Scores are competition weighted
zero-mean R² on the recent validation tail (dates 1599–1698) unless noted.

For contrast, the things that **did** work: **online refit** (a daily gradient
step on yesterday's responders) is the single biggest lever, +0.003–0.005; and
a decorrelated **ensemble** of `xgb + gru_modelr + lstm_modelr` (each online)
reached **+0.01039**, ~93% of the 8th-place private score (+0.0112). The best
single model was `lstm_modelr` on ~280 recent dates at +0.00920.

---

## 1. Path signatures as ensemble members — *shelved (kept as theory)*
We built the full machinery: Chen iteration, log-signatures over the free Lie
algebra, Volterra / Hurst `(t−s)^{H−1/2}` reweighting, the minimal Lyndon basis
via `iisignature`, and AR(p)-based channel selection to pick which groups of
features to transform.

- Findings that held up: AR channel selection beat univariate-correlation
  selection; the minimal log-signature basis worked; `sig_transformer` only
  needed a smaller refit LR (`3e-5`), not an architecture change.
- **Why discarded:** none of the signature models moved the ensemble. They are
  too correlated with the RNNs — the GRU/LSTM already capture the intraday
  temporal structure the signature encodes, so the signature member added no
  decorrelated signal.
- **Status:** the code is preserved as `janestreet.theory.signatures` (reusable
  math) and the `mlp_sig` / `sig_transformer` models remain registered, but
  they are not in the production ensemble.

## 2. Training on the full history / curriculum — *discarded*
A curriculum model warm-started across chronological batches out to 800 dates
scored **+0.00765**, *under* the 280-date recency baseline (**+0.00893**).
Old data gives a **negative static R²** on the recent tail; online refit
recovers it. The recency window sweep confirmed the knee: online R² rises
monotonically 50→400 dates (400 best at +0.00712 for the plain GRU) — recent
data helps, the oldest data hurts. → We train on a recent window, not all
history.

## 3. Symbol identity as a model feature — *mostly negative*
The models never saw `symbol_id`. Adding it as a 39-way one-hot to XGB was
**flat (−0.00002)** — the per-symbol level is already removed by the rolling
per-symbol normalization, so the dummies just add trees. Only the coarse,
temporally **stable** feature-profile cluster helped, and only modestly
(**+0.00022**). A per-symbol embedding is therefore low priority.

## 4. Symbol clustering by responder co-movement (for routing) — *discarded*
On short recent windows, symbols co-move sharply (intra/inter correlation ratio
~18×), which looked promising for specialized per-cluster models. But the
grouping is **temporally unstable**: clustering an early window vs a late window
gives an adjusted Rand index of **−0.07** (chance). You cannot route a
specialized model on a membership that doesn't survive to test time.
*(The stable axis is the feature-profile clustering, ARI +0.46 — that one we
kept; see item 3.)*

## 5. Raw lagged responders as static features — *neutral*
Feeding the 9 previous-day responder columns straight into a static XGB was
flat (**+0.00591** vs **+0.00594** base). They only pay off through the ModelR
aux-branch path and through engineered **responder-signal** features
(multi-scale momentum + venue spread), which add +0.00030.

## 6. Full-pool (898-date) single window — *infeasible + unwanted*
Materializing all 898 pool dates in one training window OOMs (~28 GB > 16 GB
laptop, and > free-Colab's ~13 GB), and per item 2 the oldest data hurts
anyway. ~400 recent dates is the sweet spot.

## 7. Vanilla transformer and Mamba — *superseded / removed*
- The vanilla transformer was superseded by the inverted / covariate designs
  (iTransformer, TimeXer). Those are built and causality-verified but **not yet
  shown to beat the RNNs** — kept as *unproven*, one GPU run from a verdict.
- **Mamba** (selective state-space) was built but never validated as helpful and
  has been **removed** to keep the model zoo lean.

## 8. Time-decay recency weighting — *inconclusive (pending)*
Soft down-weighting of old training samples (`0.5^{age/halflife}`) is
implemented but not yet run to a fair, converged comparison against the hard
recency window. Open, not disproven.
