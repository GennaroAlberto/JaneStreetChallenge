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

## 9. Symbol grouping by |correlation| of responders — *discarded (stability)*
The absolute-correlation variant of the co-movement clustering (groups
co-movers and anti-movers together). Cross-window stability (900–980 vs
1500–1580, k=3/6): ARI **+0.07 / +0.04** — better than signed corr (−0.07)
but still chance-level. Same epitaph as item 4: a frozen symbol map does
not survive to test time; the adaptive alternative is cross-sectional
attention (`gru_modelr_xsec`), which relearns relations daily.

## 10. Responder history as r6 input — *closed with prejudice*
Story lab (`artifacts/bench/story_lab`, 250 train days): given the
same-time responder state, five days of responder history add ZERO
(ceiling 0.8959 → 0.8955 with 45 lag cols; GRU-over-days 0.900 ≈ same;
log-sig L2 of the 5-day path 0.894). Deployable history-only models:
linear −0.000, MLP −0.001. The responder state is Markov for r6; the
past is already integrated into the SMAs. Corollary kept: nonlinearity
on the same-time state is worth +0.06 (0.834 linear → 0.896 MLP) — value
flows through stage-1 prediction quality (the chain), never through lags.

## 11. Hard vol-regime routing (two/three specialist models) — *discarded*
Tercile specialists routed by vol at prediction time: ensemble +0.00652 vs
single model +0.00660 — specialists lose even in their own regimes (low:
0.0161 vs 0.0166; mid: 0.0048 vs 0.0052) except a sliver in high-vol.
Data starvation beats specialization; thresholds drift. **Twist kept:**
averaging specialists WITH the single model scored +0.00721 (+0.0006) —
regime-subset training is a bagging axis, not a routing scheme.

## 12. Weight-as-feature, day-context internals, null-count — *discarded*
Data-engineering batch (`dataeng_batch.log`, standard harness): weight as
input + w×alpha interactions **−0.00036**; yesterday's market internals
(breadth/dispersion/|move|) **−0.00073** — day-level market history joins
responder history in the "history is dead" column; null-count flat
(−0.00004). Winner from the same batch kept: cross-sectional ranks of the
six atlas features, **+0.00045** → pool v3.

## 13. Within-day expanding features (XGB) — *flat*
Deviation-from-day-mean + position-in-day-range for the five atlas
features: **−0.00004** (`expanding_ablation.log`). The rolling per-symbol
deviations already in the pool (window 1000 ≈ one day) are "value vs the
day so far" in disguise — the explicit block is redundant. Pathogen:
already-normalized-away information.

## 14. Innovation labels as aux targets — *large negative*
The Wave-2.6 flagship: replace the two forward-SMA synthetic aux targets
with ridge-deconvolved next-innovations ŝ(t+1) and mean ŝ(t+1..t+4) from
the day's responder_8 path (train-time-only label engineering). Clean A/B,
identical everything else (`innov_aux_lab/result.json`, hidden [64,64],
200d, seed 42): A = **+0.00725**, B = **+0.00157**, delta **−0.00567**.
Why it fails: deconvolution *removes the smoothing that makes forward
targets learnable*. Innovations are near-white by construction — features
can't predict them, so the aux heads spend trunk capacity and gradient on
an unlearnable target and drag the shared representation down. The aux
mechanism works when targets are predictable (r7/r8/r9s/r10s all ~0.005;
spreads r0/r2 at 0.12–0.17 should be better still) — supervision SNR of
the aux target is the variable that matters, and deconvolution minimizes
it. Corollary: the deconvolution machinery stays an *analysis* tool
(atlas, lag features), not a label factory.
Bonus replication: the A arm shows hidden [64,64] @ 200d reaches +0.00725
online — a cheap-member config worth remembering for bagging.

## 15. ae_mlp (supervised AE+MLP, 2021-JS-winner family) — *rejected in config*
280d, seed 42 (`arch_wave/`): val R² peaks at epoch 1 (+0.0028) and
declines monotonically — the row-wise model overfits immediately; online
walk +0.00428, far under its pre-registered admission bar (solo >0.006).
Decorrelation is real (0.60–0.71 vs all streams) but equal-weight
admission costs −0.0002 and half-weight is a wash (+0.00003). Diagnosis:
under-regularized for this SNR, not necessarily dead — one retune
(lr 3e-4, dropout 0.3, aux_weight 2) is permitted if the machine idles;
no further budget beyond that.

## 16. gru_modelr_xsec — *first verdict VOID: epoch selection was leaked*
The initial 280d run (online +0.00371 vs twin +0.00811) is not a real
architecture verdict: fit-time validation went through the on_batch
reshape bug (future timesteps interleaved into the attention axis — val
R² 0.15, 12× the ceiling), so early stopping selected epochs on a leaked
metric. See FEATURE_RESEARCH.md "Leakage forensics". Both xsec members
retraining with the fixed, honest validation (`arch_wave2`); judgment
reserved until then. Meta-lesson: a fit-val above the noise ceiling is
a bug, not a breakthrough — and it can silently *sabotage* a model via
checkpoint selection even when every deployed number is honest.

## 17. Spread-aux heads (r0/r2 as aux targets) — *wash — VERDICT UNDER REVIEW*
**2026-07-20: its walk ran through the parquet eval path, which fed
symbol-major rows into the time-major reshape (scrambled sequences —
the bug found via the Kaggle 500d session). Checkpoint queued for a
fair raw-path rewalk before this verdict stands.**
The best-founded arch-wave bet: replace two default synthetics with the
20×-more-predictable venue spreads. Honest 280d run: solo +0.00590 vs
twin +0.00901; blend admission a wash at half weight. Refinement of the
aux-SNR rule from the innovation-labels kill: unpredictable aux targets
poison the trunk, but extra-predictable ones add nothing — the default
Volkova synthetics already saturate the regularization benefit.
lstm_xsec @ 0.5 likewise a sub-noise wash (§16's GRU twin, admitted, is
the family's representative in the stack).

## 18. Rewalking post-walk checkpoints — *leak, caught by the ceiling rule*
The lstm64 `.pt` files were saved AFTER `walk_forward_direct`, whose daily
refits absorb the entire validation tail into the weights. Rewalking those
checkpoints on the same tail "scored" +0.0142/+0.0146/+0.0153 — solo
streams above the noise ceiling (~0.014), which by the standing rule is a
bug, never a breakthrough. Fixed: checkpoints save post-fit/pre-walk;
contaminated artifacts deleted; bag retrained clean. Rule extracted:
**a checkpoint's provenance must predate every row it will be evaluated
on — post-walk weights are single-use.** Legit finding from the same
batch: the refit-lr split is real per family — GRUs prefer 3e-4
(+0.00811→+0.00845 plain), LSTMs keep 1e-3 (+0.00901 vs +0.00853 at 3e-4).

## 19. The 241-col enriched pool for RNNs — *negative at matched capacity*
lstm64 [64,64] @ 200d on pool241 (all nine XGB blocks as inputs, verified
identical to the XGB assembly): online **+0.00619** vs the 134-col twin's
**+0.00725** (−0.00106). Fit-val peaked at epoch 3 then declined — the
wider input accelerates overfit at this capacity. The blocks that pay for
trees do not transfer to small recurrent nets: the RNN already builds its
own temporal features, and the extra 107 columns cost first-layer capacity
plus noise. **Surgical follow-up run and also negative**: base + only
the 21 cross-sectional cols (ranks+chain+cluster) = **+0.00639** — same
damage as all 107 (+0.00619), so it was never drowning. Both enriched
arms show the peak-at-epoch-3 overfit signature; hypothesis: the chain
columns are near-predictions that a small online-refit RNN adopts as a
crutch. Two-dosage closure: engineered blocks stay tree-only; the RNNs'
cross-sectional information comes through xsec attention, which is
adaptive and confirmed (+0.0009 in-stack). Capacity-matched retest only
if a GPU wave has spare budget.

## 20. Fourier band-energy block (XGB) — *flat*
Causal trailing-64-step rfft band shares (low/mid/high) of the five atlas
features, 15 cols: **−0.00007** (`ablation_fourier.json`). As predicted
by the construction: the target is an SMA (low-pass) of near-white
innovations and the rolling stats already own the low-frequency content;
there is no exploitable mid/high-band structure. Same pathogen family as
§13. The frequency domain joins the closed doors — as features AND as
models (PatchTST/FEDformer-class died in the synth bench).
