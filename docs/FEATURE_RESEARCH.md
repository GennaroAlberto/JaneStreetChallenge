# Feature research — reverse-engineering the 79 features

*Applying the #555562 methodology (ACF structure, deconvolution, lead-lag) to
the features themselves. Companion to `scripts/profile_features.py` (the
"feature atlas") and `scripts/drw_feature_lab.py` (DRW-1st-style selection +
symbolic combos). Started 2026-07-09.*

## Recalibrated targets (private LB, from the leaderboard page)

The +0.0112 quoted in the tech report is Volkova's **public** score. Private:

| place | team | private R² |
|---|---|---:|
| 1 | ms capital | **0.013890** |
| 2 | Patrick Yam | 0.013273 |
| 3 | shorturl.at/LKhAD | 0.013163 |
| 4 | Haoze Hou | 0.011683 |
| 8 | Evgeniia Grigoreva (our replica base) | 0.010434 |

Our local +0.01039 tail score ≈ 8th-place level. The top-3 are well separated
from the pack (+0.0132–0.0139 vs 0.0117 for 4th) and none published — they
likely exploited structure the pack missed. Beating 1st means finding
~+0.0035 of real signal, not squeezing +0.0008 of variance reduction.

## Method

1. **Deconvolution.** responder_8 = forward SMA-4 of the underlying signal s
   (+ noise). Ridge-invert the SMA operator per (symbol, day):
   ŝ = (AᵀA + λI)⁻¹Aᵀ r8. Sanity checks all pass:
   responder_8 vs ŝ peaks at r = 0.85 (lag +1, ACF cutoff 4, triangular ramp
   R² = 0.99); responder_6 has alpha-band corr 0.37 vs nowcast 0.01 — its
   information sits entirely in the future window, confirming the #555562
   shift; responders 0/1/2 are ≈ 0 vs ŝ (A−B differences cancel the common
   signal); responders 4/7 show the SMA-120 ramp (cutoff ≈ 95–100 at our lag
   budget).
2. **Lead-lag atlas.** For every feature, corr(feature(t), ŝ(t+k)),
   k ∈ [−48, +48], averaged over ~3 000 (symbol, day) series.
   **Alpha score** = mean corr over k ∈ [+1, +20] (the target's window);
   **strict alpha** = k ∈ [+4, +20] (excludes ridge smear);
   **nowcast score** = mean over k ∈ [−20, 0].
3. **Structure stats.** ACF ramp linearity + cutoff (SMA detector), cross-day
   continuity, cross-sectional market share, intraday-profile share, tags.
4. Two windows 600 dates apart (900–980, 1500–1580) for stability.

## Findings (atlas: `artifacts/feature_atlas/win{900,1500}/`)

**F1. The alpha structure is stationary.** Strict-band alpha ranking across
the two windows: Spearman **+0.97**, sign agreement 97% among the 37 features
with |alpha| > 0.005. Unlike responder-comovement symbol clusters (ARI −0.07),
feature→signal timing is a durable property you can build on.

**F2. A feature taxonomy by timing** (win1500 numbers, strict band):

| group | members | signature |
|---|---|---|
| **Lead (true alpha)** | feature_37, 38 (α ≈ −0.067, peak +3); 18, 65 (α ≈ −0.04, peak +2, no nowcast) | correlate with *future* ŝ; 37/38 also carry the largest univariate target corr (−0.16) |
| **Reversal / nowcast block** | 39, 42, 44, 45, 46, 47, 49, 50, 53, 55, 56, 57, 58, 59, 60 | strong *negative* corr with realized ŝ at lag −3/−4 (feature_60 peaks at −0.45); several carry positive alpha (57: +0.040, 46: +0.037, 58/47: +0.027) — i.e. they encode short-horizon reversal |
| **Slow SMA-like** | 12, 67, 70 (ACF ramp R² ≈ 0.98, cutoff ≈ 160) | are themselves ~SMA-160 of some underlying series → deconvolvable, like the responders |
| **Market-wide** | 04, 05, 06, 07 (cross-sectional share ≈ 0.93) | market-level factors, alpha ≈ −0.011…−0.014 |
| forward aux ref | responder_6 strict α = +0.361 | ceiling reference |

**F3. Features ≈ descriptions of the realized path.** Nowcast magnitudes dwarf
alpha (−0.26 vs +0.04 best), consistent with #555562's "features predict the
*realized* r6(t−20) at R² ≈ 0.5". The exploitable content is the small lead
component + the reversal structure.

## What this changes (research roadmap)

1. **Innovation features from slow SMAs** — *confirmed +0.00025*
   (`artifacts/bench/ablation_innov.json`): trailing diffs of
   feature_12/67/70 at lags {1, 8, 40} on the standard XGB harness
   (0.00591 → 0.00615). Same magnitude class as rsig (+0.00030) and the
   cluster id (+0.00022); mechanisms differ so the lifts are plausibly
   partially additive — fold into the next pool rebuild.
2. **Alpha-ranked channel selection** — the signature/iTransformer variate
   channels and the DRW symbolic-combo pool should be seeded from the lead +
   reversal sets (37, 38, 18, 65, 57, 46, 58, 47, 56, 45), not univariate
   target corr (which conflates nowcast with lead) and not AR-selection.
3. **Nowcast aux heads** *(implemented, pending GPU A/B)* — realized
   responder_11 = r6.shift(20), responder_12 = r8.shift(4) as extra ModelR
   aux targets (`--realized-aux`, `num_aux=6`). Rationale: predicting the
   realized path is the R² ≈ 0.5 task; the forward head reads the future off
   a representation shaped by it.
4. **Market/idio decomposition of the reversal block** — *confirmed
   +0.00040* (`artifacts/bench/ablation_revdecomp.json`): splitting
   features 45/46/47/56/57/58/60 into per-(date,time) cross-symbol mean +
   idiosyncratic residual (14 cols, all contemporaneous/legal) lifts the
   XGB harness 0.00591 → 0.00631. Second-best block after atlas-seeded
   combos (+0.00052); the two likely overlap partially (both exploit the
   reversal family) — measure jointly at the next pool rebuild.
5. **DRW lab — first results** (`artifacts/bench/drw_lab/`, train 1399–1598,
   tail 1599–1698, XGB):

   | variant | n_feat | tail R² | Δ |
   |---|---:|---:|---:|
   | all 134 cols (baseline) | 134 | +0.00591 | — |
   | medoids only | 87 | +0.00576 | −0.00015 |
   | medoids + combos | 127 | +0.00586 | −0.00005 |
   | **all + combos** | 174 | **+0.00630** | **+0.00039** |

   Take-aways: (a) medoid pruning *hurts* here — unlike DRW's 890 raw
   features, our 134 are already curated, so keep all columns and *add*
   combos; (b) the +0.00039 combo lift beats the rsig lift (+0.00030);
   (c) 37/40 accepted combos are `min`/`max` gates, mostly on market-wide
   factors (e.g. `min(feature_07_avg, feature_36)`) — conditional structure
   depth-6 trees don't reach on their own; (d) **no rsig rediscovery**: the
   lag-responder columns never survive SHAP-consistency (raw lags are flat
   for trees), so stage C never gets to difference them — the combo pool
   must be seeded, not purely SHAP-selected.

   **v2 (atlas-seeded combos, `drw_lab_v2`)**: stage C re-run with the pool
   extended by the atlas lead + reversal sets and the lag responders
   (15 → 31 cols). `all+combos` improves to **+0.00643** (+0.00052 vs
   baseline, +0.00013 vs v1); 15/50 accepted combos pair atlas reversal
   features with market factors (e.g. `min(feature_07_avg, feature_45)`).
   Lag-responder combos still never pass — stage C screens by univariate
   target-corr, and lag differences (the rsig features) only pay through
   tree interactions (+0.0003 in the rsig ablation despite ~zero univariate
   corr). Next refinement if we push further: a tree-gain screen for
   candidates instead of the corr screen.

   **v3 (tree-gain screen, `drw_lab_v3`)**: stage C re-screened by XGB
   gain on candidate batches (no univariate-corr filter). The blind spot
   is mechanically fixed — 6/50 accepted combos touch lag responders
   (v1/v2: zero), e.g. `sub(feature_56, responder_4_lag1d)`,
   `max(feature_46, responder_8_lag1d)` — but the tail score is
   **+0.00626**, slightly *below* v2's corr-screened +0.00643. In-sample
   gain transfers a bit worse than train-corr here. Verdict: **v2's set
   stays the production choice**; the tree screen is kept as the tool for
   candidate families the corr screen structurally cannot see. Untried:
   union of v2+v3 accepted sets (likely marginal — heavy overlap in the
   market-gated reversal family both screens converge on).

6. **AE bottleneck features** — *confirmed, best block of the campaign:*
   **+0.00058** (`ae_lab_sup/ablation.json`, 0.00591 → 0.00649) from a
   **target-aware** denoising AE (recon + 2.0·weighted target head,
   134 → 8) on the broad pool. The two control arms pin the mechanism:
   plain reconstruction AE is flat (−0.00003 — reconstruction preserves
   variance, not alpha) and the supervised AE on the 31-col curated core
   is flat (+0.00003 — the core's signal is already exposed as columns;
   the AE pays by compressing nonlinear composites out of the *broad*
   pool). DRW's "AE after selection" placement did not transfer;
   supervised-on-broad did, and better. Encoder + latents persisted for
   the log-signature-of-latent-path experiments (`sig_latents.py`).

7. **Log-signature of the AE latent path** — *measured flat*
   (`sig_latents_{sup,plain}/ablation.json`): depth-2 log-signatures
   (Lévy areas, Hurst 0.1, window 32) over the 8-dim latent path, on a
   row-step-2 harness. Supervised latents: sig +0.00011, z+sig −0.00005;
   plain latents: −0.00018 / −0.00001 — all inside the noise band, vs the
   pointwise supervised latents' +0.00058. Together with the original
   signature post-mortem this is now a double-confirmed genre verdict:
   the exploitable structure here is pointwise nonlinear compression of
   the cross-section, not path geometry. Config caveat: one depth/window;
   not worth a sweep given two independent flat results.

## The pool rebuild (2026-07-10) — joint value of all confirmed blocks

Full-row standard harness (`pool_rebuild/ablation.json`):

| variant | cols | R² | Δ |
|---|---:|---:|---:|
| base | 134 | +0.00591 | — |
| + combos, decomp, rsig, innovations, cluster | 218 | +0.00644 | +0.00054 |
| **+ supervised-AE latents (all blocks)** | 226 | **+0.00665** | **+0.00074** |

Reading: individual deltas sum to +0.00227; jointly they deliver
+0.00074 — roughly **one third survives overlap** (combos/decomp/rsig all
mine the reversal-and-lag family; the AE compresses the same pool). The
AE keeps the largest *unique* marginal (+0.00021 on top of every explicit
block). **The XGB stream moves 0.00591 → 0.00665 (+12.5% relative).**
Next step: regenerate the ensemble — the enriched XGB stream (especially
its AE component) should be less correlated with the RNN streams than the
old one; re-blend against gru/lstm-online and re-measure the +0.01039
headline. Note stream row-order alignment: `pool_rebuild/preds/*.npz`
follow pool700_lags order; the run2 RNN streams follow the old pipeline
order — regenerate RNN streams from the memmap rather than reorder.

## Ensemble regeneration (2026-07-10)

Fresh streams (old preds/checkpoints were lost), all aligned by key on the
1599--1698 tail (`regen_ensemble/blend.json`):

| stream | online R² |
|---|---:|
| xgb_base (134 cols, 200d) | +0.00591 |
| xgb_all (226 cols, 200d) | +0.00665 |
| gru_modelr (280d, refit 1e-3) | +0.00811 |
| lstm_modelr (280d, refit 1e-3) | +0.00901 |

**Blends: old-style +0.00954 → enriched +0.00986 (+0.00031).** The
enriched stream keeps ~half its +0.0007 single-model lift at blend level —
its correlation to the RNNs rose 0.57→0.60 (GRU) and 0.61→0.64 (LSTM),
diluting the rest. Not comparable to the May +0.01039 headline (different
member configs). Queued upside: xgb_all on 280 days; GRU refit lr 3e-4
(May optimum); temporal-bag members (Kaggle GPU workflow now exists:
`notebooks/kaggle_train_lstm_member.ipynb`, portable parquet pool in
`artifacts/colab/`).

## Data-engineering batch (2026-07-11) — ranks win, three angles closed

Base 0.00660 (row-step-2 harness): **cross-sectional ranks of the six
atlas features +0.00045** (third-largest block ever — and it sat in
FeatureBuilder as a default-off option, never ablated). Closed: weight
as feature −0.00036, yesterday's market internals −0.00073 (day-level
market history joins responder history: history keeps testing dead),
null-count flat. **Pool v3 candidate set**: chain(+84) + AE(+58) +
combos(+52) + ranks(+45) + decomp(+40) + rsig(+30) + innovations(+25) +
cluster(+22) + volinter(+15), with the vol-scaled post-processing on top.

## Pool rebuild v2 (2026-07-11) — chain block + calibration stacked

Full-row harness (`pool_rebuild_v2/ablation.json`,
`regen_ensemble/blend_v2.json`):

| XGB variant | cols | R² | blend (+2 RNN) | calibrated |
|---|---:|---:|---:|---:|
| base | 134 | +0.00591 | +0.00954 | +0.00973 |
| all blocks | 226 | +0.00665 | +0.00986 | +0.01000 |
| **all + chain (OOF)** | 235 | **+0.00749** | +0.01003 | **+0.01019** |

The chain block (9 OOF stage-1 responder predictions; 2-fold train-side,
full-train tail-side) adds **+0.00084 on top of every other block** — the
largest single marginal of the campaign, and stacked, the XGB stream is up
**+27% relative** (0.00591 → 0.00749), closing most of its gap to the GRU
member (0.00811). Per-symbol calibration adds a consistent +0.00016 at
blend level across all variants. Current best blend: **+0.01019** (members
still below May spec: 200-day XGB window, GRU at refit 1e-3 — the cheap
upgrades queued in GPU_RESEARCH_PLAN should clear +0.0104).

## Error forensics (2026-07-11) — reverse-engineering our own residuals

Residuals of the best blend treated as a target and run through the
forensic toolkit:

1. **Heteroscedasticity blindness — the big bias.** |y| is predictable at
   centered R² **+0.12** (top features 13/14/69: previously mid-tier atlas
   entries, now identified as volatility features). Our predictions
   under-modulate with it. Fix: scale ŷ·(σ̂/σ̄)^γ with a train-window
   |y|-model; γ≈0.3 (plateau 0.2–0.3). Honest gain **+0.00036** (γ chosen
   on tail half 1, scored on half 2). Full tail: 0.01003 → **+0.01039**.
2. **Magnitude underconfidence**: E[y | pred decile] / pred ≈ 1.07–1.11
   across ALL deciles — a global amplification slope; small gain, folds
   into the vol-scaling.
3. **Mistakes are NOT feature-groupable at row level**: an XGB fit on the
   residual scores **−0.025 out of sample** (adding it to the blend hurts
   by the same). The models have cleanly extracted the row-level feature
   signal; remaining error ≈ innovation noise + the systematic effects
   above.
4. **No error persistence**: day-to-day residual corr −0.016; within-day
   ACF beyond the mechanical SMA-20 window ≈ −0.02. Nothing to harvest
   from past errors.
5. **Market-common error component**: 6.4% of residual variance is
   per-timestamp common vs a 2.6% iid floor. Partly mechanical (market
   innovations are genuinely shared); the row-level residual model
   (market-avg features included) could not harvest it → remaining
   candidate is a cross-sectional model (xsec attention, GPU queue).
6. **Tails**: top-1% |y| rows carry 23.4% of error — exactly proportional
   to their energy share (no excess); sign accuracy there is 55.8% and
   mean |pred| is 37× too small — which is precisely what the vol-scaling
   corrects in aggregate.

**Stack test verdict**: vol-scaling **subsumes** per-symbol calibration —
stacked (either order) equals vol-scale alone on the full tail and loses
~0.0000–0.0003 on the untouched half. Both are magnitude modulators; the
vol model sees per-symbol volatility features directly and does the same
job without trailing-window estimation noise. **Production post-processing
= blend → vol-scale only** (γ=0.3, train-window |y| model). Calibration
retired as subsumed. Current headline: **+0.01039 full tail / +0.01019
untouched half** — at May-headline level with members still under spec
(200-day XGB, GRU at refit 1e-3).

## Interpretation probes (2026-07-11): vol regimes & hidden states

**Vol-regime conditioning (probe A).** Feature alphas *reorganize* across
volatility terciles (proxy: feature_13): feature_37 peaks mid-vol
(+0.015) and dies high-vol (+0.002); feature_06 decays 6× from calm to
stressed; feature_60 flips sign. **The blend is a calm-market specialist:
R² +0.0212 low-vol vs +0.0073 high-vol.** The earlier "mid-day hole" is
partially confounded (mid-day is 42% low-vol) → 2-way time×vol
conditioning is the follow-up. Vol-interaction feature block (5 alpha
features × vol terciles, 15 cols): **+0.00015** on the standard harness —
positive, minor (trees already reach most of the conditioning through
splits); fold into the next pool consolidation. The market-common residual correlates
with nothing observable (all ≤0.03; third independent negative — treat
as shared innovation noise).

**Hidden states (probe B).** The trained LSTM's post-RNN state (4×96
dims/timestep) compresses to 16 PCs at 93.8% EVR, and the PCs are
interpretable: PC1 = clock+market factor (corr 0.31 time, 0.25 f06,
−0.07 with y!), PC4 = volatility (0.15 f13, 0.10 |y|). The LSTM has
internally built the same conditioners probe A found. Transfer test at
50-day fits: **16 hidden PCs alone ≈ 0.000 R² while 134 raw features
overfit to −0.019** — the state is a far more data-efficient
representation (implications for fast online adaptation). base+hidden
beat base by +0.0056 under identical (broken-baseline) conditions — NOT
a production claim; the honest version needs OOF hidden states (LSTM
trained on an earlier window) → queued as a Kaggle job ("RNN
distillation block"). Bonus: the probe surfaced and fixed a
backward-compat bug (pre-xsec checkpoints crashing on the new forward).

## Post-processing: per-symbol online calibration (+0.00014)

Fitting a per-symbol shrinkage α (weighted cov/var, ridged toward 1,
trailing 20 days, clipped [0, 1.5]) during the walk lifts the enriched
blend 0.00986 → **0.01000**. Deployable (trailing-only), free. The hard
variant (damp symbols with trailing R² < 0: ×0.3) LOSES 0.00024 — binary
gates overreact to noisy trailing estimates; smooth shrinkage only.

## Architecture triage on the synthetic world (2026-07-10)

Six models under the identical online-refit protocol, identical inputs, on
the #555562-DGP world (`artifacts/synth/world_full`, reference ceiling
+0.0488; decision rule fixed in advance: *an architecture that cannot beat
the GRU on a world designed for its own inductive bias gets no real-data
GPU budget*):

| model | online R² | % of ceiling |
|---|---:|---:|
| gru (single branch) | +0.0392 | **80%** |
| gru_modelr (aux heads) | +0.0312 | 64% |
| xgb (static) | +0.0307 | 63% |
| itransformer (causal) | +0.0273 | 56% |
| transformer (vanilla) | +0.0261 | 54% |
| timexer (inverted bridge) | +0.0219 | 45% |

Verdicts: **TimeXer fails its own mechanism test** (last, despite a planted
cross-day×clock effect built for it) → dropped from the GPU queue.
iTransformer edges vanilla attention but trails the GRU badly → ensemble
probe only, low priority. **Aux heads hurt at high SNR** (gru > gru_modelr
by 16 points of ceiling) while helping at the real problem's low SNR —
confirming aux supervision as a *regularizer*, which raises (not lowers)
the expected value of the nowcast-head A/B on real data. Caveats: one
seed, 10 epochs, one config per family; the ordering GRU ≫ attention is
too large to be config noise, the iTransformer-vs-transformer gap is not.

## Tools

```bash
# atlas on any window
uv run python scripts/profile_features.py --min-date 1500 --max-date 1580 \
    --out artifacts/feature_atlas/win1500
# post-hoc band analysis: curves.npz has the full lead-lag curves per feature

# DRW-style selection + combos + ablation on a memmap pool
uv run python scripts/drw_feature_lab.py --data artifacts/precomputed/pool700_lags \
    --train-lo 1399 --train-hi 1598 --valid-lo 1599 --valid-hi 1698 \
    --out artifacts/bench/drw_lab
```

## Wave-2.6 finale (2026-07-16) — expanding flat, innovation labels dead

The two "bigger swing" items, both on 200-date local harnesses:

- **Within-day expanding features: −0.00004** (flat). The rolling
  per-symbol deviations already in the pool (window 1000 ≈ one day) *are*
  "value vs the day so far" — the explicit block is redundant.
  Graveyard §13.
- **Innovation labels (the flagship): −0.00567.** Clean A/B on identical
  ModelR members where only the two synthetic aux targets differ
  (`innov_aux_lab/result.json`): Volkova forward-SMA synthetics
  **+0.00725** vs deconvolved ŝ(t+1)/ŝ(t+1..4) **+0.00157**. The failure
  is diagnostic, not noisy: deconvolution removes exactly the smoothing
  that makes forward targets learnable — innovations are near-white, the
  aux heads chase an unpredictable target, and the shared trunk pays.
  **Rule extracted: aux-target supervision SNR is the variable that
  matters.** This *raises* confidence in the spread-aux member
  (r0/r2 predictable at 0.12–0.17, vs ~0.005 for the SMA synthetics) —
  the opposite corner of the same axis. Graveyard §14.
- Bonus replication from the A arm: hidden [64,64] @ 200d reaches
  **+0.00725** online — cheap-member config for future bagging.

Wave-2.6 net result: ranks (+0.00045) is the sole survivor of six angles;
event framing and the metric-mass audit remain unrun.

## 280-date checkpoint (2026-07-17) — stack v3, new production number

Everything brought to the same 280d footing (train 1318–1598/1597;
`pool_rebuild_280*`, `blend_v3.json`). Verdicts on the untouched tail
half (dates 1650–1698), γ selected on half 1.

**XGB stream**: full-row all_chain_ranks = **+0.00794** (v2 200d:
+0.00749). The rs2 ladder: base +0.00686 / all_chain +0.00828 /
+ranks +0.00842 — depth alone worth ~+0.00026, ranks keeps a positive
marginal (+0.00014) on top of all nine blocks.

**Cheap lstm64 bag** ([64,64], ~20 min fits, seeds 42/1/2): solo
+0.00822/+0.00736/+0.00920; **bag average +0.00944 — beats every single
stream in the pool**, including the full-size LSTM (+0.00901) at a
tenth of the training cost. Seed bagging at small capacity > one large
member, at this SNR.

**Stack v3** (equal-weight → per-symbol calibration → vol-scale):

| blend | cal, full | cal+vol, full | cal+vol, h2 |
|---|---:|---:|---:|
| incumbent (xgb_v2+gru+lstm) | +0.01019 | +0.01043 (γ=0.2) | +0.01024 |
| v3_core (xgb_v3+gru+lstm) | +0.01026 | +0.01048 (γ=0.2) | +0.01058 |
| **v3_bag4 (+lstm64_bag)** | **+0.01036** | **+0.01055** (γ=0.15) | +0.01056 |
| v3_all6 (members flat) | +0.01025 | +0.01040 | +0.01039 |

v3_core and v3_bag4 tie on h2 (Δ=0.00002 ≪ seed noise); bag4 wins full
tail and calibrated-raw, and carries more members → **v3_bag4 is the new
production stack: +0.01055 full / +0.01056 untouched half** (+0.0003 vs
incumbent). all6 loses — flat member weighting over-weights the LSTM
family (bag~lstm corr 0.93); one vote per *stream*, not per member.
First checkpoint clearly above the Volkova-replication level (+0.010434
private). γ re-selected at 0.15–0.2 on this blend (was 0.3).

## Leakage forensics: the on_batch reshape bug (2026-07-18)

Both xsec members reported fit-time val R² **0.15–0.17** — 12× the noise
ceiling — while their walks scored +0.004/+0.007. The paper trail
(probes in scratchpad, reproduced end-to-end):

1. Clean per-day (S,T,K) forward on the lstm_xsec checkpoint: honest
   +0.0062; future-perturbation test: **zero** future dependence — the
   architecture is causal, the network innocent.
2. `_validate_one_epoch` on a rebuilt valid set: honest +0.0075 with
   `on_batch=False`, **leaky 0.1504 with `on_batch=True`** — exactly the
   flag the trainer's val loader hardcodes.
3. Root cause, `dataset.py.__getitem__`: the on_batch reshape
   `X.reshape(t,-1,k).swapaxes(0,1)` assumes time-major rows (Kaggle
   frame layout) but FitData is symbol-major. Same shape out, interior
   stride-interleaved: the cross-sectional axis carried **up to 37
   consecutive future timesteps of the same symbol** — fully covering
   the target's (t, t+20] window. The xsec attention (which mixes across
   that axis) read the future; plain RNNs merely saw scrambled
   sequences (degraded val, no inflation — why this sat undetected).

Blast radius: **all walk/production numbers unaffected** (pipeline path);
training unaffected (in-RAM presort path); fit-time early stopping was
scrambled for every memmap-trainer member — cosmetic for plain RNNs,
**epoch-selection-randomizing for the xsec pair**. Fixed with an
order-agnostic explicit (symbol,time) sort; verified 0.1504 → 0.0078.
xsec pair + spreadaux retraining with honest selection (`arch_wave2`).
Detection heuristic for the book: *any fit-val above the noise ceiling
is a bug, not a breakthrough.*

## Arch wave verdicts, honest validation (2026-07-18) — xsec_gru admitted

Reruns with fixed epoch selection (`arch_wave2/admission.json`):
gru_xsec online **+0.00371 → +0.00679** (the leaked metric had picked
epoch 4 of 15; honest picks epoch 9) — the bug was actively sabotaging
it. lstm_xsec (+0.00749) and spreadaux (+0.00590) picked the same epochs
as before by coincidence.

Admission vs stack v3 (+0.01055/+0.01057), candidates at half weight
per the weak-member convention:

- **xsec_gru @ 0.5: ADMITTED** — full +0.01062, untouched half
  **+0.01066**. Best h1 of all candidates (clean selection), confirmed
  on h2. Admitted on decorrelation (0.57 vs xgb, lowest RNN-family corr
  in the pool), not solo strength — the diversity thesis, working as
  designed. **Production stack is now v3.1: +0.01062 / +0.01066.**
- xsec_lstm @ 0.5: wash (+0.00003–4). Rejected; preds kept.
- spreadaux @ 0.5: wash. The high-SNR-aux thesis did NOT transfer:
  refined rule — *unpredictable aux targets actively hurt (innovation
  labels, −0.0057), but more-predictable-than-default targets don't
  automatically help; Volkova's synthetic set is near-optimal.*
  Graveyard §17.
- xsec_gru+spreadaux @ 0.5: h2 +0.01068 but worse h1 than xsec_gru
  alone → not taken (h2 is the verdict set, not the selection set).

Cross-sectional attention is the one architecture bet of four that paid.

## Xsec deep-dive, round 1 (2026-07-18) — refit-lr was the bottleneck

The user's call ("fit the transformer properly") pays immediately.
Walk-only refit-lr sweep on the admitted gru_xsec checkpoint
(`arch_wave2/rewalks/`):

| refit protocol | online R² |
|---|---:|
| 1e-3 (RNN-family default) | +0.00679 |
| **3e-4** | **+0.00870** |
| 3e-3 | −0.00260 |
| 1e-3, attention frozen | +0.00827 |

The attention params are brittle under 1e-3 daily steps — freezing them
alone rescues +0.0015, and 3e-4 full-model refit is better still. The
plain-RNN refit cliff does NOT transfer across the attention boundary;
the standing sweep rule catches its second architecture (after
sig-transformer's 3e-5).

**xsec_gru @ 3e-4 beats its plain twin solo (+0.00870 vs +0.00811)** —
first xsec member to do so — and admission at FULL weight gives
**stack v3.2 = v3_bag4 + xsec_gru_r34 @ 1.0 → cal+vol +0.01088 full /
+0.01088 untouched half** (from +0.01062/+0.01066). Both halves agree;
full weight beats half (+0.01078).

Pending same-session: warm-start transplants (lstm walked at the bad
1e-3 → rewalk at 3e-4 when machine frees; gru fine-tuning now), then
rewalk of the PLAIN gru stream at 3e-4 (the GPU plan's queued "May
optimum" note suggests the plain family may also prefer it).

## Stack v3.4 — the window harvest (2026-07-19)

Scale-out results (`window_scaleout_driver.log`, `pool_rebuild_500/`):
xgb_500 deployable (rs2 train, full-row valid) **+0.00839**; lstm64 400d
bag s42/s1/s2 = +0.00934/+0.00885/+0.00954, **bag +0.0102 solo** — a
3-seed cheap bag now beats every stream we had a week ago. Knee probe
(rs4 pair): 500d +0.00961 vs 700d +0.00974 — **still rising at 700d**,
decelerating; max-window training is justified for all future members.

**v3.4 = avg(xgb_500, gru_r34, lstm_modelr, bag400) + xsec_gru_r34@1.0
→ calibration → vol-scale(γ=0.2) = +0.01135 full / +0.01139 untouched
half** (v3.3: +0.01093/+0.01089). Anchor-mapped private-equivalent
≈ 0.0122–0.0124: clear 4th-place-class, ~0.0007–0.0009 (proxy) from the
3rd-place wall (0.013163). Untapped: full-size 500d members (Kaggle
queue), 700d everything, vol-bag members, distillation.

Submission pipeline: replay gate GREEN for both engines (max dev 1.9e-06
over 9 dates, all families) — the kernel's incremental features are
float32-exact twins of the offline recipe. Kaggle pool 998–1698 exported
(4.7 GB zip, 701 partitions).

## Power-axis verdicts (2026-07-19)

- **241-col pool for RNNs: negative** at matched capacity (graveyard §19;
  −0.00106 vs the 134-col twin). Surgical follow-up open: cross-sectional
  blocks only (21 cols), or capacity-matched.
- **PatchTST: +0.03022 = 62% of ceiling** on the synthetic world
  (`bench_0719_1310.json`) — best of the attention-over-time family
  (vanilla 54%, iTransformer 56%, TimeXer 45%) but 18 points under the
  GRU bar (+0.0392 / 80%). Same early-overfit signature (val peak epoch
  3). Pre-registered kill rule applies: **no real-data budget.** The
  time-attention door is now closed four ways; cross-sectional attention
  remains the only transformer that ever paid here.
- Capacity×window queue built and validated (xsec wide/h8 at 500d, core
  streams at 500d) — waiting on the Kaggle GPU session.

## Capacity night verdict (2026-07-20) — sweet spot found, bag saturated

Ladder @ 400d, seed 42 (`capacity_night.log`): [64,64] +0.00934 |
**[96,96] +0.00973** | [128,128] +0.00932 | [96,96,96] +0.00965.
Width pays to ~0.7M params then gives it back — capacity×data coupling
measured directly. Depth (96³) < width (96²) at this scale. Winner
seeded: 96²-bag(3) solo +0.0105; **cross-capacity bag9 (6×64² + 3×96²)
solo +0.01085 — strongest stream ever**. BUT stack v3.4c (bag9 core) =
+0.01144/+0.01142, identical to v3.4b: the cheap family is saturated as
a blend contributor — its marginal strength is now self-correlated.
Implications: (a) next blend jump must come from DECORRELATED sources —
the Kaggle 500d full-size members and the XGB stream; (b) for Kaggle,
128³ at 500d is a borderline bet (the sweet spot moves right with data,
but 400d already rejected 128²); (c) production stack remains
**+0.01144 / +0.01142**.

## Decorrelation axes, round 1 (2026-07-20) — fourier flat, volbag wash

- **Fourier band-energies: −0.00007** (graveyard §20). Construction
  predicted it; door closed with a number.
- **Vol-bag specialists @ 500d**: calm +0.00677 / storm +0.00537 solo,
  corr vs xgb_500 = 0.87/0.79. Admission: pair @0.5 **worse**
  (+0.01142/+0.01132); calm-only @0.5 a sub-noise wash
  (+0.01147/+0.01144 vs +0.01144/+0.01142). The §11 probe's +0.0006
  twist does NOT replicate at production scale — it was measured against
  a lone 200d rs2 model; the modern stack's vol-scaling + calibration +
  stronger XGB already subsume regime information. Graveyard §11 updated
  in spirit: vol-bagging is closed at this stack level.
- Still pending: nowcast-aux A/B, subsample-of-700 × decay members
  (wave2 driver), Kaggle 500d full-size members (user side).

## Stack v3.5 — the 500d full-size harvest (2026-07-20)

Kaggle members (fixed bundle, VALID walks): **gru_wide(128³) +0.00999**
(best member ever), xsec_gru_wide +0.00972, gru_s42@500d +0.00951,
lstm_s42@500d +0.00896 (LSTM window-saturated). Wide-at-500d confirms
the sweet-spot-moves-right prediction after 400d rejected 128-wide.

Admission (`admit3.py`): v3.4b +0.01144/+0.01142 → **v3.5 (curated:
xgb_500 + bag6 + gru_wide + xsec_wide + lstm280 @1.0; xsec_r34 +
gru500 @0.5) = +0.01182 full / +0.01177 h2** (picked on h1 ≈ 0.01187
over all-500d-six's 0.01174; six's h2 +0.01184 noted). "Everything(9)"
LOSES (+0.01161) — the 280d GRU-family streams are now pure dilution.
Anchor-mapped private-equivalent ≈ **0.0127–0.0129: between 4th
(0.011683) and 3rd (0.013163)** — the top-3 wall is ~0.0005 away on the
proxy. Checkpoints for all four members in artifacts/out (submission v2
payload). Pending: nowcast A/B + subsample×decay trio (wave2), late
submission (user side).
