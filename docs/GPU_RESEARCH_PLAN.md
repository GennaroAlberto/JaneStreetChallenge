# GPU Research Plan — 30 h/week ≈ 60 experiments

*Kaggle GPU workflow proven (10–15 min per RNN member at bs=8). This plan
allocates the budget across EDA-motivated features, selection, and new
architectures, in waves with pre-registered kill rules. Companion runner:
`notebooks/kaggle_batch_runner.ipynb`.*

## What the error EDA says the gap is made of

Decomposition of the current best blend (+0.00986) on the tail:

| axis | finding | implied experiment |
|---|---|---|
| time of day | R² 0.012–0.014 at open/close vs **0.005 mid-day** (t 121–483 = 28% of energy) | time-phase conditioning; mid-day features |
| weight bands | q3–q4 (43% of energy) at 0.006–0.007 vs 0.011+ elsewhere | weight-aware training/calibration |
| symbols | spread −0.003 (sym 28) … +0.022; several *negative* | per-symbol calibration; cross-sectional attention |
| days | 5 negative days burn 0.0084 R²-mass | day-level shrinkage / risk gate |

On "did the top-3 use transformers?": our synthetic bench says attention
over *time* and *features* underperforms the GRU at this SNR — but it never
tested attention over **symbols**, our models never mix symbols, and the
per-symbol EDA says that's where variance lives. That's the attention
variant worth the budget. The other proven-elsewhere family is the
**supervised AE+MLP** (won JS 2021; its feature-level cousin is our best
block). Both are built and registered.

## Wave 1 — architecture probes (≈10 runs, ~3 h) — READY

| runs | what | verdict criterion (pre-registered) |
|---|---|---|
| 2 | `gru_modelr_xsec`, `lstm_modelr_xsec` (seed 42, 280d) | beat plain twins (+0.00811/+0.00901) by >0.0005 **or** stream corr < 0.9 with them and blend improves |
| 2 | `ae_mlp` (2 configs: latent 16/32) | solo > +0.006 **and** corr < 0.85 vs RNNs (row-wise model — diversity is the thesis) |
| 3 | `lstm_modelr` seeds 1–3 (the seed bag begins) | 3-seed average of streams > best single by > +0.0003 |
| 3 | refit-lr sweep for the winner of the above on GPU (3e-4/1e-3/3e-3) | family cliff map, per the standing rule |

## Discovery (2026-07-10): the venue-spread responders are 20× more
predictable than the target

Per-responder predictability (stage-1 XGB, even at smoke scale): the
A−B venue-spread responders score **r2 ≈ +0.14, r0 ≈ +0.07** on the tail
vs r6's ~+0.006 — spreads are persistent/mean-reverting and the features
nowcast them. Three exploitation routes, in cheapness order:

1. **Spread-predictor aux heads**: replace/augment ModelR's aux targets
   (r7/r8/r9/r10, all ~0.005-predictable) with r0/r2 — native forward
   targets with 20× the supervision SNR. One-flag change via
   `--aux-targets responder_0,responder_2,responder_7,responder_8`.
2. **Chain features — CONFIRMED +0.00069** (`chain_lab/chain.json`,
   stage-2 0.00425 → 0.00494 on a handicapped 100-day window): stage-1
   r̂0..r̂8 appended to stage-2 inputs. Fold into the pool rebuild; try a
   GPU-NN stage-1 for r0/r2 (XGB reaches 0.12/0.17 — the ceiling study
   says stage-2 converts stage-1 gains efficiently).
3. **Algebraic route — closed**: the fixed map has a negative propagated
   ceiling (Var(r3) = 1.55×Var(r6), stage-1 R²≈0 on it). Learned stage-2
   only.
4. **Story lab verdicts** (`story_lab/story.json`): same-time nonlinear
   ceiling **0.896** (linear 0.834 — nonlinearity is real); responder
   HISTORY adds zero at the ceiling and zero deployable (graveyard §10).
   Stage-2 of the chain should be a small nonlinear net; all further
   responder work goes into stage-1 quality.

## Wave 2 — EDA-driven features & conditioning (≈15 runs)

1. **Time-phase inputs**: append sin/cos(t) × top-feature interactions and a
   3-phase one-hot; ablate on lstm_modelr. Cheap, targets the mid-day hole.
2. **Per-symbol online calibration**: during the walk, fit αₛ (per-symbol
   shrinkage) on the trailing 20 days of that symbol's predictions;
   deployable. Implement in the walk, not the model — one run per stream.
3. **Weighted-training variants**: loss weights × (1 + λ·mid-weight-band
   indicator) — teach the models to care about q3–q4 where 43% of energy is.
4. **Day-risk gate**: shrink day-d predictions by the previous day's
   realized blend R² sign (online, deployable). One run.
5. **Deconvolved-lag features**: ridge-deconvolve yesterday's r8 path into
   ŝ (atlas machinery) and feed last-k innovations as features — the raw
   underlying signal instead of SMA blur. Pool rebuild + one bag member.

## Wave 2.5 — interpretation-probe follow-ups (2026-07-11)

1. **RNN distillation block (OOF hidden states)**: train an LSTM on
   1318–1497, extract its 16 hidden PCs for 1499–1598 + tail, ablate on
   the standard harness. Motivation: 16 PCs ≈ 0 R² at 50-day fits where
   134 raw features collapse to −0.019 (extreme data-efficiency), and the
   PCs are interpretable (clock/market/vol axes). One 10-min Kaggle
   member + one local ablation.
2. **Vol-interaction feature block** (5 alpha features × vol terciles) —
   local ablation in flight; if positive, fold into pool v3.
3. **Time×vol 2-way conditioning study** — disentangle the mid-day hole
   (mid-day is 42% low-vol yet scores worst by time band).
4. **Hidden-PC online refit probe**: a tiny ridge/MLP on the 16 PCs,
   refit daily — the data-efficiency finding suggests it may adapt to
   drift faster than any full model.

## Wave 2.6 — fresh data-engineering angles (2026-07-11)

Local batch in flight (base vs +weight-as-feature, +cross-sectional
ranks, +yesterday's market internals/breadth, +null-count). Untested
bigger swings, ranked:

1. ~~**Innovation labels (flagship)**~~ — **DEAD (2026-07-16)**: clean A/B
   scored −0.00567 vs the SMA-synthetic aux baseline (graveyard §14).
   Innovations are near-white; aux heads need *predictable* targets.
   Extracted rule: aux-target supervision SNR is the variable → raises
   priority of the spread-aux member (r0/r2 at 0.12–0.17).
2. ~~**Within-day expanding features**~~ — **flat (−0.00004, 2026-07-16)**:
   the pool's rolling deviations (window 1000 ≈ one day) already encode
   day-so-far context (graveyard §13).
3. **Event framing**: time-since-last-|innovation|-spike, spike
   intensity — replaces uniform-time with event-time context.
4. **Metric-mass audit by vol regime** (before any sample-curation
   experiments): where does the denominator energy live vs where our
   skill lives (we win calm markets; storms may carry the energy).

## Wave 3 — winner scale-out (≈20 runs)

Whatever survives waves 1–2: 6–8 seed bag at 280d + window jitter
(260/280/300), blended; then the definitive ensemble vs the +0.00986
incumbent. If `xsec` wins wave 1, bag *it* instead of plain lstm.
**Add vol-bagged members**: regime-subset training (vol terciles) as a
bagging axis — specialists averaged WITH the generalist scored +0.00721
vs 0.00660 single (regime_split_test.log); routing by them loses, but as
decorrelated members they're the cheapest new diversity source since
temporal bagging. Replicate (seed/threshold) then include 2–3 per bag.

## Wave 4 — stretch (≈10 runs, only if waves deliver)

TabM-style in-model ensembles (K parallel heads, one network ≈ K seeds at
1.2× cost); 2-day input windows (yesterday's path prepended to today's
sequence — recurrent TimeXer-lite); nowcast-head A/B (`--realized-aux`,
plumbing ready since day one).

## Standing rules (from the campaign's own graveyard)

- Every architecture verdict needs its own refit-lr sweep before "it
  doesn't work" is allowed.
- Ensemble admission = decorrelation + blend improvement, not solo score
  (the 140d-LSTM and sig-latents rejections are the precedent).
- Effects < 0.0005 need 2+ seeds before they count.
- Every dead idea goes to WHAT_DIDNT_WORK.md with its number.

## Logistics

**Rerouted local (2026-07-17)**: the Wave-1 architecture batch runs on
the Mac instead of Kaggle — `scripts/run_arch_wave.sh` chains ae_mlp,
gru_modelr_xsec, lstm_spreadaux (parquet r9 pool), lstm_modelr_xsec at
the 280d window behind the 280d checkpoint program
(`run_280_checkpoint.sh`: AE re-encode → pool v3 rs2 ladder → full-row
all_chain_ranks → 3× cheap lstm64 members). Full-size seed bag (lstm
s1–3, ~4h each) deferred — the cheap lstm64 bag covers the seed axis;
revisit on GPU if a burst becomes available. Kaggle remains the fallback
for anything needing >16 GB or parallel members
(`kaggle_batch_runner.ipynb` still works; blend harness unchanged:
`regen_ensemble_blend.py` + raw_to_memmap scores everything against the
incumbent).

## Submission pipeline (late submission, frozen private test)

Three pieces, all bundled into one Kaggle dataset zip:

1. **`scripts/serving/feature_state.py`** — online replica of
   `FeatureBuilder`: keeps per-symbol rolling/EWMA state and the
   lagged-responder buffers, ingests the once-per-`date_id` `lags`
   frame plus each `(date_id, time_id)` test batch, and emits the exact
   134-column frame the pool700_lags `Preprocessor` was fitted on.
2. **`scripts/serving/kernel.py`** — the kernel proper:
   `predict(test, lags)` loads the three v1-rnn3 checkpoints
   (`lstm_modelr` refit-lr 1e-3, `gru_modelr` 3e-4, `gru_modelr_xsec`
   3e-4 — all `FullPipeline` pickles sharing the same preprocessor),
   carries RNN hidden state across intraday batches
   (`ModelR.forward(x, hidden)`), walk-refits each stream once per new
   `date_id` when `lags` arrives, blends equal-weight, then applies
   per-symbol trailing calibration (ported `calibrate()` from
   `scripts/blend_v3.py`: lookback 20 dates, ridge_w 3.0, ≥500 rows,
   α∈[0,1.5]). Served via `kaggle_evaluation`'s `JSInferenceServer`.
3. **`scripts/pack_submission_weights.py`** — the packer: verifies the
   checkpoints + `pool700_lags/preprocessor.pkl` exist, zips them with
   `src/janestreet/**` and `scripts/serving/**`, and writes
   `manifest.json` (stack `v1-rnn3`; per-checkpoint stream tag,
   refit lr, blend weight) into
   `artifacts/colab/js_submission_weights.zip`. `--dry-run` prints the
   full listing + manifest without writing.

**v2 roadmap** (hooks recorded in `manifest.json:todo_v2`, not
implemented in v1): vol-scaling `pred·(σ̂/σ̄)^γ` with an online vol
nowcaster; the pool-v3 XGB stream (needs booster-only `save_model`
packaging to dodge the torch+libomp pickle segfault); refreshed
calibration hyperparams; seed-bagged members once GPU bursts land.
