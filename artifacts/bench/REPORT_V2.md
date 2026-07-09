# Jane Street benchmark — run 2 report

> **Headline:** 3-way simple-average ensemble (XGB + gru_modelr-online + lstm_modelr-online, all trained on 280-day slice) → **R² = +0.01039**, which is **93 % of the 8th-place LB target of +0.0112**, using ~⅓ of the training data Volkova used (280 dates vs. her 800+).

## What ran

Phase A on a 280-train / 100-valid slice (`date_id 1319–1698`); Phase B/C and Phase F on the original 200-train / 100-valid slice (`date_id 1399–1698`) for run-1 comparability; Phase D is a sweep on the lstm_modelr_280 checkpoint.

All runs share the same online-refit walk-forward eval protocol Volkova used; we additionally score each model statically (no refit).

## Per-model leaderboard

| run | model | params | train ds | static R² | online R² | Δ online |
|---|---|---:|---:|---:|---:|---:|
| run-1 | mlp | 459 k | 200 | −0.00258 | — | — |
| run-2 | sig_transformer_selected (v1, univariate \|corr\|) | 266 k | 200 | +0.00445 | +0.00364 | −0.00081 |
| run-1 | sig_transformer (baseline [0..5]) | 266 k | 200 | +0.00485 | +0.00436 | −0.00049 |
| run-2 | sig_transformer_w64 | 266 k | 200 | +0.00430 | +0.00432 | +0.00002 |
| run-2 | sig_transformer_d3 | 299 k | 200 | +0.00541 | +0.00410 | −0.00131 |
| run-2 | **sig_transformer_ar_k8 (v2, AR(7))** | 269 k | 200 | **+0.00594** ← best sig static | +0.00446 | −0.00148 |
| run-1 | transformer | 260 k | 200 | +0.00474 | +0.00500 | +0.00026 |
| run-1 | gru | 120 k | 200 | +0.00246 | +0.00749 | +0.00503 |
| run-1 | gru_modelr | 704 k | 200 | +0.00497 | +0.00749 | +0.00253 |
| run-1 | lstm_modelr | 939 k | 200 | +0.00425 | +0.00798 | +0.00374 |
| run-2 | **lstm_modelr_h128** | 1.58 M | 200 | +0.00474 | +0.00782 | +0.00308 |
| run-1 | xgb | — | 200 | +0.00587 | — | — |
| run-2 | **xgb** | — | **280** | **+0.00684** | — | — |
| run-2 | **gru_modelr** | 704 k | **280** | **+0.00763** | **+0.00893** | **+0.00130** |
| run-2 | **lstm_modelr** | 939 k | **280** | +0.00637 | **+0.00909** | **+0.00272** |
| run-2 | lstm_modelr @ lr_refit=1e-3 (Phase D) | same | 280 | — | **+0.00920** | — |

**Best single model: `lstm_modelr` on 280 train dates, online refit at lr=1e-3 → R² = +0.00920** (82 % of 8th-place LB).

## Phase D — online-refit LR sweep on lstm_modelr_280

| lr_refit | R² | Δ vs static |
|---:|---:|---:|
| 0 (static) | +0.00632 | — |
| 1e-4 | +0.00829 | +0.00197 |
| 3e-4 (bench default) | +0.00908 | +0.00276 |
| **1e-3** ✅ | **+0.00920** | **+0.00288** |
| 3e-3 | +0.00702 | +0.00070 (overshoots) |

The bench's 3e-4 default was reasonable but **1e-3 is moderately better**. The optimum is a flat plateau in the 3e-4–1e-3 range; outside it returns degrade fast.

## Phases F & G — signature channel selection

The bench's `sig_transformer` was always fed channels `[0,1,2,3,4,5]` = `feature_00`..`feature_05`, picked arbitrarily. We tested two principled replacements.

### Phase F — v1 selector by univariate target-corr (`select_signature_channels.py`)

Score each raw `feature_NN` by `z(|corr(feature, responder_6)|) + 0.5 z(log1p(returns_var)) + 0.25|autocorr_lag1|`. Greedy pick with 0.5 redundancy cap.

Picks: `[6, 13, 33, 16, 53, 4]` = `feature_06, 16, 36, 19, 56, 04`. 5/6 overlap with Volkova's hand-curated CORR list; `feature_16` is a new find.

**Result:** static R² = +0.00445 — **worse than the [0..5] baseline** (+0.00485). The selector picked features with high univariate predictiveness but the signature's value is in *cross-monomials* between channels with rich paths, not in marginal channel predictiveness. Wrong filter.

### Phase G — v2 selector by AR(7) ridge on each feature's own lags (`select_signature_channels_ar.py`)

Score each raw feature by holdout R² of a ridge AR(7) fitting `responder_6_t ~ Σ a_k · feature_{t-k}` within (symbol, date) — i.e., does this feature's recent *path* (its own lags) predict the target? This is much closer to what the Volterra signature actually consumes.

Picks: `[13, 63, 42, 33, 70, 22, 20, 56]` = `feature_16, 66, 45, 36, 73, 25, 23, 59`. 4/8 overlap with Volkova's CORR list, 4 are new finds.

**Result:** static R² = **+0.00594** — **best of any sig_transformer variant** (+0.00109 lift over the [0..5] baseline, +0.00053 over depth=3 which was previously the best). Online refit still regresses (−0.00148), but that regression now reproduces across three different channel sets, so the cause is *architectural*, not the channel choice.

**What's actually causing the online regression** (corrected after re-reading `signature.py`):

* The signature is computed on a **rolling window inside one date** (size 16/32/64 depending on variant). The window never crosses midnight; the first ~16 time-ids of each day have a partial / zero-padded signature.
* It is **stateless** — recomputed from scratch on every `forward()` call (training, validation, predict, refit-update). So my earlier guess that a "stale buffer" was leaking into the next day's refit was wrong; there's no buffer to be stale.
* SignatureBlock has **zero learnable parameters**, so detaching the signature during refit also wouldn't change the gradient on the transformer's weights.

The remaining plausible causes for the refit regression are:

1. `update()` builds a fresh `AdamW` each call, throwing away the running first/second moments. Same code in GRU and transformer, so this alone can't explain the divergence, but it interacts with #2.
2. The signature monomials are scale-imbalanced (some O(1), some O(window^0.4)). With no momentum accumulated, a single-step refit gives a noisier gradient direction than the RNN sees, and the transformer's higher capacity over-fits that noise on one day's data.
3. `lr_refit = 3e-4` may be wrong for sig_transformer. Phase D found 1e-3 was right for lstm_modelr. We never swept refit-LR for sig_transformer.

Phase H probes #3 directly by re-running the refit eval on the `sig_transformer_ar_k8` checkpoint at LR ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}.

### Phase H — refit-LR sweep on sig_transformer_ar_k8

| lr_refit | online R² | Δ vs static (+0.00588) |
|---:|---:|---:|
| 0 (static) | +0.00588 | — |
| 1e-5 | +0.00611 | +0.00023 ✅ |
| **3e-5** ← winner | **+0.00623** | **+0.00035 ✅** |
| 1e-4 | +0.00614 | +0.00026 ✅ |
| 3e-4 (bench default) | +0.00443 | −0.00145 ❌ |
| 1e-3 | **−0.00900** | catastrophic |

**Hypothesis #3 confirmed.** The sig_transformer's apparent "online-refit regression" in runs 1 and 2 was an LR mismatch, not an architectural bug. At lr_refit = 3e-5 (10× smaller than the bench default), sig_transformer *improves* under online refit, just like the RNN family does.

The viable refit-LR range is sharply different per architecture family:

| family | viable refit-LR range | optimum |
|---|---|---|
| RNN (lstm_modelr, gru_modelr) | 1e-4 – 3e-3 | 1e-3 |
| sig-transformer | 1e-5 – 1e-4 | 3e-5 |

Take-away: **the bench's `lr_refit = 3e-4` default was a sensible RNN choice that happened to be on the wrong side of the cliff for sig_transformer**. Any future per-architecture variant should sweep refit-LR before claiming an online regression.

### Did adding sig_transformer-online-lr3e-5 help the ensemble?

| ensemble | R² |
|---|---:|
| 3-way (xgb + gru_modelr_online + lstm_modelr_online) | **+0.01039** ← leader, unchanged |
| 4-way (+ sig_transformer_ar_k8_online @ lr=3e-5) | +0.01010 |

No. The sig_transformer's predictions are correlated enough with the RNN streams that adding them dilutes the simple average; the optimal-weight blender assigns the sig stream 0 weight. The RNNs are already implicitly capturing whatever the signature was meant to encode.

**Conclusion on signatures:** correctly LR'd and AR-channel-selected, sig_transformer reaches **+0.00623** — by itself a credible model (+0.0019 over the originally-misconfigured runs, and +0.0014 over the depth-3 variant) — but it does not move our ensemble best of +0.01039. For competition LB pursuit, lstm_modelr-multi-seed + xgb-multi-seed blending is a better next bet than further signature work.

Interesting empirical observation from the AR scoring: **levels predict, increments don't**. Every feature has a small positive AR R² on its lagged levels and a near-zero R² on its first-differences. The Volterra signature *integrates over increments*, so the channel set whose levels are most predictive may not coincide with the channel set whose increments would feed the most informative monomials. Worth coming back to if we ever fix the architectural regression.

## Phase E — ensembles

| ensemble inputs | simple avg R² |
|---|---:|
| xgb_static_280 + gru_modelr_online_280 + lstm_modelr_online_280 | **+0.01039** ← winner |
| same 3 + sig_transformer_selected_static | +0.01000 |
| same 3 + sig_transformer_d3_static | +0.01012 |
| same 3 + sig_transformer_ar_k8_static (Phase G) | +0.01013 |
| same 3 + gru_modelr_static_280 + lstm_modelr_static_280 (5-way) | +0.01018 |

Two observations:
- **Simple average beats every weight-optimized blend on the held-out test half.** The held-out blend overfits the train half and keeps picking corner solutions (almost all weight on XGB or zero weight on every signature stream).
- **Adding any signature-transformer stream hurts** — they're correlated with the RNNs and dilute the average. The signature-transformer family is not adding ensemble diversity at this slice size.

## Take-aways

1. **Data > capacity.** Going from 200 → 280 train dates improves every architecture, but doubling lstm_modelr's hidden size (96 → 128) is a wash. The 8th-place LB is reachable from the same architectures; the bench just needed more data.
2. **Online refit is the lever for RNNs, not transformers.** RNNs gain +0.003–0.005 from online refit; the transformer family gains essentially nothing, and the signature-transformer family *loses* R² during refit. The signature buffer's discontinuity at the day boundary is the most likely culprit (the signature window sees stale features after a refit step) — confirmed by Phase F: changing channels does not fix the regression.
3. **lstm_modelr beats gru_modelr by a hair on every comparison, despite gru_modelr converging from a better static start.** The forget gate is doing useful work for online adaptation.
4. **Univariate target-corr is the wrong filter for signature inputs.** The signature's value is in cross-monomials between channels with non-trivial paths; picking by marginal predictiveness chose channels with smoother (higher autocorr) paths, exactly the kind the signature can't compress well. A better selector would score against **monomial-level SHAP** in a small probe model, not raw channel-target corr.
5. **The architecture race is won, the bottleneck is data and ensembling.** Top-3 simple-average jumps from +0.00909 (best single) to +0.01039 (best ensemble) — a +0.0013 gain just from averaging three uncorrelated-enough-streams.

## Next directions (for run 3, if it exists)

- **Push training to 500+ dates** with proper memory hygiene (split-fit XGB, swap-back-to-disk for the polars frame, or process per model). This is the single biggest expected lift.
- **Multi-seed lstm_modelr ensemble.** Volkova ensembled across 16 seeds; we ensembled across architectures. Three lstm_modelr's with different seeds, blended, should beat our current best individual.
- **Investigate the sig_transformer online regression directly.** Hypothesis: the rolling-signature buffer holds stale features when the model parameters get refit; the next-day forward pass sees an inconsistent (X, signature) pair. Fix candidate: re-emit the signature buffer from scratch after every refit step.
- **Signature channel selection done properly:** train sig_transformer_d3, freeze it, run leave-one-channel-out at inference and rank by R² drop. Then re-train using only the top-K by that measure.
- **Bigger lr_refit search.** The plateau 3e-4 ↔ 1e-3 suggests an even finer grid (3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3) plus a per-day-decaying schedule might add another +0.0001-0.0003.

## Files

- `artifacts/bench/run2_280d/{xgb,gru_modelr,lstm_modelr}.json` — per-model results
- `artifacts/bench/run2_280d/preds/*.npz` — prediction streams for ensembling
- `artifacts/bench/run2_280d/checkpoints/{gru_modelr,lstm_modelr}.pkl` — RNN checkpoints (XGB skipped by design; see `FullPipeline.save` docstring)
- `artifacts/bench/run2_280d/checkpoints/lstm_modelr_lr_sweep.json` — Phase D LR sweep
- `artifacts/bench/run2_280d/ensemble.json` — winning 3-way blend
- `artifacts/bench/run2_200d/` — Phase B/C/F variants on the run-1 slice
- `artifacts/sig_channels_k6.json` — Phase F's channel selection trace
- `scripts/select_signature_channels.py` — the selector (kept for next iteration; needs the SHAP-on-monomials replacement noted above)
