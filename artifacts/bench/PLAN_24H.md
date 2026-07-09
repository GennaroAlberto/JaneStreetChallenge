# 24-hour experiment plan — run 2

## Recap from run 1 (13.45 h on 200 train / 100 valid dates, 1399–1698)

| Model | params | fit (h) | R² static | R² online | Δ online |
|---|---:|---:|---:|---:|---:|
| **lstm_modelr** | 938 k | 3.23 | +0.00425 | **+0.00798** | +0.00374 |
| gru_modelr | 704 k | 2.94 | +0.00497 | +0.00749 | +0.00253 |
| gru | 120 k | 1.50 | +0.00246 | +0.00749 | +0.00502 |
| xgb | — | 0.13 | +0.00587 | — | — |
| sig_transformer | 266 k | 1.83 | +0.00485 | +0.00436 | **−0.00049** ⚠️ |
| transformer | 260 k | 1.58 | +0.00474 | +0.00500 | +0.00026 |
| mlp | 459 k | 1.12 | −0.00258 | — | — |

8th-place LB target: **+0.0112**. Our current best is 71 % of that on a fraction of the training data Volkova used.

## What we learned

1. **Online refit is the dominant lever for RNNs** (+0.005 on plain GRU). Transformers barely move.
2. **`sig_transformer` *loses* signal during refit** — needs investigation before we scale signature experiments.
3. **ModelR aux heads don't help on 200 train dates** — `gru` ties `gru_modelr` after refit. Suspect aux signals only pay off with more data.
4. **MLP is decisively negative** — temporal structure is doing real work.
5. **XGB is competitive and cheap** — perfect ensemble base.

## Hypotheses for this run

A. **More data flips the ModelR comparison.** With 350 train dates the aux heads start regularising correctly.
B. **The lstm_modelr is capacity-bound, not data-bound.** Hidden 128 (1.78× params) beats hidden 96.
C. **A richer Volterra signature improves static R²** but the regression at refit time is a bug to fix before we trust signature ablations.
D. **Online-refit LR ≠ 3e-4 optimum.** A small sweep {1e-4, 3e-4, 1e-3, 3e-3} on the best checkpoint costs ~minutes.
E. **Top-3 ensemble crosses 0.01.** XGB + lstm_modelr + gru_modelr predictions are partially uncorrelated.

## Crash-prevention strategy (16 GB Mac)

- Memory math: features are already `Float32` in polars. 350 train dates ≈ 13 M rows × 125 cols × 4 B = **6.5 GB** numpy + **7.5 GB** polars frame = peak ≈ 14 GB → safe with hygiene; the previous OOM was at 500 dates (≈ 19 GB peak).
- Hygiene to add in `bench.py`:
  1. After `pipe.fit()` returns, immediately `del df_tr; gc.collect()`.
  2. Wrap each model's `run_one_model` in a subprocess so the OS reclaims the address space between runs (single fork, output piped back as JSON). Belt-and-braces; lets one model's leak not kill the next.
- Hard cap at **350 train dates**. Anything bigger needs a chunked-fit path I'm not building today.
- Keep `--skip-existing` resume semantics so any crash drops us back at the last completed model.

## Experiment lineup (≈ 21 h)

| # | Phase | What | est. cost |
|---|---|---|---:|
| 0a | infra | Add checkpoint save/load to pipeline (so phases D & E are cheap) | 30 min |
| 0b | infra | Lean fit_data path: free `df_tr` + gc between models; subprocess per model | 30 min |
| 0c | debug | Investigate `sig_transformer` online refit regression (suspect: signature buffer not advanced day-over-day; refit sees stale features) | 30 min |
| A | bigger data | `lstm_modelr` [96,96,96] × 4 aux, 350/100, 8 epochs, patience 3 | 7 h |
| B | bigger model | `lstm_modelr` [128,128,128] × 4 aux, 200/100, 10 epochs | 5 h |
| C₁ | signature | `sig_transformer` depth=3, window=16, 200/100, 8 epochs | 2 h |
| C₂ | signature | `sig_transformer` depth=2, window=64, 200/100, 8 epochs | 2 h |
| D | LR sweep | Reload lstm_modelr ckpt, evaluate online refit at LR ∈ {1e-4, 3e-4, 1e-3, 3e-3} | 30 min |
| E | ensemble | Blend xgb + lstm_modelr-online + gru_modelr-online predictions (simple avg + ridge) | 30 min |
| F | report | Update `results.md` and write per-experiment commentary | 30 min |

**Total: ≈ 19.5 h plan + ≈ 4 h buffer for crashes/retries** within the 24 h envelope.

## Order of operations

Phase 0a → 0b → 0c must finish before A so the heavy training runs benefit from the new infra. After that the order is C₁ → C₂ → B → A (longest last, so partial budget overrun still yields the signature results; D and E come for free at the end from checkpoints).

## What I'll report when done

1. New leaderboard row for each experiment (static + online R²).
2. **Did hypothesis X hold?** explicit yes/no per A–E.
3. Updated take-aways and recommended next direction for run 3.

---

*This plan is conservative. If 0a/0b come in faster than expected I'll bolt on a `gru` [128] run (~3 h) — it would directly test whether the parameter gap between `gru` and `gru_modelr` was the source of the static-R² gap.*
