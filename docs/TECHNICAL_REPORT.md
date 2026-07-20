# Jane Street Real-Time Market Data Forecasting — Technical Report

*Living document. Records the problem framing, the pipeline architecture, the
model zoo, results to date, and the roadmap of things we will try.*

Last updated: 2026-07-08

---

## 1. Problem & the key insight

We predict `responder_6` (a weighted-R² competition metric) from 79 anonymized
features, per `(symbol_id, date_id, time_id)`. Target/benchmark: the 8th-place
**public** score was +0.0112; the private LB reads 8th = **+0.010434** and
1st ("ms capital") = **+0.013890** — see `FEATURE_RESEARCH.md` for the full
recalibration and the feature-atlas findings.

**What the responders actually are** (from Kaggle discussion #555562,
"Reverse Engineering the Responders", and independently consistent with our own
tag analysis):

- The responders are **forward-shifted moving averages** of a small number of
  underlying signals. ACF cutoffs pin the windows: `responder_6` = SMA-**20**,
  `responder_7` = SMA-**120**, `responder_8` = SMA-**4** of one signal
  (exchange A). `{3,4,5}` are the same three windows of a second signal
  (exchange B); `{0,1,2}` are the A−B differences. `responders.csv` tags
  confirm this grouping.
- The target is **shifted by ~20**: the `responder_6` at row *t* is effectively
  the SMA-20 over *t … t+20*. Features at *t* predict it with R²≈0.5 at lag 20.
- **Implication (sobering):** the target is a forward MA of near-white noise, so
  `E[future | present] ≈ 0`. The predictable component is genuinely tiny — this
  is why every score, ours included, plateaus near +0.010. **Variance reduction
  (bagging, ensembling, online refit) is therefore worth disproportionately more
  than chasing a better single model.**

**Deployability constraint that shapes everything:** the inference gateway
streams one `time_id` at a time and provides responders only as **daily lags**
(previous day's full responder path). So (a) features used at time *t* may only
look back to ≤ *t* within the day, and (b) the target's own history is available
only as yesterday's lagged responders — never within-day.

---

## 2. Data & leakage handling

- **Date range:** `n_times` stabilizes at 968 from `date_id = 677`; we use the
  8th-place floor of **700**. Pool for training = dates 700–1598; validation
  tail = 1599–1698 (100 dates). Clean region ≈ 999 dates, 34.8 M rows.
- **Feature engineering** (`FeatureBuilder`): raw features; per-symbol rolling
  mean/std (window 1000, **trailing** → backward-only); market average per
  `(date,time)` (**contemporaneous**, delivered together at inference); two
  synthetic aux responders; optional lagged responders + responder-signal
  features (below).
- **Leakage audited end-to-end:**
  - Model inputs are all trailing or contemporaneous — **no feature looks
    forward**.
  - Aux targets (`responder_9/10`) use forward `shift(-k)` (by design — they're
    targets). Their only cross-date reach into validation is removed by a
    **1-date embargo** between the training pool and the validation tail
    (max shift 40 « one day = 968 steps).
  - Standardization stats are fit on **train dates only**; validation uses
    frozen stats; the online scaler adapts on **past** validation days only.
  - Lagged responders are strictly **previous-day** (a date+1 self-join),
    verified backward-only.
- **Precision:** features stored `Float16` (halves footprint; roundtrip error
  ~1e-4, below the noise floor).

---

## 3. Pipeline architecture (the modular structure)

Three composable stages; each stage is a standalone script and an experiment is
one YAML file.

```
 precompute_dataset.py        train_from_memmap.py            run_experiment.py
 ────────────────────         ───────────────────             ─────────────────
 feature-engineer the    →    train ONE model on a       →    declare a bag in
 whole pool ONCE;             resample of the pool            YAML: pool + models
 fit preprocessor on          (subsample / bootstrap /        + seeds + resample;
 train dates; write a         window); eval on the tail;      trains each member,
 Float16 memmap + a           dump _static/_online .npz       blends the .npz via
 per-date row manifest        checkpoints                     ensemble_blend.py
```

- **`scripts/precompute_dataset.py`** — one memmap serves every model. Peak RAM
  ~5 GB; ~7–11 min. Flags pick the date range, the leakage boundary
  (`--train-until`), and optional `--lagged-responders`.
- **`scripts/train_from_memmap.py`** — trains a single bag member. Reads only
  its resampled dates from the memmap (so a 0.6-frac model uses ~9 GB, fits a
  16 GB Mac). Resample modes: `subsample` (frac without replacement — default),
  `bootstrap` (with replacement + synthetic group ids), `window` (contiguous).
  Supports warm-start and per-epoch checkpointing.
- **`scripts/run_experiment.py`** — the "pick dates, pick models, mix them"
  layer. One YAML (see `experiments/example_bag.yaml`) lists members
  (model × seeds × resample); the runner trains each sequentially and blends
  them. Resumable (skips completed members).
- **`scripts/ensemble_blend.py`** — simple average + held-out-fit convex blend.
- **`scripts/stack_residual.py`** — non-linear (XGB) stacker over the streams.
- **Colab** (`notebooks/colab_train_500d.ipynb`) — GPU path for the full pool,
  with cold-start / warm-start / resume-from-Drive.

**"Train on batches of the right dates, mix the models you want"** = edit the
YAML `members:` list and `data:` pool, then `run_experiment.py`.

---

## 4. Model zoo (12 registered models)

| name | family | notes |
|---|---|---|
| `xgb` | gradient boosting | fast non-DL baseline; static only |
| `mlp` | feed-forward | non-sequence baseline |
| `gru`, `lstm` | single-branch RNN | light sequence models, online refit |
| `gru_modelr`, `lstm_modelr` | 4-branch RNN + aux heads | Volkova replica; **best single model** |
| `mlp_sig` | MLP + Volterra signature | stateless-in-time, signature carries temporal info |
| `transformer` | causal attention | vanilla baseline |
| `sig_transformer` | signature + attention | |
| `mamba` | selective SSM | 3rd architecture family (linear-time) |
| `itransformer` | inverted attention (causal) | iTransformer-inspired: attention across features |
| `timexer` | endo/exo split (causal) | TimeXer-inspired: yesterday's responders (endogenous, patch+global token) × today's features (exogenous, causal) via cross-attention |

All sequence models share one training/eval/online-refit loop; adding a model =
one class + one registry line.

---

## 5. Results to date

**Weighted-R² on the 1599–1698 validation tail.** 8th-place LB = +0.0112.

### Single models (run-2, 280 train dates unless noted)
| model | static R² | online R² |
|---|---:|---:|
| lstm_modelr (lr_refit=1e-3) | +0.00637 | **+0.00920** |
| gru_modelr | +0.00763 | +0.00893 |
| xgb (280d) | +0.00684 | — |
| mlp_logsig_min (Phase J) | +0.00614 | +0.00657 |
| sig_transformer_ar_k8 @ lr=3e-5 | +0.00594 | +0.00623 |
| mlp | −0.00258 | — |

### Best ensemble so far
- **3-way simple average** (xgb + gru_modelr-online + lstm_modelr-online) →
  **+0.01039** ≈ **93 % of the 8th-place LB**.
- Non-linear stacking and adding signature/transformer streams did **not** beat
  the simple 3-way — those streams are too correlated with the RNNs.

### Key findings
1. **Data > capacity.** 200→280 train dates lifted every model; doubling
   lstm_modelr width did not.
2. **Online refit is the RNN lever** (+0.003–0.005); transformers barely move;
   signature-transformers need a much smaller refit LR (3e-5 vs 1e-3).
3. **Ensemble diversity is the wall.** Every signature/transformer stream got
   weight 0 from the blender → motivates *temporal bagging* (different date
   resamples of the same architecture) and genuinely different inputs (lagged
   responders, TimeXer).
4. **Signatures:** AR(7)-selected channels + minimal (Lyndon) log-signature is
   the best signature config, but doesn't move the ensemble.

### Pending (this session)
- Responder-signal-feature XGB ablation (base vs +lags vs +lags+signal) — *running*.
- TimeXer + lagged-responder bag members from `pool700_lags` — *first member trained; more queued*.

---

## 6. Roadmap — things we will try

Ordered by expected value given the "variance reduction > single model" insight.

1. **Temporal bagging at scale.** Many resamples (subsample frac ~0.6) of
   `lstm_modelr` / `gru_modelr` + a few seeds each, simple-averaged. This is the
   single most-aligned lever with the problem's noise ceiling. *(runner ready.)*
2. **Lagged-responder features for all models.** The reverse-engineering says
   yesterday's responder path is the dominant usable signal. Ablation in
   progress; if positive, rebuild the bag on `pool700_lags`.
3. **TimeXer head-to-head.** Does the explicit endogenous(yesterday)/
   exogenous(today) split beat feeding lags as flat features? Bag several seeds.
4. **Responder-signal features** (multi-scale momentum + venue spread). Cheap;
   keep if the ablation shows marginal lift beyond raw lags.
5. **Full 700-date pool on GPU (Colab).** Push training data to the max clean
   range; expected per-model lift ~+0.001–0.002.
6. **Multi-seed single-architecture ensembles** (Volkova ran 16 seeds). Cheaper
   diversity than new architectures, and historically effective here.
7. **Refit-LR schedules** (per-architecture; the plateau is 3e-4–1e-3 for RNNs,
   3e-5 for sig-transformers) and a small per-day decay.
8. **Tag-group feature engineering** (from `features.csv` tag structure):
   matched-triple `{tag_12,13,14}` level+slope, tag-group cross-feature factors.
   Lower priority; validate with an XGB ablation first.
9. **Mamba / itransformer as ensemble diversity** — only worth bagging if their
   streams decorrelate from the RNNs (test with a 2-seed probe first).

### Explicitly de-prioritized
- Bigger single models (capacity is not the bottleneck).
- Non-linear stacking (did not beat simple average).
- More signature-channel engineering (diminishing returns; doesn't move the
  ensemble).

---

## 7. Reproduce

```bash
# 1. precompute the pool (once)
uv run python scripts/precompute_dataset.py --min-date 700 --max-date 1698 \
    --train-until 1598 --lagged-responders 0,1,2,3,4,5,6,7,8 \
    --out artifacts/precomputed/pool700_lags

# 2. declare + run a bag
uv run python scripts/run_experiment.py --config experiments/example_bag.yaml

# 3. inspect artifacts/bench/<exp>/ensemble.json
```
