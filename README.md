# Jane Street Real-Time Market Data Forecasting

End-to-end pipeline for the [Jane Street 2024 Kaggle competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/overview). Two goals:

1. **Faithfully replicate the 8th-place (gold) solution** by Evgenia Volkova — a 4-branch GRU with auxiliary-responder heads, trained per-date with online refit during inference.
2. **Beat it** with a Volterra-signature-augmented Transformer (and any other architecture you wire in), under the same evaluation protocol.

The codebase is a Python package (no notebooks), set up with `uv`. Every architecture plugs in through a tiny [`BaseModel`](src/janestreet/models/base.py) interface — GRU, LSTM, MLP, Transformer, XGBoost, signature-Transformer are all already available.

## Quick start

```bash
# install
uv sync

# profile the dataset after Volkova-style feature engineering
uv run js profile --min-date 1500 --max-date 1530

# replicate the 8th-place GRU on a tiny date slice (CPU smoke test)
uv run js cv --model gru_modelr \
    --min-date 1690 --max-date 1698 --n-splits 1 --test-size 3 \
    --model-kwargs '{"hidden_sizes":[64,64,64],"dropout_rates":[0.1,0.1,0.1],"hidden_sizes_linear":[],"dropout_rates_linear":[],"epochs":3,"lr_refit":3e-4,"device":"cpu","num_aux":4}'

# Volterra-signature Transformer on the same slice
uv run js cv --model sig_transformer \
    --min-date 1690 --max-date 1698 --n-splits 1 --test-size 3 \
    --model-kwargs '{"d_model":128,"n_heads":4,"n_layers":3,"epochs":3,"device":"cpu","signature_depth":2,"signature_hurst":0.1,"signature_window":32,"signature_channels":[0,1,2,3,4,5]}'

# XGBoost non-DL baseline (fast)
uv run js cv --model xgb \
    --min-date 1500 --max-date 1698 --n-splits 2 --test-size 100 \
    --model-kwargs '{"n_estimators":1000,"max_depth":6,"learning_rate":0.05,"n_jobs":1}'
```

## Layout

```
src/janestreet/
├── config.py             # paths, column names, defaults — one Cfg dataclass
├── data/
│   ├── ingest.py         # lazy polars scan of partitioned parquet
│   ├── features.py       # Volkova feature engineering + optional extras
│   ├── scaler.py         # online-safe standardizer + outlier clipper
│   └── dataset.py        # PyTorch (D=symbols, T=time, K=features) per-date Dataset
├── models/
│   ├── base.py           # BaseModel ABC: fit/predict/update
│   ├── recurrent.py      # GRU/LSTM, incl. Volkova ModelR (4 aux branches)
│   ├── transformer.py    # Causal Transformer + signature-augmented variant
│   ├── signature.py      # Truncated path signature + Volterra-weighted variant
│   ├── mlp.py            # MLP baseline
│   ├── gbm.py            # XGBoost adapter
│   └── __init__.py       # registry — REGISTRY maps "gru_modelr"/"transformer"/etc → class
├── training/
│   ├── loss.py           # WeightedR2Loss
│   ├── metrics.py        # r2_weighted (numpy + torch)
│   └── cv.py             # TimeSeriesDateSplit with optional gap
├── pipeline.py           # FullPipeline + run_cv (online-refit eval loop)
└── cli.py                # `js cv` / `js profile`
```

## What's faithful to Volkova

| Choice | Source | Done |
| --- | --- | --- |
| Skip `date_id < 700`. | `config.py` | ✅ `Cfg.min_date_id = 700` |
| 79 features minus `feature_09/10/11`. | `data_processor.py` | ✅ `FeatureBuilder` |
| 16 high-corr cols → rolling mean/std @ T=1000 per symbol. | same | ✅ |
| Market average per (date, time) on the 16 cols. | same | ✅ |
| `feature_time_id = time_id`. | same | ✅ |
| Synthetic `responder_9` = `r8 + r8.shift(-4)`; `responder_10` = `r6 + shift(-20) + shift(-40)`. | same | ✅ `FeatureBuilder._add_synthetic_responders` |
| 4-branch GRU, each predicting an aux responder; linear combiner → `y`. | `models/nn.py` | ✅ `ModelR` |
| Loss = WeightedR² on `y` + on each aux responder (using last 4 cols of "other responders"). | `models/nn.py` | ✅ `RecurrentModel._loss_for_batch` |
| AdamW(lr=1e-3, wd=1e-2), batch_size=1 (one date), grad-clip 1.0, early stop. | same | ✅ |
| Online refit during validation/inference with `lr_refit=3e-4` on responder_6 only. | same | ✅ `update()` |
| TimeSeriesSplit, 2 folds × 200-date validation, optional 200-day gap. | `pipeline.py` | ✅ `TimeSeriesDateSplit` |
| Day-by-day predict, refit on previous day before predicting the next. | same | ✅ `run_cv` |

## Where to attack Volkova

1. **Volterra signature features** (`models/signature.py`). The standard truncated path signature gives polynomial features over all iterated integrals of an augmented path. The Volterra-weighted variant multiplies each increment by `(t - s)^{H − 1/2}` so the signature reflects a rough kernel — much more compact than learning the same shape from a GRU's hidden state. Channels and depth are wired through `signature_channels`, `signature_depth`, `signature_hurst`. See `SignatureBlock.forward`.
2. **Causal Transformer over the per-date sequence** (`models/transformer.py`). Same shape as the recurrent baseline (`(D, T, K)` per date), but the attention layers see the whole window non-recurrently. The signature pre-block makes its first-layer attention much more sample-efficient.
3. **Ensembling**: not packaged yet, but trivial — train several seeds / archs and average predictions. See `run_full_2.py` / `run_full_3.py` in the upstream repo for the 6-model recipe that got 0.0112 LB.

## Notes / pitfalls

- Volkova's `ModelR` requires each `(symbol, date)` group to have *exactly* `T = n_times = 968` rows. The data has this invariant from `date_id ≥ 700` onward; we set `cfg.min_date_id = 700` by default for that reason.
- On macOS the bundled `libomp` can deadlock XGBoost with multi-threading — `XGBPerHorizon` defaults to `n_jobs=1`. Raise it on Linux.
- Heavy training runs (full ~1000 dates) need a CUDA box: the writeup quotes ~100 GB RAM and 12 GB GPU. On a Mac laptop, expect to work on date slices.
- Path-signature cost is `O(T · window · depth · K^depth)` per batch — keep `signature_channels` short (4–8) and `signature_depth ≤ 3`.

## References

- Writeup: [Private LB 8th solution](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/writeups/evgeniia-grigoreva-private-lb-8th-solution)
- Code (upstream): [evgeniavolkova/kagglejanestreet](https://github.com/evgeniavolkova/kagglejanestreet)
- Submission notebook: [public-6th-place](https://www.kaggle.com/code/eivolkova/public-6th-place?scriptVersionId=217330222)
