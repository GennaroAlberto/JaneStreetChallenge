#!/bin/zsh
# Arch-wave rerun after the on_batch reshape fix (dataset.py): the fit-time
# validation was scrambled — leaked future timesteps into the xsec attention
# (val R² 0.15 vs ceiling 0.014) and degraded plain-RNN epoch selection.
# Retrain the two xsec members (selection was random under the leak) and
# spreadaux (selection was degraded). ae_mlp is arrangement-immune — stands.
set -e
cd "$(dirname "$0")/.."
OUT=artifacts/bench/arch_wave2
MM=artifacts/precomputed/pool700_lags
PQ=artifacts/colab/js_pool_1318_1698_r9
WIN="--resample-mode window --train-lo 1318 --train-hi 1597 --valid-lo 1599 --valid-hi 1698"
mkdir -p $OUT

echo "=== 1/3 lstm_modelr_xsec  $(date) ==="
uv run python scripts/train_from_memmap.py --data $MM ${=WIN} \
    --model lstm_modelr_xsec --seed 42 --device mps --tag xsec_lstm_s42 --out $OUT

echo "=== 2/3 lstm_modelr spreadaux  $(date) ==="
uv run python scripts/train_from_memmap.py --data $PQ ${=WIN} \
    --model lstm_modelr --seed 42 --device mps \
    --aux-targets responder_0,responder_2,responder_7,responder_8 \
    --tag lstm_spreadaux_s42 --out $OUT

echo "=== 3/3 gru_modelr_xsec  $(date) ==="
uv run python scripts/train_from_memmap.py --data $MM ${=WIN} \
    --model gru_modelr_xsec --seed 42 --device mps --tag xsec_gru_s42 --out $OUT

echo "=== arch rerun done  $(date) ==="
