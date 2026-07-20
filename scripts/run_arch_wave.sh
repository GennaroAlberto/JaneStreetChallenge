#!/bin/zsh
# Architecture wave, run locally instead of the Kaggle batch — the four
# genuinely-new members at the 280d production window (train 1318–1597).
# Waits for the running 280d checkpoint program to release the machine,
# then trains sequentially on MPS, cheapest verdict first:
#   1. ae_mlp        — row-wise 2021-JS-winner family (diversity thesis)
#   2. gru_modelr_xsec  — recurrence over time + attention across symbols
#   3. lstm_modelr spreadaux — aux = r0/r2 venue spreads (20x aux SNR)
#   4. lstm_modelr_xsec
# Full-size seed bag (lstm s1-3) deferred: the cheap lstm64 bag covers the
# seed axis for now.
set -e
cd "$(dirname "$0")/.."
DRIVER_LOG=artifacts/bench/pool_rebuild_280_driver.log
OUT=artifacts/bench/arch_wave
MM=artifacts/precomputed/pool700_lags
PQ=artifacts/colab/js_pool_1318_1698_r9
WIN="--resample-mode window --train-lo 1318 --train-hi 1597 --valid-lo 1599 --valid-hi 1698"
mkdir -p $OUT

echo "waiting for the 280d checkpoint program... $(date)"
while ! grep -q "all phases done" $DRIVER_LOG 2>/dev/null; do sleep 120; done
echo "machine free — starting arch wave  $(date)"

echo "=== 1/4 ae_mlp  $(date) ==="
uv run python scripts/train_from_memmap.py --data $MM ${=WIN} \
    --model ae_mlp --seed 42 --device mps --tag aemlp_s42 --out $OUT

echo "=== 2/4 gru_modelr_xsec  $(date) ==="
uv run python scripts/train_from_memmap.py --data $MM ${=WIN} \
    --model gru_modelr_xsec --seed 42 --device mps --tag xsec_gru_s42 --out $OUT

echo "=== 3/4 lstm_modelr spreadaux  $(date) ==="
uv run python scripts/train_from_memmap.py --data $PQ ${=WIN} \
    --model lstm_modelr --seed 42 --device mps \
    --aux-targets responder_0,responder_2,responder_7,responder_8 \
    --tag lstm_spreadaux_s42 --out $OUT

echo "=== 4/4 lstm_modelr_xsec  $(date) ==="
uv run python scripts/train_from_memmap.py --data $MM ${=WIN} \
    --model lstm_modelr_xsec --seed 42 --device mps --tag xsec_lstm_s42 --out $OUT

echo "=== arch wave done  $(date) ==="
