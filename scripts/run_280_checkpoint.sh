#!/bin/zsh
# 280-date local checkpoint program — sequential (one heavy job at a time,
# 16 GB budget). Phases:
#   1. AE latents re-encode for the 280d window (pure inference)
#   2. pool v3 ladder at row-step 2 (base / all_chain / all_chain_ranks),
#      computing + caching the full-row chain block
#   3. full-row all_chain_ranks fit → the deployable XGB stream
#   4. cheap LSTM members [64,64], seeds 42/1/2 (MPS, memmap-order preds)
# Blend + calibration + vol-scale run interactively afterwards.
set -e
cd "$(dirname "$0")/.."
OUT=artifacts/bench/pool_rebuild_280
mkdir -p $OUT
LAT=artifacts/bench/ae_lab_sup/latents_280.npz

echo "=== phase 1: AE latents 280d  $(date) ==="
[ -f $LAT ] || uv run python scripts/encode_ae_latents.py \
    --train-lo 1318 --train-hi 1598 --out $LAT

echo "=== phase 2: rs2 ladder  $(date) ==="
uv run python scripts/rebuild_pool_ablation.py \
    --train-lo 1318 --train-hi 1598 --row-step 2 \
    --with-chain --with-ranks --latents $LAT \
    --chain-cache $OUT/chain_cache.npz \
    --variants base,all_chain,all_chain_ranks \
    --out ${OUT}_rs2

echo "=== phase 3: full-row deployable  $(date) ==="
uv run python scripts/rebuild_pool_ablation.py \
    --train-lo 1318 --train-hi 1598 --row-step 1 \
    --with-chain --with-ranks --latents $LAT \
    --chain-cache $OUT/chain_cache.npz \
    --variants all_chain_ranks \
    --out $OUT

echo "=== phase 4: cheap LSTM bag  $(date) ==="
for SEED in 42 1 2; do
    echo "--- lstm64 seed $SEED  $(date) ---"
    uv run python scripts/train_cheap_member.py --seed $SEED \
        --out $OUT/preds
done
echo "=== all phases done  $(date) ==="
