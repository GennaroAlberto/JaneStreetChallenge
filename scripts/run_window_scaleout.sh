#!/bin/zsh
# Window scale-out after the emphatic 500d verdict (XGB base +0.00686 →
# +0.00841; lstm64 280d→400d +0.00822 → +0.00934). Sequential:
#   1. deployable 500d XGB stream: rs2 train, FULL-ROW valid preds (blend)
#   2. lstm64 400d seeds 1, 2 (upgrade the cheap bag)
#   3. window knee probe: 700d vs 500d at rs4 (comparable pair)
#   4. Kaggle pool export: dates 998-1698, all responders, zipped
set -e
cd "$(dirname "$0")/.."
OUT500=artifacts/bench/pool_rebuild_500
LAT500=artifacts/bench/ae_lab_sup/latents_500.npz
LAT700=artifacts/bench/ae_lab_sup/latents_700.npz
OUT700=artifacts/bench/pool_rebuild_700

echo "=== 1 XGB 500d deployable (full-row valid)  $(date) ==="
uv run python scripts/rebuild_pool_ablation.py \
    --train-lo 1098 --train-hi 1598 --row-step 2 --valid-row-step 1 \
    --with-chain --with-ranks --latents $LAT500 \
    --chain-cache $OUT500/chain_cache.npz \
    --variants all_chain_ranks \
    --out $OUT500/deploy

echo "=== 2 lstm64 400d seeds 1,2  $(date) ==="
for SEED in 1 2; do
    uv run python scripts/train_cheap_member.py --seed $SEED \
        --train-lo 1198 --train-hi 1597 \
        --out $OUT500/preds400
done

echo "=== 3 window knee probe: 500d vs 700d at rs4  $(date) ==="
[ -f $LAT700 ] || uv run python scripts/encode_ae_latents.py \
    --train-lo 898 --train-hi 1598 --out $LAT700
uv run python scripts/rebuild_pool_ablation.py \
    --train-lo 1098 --train-hi 1598 --row-step 4 \
    --with-chain --with-ranks --latents $LAT500 \
    --chain-cache $OUT500/chain_cache.npz \
    --variants all_chain_ranks \
    --out $OUT500/rs4_ref
mkdir -p $OUT700
uv run python scripts/rebuild_pool_ablation.py \
    --train-lo 898 --train-hi 1598 --row-step 4 \
    --with-chain --with-ranks --latents $LAT700 \
    --chain-cache $OUT700/chain_cache.npz \
    --variants all_chain_ranks \
    --out $OUT700

echo "=== 4 Kaggle pool export 998-1698  $(date) ==="
uv run python scripts/export_parquet.py \
    --memmap artifacts/precomputed/pool700_lags \
    --out artifacts/colab/js_pool_998_1698_r9 \
    --min-date 998 --max-date 1698 --include-responders
cd artifacts/colab && zip -qr js_pool_998_1698_r9.zip js_pool_998_1698_r9 && cd ../..
ls -lh artifacts/colab/js_pool_998_1698_r9.zip

echo "=== window scale-out done  $(date) ==="
