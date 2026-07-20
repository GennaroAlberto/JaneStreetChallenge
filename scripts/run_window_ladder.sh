#!/bin/zsh
# The data axis: does the 280d production window undersell the models?
# Own evidence says maybe — the May window sweep peaked AT its 400d
# boundary and never probed beyond; the "old data hurts" verdict came
# from the curriculum scheme, not a plain longer window.
# Waits for the xsec deep-dive, then:
#   0. quick rewalks at the discovered 3e-4 refit lr (warm ckpts + plain gru)
#   1. AE latents for the 500d window (pure inference)
#   2. XGB pool ladder at 500d rs2 (vs 280d rs2: base .00686 / acr .00842)
#   3. cheap lstm64 member at 400d (RAM ceiling for full-row RNN locally)
set -e
cd "$(dirname "$0")/.."
DD=artifacts/bench/arch_wave2
OUT500=artifacts/bench/pool_rebuild_500
LAT500=artifacts/bench/ae_lab_sup/latents_500.npz

echo "waiting for xsec deep-dive... $(date)"
while ! grep -q "xsec deep-dive done" $DD/deepdive_driver.log 2>/dev/null; do sleep 120; done
echo "machine free  $(date)"

echo "=== 0a rewalk warm-lstm @ 3e-4  $(date) ==="
uv run python scripts/rewalk_checkpoint.py --ckpt $DD/checkpoints/xsec_lstm_warm.pkl \
    --lr-refit 3e-4 --tag xsec_lstm_warm_r34 --out $DD/rewalks
echo "=== 0b rewalk warm-gru @ 3e-4  $(date) ==="
uv run python scripts/rewalk_checkpoint.py --ckpt $DD/checkpoints/xsec_gru_warm.pkl \
    --lr-refit 3e-4 --tag xsec_gru_warm_r34 --out $DD/rewalks
echo "=== 0c rewalk plain gru @ 3e-4  $(date) ==="
uv run python scripts/rewalk_checkpoint.py \
    --ckpt artifacts/bench/regen_ensemble/checkpoints/gru_modelr_gru_modelr.pkl \
    --lr-refit 3e-4 --tag gru_plain_r34 --out $DD/rewalks

echo "=== 1 AE latents 500d  $(date) ==="
[ -f $LAT500 ] || uv run python scripts/encode_ae_latents.py \
    --train-lo 1098 --train-hi 1598 --out $LAT500

echo "=== 2 XGB ladder 500d rs2  $(date) ==="
mkdir -p $OUT500
uv run python scripts/rebuild_pool_ablation.py \
    --train-lo 1098 --train-hi 1598 --row-step 2 \
    --with-chain --with-ranks --latents $LAT500 \
    --chain-cache $OUT500/chain_cache.npz \
    --variants base,all_chain_ranks \
    --out $OUT500

echo "=== 3 lstm64 @ 400d  $(date) ==="
uv run python scripts/train_cheap_member.py --seed 42 \
    --train-lo 1198 --train-hi 1597 \
    --out artifacts/bench/pool_rebuild_500/preds400

echo "=== window ladder done  $(date) ==="
