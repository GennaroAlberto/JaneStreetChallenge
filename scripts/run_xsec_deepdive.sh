#!/bin/zsh
# Xsec transformer deep-dive: (a) refit-lr sweep on the admitted gru_xsec
# checkpoint (walk-only — the standing rule's sweep, plus an attention-
# frozen refit variant); (b) warm-start transplants: plain twin weights +
# zero-init attention, fine-tuned with the attention at its own lr.
set -e
cd "$(dirname "$0")/.."
OUT=artifacts/bench/arch_wave2
CKPT=$OUT/checkpoints/gru_modelr_xsec_xsec_gru_s42.pkl

echo "=== 1/5 rewalk lr_refit=3e-4  $(date) ==="
uv run python scripts/rewalk_checkpoint.py --ckpt $CKPT \
    --lr-refit 3e-4 --tag xsec_gru_r3em4 --out $OUT/rewalks

echo "=== 2/5 rewalk lr_refit=3e-3  $(date) ==="
uv run python scripts/rewalk_checkpoint.py --ckpt $CKPT \
    --lr-refit 3e-3 --tag xsec_gru_r3em3 --out $OUT/rewalks

echo "=== 3/5 rewalk lr_refit=1e-3 frozen-attn  $(date) ==="
uv run python scripts/rewalk_checkpoint.py --ckpt $CKPT \
    --lr-refit 1e-3 --freeze-attn-refit --tag xsec_gru_r1em3frz --out $OUT/rewalks

echo "=== 4/5 warm-start lstm_xsec  $(date) ==="
uv run python scripts/warmstart_xsec.py --family lstm --out $OUT

echo "=== 5/5 warm-start gru_xsec  $(date) ==="
uv run python scripts/warmstart_xsec.py --family gru --out $OUT

echo "=== xsec deep-dive done  $(date) ==="
