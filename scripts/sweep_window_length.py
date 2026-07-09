"""Recency sweep: how much does older training data help (or hurt)?

Trains the same model on windows of increasing length that all END at the same
recent date (default 1597, the embargoed pool edge), and reports online/static
weighted-R² vs window length. This quantifies the recency knee suggested by the
curriculum experiment (old data gave negative static R², online refit
recovered): if the curve is flat or decreasing past some length, older data is
neutral-to-harmful and we should train on a short recent window.

Optionally combines with exponential recency weighting (``--decay-halflife``)
to see whether soft down-weighting of old data beats a hard cutoff.

Each window is trained by shelling out to ``scripts/train_from_memmap.py`` in
``window`` mode (one process each → clean memory between runs). Uses the fast
single-branch ``gru`` by default so the whole sweep finishes in a couple of
hours; pass ``--model gru_modelr`` for the stronger (slower) variant.

Usage
-----

    uv run python scripts/sweep_window_length.py \\
        --data artifacts/precomputed/pool700_lags --model gru \\
        --lengths 50,100,200,400,898 --hi 1597 \\
        --out artifacts/bench/window_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "scripts" / "train_from_memmap.py"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, default="gru")
    p.add_argument("--lengths", type=str, default="50,100,200,400,898",
                   help="comma-separated window lengths (in dates) ending at --hi")
    p.add_argument("--hi", type=int, default=1597, help="last training date (embargoed edge)")
    p.add_argument("--pool-lo", type=int, default=700)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--decay-halflife", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    # Passed through to the trainer (GPU / longer runs).
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    def log(msg: str) -> None:
        sys.stdout.write(msg + "\n"); sys.stdout.flush()
        with (out / "sweep_log.txt").open("a") as f:
            f.write(msg + "\n")

    log(f"=== window-length recency sweep: model={args.model} lengths={lengths} "
        f"ending at {args.hi}  decay_halflife={args.decay_halflife} ===")

    rows = []
    for L in lengths:
        lo = max(args.pool_lo, args.hi - L + 1)
        tag = f"win{L}" + (f"_hl{int(args.decay_halflife)}" if args.decay_halflife else "")
        res_path = out / f"{args.model}_{tag}.json"
        if not res_path.exists():
            cmd = [
                "uv", "run", "python", "-u", str(TRAINER),
                "--data", args.data, "--model", args.model,
                "--resample-mode", "window", "--train-lo", str(lo), "--train-hi", str(args.hi),
                "--pool-lo", str(args.pool_lo),
                "--valid-lo", str(args.valid_lo), "--valid-hi", str(args.valid_hi),
                "--seed", str(args.seed), "--tag", tag, "--out", str(out),
            ]
            if args.decay_halflife:
                cmd += ["--decay-halflife", str(args.decay_halflife)]
            for flag, val in (("--device", args.device), ("--epochs", args.epochs),
                              ("--batch-size", args.batch_size), ("--patience", args.patience)):
                if val is not None:
                    cmd += [flag, str(val)]
            log(f"\n--- window length {L} (dates {lo}..{args.hi}) ---")
            subprocess.run(cmd, env=env, check=False)  # noqa: S603
        if res_path.exists():
            r = json.loads(res_path.read_text())
            rows.append({"length": L, "lo": lo, "hi": args.hi,
                         "r2_static": r.get("r2_static"), "r2_online": r.get("r2_online")})
            log(f"  length {L:>4} (dates {lo}..{args.hi}): "
                f"static={r.get('r2_static'):+.5f}  online={r.get('r2_online'):+.5f}")
        else:
            log(f"  length {L}: no result (training failed?)")

    log("\n=== recency curve (online R² vs window length) ===")
    for r in rows:
        online = r["r2_online"]
        bar = "#" * max(0, int(online * 2000)) if online and online > 0 else ""
        log(f"  {r['length']:>4}d  online={online:+.5f}  {bar}")
    (out / "sweep_summary.json").write_text(json.dumps(
        {"model": args.model, "hi": args.hi, "decay_halflife": args.decay_halflife,
         "rows": rows, "timestamp": datetime.now().isoformat()}, indent=2))
    log(f"\nwritten → {out / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
