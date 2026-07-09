"""Experiment orchestrator — declare a bag in YAML, train it, blend it.

This is the "pick your dates, pick your models, mix them" layer. A config
file names:

  * the precomputed date pool (which dates + which features),
  * the validation tail,
  * a list of bag members: for each, a model + a set of seeds + a resample
    spec (subsample / bootstrap fraction of the pool),
  * optionally extra prediction files to fold into the blend (e.g. an xgb
    stream produced by scripts/bench.py).

The runner trains every (member, seed) sequentially by shelling out to the
tested ``scripts/train_from_memmap.py`` (one process each → clean memory
between members on a 16 GB Mac), then calls ``scripts/ensemble_blend.py`` on
the collected ``_online.npz`` (falling back to ``_static.npz`` for models
without online refit). Everything is resumable: members whose result JSON
already exists are skipped.

Example config (experiments/example_bag.yaml)::

    data: artifacts/precomputed/pool700_lags
    out:  artifacts/bench/exp_bag_A
    valid_lo: 1599
    valid_hi: 1698
    members:
      - model: lstm_modelr
        seeds: [1, 2, 3]
        resample_mode: subsample
        resample_frac: 0.6
      - model: timexer
        seeds: [1, 2]
        resample_mode: subsample
        resample_frac: 0.6
    extra_preds:                       # optional, blended in as-is
      - artifacts/bench/run2_280d/preds/xgb_static.npz
    blend_stream: online               # 'online' or 'static'

Usage::

    uv run python scripts/run_experiment.py --config experiments/example_bag.yaml
    uv run python scripts/run_experiment.py --config ... --dry-run   # print plan only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
TRAINER = REPO / "scripts" / "train_from_memmap.py"
BLENDER = REPO / "scripts" / "ensemble_blend.py"
# Models without an online-refit hook — blend their _static stream instead.
STATIC_ONLY = {"mlp", "xgb"}


def member_tag(model: str, seed: int, frac: float, mode: str) -> str:
    return f"{model}_{mode}{int(frac * 100):02d}_s{seed}"


def run(cmd: list[str], env: dict, log) -> int:
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env=env, check=False)  # noqa: S603  trusted local commands
    return proc.returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data = cfg["data"]
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    valid_lo, valid_hi = cfg.get("valid_lo", 1599), cfg.get("valid_hi", 1698)
    blend_stream = cfg.get("blend_stream", "online")

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with (out / "experiment_log.txt").open("a") as f:
            f.write(line)

    # Expand members × seeds into a flat plan.
    plan = []
    for m in cfg["members"]:
        model = m["model"]
        mode = m.get("resample_mode", "subsample")
        frac = float(m.get("resample_frac", 0.63))
        for seed in m.get("seeds", [m.get("seed", 1)]):
            plan.append((model, int(seed), mode, frac))

    log(f"=== experiment: {args.config} ===")
    log(f"data={data}  out={out}  valid={valid_lo}..{valid_hi}  blend={blend_stream}")
    log(f"{len(plan)} bag members:")
    for model, seed, mode, frac in plan:
        log(f"  - {member_tag(model, seed, frac, mode)}")
    if cfg.get("extra_preds"):
        log(f"extra_preds folded into blend: {cfg['extra_preds']}")
    if args.dry_run:
        log("(dry run — not training)")
        return

    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # --- Train each member (skip if its result JSON already exists) ---
    for model, seed, mode, frac in plan:
        tag = member_tag(model, seed, frac, mode)
        if (out / f"{model}_{tag}.json").exists():
            log(f"[skip] {tag} already done")
            continue
        rc = run([
            "uv", "run", "python", "-u", str(TRAINER),
            "--data", data, "--model", model,
            "--resample-mode", mode, "--resample-frac", str(frac),
            "--seed", str(seed), "--tag", tag,
            "--valid-lo", str(valid_lo), "--valid-hi", str(valid_hi),
            "--out", str(out),
        ], env, log)
        if rc != 0:
            log(f"[warn] {tag} exited with code {rc} — continuing with the rest")

    # --- Collect prediction streams for the blend ---
    preds_dir = out / "preds"
    streams: list[str] = []
    for model, seed, mode, frac in plan:
        tag = member_tag(model, seed, frac, mode)
        want = "static" if model in STATIC_ONLY else blend_stream
        f = preds_dir / f"{model}_{tag}_{want}.npz"
        if f.exists():
            streams.append(str(f))
        else:
            log(f"[warn] missing prediction stream {f.name} — excluded from blend")
    streams += [str(e) for e in cfg.get("extra_preds", [])]

    if len(streams) < 2:
        log(f"[warn] only {len(streams)} stream(s) — need >=2 to blend. Stopping.")
        return

    log(f"\nblending {len(streams)} streams:")
    for s in streams:
        log(f"  {s}")
    blend_cmd = ["uv", "run", "python", str(BLENDER)]
    for s in streams:
        blend_cmd += ["--pred", s]
    blend_cmd += ["--out", str(out / "ensemble.json")]
    run(blend_cmd, env, log)
    log(f"\ndone → {out / 'ensemble.json'}")


if __name__ == "__main__":
    main()
