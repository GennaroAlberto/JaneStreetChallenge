#!/usr/bin/env python3
"""Pack the v2 late-submission bundle for the Jane Street RTMDF kernel.

Builds ``artifacts/colab/js_submission_weights_v2.zip`` — the single archive
uploaded to Kaggle as a private dataset and mounted by the submission
kernel (``notebooks/kaggle_submission.ipynb``). The bundle contains:

* ``src/janestreet/**``            — the package the checkpoints unpickle
  against (``FullPipeline`` unpickling needs ``janestreet.*`` importable);
  ``__pycache__`` / ``*.pyc`` excluded.
* ``scripts/serving/**``           — the online feature-state replica,
  the CPU-safe checkpoint loader (``kernel_state.load_pipe_cpu``) and the
  replay check (the kernel itself is the notebook above; this packer is
  piece 3 of the submission flow).
* ``checkpoints/*.pkl``            — the four v2-500d-caughtup-4 stream
  checkpoints (flattened basenames; provenance kept in the manifest).
  These are the CAUGHT-UP Kaggle-trained 500d members produced by
  ``scripts/ship_catchup.py`` — a still-running job may not have written
  them yet; existence is checked at RUN time (``--dry-run`` lists missing
  ones as EXPECTED).
* ``preprocessor.pkl``             — the pool700_lags fit-time
  clipper+standardizer (the exact 134-col schema lives inside it and in
  each checkpoint's ``feature_cols``).
* ``manifest.json``                — stack version, per-checkpoint stream
  tag / walk-forward refit lr / blend weight / is_xsec flag (the kernel
  is manifest-driven: it loads, refits and blends exactly what is listed
  here), calibration constants, and TODO hooks (vol-scaling, XGB stream —
  still deliberately NOT implemented).

Usage
-----
    uv run python scripts/pack_submission_weights.py            # build zip
    uv run python scripts/pack_submission_weights.py --dry-run  # list only

Exit codes: 0 ok, 2 missing required inputs (each reported with a hint).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# v2-500d-caughtup-4 stack definition. All four streams consume the SAME
# 134-column preprocessed feature matrix (pool700_lags recipe: 76 kept native
# features + rolling stats / market-avg on the 16 corr features + lagged
# responders + responder-signal engineered columns — see
# src/janestreet/data/features.py and src/janestreet/config.py). Members are
# the Kaggle-trained 500d models, caught up through the held-out tail by
# scripts/ship_catchup.py (post-walk weights: ship-forward only). Blend =
# equal-weight (0.25) mean of the four raw streams with PER-ROW weights —
# rows whose symbol is absent from today's active slate exclude the xsec
# stream(s) (``is_xsec``) — then per-symbol trailing calibration
# (scripts/blend_v3.py ``calibrate``); each stream is walk-forward refitted
# once per date_id at its own lr when the lags frame arrives.
#
# NOTE: the caught-up checkpoints are produced by a RUNNING ship_catchup job
# and may not exist yet. Existence is checked at run time only — never
# assert at import.
# ---------------------------------------------------------------------------
STACK_VERSION = "v2-500d-caughtup-4"

CATCHUP_HINT = ("produced by the (possibly still running) scripts/ship_catchup.py "
                "job over the Kaggle-trained 500d member — retry once it lands")

CHECKPOINTS: list[dict] = [
    {
        "source": "artifacts/ship/checkpoints/gru_modelr_gru_wide_caughtup.pkl",
        "stream": "gru_wide",
        "refit_lr": 3e-4,
        "blend_weight": 0.25,
        "is_xsec": False,
        "hint": f"caught-up 500d wide GRU — {CATCHUP_HINT}",
    },
    {
        "source": "artifacts/ship/checkpoints/gru_modelr_xsec_xsec_gru_wide_caughtup.pkl",
        "stream": "xsec_wide",
        "refit_lr": 3e-4,
        "blend_weight": 0.25,
        "is_xsec": True,   # attends across symbols -> active-slate handling in the kernel
        "hint": f"caught-up 500d wide xsec-attention GRU — {CATCHUP_HINT}",
    },
    {
        "source": "artifacts/ship/checkpoints/gru_modelr_gru_s42_caughtup.pkl",
        "stream": "gru500",
        "refit_lr": 3e-4,
        "blend_weight": 0.25,
        "is_xsec": False,
        "hint": f"caught-up 500d GRU (seed 42) — {CATCHUP_HINT}",
    },
    {
        "source": "artifacts/ship/checkpoints/lstm_modelr_lstm_s42_caughtup.pkl",
        "stream": "lstm500",
        "refit_lr": 1e-3,
        "blend_weight": 0.25,
        "is_xsec": False,
        "hint": f"caught-up 500d LSTM (seed 42) — {CATCHUP_HINT}",
    },
]

PREPROCESSOR_SOURCE = "artifacts/precomputed/pool700_lags/preprocessor.pkl"
PREPROCESSOR_ARC = "preprocessor.pkl"

SRC_PKG = "src/janestreet"
SERVING_DIR = "scripts/serving"

# v2 zip — deliberately NOT the v1 name (artifacts/colab/js_submission_weights.zip
# stays untouched so the v1 dataset can be rebuilt/diffed).
DEFAULT_OUT = "artifacts/colab/js_submission_weights_v2.zip"

# Per-symbol trailing calibration constants — must match the kernel's port
# of scripts/blend_v3.py::calibrate.
CALIBRATION = {
    "source": "scripts/blend_v3.py::calibrate",
    "lookback_dates": 20,
    "ridge_w": 3.0,
    "min_history_rows": 500,
    "alpha_clip": [0.0, 1.5],
}

# Future hooks — recorded so the kernel/manifest consumers know what is
# STILL deliberately absent (unchanged from v1). DO NOT implement here.
TODO_V2 = [
    "vol_scaling: pred * (sigma_hat/sigma_bar)^gamma with an online vol "
    "nowcaster (port from scripts/blend_v3.py; gamma selected on holdout-1)",
    "xgb_stream: add the pool-v3 XGB stream — needs booster-only packaging "
    "(save_model JSON) to dodge the torch+libomp pickle segfault "
    "(see FullPipeline.save docstring)",
]

EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_NAMES = {".DS_Store"}


# ---------------------------------------------------------------------------
def _iter_tree(root: Path) -> list[Path]:
    """All files under ``root`` (sorted), minus caches/junk."""
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix in EXCLUDE_SUFFIXES or p.name in EXCLUDE_NAMES:
            continue
        out.append(p)
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:>8.1f} {unit}" if unit != "B" else f"{n:>8d} B "
        n /= 1024  # type: ignore[assignment]
    return f"{n} B"


def collect_entries(
    repo: Path,
) -> tuple[list[tuple[str, Path]], list[str], list[str]]:
    """Resolve (arcname, source_path) pairs.

    Returns ``(entries, problems, expected)`` where ``expected`` holds the
    not-yet-built caught-up checkpoints (the ship_catchup job may still be
    running) — reported as EXPECTED in --dry-run, and as a build blocker
    (with a retry hint) in a real build. Existence is checked HERE, at run
    time — never at import.
    """
    entries: list[tuple[str, Path]] = []
    problems: list[str] = []
    expected: list[str] = []

    # 1. src/janestreet package -------------------------------------------
    pkg = repo / SRC_PKG
    if pkg.is_dir():
        for f in _iter_tree(pkg):
            entries.append((f.relative_to(repo).as_posix(), f))
    else:
        problems.append(f"{SRC_PKG}/ not found under {repo} — wrong --repo-root?")

    # 2. serving kernel + feature-state -----------------------------------
    serving = repo / SERVING_DIR
    if serving.is_dir():
        files = _iter_tree(serving)
        if not files:
            problems.append(f"{SERVING_DIR}/ exists but is empty — write the "
                            "online feature-state (feature_state.py) "
                            "before packing.")
        for f in files:
            entries.append((f.relative_to(repo).as_posix(), f))
    else:
        problems.append(
            f"{SERVING_DIR}/ not found — the online feature-state "
            "(feature_state.py) must exist before the bundle is "
            "buildable. See docs/GPU_RESEARCH_PLAN.md 'Submission pipeline'.")

    # 3. the four v2 caught-up checkpoints --------------------------------
    for spec in CHECKPOINTS:
        src = repo / spec["source"]
        if src.is_file():
            entries.append((f"checkpoints/{src.name}", src))
        else:
            expected.append(
                f"checkpoint not built yet: {spec['source']} "
                f"(stream '{spec['stream']}') — {spec['hint']}")

    # 4. preprocessor ------------------------------------------------------
    prep = repo / PREPROCESSOR_SOURCE
    if prep.is_file():
        entries.append((PREPROCESSOR_ARC, prep))
    else:
        problems.append(
            f"preprocessor missing: {PREPROCESSOR_SOURCE} — regenerate the "
            "pool700_lags precompute (scripts/precompute_dataset.py) or "
            "point at the memmap dir that holds it.")

    # Defensive: arcnames must be unique.
    seen: set[str] = set()
    for arc, _ in entries:
        if arc in seen:
            problems.append(f"duplicate arcname in bundle: {arc}")
        seen.add(arc)

    return entries, problems, expected


def build_manifest(repo: Path, with_hashes: bool) -> dict:
    ckpts: dict[str, dict] = {}
    for spec in CHECKPOINTS:
        src = repo / spec["source"]
        arc = f"checkpoints/{Path(spec['source']).name}"
        entry = {
            "stream": spec["stream"],
            "refit_lr": spec["refit_lr"],
            "blend_weight": spec["blend_weight"],
            "is_xsec": spec["is_xsec"],
            "source": spec["source"],
        }
        if src.is_file():
            entry["bytes"] = src.stat().st_size
            if with_hashes:
                entry["sha256"] = _sha256(src)
        ckpts[arc] = entry
    return {
        "stack_version": STACK_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoints": ckpts,
        "preprocessor": PREPROCESSOR_ARC,
        "feature_schema": (
            "134-column pool700_lags recipe; authoritative list is "
            "feature_cols inside each checkpoint pickle (FullPipeline.load) "
            "and must match Preprocessor.feature_cols in preprocessor.pkl"
        ),
        "blend": {
            "type": "manifest_weighted_mean",
            "per_row": (
                "per-row weights: rows whose symbol is absent from today's "
                "active slate exclude the is_xsec streams from the weighted "
                "mean (remaining weights renormalize)"
            ),
            "post": "per_symbol_trailing_calibration",
            "calibration": CALIBRATION,
        },
        "todo_v2": TODO_V2,
    }


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="repo root (default: parent of scripts/)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output zip path (default: <repo>/{DEFAULT_OUT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="list bundle contents + manifest; write nothing")
    args = ap.parse_args(argv)

    repo = args.repo_root.resolve()
    out = (args.out if args.out is not None else repo / DEFAULT_OUT).resolve()

    entries, problems, expected = collect_entries(repo)
    manifest = build_manifest(repo, with_hashes=not args.dry_run)

    if args.dry_run:
        print(f"[dry-run] bundle plan for {out} (stack {STACK_VERSION})")
        total = 0
        for arc, src in entries:
            size = src.stat().st_size
            total += size
            print(f"  {_fmt_size(size)}  {arc}    <- {src.relative_to(repo)}")
        print(f"  {'':>11}  manifest.json    <- (generated)")
        print(f"[dry-run] {len(entries) + 1} files, ~{total / 2**20:.1f} MB uncompressed")
        if expected:
            print(f"[dry-run] {len(expected)} checkpoint(s) EXPECTED but not "
                  "built yet (ship_catchup job still running?):")
            for e in expected:
                print(f"  EXPECTED: {e}")
        if problems:
            print(f"[dry-run] {len(problems)} problem(s) — a real build would fail:")
            for p in problems:
                print(f"  MISSING/ERROR: {p}")
        print("[dry-run] manifest.json would be:")
        print(json.dumps(manifest, indent=2))
        return 0

    if problems or expected:
        print("cannot build bundle — fix the following first:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        for e in expected:
            print(f"  - {e}", file=sys.stderr)
        if expected:
            print("hint: the caught-up checkpoints come from the running "
                  "scripts/ship_catchup.py job — wait for it to finish, then "
                  "re-run this packer.", file=sys.stderr)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arc, src in entries:
            zf.write(src, arcname=arc)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB, "
          f"{len(entries) + 1} files, stack {STACK_VERSION})")
    for arc, meta in manifest["checkpoints"].items():
        tag = " [xsec]" if meta["is_xsec"] else ""
        print(f"  {meta['stream']:<12} refit_lr={meta['refit_lr']:g} "
              f"w={meta['blend_weight']:.4f}{tag}  {arc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
