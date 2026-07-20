"""Command-line entrypoints.

Examples
--------
Run CV with the Volkova replica on a small date range::

    uv run js cv --model gru_modelr --min-date 1400 --max-date 1698 --epochs 3

Try the Volterra-signature Transformer::

    uv run js cv --model sig_transformer --min-date 1400 --max-date 1698 \\
        --model-kwargs '{"n_layers": 4, "signature_depth": 2}'

Profile data quickly::

    uv run js profile
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from janestreet.config import Cfg
from janestreet.pipeline import prepare_dataset, run_cv


def _parse_cfg(args: argparse.Namespace) -> Cfg:
    cfg = Cfg()
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    if args.artifacts_root:
        cfg.artifacts_root = Path(args.artifacts_root)
    if args.min_date is not None:
        cfg.min_date_id = args.min_date
    if args.max_date is not None:
        cfg.max_date_id = args.max_date
    if args.model:
        cfg.model_name = args.model
    if args.model_kwargs:
        cfg.model_kwargs = json.loads(args.model_kwargs)
    if args.epochs is not None:
        cfg.epochs = args.epochs
        cfg.model_kwargs.setdefault("epochs", args.epochs)
    if args.n_splits is not None:
        cfg.n_splits = args.n_splits
    if args.cv_gap is not None:
        cfg.cv_gap_dates = args.cv_gap
    if args.test_size is not None:
        cfg.test_size_dates = args.test_size
    if args.device:
        cfg.device = args.device
        cfg.model_kwargs.setdefault("device", args.device)
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.model_kwargs.setdefault("seed", args.seed)
    if args.aux_targets:
        cfg.aux_targets = args.aux_targets.split(",")
    if args.realized_aux:
        cfg.realized_aux = True
    cfg.verbose = not args.quiet
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", default=None)
    p.add_argument("--artifacts-root", default=None)
    p.add_argument("--min-date", type=int, default=None)
    p.add_argument("--max-date", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--model-kwargs", default=None,
                   help="JSON dict of kwargs passed to the model constructor")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument("--cv-gap", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--aux-targets", default=None,
                   help="comma-separated aux responder columns (overrides cfg)")
    p.add_argument("--realized-aux", action="store_true",
                   help="synthesize realized (nowcast) responder_11/12 aux targets")
    p.add_argument("--quiet", action="store_true")


def cmd_cv(args: argparse.Namespace) -> None:
    cfg = _parse_cfg(args)
    scores = run_cv(cfg)
    print(json.dumps({"folds": scores, "mean": sum(scores) / len(scores)}, indent=2))


def cmd_profile(args: argparse.Namespace) -> None:
    cfg = _parse_cfg(args)
    df = prepare_dataset(cfg)
    print(f"rows = {df.height}, cols = {df.width}")
    print(f"date range: {df['date_id'].min()} → {df['date_id'].max()}")
    print(f"symbols    : {df['symbol_id'].n_unique()}")
    print(f"time_ids   : {df['time_id'].n_unique()} (min {df['time_id'].min()}, max {df['time_id'].max()})")
    print("first 5 columns:", df.columns[:8], "…")


def main() -> None:
    p = argparse.ArgumentParser(prog="js", description="Jane Street pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_cv = sub.add_parser("cv", help="run cross-validation with the configured model")
    _add_common(p_cv)
    p_cv.set_defaults(func=cmd_cv)

    p_pf = sub.add_parser("profile", help="profile the dataset after feature engineering")
    _add_common(p_pf)
    p_pf.set_defaults(func=cmd_profile)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
