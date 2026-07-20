"""SETOL stopping rules + calibration harness.

The user's chosen protocol for retrains without a validation oracle
(e.g. the shifted 1298–1698 window): pick the stopping epoch from layer
SPECTRA alone, then epoch-bag {peak−1, peak, peak+1}.

Rules operate on a tslab ``TrainingWatcher.history()`` dump (the
``history`` key of a ``*_health.json``):

  R1 vh_guard   — last epoch BEFORE the count of very-heavy layers
                  (alpha < 2) rises above its running minimum + 1.
  R2 alpha_knee — epoch where the mean-alpha decline flattens: first e
                  with (mean_a[e-1] - mean_a[e]) < eps after at least 3
                  falling epochs (eps = 2% of the total fall).
  R3 ideal_peak — epoch maximizing the count of layers in the ideal
                  band (2 <= alpha <= 4).

Calibration mode: given health jsons from runs where per-epoch VAL was
recorded, report each rule's pick vs the val-argmax — a rule earns
trust by reproducing known peaks before being used blind.

Usage
-----
    uv run python scripts/setol_stop.py calibrate out/*_health.json
    uv run python scripts/setol_stop.py pick some_health.json --rule vh_guard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _per_epoch(history: dict) -> dict[int, dict]:
    """Regroup column-oriented history into {epoch: {layer: alpha}} plus val.

    The final snapshot (restored-best, taken after fit) repeats an epoch
    id already seen — keep the FIRST occurrence of each step so the
    trajectory stays the in-loop one.
    """
    steps = history["step"]
    layers = history["layer"]
    alphas = history["alpha"]
    vals = history.get("val_metric", [None] * len(steps))
    out: dict[int, dict] = {}
    for s, l, a, v in zip(steps, layers, alphas, vals):
        e = out.setdefault(int(s), {"alphas": {}, "val": None})
        if l not in e["alphas"]:
            e["alphas"][l] = a
        if v is not None and e["val"] is None:
            e["val"] = float(v)
    return dict(sorted(out.items()))


def _series(ep: dict[int, dict]):
    epochs = [e for e in ep if e >= 0]
    vh = [sum(1 for a in ep[e]["alphas"].values() if a == a and a < 2.0)
          for e in epochs]
    ideal = [sum(1 for a in ep[e]["alphas"].values() if a == a and 2.0 <= a <= 4.0)
             for e in epochs]
    mean_a = [float(np.nanmean(list(ep[e]["alphas"].values()))) for e in epochs]
    val = [ep[e]["val"] for e in epochs]
    return epochs, vh, ideal, mean_a, val


def pick(history: dict, rule: str) -> int:
    epochs, vh, ideal, mean_a, _ = _series(_per_epoch(history))
    if rule == "vh_guard":
        run_min = vh[0]
        for i in range(1, len(epochs)):
            run_min = min(run_min, vh[i - 1])
            if vh[i] > run_min + 1:
                return epochs[max(0, i - 1)]
        return epochs[-1]
    if rule == "alpha_knee":
        total_fall = max(1e-9, mean_a[0] - min(mean_a))
        eps = 0.02 * total_fall
        falling = 0
        for i in range(1, len(epochs)):
            drop = mean_a[i - 1] - mean_a[i]
            if drop > eps:
                falling += 1
            elif falling >= 3:
                return epochs[i - 1]
        return epochs[-1]
    if rule == "ideal_peak":
        return epochs[int(np.argmax(ideal))]
    raise ValueError(rule)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("jsons", nargs="+")
    k = sub.add_parser("pick")
    k.add_argument("json")
    k.add_argument("--rule", default="vh_guard",
                   choices=["vh_guard", "alpha_knee", "ideal_peak"])
    args = p.parse_args()

    if args.cmd == "pick":
        h = json.loads(Path(args.json).read_text())["history"]
        print(pick(h, args.rule))
        return

    print(f"{'run':38s} {'val_argmax':>10} {'vh_guard':>9} {'alpha_knee':>10} "
          f"{'ideal_peak':>10}")
    for j in args.jsons:
        blob = json.loads(Path(j).read_text())
        h = blob["history"]
        epochs, _, _, _, val = _series(_per_epoch(h))
        va = (epochs[int(np.nanargmax([v if v is not None else -9 for v in val]))]
              if any(v is not None for v in val) else None)
        picks = {r: pick(h, r) for r in ("vh_guard", "alpha_knee", "ideal_peak")}
        print(f"{Path(j).stem:38s} {str(va):>10} {picks['vh_guard']:>9} "
              f"{picks['alpha_knee']:>10} {picks['ideal_peak']:>10}")


if __name__ == "__main__":
    main()
