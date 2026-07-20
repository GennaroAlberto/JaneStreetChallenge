"""DRW-1st-place-style feature lab on the precomputed memmap pool.

Ports the winning DRW crypto recipe (writeup: kaggle.com/competitions/
drw-crypto-market-prediction/writeups/drw-solution-1st) to Jane Street:

  A. **Correlation clustering → medoids.** Features cluster heavily
     (|ρ| ≥ 0.6). Distance 1−|ρ|, complete linkage; medoid = the member with
     the highest Σ|ρ| to its cluster (the most expensive point to replicate).
  B. **SHAP-consistency selection.** XGB per contiguous time fold; a feature
     survives if it makes the top-K mean-|SHAP| list in ≥ min_folds folds.
     Selects features that are *robustly* important, not fold-lucky.
  C. **Symbolic 2nd-order combos.** x−y, x*y, x/y, min, max over the core
     set, screened by |corr(candidate, target)| on train rows with a
     redundancy cap against existing columns and each other.
  D. **Ablation.** Full-train XGB → weighted R² on the validation tail:
     all-cols baseline vs medoids vs medoids+combos vs all+combos.

Stages are cached to --out; rerun with --stages to redo a subset.

Usage
-----
    uv run python scripts/drw_feature_lab.py \\
        --data artifacts/precomputed/pool700_lags \\
        --train-lo 1399 --train-hi 1598 --valid-lo 1599 --valid-hi 1698 \\
        --out artifacts/bench/drw_lab
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def r2_weighted(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest: dict, lo: int, hi: int, step: int = 1) -> np.ndarray:
    ranges = manifest["date_row_ranges"]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64)
             for d in range(lo, hi + 1) if str(d) in ranges]
    rows = np.concatenate(parts)
    return rows[::step] if step > 1 else rows


# ---------------------------------------------------------------------------
# Stage A — correlation clustering → medoid representatives
# ---------------------------------------------------------------------------

def stage_cluster(X: np.ndarray, names: list[str], thresh: float, log) -> dict:
    log(f"[A] corr matrix on {X.shape[0]:,} × {X.shape[1]} …")
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="complete")
    labels = fcluster(Z, t=1.0 - thresh, criterion="distance")

    clusters: dict[int, list[int]] = {}
    for i, lb in enumerate(labels):
        clusters.setdefault(int(lb), []).append(i)
    medoids = []
    for _lb, members in sorted(clusters.items()):
        if len(members) == 1:
            medoids.append(members[0])
            continue
        sub = np.abs(C[np.ix_(members, members)])
        medoids.append(members[int(np.argmax(sub.sum(axis=0)))])
    medoids = sorted(medoids)
    log(f"[A] {len(clusters)} clusters at |ρ|≥{thresh} → {len(medoids)} medoids "
        f"(from {len(names)} cols)")
    return {
        "threshold": thresh,
        "clusters": {str(lb): [names[i] for i in m] for lb, m in sorted(clusters.items())},
        "medoids": [names[i] for i in medoids],
        "medoid_idx": medoids,
    }


# ---------------------------------------------------------------------------
# Stage B — SHAP-consistency across contiguous time folds
# ---------------------------------------------------------------------------

def stage_shap(
    X_mm, y, w, manifest, medoid_idx: list[int], names: list[str],
    train_lo: int, train_hi: int, n_folds: int, top_k: int, min_folds: int,
    row_step: int, seed: int, log,
) -> dict:
    bounds = np.linspace(train_lo, train_hi + 1, n_folds + 1).astype(int)
    per_fold_top: list[list[str]] = []
    med_names = [names[i] for i in medoid_idx]

    for f in range(n_folds):
        lo, hi = int(bounds[f]), int(bounds[f + 1] - 1)
        rows = rows_for_dates(manifest, lo, hi, step=row_step)
        Xf = np.ascontiguousarray(X_mm[rows][:, medoid_idx]).astype(np.float32)
        yf, wf = np.ascontiguousarray(y[rows]), np.ascontiguousarray(w[rows])
        t0 = time.time()
        model = xgb.XGBRegressor(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=10,
            reg_lambda=1.0, tree_method="hist", max_bin=128, n_jobs=1,
            random_state=seed, objective="reg:squarederror",
        )
        model.fit(Xf, yf, sample_weight=wf, verbose=False)
        sub = np.random.default_rng(seed).choice(len(Xf), min(100_000, len(Xf)), replace=False)
        contribs = model.get_booster().predict(
            xgb.DMatrix(Xf[sub], feature_names=med_names), pred_contribs=True
        )
        imp = np.abs(contribs[:, :-1]).mean(axis=0)  # drop bias column
        order = np.argsort(-imp)[:top_k]
        top = [med_names[i] for i in order]
        per_fold_top.append(top)
        log(f"[B] fold {f + 1}/{n_folds} dates {lo}..{hi} rows={len(rows):,} "
            f"({time.time() - t0:.0f}s) top5={top[:5]}")
        del Xf, yf, wf, model
        gc.collect()

    counts: dict[str, int] = {}
    for top in per_fold_top:
        for name in top:
            counts[name] = counts.get(name, 0) + 1
    core = sorted([n for n, c in counts.items() if c >= min_folds])
    log(f"[B] core = {len(core)} features in top-{top_k} of ≥{min_folds}/{n_folds} folds")
    return {"per_fold_top": per_fold_top, "counts": counts, "core": core,
            "top_k": top_k, "min_folds": min_folds}


# ---------------------------------------------------------------------------
# Stage C — symbolic 2nd-order combos, screened by target corr or tree gain
# ---------------------------------------------------------------------------

OPS = {
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / np.where(np.abs(b) < 0.1, np.sign(b) * 0.1 + (b == 0) * 0.1, b),
    "min": np.minimum,
    "max": np.maximum,
}


def stage_combos(
    X: np.ndarray, y: np.ndarray, names: list[str], core: list[str],
    keep: int, redundancy_cap: float, log,
) -> dict:
    idx = [names.index(c) for c in core]
    Xc = X[:, idx]
    yz = (y - y.mean()) / (y.std() + 1e-12)

    def corr_with_target(v: np.ndarray) -> float:
        sd = v.std()
        if sd < 1e-9 or not np.isfinite(sd):
            return 0.0
        return float(np.mean((v - v.mean()) / sd * yz))

    base_corr = {c: corr_with_target(Xc[:, j].astype(np.float64)) for j, c in enumerate(core)}
    log(f"[C] screening {len(OPS)} ops × C({len(core)},2) pairs "
        f"= {len(OPS) * len(core) * (len(core) - 1) // 2} candidates on "
        f"{X.shape[0]:,} rows")

    cands: list[tuple[float, str, int, int, str]] = []
    for a in range(len(core)):
        va = Xc[:, a].astype(np.float64)
        for b in range(a + 1, len(core)):
            vb = Xc[:, b].astype(np.float64)
            for op, fn in OPS.items():
                r = corr_with_target(fn(va, vb))
                # only worth keeping if it beats both parents
                if abs(r) > max(abs(base_corr[core[a]]), abs(base_corr[core[b]])):
                    cands.append((r, op, a, b, f"{op}({core[a]},{core[b]})"))
    cands.sort(key=lambda t: -abs(t[0]))
    log(f"[C] {len(cands)} candidates beat both parents; applying redundancy cap")

    accepted: list[dict] = []
    accepted_vecs: list[np.ndarray] = []
    for r, op, a, b, label in cands:
        if len(accepted) >= keep:
            break
        v = OPS[op](Xc[:, a].astype(np.float64), Xc[:, b].astype(np.float64))
        sd = v.std()
        if sd < 1e-9:
            continue
        vz = (v - v.mean()) / sd
        redundant = False
        for u in accepted_vecs:
            if abs(float(np.mean(vz * u))) > redundancy_cap:
                redundant = True
                break
        if not redundant:
            for parent in (a, b):
                pz = Xc[:, parent].astype(np.float64)
                pz = (pz - pz.mean()) / (pz.std() + 1e-12)
                if abs(float(np.mean(vz * pz))) > redundancy_cap:
                    redundant = True
                    break
        if redundant:
            continue
        accepted.append({"label": label, "op": op, "a": core[a], "b": core[b],
                         "train_corr": r})
        accepted_vecs.append(vz)
    log(f"[C] accepted {len(accepted)} combos; best: "
        + ", ".join(f"{c['label']}({c['train_corr']:+.4f})" for c in accepted[:5]))
    return {"base_corr": base_corr, "accepted": accepted}


def stage_combos_tree(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, names: list[str],
    core: list[str], keep: int, redundancy_cap: float, batch: int,
    seed: int, log,
) -> dict:
    """Tree-gain screen: unblinds interaction-only candidates.

    The correlation screen cannot see candidates whose parents (and the
    candidate itself) have ~zero univariate target correlation but real
    value through splits — e.g. lag-responder differences (the rsig block:
    +0.0003 via trees, ~0 univariate). Here we fit shallow XGBs on batches
    of candidates ALONE (no base features, so gain is not diluted by
    already-known columns) and rank every candidate by total gain.
    """
    idx = [names.index(c) for c in core]
    Xc = X[:, idx].astype(np.float32)

    labels: list[tuple[str, str, int, int]] = []
    for a in range(len(core)):
        for b in range(a + 1, len(core)):
            for op in OPS:
                labels.append((f"{op}({core[a]},{core[b]})", op, a, b))
    log(f"[C/tree] {len(labels)} candidates, batches of {batch}")

    scored: list[tuple[float, tuple]] = []
    for lo in range(0, len(labels), batch):
        chunk = labels[lo:lo + batch]
        M = np.column_stack([
            OPS[op](Xc[:, a].astype(np.float64), Xc[:, b].astype(np.float64))
            for _, op, a, b in chunk
        ]).astype(np.float32)
        model = xgb.XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
            tree_method="hist", max_bin=64, n_jobs=1, random_state=seed,
            importance_type="gain", objective="reg:squarederror",
        )
        model.fit(M, y, sample_weight=w, verbose=False)
        gains = model.feature_importances_
        scored.extend(zip(gains.tolist(), chunk, strict=True))
        log(f"[C/tree] batch {lo // batch + 1}/"
            f"{(len(labels) + batch - 1) // batch} done "
            f"(top gain {max(gains):.4f})")
        del M, model
        gc.collect()

    scored.sort(key=lambda t: -t[0])
    accepted: list[dict] = []
    accepted_vecs: list[np.ndarray] = []
    for gain, (label, op, a, b) in scored:
        if len(accepted) >= keep or gain <= 0:
            break
        v = OPS[op](Xc[:, a].astype(np.float64), Xc[:, b].astype(np.float64))
        sd = v.std()
        if sd < 1e-9 or not np.isfinite(sd):
            continue
        vz = (v - v.mean()) / sd
        redundant = False
        for u in accepted_vecs:
            if abs(float(np.mean(vz * u))) > redundancy_cap:
                redundant = True
                break
        if not redundant:
            for parent in (a, b):
                pz = Xc[:, parent].astype(np.float64)
                psd = pz.std()
                if psd < 1e-9:
                    continue
                pz = (pz - pz.mean()) / psd
                if abs(float(np.mean(vz * pz))) > redundancy_cap:
                    redundant = True
                    break
        if redundant:
            continue
        accepted.append({"label": label, "op": op, "a": core[a], "b": core[b],
                         "gain": float(gain)})
        accepted_vecs.append(vz)
    log(f"[C/tree] accepted {len(accepted)}; best: "
        + ", ".join(f"{c['label']}(g={c['gain']:.3f})" for c in accepted[:5]))
    return {"accepted": accepted, "screen": "tree"}


def combo_matrix(X: np.ndarray, names: list[str], accepted: list[dict]) -> np.ndarray:
    cols = []
    for c in accepted:
        va = X[:, names.index(c["a"])].astype(np.float32)
        vb = X[:, names.index(c["b"])].astype(np.float32)
        cols.append(OPS[c["op"]](va, vb))
    return np.column_stack(cols).astype(np.float32) if cols else np.zeros((len(X), 0), np.float32)


# ---------------------------------------------------------------------------
# Stage D — validation ablation
# ---------------------------------------------------------------------------

def fit_eval(Xtr, ytr, wtr, Xva, yva, wva, n_estimators, seed):
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", max_bin=128, n_jobs=1,
        early_stopping_rounds=50, random_state=seed,
        objective="reg:squarederror",
    )
    model.fit(Xtr, ytr, sample_weight=wtr,
              eval_set=[(Xva, yva)], sample_weight_eval_set=[wva], verbose=False)
    pred = model.predict(Xva)
    best_it = int(getattr(model, "best_iteration", n_estimators) or n_estimators)
    return r2_weighted(yva, pred, wva), best_it, pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="artifacts/precomputed/pool700_lags")
    p.add_argument("--train-lo", type=int, default=1399)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--out", type=str, default="artifacts/bench/drw_lab")
    p.add_argument("--stages", type=str, default="A,B,C,D")
    p.add_argument("--corr-thresh", type=float, default=0.6)
    p.add_argument("--screen-row-step", type=int, default=8,
                   help="row subsample step for corr/screening stages")
    p.add_argument("--fold-row-step", type=int, default=2,
                   help="row subsample step for stage-B fold fits")
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-folds", type=int, default=3)
    p.add_argument("--extra-core", type=str, default=None,
                   help="comma-separated column names merged into the stage-C "
                        "combo pool (e.g. atlas lead/reversal features + lag "
                        "responders that SHAP-consistency filters out)")
    p.add_argument("--screen", choices=["corr", "tree"], default="corr",
                   help="stage-C candidate screen: univariate target corr "
                        "(fast, blind to interaction-only value) or tree gain "
                        "(slower, sees it)")
    p.add_argument("--tree-batch", type=int, default=400)
    p.add_argument("--keep-combos", type=int, default=40)
    p.add_argument("--redundancy-cap", type=float, default=0.9)
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--variants", type=str,
                   default="all_cols,medoids,medoids+combos,all+combos",
                   help="comma-separated stage-D variants to fit")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    log_path = out / "lab_log.txt"

    def log(msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    names: list[str] = manifest["feature_cols"]
    K = manifest["K"]
    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")

    screen_rows = rows_for_dates(manifest, args.train_lo, args.train_hi, args.screen_row_step)
    X_screen = np.ascontiguousarray(X_mm[screen_rows]).astype(np.float32)
    y_screen = np.ascontiguousarray(y[screen_rows]).astype(np.float64)
    w_screen = np.ascontiguousarray(w[screen_rows]).astype(np.float32)
    log(f"screen slice: {X_screen.shape[0]:,} rows (step {args.screen_row_step})")

    # ---- A ----
    if "A" in stages:
        clus = stage_cluster(X_screen, names, args.corr_thresh, log)
        (out / "clusters.json").write_text(json.dumps(clus, indent=1))
    else:
        clus = json.loads((out / "clusters.json").read_text())

    # ---- B ----
    if "B" in stages:
        shap_res = stage_shap(
            X_mm, y, w, manifest, clus["medoid_idx"], names,
            args.train_lo, args.train_hi, args.n_folds, args.top_k,
            args.min_folds, args.fold_row_step, args.seed, log,
        )
        (out / "shap_core.json").write_text(json.dumps(shap_res, indent=1))
    else:
        shap_res = json.loads((out / "shap_core.json").read_text())

    # ---- C ----
    if "C" in stages:
        core = list(shap_res["core"])
        if args.extra_core:
            extra = [c for c in args.extra_core.split(",") if c in names]
            core += [c for c in extra if c not in core]
            log(f"[C] core extended by --extra-core: {len(shap_res['core'])} → {len(core)}")
        if args.screen == "tree":
            combos = stage_combos_tree(
                X_screen, y_screen, w_screen, names, core,
                args.keep_combos, args.redundancy_cap, args.tree_batch,
                args.seed, log,
            )
        else:
            combos = stage_combos(
                X_screen, y_screen, names, core,
                args.keep_combos, args.redundancy_cap, log,
            )
        (out / "combos.json").write_text(json.dumps(combos, indent=1))
    else:
        combos = json.loads((out / "combos.json").read_text())
    del X_screen, y_screen
    gc.collect()

    # ---- D ----
    if "D" not in stages:
        return
    tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    log(f"[D] train rows={len(tr):,}  valid rows={len(va):,}")
    Xtr_all = np.ascontiguousarray(X_mm[tr]).astype(np.float32)
    Xva_all = np.ascontiguousarray(X_mm[va]).astype(np.float32)
    ytr, wtr = np.ascontiguousarray(y[tr]), np.ascontiguousarray(w[tr])
    yva, wva = np.ascontiguousarray(y[va]), np.ascontiguousarray(w[va])
    med = clus["medoid_idx"]
    acc = combos["accepted"]

    variants = {
        "all_cols": lambda: (Xtr_all, Xva_all),
        "medoids": lambda: (np.ascontiguousarray(Xtr_all[:, med]),
                            np.ascontiguousarray(Xva_all[:, med])),
        "medoids+combos": lambda: (
            np.hstack([Xtr_all[:, med], combo_matrix(Xtr_all, names, acc)]),
            np.hstack([Xva_all[:, med], combo_matrix(Xva_all, names, acc)]),
        ),
        "all+combos": lambda: (
            np.hstack([Xtr_all, combo_matrix(Xtr_all, names, acc)]),
            np.hstack([Xva_all, combo_matrix(Xva_all, names, acc)]),
        ),
    }
    results = []
    preds_dir = out / "preds"
    preds_dir.mkdir(exist_ok=True)
    wanted = [v.strip() for v in args.variants.split(",")]
    variants = {k: v for k, v in variants.items() if k in wanted}
    for vname, build in variants.items():
        xtr, xva = build()
        t0 = time.time()
        r2, best_it, pred = fit_eval(xtr, ytr, wtr, xva, yva, wva,
                                     args.n_estimators, args.seed)
        log(f"[D] {vname:16s} n_feat={xtr.shape[1]:>3} best_iter={best_it:>4} "
            f"R²={r2:+.5f}  ({time.time() - t0:.0f}s)")
        np.savez_compressed(preds_dir / f"xgb_{vname.replace('+', '_')}.npz",
                            pred=pred.astype(np.float32), rows=va)
        results.append({"variant": vname, "n_features": int(xtr.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2})
        del xtr, xva
        gc.collect()

    base = results[0]["r2_weighted"]
    for r in results:
        r["delta_vs_all_cols"] = r["r2_weighted"] - base
    (out / "ablation.json").write_text(json.dumps({
        "data": str(data),
        "train": [args.train_lo, args.train_hi],
        "valid": [args.valid_lo, args.valid_hi],
        "results": results, "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log("[D] done → ablation.json")


if __name__ == "__main__":
    main()
