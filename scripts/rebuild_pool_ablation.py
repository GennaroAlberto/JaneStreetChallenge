"""Joint ablation of every confirmed feature block — the pool rebuild.

Individually confirmed on the standard harness (base = 134 cols, +0.00591):

    supervised-AE bottleneck   8 cols   +0.00058   (ae_lab_sup)
    atlas-seeded combos v2    50 cols   +0.00052   (drw_lab_v2)
    reversal mkt/idio decomp  14 cols   +0.00040   (ablation_revdecomp)
    rsig lag differences       5 cols   +0.00030
    slow-SMA innovations       9 cols   +0.00025   (ablation_innov)
    stable cluster one-hot     6 cols   +0.00022   (ablation_symbol)

Fully additive would be +0.00227 → ~0.0082. The blocks partially overlap
(combos and decomp both mine the reversal family; the AE compresses the
same pool the combos recombine), so the joint number is the real question.

Variants (nested prefixes of ONE matrix — no hstack doubling, so the
280d window fits):
    base            — 134
    all_but_ae      — base + combos + decomp + rsig + innov + cluster (218)
    all             — + the AE latents (226): the AE's *marginal* value on
                      top of every explicit block is the redundancy check
    all_chain       — + 9 OOF stage-1 responder predictions (--with-chain)
    all_chain_ranks — + 6 cross-sectional ranks (--with-ranks; the
                      data-engineering batch winner, +0.00045 standalone)

`--variants` trims the ladder; `--chain-cache` persists the full-row
stage-1 predictions so the expensive OOF fits run once per window
(the rs2 ladder and the full-row deployable fit share them).

Memory: single in-place allocation; the widest variant is the matrix
itself (contiguous, no copy). Peak ≈ 17 GB full-row at 280d on the 16 GB
machine (some swap), provided nothing heavy runs alongside.

Usage
-----
    uv run python scripts/rebuild_pool_ablation.py \\
        --data artifacts/precomputed/pool700_lags \\
        --out artifacts/bench/pool_rebuild
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.append(str(Path(__file__).parent))
from ablate_innovations import DELTAS as INNOV_DELTAS  # noqa: E402
from ablate_innovations import TRIO, innovation_features  # noqa: E402
from ablate_responder_signal import signal_features  # noqa: E402
from ablate_reversal_decomp import BLOCK, decomp_features  # noqa: E402
from drw_feature_lab import OPS  # noqa: E402
from responder_chain import load_raw_responders, make_xgb  # noqa: E402


def r2_weighted(y, p, w):
    num = float(np.average((p - y) ** 2, weights=w))
    den = float(np.average(y ** 2, weights=w)) + 1e-38
    return 1.0 - num / den


def rows_for_dates(manifest, lo, hi):
    ranges = manifest["date_row_ranges"]
    dates = [d for d in range(lo, hi + 1) if str(d) in ranges]
    parts = [np.arange(*ranges[str(d)], dtype=np.int64) for d in dates]
    return dates, np.concatenate(parts)


def combo_block(X, names, accepted):
    cols = []
    for c in accepted:
        va = X[:, names.index(c["a"])].astype(np.float32)
        vb = X[:, names.index(c["b"])].astype(np.float32)
        cols.append(OPS[c["op"]](va, vb).astype(np.float32))
    return np.column_stack(cols)


RANK_COLS = ["feature_37", "feature_38", "feature_60", "feature_45",
             "feature_06", "feature_13"]


def rank_block(M, names, date_spans, step):
    """Cross-sectional rank in [0,1] per (date,time) for the 6 atlas
    features, computed from the assembled base columns.

    Memmap order within a date is symbol-major blocks of 968 rows; global
    ::step preserves per-date phase (968 % step == 0), so each date span
    reshapes to (S, 968//step) and ranking runs across the symbol axis —
    all symbols present at every retained timestep."""
    t_s = 968 // step
    assert 968 % step == 0
    idx = [names.index(c) for c in RANK_COLS]
    out = np.empty((M.shape[0], len(idx)), np.float32)
    for a, b in date_spans:
        s_d = (b - a) // t_s
        for j, ci in enumerate(idx):
            v = M[a:b, ci].reshape(s_d, t_s)
            rk = np.argsort(np.argsort(v, axis=0), axis=0).astype(np.float32)
            out[a:b, j] = (rk / max(1, s_d - 1)).reshape(-1)
    return out


def date_spans_subsampled(manifest, dates, step):
    """Per-date [start, end) spans inside the ::step-subsampled row array."""
    ranges = manifest["date_row_ranges"]
    spans, c = [], 0
    for d in dates:
        lo, hi = ranges[str(d)]
        n = (hi - lo) // step
        spans.append((c, c + n))
        c += n
    return spans


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="artifacts/precomputed/pool700_lags")
    p.add_argument("--latents", default="artifacts/bench/ae_lab_sup/latents.npz")
    p.add_argument("--combos", default="artifacts/bench/drw_lab_v2/combos.json")
    p.add_argument("--clusters", default="artifacts/bench/ablation_symbol.json")
    p.add_argument("--train-lo", type=int, default=1399)
    p.add_argument("--train-hi", type=int, default=1598)
    p.add_argument("--valid-lo", type=int, default=1599)
    p.add_argument("--valid-hi", type=int, default=1698)
    p.add_argument("--row-step", type=int, default=1)
    p.add_argument("--valid-row-step", type=int, default=None,
                   help="valid-side subsample (default: same as --row-step). "
                        "Set 1 with a subsampled train to emit FULL-ROW tail "
                        "predictions for the blend from a window too large "
                        "to train full-row locally")
    p.add_argument("--n-estimators", type=int, default=1500)
    p.add_argument("--with-chain", action="store_true",
                   help="add the 9 stage-1 responder predictions (chain "
                        "block, +0.00069 standalone). Train-side features "
                        "come from 2-fold out-of-fold stage-1 fits; the "
                        "tail's from a full-train refit — textbook stacking "
                        "hygiene, no in-sample stage-1 anywhere")
    p.add_argument("--chain-row-step", type=int, default=2,
                   help="row subsample for stage-1 fits (predictions are "
                        "made on full rows regardless)")
    p.add_argument("--chain-estimators", type=int, default=800)
    p.add_argument("--chain-cache", default=None,
                   help="npz path for the full-row stage-1 predictions; "
                        "loaded if it exists, written after computing "
                        "otherwise — the OOF fits run once per window")
    p.add_argument("--with-ranks", action="store_true",
                   help="add 6 cross-sectional rank columns (per-(date,"
                        "time) rank across symbols of the atlas features)")
    p.add_argument("--variants", default=None,
                   help="comma-separated subset of base,all_but_ae,all,"
                        "all_chain,all_chain_ranks (default: all valid "
                        "for the flags given)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/bench/pool_rebuild")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"

    def log(msg):
        line = msg if msg.endswith("\n") else msg + "\n"
        sys.stdout.write(line); sys.stdout.flush()
        with log_path.open("a") as f:
            f.write(line)

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    names = manifest["feature_cols"]
    K = manifest["K"]
    n_base_lagfree = K - len(manifest.get("lag_cols", []))
    X_mm = np.memmap(data / manifest["X_file"],
                     dtype=np.float16 if manifest["precision"] == "float16" else np.float32,
                     mode="r", shape=(manifest["N"], K))
    y = np.load(data / "y.f32.npy", mmap_mode="r")
    w = np.load(data / "w.f32.npy", mmap_mode="r")
    syms = np.load(data / "symbols.i16.npy", mmap_mode="r")

    accepted = json.loads(Path(args.combos).read_text())["accepted"]
    cluster_map = {int(k): int(v) for k, v in
                   json.loads(Path(args.clusters).read_text())["cluster_map"].items()}
    n_clusters = max(cluster_map.values())
    blob = np.load(args.latents)

    tr_dates, tr = rows_for_dates(manifest, args.train_lo, args.train_hi)
    va_dates, va = rows_for_dates(manifest, args.valid_lo, args.valid_hi)
    if not (np.array_equal(blob["rows_train"], tr)
            and np.array_equal(blob["rows_valid"], va)):
        raise ValueError("AE latent rows do not match the harness rows")

    n_ae = blob["z_train"].shape[1]
    n_combo = len(accepted)
    n_dec, n_innov, n_rsig = 2 * len(BLOCK), len(TRIO) * len(INNOV_DELTAS), 5
    n_chain = 9 if args.with_chain else 0
    n_ranks = len(RANK_COLS) if args.with_ranks else 0
    total = K + n_combo + n_dec + n_rsig + n_innov + n_clusters + n_ae
    total_ext = total + n_chain + n_ranks
    log(f"blocks: base {K} + combos {n_combo} + decomp {n_dec} + rsig {n_rsig}"
        f" + innov {n_innov} + cluster {n_clusters} + ae {n_ae}"
        + (f" + chain {n_chain}" if n_chain else "")
        + (f" + ranks {n_ranks}" if n_ranks else "")
        + f" = {total_ext}")

    step = args.row_step
    vstep = args.valid_row_step if args.valid_row_step is not None else step

    def assemble(dates, rows, z, chain_full, step):
        """In-place block assembly — no hstack doubling. Per-date computed
        blocks (decomp, innovations) are built full-range then subsampled;
        everything else is subsampled before allocation. Chain and rank
        columns live at the tail of the same allocation so every variant
        is a prefix slice of one matrix."""
        rows_s = rows[::step]
        n = len(rows_s)
        M = np.empty((n, total_ext), dtype=np.float32)
        M[:, :K] = X_mm[rows_s]
        c = K
        M[:, c:c + n_combo] = combo_block(M[:, :K], names, accepted); c += n_combo
        M[:, c:c + n_dec] = decomp_features(X_mm, manifest, dates,
                                            [names.index(b) for b in BLOCK])[::step]
        c += n_dec
        M[:, c:c + n_rsig] = signal_features(M[:, n_base_lagfree:K]); c += n_rsig
        M[:, c:c + n_innov] = innovation_features(X_mm, manifest, dates,
                                                  [names.index(t) for t in TRIO])[::step]
        c += n_innov
        cl = np.array([cluster_map.get(int(s), 0) for s in syms[rows_s]])
        for j in range(n_clusters):
            M[:, c + j] = (cl == j + 1).astype(np.float32)
        c += n_clusters
        M[:, c:c + n_ae] = z[::step]; c += n_ae
        if n_chain:
            M[:, c:c + n_chain] = chain_full[::step]; c += n_chain
        if n_ranks:
            spans = date_spans_subsampled(manifest, dates, step)
            M[:, c:c + n_ranks] = rank_block(M, names, spans, step)
            c += n_ranks
        assert c == total_ext
        return M

    # ---- optional chain block (BEFORE the big assembly, to bound memory) ---
    chain_tr = chain_va = None
    cache = Path(args.chain_cache) if args.chain_cache else None
    if args.with_chain and cache is not None and cache.exists():
        blob_c = np.load(cache)
        chain_tr, chain_va = blob_c["chain_tr"], blob_c["chain_va"]
        if len(chain_tr) != len(tr) or len(chain_va) != len(va):
            raise ValueError("chain cache rows do not match the harness rows")
        log(f"[chain] loaded full-row cache {cache}")
    elif args.with_chain:
        log("[chain] stage-1: 9 responders × 3 fits (2-fold OOF + full-train)")
        cs = args.chain_row_step
        mid = (args.train_lo + args.train_hi) // 2
        folds = [  # (fit_lo, fit_hi, predict_rows_selector)
            (args.train_lo, mid, "B"),      # fit first half  → predict second
            (mid + 1, args.train_hi, "A"),  # fit second half → predict first
            (args.train_lo, args.train_hi, "V"),  # full train → predict tail
        ]
        _, ra = rows_for_dates(manifest, args.train_lo, mid)
        _, rb = rows_for_dates(manifest, mid + 1, args.train_hi)
        Ra = load_raw_responders(args.train_lo, mid)
        Rb = load_raw_responders(mid + 1, args.train_hi)
        chain_tr = np.empty((len(tr), 9), np.float32)
        chain_va = np.empty((len(va), 9), np.float32)
        n_a = len(ra)
        for lo_f, hi_f, dest in folds:
            _, rf = rows_for_dates(manifest, lo_f, hi_f)
            rf_s = rf[::cs]
            Xf = np.ascontiguousarray(X_mm[rf_s]).astype(np.float32)
            Rf = (np.concatenate([Ra, Rb]) if dest == "V"
                  else (Ra if dest == "B" else Rb))[::cs]
            wf = np.ascontiguousarray(w[rf_s])
            if dest == "V":
                Xp = np.ascontiguousarray(X_mm[va]).astype(np.float32)
            elif dest == "B":
                Xp = np.ascontiguousarray(X_mm[rb]).astype(np.float32)
            else:
                Xp = np.ascontiguousarray(X_mm[ra]).astype(np.float32)
            for i_r in range(9):
                m = make_xgb(args.chain_estimators, args.seed)
                m.set_params(early_stopping_rounds=None)  # OOF: no eval set
                m.fit(Xf, Rf[:, i_r], sample_weight=wf, verbose=False)
                pred = m.predict(Xp)
                if dest == "V":
                    chain_va[:, i_r] = pred
                elif dest == "B":
                    chain_tr[n_a:, i_r] = pred
                else:
                    chain_tr[:n_a, i_r] = pred
                del m
                gc.collect()
            log(f"[chain] fold fit {lo_f}-{hi_f} → {dest} done")
            del Xf, Xp
            gc.collect()
        del Ra, Rb
        gc.collect()
        if cache is not None:
            np.savez_compressed(cache, chain_tr=chain_tr, chain_va=chain_va)
            log(f"[chain] full-row cache saved → {cache}")

    Xtr = assemble(tr_dates, tr, blob["z_train"], chain_tr, step)
    del chain_tr
    Xva = assemble(va_dates, va, blob["z_valid"], chain_va, vstep)
    del chain_va
    ytr, wtr = np.ascontiguousarray(y[tr][::step]), np.ascontiguousarray(w[tr][::step])
    yva, wva = np.ascontiguousarray(y[va][::vstep]), np.ascontiguousarray(w[va][::vstep])
    gc.collect()
    log(f"assembled: train {Xtr.shape}, valid {Xva.shape}")

    n_no_ae = total - n_ae
    widths = {"base": K, "all_but_ae": n_no_ae, "all": total}
    if args.with_chain:
        widths["all_chain"] = total + n_chain
    if args.with_ranks:
        widths["all_chain_ranks" if args.with_chain else "all_ranks"] = total_ext
    if args.variants:
        wanted = args.variants.split(",")
        unknown = [v for v in wanted if v not in widths]
        if unknown:
            raise ValueError(f"unknown variants {unknown}; have {list(widths)}")
        widths = {v: widths[v] for v in wanted}
    variants = {v: (Xtr[:, :n_c], Xva[:, :n_c]) for v, n_c in widths.items()}
    results = []
    preds_dir = out / "preds"
    preds_dir.mkdir(exist_ok=True)
    for vname, (xt, xv) in variants.items():
        # slices of a C-contiguous matrix: XGB copies internally per fit,
        # one variant at a time — peak stays bounded
        r2, best_it, pred = fit_eval(np.ascontiguousarray(xt), ytr, wtr,
                                     np.ascontiguousarray(xv), yva, wva,
                                     args.n_estimators, args.seed)
        log(f"  {vname:11s} n_feat={xt.shape[1]:>3} best_iter={best_it:>4} "
            f"R²={r2:+.5f}")
        np.savez_compressed(preds_dir / f"xgb_{vname}.npz",
                            pred=pred.astype(np.float32))
        results.append({"variant": vname, "n_features": int(xt.shape[1]),
                        "best_iter": best_it, "r2_weighted": r2})
        gc.collect()

    base_r2 = results[0]["r2_weighted"]
    for r in results:
        r["delta_vs_base"] = r["r2_weighted"] - base_r2
    (out / "ablation.json").write_text(json.dumps({
        "data": str(data), "row_step": step,
        "blocks": {"combos": n_combo, "decomp": n_dec, "rsig": n_rsig,
                   "innov": n_innov, "cluster": n_clusters, "ae": n_ae,
                   "chain": n_chain, "ranks": n_ranks},
        "train": [args.train_lo, args.train_hi],
        "valid": [args.valid_lo, args.valid_hi],
        "results": results, "timestamp": datetime.now().isoformat(),
    }, indent=2))
    log("deltas vs base: " + "  ".join(
        f"{r['variant']} {r['delta_vs_base']:+.5f}" for r in results[1:])
        + f" → {out}/ablation.json")


if __name__ == "__main__":
    main()
