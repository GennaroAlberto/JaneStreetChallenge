"""Cluster symbols by responder co-movement, and check the clusters are stable.

The covariates are anonymized, but symbols are not homogeneous: their
``responder_6`` series fall into a few co-moving groups (plausibly asset
classes / sectors). If those groups are **temporally stable** — a symbol keeps
its peers across time — we can route each symbol to a specialized model, or
feed the cluster id / a per-cluster cross-sectional average as a feature.

This script:

1. Builds the symbol x symbol ``responder_6`` correlation matrix over a date
   range and clusters it (average-linkage hierarchical, ``k`` clusters).
2. **Stability**: clusters an early window and a late window independently and
   compares the assignments (adjusted Rand index) on the symbols common to
   both. High ARI => membership is durable => routing at test time is safe.
3. Saves the canonical map (clustered on the whole train pool) to
   ``artifacts/clusters/symbol_clusters_k{K}.json`` for the feature builder.

Usage
-----

    uv run python scripts/cluster_symbols.py --k 3 \\
        --pool-lo 700 --pool-hi 1597 --early 800-1099 --late 1298-1597
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")

from janestreet.config import Cfg  # noqa: E402
from janestreet.data.ingest import scan_train_dates  # noqa: E402

TARGET = "responder_6"


def corr_matrix(cfg: Cfg, lo: int, hi: int) -> tuple[list[int], np.ndarray]:
    """symbol x symbol correlation of ``responder_6`` over dates [lo, hi]."""
    df = (
        scan_train_dates(cfg, min_date=lo, max_date=hi)
        .select(["date_id", "time_id", "symbol_id", TARGET])
        .collect()
    )
    piv = df.pivot(values=TARGET, index=["date_id", "time_id"], on="symbol_id")
    cols = [c for c in piv.columns if c not in ("date_id", "time_id")]
    M = piv.select(cols).to_numpy()
    n = len(cols)
    C = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i, n):
            a, b = M[:, i], M[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 500:
                c = np.corrcoef(a[m], b[m])[0, 1]
                C[i, j] = C[j, i] = c
    return [int(c) for c in cols], C


def cluster(cols: list[int], C: np.ndarray, k: int) -> dict[int, int]:
    """Average-linkage clustering of symbols into ``k`` groups. -> {symbol: label}."""
    Cf = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(Cf, 1.0)
    d = 1.0 - Cf
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    Z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(Z, k, criterion="maxclust")
    return {int(s): int(lab) for s, lab in zip(cols, labels, strict=True)}


def intra_inter(cols: list[int], C: np.ndarray, lab: dict[int, int]) -> tuple[float, float]:
    intra, inter = [], []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = C[i, j]
            if not np.isfinite(c):
                continue
            (intra if lab[cols[i]] == lab[cols[j]] else inter).append(c)
    return float(np.mean(intra)), float(np.mean(inter))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--pool-lo", type=int, default=700)
    p.add_argument("--pool-hi", type=int, default=1597)
    p.add_argument("--early", type=str, default="800-1099")
    p.add_argument("--late", type=str, default="1298-1597")
    p.add_argument("--out", type=str, default="artifacts/clusters")
    args = p.parse_args()
    cfg = Cfg()

    e_lo, e_hi = (int(x) for x in args.early.split("-"))
    l_lo, l_hi = (int(x) for x in args.late.split("-"))

    # --- Stability: cluster early and late windows independently ---
    ce_cols, ce = corr_matrix(cfg, e_lo, e_hi)
    cl_cols, cl = corr_matrix(cfg, l_lo, l_hi)
    lab_e = cluster(ce_cols, ce, args.k)
    lab_l = cluster(cl_cols, cl, args.k)
    common = sorted(set(lab_e) & set(lab_l))
    ari = adjusted_rand_score([lab_e[s] for s in common], [lab_l[s] for s in common])
    print(f"=== stability (k={args.k}) ===")
    print(f"  early {e_lo}-{e_hi}: {len(ce_cols)} symbols   late {l_lo}-{l_hi}: {len(cl_cols)} symbols"
          f"   common: {len(common)}")
    print(f"  adjusted Rand index (early vs late): {ari:+.3f}"
          f"   [1=identical, 0=chance]  ->  {'STABLE' if ari > 0.4 else 'UNSTABLE'}")

    # --- Canonical map on the full train pool ---
    cols, C = corr_matrix(cfg, args.pool_lo, args.pool_hi)
    lab = cluster(cols, C, args.k)
    intra, inter = intra_inter(cols, C, lab)
    sizes = np.bincount(list(lab.values()))[1:]
    print(f"\n=== canonical clustering on pool {args.pool_lo}-{args.pool_hi} ===")
    print(f"  {len(cols)} symbols -> k={args.k}: sizes={sorted(sizes.tolist(), reverse=True)}")
    print(f"  intra-cluster corr={intra:+.3f}   inter-cluster corr={inter:+.3f}"
          f"   (ratio {intra / max(inter, 1e-6):.1f}x)")
    for c in sorted(set(lab.values())):
        members = sorted(s for s, la in lab.items() if la == c)
        print(f"    cluster {c}: {members}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"symbol_clusters_k{args.k}.json"
    dest.write_text(json.dumps(
        {"k": args.k, "pool": [args.pool_lo, args.pool_hi],
         "stability_ari": ari, "intra_corr": intra, "inter_corr": inter,
         "map": {str(s): int(la) for s, la in sorted(lab.items())}}, indent=2))
    print(f"\nsaved canonical map -> {dest}")


if __name__ == "__main__":
    main()
