"""Feature atlas — 555562-style reverse-engineering research on the 79 features.

Kaggle discussion #555562 (John Payne) reverse-engineered the *responders*:
ACF cutoffs pin them as forward-shifted SMAs (windows 4/20/120) of a common
underlying signal, on two exchanges, plus noise. Features at time t predict
the *realized* responder_6(t-20) with linear R² ≈ 0.5 — so the features are
mostly a rich description of the recent path, and the target is its forward
continuation.

This script applies the same methodology to the features themselves, plus a
step nobody published: **deconvolve responder_8** (the finest SMA-4) back
into an estimate of the underlying per-minute signal ŝ, then map every
feature's lead-lag correlation against ŝ. Features correlating with ŝ at
k ≥ +1 (strictly future signal) carry genuine alpha; features correlating at
k ≤ 0 are nowcast/descriptive.

Per raw feature (and, as sanity references, each responder):
  * null rate
  * level-ACF: lag-1 persistence, half-life, cutoff lag, triangular-ramp
    linearity (SMA-of-noise signature — flags features that are themselves
    moving averages, hence deconvolvable)
  * cross-day continuity (continuous series vs daily reset)
  * cross-sectional market share (avg pairwise corr across symbols) — is the
    feature market-wide or idiosyncratic?
  * intraday-profile share (deterministic time-of-day component)
  * lead-lag curve vs deconvolved ŝ and vs the Δr8 increment proxy:
    alpha score = mean corr over k ∈ [+1, +20] (the target's window),
    nowcast score = mean corr over k ∈ [-20, 0], peak corr + peak lag
  * univariate corr with the actual target responder_6
  * tags from features.csv

Outputs (default artifacts/feature_atlas/):
  atlas.json    per-feature metrics
  ATLAS.md      human-readable ranked tables
  curves.npz    mean ACF + lead-lag curves per feature (for plotting)

Usage
-----
    uv run python scripts/profile_features.py \\
        --min-date 1500 --max-date 1580 --out artifacts/feature_atlas
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.config import COL_DATE, COL_ID, COL_TIME, COLS_FEATURES_INIT, Cfg
from janestreet.data.ingest import scan_train_dates

RESPONDERS = [f"responder_{i}" for i in range(9)]
SMA_W = 4               # responder_8 window (ACF cutoff at lag 4, #555562)
TARGET_BAND = (1, 20)   # responder_6 = forward SMA-20 → alpha band k ∈ [1, 20]
NOWCAST_BAND = (-20, 0)


# ---------------------------------------------------------------------------
# series extraction
# ---------------------------------------------------------------------------

def load_series(cfg: Cfg, lo: int, hi: int, cols: list[str]):
    """Return (data, meta): data[c] = (n_series, T) float32, one row per
    complete (symbol, date) group; meta rows are (symbol, date)."""
    lf = scan_train_dates(cfg, lo, hi).select(
        [COL_ID, COL_DATE, COL_TIME, *cols]
    )
    df = lf.collect().sort([COL_ID, COL_DATE, COL_TIME])
    t_per = cfg.n_times_per_date

    counts = df.group_by([COL_ID, COL_DATE], maintain_order=True).len()
    complete = counts.filter(pl.col("len") == t_per)
    if complete.height < counts.height:
        df = df.join(
            complete.select([COL_ID, COL_DATE]), on=[COL_ID, COL_DATE], how="inner"
        ).sort([COL_ID, COL_DATE, COL_TIME])
    n_series = df.height // t_per
    meta = (
        df.select([COL_ID, COL_DATE])
        .gather_every(t_per)
        .to_numpy()
        .astype(np.int32)
    )

    data = {}
    for c in cols:
        v = df.get_column(c).to_numpy().astype(np.float32)
        data[c] = v.reshape(n_series, t_per)
    return data, meta


# ---------------------------------------------------------------------------
# FFT correlation machinery
# ---------------------------------------------------------------------------

def _normalize_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-impute NaNs, demean, unit-std per row. Returns (z, valid_mask)."""
    x = x.astype(np.float64, copy=True)
    nan = np.isnan(x)
    null_frac = nan.mean(axis=1)
    mu = np.nanmean(np.where(nan, np.nan, x), axis=1, keepdims=True)
    mu = np.nan_to_num(mu)
    x = np.where(nan, mu, x) - mu
    sd = x.std(axis=1, keepdims=True)
    valid = (sd[:, 0] > 1e-9) & (null_frac < 0.5)
    z = np.divide(x, sd, out=np.zeros_like(x), where=sd > 1e-9)
    return z, valid


def crosscorr(zf: np.ndarray, zs: np.ndarray, max_shift: int) -> np.ndarray:
    """Per-row corr(f(t), s(t+k)) for k in [-max_shift, +max_shift].

    Rows must be demeaned/unit-std. Returns (n, 2*max_shift+1).
    """
    n, t = zf.shape
    fft_len = 1 << int(np.ceil(np.log2(2 * t)))
    ff = np.fft.rfft(zf, n=fft_len, axis=1)
    sf = np.fft.rfft(zs, n=fft_len, axis=1)
    cc = np.fft.irfft(np.conj(ff) * sf, n=fft_len, axis=1)
    ks = np.arange(-max_shift, max_shift + 1)
    idx = np.where(ks >= 0, ks, fft_len + ks)
    return cc[:, idx] / (t - np.abs(ks))[None, :]


def mean_acf(z: np.ndarray, valid: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean autocorrelation (lags 0..max_lag) over valid rows."""
    if valid.sum() == 0:
        return np.zeros(max_lag + 1)
    cc = crosscorr(z[valid], z[valid], max_lag)
    acf = cc[:, max_lag:]              # k >= 0
    a0 = acf[:, :1]
    acf = np.divide(acf, a0, out=np.zeros_like(acf), where=a0 > 1e-9)
    return acf.mean(axis=0)


def acf_stats(acf: np.ndarray) -> dict:
    """Persistence, half-life, cutoff, and triangular-ramp linearity."""
    lags = np.arange(len(acf))
    below = np.where(acf < 0.05)[0]
    cutoff = int(below[0]) if len(below) else -1
    half = np.where(acf < 0.5)[0]
    halflife = int(half[0]) if len(half) else -1
    lin_r2 = float("nan")
    if 3 <= cutoff <= len(acf) - 1:
        seg, x = acf[: cutoff + 1], lags[: cutoff + 1]
        coef = np.polyfit(x, seg, 1)
        pred = np.polyval(coef, x)
        ss = np.sum((seg - seg.mean()) ** 2)
        lin_r2 = float(1 - np.sum((seg - pred) ** 2) / ss) if ss > 1e-12 else float("nan")
    return {
        "acf1": float(acf[1]) if len(acf) > 1 else float("nan"),
        "halflife": halflife,
        "cutoff_005": cutoff,
        "ramp_linearity_r2": lin_r2,
    }


# ---------------------------------------------------------------------------
# signal recovery
# ---------------------------------------------------------------------------

def deconvolution_matrix(t: int, w: int, lam: float) -> np.ndarray:
    """Ridge inverse of the forward-SMA operator: r(τ) = mean(s[τ..τ+w-1]).

    Returns M (t+w-1, t) with ŝ = M @ r per series. λ regularizes ||s||²
    (the responders carry added noise per #555562, so exact inversion is
    ill-posed).
    """
    ts = t + w - 1
    a = np.zeros((t, ts))
    for i in range(t):
        a[i, i : i + w] = 1.0 / w
    return np.linalg.solve(a.T @ a + lam * np.eye(ts), a.T)


# ---------------------------------------------------------------------------
# aggregate diagnostics
# ---------------------------------------------------------------------------

def cross_day_continuity(x: np.ndarray, meta: np.ndarray) -> float:
    """corr(last value of day d, first value of day d+1) per symbol."""
    tail_v, head_v = [], []
    order = np.lexsort((meta[:, 1], meta[:, 0]))   # by symbol, then date
    m, xs = meta[order], x[order]
    for i in range(len(m) - 1):
        if m[i, 0] == m[i + 1, 0] and m[i + 1, 1] == m[i, 1] + 1:
            a, b = xs[i, -1], xs[i + 1, 0]
            if np.isfinite(a) and np.isfinite(b):
                tail_v.append(a)
                head_v.append(b)
    if len(tail_v) < 30:
        return float("nan")
    tail_a, head_a = np.array(tail_v), np.array(head_v)
    if tail_a.std() < 1e-9 or head_a.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(tail_a, head_a)[0, 1])


def market_share(x: np.ndarray, meta: np.ndarray) -> float:
    """Share of variance explained by the per-(date,time) market mean."""
    dates = np.unique(meta[:, 1])
    num = den = 0.0
    for d in dates:
        rows = meta[:, 1] == d
        if rows.sum() < 3:
            continue
        xd = np.nan_to_num(x[rows])          # (n_sym, T)
        mkt = xd.mean(axis=0, keepdims=True)
        den += float(((xd - xd.mean()) ** 2).sum())
        num += float(((mkt - xd.mean()) ** 2).sum()) * xd.shape[0]
    return num / den if den > 0 else float("nan")


def intraday_profile_share(x: np.ndarray) -> float:
    """Share of variance from the deterministic time-of-day profile."""
    xd = np.nan_to_num(x)
    prof = xd.mean(axis=0)
    tot = float(((xd - xd.mean()) ** 2).mean())
    return float(((prof - prof.mean()) ** 2).mean()) / tot if tot > 1e-12 else float("nan")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-date", type=int, default=1500)
    p.add_argument("--max-date", type=int, default=1580)
    p.add_argument("--out", type=str, default="artifacts/feature_atlas")
    p.add_argument("--max-shift", type=int, default=48)
    p.add_argument("--max-acf-lag", type=int, default=200)
    p.add_argument("--ridge-lambda", type=float, default=0.5)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Cfg()
    t_per = cfg.n_times_per_date

    cols = COLS_FEATURES_INIT + RESPONDERS
    print(f"loading dates {args.min_date}..{args.max_date} ({len(cols)} cols)…", flush=True)
    data, meta = load_series(cfg, args.min_date, args.max_date, cols)
    n_series = meta.shape[0]
    print(f"  {n_series} complete (symbol, date) series × {t_per} steps", flush=True)

    # --- recover the underlying signal from responder_8 (SMA-4) ------------
    print("deconvolving responder_8 → ŝ (ridge SMA-4 inverse)…", flush=True)
    m_inv = deconvolution_matrix(t_per, SMA_W, args.ridge_lambda)
    r8 = np.nan_to_num(data["responder_8"].astype(np.float64))
    s_hat = (r8 @ m_inv.T)[:, :t_per]               # align: ŝ[t] ~ signal at t
    z_sig, sig_valid = _normalize_rows(s_hat)
    # increment proxy: Δr8(t) ∝ s(t+3) − s(t−1) — deconvolution-free check
    d8 = np.diff(r8, axis=1, prepend=r8[:, :1])
    z_d8, d8_valid = _normalize_rows(d8)

    tags = {}
    tag_path = cfg.data_root / "features.csv"
    if tag_path.exists():
        tf = pl.read_csv(tag_path)
        tag_cols = [c for c in tf.columns if c.startswith("tag_")]
        for row in tf.iter_rows(named=True):
            tags[row["feature"]] = [c for c in tag_cols if row[c]]

    ks = np.arange(-args.max_shift, args.max_shift + 1)
    a_lo, a_hi = TARGET_BAND
    n_lo, n_hi = NOWCAST_BAND
    alpha_band = (ks >= a_lo) & (ks <= a_hi)
    nowcast_band = (ks >= n_lo) & (ks <= n_hi)

    z_tgt, tgt_valid = _normalize_rows(np.nan_to_num(data["responder_6"].astype(np.float64)))

    atlas: dict[str, dict] = {}
    curves_acf, curves_ll = {}, {}
    print("profiling columns…", flush=True)
    for i, c in enumerate(cols):
        x = data[c]
        null_rate = float(np.isnan(x).mean())
        z, valid = _normalize_rows(x)
        v = valid & sig_valid

        acf = mean_acf(z, valid, args.max_acf_lag)
        row: dict = {"null_rate": null_rate, "n_valid_series": int(valid.sum())}
        row.update(acf_stats(acf))
        row["cross_day_continuity"] = cross_day_continuity(x, meta)
        row["market_share"] = market_share(x, meta)
        row["intraday_profile_share"] = intraday_profile_share(x)

        if v.sum() > 30:
            ll = crosscorr(z[v], z_sig[v], args.max_shift).mean(axis=0)
            ll_d8 = crosscorr(z[v & d8_valid], z_d8[v & d8_valid], args.max_shift).mean(axis=0)
            pk = int(np.argmax(np.abs(ll)))
            row["alpha_score"] = float(ll[alpha_band].mean())
            row["alpha_score_d8"] = float(ll_d8[alpha_band].mean())
            row["nowcast_score"] = float(ll[nowcast_band].mean())
            row["peak_corr"] = float(ll[pk])
            row["peak_lag"] = int(ks[pk])
            curves_ll[c] = ll.astype(np.float32)
        vt = valid & tgt_valid
        if vt.sum() > 30:
            row["corr_target_t0"] = float((z[vt] * z_tgt[vt]).mean(axis=1).mean())
        row["tags"] = tags.get(c, [])
        atlas[c] = row
        curves_acf[c] = acf.astype(np.float32)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cols)}", flush=True)

    # ---- write outputs -----------------------------------------------------
    meta_out = {
        "min_date": args.min_date, "max_date": args.max_date,
        "n_series": n_series, "ridge_lambda": args.ridge_lambda,
        "alpha_band": list(TARGET_BAND), "nowcast_band": list(NOWCAST_BAND),
        "timestamp": datetime.now().isoformat(),
    }
    (out / "atlas.json").write_text(json.dumps({"meta": meta_out, "columns": atlas}, indent=1))
    np.savez_compressed(
        out / "curves.npz",
        lags=ks,
        **{f"acf__{c}": curves_acf[c] for c in curves_acf},
        **{f"ll__{c}": curves_ll[c] for c in curves_ll},
    )
    write_markdown(out / "ATLAS.md", atlas, meta_out)
    print(f"written → {out}/atlas.json, ATLAS.md, curves.npz", flush=True)


def write_markdown(path: Path, atlas: dict, meta: dict) -> None:
    feats = {c: r for c, r in atlas.items() if c.startswith("feature_")}
    resps = {c: r for c, r in atlas.items() if c.startswith("responder_")}

    def fmt(v, spec=".4f"):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        return f"{v:{spec}}"

    def table(rows, cols, headers):
        lines = ["| col | " + " | ".join(headers) + " |",
                 "|---|" + "---|" * len(headers)]
        for name, r in rows:
            lines.append(
                "| " + name + " | "
                + " | ".join(fmt(r.get(k)) if not isinstance(r.get(k), (int, list)) else str(r.get(k)) for k in cols)
                + " |"
            )
        return "\n".join(lines)

    md = [f"# Feature atlas — dates {meta['min_date']}–{meta['max_date']} "
          f"({meta['n_series']} series)\n",
          f"*Generated {meta['timestamp']}. Alpha band k∈{meta['alpha_band']} on the "
          f"deconvolved responder_8 signal ŝ; nowcast band k∈{meta['nowcast_band']}.*\n"]

    md.append("\n## Sanity: responders vs ŝ (deconvolution check)\n")
    md.append(table(sorted(resps.items()),
                    ["alpha_score", "nowcast_score", "peak_corr", "peak_lag",
                     "cutoff_005", "ramp_linearity_r2"],
                    ["alpha", "nowcast", "peak r", "peak lag", "ACF cutoff", "ramp R²"]))

    by_alpha = sorted(feats.items(), key=lambda kv: -abs(kv[1].get("alpha_score") or 0))
    md.append("\n\n## Top 25 features by |alpha score| (lead the future signal)\n")
    md.append(table(by_alpha[:25],
                    ["alpha_score", "alpha_score_d8", "nowcast_score", "peak_corr",
                     "peak_lag", "corr_target_t0", "market_share"],
                    ["alpha", "alpha(Δr8)", "nowcast", "peak r", "peak lag",
                     "corr r6(t)", "mkt share"]))

    by_now = sorted(feats.items(), key=lambda kv: -abs(kv[1].get("nowcast_score") or 0))
    md.append("\n\n## Top 15 nowcast features (describe the realized path)\n")
    md.append(table(by_now[:15],
                    ["nowcast_score", "alpha_score", "peak_lag", "corr_target_t0"],
                    ["nowcast", "alpha", "peak lag", "corr r6(t)"]))

    sma_like = [(c, r) for c, r in feats.items()
                if (r.get("ramp_linearity_r2") or 0) > 0.97 and 0 < r.get("cutoff_005", -1) <= 300]
    sma_like.sort(key=lambda kv: kv[1]["cutoff_005"])
    md.append("\n\n## SMA-like features (triangular ACF ramp → themselves moving averages?)\n")
    md.append(table(sma_like,
                    ["cutoff_005", "ramp_linearity_r2", "acf1", "alpha_score"],
                    ["ACF cutoff", "ramp R²", "acf1", "alpha"]) if sma_like else "*(none)*")

    by_mkt = sorted(feats.items(), key=lambda kv: -(kv[1].get("market_share") or 0))
    md.append("\n\n## Most market-wide features (top 15 by cross-sectional share)\n")
    md.append(table(by_mkt[:15],
                    ["market_share", "intraday_profile_share", "cross_day_continuity", "alpha_score"],
                    ["mkt share", "intraday share", "x-day cont.", "alpha"]))

    md.append("\n\n## Structure overview (all features)\n")
    md.append(table(sorted(feats.items()),
                    ["acf1", "halflife", "cutoff_005", "cross_day_continuity",
                     "market_share", "null_rate", "alpha_score", "tags"],
                    ["acf1", "halflife", "cutoff", "x-day", "mkt", "nulls", "alpha", "tags"]))
    path.write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
