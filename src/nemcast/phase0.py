"""Phase 0 spike: zero-shot Timestral vs AEMO forecasts and naive baselines.

Times are NEM time (AEST, naive timestamps) and label the *end* of each interval, as in AEMO data.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

REGIONS = ["NSW1", "QLD1", "SA1", "TAS1", "VIC1"]
LAKE = Path("data/lake")
RESULTS = Path("data/results")
QUANTILE_LEVELS = (0.01,) + tuple(round(0.05 * i, 2) for i in range(1, 20)) + (0.99,)  # Timestral's 21 knots
Q_COLS = [f"q{int(round(q * 100)):02d}" for q in QUANTILE_LEVELS]
PD_HISTORY_LEAD = pd.Timedelta(hours=12)
PD_CLIP = 1000.0  # $/MWh; PD often shows cap prices a day ahead that never happen


# ---------------------------------------------------------------- data


def load_prices_5m(mmsdm_parquet: str | None = None) -> pd.DataFrame:
    """Wide 5-min RRP (index: interval end, columns: regions) on a complete grid."""
    parts = [f"SELECT SETTLEMENTDATE, REGIONID, RRP FROM read_parquet('{LAKE}/dispatch_price/*.parquet')"]
    if mmsdm_parquet:
        parts.append(
            f"SELECT SETTLEMENTDATE, REGIONID, RRP FROM read_parquet('{mmsdm_parquet}') WHERE INTERVENTION = 0"
        )
    sql = " UNION ALL ".join(parts)
    df = duckdb.sql(f"SELECT DISTINCT * FROM ({sql})").df()
    wide = df.pivot_table(index="SETTLEMENTDATE", columns="REGIONID", values="RRP", aggfunc="mean")[REGIONS]
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq="5min")
    return wide.reindex(grid)


def to_30m(p5: pd.DataFrame) -> pd.DataFrame:
    """Trading-interval price: mean of the six 5-min prices ending in (t-30min, t]."""
    g = p5.resample("30min", label="right", closed="right")
    return g.mean().where(g.count() == 6)


def load_predispatch() -> pd.DataFrame:
    """Pre-dispatch runs with RRP and reserve margin (`RESERVE` = available generation - demand, MW)."""
    return duckdb.sql(
        f"""
        SELECT p.RUN_DATETIME, p.PUBLISHED, p.DATETIME, p.REGIONID, p.RRP,
               s.AVAILABLEGENERATION - s.TOTALDEMAND AS RESERVE
        FROM read_parquet('{LAKE}/predispatch_region_prices/*.parquet') p
        LEFT JOIN read_parquet('{LAKE}/predispatch_region_solution/*.parquet') s
          USING (RUN_DATETIME, PUBLISHED, DATETIME, REGIONID)
        """
    ).df()


def load_p5min() -> pd.DataFrame:
    return duckdb.sql(
        f"SELECT RUN_DATETIME, INTERVAL_DATETIME, REGIONID, RRP "
        f"FROM read_parquet('{LAKE}/p5min_regionsolution/*.parquet')"
    ).df()


def pd_latest(pdf: pd.DataFrame, issue: pd.Timestamp, col: str = "RRP") -> pd.DataFrame:
    """Latest pre-dispatch run published at or before `issue`, wide (DATETIME x region)."""
    avail = pdf[pdf["PUBLISHED"] <= issue]
    run = avail["RUN_DATETIME"].max()
    one = avail[avail["RUN_DATETIME"] == run]
    return one.pivot_table(index="DATETIME", columns="REGIONID", values=col)[REGIONS]


def pd_history(pdf: pd.DataFrame, lead: pd.Timedelta = PD_HISTORY_LEAD, col: str = "RRP") -> pd.DataFrame:
    """For each interval, the pre-dispatch price from the latest run published at least `lead` before it."""
    ok = pdf[pdf["PUBLISHED"] <= pdf["DATETIME"] - lead].sort_values("PUBLISHED")
    last = ok.groupby(["DATETIME", "REGIONID"], as_index=False).last()
    return last.pivot_table(index="DATETIME", columns="REGIONID", values=col)[REGIONS]


# ---------------------------------------------------------------- model


def load_model(size: str) -> TimestralForecaster:
    return TimestralForecaster.from_pretrained(f"mistralai/Timestral-{size}")


def forecast(
    fc: TimestralForecaster,
    context: np.ndarray,
    horizon: int,
    known: np.ndarray | None = None,
    symmetric: bool = True,
) -> np.ndarray:
    """Quantiles `(B, V, H, 21)`; context `(B, V, T)` is trimmed to the model's max context."""
    t = min(context.shape[-1], fc.max_context)
    ctx = context[..., -t:]
    kc = known[..., -(t + horizon) :] if known is not None else None
    with torch.inference_mode():
        out = fc.forecast(
            ctx.astype(np.float32),
            prediction_length=horizon,
            known_covariates=None if kc is None else kc.astype(np.float32),
            symmetric_averaging=symmetric,
            clip_nonnegative=False,
        )
    return out.quantiles.cpu().numpy()


def _quantile_frame(q: np.ndarray, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """`q (5, H, 21)` -> {region: DataFrame(index=valid_time, columns=Q_COLS)}."""
    return {r: pd.DataFrame(q[i], index=index, columns=Q_COLS) for i, r in enumerate(REGIONS)}


# ---------------------------------------------------------------- backtests


def _same_time_last_known(series: pd.DataFrame, valid: pd.DatetimeIndex, last_known: pd.Timestamp) -> pd.DataFrame:
    """Seasonal-naive: value at the same time of day on the most recent day already observed."""
    days = np.maximum(np.ceil((valid - last_known) / pd.Timedelta(days=1)).to_numpy().astype(int), 1)
    src = valid - pd.to_timedelta(days, unit="D")
    return series.reindex(src).set_axis(valid)


def run_day_ahead(
    p30: pd.DataFrame,
    pdf: pd.DataFrame,
    pd_hist: dict[str, pd.DataFrame],
    fc: TimestralForecaster,
    days: list[pd.Timestamp],
    mode: str,
    symmetric: bool = True,
    issue_hm: str = "12:45",
) -> pd.DataFrame:
    """DA-30: issue on day D at `issue_hm`, score the next trading day (D+1 04:30 .. D+2 04:00).

    mode: "uni" (5 series x 1 variate), "multi" (1 series x 5 variates), "uni_pd" (uni + PD RRP as known covariate),
    "uni_pd_rm" (uni_pd + PD reserve margin as a second known covariate); "uni_pdc" / "uni_pdc_rm" are the same
    with the PD price covariate clipped to +/-`PD_CLIP`.
    Known covariates use, over the context, the PD value issued `PD_HISTORY_LEAD` ahead, and over the horizon
    the latest PD run at issue time, so no actuals leak in.
    """
    rows = []
    for day in days:
        issue = day + pd.Timedelta(issue_hm + ":00")
        last_known = issue.floor("30min")
        horizon_end = day + pd.Timedelta(days=2, hours=4)
        horizon_idx = pd.date_range(last_known + pd.Timedelta("30min"), horizon_end, freq="30min")
        target_idx = horizon_idx[horizon_idx > day + pd.Timedelta(days=1, hours=4)]
        hist = p30.loc[:last_known]
        ctx = hist.to_numpy().T  # (5, T)
        h = len(horizon_idx)
        latest = pd_latest(pdf, issue).reindex(horizon_idx)
        known = None
        if mode == "multi":
            q = forecast(fc, ctx[None], h, symmetric=symmetric)[0]
        else:
            if mode.startswith("uni_pd"):
                cols = ["RRP", "RESERVE"] if mode.endswith("_rm") else ["RRP"]
                known = np.stack(
                    [
                        np.concatenate(
                            [
                                pd_hist[c].reindex(hist.index).to_numpy().T,
                                pd_latest(pdf, issue, c).reindex(horizon_idx).to_numpy().T,
                            ],
                            axis=1,
                        )
                        for c in cols
                    ],
                    axis=1,
                )  # (5, C, T + H)
                if mode.startswith("uni_pdc"):
                    known[:, 0] = np.clip(known[:, 0], -PD_CLIP, PD_CLIP)
            q = forecast(fc, ctx[:, None, :], h, known=known, symmetric=symmetric)[:, 0]
        frames = _quantile_frame(q, horizon_idx)
        naive_day = _same_time_last_known(p30, target_idx, last_known)
        naive_week = p30.reindex(target_idx - pd.Timedelta(days=7)).set_axis(target_idx)
        actual = p30.reindex(target_idx)
        for r in REGIONS:
            f = frames[r].loc[target_idx]
            rows.append(
                f.assign(
                    issue=issue,
                    region=r,
                    actual=actual[r].to_numpy(),
                    aemo_pd=latest[r].reindex(target_idx).to_numpy(),
                    naive_day=naive_day[r].to_numpy(),
                    naive_week=naive_week[r].to_numpy(),
                )
                .rename_axis("valid_time")
                .reset_index()
            )
    return pd.concat(rows, ignore_index=True)


def run_intraday_5m(
    p5: pd.DataFrame,
    p5min: pd.DataFrame,
    pdf: pd.DataFrame,
    fc: TimestralForecaster,
    issues: list[pd.Timestamp],
    horizon: int = 144,
    symmetric: bool = True,
) -> pd.DataFrame:
    """ID-5: at each issue time (a 5-min boundary, after that interval's dispatch price is out), forecast 12h."""
    rows = []
    p5min_runs = {k: g for k, g in p5min.groupby("RUN_DATETIME")}
    for issue in issues:
        horizon_idx = pd.date_range(issue + pd.Timedelta("5min"), periods=horizon, freq="5min")
        hist = p5.loc[:issue]
        q = forecast(fc, hist.to_numpy().T[:, None, :], horizon, symmetric=symmetric)[:, 0]
        frames = _quantile_frame(q, horizon_idx)
        run = p5min_runs.get(issue + pd.Timedelta("5min"))
        p5m = (
            run.pivot_table(index="INTERVAL_DATETIME", columns="REGIONID", values="RRP")[REGIONS].reindex(horizon_idx)
            if run is not None
            else pd.DataFrame(np.nan, index=horizon_idx, columns=REGIONS)
        )
        pd30 = pd_latest(pdf, issue)
        pd5 = pd30.reindex(horizon_idx.ceil("30min")).set_axis(horizon_idx)
        actual = p5.reindex(horizon_idx)
        naive_day = p5.reindex(horizon_idx - pd.Timedelta(days=1)).set_axis(horizon_idx)
        persist = p5.loc[issue]
        for r in REGIONS:
            rows.append(
                frames[r]
                .assign(
                    issue=issue,
                    region=r,
                    lead_min=np.arange(1, horizon + 1) * 5,
                    actual=actual[r].to_numpy(),
                    aemo_p5min=p5m[r].to_numpy(),
                    aemo_pd=pd5[r].to_numpy(),
                    naive_day=naive_day[r].to_numpy(),
                    persistence=persist[r],
                )
                .rename_axis("valid_time")
                .reset_index()
            )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- metrics


def pinball(df: pd.DataFrame) -> pd.Series:
    """Mean pinball loss across the 21 quantiles (a discrete approximation of CRPS)."""
    y = df["actual"].to_numpy()[:, None]
    q = df[Q_COLS].to_numpy()
    tau = np.array(QUANTILE_LEVELS)[None, :]
    diff = y - q
    return pd.Series(np.maximum(tau * diff, (tau - 1) * diff).mean(axis=1), index=df.index)


def score(df: pd.DataFrame, point_cols: list[str], by: list[str]) -> pd.DataFrame:
    df = df.dropna(subset=["actual"]).copy()
    df["timestral_p50"] = df["q50"]
    df["_pinball"] = pinball(df)
    df["_cover80"] = (df["actual"] >= df["q10"]) & (df["actual"] <= df["q90"])
    out = {}
    g = df.groupby(by) if by else [("all", df)]
    for key, d in g:
        row = {"n": len(d), "timestral_pinball": d["_pinball"].mean(), "timestral_cover80": d["_cover80"].mean()}
        for c in ["timestral_p50", *point_cols]:
            ok = d[c].notna()
            err = d.loc[ok, c] - d.loc[ok, "actual"]
            row[f"{c}_mae"] = err.abs().mean()
            row[f"{c}_medae"] = err.abs().median()
            row[f"{c}_rmse"] = np.sqrt((err**2).mean())
        out[key] = row
    res = pd.DataFrame.from_dict(out, orient="index")
    if by:
        res.index.names = by
    return res


# ---------------------------------------------------------------- CLI


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mmsdm",
        default=None,
        help="optional MMSDM DISPATCHPRICE Parquet (e.g. from NEMOSIS) to extend price history for long context",
    )
    p.add_argument("--start", default="2026-10-01")
    p.add_argument("--end", default="2026-10-31")
    p.add_argument("--da-models", nargs="*", default=["28M"])
    p.add_argument("--every-days", type=int, default=1, help="DA-30: use every n-th day (to save CPU)")
    p.add_argument("--da-modes", nargs="*", default=["uni", "multi", "uni_pd", "uni_pd_rm", "uni_pdc", "uni_pdc_rm"])
    p.add_argument("--id-model", default="28M")
    p.add_argument("--id-every-hours", type=int, default=3)
    p.add_argument("--id-symmetric", action="store_true")
    p.add_argument("--skip-da", action="store_true")
    p.add_argument("--skip-id", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    p5 = load_prices_5m(args.mmsdm)
    p30 = to_30m(p5)
    pdf = load_predispatch()
    days = list(pd.date_range(args.start, args.end, freq="D"))

    if not args.skip_da:
        hist = {c: pd_history(pdf, col=c) for c in ("RRP", "RESERVE")}
        for size in args.da_models:
            fc = load_model(size)
            for mode in args.da_modes:
                t0 = time.perf_counter()
                res = run_day_ahead(p30, pdf, hist, fc, days[:: args.every_days], mode)
                res.assign(model=size, mode=mode).to_parquet(RESULTS / f"da30_{size}_{mode}.parquet", index=False)
                print(f"DA-30 {size} {mode}: {time.perf_counter() - t0:.0f}s", flush=True)

    if not args.skip_id:
        p5min = load_p5min()
        fc = load_model(args.id_model)
        issues = [
            ts
            for ts in pd.date_range(args.start, pd.Timestamp(args.end) + pd.Timedelta("23:55:00"), freq="5min")
            if ts.minute == 0 and ts.hour % args.id_every_hours == 0
        ]
        t0 = time.perf_counter()
        res = run_intraday_5m(p5, p5min, pdf, fc, issues, symmetric=args.id_symmetric)
        res.assign(model=args.id_model).to_parquet(RESULTS / f"id5_{args.id_model}.parquet", index=False)
        print(f"ID-5 {args.id_model}: {len(issues)} issues, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
