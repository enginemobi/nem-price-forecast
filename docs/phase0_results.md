# Phase 0 results: zero-shot Timestral on NEM prices

Backtest: **1–31 Oct 2026**, all 5 regions, no fine-tuning. Reproduce with
`uv run python -m nemcast.phase0 --mmsdm <MMSDM DISPATCHPRICE parquet>` (28M, all DA modes, ID-5) and
`uv run python -m nemcast.phase0 --mmsdm <…> --skip-id --da-models 209M --da-modes uni uni_pdc --every-days 2`
(see `notebooks/phase0_spike.ipynb`). Prices are $/MWh. MAE is dominated by a handful of spikes, so median
absolute error (MedAE) is shown too. Pinball = mean pinball loss over Timestral's 21 quantiles (≈ CRPS).

## Verdict: go

- **Day-ahead 30-min:** Timestral-28M with AEMO pre-dispatch (PD) price (clipped to ±$1,000) and PD reserve margin
  as known covariates has **MAE 34.9** vs **330.8 for AEMO PD**, **55.2 for same-time-yesterday** and 62.3 for
  same-time-last-week. MedAE 18.7 vs 30.1 (PD) and 32.1 (yesterday). The P10–P90 band held 86% of outcomes.
  Even target-only Timestral (MAE 41.2) beats all baselines. Covariates mainly improve typical error, not the tails.
- **Model size:** target-only, 209M ≈ 28M (MAE 42.6 vs 43.5 on the same days). With clipped PD, 209M is the best
  config tested (pinball 15.1, MAE 36.8 on those days) but costs ~5× the CPU, ~80 s per day-ahead run with covariates.
- **Intraday 5-min:** Timestral-28M (target only) beats AEMO P5MIN, PD and persistence on MAE at every lead
  time (0–1 h: 15.8 vs 25.9 P5MIN, 20.8 persistence). On MedAE PD is slightly better beyond 3 h.
- **CPU:** 28M fits easily on a 2-vCPU box. A day-ahead forecast takes about 6 s for all 5 regions; a 5-min forecast
  about 3 s with symmetric averaging off. 123M/209M take about 25–30 s per call.

## DA-30 (issued 12:45 on day D, scored on trading day D+1, 31 issues × 48 intervals × 5 regions)

| Timestral-28M mode | Pinball | P10–P90 cover | MAE | MedAE | RMSE |
|---|---:|---:|---:|---:|---:|
| `uni` (each region alone) | 15.0 | 0.78 | 41.2 | 25.4 | 179.0 |
| `multi` (5 regions jointly) | 14.7 | 0.80 | 40.6 | 26.0 | 178.6 |
| `uni_pd` (+ raw PD price) | 61.5 | 0.87 | 64.6 | 19.3 | 467.2 |
| `uni_pd_rm` (+ raw PD price, PD reserve) | 41.5 | 0.88 | 53.4 | 18.6 | 321.6 |
| `uni_pdc` (+ clipped PD price) | 15.3 | 0.86 | 35.8 | 18.3 | 179.5 |
| **`uni_pdc_rm` (+ clipped PD price, PD reserve)** | **14.5** | 0.86 | **34.9** | 18.7 | **176.5** |
| AEMO PD | – | – | 330.8 | 30.1 | 2224.6 |
| Same time yesterday | – | – | 55.2 | 32.1 | 249.3 |
| Same time last week | – | – | 62.3 | – | 251.5 |

By region (MAE; `uni` → `uni_pdc_rm`, vs AEMO PD):

| Region | Timestral `uni` | Timestral `uni_pdc_rm` | AEMO PD | Yesterday |
|---|---:|---:|---:|---:|
| NSW1 | 53.0 | 52.3 | 885.8 | 78.5 |
| QLD1 | 32.9 | 28.9 | 458.2 | 39.8 |
| SA1 | 42.7 | 36.0 | 145.4 | 59.9 |
| TAS1 | 34.8 | 27.4 | 61.2 | 42.8 |
| VIC1 | 42.3 | 29.6 | 103.3 | 55.2 |

### Model size (every 2nd day, 16 issues, same days for all rows)

| Model / mode | Pinball | MAE | MedAE | RMSE |
|---|---:|---:|---:|---:|
| 28M `uni` | 16.1 | 43.5 | 24.9 | 242.2 |
| 28M `multi` | 15.8 | 42.8 | 25.4 | 241.8 |
| 28M `uni_pdc` | 18.9 | 42.4 | 19.7 | 245.4 |
| 28M `uni_pdc_rm` | 17.6 | 41.2 | 20.3 | 241.5 |
| 209M `uni` | 16.0 | 42.6 | 24.3 | 242.4 |
| **209M `uni_pdc`** | **15.1** | **36.8** | **19.7** | **239.4** |
| AEMO PD | – | 381.8 | 31.0 | – |
| Same time yesterday | – | 61.9 | – | – |

Target-only, 209M ≈ 28M. With the clipped PD covariate, 209M uses it better: it improves on every metric, while
28M improves MAE/MedAE but its pinball gets worse. 209M + clipped PD + reserve was not run (≈3 h of CPU here).

## ID-5 (Timestral-28M, issued every 3 h, 12 h ahead at 5-min, 248 issues)

| Lead | Pinball | Cover | Timestral P50 MAE | P5MIN | PD | Persistence | Yesterday |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–1 h | 5.6 | 0.80 | **15.8** | 25.9 | 34.5 | 20.8 | 48.4 |
| 1–3 h | 10.3 | 0.78 | **25.9** | – | 45.6 | 40.6 | 53.7 |
| 3–6 h | 12.2 | 0.78 | **32.1** | – | 54.8 | 56.2 | 51.9 |
| 6–12 h | 14.2 | 0.79 | **38.6** | – | 67.9 | 73.9 | 51.8 |

MedAE at 0–1 h: Timestral 5.9, P5MIN 6.7, persistence 6.6, PD 8.5. At 6–12 h: Timestral 21.6, PD 17.5.

## What we learned

1. **AEMO PD is a poor point forecast a day ahead.** It often shows cap prices ($23,200) for NSW/QLD evening peaks
   that don't happen (e.g. issued 9 Oct for the evening of 10 Oct: PD $23,200 vs about $150 actual). Its typical error
   (MedAE) is fine, but the misses make the MAE huge.
2. **Feed PD into Timestral clipped, not raw.** Passing raw PD prices as a covariate pulls the forecast towards these
   false spikes (pinball rises 4×). Clipping to ±$1,000 keeps the useful information: MAE falls 13% vs target only,
   with the largest gains in VIC/TAS/SA, and MedAE falls from 25.4 to 18.3. Pinball does not improve consistently:
   on the every-other-day subset it is worse with clipped PD (18.9 vs 16.1), because clipped PD still drags the
   upper quantiles towards false spikes. A better PD transform (or using PD quantile spread as a feature) is next.
3. **Reserve margin helps a little** on top of clipped PD (31 days: pinball 15.3 → 14.5, RMSE 179.5 → 176.5;
   every-other-day subset: 18.9 → 17.6). It should matter more in winter/summer peaks and with PASA reserve beyond
   the PD horizon.
4. **Joint regions (`multi`) ≈ separate regions** zero-shot; not worth the extra complexity yet.
5. **Covariates cost CPU:** each known covariate adds roughly 5–6× the target-only time at full 32k context,
   because the covariate rows are processed at full length. Trimming covariate history is the next speed-up.

## Caveats

- One month (October, shoulder season) with few real spikes; Timestral's P50 will under-forecast spikes by design.
  The tails (P90/P99) and a summer/winter backtest matter before trusting it for risk.
- PD covariate history is only available from 27 Sep 2026 in this spike, so most of the 32k context has no covariate.
  A NEMSEER backfill of PD vintages should make the covariate more useful.
- P5MIN only covers the first hour, so it is scored at 0–1 h only. P5MIN runs published just after the issue time are
  used, which slightly favours AEMO.
- Benchmarks are from a shared 2-vCPU VM (timings are approximate; some runs overlapped).
