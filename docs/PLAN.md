# Plan: NEM Spot Price Forecasting Web App (Timestral)

**Goal:** a web app for **personal research** that publishes probabilistic spot price forecasts for all 5 NEM regions (NSW1, QLD1, VIC1, SA1, TAS1), at **30-min and 5-min** resolution. That covers both day-ahead and intraday forecasts. It uses Mistral's **Timestral** time-series foundation model and only public data: AEMO NEMWeb, open weather APIs and calendars. Data is stored in **DuckDB + Parquet**. The app is built **local-first**, with an optional free hosting path (section 8).

**Decisions so far:** personal research use · 30-min + 5-min · all 5 regions · DuckDB + Parquet · new GitHub repo · free hosting if feasible, otherwise run locally first.

---

## 1. What Timestral gives us (checked against the repo)

| Capability | Why it matters here |
|---|---|
| Zero-shot, **inference-only** package (`TimestralForecaster.from_pretrained`). The repo has no training or fine-tuning code. | No model training needed. The work goes into data, covariates and evaluation. |
| **21 quantiles** per step (0.01 … 0.99) | Fan charts, P10/P50/P90, and spike / negative-price probabilities. |
| **Multivariate** targets, forecast jointly | The 5 regional prices can be forecast together, so the model can use the coupling through interconnectors. |
| **Past-only, known-future and categorical covariates** | Known-future: AEMO forecasts (price, demand, reserve), weather forecasts, calendar. Past-only: actual flows and reserve. |
| Context up to **32,768 steps** | About 1.9 years at 30-min; about 114 days at 5-min. |
| CPU inference; 4 sizes (5M / 28M / 123M / 209M) | Fits a laptop or a free VM. The 5-min cadence may need a smaller model (benchmark this). |

Gotchas found in the code:
- `clip_nonnegative=True` (default) clamps forecasts at 0 when every observed context value is ≥ 0. Always pass `False`, because NEM prices go negative.
- `symmetric_averaging=True` (default) runs a mirrored second pass, which doubles inference cost. Benchmark with it on and off, especially for the 5-min product.
- Weights licence: Hugging Face "Modified MIT" with a US$20M monthly-revenue cap. **Personal research use is fine.** Keep the attribution notice.

## 2. Forecast products

All timestamps are stored in UTC and shown in **NEM time (AEST, UTC+10, no daylight saving)**. History used starts **Oct 2021**, when 5-minute settlement began, so the market regime is consistent.

| Product | Resolution | Issued | Horizon | Steps | Main known-future covariates |
|---|---|---|---|---|---|
| **Day-ahead (DA-30)** | 30 min | Daily ~12:45 NEM time (after the midday pre-dispatch run) | Next trading day 04:00→04:00 | ~78 | Pre-dispatch (PD) price/demand/reserve, PDPASA, weather, calendar |
| **Intraday (ID-30)** | 30 min | Every 30 min | 1–48 steps (24h) | 48 | PD, PDPASA, weather, calendar |
| **Intraday (ID-5)** | 5 min | Every 5 min | 1–144 steps (12h) | 144 | P5MIN (first hour), PD forward-filled to 5-min after that, weather interpolated, calendar |
| **Day-ahead 5-min (DA-5, experimental)** | 5 min | Daily ~12:45 | Next trading day | ~470 | As DA-30, upsampled |

- **30-min target:** mean of the six 5-min dispatch RRPs in the trading interval. Since 5-minute settlement, this is the trading price.
- **5-min target:** dispatch RRP. Context is capped at about 114 days by the model's 32,768-step limit.
- **DA-5 experiment:** compare a native 5-min run against the DA-30 forecast spread over 5-min intervals using each half hour's typical 5-min profile. Keep whichever scores better.

## 3. Public data sources

| Source | Tables / data | Use | Access (verified today) |
|---|---|---|---|
| NEMWeb `DispatchIS_Reports` | `DISPATCHPRICE` (RRP), `DISPATCHREGIONSUM` (TOTALDEMAND, AVAILABLEGENERATION, NETINTERCHANGE…), `DISPATCHINTERCONNECTORRES` | Targets + past covariates + actual reserve | 5-min ZIP/CSV ✓ |
| NEMWeb `PredispatchIS_Reports` | PD price + region summary (demand, available gen) | Known-future covariates + **baseline** | 30-min ZIP/CSV ✓ |
| NEMWeb `P5_Reports` | P5MIN region solution (next 12 × 5-min) | Known-future covariates for ID-5 | 5-min ZIP/CSV ✓ |
| NEMWeb `PDPASA` | PDPASA region solution (surplus reserve, LOR condition) | **Reserve margin** known-future | ZIP/CSV ✓ |
| NEMWeb `Short_Term_PASA_Reports` | STPASA region solution (demand, surplus reserve, up to 7 days) | Reserve / demand beyond the PD horizon | ZIP/CSV ✓ |
| NEMWeb `PD7Day` | 7-day pre-dispatch | Longer known-future covariates | ZIP/CSV ✓ |
| NEMWeb `Operational_Demand`, `ROOFTOP_PV` | Actual + forecast operational demand, rooftop PV | Covariates (drive negative midday prices) | ZIP/CSV ✓ |
| NEMWeb `MMSDM` archive | Monthly history of all MMS tables, including historical PD/P5MIN/PASA runs | Backfill + leakage-free backtests | Monthly ZIPs ✓ |
| **NEMOSIS** (UNSW CEEM, v3.8) | Downloader for dispatch/actuals tables (DISPATCHPRICE, DISPATCHREGIONSUM, interconnectors, rooftop PV…) | Historical actuals backfill | `pip install nemosis` |
| **NEMSEER** (UNSW CEEM, v1.0) | Downloader for historical AEMO forecast runs (PREDISPATCH, P5MIN, PDPASA, STPASA) | Historical forecast vintages backfill | `pip install nemseer` |
| Custom MMS CSV parser | Live files from `Reports/Current/*` | Real-time ingest (only the few tables we need) | — |
| **Open-Meteo** Forecast API | Hourly temperature, wind speed 10/100 m, GHI/DNI, cloud cover | Known-future weather | Free, no key, for non-commercial use ✓ |
| Open-Meteo Historical Forecast API | Archived *forecasts* | Leakage-free backtests | ✓ |
| Open-Meteo Archive (ERA5) | Historical actuals | Analysis / past covariates | ✓ |
| `holidays` package | AU national + state holidays | Categorical known covariates | pip |

Weather sites per region: the capital-city temperature, plus 2–3 wind and solar sites weighted by installed capacity. The site list is versioned in `config/weather_sites.yaml`.

## 4. Covariates for Timestral

**Reserve margin (added)**: built per region, both as an observed history and as a forecast:
- **Past (actual):** `reserve_mw = AVAILABLEGENERATION − TOTALDEMAND` from `DISPATCHREGIONSUM`, and `reserve_ratio = reserve_mw / TOTALDEMAND`. An optional variant adds interconnector import headroom.
- **Known-future (forecast):**
  - next hour: the same formula on **P5MIN**
  - to the end of the next trading day: on **pre-dispatch**
  - **PDPASA** `SURPLUSRESERVE`
  - beyond the PD horizon: **STPASA** surplus reserve
- **LOR condition** (none/LOR1/LOR2/LOR3), passed as a *categorical* known covariate.
- Low reserve is the main driver of price spikes, so it should help most in the upper quantiles.

Full tensor layout per issue time:
- `context` (B, V=5, T): RRP for NSW1/QLD1/VIC1/SA1/TAS1 (multivariate). Alternative: univariate per region, B=5.
- `known_covariates` (B, C_known, T+H), per region:
  - PD (or P5MIN) price and demand, as issued for the history and the latest run for the horizon
  - forecast reserve margin and PDPASA surplus reserve
  - rooftop PV and operational demand forecasts
  - weather forecasts
  - calendar: hour-of-day sin/cos, day-of-week, holiday 0/1 flag
- `categorical_known_covariates`: LOR condition, day type.
- `past_covariates`: interconnector flows, actual reserve margin, actual demand.
- Missing values → NaN (the model handles them natively).

Experiments, run in order and separately for 30-min and 5-min:
1. Target only (zero-shot)
2. + calendar
3. + AEMO PD/P5MIN prices and demand
4. + **reserve margin** (+ LOR)
5. + weather
6. Multivariate vs univariate
7. raw $/MWh vs `asinh(price/k)` transform
8. Model size 28M / 123M / 209M; `symmetric_averaging` on/off (latency vs accuracy)

## 5. Evaluation

- **Rolling-origin backtest** over ≥ 12 months: daily for DA, hourly sampling for ID-30/ID-5. Use only data available at issue time, which means archived AEMO forecast vintages and archived weather forecasts.
- **Baselines:**
  - AEMO PD price (for 30-min) and P5MIN (for 5-min, first hour), the benchmarks to beat
  - seasonal naive: same interval yesterday / last week
  - LightGBM quantile regression on the same features
- **Metrics:** MAE, RMSE, pinball loss / CRPS, P10–P90 and P5–P95 coverage, spike classification (RRP > $300: precision/recall from P(exceed)), and negative-price classification.
- Break results down by region × season × hour of day. Store them in the lake so the UI can show them.

## 6. Storage: DuckDB + Parquet

**Layout** (Hive-partitioned Parquet "lake" under `data/lake/`; raw ZIPs are optionally cached under `data/raw/` and pruned):

```
data/lake/
  dispatch_price/          region=NSW1/month=2026-11/part-*.parquet   # 5-min RRP
  dispatch_regionsum/      region=…/month=…/                           # demand, available gen, interchange
  interconnector_flow/     month=…/
  predispatch_region/      month=…/   (run_datetime, interval_datetime, region, rrp, demand, avail_gen)
  p5min_region/            month=…/   (run_datetime, interval_datetime, …)
  pdpasa_region/           month=…/   (run_datetime, interval_datetime, surplus_reserve, lor)
  stpasa_region/           month=…/
  rooftop_pv/, op_demand/  month=…/
  weather_forecast/        site=…/month=…/ (issue_time, valid_time, vars…)
  price_30m/               derived: 30-min trading price + all features aligned (feature store)
  forecast_run/            (run_id, product, issue_time, model, config_hash)
  forecast_quantile/       product=…/month=…/ (run_id, region, valid_time, q01…q99)
  metrics/                 backtest + live scores
```

- **Single writer:** one worker process does all ingest and forecast writes. Each batch writes a new small part file, and a nightly **compaction** job merges the parts in each partition. This avoids DuckDB file-locking problems.
- **Readers:** the API uses an in-memory DuckDB connection with views over `read_parquet('data/lake/<table>/**/*.parquet', hive_partitioning=true)`. Because the queries read the Parquet files directly, there is no lock on a `.duckdb` file. A `nem.duckdb` file is used only for notebooks and ad-hoc analysis.
- **Size estimate:** PD ≈ 5 regions × ~80 intervals × 17.5k runs/yr ≈ 7M rows/yr. P5MIN is about 6M rows/yr. Dispatch is about 0.5M rows/yr per table. Since Oct 2021 that totals a few GB of Parquet, which is fine on a laptop. The MMSDM backfill *downloads* are large, so download only the needed tables and columns, then delete the raw ZIPs.
- **Portability:** the lake is plain files, so it can be synced to a Hugging Face Dataset repo or object storage for the free hosting option (section 8).

## 7. Application architecture (local-first)

```
                    worker (single process, APScheduler)
  NEMWeb ──► ingest_dispatch (5m) · ingest_p5min (5m) · ingest_predispatch/pdpasa (30m)
  Open-Meteo ► ingest_weather (1h) · ingest_stpasa (daily-ish) · compact (nightly)
                    │ writes Parquet
                    ▼
            data/lake/*.parquet  ◄──── DuckDB (in-process, read)
                    │                        ▲
   forecast jobs: ID-5 (every 5m) · ID-30 (30m) · DA-30/DA-5 (12:45) · score (nightly)
                                             │
                     API (FastAPI): /regions /forecasts /actuals /metrics /runs
                                             │
                     Web UI (Vite + React + ECharts, built to static files and served by FastAPI)
```

- **Repo layout:**
  - `nemcast/ingest/` (MMS parser, NEMOSIS/NEMSEER backfill, Open-Meteo client)
  - `nemcast/features/`
  - `nemcast/model/` (Timestral wrapper, pinned `revision`)
  - `nemcast/jobs/`
  - `nemcast/api/`
  - `web/`
  - `notebooks/`
  - `tests/`
- **Tooling:** Python 3.11+, `uv`, ruff, pytest, pre-commit; Node 20 for `web/`. GitHub Actions CI runs lint, tests and a small backtest smoke test.
- **Run locally:** `uv run nemcast backfill --since 2021-10-01`, then `uv run nemcast serve`, which starts the API, the UI and the scheduler together. A Docker Compose file is also provided.
- **UI pages:**
  1. Dashboard: region and product selector; fan chart (P1–P99 shading, P10/P50/P90 lines) vs actuals and the AEMO PD/P5MIN price; spike and negative-price probability strip; reserve margin panel.
  2. Regions overview: 5 small multiples.
  3. Forecast replay: pick a past issue time.
  4. Accuracy vs baselines.
  5. Drivers: the covariates used.
  6. CSV/Parquet download.
- **Laptop resources:** about 8 GB RAM is enough for the 123M model. The ID-5 job must finish well within 5 minutes. Target < 60 s for all 5 regions in one batch, dropping to 28M or turning symmetric averaging off if needed.

## 8. Hosting (free options checked against provider docs today)

| Option | Spec | 30-min + DA | 5-min (every 5 min) | Verdict |
|---|---|---|---|---|
| **Local machine** | Your laptop/desktop | ✓ | ✓ (while it's on) | **Build and run here first** |
| **Oracle Cloud Always Free, Ampere A1 VM** | 2 OCPU Arm + 12 GB RAM, 200 GB block storage | ✓ | ✓ if the benchmark passes (likely with 28M/123M) | **Best free host:** runs the same app 24/7 unchanged. Needs card verification at signup, and A1 capacity can be unavailable in some regions. Check Oracle's idle-instance reclamation policy. |
| **GitHub Actions cron + static site** (public repo) | Free, unlimited 4 vCPU / 16 GB runners for public repos; cron ≥ 5 min | ✓: ingest + forecast every 30 min, publish Parquet to a Hugging Face Dataset repo; static React + DuckDB-WASM site on GitHub Pages | ✗: schedules "can be delayed… some queued jobs may be dropped" under load | Good **serverless fallback** for the 30-min/DA products. Requirements and risks: the repo must be public; GitHub's terms discourage using Actions "as part of a serverless application"; scheduled workflows are disabled after 60 days of repo inactivity. |
| Hugging Face Spaces | 2 vCPU / 16 GB | — | — | Compute (Docker/Gradio) Spaces **now require a paid plan**; only Static Spaces are free, and they could host the static UI. |
| Other free PaaS tiers | Typically < 1 GB RAM, sleep when idle | ✗ | ✗ | Too small for PyTorch + model weights. |

**Recommendation:**
1. Build and run locally first.
2. Deploy the same Docker Compose stack to an Oracle Always Free A1 VM, after checking the Arm build and the latency benchmark.
3. Optionally publish a read-only static dashboard (DuckDB-WASM over published Parquet) on GitHub Pages.

## 9. Delivery phases

| Phase | Scope | Exit criteria |
|---|---|---|
| 0. Spike (½–1 day) | Notebook: 2 years of 30-min and 114 days of 5-min RRP for NSW1; zero-shot Timestral DA-30 and ID-5 vs naive and AEMO PD/P5MIN over 30 days; latency per model size on the laptop | Go/no-go; model size chosen for each product |
| 1. Repo + data lake | Repo scaffold (uv, ruff, pytest, pre-commit, CI); MMS parser; NEMOSIS/NEMSEER backfill since Oct 2021; Open-Meteo historical forecasts; Parquet layout + compaction; data freshness checks | All tables populated; `nemcast backfill` reproducible |
| 2. Forecasting core | Feature builder (incl. reserve margin) → tensors; Timestral wrapper; experiments 1–8; rolling backtest harness for 30-min and 5-min | Chosen configs documented, with metrics vs AEMO baselines for all 5 regions |
| 3. Live jobs + API | Scheduler (ingest, ID-5, ID-30, DA, scoring); FastAPI endpoints | Live forecasts updating locally for 24h without errors |
| 4. Web UI | Dashboard, regions, replay, accuracy, drivers pages | All pages working against live data |
| 5. Packaging + hosting | Docker Compose (multi-arch, including arm64); Oracle A1 deployment guide; optional static GitHub Pages dashboard | Runs unattended for 7 days |
| 6. Extensions | LightGBM ensemble, conformal recalibration, spike alerts, FCAS prices | — |

Rough effort: phases 0–5 take about 4–6 Devin sessions. The biggest unknowns are the historical forecast-vintage backfill (large MMSDM downloads) and 5-min latency on CPU.

## 10. Risks & mitigations

- **Spikes and the heavy tail:** a zero-shot model will under-predict extreme prices. Mitigations: reserve margin and LOR covariates, asinh transform, an upper-quantile view, and a spike classifier if needed.
- **Backtest leakage:** use only archived forecasts as "future" covariates, never actual values.
- **5-min latency:** batch the regions, use a smaller model, turn off symmetric averaging, and keep the model resident in memory.
- **NEMWeb format changes / outages:** schema checks on ingest, retries, and freshness alerts in the UI.
- **Market price cap/floor change each July:** keep them as config.
- **No fine-tuning path:** if accuracy falls short, use ensembling or recalibration rather than fine-tuning.
- **Licences:** Timestral weights (personal use OK, keep the attribution); Open-Meteo free tier (non-commercial OK); add AEMO data attribution in the UI footer.
