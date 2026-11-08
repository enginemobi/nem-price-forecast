# nem-price-forecast

Probabilistic short-term spot price forecasts for the Australian National Electricity Market (NEM). It covers all 5 regions (NSW1, QLD1, VIC1, SA1, TAS1) at 30-min and 5-min resolution, for day-ahead and intraday horizons.

- **Model:** [Timestral](https://github.com/mistralai/timestral), Mistral AI's time-series foundation model (zero-shot, quantile forecasts, covariates)
- **Data:** AEMO [NEMWeb](https://nemweb.com.au) (dispatch, pre-dispatch, P5MIN, PASA, rooftop PV), [Open-Meteo](https://open-meteo.com) weather forecasts, AU public holidays
- **Storage:** DuckDB + Parquet
- **App:** FastAPI + a React dashboard, built to run on your own machine first

Status: planning. See [docs/PLAN.md](docs/PLAN.md).

For personal research use. Timestral weights are under Mistral AI's Modified MIT licence. NEM data © AEMO.
