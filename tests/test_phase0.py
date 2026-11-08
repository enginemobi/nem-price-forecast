import numpy as np
import pandas as pd

from nemcast.phase0 import Q_COLS, QUANTILE_LEVELS, REGIONS, pinball, score, to_30m


def test_quantile_columns_match_timestral_knots():
    assert len(QUANTILE_LEVELS) == len(Q_COLS) == 21
    assert Q_COLS[0] == "q01" and Q_COLS[10] == "q50" and Q_COLS[-1] == "q99"


def test_to_30m_averages_six_intervals_ending_on_the_half_hour():
    idx = pd.date_range("2026-10-01 00:05", periods=12, freq="5min")
    p5 = pd.DataFrame({r: np.arange(12.0) for r in REGIONS}, index=idx)
    p30 = to_30m(p5)
    assert p30.loc["2026-10-01 00:30", "NSW1"] == np.mean(range(6))
    assert p30.loc["2026-10-01 01:00", "NSW1"] == np.mean(range(6, 12))


def test_pinball_is_zero_for_a_perfect_forecast_and_score_runs_without_groups():
    df = pd.DataFrame({c: [50.0, 60.0] for c in Q_COLS})
    df["actual"] = [50.0, 60.0]
    df["aemo_pd"] = [40.0, 80.0]
    assert np.allclose(pinball(df), 0)
    res = score(df, ["aemo_pd"], [])
    assert res.loc["all", "timestral_p50_mae"] == 0
    assert res.loc["all", "aemo_pd_mae"] == 15
