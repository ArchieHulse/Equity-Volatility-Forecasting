import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from volatility_project.features import make_features, split_samples


def test_forward_realised_variance_alignment():
    idx = pd.bdate_range("2000-01-03", periods=100)
    returns = np.arange(1, 101, dtype=float) / 10000
    panel = pd.DataFrame({"market_return": returns, "market_excess": returns,
                          "rf": 0.0, "vix": 20.0, "vix_decimal": 0.2,
                          "vix_implied_variance": 0.04, "smb": 0.0,
                          "hml": 0.0, "rmw": 0.0, "cma": 0.0}, index=idx)
    x, targets = make_features(panel, horizons=(1, 5, 21))
    # At origin 0, the 1-session target is the squared return at index 1, annualised by 252.
    expected = 252.0 * returns[1] ** 2
    assert np.isclose(targets[1].iloc[0]["rv_var_1"], expected)
    assert targets[5].iloc[0]["target_end_5"] == idx[5]
    assert np.isnan(targets[21].iloc[-1]["rv_var_21"])


def test_split_purges_forward_labels():
    idx = pd.bdate_range("2000-01-03", periods=240)
    returns = np.random.default_rng(1).normal(0, 0.01, len(idx))
    panel = pd.DataFrame({"market_return": returns, "market_excess": returns,
                          "rf": 0.0, "vix": 20.0, "vix_decimal": 0.2,
                          "vix_implied_variance": 0.04, "smb": 0.0,
                          "hml": 0.0, "rmw": 0.0, "cma": 0.0}, index=idx)
    x, targets = make_features(panel, horizons=(5,))
    s = split_samples(x, targets[5], 5, train_end=str(idx[149].date()),
                      validation_end=str(idx[189].date()), test_start=str(idx[190].date()))
    assert len(s["X_train"]) == len(s["y_train"])
    assert s["X_train"].index.max() < idx[149]


def test_volatility_target_backtest_uses_previous_session_forecast():
    from volatility_project.evaluation import volatility_target_backtest

    idx = pd.bdate_range("2020-01-02", periods=5)
    panel = pd.DataFrame({
        "market_return": [0.01, -0.01, 0.02, 0.00, 0.01],
        "market_excess": [0.009, -0.011, 0.019, -0.001, 0.009],
        "rf": [0.001] * 5,
    }, index=idx)
    # Deliberately omit a forecast on idx[1]. The idx[2] return must not
    # incorrectly use the idx[0] signal by shifting across available rows.
    forecasts = pd.DataFrame({
        "date": [idx[0], idx[2], idx[3]],
        "horizon": [1, 1, 1],
        "model": ["test_model"] * 3,
        "forecast_variance": [0.04, 0.01, 0.01],
    })
    series, _ = volatility_target_backtest(panel, forecasts, horizon=1,
                                           target_vol=0.10, max_leverage=1.5,
                                           cost_bps=0.0)
    strat = series[series["strategy"].str.startswith("test_model")].set_index("date")
    # idx[1] receives the idx[0] signal: 10% / 20% = 0.5 exposure.
    assert np.isclose(strat.loc[idx[1], "weight"], 0.5)
    # idx[2] has no previous-session signal and is excluded rather than
    # reusing idx[0]'s stale forecast.
    assert idx[2] not in strat.index
    # idx[3] receives the idx[2] forecast: 10% / 10% = 1.0 exposure.
    assert np.isclose(strat.loc[idx[3], "weight"], 1.0)


def test_dm_hac_uses_nonzero_mean_loss_difference():
    from volatility_project.evaluation import hac_dm_test
    a = np.full(200, 2.0)
    b = np.full(200, 1.0)
    result = hac_dm_test(a, b, max_lag=21)
    assert np.isclose(result["mean_loss_diff_a_minus_b"], 1.0)
    assert result["dm_stat"] > 0


def test_vix_ablation_dm_reports_with_minus_without_sign():
    from volatility_project.evaluation import vix_ablation_dm
    dates = pd.bdate_range("2020-01-01", periods=150)
    rows = []
    for h in (1, 5, 21):
        for i, dt in enumerate(dates):
            actual = 0.02 + (i % 7) * 0.001
            # Give with-VIX lower forecast loss in a deterministic example.
            rows.append({"date": dt, "horizon": h, "model": "HAR_RV_with_VIX",
                         "forecast_variance": actual})
            rows.append({"date": dt, "horizon": h, "model": "HAR_RV_no_VIX",
                         "forecast_variance": actual * 1.5})
            rows.append({"date": dt, "horizon": h, "model": "XGBoost_with_VIX",
                         "forecast_variance": actual})
            rows.append({"date": dt, "horizon": h, "model": "XGBoost_no_VIX",
                         "forecast_variance": actual * 1.5})
    frame = pd.DataFrame(rows)
    # Include a common positive target for every date/horizon.
    frame["realised_variance"] = 0.02 + (frame["date"].map(lambda d: dates.get_loc(d) % 7)) * 0.001
    result = vix_ablation_dm(frame)
    assert len(result) == 6
    assert (result["mean_loss_diff_a_minus_b"] < 0).all()
