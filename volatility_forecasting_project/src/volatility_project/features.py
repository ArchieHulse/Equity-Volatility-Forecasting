from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 21)


def make_features(panel: pd.DataFrame, horizons=HORIZONS) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Create features known at close t and forward RV targets from t+1 onward."""
    x = pd.DataFrame(index=panel.index)
    r = panel["market_return"].astype(float)
    r2 = r.pow(2)
    for lag in (0, 1, 2, 3, 5, 10, 21):
        x[f"market_return_lag_{lag}"] = r.shift(lag)
        x[f"abs_return_lag_{lag}"] = r.abs().shift(lag)
        x[f"squared_return_lag_{lag}"] = r2.shift(lag)
    for window in (5, 22, 63):
        x[f"rv_mean_{window}"] = r2.rolling(window, min_periods=window).mean()
        x[f"rv_sum_{window}"] = r2.rolling(window, min_periods=window).sum()
        x[f"vol_ann_{window}"] = np.sqrt(252.0 * x[f"rv_mean_{window}"])
    # Market volatility and its recent change are observable at the forecast origin.
    x["vix_decimal"] = panel["vix_decimal"]
    x["vix_squared"] = panel["vix_implied_variance"]
    x["vix_change_1"] = panel["vix"].pct_change(1, fill_method=None)
    x["vix_change_5"] = panel["vix"].pct_change(5, fill_method=None)
    x["vix_mean_5"] = panel["vix"].rolling(5).mean() / 100.0
    x["vix_mean_22"] = panel["vix"].rolling(22).mean() / 100.0
    # Factor returns are known after close t; their lags help distinguish market-wide from factor shocks.
    for col in ("smb", "hml", "rmw", "cma"):
        if col in panel:
            x[f"{col}_lag_0"] = panel[col]
            x[f"{col}_lag_1"] = panel[col].shift(1)
    x = x.replace([np.inf, -np.inf], np.nan)

    targets = {}
    for h in horizons:
        forward_sum = r2.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
        target = pd.DataFrame(index=panel.index)
        target[f"rv_var_{h}"] = (252.0 / h) * forward_sum
        target[f"rv_vol_{h}"] = np.sqrt(target[f"rv_var_{h}"].clip(lower=0))
        target[f"target_end_{h}"] = pd.Series(panel.index, index=panel.index).shift(-h)
        targets[h] = target
    return x, targets


def split_samples(x: pd.DataFrame, target: pd.DataFrame, horizon: int,
                  train_end="2009-12-31", validation_end="2016-12-31", test_start="2017-01-01"):
    """Chronological split; purge samples whose forward target crosses a boundary."""
    ycol = f"rv_var_{horizon}"
    endcol = f"target_end_{horizon}"
    data = x.join(target[[ycol, endcol]], how="inner").dropna()
    dates = data.index
    target_end = pd.to_datetime(data[endcol])
    tr_end = pd.Timestamp(train_end)
    va_end = pd.Timestamp(validation_end)
    te_start = pd.Timestamp(test_start)
    train_mask = (dates <= tr_end) & (target_end <= tr_end)
    val_mask = (dates > tr_end) & (dates <= va_end) & (target_end <= va_end)
    test_mask = (dates >= te_start)
    feature_cols = list(x.columns)
    return {
        "X_train": data.loc[train_mask, feature_cols], "y_train": data.loc[train_mask, ycol],
        "X_val": data.loc[val_mask, feature_cols], "y_val": data.loc[val_mask, ycol],
        "X_pretest": data.loc[dates < te_start, feature_cols], "y_pretest": data.loc[(dates < te_start) & (target_end < te_start), ycol],
        "X_test": data.loc[test_mask, feature_cols], "y_test": data.loc[test_mask, ycol],
        "test_target_end": target_end.loc[test_mask],
    }
