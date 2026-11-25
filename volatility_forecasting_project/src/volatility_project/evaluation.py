from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def attach_realised_targets(forecasts: pd.DataFrame, targets: dict) -> pd.DataFrame:
    pieces = []
    for h, target in targets.items():
        tmp = forecasts.loc[forecasts["horizon"] == h].copy()
        actual = target[[f"rv_var_{h}", f"rv_vol_{h}", f"target_end_{h}"]].copy()
        tmp = tmp.set_index("date").join(actual, how="left").reset_index()
        tmp = tmp.rename(columns={f"rv_var_{h}": "realised_variance", f"rv_vol_{h}": "realised_volatility"})
        tmp = tmp.dropna(subset=["forecast_variance", "realised_variance"])
        tmp = tmp[(tmp["forecast_variance"] > 0) & (tmp["realised_variance"] >= 0)]
        pieces.append(tmp)
    return pd.concat(pieces, ignore_index=True)


def forecast_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, model), g in joined.groupby(["horizon", "model"]):
        fvar = np.maximum(g["forecast_variance"].to_numpy(float), 1e-12)
        rvar = np.maximum(g["realised_variance"].to_numpy(float), 0.0)
        fvol, rvol = np.sqrt(fvar), np.sqrt(rvar)
        rows.append({"horizon": int(h), "model": model, "n": len(g),
                     "rmse_vol": float(np.sqrt(np.mean((fvol-rvol)**2))),
                     "mae_vol": float(np.mean(np.abs(fvol-rvol))),
                     "qlike": float(np.mean(np.log(fvar) + rvar/fvar)),
                     "mean_forecast_vol": float(np.mean(fvol)),
                     "mean_realised_vol": float(np.mean(rvol))})
    return pd.DataFrame(rows).sort_values(["horizon", "qlike"])



def forecast_metrics_by_vix_regime(joined: pd.DataFrame, panel: pd.DataFrame,
                                  train_end: str = "2009-12-31", quantile: float = 0.75) -> pd.DataFrame:
    """Score forecasts in low/high VIX regimes using a threshold fixed on training data."""
    training_vix = panel.loc[:pd.Timestamp(train_end), "vix"].dropna()
    if training_vix.empty:
        raise ValueError("No VIX observations available in the training period")
    threshold = float(training_vix.quantile(quantile))
    vix_at_origin = panel["vix"].rename("vix_at_origin")
    frame = joined.join(vix_at_origin, on="date")
    frame = frame.dropna(subset=["vix_at_origin"]).copy()
    frame["vix_regime"] = np.where(frame["vix_at_origin"] >= threshold, "high_vix", "normal_vix")
    rows = []
    for regime, group in frame.groupby("vix_regime"):
        metrics = forecast_metrics(group)
        metrics["vix_regime"] = regime
        metrics["vix_threshold_training_q75"] = threshold
        metrics["n_forecast_origins"] = int(group["date"].nunique())
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def hac_dm_test(loss_a, loss_b, max_lag: int) -> dict:
    """Two-sided DM-style test using Bartlett/Newey-West long-run variance."""
    a = np.asarray(loss_a, float)
    b = np.asarray(loss_b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    raw_diff = a[valid] - b[valid]
    n = len(raw_diff)
    if n < max(20, max_lag + 5):
        return {"n": n, "mean_loss_diff_a_minus_b": np.nan, "dm_stat": np.nan, "p_value": np.nan}
    mean_diff = float(np.mean(raw_diff))
    d = raw_diff - mean_diff
    gamma0 = np.dot(d, d) / n
    lrv = gamma0
    for lag in range(1, min(max_lag, n-1) + 1):
        gamma = np.dot(d[lag:], d[:-lag]) / n
        weight = 1.0 - lag/(max_lag+1.0)
        lrv += 2.0 * weight * gamma
    se = np.sqrt(max(lrv, 1e-18) / n)
    stat = mean_diff / se
    return {"n": n, "mean_loss_diff_a_minus_b": mean_diff,
            "dm_stat": float(stat), "p_value": float(2.0 * norm.sf(abs(stat)))}


def pairwise_dm(joined: pd.DataFrame, benchmark="GARCH_1_1") -> pd.DataFrame:
    out = []
    for h, g in joined.groupby("horizon"):
        pivot = g.assign(qlike_obs=np.log(g["forecast_variance"].clip(lower=1e-12)) +
                         g["realised_variance"]/g["forecast_variance"].clip(lower=1e-12))
        pivot = pivot.pivot_table(index="date", columns="model", values="qlike_obs")
        if benchmark not in pivot:
            continue
        for model in pivot.columns:
            if model == benchmark: continue
            common = pivot[[model, benchmark]].dropna()
            # Forecast losses are persistent beyond the mechanical overlap of
            # h-day targets; use at least 21 Newey-West lags for every horizon.
            result = hac_dm_test(common[model].values, common[benchmark].values,
                                 max_lag=max(21, int(h)-1))
            out.append({"horizon": int(h), "model": model, "benchmark": benchmark,
                        "hac_lags": max(21, int(h)-1), **result})
    result = pd.DataFrame(out)
    if not result.empty:
        from statsmodels.stats.multitest import multipletests
        valid = result["p_value"].notna()
        result["p_value_holm_global"] = np.nan
        result["reject_5pct_holm_global"] = False
        if valid.any():
            reject, adjusted, _, _ = multipletests(result.loc[valid, "p_value"].to_numpy(),
                                                    alpha=0.05, method="holm")
            result.loc[valid, "p_value_holm_global"] = adjusted
            result.loc[valid, "reject_5pct_holm_global"] = reject
    return result



def vix_ablation_dm(joined: pd.DataFrame) -> pd.DataFrame:
    """Paired HAC tests of otherwise identical model families with vs without VIX."""
    pairs = (("HAR_RV_with_VIX", "HAR_RV_no_VIX"),
             ("XGBoost_with_VIX", "XGBoost_no_VIX"))
    out = []
    for h, group in joined.groupby("horizon"):
        q = group.assign(qlike_obs=np.log(group["forecast_variance"].clip(lower=1e-12))
                         + group["realised_variance"] / group["forecast_variance"].clip(lower=1e-12))
        pivot = q.pivot_table(index="date", columns="model", values="qlike_obs")
        for with_vix, no_vix in pairs:
            if with_vix not in pivot or no_vix not in pivot:
                continue
            common = pivot[[with_vix, no_vix]].dropna()
            result = hac_dm_test(common[with_vix].values, common[no_vix].values,
                                 max_lag=max(21, int(h)-1))
            out.append({"horizon": int(h), "model_with_VIX": with_vix,
                        "model_without_VIX": no_vix,
                        "hac_lags": max(21, int(h)-1), **result})
    result = pd.DataFrame(out)
    if not result.empty:
        from statsmodels.stats.multitest import multipletests
        valid = result["p_value"].notna()
        result["p_value_holm_global"] = np.nan
        result["reject_5pct_holm_global"] = False
        if valid.any():
            reject, adjusted, _, _ = multipletests(result.loc[valid, "p_value"].to_numpy(),
                                                    alpha=0.05, method="holm")
            result.loc[valid, "p_value_holm_global"] = adjusted
            result.loc[valid, "reject_5pct_holm_global"] = reject
    return result

def portfolio_metrics(returns: pd.Series, rf: pd.Series, name: str) -> dict:
    aligned = pd.concat([returns.rename("ret"), rf.rename("rf")], axis=1).dropna()
    r, riskfree = aligned["ret"], aligned["rf"]
    wealth = (1.0 + r).cumprod()
    ann_return = float(wealth.iloc[-1] ** (252.0/len(wealth)) - 1.0) if len(wealth) else np.nan
    ann_vol = float(r.std(ddof=1) * np.sqrt(252.0)) if len(r) > 1 else np.nan
    sharpe = float(((r-riskfree).mean()*252.0)/(r.std(ddof=1)*np.sqrt(252.0))) if len(r)>1 and r.std(ddof=1)>0 else np.nan
    drawdown = wealth/wealth.cummax()-1.0
    return {"strategy": name, "n": len(r), "cagr": ann_return,
            "annualised_volatility": ann_vol, "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()), "total_return": float(wealth.iloc[-1]-1.0) if len(wealth) else np.nan}


def volatility_target_backtest(panel: pd.DataFrame, forecasts: pd.DataFrame, horizon=21,
                               target_vol=0.10, max_leverage=1.5, cost_bps=5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply forecast-only close-t signal to next session's excess return; cash earns RF."""
    g = forecasts[(forecasts["horizon"] == horizon)].copy()
    g = g.sort_values("date")
    pivot = g.pivot(index="date", columns="model", values="forecast_variance")
    # Forecasts at t are used to set exposure for return on t+1.
    rows=[]
    for model in pivot.columns:
        # Reindex to the full market-session index before shifting. This ensures a
        # close-t signal is applied to the next market session, even if a model has
        # a missing forecast on an intervening date.
        forecast_var = pivot[model].reindex(panel.index)
        weight = (target_vol / np.sqrt(forecast_var.clip(lower=1e-10))).clip(0, max_leverage).shift(1)
        data = pd.concat([panel["market_excess"], panel["rf"], weight.rename("weight")], axis=1).dropna()
        turnover = data["weight"].diff().abs().fillna(data["weight"].abs())
        cost = (cost_bps/10000.0) * turnover
        strategy_ret = data["rf"] + data["weight"]*data["market_excess"] - cost
        strategy_name = f"{model}_target{int(round(target_vol*100))}_cost{int(cost_bps)}bps"
        rows.append(pd.DataFrame({"date": data.index, "strategy": strategy_name, "return": strategy_ret.values,
                                  "weight": data["weight"].values, "turnover": turnover.values}))
    # Passive benchmark in the same period.
    bh = panel[["market_return", "rf"]].copy()
    bh = bh.loc[rows[0]["date"].min():rows[0]["date"].max()]
    rows.append(pd.DataFrame({"date": bh.index, "strategy": "BuyAndHold", "return": bh["market_return"].values,
                              "weight": 1.0, "turnover": 0.0}))
    series = pd.concat(rows, ignore_index=True)
    metrics=[]
    for strategy, s in series.groupby("strategy"):
        r=s.set_index("date")["return"]
        rf=panel["rf"].reindex(r.index)
        m=portfolio_metrics(r, rf, strategy)
        m["average_exposure"] = float(s["weight"].mean())
        m["annualised_turnover"] = float(s["turnover"].mean()*252.0)
        metrics.append(m)
    return series, pd.DataFrame(metrics)
