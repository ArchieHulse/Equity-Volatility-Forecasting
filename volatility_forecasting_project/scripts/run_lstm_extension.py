from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volatility_project.data import build_market_panel  # noqa: E402
from volatility_project.evaluation import (  # noqa: E402
    attach_realised_targets,
    forecast_metrics,
    forecast_metrics_by_vix_regime,
    pairwise_dm,
    vix_ablation_dm,
    volatility_target_backtest,
)
from volatility_project.features import make_features  # noqa: E402
from volatility_project.models import lstm_forecasts  # noqa: E402


def _data_file(raw_dir: Path, stem: str) -> Path:
    for suffix in (".zip", ".csv"):
        candidate = raw_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {stem}.zip or {stem}.csv in {raw_dir}")


def _run_or_load(panel: pd.DataFrame, x: pd.DataFrame, targets: dict,
                 path: Path, reuse_existing: bool, epochs: int) -> pd.DataFrame:
    if reuse_existing and path.exists():
        return pd.read_csv(path, parse_dates=["date"], low_memory=False)
    forecasts = []
    for horizon in (1, 5, 21):
        print(f"Training LSTM for {horizon}-session horizon on {path.name}...", flush=True)
        forecasts.append(lstm_forecasts(x, targets, horizons=(horizon,), epochs=epochs))
    result = pd.concat(forecasts, ignore_index=True)
    result.to_csv(path, index=False)
    return result


def _merge_lstm(existing_path: Path, lstm: pd.DataFrame,
                targets: dict) -> pd.DataFrame:
    new = attach_realised_targets(lstm, targets)
    if existing_path.exists():
        old = pd.read_csv(existing_path, parse_dates=["date"], low_memory=False)
        old = old[old["model"] != "LSTM"]
        combined = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        combined = new
    combined = combined.sort_values(["horizon", "date", "model"]).reset_index(drop=True)
    combined.to_csv(existing_path, index=False)
    return combined


def _write_strategy_outputs(panel: pd.DataFrame, forecasts: pd.DataFrame,
                            outputs: Path, figures: Path) -> None:
    series_parts, metric_parts = [], []
    benchmark_added = False
    for target_vol in (0.08, 0.10, 0.12):
        for cost_bps in (0.0, 5.0, 10.0):
            series, summary = volatility_target_backtest(
                panel, forecasts, horizon=21, target_vol=target_vol,
                max_leverage=1.5, cost_bps=cost_bps,
            )
            if benchmark_added:
                series = series[series["strategy"] != "BuyAndHold"]
                summary = summary[summary["strategy"] != "BuyAndHold"]
            else:
                benchmark_added = True
            series_parts.append(series)
            summary["target_volatility"] = target_vol
            summary["cost_bps"] = cost_bps
            metric_parts.append(summary)
    strategy_series = pd.concat(series_parts, ignore_index=True)
    strategy_metrics = pd.concat(metric_parts, ignore_index=True)
    strategy_series.to_csv(outputs / "volatility_target_returns_sensitivity.csv", index=False)
    strategy_metrics.to_csv(outputs / "volatility_target_metrics_sensitivity.csv", index=False)

    selected = {
        "BuyAndHold": "Buy and hold",
        "GARCH_1_1_target10_cost5bps": "GARCH target 10%",
        "EGARCH_1_1_target10_cost5bps": "EGARCH target 10%",
        "HAR_RV_with_VIX_target10_cost5bps": "HAR + VIX target 10%",
        "XGBoost_with_VIX_target10_cost5bps": "XGBoost + VIX target 10%",
        "LSTM_target10_cost5bps": "LSTM target 10%",
    }
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, label in selected.items():
        group = strategy_series[strategy_series["strategy"] == name]
        if not group.empty:
            s = group.sort_values("date").set_index("date")
            (1 + s["return"]).cumprod().plot(ax=ax, label=label, linewidth=1.1)
    ax.set_title("Out-of-sample $1 growth — 10% volatility target, 5 bp turnover cost")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "volatility_target_equity_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and integrate the optional LSTM extension")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Use outputs/lstm_forecasts.csv and outputs/fiz_lstm_forecasts.csv if present")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    raw = ROOT / "data" / "raw"
    outputs = ROOT / "outputs"
    figures = outputs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ciz_file = _data_file(raw, "fama_french_5_factors_daily_ciz")
    fiz_file = _data_file(raw, "fama_french_5_factors_daily_fiz_2024")
    vix_file = raw / "fred_vixcls.csv"
    if not vix_file.exists():
        raise FileNotFoundError(vix_file)

    ciz_panel = build_market_panel(ciz_file, vix_file)
    ciz_x, ciz_targets = make_features(ciz_panel)
    ciz_lstm = _run_or_load(ciz_panel, ciz_x, ciz_targets,
                            outputs / "lstm_forecasts.csv", args.reuse_existing, args.epochs)
    ciz_joined = _merge_lstm(outputs / "out_of_sample_forecasts.csv", ciz_lstm, ciz_targets)

    fiz_panel = build_market_panel(fiz_file, vix_file)
    fiz_x, fiz_targets = make_features(fiz_panel)
    fiz_lstm = _run_or_load(fiz_panel, fiz_x, fiz_targets,
                            outputs / "fiz_lstm_forecasts.csv", args.reuse_existing, args.epochs)
    fiz_joined = _merge_lstm(outputs / "fiz_archive_out_of_sample_forecasts.csv", fiz_lstm, fiz_targets)

    # Recompute all current-vintage inference tables after adding the LSTM.
    forecast_metrics(ciz_joined).to_csv(outputs / "forecast_metrics.csv", index=False)
    pairwise_dm(ciz_joined).to_csv(outputs / "dm_tests_vs_garch.csv", index=False)
    vix_ablation_dm(ciz_joined).to_csv(outputs / "vix_ablation_dm_tests.csv", index=False)
    forecast_metrics_by_vix_regime(ciz_joined, ciz_panel).to_csv(
        outputs / "forecast_metrics_by_vix_regime.csv", index=False
    )

    common_end = min(ciz_panel.index.max(), fiz_panel.index.max())
    def _common_forward_window(frame: pd.DataFrame) -> pd.DataFrame:
        # Enforce identical forecast-origin and full forward-target coverage
        # when comparing the current CIZ data vintage with the Dec-2024 FIZ archive.
        pieces = []
        for horizon, group in frame.groupby("horizon", sort=False):
            end_col = f"target_end_{int(horizon)}"
            target_end = pd.to_datetime(group[end_col], errors="coerce")
            keep = (group["date"] >= pd.Timestamp("2017-01-01")) & (
                group["date"] <= common_end
            ) & (target_end <= common_end)
            pieces.append(group.loc[keep])
        return pd.concat(pieces, ignore_index=True) if pieces else frame.iloc[0:0].copy()

    ciz_common = _common_forward_window(ciz_joined)
    fiz_common = _common_forward_window(fiz_joined)
    vintage_rows = []
    for vintage, frame in (("current_CIZ", ciz_common), ("Dec_2024_FIZ_archive", fiz_common)):
        m = forecast_metrics(frame)
        m["data_vintage"] = vintage
        vintage_rows.append(m)
    pd.concat(vintage_rows, ignore_index=True).to_csv(
        outputs / "forecast_metrics_by_data_vintage.csv", index=False
    )

    rf_rows = []
    for vintage, frame in (("current_CIZ", ciz_joined), ("Dec_2024_FIZ_archive", fiz_joined)):
        for period, lo, hi in (("before_2024_06", "2017-01-01", "2024-05-31"),
                               ("from_2024_06", "2024-06-01", None)):
            sub = frame[frame["date"] >= lo]
            if hi is not None:
                sub = sub[sub["date"] <= hi]
            if not sub.empty:
                m = forecast_metrics(sub)
                m["data_vintage"] = vintage
                m["period"] = period
                rf_rows.append(m)
    pd.concat(rf_rows, ignore_index=True).to_csv(
        outputs / "forecast_metrics_by_rf_period.csv", index=False
    )

    _write_strategy_outputs(ciz_panel, ciz_joined, outputs, figures)

    # Update comparison plots with the additional LSTM forecast curve.
    for horizon in (1, 5, 21):
        subset = ciz_joined[ciz_joined["horizon"] == horizon]
        pivot = subset.pivot(index="date", columns="model", values="forecast_variance")
        realised = subset.drop_duplicates("date").set_index("date")["realised_variance"]
        fig, ax = plt.subplots(figsize=(13, 5))
        (np.sqrt(realised) * 100).plot(ax=ax, color="black", linewidth=0.8, label="Realised")
        for column in pivot.columns:
            (np.sqrt(pivot[column]) * 100).rolling(5, min_periods=1).mean().plot(
                ax=ax, alpha=0.8, linewidth=0.8, label=column
            )
        ax.set_title(f"Out-of-sample annualised volatility forecasts — {horizon}-session horizon")
        ax.set_ylabel("Annualised volatility (%)")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"forecast_comparison_{horizon}d.png", dpi=160)
        plt.close(fig)

    metadata = {
        "lstm_extension": {
            "epochs_max": args.epochs,
            "sequence_length": 22,
            "hidden_size": 32,
            "dropout": 0.15,
            "optimizer": "Adam",
            "early_stopping_patience": 7,
            "data_vintages": ["current_CIZ", "Dec_2024_FIZ_archive"],
            "current_forecast_counts": ciz_lstm.groupby("horizon").size().to_dict(),
            "fiz_forecast_counts": fiz_lstm.groupby("horizon").size().to_dict(),
        }
    }
    (outputs / "lstm_training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("LSTM extension integrated. Updated metrics, statistical tests, vintage comparison and risk outputs.")
    print(f"Current CIZ forecasts: {len(ciz_joined):,}; FIZ archive forecasts: {len(fiz_joined):,}")
    print(f"Outputs: {outputs}")
    print(forecast_metrics(ciz_joined).to_string(index=False))


if __name__ == "__main__":
    main()
