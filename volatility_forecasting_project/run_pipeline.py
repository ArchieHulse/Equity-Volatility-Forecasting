from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from volatility_project.data import (  # noqa: E402
    build_market_panel,
    compare_data_vintages,
    download_sources,
    save_metadata,
    sha256_file,
)
from volatility_project.evaluation import (  # noqa: E402
    attach_realised_targets,
    forecast_metrics,
    forecast_metrics_by_vix_regime,
    pairwise_dm,
    vix_ablation_dm,
    volatility_target_backtest,
)
from volatility_project.features import make_features  # noqa: E402
from volatility_project.models import (  # noqa: E402
    arch_family_forecasts,
    har_forecasts,
    simple_benchmark_forecasts,
    xgboost_forecasts,
)


def generate_forecasts(panel: pd.DataFrame, x: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """Run the same model specification on a single data vintage."""
    parts = [
        simple_benchmark_forecasts(panel, x),
        arch_family_forecasts(panel),
        har_forecasts(x, targets, include_vix=True),
        har_forecasts(x, targets, include_vix=False),
        xgboost_forecasts(x, targets, include_vix=True),
        xgboost_forecasts(x, targets, include_vix=False),
    ]
    forecasts = pd.concat(parts, ignore_index=True)
    forecasts = forecasts[
        (forecasts["date"] >= pd.Timestamp("2017-01-01"))
        & forecasts["forecast_variance"].notna()
    ]
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the volatility forecasting research pipeline")
    parser.add_argument("--download", action="store_true", help="Download current CIZ data, FIZ archive and FRED VIX")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of raw data")
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    outputs = ROOT / "outputs"
    figures = outputs / "figures"
    for path in (raw_dir, processed, outputs, figures):
        path.mkdir(parents=True, exist_ok=True)

    metadata = download_sources(raw_dir, refresh=args.refresh) if args.download else {
        "note": "Using user-provided raw files staged in data/raw."
    }
    # Accept both the original publisher ZIPs and user-supplied extracted CSVs.
    ciz_candidates = [raw_dir / "fama_french_5_factors_daily_ciz.zip",
                      raw_dir / "fama_french_5_factors_daily_ciz.csv"]
    fiz_candidates = [raw_dir / "fama_french_5_factors_daily_fiz_2024.zip",
                      raw_dir / "fama_french_5_factors_daily_fiz_2024.csv"]
    ciz_file = next((p for p in ciz_candidates if p.exists()), ciz_candidates[0])
    fiz_file = next((p for p in fiz_candidates if p.exists()), fiz_candidates[0])
    vix_file = raw_dir / "fred_vixcls.csv"
    missing = [p.name for p in (ciz_file, fiz_file, vix_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required raw source files are missing: " + ", ".join(missing)
            + ". Run: python run_pipeline.py --download"
        )

    # Main analysis uses the current Fama/French series (CIZ-based data vintage).
    ciz_panel = build_market_panel(ciz_file, vix_file)
    ciz_x, ciz_targets = make_features(ciz_panel)
    ciz_forecasts = generate_forecasts(ciz_panel, ciz_x, ciz_targets)
    ciz_joined = attach_realised_targets(ciz_forecasts, ciz_targets)
    ciz_joined.to_csv(outputs / "out_of_sample_forecasts.csv", index=False)
    ciz_panel.to_csv(processed / "market_panel_ciz.csv", index_label="date")

    # Re-run the same methodology on the official December-2024 FIZ archive.
    # This is a true data-vintage comparison over common historical dates, unlike
    # splitting the current CIZ series into pre- and post-2025 observation dates.
    fiz_panel = build_market_panel(fiz_file, vix_file)
    fiz_x, fiz_targets = make_features(fiz_panel)
    fiz_forecasts = generate_forecasts(fiz_panel, fiz_x, fiz_targets)
    fiz_joined = attach_realised_targets(fiz_forecasts, fiz_targets)
    fiz_joined.to_csv(outputs / "fiz_archive_out_of_sample_forecasts.csv", index=False)
    fit_cols = [c for c in ciz_forecasts.columns if c.startswith("fit_") or c.startswith("parameter_")]
    if fit_cols:
        ciz_params = ciz_forecasts.loc[ciz_forecasts["model"].isin(
            ["GARCH_1_1", "EGARCH_1_1", "GJR_GARCH_1_1"]), ["model"] + fit_cols].drop_duplicates()
        ciz_params.insert(0, "data_vintage", "current_CIZ")
        fiz_params = fiz_forecasts.loc[fiz_forecasts["model"].isin(
            ["GARCH_1_1", "EGARCH_1_1", "GJR_GARCH_1_1"]), ["model"] + fit_cols].drop_duplicates()
        fiz_params.insert(0, "data_vintage", "Dec_2024_FIZ_archive")
        pd.concat([ciz_params, fiz_params], ignore_index=True).to_csv(
            outputs / "garch_family_parameter_estimates.csv", index=False
        )
    fiz_panel.to_csv(processed / "market_panel_fiz_archive.csv", index_label="date")

    metadata.update({
        "current_ciz_sample": {
            "start": str(ciz_panel.index.min().date()),
            "end": str(ciz_panel.index.max().date()),
            "n_observations": int(len(ciz_panel)),
        },
        "dec_2024_fiz_archive_sample": {
            "start": str(fiz_panel.index.min().date()),
            "end": str(fiz_panel.index.max().date()),
            "n_observations": int(len(fiz_panel)),
        },
        "split_dates": {
            "train_end": "2009-12-31",
            "validation_end": "2016-12-31",
            "test_start": "2017-01-01",
        },
    })
    metadata["user_provided_sources"] = {
        "current_CIZ": {"file": ciz_file.name, "sha256": sha256_file(ciz_file), "bytes": ciz_file.stat().st_size},
        "Dec_2024_FIZ_archive": {"file": fiz_file.name, "sha256": sha256_file(fiz_file), "bytes": fiz_file.stat().st_size},
        "FRED_VIXCLS": {"file": vix_file.name, "sha256": sha256_file(vix_file), "bytes": vix_file.stat().st_size},
    }
    save_metadata(metadata, processed / "source_metadata.json")

    # Primary current-vintage out-of-sample metrics and HAC loss comparisons.
    metrics = forecast_metrics(ciz_joined)
    metrics.to_csv(outputs / "forecast_metrics.csv", index=False)
    forecast_metrics_by_vix_regime(ciz_joined, ciz_panel).to_csv(
        outputs / "forecast_metrics_by_vix_regime.csv", index=False
    )
    pairwise_dm(ciz_joined).to_csv(outputs / "dm_tests_vs_garch.csv", index=False)
    vix_ablation_dm(ciz_joined).to_csv(outputs / "vix_ablation_dm_tests.csv", index=False)

    # Compare the two source vintages on exactly the same test dates.
    common_end = min(ciz_panel.index.max(), fiz_panel.index.max())
    def _common_forward_window(frame: pd.DataFrame) -> pd.DataFrame:
        # Keep only origins whose complete forward realised-variance target is
        # observable in BOTH vintages. This avoids giving CIZ extra late-2024
        # forecast origins whose target window extends beyond the FIZ archive.
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
    vintage_metrics = []
    for vintage, frame in (("current_CIZ", ciz_common), ("Dec_2024_FIZ_archive", fiz_common)):
        if not frame.empty:
            vm = forecast_metrics(frame)
            vm["data_vintage"] = vintage
            vintage_metrics.append(vm)
    if vintage_metrics:
        pd.concat(vintage_metrics, ignore_index=True).to_csv(
            outputs / "forecast_metrics_by_data_vintage.csv", index=False
        )

    vintage_diffs = []
    for period, start in (("full_common_history", None), ("test_period_common_dates", "2017-01-01")):
        ciz_slice = ciz_panel if start is None else ciz_panel.loc[start:common_end]
        fiz_slice = fiz_panel if start is None else fiz_panel.loc[start:common_end]
        if not ciz_slice.empty and not fiz_slice.empty:
            vd = compare_data_vintages(ciz_slice, fiz_slice)
            vd["period"] = period
            vintage_diffs.append(vd)
    if vintage_diffs:
        pd.concat(vintage_diffs, ignore_index=True).to_csv(
            outputs / "data_vintage_comparison.csv", index=False
        )

    # Descriptive break-period sensitivity around the June-2024 RF-source change.
    # These are observation-period diagnostics, not a causal estimate of the source effect.
    rf_period_metrics = []
    for vintage, frame in (("current_CIZ", ciz_common), ("Dec_2024_FIZ_archive", fiz_common)):
        for period, lo, hi in (
            ("before_2024_06", "2017-01-01", "2024-05-31"),
            ("from_2024_06", "2024-06-01", None),
        ):
            sub = frame[frame["date"] >= lo]
            if hi is not None:
                sub = sub[sub["date"] <= hi]
            if not sub.empty:
                rm = forecast_metrics(sub)
                rm["data_vintage"] = vintage
                rm["period"] = period
                rf_period_metrics.append(rm)
    if rf_period_metrics:
        pd.concat(rf_period_metrics, ignore_index=True).to_csv(
            outputs / "forecast_metrics_by_rf_period.csv", index=False
        )

    # Volatility-targeting sensitivity for every model, with no model selected
    # using final test performance. Selection is left to validation-period evidence.
    strategy_series_parts = []
    strategy_metric_parts = []
    benchmark_added = False
    for target_vol in (0.08, 0.10, 0.12):
        for cost_bps in (0.0, 5.0, 10.0):
            series, summary = volatility_target_backtest(
                ciz_panel,
                ciz_forecasts,
                horizon=21,
                target_vol=target_vol,
                max_leverage=1.5,
                cost_bps=cost_bps,
            )
            if benchmark_added:
                series = series[series["strategy"] != "BuyAndHold"]
                summary = summary[summary["strategy"] != "BuyAndHold"]
            else:
                benchmark_added = True
            strategy_series_parts.append(series)
            summary["target_volatility"] = target_vol
            summary["cost_bps"] = cost_bps
            strategy_metric_parts.append(summary)

    strategy_series = pd.concat(strategy_series_parts, ignore_index=True)
    strategy_metrics = pd.concat(strategy_metric_parts, ignore_index=True)
    strategy_series.to_csv(outputs / "volatility_target_returns_sensitivity.csv", index=False)
    strategy_metrics.to_csv(outputs / "volatility_target_metrics_sensitivity.csv", index=False)

    # Charts for the full current-vintage test sample.
    fig, ax = plt.subplots(figsize=(12, 5))
    ciz_panel["vix"].plot(ax=ax, linewidth=0.8, title="CBOE VIX daily close")
    ax.set_ylabel("VIX index")
    fig.tight_layout()
    fig.savefig(figures / "vix_history.png", dpi=160)
    plt.close(fig)

    for horizon in (1, 5, 21):
        subset = ciz_joined[ciz_joined["horizon"] == horizon]
        if subset.empty:
            continue
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

    fig, ax = plt.subplots(figsize=(12, 5))
    selected_strategies = {
        "BuyAndHold": "Buy and hold",
        "GARCH_1_1_target10_cost5bps": "GARCH target 10%",
        "EGARCH_1_1_target10_cost5bps": "EGARCH target 10%",
        "HAR_RV_with_VIX_target10_cost5bps": "HAR + VIX target 10%",
        "XGBoost_with_VIX_target10_cost5bps": "XGBoost + VIX target 10%",
        "LSTM_target10_cost5bps": "LSTM target 10%",
    }
    for strategy, label in selected_strategies.items():
        group = strategy_series[strategy_series["strategy"] == strategy]
        if group.empty:
            continue
        data = group.sort_values("date").set_index("date")
        (1 + data["return"]).cumprod().plot(ax=ax, label=label, linewidth=1.1)
    ax.set_title("Out-of-sample $1 growth — 10% volatility target, 5 bp turnover cost")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "volatility_target_equity_curves.png", dpi=160)
    plt.close(fig)

    versions = {}
    import_names = {"scikit-learn": "sklearn"}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "torch", "numba"):
        try:
            mod = __import__(import_names.get(package, package))
            versions[package] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[package] = "not available"
    metadata["package_versions"] = versions
    save_metadata(metadata, processed / "source_metadata.json")

    print(
        "Completed. Current CIZ sample: "
        f"{ciz_panel.index.min().date()} to {ciz_panel.index.max().date()} "
        f"({len(ciz_panel):,} rows)"
    )
    print("FIZ archive sample:", fiz_panel.index.min().date(), "to", fiz_panel.index.max().date())
    print("Outputs:", outputs)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
