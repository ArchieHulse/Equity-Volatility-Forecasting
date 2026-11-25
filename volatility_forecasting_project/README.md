# Machine Learning and Econometric Approaches to US Equity-Market Volatility Forecasting

**Research question:** Can machine-learning models improve out-of-sample forecasts of future US equity-market volatility relative to established econometric benchmarks, and can forecast-based volatility targeting improve the realised risk profile of a passive market exposure?

This repository contains the research pipeline, executed analysis notebooks, forecast outputs and a results-led report. The main study uses the current Fama/French CRSP CIZ-based daily five-factor data; a December 2024 FIZ archive is used for a direct data-vintage robustness comparison; FRED's CBOE VIX daily close is used as an optional predictor.

## Key findings

- **1-session horizon:** HAR with VIX achieves the best QLIKE; EGARCH records the lowest RMSE of annualised volatility.
- **5-session horizon:** XGBoost with VIX has the best QLIKE and RMSE, with HAR + VIX close on QLIKE.
- **21-session horizon:** XGBoost + VIX has the best QLIKE, while EGARCH has the lowest RMSE/MAE of volatility.
- VIX significantly improves the HAR forecasts at 1- and 5-session horizons after Holm correction. VIX's incremental gain for XGBoost is not statistically significant after that correction.
- LSTM is competitive but does not lead any horizon.
- A 10% volatility-targeting illustration reduces realised volatility and drawdown relative to buy-and-hold, at the cost of lower CAGR. This is a risk-scaling result, not evidence of alpha.

See [`reports/empirical_results.md`](reports/empirical_results.md) for full tables, statistical comparisons, data-vintage checks and limitations.

## Data sources

1. [Kenneth R. French Data Library — current daily Fama/French five-factor data](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (current CIZ-based data vintage; uploaded file identifies the September 2026 CRSP database).
2. [Kenneth R. French Data Library — December 2024 historical archive](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library_202412_archive.html) (FIZ data vintage).
3. [FRED — CBOE Volatility Index (VIXCLS)](https://fred.stlouisfed.org/series/VIXCLS) (daily close, sourced from CBOE; observe the source's attribution/copyright notice).

The CIZ and FIZ files are not identical. The publisher states that CRSP FIZ was discontinued after the December 2024 release and the newer CIZ-based data began in January 2025. The current file also reports that the one-month Treasury-bill proxy used for `RF` changed from Ibbotson through May 2024 to ICE BofA US 1-Month Treasury Bill Index from June 2024. The project compares the two complete data vintages directly over common dates and separately reports a descriptive pre-/post-June-2024 sensitivity; neither comparison should be interpreted as a causal attribution of data-source effects.

## Reproducing the analysis

Python 3.11 is recommended. The main pipeline accepts either official ZIP files or extracted CSV files. Place the three files under `data/raw/` with these names:

```text
data/raw/fama_french_5_factors_daily_ciz.csv
data/raw/fama_french_5_factors_daily_fiz_2024.csv
data/raw/fred_vixcls.csv
```

Install dependencies and run the main econometric/XGBoost pipeline:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

To download the official source files automatically instead, run `python run_pipeline.py --download` (or `--download --refresh` to force a refresh). Check the upstream data terms before redistributing raw source files.

The LSTM extension is optional and requires PyTorch:

```bash
pip install -r requirements-lstm.txt
python scripts/run_lstm_extension.py
```

The main pipeline regenerates the CIZ/FIZ econometric and XGBoost forecasts, diagnostics, sensitivity outputs and figures. The optional LSTM script trains and adds the LSTM forecasts, then updates combined metrics and figures. Notebooks are designed as a guided exploration of source data and generated results; run the pipeline before running notebooks 02–04.

## Project structure

```text
.
├── data/
│   ├── raw/                   # user-downloaded source files; excluded from public package
│   └── processed/             # generated aligned market panels
├── notebooks/
│   ├── 01_data_and_eda.ipynb
│   ├── 02_econometric_models.ipynb
│   ├── 03_machine_learning.ipynb
│   └── 04_evaluation_and_risk.ipynb
├── outputs/                   # selected summary tables and figures
├── reports/
│   ├── research_design.md
│   └── empirical_results.md
├── scripts/run_lstm_extension.py
├── src/volatility_project/    # data, feature, model and evaluation functions
├── tests/
└── run_pipeline.py
```

## Model and evaluation details

- Horizons: 1, 5 and 21 trading sessions.
- Target: annualised forward sum of daily squared market returns, with the square root used for volatility-error metrics. This is a daily-close-return proxy, not intraday realised variance.
- Splits: training 1990–2009; validation 2010–2016; final out-of-sample testing from 2017 onward. Forward labels crossing a split boundary are purged.
- Models: historical volatility, EWMA, Gaussian-QMLE GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), HAR-style Ridge regression, XGBoost, and optional LSTM.
- Metrics: QLIKE on annualised variance; RMSE and MAE on annualised volatility; HAC-adjusted DM-style comparisons with Holm correction.
- VIX: predictor, not target. VIX is S&P 500 option-implied volatility, while the target is constructed from a broader US equity-market return series.

## Reproducibility notes

- `data/processed/source_metadata.json` records file names, SHA-256 checksums, data coverage and split dates for the run.
- Current model specifications and data transformations are in `src/volatility_project/` and `run_pipeline.py`.
- GARCH-family optimisation is implemented using SciPy Gaussian quasi-maximum likelihood; the `arch` package is not required.
- LSTM random initialisation may create small run-to-run variation. Its architecture, maximum epoch count and early-stopping settings are recorded in `outputs/lstm_training_metadata.json`.
- The test period is historical. Forecast performance does not imply future superiority or investment returns.

## References

- Corsi, F. (2009). “A Simple Approximate Long-Memory Model of Realized Volatility.” *Journal of Financial Econometrics*, 7(2), 174–196.
- Engle, R. (1982). “Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation.” *Econometrica*, 50(4), 987–1007.
- Bollerslev, T. (1986). “Generalized Autoregressive Conditional Heteroskedasticity.” *Journal of Econometrics*, 31(3), 307–327.
- Nelson, D. B. (1991). “Conditional Heteroskedasticity in Asset Returns: A New Approach.” *Econometrica*, 59(2), 347–370.
