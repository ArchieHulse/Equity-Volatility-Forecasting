# Research Design

## 1. Objective

Evaluate whether machine-learning models improve forecasts of future US equity-market volatility relative to established econometric models, and test whether forecast-based volatility targeting improves the realised risk profile of a passive market exposure.

## 2. Hypotheses

- **H1:** Machine-learning models improve out-of-sample volatility forecasts relative to historical-volatility and EWMA benchmarks.
- **H2:** GARCH-family models capture volatility persistence and asymmetric responses to negative returns better than simple rolling baselines.
- **H3:** VIX adds incremental forecasting information beyond realised-volatility history alone.
- **H4:** Forecast-based volatility targeting can reduce realised volatility and drawdown relative to an unscaled market exposure, without assuming that it necessarily improves CAGR or Sharpe ratio.

These are testable hypotheses, not expected findings. Results are reported whether or not they support the hypotheses.

## 3. Data and variables

The market series is the daily US market research return in the Kenneth R. French Data Library's daily Fama/French five-factor file. Total market return is reconstructed as `Mkt-RF + RF`; the risk-free rate is retained for portfolio evaluation. VIXCLS is the CBOE VIX daily close distributed through FRED. VIX is an implied-volatility measure and is not treated as realised volatility.

The analysis uses two data vintages: the current CRSP CIZ-based file and the publisher's December 2024 CRSP FIZ archive. The current file contains the September 2026 CRSP database and data through 30 September 2026. The archived FIZ file ends on 31 December 2024. The VIX source file extends to 23 November 2026, but the joined return-volatility analysis ends when the current market-return file ends.

Fama/French returns are percent units and are converted to decimals. The VIX import detects comma or semicolon delimiters. The market panel is left-joined to VIX so missing VIX observations do not remove market-return sessions or shift forward target dates.

## 4. Target construction

For a forecast origin at close *t* and horizon *h* in {1, 5, 21}, define:

`RV(t,h) = (252 / h) * sum(r[t+j]^2 for j=1,...,h)`

The forecast target is annualised forward realised variance; annualised forward realised volatility is its square root. This is a daily-return proxy, not intraday realised variance from high-frequency data.

## 5. Features

Feature families include lagged market returns, absolute and squared returns, trailing realised variance over 5/22/63 sessions, volatility estimates, VIX level and changes, and lagged Fama/French factor returns. Every feature is available at the forecast origin. VIX observed at close *t* is used to forecast returns from *t+1* onward.

## 6. Models

- Historical realised-volatility benchmark: trailing 22-session variance.
- EWMA with lambda 0.94.
- GARCH(1,1), EGARCH(1,1), and GJR-GARCH(1,1), estimated using Gaussian quasi-maximum likelihood with constrained numerical optimisation. Parameters are estimated on pre-test observations and conditional variance is filtered through the test period using observed returns. Multi-step EGARCH forecasts use Monte Carlo future innovations.
- HAR-style regression using log-variance features, standardisation and Ridge regularisation; the regularisation parameter is selected on validation data.
- XGBoost regression on lagged market, factor and optional VIX features. The main specification is fixed rather than exhaustively tuned against the test period.
- Optional LSTM sequence model using a 22-session window, 32 hidden units, 0.15 dropout, Adam optimisation, up to 25 epochs and early stopping on validation loss (patience 7).

## 7. Chronological evaluation

- Training: 1990-01-01 to 2009-12-31.
- Validation: 2010-01-01 to 2016-12-31.
- Test: 2017-01-01 to the latest date available in the current market-return vintage.

Origins are assigned chronologically and observations are purged if the forward target window crosses a training or validation boundary. No random train/test split is used. The final test period is not used for feature selection or hyperparameter tuning.

## 8. Metrics and inference

Primary forecast loss is QLIKE on annualised variance (lower is better). Secondary metrics are RMSE and MAE on annualised volatility. Pairwise comparisons use a Diebold–Mariano-style statistic with a Newey–West/HAC long-run variance estimator using 21 lags. Holm correction is applied across model-by-horizon comparisons. These are HAC-adjusted DM-style tests with a normal approximation, not exact small-sample inference.

A VIX ablation directly compares matched VIX-inclusive and VIX-exclusive forecasts. Regime tests use a high-VIX threshold equal to the 75th percentile of VIX in the training sample only.

## 9. Portfolio illustration

Each model with a valid 21-session forecast is evaluated in a separate volatility-targeting simulation. The close-*t* forecast determines next-session exposure to market excess return, subject to an 8%, 10%, or 12% target and a 1.5x leverage cap. The cash allocation earns the observed daily risk-free rate. Cost is charged at 0, 5 or 10 basis points per unit of absolute exposure change. The portfolio results are evaluated using CAGR, annualised volatility, Sharpe ratio, maximum drawdown and turnover. This is an illustrative risk-management experiment, not an investable strategy recommendation.

## 10. Robustness checks

1. VIX-free versus VIX-inclusive model variants.
2. Direct current-CIZ versus December-2024-FIZ archive comparison over common dates.
3. Descriptive forecast-period sensitivity before and after June 2024, when the current file's `RF` source changed. This is not a causal estimate of the source effect.
4. Transaction-cost sensitivity at 0, 5 and 10 basis points.
5. Target-volatility sensitivity at 8%, 10% and 12%.
6. Separate high- and normal-VIX regimes using a threshold fixed from training data.
7. LSTM comparison on both data vintages, with the same general training design.

## 11. Limitations

Daily close returns do not measure intraday realised variance. VIX is based on S&P 500 options and is a roughly 30-day implied-volatility measure, whereas the target uses a broad US market return series and 1-, 5- and 21-session horizons. The 5- and 21-session forward targets overlap. The current Fama/French series is CIZ-based, while the historical archive is a December-2024 FIZ vintage. The current file also reports a change in the one-month Treasury-bill proxy used for `RF` in June 2024, complicating vintage attribution. Gaussian QMLE and simplified portfolio assumptions may affect results. Model rankings are historical and do not ensure future performance.
