# Empirical Results

**Project:** Machine Learning and Econometric Approaches to US Equity-Market Volatility Forecasting  
**Primary data vintage:** Current Fama/French CRSP CIZ-based daily factors  
**Out-of-sample period:** 3 January 2017–30 September 2026 (forecast-origin dates; the precise first origin varies by feature availability)

## Executive summary

This study compares daily volatility forecasts at 1-, 5- and 21-trading-session horizons. The current CIZ-based market series is the primary analysis. Models include rolling historical volatility, EWMA, GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), HAR-style realised-volatility regressions, XGBoost and an LSTM sequence model. VIX-inclusive and VIX-excluded supervised specifications are evaluated separately.

The principal finding is **horizon-dependent performance**:

- At one day, the HAR model with VIX has the lowest QLIKE loss. EGARCH has the lowest RMSE of annualised volatility, showing that the preferred model depends on the loss function.
- At five days, XGBoost with VIX has the lowest QLIKE and RMSE, although HAR with VIX is close on QLIKE.
- At 21 days, XGBoost with VIX has the lowest QLIKE, while EGARCH has the lowest RMSE and MAE of annualised volatility. XGBoost's QLIKE lead over GARCH-family alternatives is small and should not be interpreted as a decisive model victory.
- The LSTM is competitive but is not the top model at any horizon. It improves on the GARCH(1,1) QLIKE benchmark at one and five days under the study's multiplicity-adjusted HAC comparisons, but it does not displace the best-performing model.
- VIX adds statistically significant information to the HAR specification at one- and five-day horizons after Holm correction. Its incremental contribution to XGBoost does not remain statistically significant after the same correction.
- The volatility-targeting illustration lowers realised volatility and drawdown relative to buy-and-hold, but it also lowers CAGR. This is a risk-control trade-off, not evidence of alpha.

## 1. Data and sample construction

### Sources

1. **Kenneth R. French Data Library — daily Fama/French five factors, current CRSP CIZ-based data vintage.** The uploaded file identifies the September 2026 CRSP database and provides observations through 30 September 2026.
2. **Kenneth R. French Data Library — December 2024 archive.** The uploaded archive identifies the December 2024 CRSP database and supplies the legacy FIZ-based series through 31 December 2024.
3. **FRED series VIXCLS.** Daily CBOE VIX closing values in the uploaded file extend through 23 November 2026. The usable joined market-return sample ends on 30 September 2026 because the CIZ market-return file ends on that date.

The Fama/French files report returns in percent; these are converted to decimal returns. The broad US market total return is reconstructed as `Mkt-RF + RF`. VIX is an implied-volatility predictor, not the realised-volatility target. The VIX file supplied by the user uses a semicolon delimiter; the loader detects this format.

### Coverage and data integrity

| Series | Observations | Start | End |
|---|---:|---|---|
| Current CIZ market panel after VIX left join | 9,254 | 2 Jan 1990 | 30 Sep 2026 |
| December 2024 FIZ archive panel after VIX left join | 8,817 | 2 Jan 1990 | 31 Dec 2024 |
| Common CIZ/FIZ market dates | 8,817 | 2 Jan 1990 | 31 Dec 2024 |
| VIX source file | 9,323 | 2 Jan 1990 | 23 Nov 2026 |

The CIZ and FIZ panels each retain all market-return trading sessions. Three dates in the VIX source are unavailable after cleaning; these remain in the market panels, but VIX-dependent specifications cannot use those observations. Duplicate dates are checked by the ingestion code.

The current CIZ file's header reports that the `RF` source is Ibbotson through May 2024 and the ICE BofA US 1-Month Treasury Bill Index from June 2024. This is a separate consideration from the CIZ/FIZ data-vintage change.

## 2. Forecasting methodology

### Target

For forecast origin \(t\) and horizon \(h\in\{1,5,21\}\), the annualised forward realised-variance proxy is

\[
RV_{t,h}=\frac{252}{h}\sum_{j=1}^{h}r_{t+j}^{2}.
\]

Annualised realised volatility is \(\sqrt{RV_{t,h}}\). This is a daily-close-return proxy; it is not an intraday realised-variance measure constructed from high-frequency observations.

Features use information available at the close of the forecast origin. VIX is observed at close \(t\) and forecasts returns beginning at \(t+1\). Chronological splits are used:

- Training: 1990–2009.
- Validation: 2010–2016.
- Out-of-sample test: 2017 onward.

Forward labels crossing a training or validation boundary are purged. The test set is not used for feature selection or hyperparameter tuning.

### Models

- Historical volatility: trailing 22-session variance.
- EWMA: RiskMetrics decay parameter \(\lambda=0.94\).
- GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1): Gaussian quasi-maximum-likelihood estimation implemented with constrained numerical optimisation. GARCH parameters are estimated before the test period and conditional variance is filtered forward using observed returns. Multi-step EGARCH forecasts use Monte Carlo innovation paths.
- HAR-style realised-volatility regression: log-variance features, standardisation and Ridge regularisation; regularisation is selected using validation data.
- XGBoost: fixed model specification, with and without VIX features.
- LSTM: 22-session sequence window, 32 hidden units, 0.15 dropout, Adam optimiser, maximum 25 epochs and validation early stopping (patience 7). The LSTM is an optional extension and requires PyTorch.

### Evaluation

The primary loss is QLIKE on annualised variance (lower is better). Secondary measures are RMSE and MAE on annualised volatility. Forecast-loss comparisons use a Diebold–Mariano-style statistic with a Newey–West/HAC variance estimate using 21 lags; Holm correction is applied across the model/horizon comparison family. These are reported as HAC-adjusted DM-style comparisons, not as exact small-sample tests.

## 3. Main out-of-sample forecast results

The table reports the best three models by QLIKE at each horizon. Volatility errors are expressed as annualised decimal volatility; multiply by 100 for percentage points.

| Horizon | Model | N | QLIKE | RMSE volatility | MAE volatility |
|---:|---|---:|---:|---:|---:|
| 1 day | HAR + VIX | 2,448 | **−2.9196** | 0.1224 | 0.0886 |
| 1 day | LSTM | 2,448 | −2.8870 | 0.1321 | 0.0958 |
| 1 day | EGARCH | 2,448 | −2.8630 | **0.1219** | **0.0874** |
| 5 days | XGBoost + VIX | 2,444 | **−2.8414** | **0.0762** | 0.0523 |
| 5 days | HAR + VIX | 2,444 | −2.8363 | 0.0785 | **0.0507** |
| 5 days | LSTM | 2,444 | −2.8186 | 0.0841 | 0.0573 |
| 21 days | XGBoost + VIX | 2,428 | **−2.5600** | 0.0863 | 0.0550 |
| 21 days | GJR-GARCH | 2,428 | −2.5555 | 0.0861 | 0.0518 |
| 21 days | LSTM | 2,428 | −2.5426 | 0.0888 | 0.0525 |

For 21-day forecasts, EGARCH records the lowest RMSE (0.0819) and MAE (0.0493) among the models, even though XGBoost + VIX has the best QLIKE. The result reinforces that model rankings depend on the forecast loss function.

### HAC-adjusted comparisons against GARCH(1,1)

After Holm correction across model-by-horizon comparisons:

- **1 day:** HAR + VIX and LSTM have lower QLIKE than GARCH; GJR-GARCH also improves on GARCH. EWMA and trailing historical volatility are significantly worse.
- **5 days:** HAR + VIX, LSTM, XGBoost without VIX and XGBoost with VIX have lower QLIKE than GARCH. The trailing historical-volatility baseline is significantly worse.
- **21 days:** no alternative model significantly improves on GARCH after correction; the trailing historical-volatility baseline is significantly worse.

This is evidence of comparative predictive performance within this specific test design, not proof that the models will retain their rankings in future markets.

## 4. Does VIX add information?

The VIX ablation compares otherwise equivalent VIX-inclusive and VIX-excluded specifications on matched forecast dates. Mean loss differences are calculated as `QLIKE(with VIX) − QLIKE(without VIX)`; negative values favour the VIX-inclusive model.

| Horizon | Model pair | Mean loss difference | Holm-adjusted p-value | Significant at 5%? |
|---:|---|---:|---:|---|
| 1 day | HAR + VIX vs HAR | −0.1260 | <0.00001 | Yes |
| 1 day | XGBoost + VIX vs XGBoost | −0.0098 | 0.5610 | No |
| 5 days | HAR + VIX vs HAR | −0.1041 | 0.000005 | Yes |
| 5 days | XGBoost + VIX vs XGBoost | −0.0277 | 0.0868 | No |
| 21 days | HAR + VIX vs HAR | −0.0660 | 0.0868 | No |
| 21 days | XGBoost + VIX vs XGBoost | −0.0403 | 0.3854 | No |

VIX is a useful incremental predictor for HAR at one- and five-day horizons in this sample. Although point estimates generally favour adding VIX to XGBoost, the evidence is not statistically significant after Holm correction.

## 5. Market regimes

The high-VIX regime is defined as VIX above the 75th percentile of the **training-period** VIX distribution (1990–2009), a threshold of 23.9425. This threshold is fixed before evaluating the test period. The test sample contains 430 high-VIX and 2,018 normal-VIX one-day forecast origins (horizon-specific valid counts differ slightly).

Average realised annualised volatility is materially higher in the high-VIX regime:

| Horizon | High-VIX mean realised volatility | Normal-VIX mean realised volatility |
|---:|---:|---:|
| 1 day | 24.38% | 9.70% |
| 5 days | 27.86% | 12.04% |
| 21 days | 25.92% | 13.55% |

The leading QLIKE model varies by horizon and regime: HAR + VIX is strongest for one-day forecasts in both regimes; XGBoost + VIX is strongest at five days; and GJR-GARCH has the best high-VIX 21-day QLIKE, while XGBoost + VIX is strongest in normal-VIX 21-day conditions. Regime-specific rankings are descriptive and are based on smaller, unequal samples; they should not be treated as stable strategy-selection rules.

## 6. Data-vintage robustness: CIZ versus FIZ

Over the full common return history (8,817 dates from 2 January 1990 to 31 December 2024), the current CIZ and December 2024 FIZ series have:

- Market total-return correlation: **0.999982**.
- Mean absolute daily market-return difference: **0.000041** in decimal-return units (0.0041 percentage points).
- Maximum absolute daily market-return difference: **0.00080** (0.08 percentage points).

For the forecast comparison, each horizon is restricted to origins whose **entire forward target window** ends by 31 December 2024. This produces identical CIZ/FIZ forecast-origin counts of 2,011 (1 day), 2,007 (5 days) and 1,991 (21 days). On 2017–2024 test-period common market-return dates, market-return correlation is 0.999974. Forecast rankings on the archived period are broadly similar: HAR + VIX is best by QLIKE at one day; HAR + VIX and XGBoost + VIX are nearly tied at five days; XGBoost + VIX is best at 21 days. The LSTM is not the top-ranked model on the archived test period either.

The two vintages should not be treated as identical. In addition to the FIZ-to-CIZ return-data construction change, the current file reports a change in the `RF` source from June 2024. Hence the comparison is a full data-vintage comparison, not an isolated causal test of the return-construction change.

## 7. Risk-management illustration: volatility targeting

A separate long-only volatility-targeting simulation uses the 21-session forecast to scale next-session market exposure. It targets 10% annualised volatility, caps exposure at 1.5x, accrues the observed daily risk-free rate on the cash allocation and charges 5 basis points per unit of absolute turnover. The strategy is illustrative and does not include all real-world trading frictions, financing spreads or instrument-tracking effects.

| Strategy | CAGR | Annualised volatility | Sharpe | Maximum drawdown | Average exposure |
|---|---:|---:|---:|---:|---:|
| Buy and hold | 15.17% | 18.79% | 0.712 | −34.22% | 1.00x |
| GARCH target 10% | 9.86% | 9.74% | 0.755 | −14.47% | 0.69x |
| HAR + VIX target 10% | 9.46% | 9.22% | 0.753 | −14.42% | 0.68x |
| XGBoost + VIX target 10% | 8.92% | 8.63% | 0.741 | −12.85% | 0.63x |
| LSTM target 10% | 9.01% | 9.35% | 0.700 | −14.87% | 0.66x |

The risk-targeted strategies reduce volatility and drawdown, as intended, but they also have lower CAGR because average exposure is below one. Sharpe ratios are only modestly higher for selected strategies; the LSTM does not improve on buy-and-hold Sharpe in this illustration. These results should be interpreted as a risk-scaling demonstration, not evidence that volatility forecasts create alpha.

## 8. Limitations and interpretation

1. **Daily realised-variance proxy:** the target sums daily squared close-to-close returns. It does not use intraday high-frequency realised variance and is affected by close-to-close jumps.
2. **Index mismatch:** VIX is derived from S&P 500 options, whereas the Fama/French market factor represents a broad US equity-market portfolio. VIX is therefore a closely related market-wide proxy, not a perfectly matched implied-volatility measure for the target portfolio.
3. **Different forecast horizons:** VIX represents roughly 30-day option-implied volatility; the target horizons are 1, 5 and 21 trading sessions.
4. **Overlapping targets:** daily 5- and 21-session forecasts have overlapping realised outcomes. HAC inference helps account for serial dependence, but statistical tests remain approximations.
5. **Data vintage and RF source:** CIZ and FIZ differ slightly. The `RF` source change in June 2024 overlaps the archived period's end and complicates attribution of differences.
6. **Model scope:** the XGBoost parameters and LSTM architecture are a limited, pre-specified comparison, not an exhaustive hyperparameter search. Deep models can be sensitive to random seeds and training choices.
7. **GARCH assumptions:** Gaussian QMLE is used; EGARCH multi-step forecasts rely on simulated Gaussian innovations. Alternative innovation distributions and estimation methods could change results.
8. **Portfolio assumptions:** the risk-targeting experiment is a simplified long-only scaling rule with a fixed transaction-cost assumption, observed risk-free cash accrual and a leverage cap. It is not a fully costed investment strategy.
9. **No guaranteed persistence:** the out-of-sample period is historical. The reported rankings do not imply future forecast superiority or investment returns.

## Conclusion

The empirical results favour model diversity rather than a single universally superior approach. HAR with VIX is particularly effective at short horizons, while XGBoost with VIX is strongest by QLIKE at five and 21 sessions. GARCH-family models remain competitive, and EGARCH often performs well under volatility RMSE. The LSTM adds a meaningful deep-learning comparison but does not dominate the conventional models. VIX provides the clearest incremental benefit to HAR at short horizons. Finally, volatility targeting achieves a materially smoother historical return path at the cost of lower growth. The main contribution is a reproducible, multi-horizon comparison with explicit data-vintage checks and leakage-aware out-of-sample evaluation.

## Reproduction

From the repository root, place the three downloaded source files under `data/raw/` using the filenames described in the README, install `requirements.txt`, and run:

```bash
python run_pipeline.py
pip install -r requirements-lstm.txt
python scripts/run_lstm_extension.py
```

The LSTM extension can be skipped when PyTorch is unavailable. The main pipeline regenerates econometric and XGBoost outputs; the extension adds the LSTM forecasts and recomputes combined evaluation outputs.
