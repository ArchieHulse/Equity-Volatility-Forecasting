from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .features import split_samples


def simple_benchmark_forecasts(panel: pd.DataFrame, x: pd.DataFrame, horizons=(1, 5, 21)) -> pd.DataFrame:
    """Historical-volatility and EWMA forecasts at every origin date."""
    r2 = panel["market_return"].pow(2)
    ewma_daily_var = r2.ewm(alpha=1.0 - 0.94, adjust=False).mean()
    out = []
    for h in horizons:
        hist_var = 252.0 * r2.rolling(22, min_periods=22).mean()
        ewma_var = 252.0 * ewma_daily_var
        for model, series in (("HistoricalVol_22d", hist_var), ("EWMA_0.94", ewma_var)):
            tmp = pd.DataFrame({"date": panel.index, "horizon": h, "model": model,
                                "forecast_variance": series.values})
            out.append(tmp)
    return pd.concat(out, ignore_index=True)


def _filter_conditional_variance(returns: np.ndarray, params: np.ndarray, kind: str,
                                initial_variance: float) -> np.ndarray:
    """Filter conditional variance for Gaussian-QMLE GARCH-family models.

    Returns are expressed in percentage points. The function is deliberately
    implemented without the external `arch` package so the core project stays
    reproducible in a minimal SciPy environment.
    """
    e = np.asarray(returns, dtype=float)
    h = np.empty(len(e), dtype=float)
    h[0] = max(float(initial_variance), 1e-8)
    if kind == "GARCH_1_1":
        omega, alpha, beta = params
        for t in range(1, len(e)):
            h[t] = omega + alpha * e[t-1]**2 + beta * h[t-1]
    elif kind == "GJR_GARCH_1_1":
        omega, alpha, gamma, beta = params
        for t in range(1, len(e)):
            shock = e[t-1]**2 * (alpha + gamma * (e[t-1] < 0.0))
            h[t] = omega + shock + beta * h[t-1]
    elif kind == "EGARCH_1_1":
        omega, alpha, gamma, beta = params
        log_h = np.log(h[0])
        expected_abs_normal = np.sqrt(2.0 / np.pi)
        for t in range(1, len(e)):
            z = e[t-1] / np.sqrt(h[t-1])
            log_h = (omega + beta * log_h
                     + alpha * (abs(z) - expected_abs_normal)
                     + gamma * z)
            h[t] = np.exp(np.clip(log_h, -20.0, 20.0))
    else:
        raise ValueError(f"Unknown variance model: {kind}")
    return np.maximum(h, 1e-10)


def _fit_garch_family(y_train: np.ndarray, kind: str) -> dict:
    """Fit a Gaussian quasi-maximum-likelihood GARCH-family model."""
    from scipy.optimize import minimize

    y_train = np.asarray(y_train, dtype=float)
    y_train = y_train[np.isfinite(y_train)]
    if len(y_train) < 500:
        raise ValueError(f"Not enough observations to fit {kind}: {len(y_train)}")
    sample_var = max(float(np.var(y_train)), 1e-6)
    if kind == "GARCH_1_1":
        x0 = np.array([sample_var * 0.05, 0.06, 0.90])
        bounds = [(1e-8, max(10.0, sample_var * 5)), (1e-5, 0.60), (1e-5, 0.999)]
    elif kind == "GJR_GARCH_1_1":
        x0 = np.array([sample_var * 0.05, 0.04, 0.06, 0.88])
        bounds = [(1e-8, max(10.0, sample_var * 5)), (1e-5, 0.60), (0.0, 0.80), (1e-5, 0.999)]
    elif kind == "EGARCH_1_1":
        x0 = np.array([-0.10, 0.15, -0.05, 0.95])
        bounds = [(-5.0, 2.0), (1e-5, 2.0), (-2.0, 2.0), (0.001, 0.998)]
    else:
        raise ValueError(kind)

    def objective(params: np.ndarray) -> float:
        if kind == "GARCH_1_1":
            persistence = params[1] + params[2]
        elif kind == "GJR_GARCH_1_1":
            persistence = params[1] + params[2] * 0.5 + params[3]
        else:
            persistence = params[3]
        penalty = 0.0
        if kind != "EGARCH_1_1" and persistence >= 0.9985:
            excess = persistence - 0.9985
            penalty = 1e6 + 1e9 * excess * excess
        e = y_train
        h = _filter_conditional_variance(e, params, kind, sample_var)
        if not np.isfinite(h).all() or (h <= 0).any():
            return 1e12
        burn = min(50, len(e) // 10)
        ll = 0.5 * np.sum(np.log(h[burn:]) + (e[burn:] ** 2) / h[burn:])
        return float(ll + penalty) if np.isfinite(ll) else 1e12

    fit = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 1200, "ftol": 1e-10, "maxls": 40})
    params = np.asarray(fit.x, dtype=float)
    if kind == "GARCH_1_1":
        persistence = float(params[1] + params[2])
    elif kind == "GJR_GARCH_1_1":
        persistence = float(params[1] + 0.5 * params[2] + params[3])
    else:
        persistence = float(params[3])
    if not np.isfinite(fit.fun) or fit.fun >= 1e11:
        raise RuntimeError(f"Could not estimate {kind}: {fit.message}")
    return {"params": params, "objective": float(fit.fun), "converged": bool(fit.success),
            "message": str(fit.message), "persistence": persistence,
            "n_train": int(len(y_train))}


def arch_family_forecasts(panel: pd.DataFrame, horizons=(1, 5, 21), test_start="2017-01-01",
                          egarch_simulations: int = 300) -> pd.DataFrame:
    """Fit GARCH, EGARCH and GJR-GARCH pre-test, then make sequential OOS forecasts.

    The estimation uses Gaussian quasi-maximum likelihood on returns in percent.
    Parameters are estimated only with observations before ``test_start`` and are
    held fixed during the test. The variance recursion is updated using each newly
    observed return, as would be possible in live forecasting. Multi-step GJR
    forecasts use the symmetric-normal expected shock; EGARCH multi-step forecasts
    use reproducible Monte Carlo integration of future standard-normal innovations.
    Forecasts are returned as annualised average variance in decimal-return units.
    """
    y = (panel["market_return"] * 100.0).astype(float)
    start = pd.Timestamp(test_start)
    train = y.loc[y.index < start]
    test_positions = np.flatnonzero(y.index >= start)
    if len(test_positions) == 0:
        return pd.DataFrame(columns=["date", "horizon", "model", "forecast_variance"])
    specs = ("GARCH_1_1", "EGARCH_1_1", "GJR_GARCH_1_1")
    out = []
    rng = np.random.default_rng(2026)
    max_h = max(horizons)
    expected_abs_normal = np.sqrt(2.0 / np.pi)

    for name in specs:
        fit = _fit_garch_family(train.to_numpy(), name)
        params = fit["params"]
        h_filtered = _filter_conditional_variance(y.to_numpy(), params, name,
                                                   max(float(np.var(train)), 1e-6))
        values = y.to_numpy()
        rows = {h: [] for h in horizons}
        dates = []
        for i in test_positions:
            h_t = h_filtered[i]
            e_t = values[i]
            if name == "GARCH_1_1":
                omega, alpha, beta = params
                v1 = omega + alpha * e_t**2 + beta * h_t
                persistence = alpha + beta
                path = np.empty(max_h, dtype=float)
                path[0] = v1
                for k in range(1, max_h):
                    path[k] = omega + persistence * path[k-1]
            elif name == "GJR_GARCH_1_1":
                omega, alpha, gamma, beta = params
                v1 = omega + alpha * e_t**2 + gamma * (e_t < 0.0) * e_t**2 + beta * h_t
                persistence = alpha + 0.5 * gamma + beta
                path = np.empty(max_h, dtype=float)
                path[0] = v1
                for k in range(1, max_h):
                    path[k] = omega + persistence * path[k-1]
            else:
                omega, alpha, gamma, beta = params
                z_t = e_t / np.sqrt(h_t)
                log_v1 = (omega + beta * np.log(h_t)
                          + alpha * (abs(z_t) - expected_abs_normal) + gamma * z_t)
                log_paths = np.full(egarch_simulations, log_v1, dtype=float)
                path = np.empty(max_h, dtype=float)
                for k in range(max_h):
                    var_paths = np.exp(np.clip(log_paths, -20.0, 20.0))
                    path[k] = float(var_paths.mean())
                    if k < max_h - 1:
                        z_future = rng.standard_normal(egarch_simulations)
                        log_paths = (omega + beta * log_paths
                                     + alpha * (np.abs(z_future) - expected_abs_normal)
                                     + gamma * z_future)
            for h in horizons:
                rows[h].append(float((252.0 / 10000.0) * np.mean(path[:h])))
            dates.append(y.index[i])

        for h in horizons:
            tmp = pd.DataFrame({"date": dates, "horizon": h, "model": name,
                                "forecast_variance": rows[h],
                                "fit_objective": fit["objective"],
                                "fit_converged": fit["converged"],
                                "fit_persistence": fit["persistence"],
                                "fit_n_train": fit["n_train"]})
            for j, value in enumerate(params):
                tmp[f"parameter_{j+1}"] = float(value)
            out.append(tmp)
    return pd.concat(out, ignore_index=True)

def har_forecasts(x: pd.DataFrame, targets: dict, horizons=(1, 5, 21),
                  test_start="2017-01-01", include_vix=True) -> pd.DataFrame:
    """HAR-RV log-variance model with validation-selected ridge regularisation.

    Inputs are logarithms of lagged daily/weekly/monthly realised-variance
    proxies. Features are standardised inside each training fit. Ridge strength
    is selected using validation log-RV MSE; a Duan smearing correction is
    estimated on validation residuals before the final pre-test refit.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    base_cols = ["rv_mean_5", "rv_mean_22", "rv_mean_63"]
    if include_vix and "vix_squared" in x.columns:
        base_cols.append("vix_squared")
    out = []
    test_start = pd.Timestamp(test_start)
    alphas = (0.01, 0.1, 1.0, 10.0, 100.0)

    def transform_features(frame: pd.DataFrame) -> pd.DataFrame:
        z = frame[base_cols].clip(lower=1e-12).apply(np.log)
        z.columns = [f"log_{c}" for c in z.columns]
        return z.replace([np.inf, -np.inf], np.nan)

    for h in horizons:
        s = split_samples(x, targets[h], h)
        Xtr = transform_features(s["X_train"])
        ytr = np.log(s["y_train"].clip(lower=1e-12))
        Xva = transform_features(s["X_val"])
        yva = np.log(s["y_val"].clip(lower=1e-12))
        train_ok = Xtr.notna().all(axis=1) & ytr.notna()
        val_ok = Xva.notna().all(axis=1) & yva.notna()
        if train_ok.sum() == 0 or val_ok.sum() == 0:
            continue

        best_alpha, best_score = None, float("inf")
        best_calibrator = None
        for alpha in alphas:
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            candidate.fit(Xtr.loc[train_ok], ytr.loc[train_ok])
            val_pred = candidate.predict(Xva.loc[val_ok])
            score = float(np.mean((yva.loc[val_ok].to_numpy() - val_pred) ** 2))
            if score < best_score:
                best_alpha, best_score, best_calibrator = alpha, score, candidate
        val_resid = yva.loc[val_ok].to_numpy() - best_calibrator.predict(Xva.loc[val_ok])
        smear = float(np.mean(np.exp(np.clip(val_resid, -20, 20))))

        pretest_ix = s["X_pretest"].index.intersection(s["y_pretest"].index)
        Xpre = transform_features(s["X_pretest"].loc[pretest_ix])
        ypre = np.log(s["y_pretest"].loc[pretest_ix].clip(lower=1e-12))
        pre_ok = Xpre.notna().all(axis=1) & ypre.notna()
        if pre_ok.sum() == 0:
            continue
        final_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        final_model.fit(Xpre.loc[pre_ok], ypre.loc[pre_ok])
        Xtest = transform_features(x.loc[x.index >= test_start])
        test_ok = Xtest.notna().all(axis=1)
        pred_log = final_model.predict(Xtest.loc[test_ok])
        pred = np.exp(np.clip(pred_log, -20, 20)) * smear
        out.append(pd.DataFrame({
            "date": Xtest.loc[test_ok].index,
            "horizon": h,
            "model": "HAR_RV_with_VIX" if include_vix else "HAR_RV_no_VIX",
            "forecast_variance": pred,
            "selected_alpha": float(best_alpha),
            "validation_log_mse": float(best_score),
            "smearing_factor": float(smear),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["date", "horizon", "model", "forecast_variance"])

def xgboost_forecasts(x: pd.DataFrame, targets: dict, horizons=(1, 5, 21), test_start="2017-01-01", include_vix=True) -> pd.DataFrame:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install xgboost with: pip install xgboost") from exc

    out = []
    test_start = pd.Timestamp(test_start)
    feature_cols = list(x.columns)
    if not include_vix:
        feature_cols = [c for c in feature_cols if not c.startswith("vix_")]

    def new_model():
        return XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                            reg_lambda=5.0, objective="reg:squarederror",
                            tree_method="hist", n_jobs=-1, random_state=2026)

    for h in horizons:
        s = split_samples(x, targets[h], h)
        Xtr, ytr = s["X_train"][feature_cols], np.log(s["y_train"].clip(lower=1e-12))
        Xva, yva = s["X_val"][feature_cols], np.log(s["y_val"].clip(lower=1e-12))
        tr_valid = Xtr.notna().all(axis=1) & ytr.notna()
        va_valid = Xva.notna().all(axis=1) & yva.notna()
        calibration_model = new_model().fit(Xtr.loc[tr_valid], ytr.loc[tr_valid])
        val_resid = yva.loc[va_valid].values - calibration_model.predict(Xva.loc[va_valid])
        smear = float(np.mean(np.exp(val_resid))) if len(val_resid) else 1.0

        train_dates = s["X_pretest"].index.intersection(s["y_pretest"].index)
        Xfit = s["X_pretest"].loc[train_dates, feature_cols]
        yfit = np.log(s["y_pretest"].loc[train_dates].clip(lower=1e-12))
        fit_mask = Xfit.notna().all(axis=1) & yfit.notna()
        Xtest = x.loc[x.index >= test_start, feature_cols]
        test_valid = Xtest.notna().all(axis=1)
        final_model = new_model().fit(Xfit.loc[fit_mask], yfit.loc[fit_mask])
        pred_log = final_model.predict(Xtest.loc[test_valid])
        pred = np.exp(pred_log) * smear
        out.append(pd.DataFrame({"date": Xtest.loc[test_valid].index, "horizon": h,
                                 "model": "XGBoost_with_VIX" if include_vix else "XGBoost_no_VIX", "forecast_variance": pred}))
    return pd.concat(out, ignore_index=True)


# Optional LSTM extension. It is kept separate so the main pipeline can run without a GPU.
def lstm_forecasts(x: pd.DataFrame, targets: dict, horizons=(1, 5, 21),
                   train_end="2009-12-31", validation_end="2016-12-31",
                   test_start="2017-01-01", seq_len=22, epochs=25) -> pd.DataFrame:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise ImportError("Install PyTorch with: pip install torch") from exc

    torch.manual_seed(2026)
    np.random.seed(2026)
    feature_cols = list(x.columns)
    raw = x[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    out = []

    class LSTMRegressor(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=32, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(0.15)
            self.head = nn.Linear(32, 1)
        def forward(self, z):
            seq, _ = self.lstm(z)
            return self.head(self.dropout(seq[:, -1, :])).squeeze(-1)

    for h in horizons:
        s = split_samples(raw, targets[h], h, train_end, validation_end, test_start)
        def sequence_array(index, y_series=None):
            Xs, ys, ds = [], [], []
            for dt in index:
                pos = raw.index.get_indexer([dt])[0]
                if pos < seq_len - 1:
                    continue
                seq = raw.iloc[pos-seq_len+1:pos+1].values
                if not np.isfinite(seq).all():
                    continue
                Xs.append(seq)
                ds.append(dt)
                if y_series is not None:
                    ys.append(float(y_series.loc[dt]))
            return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.Index(ds)
        train_ix = s["X_train"].index.intersection(s["y_train"].index)
        val_ix = s["X_val"].index.intersection(s["y_val"].index)
        pretest_ix = s["X_pretest"].index.intersection(s["y_pretest"].index)
        test_ix = s["X_test"].index
        Xtr, ytr, dtr = sequence_array(train_ix, np.log(s["y_train"].clip(lower=1e-12)))
        Xva, yva, dva = sequence_array(val_ix, np.log(s["y_val"].clip(lower=1e-12)))
        Xpre, ypre, dpre = sequence_array(pretest_ix, np.log(s["y_pretest"].clip(lower=1e-12)))
        Xte, _, dte = sequence_array(test_ix)
        scaler = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
        def scale(z):
            return scaler.transform(z.reshape(-1, z.shape[-1])).reshape(z.shape).astype(np.float32)
        Xtr, Xva, Xpre, Xte = map(scale, (Xtr, Xva, Xpre, Xte))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = LSTMRegressor(Xtr.shape[-1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
        loss_fn = nn.MSELoss()
        train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=64, shuffle=False)
        best_loss, best_epoch, best_state, patience = float("inf"), 1, None, 0
        for epoch in range(1, epochs + 1):
            net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = loss_fn(net(xb), yb); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(torch.tensor(Xva, device=device)), torch.tensor(yva, device=device)).item()
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, best_state, patience = val_loss, epoch, {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 7: break
        # Validation residual smearing corrects the log-to-level retransformation.
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            val_pred_log = net(torch.tensor(Xva, device=device)).cpu().numpy()
        smear = float(np.mean(np.exp(yva - val_pred_log))) if len(yva) else 1.0
        # Refit on all pre-test examples for the selected epoch count.
        Xall, yall = Xpre, ypre
        net = LSTMRegressor(Xall.shape[-1]).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
        loader = DataLoader(TensorDataset(torch.tensor(Xall), torch.tensor(yall)), batch_size=64, shuffle=False)
        for _ in range(best_epoch):
            net.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(); loss = loss_fn(net(xb), yb); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad(): pred_log = net(torch.tensor(Xte, device=device)).cpu().numpy()
        pred = np.exp(pred_log) * smear
        out.append(pd.DataFrame({"date": dte, "horizon": h, "model": "LSTM", "forecast_variance": pred}))
    return pd.concat(out, ignore_index=True)
