import pandas as pd
import numpy as np
from pathlib import Path

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

DATA_PATH = "data/processed/geo_lep_clean.csv"

def loocv_rmse(X: np.ndarray, y: np.ndarray, model) -> float:
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    return float(np.sqrt(mean_squared_error(y, preds)))

def main():
    lep = pd.read_csv(DATA_PATH)

    # Basic sanity checks (fail fast)
    assert lep.shape[0] == lep["local_enterprise_partnership_name"].nunique(), "LEP rows not unique"
    assert lep["achievement_rate"].between(0, 1).all(), "achievement_rate outside [0,1]"
    assert (lep["starts"] > 0).all(), "starts must be > 0"

    y = lep["achievement_rate"].to_numpy()

    # -----------------------------
    # Model 0: Baseline (mean)
    # -----------------------------
    y_mean = y.mean()
    baseline_rmse = float(np.sqrt(np.mean((y - y_mean) ** 2)))

    # -----------------------------
    # Model 1: OLS (unweighted)
    # -----------------------------
    X1 = lep[["starts"]]
    X1_sm = sm.add_constant(X1)
    ols = sm.OLS(y, X1_sm).fit()

    # LOOCV using sklearn for comparable RMSE
    X1_np = X1.to_numpy()
    lr = LinearRegression()
    ols_loocv_rmse = loocv_rmse(X1_np, y, lr)

    # -----------------------------
    # Model 2: WLS (weights = starts)
    # Rationale: higher starts => more precise rate
    # -----------------------------
    w = lep["starts"].to_numpy()
    wls = sm.WLS(y, X1_sm, weights=w).fit()

    # -----------------------------
    # Optional: log(starts) model
    # Useful if starts is highly skewed (London outlier)
    # -----------------------------
    lep["log_starts"] = np.log(lep["starts"])
    Xlog = lep[["log_starts"]]
    Xlog_sm = sm.add_constant(Xlog)
    ols_log = sm.OLS(y, Xlog_sm).fit()

    Xlog_np = Xlog.to_numpy()
    ols_log_loocv_rmse = loocv_rmse(Xlog_np, y, lr)

    # -----------------------------
    # Save results for your report appendix
    # -----------------------------
    results = pd.DataFrame([
        {
            "model": "Baseline (mean)",
            "coef_starts": np.nan,
            "pvalue_starts": np.nan,
            "r2": np.nan,
            "rmse_loocv": np.nan,
            "rmse_in_sample": baseline_rmse
        },
        {
            "model": "OLS: achievement_rate ~ starts",
            "coef_starts": ols.params["starts"],
            "pvalue_starts": ols.pvalues["starts"],
            "r2": ols.rsquared,
            "rmse_loocv": ols_loocv_rmse,
            "rmse_in_sample": float(np.sqrt(np.mean(ols.resid ** 2)))
        },
        {
            "model": "WLS (weights=starts): achievement_rate ~ starts",
            "coef_starts": wls.params["starts"],
            "pvalue_starts": wls.pvalues["starts"],
            "r2": wls.rsquared,
            "rmse_loocv": np.nan,  # keep simple; optional to implement weighted CV
            "rmse_in_sample": float(np.sqrt(np.mean(wls.resid ** 2)))
        },
        {
            "model": "OLS: achievement_rate ~ log(starts)",
            "coef_starts": ols_log.params["log_starts"],
            "pvalue_starts": ols_log.pvalues["log_starts"],
            "r2": ols_log.rsquared,
            "rmse_loocv": ols_log_loocv_rmse,
            "rmse_in_sample": float(np.sqrt(np.mean(ols_log.resid ** 2)))
        }
    ])

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    results.to_csv("outputs/models/lep_model_results.csv", index=False)

    # Print human-readable summaries for you while developing
    print("\n=== Baseline RMSE (mean predictor) ===")
    print(baseline_rmse)

    print("\n=== OLS summary: achievement_rate ~ starts ===")
    print(ols.summary())

    print("\n=== WLS summary (weights=starts): achievement_rate ~ starts ===")
    print(wls.summary())

    print("\n=== OLS summary: achievement_rate ~ log(starts) ===")
    print(ols_log.summary())

    print("\nSaved: outputs/models/lep_model_results.csv")

if __name__ == "__main__":
    main()
