import pandas as pd
import numpy as np
from pathlib import Path

import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

DATA_PATH = "data/processed/geo_lep_clean.csv"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def loocv_rmse_ols(X: np.ndarray, y: np.ndarray) -> float:
    """LOOCV RMSE for OLS using statsmodels."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = sm.add_constant(X[train_idx])
        X_test = sm.add_constant(X[test_idx], has_constant="add")

        m = sm.OLS(y[train_idx], X_train).fit()
        preds[test_idx] = m.predict(X_test)[0]

    return rmse(y, preds)


def loocv_rmse_wls(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """LOOCV RMSE for WLS using statsmodels."""
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = sm.add_constant(X[train_idx])
        X_test = sm.add_constant(X[test_idx], has_constant="add")

        m = sm.WLS(y[train_idx], X_train, weights=weights[train_idx]).fit()
        preds[test_idx] = m.predict(X_test)[0]

    return rmse(y, preds)


def loocv_rmse_binom_glm(X: np.ndarray, success: np.ndarray, total: np.ndarray) -> float:
    """
    LOOCV RMSE for a Binomial GLM.
    We fit a binomial model on counts, then predict a probability p for the held-out row.
    RMSE is computed on the achievement RATE (success/total).
    """
    loo = LeaveOneOut()

    y_rate = success / total
    preds_rate = np.zeros(len(y_rate), dtype=float)

    for train_idx, test_idx in loo.split(X):
        X_train = sm.add_constant(X[train_idx])
        X_test = sm.add_constant(X[test_idx], has_constant="add")

        succ_train = success[train_idx]
        tot_train = total[train_idx]
        fail_train = tot_train - succ_train

        endog_train = np.column_stack([succ_train, fail_train])

        glm = sm.GLM(endog_train, X_train, family=sm.families.Binomial()).fit()
        p_hat = glm.predict(X_test)[0]  # predicted probability of success

        # Convert predicted probability into predicted rate (same thing)
        preds_rate[test_idx] = p_hat

    return rmse(y_rate, preds_rate)


def main():
    lep = pd.read_csv(DATA_PATH)

    # ---- Sanity checks ----
    assert lep.shape[0] == lep["local_enterprise_partnership_name"].nunique(), "LEP rows not unique"
    assert (lep["starts"] > 0).all(), "starts must be > 0"
    assert (lep["achievements"] >= 0).all(), "achievements must be >= 0"
    assert (lep["achievements"] <= lep["starts"]).all(), "achievements cannot exceed starts"
    assert lep["achievement_rate"].between(0, 1).all(), "achievement_rate outside [0,1]"

    # ---- Variables ----
    starts = lep["starts"].to_numpy()
    achievements = lep["achievements"].to_numpy()
    y_rate = lep["achievement_rate"].to_numpy()

    # Rescale starts for stability + interpretability
    lep["starts_10k"] = lep["starts"] / 10_000
    X = lep[["starts_10k"]].to_numpy()

    # Optional log predictor
    lep["log_starts"] = np.log(lep["starts"])
    X_log = lep[["log_starts"]].to_numpy()

    # ---- Proper LOOCV baseline: "predict the training mean rate" ----
    loo = LeaveOneOut()
    baseline_preds = np.zeros(len(y_rate), dtype=float)

    for train_idx, test_idx in loo.split(y_rate):
        baseline_preds[test_idx] = y_rate[train_idx].mean()

    baseline_rmse_loocv = rmse(y_rate, baseline_preds)
    baseline_rmse_in_sample = rmse(y_rate, np.full_like(y_rate, y_rate.mean(), dtype=float))

    print("\n=== Baseline (mean rate) ===")
    print("In-sample RMSE:", baseline_rmse_in_sample)
    print("LOOCV RMSE:", baseline_rmse_loocv)

    # ---- OLS on rates ----
    X_sm = sm.add_constant(lep[["starts_10k"]])
    ols = sm.OLS(y_rate, X_sm).fit()
    ols_rmse_loocv = loocv_rmse_ols(X, y_rate)

    print("\n=== OLS: achievement_rate ~ starts_10k ===")
    print(ols.summary())
    print("LOOCV RMSE:", ols_rmse_loocv)

    # ---- WLS on rates (weights = starts) ----
    wls = sm.WLS(y_rate, X_sm, weights=starts).fit()
    wls_rmse_loocv = loocv_rmse_wls(X, y_rate, weights=starts)

    print("\n=== WLS (weights=starts): achievement_rate ~ starts_10k ===")
    print(wls.summary())
    print("LOOCV RMSE:", wls_rmse_loocv)

    # ---- OLS with log(starts) (optional check) ----
    X_log_sm = sm.add_constant(lep[["log_starts"]])
    ols_log = sm.OLS(y_rate, X_log_sm).fit()
    ols_log_rmse_loocv = loocv_rmse_ols(X_log, y_rate)

    print("\n=== OLS: achievement_rate ~ log(starts) ===")
    print(ols_log.summary())
    print("LOOCV RMSE:", ols_log_rmse_loocv)

    # ---- Binomial GLM on counts (recommended) ----
    lep["failures"] = lep["starts"] - lep["achievements"]
    endog = np.column_stack([lep["achievements"], lep["failures"]])

    glm = sm.GLM(endog, X_sm, family=sm.families.Binomial()).fit()
    glm_rmse_loocv = loocv_rmse_binom_glm(X, achievements, starts)

    print("\n=== Binomial GLM: achievements out of starts ~ starts_10k ===")
    print(glm.summary())
    print("LOOCV RMSE (on achievement_rate):", glm_rmse_loocv)

    # ---- Collect results table ----
    results = pd.DataFrame(
        [
            {
                "model": "Baseline (mean rate)",
                "predictor": "const",
                "coef": np.nan,
                "pvalue": np.nan,
                "r2": np.nan,
                "rmse_loocv": baseline_rmse_loocv,
            },
            {
                "model": "OLS (rate)",
                "predictor": "starts_10k",
                "coef": float(ols.params["starts_10k"]),
                "pvalue": float(ols.pvalues["starts_10k"]),
                "r2": float(ols.rsquared),
                "rmse_loocv": float(ols_rmse_loocv),
            },
            {
                "model": "WLS (rate, weights=starts)",
                "predictor": "starts_10k",
                "coef": float(wls.params["starts_10k"]),
                "pvalue": float(wls.pvalues["starts_10k"]),
                "r2": float(wls.rsquared),
                "rmse_loocv": float(wls_rmse_loocv),
            },
            {
                "model": "OLS (rate)",
                "predictor": "log_starts",
                "coef": float(ols_log.params["log_starts"]),
                "pvalue": float(ols_log.pvalues["log_starts"]),
                "r2": float(ols_log.rsquared),
                "rmse_loocv": float(ols_log_rmse_loocv),
            },
            {
                "model": "Binomial GLM (counts)",
                "predictor": "starts_10k",
                "coef": float(glm.params["starts_10k"]),
                "pvalue": float(glm.pvalues["starts_10k"]),
                "r2": np.nan,  # not directly comparable to OLS R^2
                "rmse_loocv": float(glm_rmse_loocv),
            },
        ]
    )

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    results.to_csv("outputs/models/lep_model_results_with_glm.csv", index=False)

    print("\n=== Model comparison table ===")
    print(results.to_string(index=False))
    print("\nSaved: outputs/models/lep_model_results_with_glm.csv")


if __name__ == "__main__":
    main()
