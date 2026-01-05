# robustness_lep.py
# London influence robustness check + cleaner output formatting

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


def fit_models(df: pd.DataFrame) -> dict:
    """
    Fit baseline, OLS(rate), WLS(rate, weights=starts).
    Returns dict of key numbers.
    """
    df = df.copy()

    # sanity
    assert df["achievement_rate"].between(0, 1).all()
    assert (df["starts"] > 0).all()

    # scale starts for interpretability (per 10k)
    df["starts_10k"] = df["starts"] / 10_000.0

    y = df["achievement_rate"].to_numpy(dtype=float)

    # baseline (mean predictor) LOOCV is identical to in-sample for mean
    y_mean = float(y.mean())
    baseline_loocv = float(np.sqrt(np.mean((y - y_mean) ** 2)))

    # OLS (rate)
    X = df[["starts_10k"]].to_numpy(dtype=float)
    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit()
    ols_loocv = loocv_rmse(X, y, LinearRegression())

    # WLS (rate, weights=starts)
    w = df["starts"].to_numpy(dtype=float)
    wls = sm.WLS(y, X_sm, weights=w).fit()

    return {
        "n": int(df.shape[0]),
        "baseline_loocv_rmse": baseline_loocv,
        "ols_coef": float(ols.params[1]),
        "ols_pvalue": float(ols.pvalues[1]),
        "ols_r2": float(ols.rsquared),
        "ols_loocv_rmse": ols_loocv,
        "wls_coef": float(wls.params[1]),
        "wls_pvalue": float(wls.pvalues[1]),
        "wls_r2": float(wls.rsquared),
    }


def influence_diagnostics(df: pd.DataFrame) -> dict:
    """
    Influence diagnostics for OLS(rate) model: Cook's distance & leverage.
    """
    df = df.copy()
    df["starts_10k"] = df["starts"] / 10_000.0

    y = df["achievement_rate"].to_numpy(dtype=float)
    X = df[["starts_10k"]].to_numpy(dtype=float)
    X_sm = sm.add_constant(X)

    ols = sm.OLS(y, X_sm).fit()
    infl = ols.get_influence()

    cooks_d = infl.cooks_distance[0]        # array
    leverage = infl.hat_matrix_diag         # array

    i_cook = int(np.argmax(cooks_d))
    i_lev = int(np.argmax(leverage))

    return {
        "most_influential_lep": str(df.iloc[i_cook]["local_enterprise_partnership_name"]),
        "max_cooks_d": float(cooks_d[i_cook]),
        "max_leverage": float(leverage[i_lev]),
    }


def main():
    lep = pd.read_csv(DATA_PATH)

    # basic fail-fast checks
    assert lep.shape[0] == lep["local_enterprise_partnership_name"].nunique(), "LEP rows not unique"
    assert lep["achievement_rate"].between(0, 1).all(), "achievement_rate outside [0,1]"
    assert (lep["starts"] > 0).all(), "starts must be > 0"

    # --- run for all LEPs ---
    all_stats = fit_models(lep)
    all_infl = influence_diagnostics(lep)

    # --- run excluding London ---
    lep_no_london = lep[lep["local_enterprise_partnership_name"] != "London"].copy()
    no_ldn_stats = fit_models(lep_no_london)
    no_ldn_infl = influence_diagnostics(lep_no_london)

    out = pd.DataFrame([
        {"subset": "All LEPs", **all_stats, **all_infl},
        {"subset": "Exclude London", **no_ldn_stats, **no_ldn_infl},
    ])

    # -----------------------------
    # EDIT YOU ASKED FOR:
    # Clean up any weird whitespace/newlines in the LEP name before printing/saving
    # -----------------------------
    out["most_influential_lep"] = (
        out["most_influential_lep"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # nicer numeric formatting in console
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\n=== London influence robustness check (LEP) ===")
    print(out)

    # save
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    out.to_csv("outputs/models/lep_london_robustness_check.csv", index=False)
    print("\nSaved: outputs/models/lep_london_robustness_check.csv")


if __name__ == "__main__":
    main()
