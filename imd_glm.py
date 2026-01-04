import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

DATA_PATH = "data/processed/imd_clean.csv"  # change if needed


def main():
    imd = pd.read_csv(DATA_PATH)

    # Standardise column names
    imd.columns = imd.columns.str.lower().str.strip().str.replace(" ", "_")

    required = {"learner_home_depriv", "starts", "achievements"}
    missing = required - set(imd.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found columns: {list(imd.columns)}")

    # Keep only the five IMD quintiles (drop Total/Unknown)
    keep = [
        "One (most deprived)",
        "Two",
        "Three",
        "Four",
        "Five (least deprived)",
    ]
    imd = imd[imd["learner_home_depriv"].isin(keep)].copy()

    # Force numeric for counts
    imd["starts"] = pd.to_numeric(imd["starts"], errors="coerce")
    imd["achievements"] = pd.to_numeric(imd["achievements"], errors="coerce")

    # Drop bad rows
    imd = imd.dropna(subset=["starts", "achievements"])
    imd = imd[(imd["starts"] > 0) & (imd["achievements"] >= 0) & (imd["achievements"] <= imd["starts"])].copy()

    # Compute derived fields
    imd["failures"] = imd["starts"] - imd["achievements"]
    imd["achievement_rate"] = imd["achievements"] / imd["starts"]

    # Ordered categorical
    imd["imd_group"] = pd.Categorical(imd["learner_home_depriv"], categories=keep, ordered=True)

    # Build design matrix: intercept + dummies (reference = "One (most deprived)")
    X = pd.get_dummies(imd["imd_group"], drop_first=True)
    X = sm.add_constant(X)

    # ---- HARD FIXES (these prevent your crash) ----
    # Ensure numeric float matrix
    X = X.astype(float)

    # Build endog as (successes, failures) numeric array
    endog = np.column_stack([
        imd["achievements"].astype(float).to_numpy(),
        imd["failures"].astype(float).to_numpy(),
    ])

    # Final safety checks BEFORE fit (prints info if something is wrong)
    if np.isnan(endog).any() or np.isinf(endog).any():
        raise ValueError("endog contains NaN/inf after cleaning.")
    if np.isnan(X.to_numpy()).any() or np.isinf(X.to_numpy()).any():
        raise ValueError("X contains NaN/inf after cleaning.")
    if endog.shape[0] != X.shape[0]:
        raise ValueError(f"Row mismatch: endog rows={endog.shape[0]}, X rows={X.shape[0]}")

    # Fit Binomial GLM
    glm = sm.GLM(endog, X, family=sm.families.Binomial()).fit()

    print("\n=== IMD Binomial GLM: achievement by deprivation group ===")
    print(glm.summary())

    # Odds ratios table
    params = glm.params
    conf = glm.conf_int()
    or_table = pd.DataFrame({
        "term": params.index,
        "coef_logodds": params.values,
        "odds_ratio": np.exp(params.values),
        "or_ci_lower": np.exp(conf[0].values),
        "or_ci_upper": np.exp(conf[1].values),
        "pvalue": glm.pvalues.values,
    })

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    or_table.to_csv("outputs/models/imd_glm_odds_ratios.csv", index=False)
    print("\nSaved: outputs/models/imd_glm_odds_ratios.csv")

    # Save plotting table
    plot_df = imd[["learner_home_depriv", "starts", "achievements", "achievement_rate"]].copy()
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    plot_df.to_csv("data/processed/imd_quintiles_for_plot.csv", index=False)
    print("Saved: data/processed/imd_quintiles_for_plot.csv")

    print("\nRows used:", len(imd))
    print("Groups:", imd["learner_home_depriv"].tolist())


if __name__ == "__main__":
    main()
