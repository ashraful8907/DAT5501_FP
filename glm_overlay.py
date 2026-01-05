import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm

DATA_PATH = "data/processed/imd_quintiles_for_plot.csv"
FIG_PATH = "outputs/figures/fig_imd_glm_overlay.png"
MODEL_OUT = "outputs/models/imd_glm_overlay_results.csv"

def main():
    # --- Load ---
    df = pd.read_csv(DATA_PATH)

    # --- Validate columns ---
    required = {"learner_home_depriv", "starts", "achievements"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {DATA_PATH}: {missing}")

    # --- Map quintiles ---
    mapping = {
        "One (most deprived)": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five (least deprived)": 5,
    }
    df = df[df["learner_home_depriv"].isin(mapping.keys())].copy()
    df["imd_q"] = df["learner_home_depriv"].map(mapping).astype(int)
    df = df.sort_values("imd_q")

    # --- Numeric + derived ---
    df["starts"] = pd.to_numeric(df["starts"], errors="coerce")
    df["achievements"] = pd.to_numeric(df["achievements"], errors="coerce")
    df = df.dropna(subset=["starts", "achievements"])
    df = df[df["starts"] > 0].copy()

    df["failures"] = df["starts"] - df["achievements"]
    df["rate"] = df["achievements"] / df["starts"]

    endog = np.column_stack([df["achievements"], df["failures"]])

    # --- Models ---
    X1 = sm.add_constant(df[["imd_q"]].astype(float))
    m1 = sm.GLM(endog, X1, family=sm.families.Binomial()).fit()

    df["imd_q2"] = df["imd_q"] ** 2
    X2 = sm.add_constant(df[["imd_q", "imd_q2"]].astype(float))
    m2 = sm.GLM(endog, X2, family=sm.families.Binomial()).fit()

    # --- Predictions ---
    grid = np.linspace(1, 5, 200)
    grid_df = pd.DataFrame({"imd_q": grid})

    grid_X1 = sm.add_constant(grid_df, has_constant="add")
    pred1 = m1.predict(grid_X1)

    grid_df["imd_q2"] = grid_df["imd_q"] ** 2
    grid_X2 = sm.add_constant(grid_df[["imd_q", "imd_q2"]], has_constant="add")
    pred2 = m2.predict(grid_X2)

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    plt.scatter(df["imd_q"], df["rate"])
    plt.plot(grid, pred1, label="GLM trend")
    plt.plot(grid, pred2, label="GLM quadratic")

    plt.xticks([1, 2, 3, 4, 5], ["1 (Most deprived)", "2", "3", "4", "5 (Least deprived)"])
    plt.ylabel("Achievement rate")
    plt.xlabel("IMD deprivation quintile")
    plt.title("Achievement rate by deprivation quintile with GLM fits (England, 2024/25)")
    plt.legend()
    plt.tight_layout()

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_PATH, dpi=300)
    plt.show()

    # --- Save model summary table for report ---
    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame([
        {
            "model": "GLM trend",
            "coef_const": float(m1.params["const"]),
            "coef_imd_q": float(m1.params["imd_q"]),
            "aic": float(m1.aic),
        },
        {
            "model": "GLM quadratic",
            "coef_const": float(m2.params["const"]),
            "coef_imd_q": float(m2.params["imd_q"]),
            "coef_imd_q2": float(m2.params["imd_q2"]),
            "aic": float(m2.aic),
        },
    ])
    out.to_csv(MODEL_OUT, index=False)
    print(f"Saved figure: {FIG_PATH}")
    print(f"Saved model table: {MODEL_OUT}")

if __name__ == "__main__":
    main()
