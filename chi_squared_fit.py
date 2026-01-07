import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm

DATA_PATH = "data/processed/imd_clean.csv"
OUT_FIG = "outputs/figures/fig_imd_polyfit_overlay.png"
OUT_TABLE = "outputs/models/imd_polyfit_model_comparison.csv"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

keep = [
    "One (most deprived)",
    "Two",
    "Three",
    "Four",
    "Five (least deprived)",
]
df = df[df["learner_home_depriv"].isin(keep)].copy()

df["starts"] = pd.to_numeric(df["starts"], errors="coerce")
df["achievements"] = pd.to_numeric(df["achievements"], errors="coerce")
df = df.dropna(subset=["starts", "achievements"])

df = df[(df["starts"] > 0) & (df["achievements"] >= 0) & (df["achievements"] <= df["starts"])].copy()

df["achievement_rate"] = df["achievements"] / df["starts"]

# Map IMD label -> 1..5 
mapping = {
    "One (most deprived)": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five (least deprived)": 5,
}
df["imd_q"] = df["learner_home_depriv"].map(mapping).astype(int)
df = df.sort_values("imd_q")

y = df["achievement_rate"].to_numpy(dtype=float)
w = df["starts"].to_numpy(dtype=float)  # weights = starts
X1 = sm.add_constant(df[["imd_q"]].astype(float))

# Model 1 (linear)
m1 = sm.WLS(y, X1, weights=w).fit()

# Model 2 (quadratic)
df["imd_q2"] = df["imd_q"] ** 2
X2 = sm.add_constant(df[["imd_q", "imd_q2"]].astype(float))
m2 = sm.WLS(y, X2, weights=w).fit()

# Chi-squared style fit (weighted SSE) + per DoF
pred1 = m1.predict(X1).to_numpy(dtype=float)
pred2 = m2.predict(X2).to_numpy(dtype=float)

chi2_1 = float(np.sum(w * (y - pred1) ** 2))
chi2_2 = float(np.sum(w * (y - pred2) ** 2))

chi2_dof_1 = chi2_1 / float(m1.df_resid)
chi2_dof_2 = chi2_2 / float(m2.df_resid)

print("\n=== Model fit ===")
print("Linear:   chi2/DoF =", round(chi2_dof_1, 6), "| BIC =", round(m1.bic, 3))
print("Quadratic:chi2/DoF =", round(chi2_dof_2, 6), "| BIC =", round(m2.bic, 3))

# Save results table (for report)
out = pd.DataFrame([
    {
        "model": "Linear",
        "chi2": chi2_1,
        "chi2_per_dof": chi2_dof_1,
        "bic": float(m1.bic),
        "aic": float(m1.aic),
        "coef_const": float(m1.params["const"]),
        "coef_imd_q": float(m1.params["imd_q"]),
        "p_imd_q": float(m1.pvalues["imd_q"]),
    },
    {
        "model": "Quadratic",
        "chi2": chi2_2,
        "chi2_per_dof": chi2_dof_2,
        "bic": float(m2.bic),
        "aic": float(m2.aic),
        "coef_const": float(m2.params["const"]),
        "coef_imd_q": float(m2.params["imd_q"]),
        "coef_imd_q2": float(m2.params["imd_q2"]),
        "p_imd_q": float(m2.pvalues["imd_q"]),
        "p_imd_q2": float(m2.pvalues["imd_q2"]),
    },
])

Path("outputs/models").mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_TABLE, index=False)
print("Saved:", OUT_TABLE)

grid = np.linspace(1, 5, 200)
grid_df = pd.DataFrame({"imd_q": grid})

grid_X1 = sm.add_constant(grid_df, has_constant="add")
grid_pred1 = m1.predict(grid_X1)

grid_df["imd_q2"] = grid_df["imd_q"] ** 2
grid_X2 = sm.add_constant(grid_df[["imd_q", "imd_q2"]], has_constant="add")
grid_pred2 = m2.predict(grid_X2)

# Plot
plt.figure(figsize=(9, 5))
plt.scatter(df["imd_q"], df["achievement_rate"])
plt.plot(grid, grid_pred1, label="Linear fit (WLS)")
plt.plot(grid, grid_pred2, label="Quadratic fit (WLS)")

plt.xticks([1, 2, 3, 4, 5], ["1 (Most deprived)", "2", "3", "4", "5 (Least deprived)"])
plt.ylabel("Achievement rate")
plt.xlabel("IMD deprivation quintile")
plt.title("Achievement rate by deprivation quintile with polynomial fits (England, 2024/25)")
plt.legend()
plt.tight_layout()

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print("Saved:", OUT_FIG)
