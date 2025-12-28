import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load cleaned data
imd = pd.read_csv("data/processed/imd_clean.csv")

x = imd["imd_quintile"]
y = imd["achievement_rate"]

plt.figure(figsize=(8, 5))
plt.bar(x, y)

plt.xticks(
    [1, 2, 3, 4, 5],
    ["1 (Most deprived)", "2", "3", "4", "5 (Least deprived)"]
)
plt.ylabel("Achievement rate")
plt.xlabel("IMD deprivation quintile")
plt.title("Apprenticeship achievement rate by deprivation quintile (England, 2024/25)")

# Value labels
for xi, yi in zip(x, y):
    plt.text(xi, yi + 0.002, f"{yi:.3f}", ha="center", fontsize=9)

plt.ylim(0, max(y) + 0.05)
plt.tight_layout()

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/fig1_imd_achievement_rate.png", dpi=300)
plt.show()
