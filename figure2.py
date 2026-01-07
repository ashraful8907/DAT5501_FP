import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

lep = pd.read_csv("data/processed/geo_lep_clean.csv")

x = lep["starts"]
y = lep["achievement_rate"]

plt.figure(figsize=(8, 5))
plt.scatter(x, y)

plt.xlabel("Apprenticeship starts (LEP, 2024/25)")
plt.ylabel("Achievement rate")
plt.title("LEP apprenticeship starts vs achievement rate (England, 2024/25)")

plt.tight_layout()

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/figures/fig_s2_lep_starts_vs_achievement.png", dpi=300)

plt.show()
