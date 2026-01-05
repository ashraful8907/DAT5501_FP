# figure_lep_top_bottom.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "data/processed/geo_lep_clean_2.csv"
OUT_PATH = "outputs/figures/fig_lep_top_bottom.png"

def main():
    df = pd.read_csv(DATA_PATH).copy()
    df["achievement_rate"] = pd.to_numeric(df["achievement_rate"], errors="coerce")
    df = df.dropna(subset=["achievement_rate"]).sort_values("achievement_rate")

    bottom = df.head(10).copy()
    top = df.tail(10).copy()

    bottom["group"] = "Bottom 10"
    top["group"] = "Top 10"

    plot_df = pd.concat([bottom, top], ignore_index=True)
    plot_df["label"] = plot_df["local_enterprise_partnership_name"].astype(str)

    # colours: bottom red, top green
    colors = plot_df["group"].map({"Bottom 10": "tab:red", "Top 10": "tab:green"})

    plt.figure(figsize=(12, 7))
    plt.barh(plot_df["label"], plot_df["achievement_rate"], color=colors)

    # separator line between bottom and top
    plt.axhline(9.5, linewidth=1)

    plt.xlabel("Achievement rate")
    plt.title("Top 10 and Bottom 10 LEPs by apprenticeship achievement rate (2024/25)", pad=15)

    # OPTIONAL: add labels at end of bars
    for y, val in enumerate(plot_df["achievement_rate"]):
        plt.text(val + 0.003, y, f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # stops title clipping

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)
    plt.show()

    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
