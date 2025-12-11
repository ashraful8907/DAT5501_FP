import pandas as pd
from pathlib import Path

imd = pd.read_csv("app-learner-deprivation-202425-q4.csv", low_memory = False)
imd.columns = imd.columns.str.lower().str.strip().str.replace(" ", "_")

mask = (
    (imd["time_period"].astype(str) == "202425") &
    (imd["geographic_level"] == "National") &
    (imd["age_summary"] == "Total") &
    (imd["apps_level"] == "Total") &
    (imd["ssa_tier_1"] == "Total")
)

imd_clean = imd.loc[mask].copy()
print("Rows after IMD filter:", len(imd_clean))
print(imd_clean[["learner_home_depriv", "starts", "achievements"]].head())

imd_clean = imd_clean.replace({"x" : None, "c" : None, "z" : None, "low" : 0})
imd_clean["starts"] = pd.to_numeric(imd_clean["starts"], errors = "coerce")
imd_clean["achievements"] = pd.to_numeric(imd_clean["achievements"], errors = "coerce")

imd_clean = (
    imd_clean
    .groupby("learner_home_depriv", as_index = False)
    .agg({"starts": "sum", "achievements" : "sum"})
)

mask_nonzero = imd_clean["starts"] > 0

imd_clean.loc[mask_nonzero, "achievement_rate"] = (
    imd_clean.loc[mask_nonzero, "achievements"] / imd_clean.loc[mask_nonzero, "starts"]
)
imd_clean.loc[mask_nonzero, "dropoff_rate"] = 1 - imd_clean.loc[mask_nonzero, "achievement_rate"]

imd_clean.loc[~mask_nonzero, ["achievement_rate", "dropoff_rate"]] = None

print(imd_clean)

Path("data/processed").mkdir(parents=True, exist_ok=True)
imd_clean.to_csv("data/processed/imd_clean.csv", index=False)
