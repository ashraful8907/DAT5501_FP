import pandas as pd
from pathlib import Path

# Load geo 
geo_raw = pd.read_csv(
    "app-geography-lep-eda-202425-q4.csv",
    dtype={
        "local_enterprise_partnership_code": "string",
        "local_enterprise_partnership_name": "string",
        "english_devolved_area_code": "string",
        "english_devolved_area_name": "string",
    },
    low_memory=False,
)

geo = geo_raw.copy()

# Standardise column names
geo.columns = geo.columns.str.lower().str.strip().str.replace(" ", "_")

# Checking
print("Columns:", geo.columns.tolist())
print("Unique time_period:", geo["time_period"].unique())
print("Unique geographic_level:", geo["geographic_level"].unique())

# Build filter mask carefully
mask = geo["time_period"].astype(str).eq("202425")
mask &= geo["geographic_level"] == "Local enterprise partnership"
mask &= geo["ethnicity_major"] == "Total"
mask &= geo["apps_level"] == "Total"
mask &= geo["age_summary"] == "Total"
mask &= geo["sex"] == "Total"
mask &= geo["ssa_tier_1"] == "Total"

geo_clean = geo.loc[mask].copy()

print("Rows after filter:", len(geo_clean))

geo_clean = geo_clean.replace({"x": None, "c": None, "z": None, "low": 0})

geo_clean["starts"] = pd.to_numeric(geo_clean["starts"], errors="coerce")
geo_clean["achievements"] = pd.to_numeric(geo_clean["achievements"], errors="coerce")

print(geo_clean[["starts", "achievements"]].dtypes)

# Aggregate to one row per LEP
geo_clean = (
    geo_clean
    .groupby(
        ["local_enterprise_partnership_code",
         "local_enterprise_partnership_name"],
        as_index=False,
    )
    .agg({"starts": "sum", "achievements": "sum"})
)

# Compute achievement_rate and dropoff_rate
# avoid division by zero
mask_nonzero = geo_clean["starts"] > 0

geo_clean.loc[mask_nonzero, "achievement_rate"] = (
    geo_clean.loc[mask_nonzero, "achievements"] / geo_clean.loc[mask_nonzero, "starts"]
)
geo_clean.loc[mask_nonzero, "dropoff_rate"] = 1 - geo_clean.loc[mask_nonzero, "achievement_rate"]

geo_clean.loc[~mask_nonzero, "achievement_rate"] = None
geo_clean.loc[~mask_nonzero, "dropoff_rate"] = None


print(geo_clean.head())
print(geo_clean.tail())

# Save cleaned dataset
Path("data/processed").mkdir(parents=True, exist_ok=True)
geo_clean.to_csv("data/processed/geo_lep_clean.csv", index=False)
