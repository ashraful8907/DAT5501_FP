import pandas as pd

# --- Load datasets with correct dtypes for geo ---
geo = pd.read_csv(
    "app-geography-lep-eda-202425-q4.csv",
    dtype={
        "local_enterprise_partnership_code": "string",
        "local_enterprise_partnership_name": "string",
        "english_devolved_area_code": "string",
        "english_devolved_area_name": "string",
    },
    low_memory=False,
)

imd = pd.read_csv("app-learner-deprivation-202425-q4.csv")

# --- Standardise column names ---
geo.columns = geo.columns.str.lower().str.strip().str.replace(" ", "_")

# --- Filter rows ---
geo_filter = geo[
    (geo["time_period"] == "202425")
    & (geo["geographic_level"] == "Local enterprise partnership")
    & (geo["ethnicity_major"] == "Total")
    & (geo["age_summary"] == "Total")
    & (geo["apps_level"] == "Total")
    & (geo["sex"] == "Total")
    & (geo["ssa_tier_1"] == "Total")
]

geo_filter = geo_filter.replace({
    "low":0
})

geo_filter["starts"] = pd.to_numeric(geo_filter["starts"])
geo_filter["achievements"] = pd.to_numeric(geo_filter["achievements"])

geo_filter = geo

# Aggregate to one row per LEP

geo["achievement_rate"] = geo["achievements"] / geo["starts"]
geo["dropoff_rate"] = 1 - geo["achievement_rate"]
geo.loc[geo["starts"] == 0, "achievement_rate"] = None
geo.loc[geo["starts"] == 0, "dropoff_rate"] = None

geo = geo.dropna(subset=["starts"])
geo = geo[(geo["starts"] > 0)]


print(geo.head())
print(geo.tail())
