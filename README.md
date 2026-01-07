# DAT5501_FP — Final Project (DAT5501: Analysis, Software and Career Practice)

This repository contains a reproducible Python analysis of **apprenticeship achievement rates (England, 2024/25)**. It focuses on how achievement outcomes vary by:

- **Local Enterprise Partnership (LEP)** (regional variation + starts vs achievement rate)
- **IMD deprivation quintile** (national deprivation pattern)

The workflow is:

1. Clean raw datasets  
2. Generate figures  
3. Run models + robustness checks  
4. Run unit tests (CI-ready)

---

## Repository structure

DAT5501_FP/
├── .circleci/
│   └── config.yml
├── data/
│   └── processed/
│       ├── geo_lep_clean.csv
│       ├── geo_lep_clean_2.csv
│       ├── imd_clean.csv
│       └── imd_quintiles_for_plot.csv
├── outputs/
│   ├── figures/
│   │   ├── fig_imd_glm_overlay.png
│   │   ├── fig_imd_polyfit_overlay.png
│   │   ├── fig_lep_top_bottom.png
│   │   ├── fig_s2_lep_starts_vs_achievement.png
│   │   └── fig1_imd_achievement_rate.png
│   └── models/
│       ├── imd_glm_odds_ratios.csv
│       ├── imd_glm_overlay_results.csv
│       ├── imd_glm_trend_results.csv
│       ├── imd_polyfit_model_comparison.csv
│       ├── lep_london_robustness_check.csv
│       ├── lep_london_robustness.csv
│       ├── lep_model_results_with_glm.csv
│       └── lep_model_results.csv
├── raw_data/
│   ├── app-geography-lep-eda-202425-q4.csv
│   └── app-learner-deprivation-202425-q4.csv
├── chi_squared_fit.py
├── cleaning_geo.py
├── cleaning_imd.py
├── figure1.py
├── figure2.py
├── london.py
├── model.py
├── top_bottom_lep.py
├── test.py
├── requirements.txt
└── README.md

## Data inputs

Raw data is stored in `raw_data/`:

- `raw_data/app-geography-lep-eda-202425-q4.csv` (LEP geography dataset)
- `raw_data/app-learner-deprivation-202425-q4.csv` (IMD deprivation dataset)

> These CSVs can be large. If you publish this repo publicly, consider adding `raw_data/` to `.gitignore` (or use Git LFS) and document how to obtain the datasets.

---

## Processed data outputs

Cleaning scripts create files in `data/processed/`:

- `geo_lep_clean.csv` / `geo_lep_clean_2.csv`  
  Cleaned LEP-level dataset (starts, achievements, `achievement_rate`, `dropoff_rate`)

- `imd_clean.csv`  
  Cleaned IMD quintile dataset (five rows: quintiles 1–5)

- `imd_quintiles_for_plot.csv`  
  IMD dataset prepared for plotting and overlays

---

## Figures created

Saved to `outputs/figures/`:

- `fig1_imd_achievement_rate.png` — achievement rate by IMD quintile
- `fig_s2_lep_starts_vs_achievement.png` — LEP starts vs achievement rate scatter
- `fig_lep_top_bottom.png` — top and bottom LEPs by achievement rate
- `fig_imd_polyfit_overlay.png` — IMD points + polynomial fit overlay (descriptive)
- `fig_imd_glm_overlay.png` — IMD points + GLM trend overlay

---

## Model outputs

Saved to `outputs/models/`:

### LEP modelling

- `lep_model_results.csv`
- `lep_model_results_with_glm.csv` — includes GLM comparison
- `lep_london_robustness.csv`
- `lep_london_robustness_check.csv` — influence/sensitivity check excluding London

### IMD modelling

- `imd_polyfit_model_comparison.csv`
- `imd_glm_trend_results.csv`
- `imd_glm_overlay_results.csv`
- `imd_glm_odds_ratios.csv`

---

## Scripts and what they do

- `cleaning_geo.py` — cleans LEP geography data and produces `geo_lep_clean*.csv`
- `cleaning_imd.py` — cleans IMD dataset and produces `imd_clean.csv` / plot-ready files
- `figure1.py` — generates IMD figure(s)
- `figure2.py` — generates LEP scatter figure(s)
- `top_bottom_lep.py` — generates top/bottom LEP ranking figure
- `model.py` — runs LEP models (baseline/OLS/WLS/GLM comparisons) and exports results
- `london.py` — robustness checks for influential observations (London)
- `chi_squared_fit.py` — additional statistical check related to distribution/fit (supporting analysis)
- `test.py` — unit tests / data quality checks (pytest)

---
