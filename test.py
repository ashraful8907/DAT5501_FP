"""
Unit tests for DAT5501 FP - Apprenticeship achievement analysis

Performance decision:
- Use pytest fixtures to load each CSV once per test session.
  This avoids repeated I/O and speeds up CI runs.
"""

from pathlib import Path
from typing import List

import pandas as pd
import pytest


# Paths
GEO_PATH = Path("data/processed/geo_lep_clean.csv")
IMD_PATH = Path("data/processed/imd_clean.csv")
LONDON_OUT_PATH = Path("outputs/models/lep_london_robustness_check.csv")
MODEL_OUT_PATH = Path("outputs/models/lep_model_results_with_glm.csv")


# Fixtures
@pytest.fixture(scope="session")
def geo_df() -> pd.DataFrame:
    assert GEO_PATH.exists(), f"Missing {GEO_PATH}. Run cleaning_geo.py first."
    return pd.read_csv(GEO_PATH)


@pytest.fixture(scope="session")
def imd_df() -> pd.DataFrame:
    assert IMD_PATH.exists(), f"Missing {IMD_PATH}. Run cleaning_imd.py first."
    return pd.read_csv(IMD_PATH)


# Helper assertions
def assert_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"{name} missing columns: {missing}"


def assert_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Convert a column to numeric and fail clearly if conversion isn't possible."""
    try:
        return pd.to_numeric(df[col], errors="raise")
    except Exception as e:
        raise AssertionError(f"Column '{col}' should be numeric but isn't: {e}") from e


# GEO (LEP) tests
def test_geo_schema_and_rows(geo_df: pd.DataFrame):
    """Schema checks prevent silent downstream errors in figures/models."""
    assert len(geo_df) > 0, "geo_lep_clean.csv is empty"

    required = [
        "local_enterprise_partnership_code",
        "starts",
        "achievements",
        "achievement_rate",
        "dropoff_rate",
    ]
    assert_required_columns(geo_df, required, "geo_lep_clean.csv")


def test_geo_counts_valid(geo_df: pd.DataFrame):
    """Counts must be non-negative and achievements cannot exceed starts."""
    starts = assert_numeric_series(geo_df, "starts")
    ach = assert_numeric_series(geo_df, "achievements")

    assert (starts >= 0).all(), "starts has negative values"
    assert (ach >= 0).all(), "achievements has negative values"
    assert (ach <= starts).all(), "Found achievements > starts (logical error)"


def test_geo_rates_valid_probability(geo_df: pd.DataFrame):
    """
    Achievement/dropoff rates should be interpretable as probabilities.
    This directly supports the report insight that compares achievement rates across regions.
    """
    for col in ["achievement_rate", "dropoff_rate"]:
        rates = assert_numeric_series(geo_df, col)
        assert rates.between(0, 1).all(), f"{col} has values outside [0, 1]"


def test_geo_achievement_rate_correct(geo_df: pd.DataFrame):
    """
    Metric integrity test: verifies achievement_rate == achievements/starts when starts > 0.
    This is the key KPI used in the report, so correctness is essential.
    """
    starts = assert_numeric_series(geo_df, "starts")
    ach = assert_numeric_series(geo_df, "achievements")
    rate = assert_numeric_series(geo_df, "achievement_rate")

    mask = starts > 0
    expected = ach[mask] / starts[mask]

    # Use a tolerance in case rates are rounded in cleaning scripts.
    assert (rate[mask] - expected).abs().max() < 1e-6, (
        "achievement_rate does not match achievements/starts"
    )


def test_geo_lep_codes_unique(geo_df: pd.DataFrame):
    """Ensures each LEP appears once (prevents double counting in rankings/models)."""
    codes = geo_df["local_enterprise_partnership_code"]
    assert codes.notna().all(), "LEP codes contain missing values"
    assert codes.is_unique, "LEP codes are not unique"


# IMD tests
def test_imd_schema_and_quintiles(imd_df: pd.DataFrame):
    """IMD dataset must represent exactly 5 deprivation quintiles."""
    required = ["imd_quintile", "starts", "achievements", "achievement_rate", "dropoff_rate"]
    assert_required_columns(imd_df, required, "imd_clean.csv")

    assert len(imd_df) == 5, f"Expected 5 rows (quintiles), got {len(imd_df)}"

    quintiles = assert_numeric_series(imd_df, "imd_quintile").tolist()
    assert sorted(quintiles) == [1, 2, 3, 4, 5], f"Unexpected quintiles: {quintiles}"
    assert len(set(quintiles)) == 5, "Duplicate quintiles found"


def test_imd_counts_and_rates_valid(imd_df: pd.DataFrame):
    """Same integrity rules as LEP dataset: valid counts and probability rates."""
    starts = assert_numeric_series(imd_df, "starts")
    ach = assert_numeric_series(imd_df, "achievements")

    assert (starts >= 0).all(), "IMD starts has negative values"
    assert (ach >= 0).all(), "IMD achievements has negative values"
    assert (ach <= starts).all(), "IMD has achievements > starts"

    for col in ["achievement_rate", "dropoff_rate"]:
        rates = assert_numeric_series(imd_df, col)
        assert rates.between(0, 1).all(), f"IMD {col} has values outside [0, 1]"


def test_imd_achievement_rate_correct(imd_df: pd.DataFrame):
    """IMD achievement_rate is also used in Figure 1; verify calculation is correct."""
    starts = assert_numeric_series(imd_df, "starts")
    ach = assert_numeric_series(imd_df, "achievements")
    rate = assert_numeric_series(imd_df, "achievement_rate")

    mask = starts > 0
    expected = ach[mask] / starts[mask]

    assert (rate[mask] - expected).abs().max() < 1e-6, (
        "IMD achievement_rate does not match achievements/starts"
    )


# Reproducibility tests
def test_model_outputs_exist():
    """
    Ensures modelling steps ran and produced outputs used in the report appendix.
    These are lightweight file-existence checks that help CI detect broken pipelines.
    """
    assert MODEL_OUT_PATH.exists(), f"Missing {MODEL_OUT_PATH}. Run model.py"
    assert LONDON_OUT_PATH.exists(), f"Missing {LONDON_OUT_PATH}. Run london.py"


def test_model_outputs_have_rows_if_present():
    """Lightweight sanity checks that model output tables are not empty."""
    if MODEL_OUT_PATH.exists():
        df = pd.read_csv(MODEL_OUT_PATH)
        assert len(df) > 0, "lep_model_results_with_glm.csv is empty"

    if LONDON_OUT_PATH.exists():
        df = pd.read_csv(LONDON_OUT_PATH)
        assert len(df) > 0, "lep_london_robustness_check.csv is empty"
