import pandas as pd

def test_neg_values():
    df = pd.read_csv("data/processed/geo_lep_clean.csv")
    assert (df["starts"] >= 0).all()
    assert (df["achievements"] >= 0).all()

def test_rates_between_zero_and_one():
    df = pd.read_csv("data/processed/geo_lep_clean.csv")
    assert df["achievement_rate"].between(0, 1).all()
    assert df["dropoff_rate"].between(0, 1).all()

def test_unique_lep_codes():
    df = pd.read_csv("data/processed/geo_lep_clean.csv")
    assert df["local_enterprise_partnership_code"].is_unique