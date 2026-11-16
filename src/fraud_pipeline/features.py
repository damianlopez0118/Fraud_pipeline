"""Feature engineering shared by training and scoring."""

import numpy as np
import pandas as pd

from .data_gen import FRAUD_DOMAINS, HIGH_RISK_COUNTRIES

TARGET = "is_fraud"
NUMERIC = [
    "order_amount_log",
    "num_lines",
    "account_age_days",
    "is_high_risk_country",
    "is_fraud_domain",
    "is_new_account",
    "hour",
    "is_weekend",
]


def build_features(df: pd.DataFrame, item_types: list[str] | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["order_amount_log"] = np.log1p(df["order_amount"])
    out["num_lines"] = df["num_lines"]
    out["account_age_days"] = df["account_age_days"]
    out["is_high_risk_country"] = df["country_cd"].isin(HIGH_RISK_COUNTRIES).astype(int)
    out["is_fraud_domain"] = df["email_domain"].str.lower().isin(FRAUD_DOMAINS).astype(int)
    out["is_new_account"] = (df["account_age_days"] < 30).astype(int)
    out["hour"] = df["created_ts"].dt.hour
    out["is_weekend"] = (df["created_ts"].dt.day_of_week >= 5).astype(int)

    # Fixed category list keeps train/score columns identical
    cats = item_types or sorted(df["item_type"].unique())
    for c in cats:
        out[f"item_type_{c}"] = (df["item_type"] == c).astype(int)
    return out


def time_split(df: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-time split: train on earlier orders, test on later ones (no leakage)."""
    df = df.sort_values("created_ts")
    cut = int(len(df) * (1 - test_fraction))
    return df.iloc[:cut], df.iloc[cut:]
