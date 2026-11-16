"""Synthetic order data generator (no proprietary data is used in this portfolio)."""

import numpy as np
import pandas as pd

HIGH_RISK_COUNTRIES = ["DOM", "GHA", "NGA"]
SAFE_COUNTRIES = ["USA", "CAN", "MEX"]
FRAUD_DOMAINS = ["spamok.com", "mask.me", "proton.me", "oletters.com"]
COMMON_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "comcast.net", "icloud.com"]
ITEM_TYPES = ["BYOD", "DEVICE", "SIM", "ACCESSORY"]


def generate_orders(
    n: int = 20_000,
    fraud_rate: float = 0.03,
    start: str = "2025-01-01",
    days: int = 180,
    seed: int = 42,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Generate orders. `drift` shifts order amounts to simulate data drift."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate

    created = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days * 24 * 3600, n), unit="s")
    amount = np.where(
        is_fraud,
        rng.lognormal(5.5 + drift, 0.8, n),
        rng.lognormal(5.0 + drift, 0.8, n),
    ).round(2)
    country = np.where(
        is_fraud & (rng.random(n) < 0.35),
        rng.choice(HIGH_RISK_COUNTRIES, n),
        rng.choice(SAFE_COUNTRIES + HIGH_RISK_COUNTRIES, n, p=[0.8, 0.07, 0.06, 0.02, 0.03, 0.02]),
    )
    domain = np.where(
        is_fraud & (rng.random(n) < 0.3),
        rng.choice(FRAUD_DOMAINS, n),
        rng.choice(COMMON_DOMAINS + FRAUD_DOMAINS[:1], n, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),
    )
    lines = np.where(is_fraud, rng.integers(1, 7, n), rng.integers(1, 5, n))
    acct_age = np.where(is_fraud, rng.exponential(90, n), rng.exponential(500, n)).round(0)

    return pd.DataFrame(
        {
            "order_id": np.arange(1, n + 1),
            "created_ts": created,
            "order_amount": amount,
            "country_cd": country,
            "email_domain": domain,
            "item_type": rng.choice(ITEM_TYPES, n, p=[0.3, 0.4, 0.2, 0.1]),
            "num_lines": lines,
            "account_age_days": acct_age,
            "is_fraud": is_fraud.astype(int),
        }
    ).sort_values("created_ts", ignore_index=True)
