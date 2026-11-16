import numpy as np
import pandas as pd
import pytest

from fraud_pipeline.data_gen import generate_orders
from fraud_pipeline.drift import psi
from fraud_pipeline.features import build_features, time_split
from fraud_pipeline.pipeline import run
from fraud_pipeline.quality import (
    check_range,
    check_unique,
    detect_volume_anomalies,
    run_order_checks,
)


@pytest.fixture(scope="module")
def orders():
    return generate_orders(n=3_000, seed=1)


def test_generator_is_deterministic():
    a, b = generate_orders(n=500, seed=3), generate_orders(n=500, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_quality_checks_pass_on_clean_data(orders):
    assert all(r.passed for r in run_order_checks(orders))


def test_quality_checks_catch_bad_data(orders):
    bad = orders.copy()
    bad.loc[0, "order_id"] = bad.loc[1, "order_id"]
    bad.loc[2, "order_amount"] = -5
    assert not check_unique(bad, "order_id").passed
    assert not check_range(bad, "order_amount", 0, 1e6).passed


def test_missing_column_fails_schema(orders):
    results = run_order_checks(orders.drop(columns=["country_cd"]))
    assert not results[0].passed


def test_volume_anomaly_detected():
    ts = pd.date_range("2025-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    rows = [t for t in ts for _ in range(100 + int(rng.integers(-3, 4)))]
    rows += [ts[-1]] * 400  # spike on the last day
    daily = detect_volume_anomalies(pd.DataFrame({"created_ts": rows}), "created_ts")
    assert bool(daily["anomaly"].iloc[-1])


def test_time_split_has_no_leakage(orders):
    train, test = time_split(orders)
    assert train["created_ts"].max() <= test["created_ts"].min()


def test_features_have_stable_columns(orders):
    cats = sorted(orders["item_type"].unique())
    full = build_features(orders, cats)
    subset = build_features(orders.head(5), cats)
    assert list(full.columns) == list(subset.columns)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5_000))
    assert psi(base, pd.Series(rng.normal(0, 1, 5_000))) < 0.1
    assert psi(base, pd.Series(rng.normal(2, 1, 5_000))) > 0.25


def test_end_to_end_pipeline(tmp_path):
    metrics = run(tmp_path, n=6_000, seed=5)
    assert metrics["roc_auc"] > 0.8
    for name in ("quality_report.csv", "drift_report.csv", "dashboard.html", "metrics.json"):
        assert (tmp_path / name).exists()
