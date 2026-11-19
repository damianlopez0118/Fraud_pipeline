import json

import pandas as pd
import pytest

from fraud_pipeline.socure import MonitoringPolicy, summarize_socure
from fraud_pipeline.socure_refresh import prepare_metric, refresh_socure, validate_curves


def evaluations(values, score_type="Fraud Risk Score"):
    return pd.DataFrame(
        [{"month": month, "score_type": score_type, "value": value}
         for month, value in values]
    )


def test_socure_alerts_and_derived_auc():
    report = summarize_socure(
        evaluations([("2026-09", 0.42), ("2026-10", 0.18)]),
        evaluations([("2026-09", 0.41), ("2026-10", 0.20)]),
        as_of="2026-11-19",
    )
    assert report["latest"][0]["status"] == "alert"
    assert len(report["latest"][0]["reasons"]) == 4
    assert report["monthly"][-1]["roc_auc"] == pytest.approx(0.59)
    json.dumps(report, allow_nan=False)


def test_missing_month_is_a_gap_not_an_improvement():
    report = summarize_socure(
        evaluations([("2026-08", 0.45), ("2026-10", 0.46)]),
        evaluations([("2026-08", 0.41)]), as_of="2026-11-19",
    )
    assert report["monthly"][1]["gini"] is None
    assert report["monthly"][-1]["gini_change"] is None
    assert "Missing Gini or KS evaluation" in report["latest"][0]["reasons"]
    assert report["latest"][0]["status"] == "alert"


def test_stale_results_cannot_look_healthy():
    report = summarize_socure(
        evaluations([("2026-01", 0.6)]), evaluations([("2026-01", 0.5)]),
        as_of="2026-11-19",
    )
    assert report["latest"][0]["status"] == "alert"


@pytest.mark.parametrize("value", [-0.1, 20, float("inf")])
def test_rejects_invalid_ks_scale(value):
    with pytest.raises(ValueError, match="convert percentages"):
        summarize_socure(evaluations([("2026-10", 0.4)]),
                         evaluations([("2026-10", value)]), as_of="2026-11-19")


def test_duplicate_cohorts_are_not_silently_averaged():
    with pytest.raises(ValueError, match="population/version"):
        summarize_socure(evaluations([("2026-10", 0.4), ("2026-10", 0.6)]),
                         evaluations([("2026-10", 0.4)]), as_of="2026-11-19")


def test_ambiguous_live_evaluations_preserve_range_without_point_estimate():
    source = pd.DataFrame([{"month": "2026-10", "score_type": "fraud_score",
                            "minimum": 0.2, "maximum": 0.3,
                            "source_rows": 8, "non_null_rows": 8}])
    report = summarize_socure(prepare_metric(source, "gini"), prepare_metric(source, "ks"),
                              as_of="2026-11-19")
    assert report["monthly"][0]["gini"] is None
    assert report["monthly"][0]["gini_min"] == 0.2
    assert report["monthly"][0]["gini_max"] == 0.3
    assert "Ambiguous GINI: multiple source evaluations" in report["latest"][0]["reasons"]
    json.dumps(report, allow_nan=False)


def test_current_month_is_provisional_even_with_good_metrics():
    report = summarize_socure(evaluations([("2026-11", 0.6)]),
                              evaluations([("2026-11", 0.5)]), as_of="2026-11-19")
    assert report["latest"][0]["status"] == "alert"
    assert report["latest"][0]["age_days"] == 0


def test_invalid_or_ambiguous_curves_are_rejected():
    curve = pd.DataFrame([{"month": "2026-10", "score_type": "fraud_score", "score": 0.5,
                           "orders_fraction": 0.2, "fraud_fraction": 0.4}])
    assert len(validate_curves(curve)) == 1
    with pytest.raises(ValueError, match="Ambiguous"):
        validate_curves(pd.concat([curve, curve]))
    curve["fraud_fraction"] = 40
    with pytest.raises(ValueError, match="fraction"):
        validate_curves(curve)


def test_policy_rejects_reversed_thresholds():
    with pytest.raises(ValueError, match="ordered fractions"):
        MonitoringPolicy(ks_weak=0.4, ks_good=0.3)


def test_refresh_writes_closed_month_snapshot_and_flags_absent_scores(tmp_path, monkeypatch):
    queries = []

    def fake_query(client, warehouse_id, statement):
        queries.append(statement)
        if "cum_orders_perc" in statement:
            return pd.DataFrame([{
                "month": "2026-10", "score_type": "fraud_score", "score": 0.5,
                "orders_fraction": 0.2, "fraud_fraction": 0.4,
            }])
        return pd.DataFrame([{
            "month": "2026-10", "score_type": "fraud_score", "minimum": 0.21,
            "maximum": 0.25, "source_rows": 8, "non_null_rows": 8,
        }])

    monkeypatch.setattr("fraud_pipeline.socure_refresh.query_frame", fake_query)
    report = refresh_socure(None, "warehouse", tmp_path, as_of="2026-11-19")
    assert report["decision"] == "not_assessable"
    assert report["window"]["end"] == "2026-10"
    assert len(report["latest"]) == 3
    assert all(score["status"] == "alert" for score in report["latest"])
    fraud = next(score for score in report["latest"] if score["score_type"] == "fraud_score")
    assert "All observed KS values below weak reference" in fraud["reasons"]
    assert all("2026-11'" not in query for query in queries)
    assert all("order_current_vw_all_var" not in query for query in queries)
    assert json.loads((tmp_path / "socure_monitoring.json").read_text())["decision"] == report["decision"]
    html = (tmp_path / "socure.html").read_text(encoding="utf-8")
    assert "NOT ASSESSABLE" in html
    assert "0.210 to 0.250 (range)" in html
    assert "Socure Score Performance" in html


def test_refresh_blocks_invalid_windows_before_querying(tmp_path):
    with pytest.raises(ValueError, match="history months"):
        refresh_socure(None, "warehouse", tmp_path, months=0)