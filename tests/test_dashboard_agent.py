import json

import pandas as pd

from fraud_pipeline.agent import create_report_tools
from fraud_pipeline.dashboard_page import render_dashboard
from fraud_pipeline.dashboard_server import ChatRequest, ChatTurn, chat


def test_report_tools_read_pipeline_artifacts(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"pr_auc": 0.58}), encoding="utf-8")
    pd.DataFrame(
        [
            {"name": "schema", "passed": True, "detail": "ok"},
            {"name": "unique:order_id", "passed": False, "detail": "duplicates=2"},
        ]
    ).to_csv(tmp_path / "quality_report.csv", index=False)
    pd.DataFrame(
        [{"feature": "order_amount_log", "psi": 0.31, "status": "major"}]
    ).to_csv(tmp_path / "drift_report.csv", index=False)
    tools = {report_tool.name: report_tool for report_tool in create_report_tools(tmp_path)}

    assert '"pr_auc": 0.58' in tools["read_model_metrics"].invoke({})
    quality = json.loads(tools["read_quality_checks"].invoke({}))
    assert quality["passed_count"] == 1
    assert quality["failures"][0]["name"] == "unique:order_id"
    drift = json.loads(tools["read_feature_drift"].invoke({}))
    assert drift[0]["status"] == "major"


def test_report_tools_explain_missing_reports(tmp_path):
    tools = {report_tool.name: report_tool for report_tool in create_report_tools(tmp_path)}

    assert tools["read_model_metrics"].invoke({}) == "metrics.json is unavailable."
    assert tools["read_quality_checks"].invoke({}) == "quality_report.csv is unavailable."
    assert tools["read_feature_drift"].invoke({}) == "drift_report.csv is unavailable."


def test_dashboard_shell_has_local_assistant_and_plot():
    html = render_dashboard('<div id="plotly-graph"></div>')

    assert '<div id="plotly-graph"></div>' in html
    assert 'fetch("/api/chat"' in html
    assert "Fraud assistant" in html


def test_greeting_gets_dashboard_specific_reply_without_model_call():
    request = ChatRequest(messages=[ChatTurn(role="user", content="  HI!  ")])

    assert chat(request) == {
        "answer": (
            "Hello! This is Gianna your fraud expert. How can I assist you today with the "
            "fraud-pipeline report?"
        )
    }