import json

import pandas as pd

from fraud_pipeline.agent import build_agent, create_report_tools, current_socure_context
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
    assert "unavailable" in tools["read_socure_monitoring"].invoke({})


def test_assistant_reads_socure_snapshot_with_constraints(tmp_path):
    snapshot = {"as_of": "2026-11-19", "decision": "not_assessable",
                "limitations": ["Duplicate evaluations"], "curves": [{"score": 0.1}]}
    (tmp_path / "socure_monitoring.json").write_text(json.dumps(snapshot), encoding="utf-8")
    tools = {report_tool.name: report_tool for report_tool in create_report_tools(tmp_path)}
    result = json.loads(tools["read_socure_monitoring"].invoke({}))
    assert result["decision"] == "not_assessable"
    assert result["limitations"] == ["Duplicate evaluations"]
    assert "curves" not in result


def test_current_context_uses_latest_month_and_rereads_refreshes(tmp_path):
    path = tmp_path / "socure_monitoring.json"
    snapshot = {
        "as_of": "2026-11-19", "refreshed_at": "2026-11-19T10:00:00Z",
        "decision": "not_assessable", "limitations": ["Duplicate evaluations"],
        "monthly": [{"month": "2026-09", "gini": 0.3},
                    {"month": "2026-10", "gini": None, "gini_min": -0.03}],
        "curves": [{"score": 0.1}],
    }
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    result = json.loads(current_socure_context(tmp_path))
    assert result["latest_monthly_metrics"] == [snapshot["monthly"][-1]]
    assert result["limitations"] == ["Duplicate evaluations"]
    assert result["refreshed_at"] == snapshot["refreshed_at"]
    assert "monthly" not in result and "curves" not in result
    snapshot["decision"] = "review_required"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    assert json.loads(current_socure_context(tmp_path))["decision"] == "review_required"


def test_current_context_handles_missing_or_corrupt_snapshot(tmp_path):
    assert "unavailable" in current_socure_context(tmp_path)
    (tmp_path / "socure_monitoring.json").write_text("invalid JSON", encoding="utf-8")
    assert "unreadable" in current_socure_context(tmp_path)


def test_latest_update_chat_answers_from_snapshot_without_model(tmp_path, monkeypatch):
    snapshot = {
        "as_of": "2026-11-19", "refreshed_at": "2026-11-19T10:00:00Z",
        "decision": "not_assessable", "window": {"start": "2025-11", "end": "2026-10"},
        "limitations": ["Duplicate evaluations have no run key", "Invalid curve withheld"],
        "monthly": [{"month": "2026-10", "score_type": "emailrisk_score", "gini": None,
                     "gini_min": -0.03, "gini_max": -0.02, "ks": None,
                     "gini_source_rows": 8, "ks_source_rows": 8}],
        "latest": [{"score_type": "emailrisk_score", "reasons": ["Ambiguous GINI"]}],
    }
    (tmp_path / "socure_monitoring.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setenv("FRAUD_PIPELINE_ARTIFACTS", str(tmp_path))

    def unexpected_model(*args, **kwargs):
        raise AssertionError("Latest-update answers must not depend on Ollama")

    monkeypatch.setattr("fraud_pipeline.dashboard_server.get_agent", unexpected_model)
    for question in ("What was added in the latest Socure update?",
                     "Why is assessment blocked?", "Review the Socure monitoring snapshot"):
        answer = chat(ChatRequest(messages=[ChatTurn(role="user", content=question)]))["answer"]
        assert "2026-10" in answer and "2026-11-19T10:00:00Z" in answer
        assert "not assessable" in answer
        assert "-0.0300 to -0.0200" in answer
        assert "Duplicate evaluations" in answer
        assert "Invalid curve withheld" in answer


def test_latest_update_chat_explains_unavailable_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("FRAUD_PIPELINE_ARTIFACTS", str(tmp_path))
    answer = chat(ChatRequest(messages=[ChatTurn(role="user", content="What is the latest update?")]))
    assert "snapshot unavailable" in answer["answer"]


def test_agent_receives_current_socure_context_without_calling_a_tool(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage

    captured = []

    class FakeModel:
        def __init__(self, **kwargs):
            assert kwargs["num_ctx"] == 4096
            assert kwargs["num_predict"] == 512
            assert kwargs["client_kwargs"]["timeout"] == 60.0

        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            captured.append(messages)
            return AIMessage(content="Assessment is blocked by duplicate evaluations.")

    monkeypatch.setattr("fraud_pipeline.agent.ChatOllama", FakeModel)
    snapshot = {"as_of": "2026-11-19", "decision": "not_assessable",
                "limitations": ["Duplicate evaluations"],
                "monthly": [{"month": "2026-10", "gini": None, "ks": None}]}
    path = tmp_path / "socure_monitoring.json"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    graph = build_agent(tmp_path)
    graph.invoke({"messages": [HumanMessage(content="What was the latest update?")]})
    assert "latest dashboard update added Socure monitoring" in captured[-1][0].content
    assert '"decision": "not_assessable"' in captured[-1][1].content
    assert "2026-10" in captured[-1][1].content
    snapshot["as_of"] = "2026-11-20"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    graph.invoke({"messages": [HumanMessage(content="Has the report been refreshed?")]})
    assert "2026-11-20" in captured[-1][1].content


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