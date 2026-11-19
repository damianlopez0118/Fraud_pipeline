"""Refresh aggregate-only Socure artifacts from verified Unity Catalog schemas."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .socure import MonitoringPolicy, summarize_socure

SCHEMA = "xm_risk_dev.xm_risk"
TABLES = {"gini": f"{SCHEMA}.gini_results", "ks": f"{SCHEMA}.ks",
          "curves": f"{SCHEMA}.lorenz_curve", "orders": f"{SCHEMA}.order_current_vw_all_var"}
EXPECTED_SCORES = ("emailrisk_score", "fraud_score", "phonerisk_score")


def query_frame(client, warehouse_id: str, statement: str) -> pd.DataFrame:
    response = client.statement_execution.execute_statement(
        warehouse_id=warehouse_id, statement=statement, wait_timeout="50s",
    )
    state = response.status.state.value if response.status and response.status.state else "UNKNOWN"
    if state != "SUCCEEDED":
        if state in {"PENDING", "RUNNING"}:
            client.statement_execution.cancel_execution(response.statement_id)
        raise RuntimeError(f"Read-only monitoring query did not finish: {state}")
    manifest = response.manifest
    if not manifest or manifest.truncated or manifest.total_row_count > 10000:
        raise RuntimeError("Monitoring query exceeded the bounded aggregate result size")
    rows = list(response.result.data_array or []) if response.result else []
    for chunk in (manifest.chunks or [])[1:]:
        result = client.statement_execution.get_statement_result_chunk_n(
            response.statement_id, chunk.chunk_index
        )
        rows.extend(result.data_array or [])
    return pd.DataFrame(rows, columns=[column.name for column in manifest.schema.columns])


def prepare_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = frame.copy()
    for column in ("minimum", "maximum", "source_rows", "non_null_rows"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    authoritative = frame["source_rows"].eq(1) & frame["non_null_rows"].eq(1)
    frame["value"] = frame["minimum"].where(authoritative)
    return frame.rename(columns={"minimum": f"{metric}_min", "maximum": f"{metric}_max",
                                 "source_rows": f"{metric}_source_rows"})


def validate_curves(frame: pd.DataFrame) -> list[dict]:
    frame = frame.copy()
    for column in ("score", "orders_fraction", "fraud_fraction"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in ("orders_fraction", "fraud_fraction"):
        if frame[column].isna().any() or not frame[column].between(0, 1).all():
            raise ValueError(f"Curve {column} must be a non-null fraction between 0 and 1")
    if frame["score"].isna().any():
        raise ValueError("Curve score cannot be null")
    if frame.duplicated(["month", "score_type", "score"]).any():
        raise ValueError("Ambiguous capture curves: multiple rows per month/score/threshold")
    for _, curve in frame.groupby(["month", "score_type"]):
        curve = curve.sort_values("orders_fraction")
        if (curve["fraud_fraction"].diff().dropna() < -1e-8).any():
            raise ValueError("Capture curve is not cumulative; verify source cohort/run")
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def refresh_socure(
    client, warehouse_id: str, out_dir: Path, *, as_of: str | None = None,
    months: int = 12, label_lag_months: int = 1, policy: MonitoringPolicy | None = None,
) -> dict:
    if not 1 <= months <= 60 or not 1 <= label_lag_months <= 24:
        raise ValueError("Use 1-60 history months and 1-24 months of label lag")
    today = pd.Timestamp(as_of or pd.Timestamp.now().date()).normalize()
    end = today.to_period("M") - label_lag_months
    start = end - months + 1
    predicate = f"fiscal_month BETWEEN '{start}' AND '{end}'"
    frames = {}
    for metric, column in (("gini", "Gini"), ("ks", "KS")):
        result = query_frame(client, warehouse_id, f"""
            SELECT fiscal_month AS month, fraud_var AS score_type,
                   MIN({column}) AS minimum, MAX({column}) AS maximum,
                   COUNT(*) AS source_rows, COUNT({column}) AS non_null_rows
            FROM {TABLES[metric]} WHERE {predicate}
            GROUP BY fiscal_month, fraud_var
            ORDER BY fiscal_month, fraud_var LIMIT 10001
        """)
        frames[metric] = prepare_metric(result, metric)
        for score_type in EXPECTED_SCORES:
            present = (frames[metric]["month"].eq(str(end))
                       & frames[metric]["score_type"].eq(score_type))
            if not present.any():
                missing = pd.DataFrame([{
                    "month": str(end), "score_type": score_type, "value": None,
                    f"{metric}_min": None, f"{metric}_max": None,
                    f"{metric}_source_rows": 0,
                }])
                frames[metric] = pd.concat([frames[metric], missing], ignore_index=True)
    report = summarize_socure(frames["gini"], frames["ks"], policy=policy, as_of=str(today.date()))
    report["sources"] = TABLES
    report["window"] = {"start": str(start), "end": str(end), "label_lag_months": label_lag_months}
    report["refreshed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    report["limitations"] = [
        "Reference thresholds are monitoring heuristics, not model acceptance criteria.",
        "Duplicate evaluations have no run/cohort key: ranges are not authoritative estimates.",
        "Label maturity, sample sizes, confidence intervals and cohort alignment are unverified.",
        "A calendar label lag does not establish that fraud outcomes are fully observed.",
        ("Order view has no Socure score fields; score coverage, PSI, PR-AUC and precision/recall "
         "require a governed score join and a confirmed fraud-label definition."),
        "Capture points retain source accumulation order; high-risk-first direction is unverified.",
    ]
    curves = query_frame(client, warehouse_id, f"""
        SELECT fiscal_month AS month, fraud_var AS score_type, score,
               cum_orders_perc AS orders_fraction, cum_frauds_perc AS fraud_fraction
        FROM {TABLES['curves']} WHERE fiscal_month = '{end}'
        ORDER BY fraud_var, cum_orders_perc LIMIT 10001
    """)
    try:
        report["curves"] = validate_curves(curves)
    except ValueError as error:
        report["curves"] = []
        report["limitations"].append(str(error))
    if report["monthly"] and max(row["month"] for row in report["monthly"]) < str(end):
        report["limitations"].append(f"Expected evaluation month {end} is unavailable.")
        for score in report["latest"]:
            score["status"] = "alert"
            score["reasons"].append(f"Expected evaluation month {end} is unavailable")
    report["decision"] = (
        "not_assessable" if not report["latest"] or any(
            "Missing Gini or KS evaluation" in score["reasons"] for score in report["latest"]
        ) else "review_required" if any(
            score["status"] != "within_reference" for score in report["latest"]
        ) else "within_reference_not_validated"
    )
    from .report import render_socure_dashboard

    html = render_socure_dashboard(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "socure_monitoring.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    (out_dir / "socure.html").write_text(html, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--label-lag-months", type=int, default=1)
    parser.add_argument("--policy", type=Path, help="JSON overrides for MonitoringPolicy fields")
    args = parser.parse_args()
    from databricks.sdk import WorkspaceClient

    policy = MonitoringPolicy(**json.loads(args.policy.read_text())) if args.policy else None
    report = refresh_socure(
        WorkspaceClient(profile=args.profile), args.warehouse_id, args.out,
        as_of=args.as_of, months=args.months, label_lag_months=args.label_lag_months, policy=policy,
    )
    print(json.dumps({"decision": report["decision"], "window": report["window"],
                      "latest": report["latest"]}, indent=2))


if __name__ == "__main__":
    main()