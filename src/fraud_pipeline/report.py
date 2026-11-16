"""Self-contained HTML dashboard (Plotly) for EDA, model quality, and monitoring."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve


def build_dashboard(
    orders: pd.DataFrame,
    y_test,
    scores,
    drift: pd.DataFrame,
    daily_volume: pd.DataFrame,
    metrics: dict[str, float],
    out_path: Path,
) -> Path:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Daily order volume (anomalies in red)",
            "Fraud rate by country",
            "Precision-recall (out-of-time test)",
            "Feature drift (PSI)",
        ),
    )
    fig.add_trace(go.Scatter(x=daily_volume.index, y=daily_volume["orders"], name="orders"), 1, 1)
    anomalies = daily_volume[daily_volume["anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies.index,
            y=anomalies["orders"],
            mode="markers",
            marker={"color": "red", "size": 9},
            name="anomaly",
        ),
        1,
        1,
    )

    by_country = orders.groupby("country_cd")["is_fraud"].mean().sort_values(ascending=False)
    fig.add_trace(go.Bar(x=by_country.index, y=by_country.values, name="fraud rate"), 1, 2)

    precision, recall, _ = precision_recall_curve(y_test, scores)
    fig.add_trace(go.Scatter(x=recall, y=precision, name="PR curve"), 2, 1)

    fig.add_trace(go.Bar(x=drift["feature"], y=drift["psi"], name="PSI"), 2, 2)
    fig.add_hline(y=0.25, line_dash="dash", line_color="red", row=2, col=2)

    title = " | ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    fig.update_layout(title=f"Fraud pipeline dashboard - {title}", height=800, showlegend=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path
