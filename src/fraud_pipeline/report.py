"""Self-contained HTML dashboard (Plotly) for EDA, model quality, and monitoring."""

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve

from .dashboard_page import render_dashboard


def render_socure_dashboard(report: dict) -> str:
    monthly = pd.DataFrame(report["monthly"])
    colors = {"emailrisk_score": "#9747df", "fraud_score": "#1265db",
              "phonerisk_score": "#d87500"}
    labels = {"emailrisk_score": "Email Risk Score", "fraud_score": "Fraud Risk Score",
              "phonerisk_score": "Phone Risk Score"}
    policy = report["policy"]
    charts = []
    for metric, title in (("gini", "Gini by Month"), ("ks", "KS by Month")):
        figure = go.Figure()
        if not monthly.empty:
            for index, (score_type, history) in enumerate(monthly.groupby("score_type")):
                color = colors.get(score_type, ["#1265db", "#9747df", "#d87500"][index % 3])
                name = labels.get(score_type, score_type)
                red, green, blue = (int(color[offset:offset + 2], 16) for offset in (1, 3, 5))
                if f"{metric}_min" in history:
                    for bound in ("min", "max"):
                        figure.add_trace(go.Scatter(
                            x=history["month"], y=history[f"{metric}_{bound}"],
                            name=f"{name} observed range" if bound == "min" else name,
                            legendgroup=score_type, showlegend=bound == "min",
                            mode="lines+markers", connectgaps=False,
                            line={"color": color, "width": 1, "dash": "dot"},
                            fill="tonexty" if bound == "max" else None,
                            fillcolor=f"rgba({red},{green},{blue},0.12)",
                            hovertemplate=f"%{{x}}<br>{bound}: %{{y:.4f}}<extra>{name}</extra>",
                        ))
                figure.add_trace(go.Scatter(
                    x=history["month"], y=history[metric], name=name,
                    legendgroup=score_type, showlegend=f"{metric}_min" not in history,
                    mode="lines+markers", connectgaps=False, line={"color": color, "width": 2},
                ))
        for level, color in (("weak", "#c44536"), ("good", "#218454")):
            threshold = policy[f"{metric}_{level}"]
            figure.add_hline(y=threshold, line_color=color, line_dash="dash",
                             annotation_text=f"{level.title()} reference {threshold:.2f}",
                             annotation_position="top right")
        figure.update_layout(
            title=title, template="plotly_white", height=390,
            margin={"l": 48, "r": 20, "t": 50, "b": 100},
            legend={"orientation": "h", "y": -0.27, "font": {"size": 10}},
            xaxis={"type": "category", "tickangle": -45}, yaxis={"title": metric.upper()},
        )
        charts.append(figure.to_html(full_html=False, include_plotlyjs="cdn" if not charts else False,
                                     config={"responsive": True, "displaylogo": False}))
    capture = go.Figure()
    curves = pd.DataFrame(report.get("curves", []))
    if not curves.empty:
        for score_type, curve in curves.groupby("score_type"):
            capture.add_trace(go.Scatter(
                x=curve["orders_fraction"], y=curve["fraud_fraction"],
                mode="markers", name=labels.get(score_type, score_type),
                marker={"color": colors.get(score_type, "#1265db"), "size": 7},
                customdata=curve["score"],
                hovertemplate="Orders %{x:.1%}<br>Fraud %{y:.1%}<br>Score %{customdata}<extra>%{fullData.name}</extra>",
            ))
    else:
        capture.add_annotation(text="No validated source capture points for this month", showarrow=False)
    capture.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line={"color": "#8c979d", "dash": "dash"})
    capture.update_layout(
        title=f"Cumulative Fraud / Orders - {report['window']['end']}",
        template="plotly_white", height=390,
        margin={"l": 55, "r": 20, "t": 50, "b": 70},
        xaxis={"title": "Cumulative orders (source order)", "tickformat": ".0%", "range": [0, 1]},
        yaxis={"title": "Cumulative fraud", "tickformat": ".0%", "range": [0, 1]},
        legend={"orientation": "h", "y": -0.24},
    )
    capture_html = capture.to_html(full_html=False, include_plotlyjs=False,
                                    config={"responsive": True, "displaylogo": False})

    def metric_value(row: dict, metric: str) -> str:
        value = row.get(metric)
        if value is not None:
            return f"{value:.3f}"
        lower, upper = row.get(f"{metric}_min"), row.get(f"{metric}_max")
        if lower is not None and upper is not None:
            return f"{lower:.3f} to {upper:.3f} (range)"
        return "Unavailable"

    def change_value(row: dict, metric: str) -> str:
        value = row.get(f"{metric}_change")
        return "Unavailable" if value is None else f"{value:+.3f}"

    cards = []
    for score in report["latest"]:
        history = [row for row in report["monthly"] if row["score_type"] == score["score_type"]]
        row = history[-1]
        reasons = "; ".join(score["reasons"]) or "Within reference thresholds; validation still required"
        auc = metric_value(row, "roc_auc")
        if auc == "Unavailable" and row.get("gini_min") is not None:
            auc = f"{(row['gini_min'] + 1) / 2:.3f} to {(row['gini_max'] + 1) / 2:.3f} (range)"
        cards.append(f"""<article class="score-card">
          <h3>{escape(labels.get(score['score_type'], score['score_type']))}</h3>
          <p class="score-status {escape(score['status'])}">{escape(score['status'].replace('_', ' ').upper())}</p>
          <dl><dt>Gini</dt><dd>{metric_value(row, 'gini')}</dd>
          <dt>ROC-AUC (from Gini)</dt><dd>{auc}</dd>
          <dt>KS</dt><dd>{metric_value(row, 'ks')}</dd>
          <dt>Gini / KS monthly change</dt><dd>{change_value(row, 'gini')} / {change_value(row, 'ks')}</dd>
          <dt>Source rows: Gini / KS</dt><dd>{row.get('gini_source_rows') or 0:g} / {row.get('ks_source_rows') or 0:g}</dd></dl>
          <p class="score-reasons">{escape(reasons)}</p></article>""")
    limitations = "".join(f"<li>{escape(text)}</li>" for text in report["limitations"])
    sources = "".join(f"<li><code>{escape(table)}</code></li>" for table in report["sources"].values())
    content = f"""<style>
      .socure-header {{ border-bottom: 1px solid var(--line); padding: 8px 0 18px; }}
      .socure-header h1 {{ font-size: 28px; }}
      .snapshot {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
      .decision {{ color: #a33225; font-weight: 700; }}
      .score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr)); gap: 12px; margin: 20px 0; }}
      .score-card {{ background: white; border: 1px solid var(--line); border-radius: 6px; padding: 16px; min-width: 0; }}
      .score-card h3 {{ margin: 0; font-size: 16px; }}
      .score-status {{ font-size: 11px; font-weight: 700; color: #985800; }}
      .score-status.alert {{ color: #b43829; }}
      .score-status.within_reference {{ color: #147649; }}
      .score-card dl {{ font-size: 12px; margin-bottom: 0; }}
      .score-card dt {{ color: var(--muted); margin-top: 10px; }}
      .score-card dd {{ margin: 3px 0; overflow-wrap: anywhere; font-weight: 600; }}
      .score-reasons {{ font-size: 12px; line-height: 1.5; color: #a33225; overflow-wrap: anywhere; }}
      .socure-charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
      .socure-chart {{ min-width: 0; }}
      .source-notes {{ border-top: 1px solid var(--line); margin-top: 20px; font-size: 12px; line-height: 1.65; overflow-wrap: anywhere; }}
      .source-notes h2 {{ font-size: 16px; }}
      @media (max-width: 1250px) {{ .socure-charts {{ grid-template-columns: minmax(0, 1fr); }} }}
      @media (max-width: 820px) {{ .socure .plotly-graph-div {{ width: 100% !important; min-width: 0 !important; }} }}
    </style><section class="socure">
      <header class="socure-header"><p class="eyebrow">MODEL MONITORING / UNITY CATALOG</p>
      <h1>Socure Score Performance</h1>
      <p class="decision">{escape(report['decision'].replace('_', ' ').upper())}</p>
      <p class="snapshot">Evaluation window {escape(report['window']['start'])} to {escape(report['window']['end'])}
      | As of {escape(report['as_of'])} | Refreshed {escape(report['refreshed_at'])}</p></header>
      <div class="score-grid">{''.join(cards) or '<p>No evaluations available in this window.</p>'}</div>
      <div class="socure-charts"><div class="socure-chart">{charts[0]}</div><div class="socure-chart">{charts[1]}</div></div>
      <div class="socure-chart">{capture_html}</div>
      <section class="source-notes"><h2>Assessment Constraints</h2><ul>{limitations}</ul>
      <h2>Governed Sources</h2><ul>{sources}</ul></section></section>"""
    return render_dashboard(content)


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
    fig.update_layout(
        title=f"Fraud pipeline dashboard - {title}", height=800, autosize=True, showlegend=False
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_html = fig.to_html(
        full_html=False, include_plotlyjs="cdn", config={"responsive": True}
    )
    out_path.write_text(render_dashboard(plot_html), encoding="utf-8")
    return out_path
