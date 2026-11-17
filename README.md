# Fraud Pipeline: End-to-End Data and ML Portfolio Project

A production-style pipeline for order-fraud detection: **ingest → data-quality gate → features → XGBoost → MLflow tracking → drift monitoring → dashboard**, with tests, CI, a Databricks Job bundle, and a PySpark example.

> Uses **synthetic data only**. No company data, credentials or schemas are included.

## What this demonstrates

| Job requirement | Where |
|---|---|
| SQL/Python data pipelines | [pipeline.py](src/fraud_pipeline/pipeline.py): stages are plain functions and idempotent |
| Spark / Databricks / cloud | [spark_job.py](src/fraud_pipeline/spark_job.py): partitioned daily aggregates (needs Java; not run locally) |
| Data quality, anomaly detection | [quality.py](src/fraud_pipeline/quality.py): schema, null, unique, range and allowed-value checks; z-score volume anomalies; the pipeline fails fast |
| ML workflow: train, evaluate, monitor | [train.py](src/fraud_pipeline/train.py) (out-of-time split, class weighting, PR-AUC, MLflow), [drift.py](src/fraud_pipeline/drift.py) (PSI) |
| Orchestration | [resources/fraud_pipeline.job.yml](resources/fraud_pipeline.job.yml): scheduled Databricks Job with retries |
| Testing, CI/CD | [tests/](tests/), [.github/workflows/ci.yml](.github/workflows/ci.yml): ruff and pytest on every push |
| BI / EDA | [report.py](src/fraud_pipeline/report.py): Plotly dashboard (volume, fraud by country, PR curve, drift) |

## Run it

```powershell
pip install -e ".[dev,agent]"
pytest -q
fraud-pipeline --out artifacts      # writes quality_report.csv, drift_report.csv, dashboard.html, metrics.json
```

## Local dashboard assistant

The dashboard has a read-only LangGraph assistant for run metrics, data-quality results, and feature drift. It uses Ollama locally; the default model is `qwen2.5:3b`.

```powershell
ollama pull qwen2.5:3b
fraud-dashboard --artifacts artifacts
```

Open `http://127.0.0.1:8000`. If Ollama is not running, start it with `ollama serve` in a separate terminal. Set `OLLAMA_MODEL` or `OLLAMA_BASE_URL` to use another local Ollama model or endpoint. The browser connects only to the local dashboard API; model credentials are not placed in the page.

For a later Databricks migration, the dashboard UI and report-reading tools can remain; the model adapter and artifact location can be switched to Databricks services.

## Run on Databricks

Requires Databricks CLI 0.292.0 or newer and valid workspace authentication. Create a writable Unity Catalog Volume, then set `output_path` to its path.

```powershell
databricks bundle validate --target dev --var="output_path=/Volumes/your_catalog/your_schema/your_volume/fraud_pipeline"
databricks bundle deploy --target dev --var="output_path=/Volumes/your_catalog/your_schema/your_volume/fraud_pipeline"
databricks bundle run --target dev --var="output_path=/Volumes/your_catalog/your_schema/your_volume/fraud_pipeline" fraud_pipeline
```

The daily schedule is paused by default to avoid starting recurring compute before you are ready. Unpause it in the Databricks Jobs UI after confirming the volume path and workspace permissions. Job runs use Databricks MLflow tracking; local CLI runs continue to use SQLite.

## Results (synthetic, out-of-time test set)

ROC-AUC ≈ 0.95, PR-AUC ≈ 0.58, recall ≈ 0.41 at ≥80% precision (about 3% fraud rate). PR-AUC and recall at fixed precision are the headline metrics because ROC-AUC looks flattering on imbalanced data. The simulated production batch shifts order amounts, and PSI correctly flags `order_amount_log` as major drift.

## Design decisions (good interview talking points)

- **Out-of-time split** prevents leakage that a random split would hide.
- **Validate before train.** Bad data fails the run instead of silently producing a bad model.
- **Stable feature columns** between training and scoring (fixed category list).
- **Monitoring is a first-class stage**, not an afterthought.

## Known limits and next steps

- The PySpark aggregation example is separate from the fraud CLI. The Databricks Job bundle uses serverless compute and requires a configured workspace and writable Unity Catalog Volume.
- Next: Delta Lake output, a model registry stage with promotion gates, and a RAG/agent extension (see [docs/interview_prep.md](docs/interview_prep.md)).
