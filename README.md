# Fraud Pipeline: End-to-End Data and ML Portfolio Project

A production-style pipeline for order-fraud detection: **ingest → data-quality gate → features → XGBoost → MLflow tracking → drift monitoring → dashboard**, with tests, CI, an Airflow DAG and a PySpark job.

> Uses **synthetic data only**. No company data, credentials or schemas are included.

## What this demonstrates

| Job requirement | Where |
|---|---|
| SQL/Python data pipelines | [pipeline.py](src/fraud_pipeline/pipeline.py): stages are plain functions and idempotent |
| Spark / Databricks / cloud | [spark_job.py](src/fraud_pipeline/spark_job.py): partitioned daily aggregates (needs Java; not run locally) |
| Data quality, anomaly detection | [quality.py](src/fraud_pipeline/quality.py): schema, null, unique, range and allowed-value checks; z-score volume anomalies; the pipeline fails fast |
| ML workflow: train, evaluate, monitor | [train.py](src/fraud_pipeline/train.py) (out-of-time split, class weighting, PR-AUC, MLflow), [drift.py](src/fraud_pipeline/drift.py) (PSI) |
| Orchestration | [dags/fraud_pipeline_dag.py](dags/fraud_pipeline_dag.py): Airflow DAG with retries |
| Testing, CI/CD | [tests/](tests/), [.github/workflows/ci.yml](.github/workflows/ci.yml): ruff and pytest on every push |
| BI / EDA | [report.py](src/fraud_pipeline/report.py): Plotly dashboard (volume, fraud by country, PR curve, drift) |

## Run it

```powershell
pip install -e ".[dev]"
pytest -q
fraud-pipeline --out artifacts      # writes quality_report.csv, drift_report.csv, dashboard.html, metrics.json
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db
```

## Results (synthetic, out-of-time test set)

ROC-AUC ≈ 0.95, PR-AUC ≈ 0.58, recall ≈ 0.41 at ≥80% precision (about 3% fraud rate). PR-AUC and recall at fixed precision are the headline metrics because ROC-AUC looks flattering on imbalanced data. The simulated production batch shifts order amounts, and PSI correctly flags `order_amount_log` as major drift.

## Design decisions (good interview talking points)

- **Out-of-time split** prevents leakage that a random split would hide.
- **Validate before train.** Bad data fails the DAG instead of silently producing a bad model.
- **Stable feature columns** between training and scoring (fixed category list).
- **Monitoring is a first-class stage**, not an afterthought.

## Known limits and next steps

- Spark job and Airflow DAG are written but not executed locally (no Java; Airflow is not supported on Windows). Test them on Databricks Community Edition / Docker.
- Next: Delta Lake output, a model registry stage with promotion gates, a small RAG/agent demo (see [docs/interview_prep.md](docs/interview_prep.md)).
