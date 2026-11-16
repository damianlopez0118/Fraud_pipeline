"""Airflow DAG for the fraud pipeline (deploy to an Airflow environment; not run locally on Windows)."""

from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

OUT = Path("/opt/airflow/data/fraud_pipeline")


@dag(
    dag_id="fraud_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["fraud", "ml"],
)
def fraud_pipeline():
    @task
    def ingest_task() -> str:
        from fraud_pipeline.pipeline import ingest

        return str(ingest(OUT, n=20_000, seed=42))

    @task
    def validate_task(path: str) -> str:
        from fraud_pipeline.pipeline import validate

        if not validate(Path(path), OUT):
            raise ValueError("Data quality checks failed")  # fails the run and triggers alerting
        return path

    @task
    def train_task(path: str) -> dict:
        from fraud_pipeline.pipeline import train_and_monitor

        return train_and_monitor(Path(path), OUT)

    train_task(validate_task(ingest_task()))


fraud_pipeline()
