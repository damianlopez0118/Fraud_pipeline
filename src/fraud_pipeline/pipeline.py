"""Pipeline stages and CLI entry points for local and scheduled runs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data_gen import generate_orders
from .drift import drift_report
from .features import NUMERIC, build_features, time_split
from .quality import detect_volume_anomalies, run_order_checks, summarize
from .report import build_dashboard
from .train import train_model


def ingest(out_dir: Path, n: int, seed: int) -> Path:
    path = out_dir / "orders.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_orders(n=n, seed=seed).to_parquet(path, index=False)
    return path


def validate(orders_path: Path, out_dir: Path) -> bool:
    """Fail fast: a failing check stops the pipeline before training."""
    results = summarize(run_order_checks(pd.read_parquet(orders_path)))
    results.to_csv(out_dir / "quality_report.csv", index=False)
    return bool(results["passed"].all())


def train_and_monitor(
    orders_path: Path, out_dir: Path, tracking_uri: str | None = None
) -> dict:
    orders = pd.read_parquet(orders_path)
    _, metrics, x_train, _, scores = train_model(
        orders, tracking_dir=out_dir / "mlflow", tracking_uri=tracking_uri
    )
    _, test_df = time_split(orders)

    # Simulate a later production batch with shifted amounts to exercise drift monitoring
    prod = generate_orders(n=len(test_df), seed=7, drift=0.6)
    item_types = sorted(orders["item_type"].unique())
    drift = drift_report(x_train, build_features(prod, item_types), NUMERIC)
    drift.to_csv(out_dir / "drift_report.csv", index=False)

    daily = detect_volume_anomalies(orders, "created_ts")
    build_dashboard(
        orders, test_df["is_fraud"], scores, drift, daily, metrics, out_dir / "dashboard.html"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run(
    out_dir: Path,
    n: int = 20_000,
    seed: int = 42,
    tracking_uri: str | None = None,
) -> dict:
    orders_path = ingest(out_dir, n, seed)
    if not validate(orders_path, out_dir):
        raise RuntimeError("Data quality checks failed; see quality_report.csv")
    return train_and_monitor(orders_path, out_dir, tracking_uri=tracking_uri)


def run_databricks(output_path: str) -> dict:
    return run(Path(output_path), tracking_uri="databricks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fraud pipeline end to end")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.rows, args.seed), indent=2))


if __name__ == "__main__":
    main()
