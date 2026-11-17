"""XGBoost training and evaluation with MLflow tracking."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from .features import TARGET, build_features, time_split


def recall_at_precision(y_true, scores, min_precision: float = 0.8) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y_true, scores) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_p80": recall_at_precision(y_true, scores, 0.8),
    }


def train_model(
    df: pd.DataFrame,
    tracking_dir: Path | None = None,
    seed: int = 42,
    tracking_uri: str | None = None,
):
    """Train on the earlier orders, evaluate out-of-time, and log everything to MLflow."""
    train_df, test_df = time_split(df)
    item_types = sorted(df["item_type"].unique())
    x_train, x_test = build_features(train_df, item_types), build_features(test_df, item_types)
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        # Class imbalance: weight positives by the neg/pos ratio
        "scale_pos_weight": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        "eval_metric": "aucpr",
        "random_state": seed,
        "n_jobs": 1,
    }
    model = XGBClassifier(**params)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test, scores)

    if tracking_dir is not None:
        tracking_dir.mkdir(parents=True, exist_ok=True)
        default_tracking_uri = f"sqlite:///{(tracking_dir / 'mlflow.db').as_posix()}"
        mlflow.set_tracking_uri(tracking_uri or default_tracking_uri)
        mlflow.set_experiment("fraud_orders")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_params({"train_rows": len(train_df), "test_rows": len(test_df)})
            mlflow.log_metrics(metrics)
            importance = pd.Series(model.feature_importances_, index=x_train.columns)
            mlflow.log_dict(
                importance.sort_values(ascending=False).round(4).to_dict(),
                "feature_importance.json",
            )

    return model, metrics, x_train, x_test, np.asarray(scores)
