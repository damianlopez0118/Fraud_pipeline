"""Lightweight data-quality framework: declarative checks, results, and volume anomaly detection."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def check_not_null(
    df: pd.DataFrame, columns: list[str], max_null_rate: float = 0.0
) -> list[CheckResult]:
    out = []
    for col in columns:
        rate = float(df[col].isna().mean()) if col in df else 1.0
        out.append(CheckResult(f"not_null:{col}", rate <= max_null_rate, f"null_rate={rate:.4f}"))
    return out


def check_unique(df: pd.DataFrame, column: str) -> CheckResult:
    dupes = int(df[column].duplicated().sum())
    return CheckResult(f"unique:{column}", dupes == 0, f"duplicates={dupes}")


def check_range(df: pd.DataFrame, column: str, low: float, high: float) -> CheckResult:
    bad = int((~df[column].between(low, high)).sum())
    return CheckResult(f"range:{column}[{low},{high}]", bad == 0, f"out_of_range={bad}")


def check_allowed_values(df: pd.DataFrame, column: str, allowed: set) -> CheckResult:
    bad = sorted(set(df[column].dropna().unique()) - allowed)
    return CheckResult(f"allowed:{column}", not bad, f"unexpected={bad[:5]}")


def check_schema(df: pd.DataFrame, required: set[str]) -> CheckResult:
    missing = sorted(required - set(df.columns))
    return CheckResult("schema", not missing, f"missing={missing}")


def detect_volume_anomalies(
    df: pd.DataFrame, ts_col: str, window: int = 14, z_threshold: float = 3.0
) -> pd.DataFrame:
    """Flag days whose order volume deviates from the trailing mean by more than z_threshold sigmas."""
    daily = df.set_index(ts_col).resample("D").size().rename("orders").to_frame()
    roll = daily["orders"].shift(1).rolling(window, min_periods=window // 2)
    daily["z"] = (daily["orders"] - roll.mean()) / roll.std().replace(0, float("nan"))
    daily["anomaly"] = daily["z"].abs() > z_threshold
    return daily


def run_order_checks(df: pd.DataFrame) -> list[CheckResult]:
    required = {"order_id", "created_ts", "order_amount", "country_cd", "is_fraud"}
    results = [check_schema(df, required)]
    if not results[0].passed:
        return results
    results += check_not_null(df, sorted(required))
    results.append(check_unique(df, "order_id"))
    results.append(check_range(df, "order_amount", 0, 1_000_000))
    results.append(check_allowed_values(df, "is_fraud", {0, 1}))
    return results


def summarize(results: list[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])
