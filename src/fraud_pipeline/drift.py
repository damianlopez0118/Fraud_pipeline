"""Population Stability Index (PSI) drift monitoring."""

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """PSI between reference and current distributions: <0.1 stable, 0.1-0.25 moderate, >0.25 major."""
    edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected.dropna(), edges)[0] / max(expected.notna().sum(), 1)
    a = np.histogram(actual.dropna(), edges)[0] / max(actual.notna().sum(), 1)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    rows = []
    for col in columns:
        value = psi(reference[col], current[col])
        status = "major" if value > 0.25 else "moderate" if value > 0.1 else "stable"
        rows.append({"feature": col, "psi": round(value, 4), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
