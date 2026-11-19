"""Read-only monitoring of monthly, independently evaluated Socure scores."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MonitoringPolicy:
    gini_weak: float = 0.20
    gini_good: float = 0.40
    ks_weak: float = 0.30
    ks_good: float = 0.40
    decline_alert: float = 0.05
    max_age_days: int = 62

    def __post_init__(self):
        if not -1 <= self.gini_weak < self.gini_good <= 1:
            raise ValueError("Gini references must be ordered fractions in [-1, 1]")
        if not 0 <= self.ks_weak < self.ks_good <= 1:
            raise ValueError("KS references must be ordered fractions in [0, 1]")
        if not 0 < self.decline_alert <= 2 or not self.max_age_days > 0:
            raise ValueError("Decline threshold and maximum age must be positive")


def summarize_socure(
    gini: pd.DataFrame,
    ks: pd.DataFrame,
    *,
    policy: MonitoringPolicy | None = None,
    as_of: str | None = None,
) -> dict:
    """Use canonical month/score_type/value columns; never fill missing evaluations."""
    policy = policy or MonitoringPolicy()
    today = pd.Timestamp(as_of or pd.Timestamp.now().date()).normalize()
    frames = []
    for metric, source, bounds in (("gini", gini, (-1, 1)), ("ks", ks, (0, 1))):
        required = {"month", "score_type", "value"}
        if not required.issubset(source.columns):
            raise ValueError(f"{metric} requires columns: {sorted(required)}")
        frame = source[sorted(required)].copy()
        frame["month"] = pd.to_datetime(frame["month"], errors="raise").dt.to_period("M")
        if frame["month"].isna().any() or frame["score_type"].isna().any():
            raise ValueError(f"{metric}: month and score_type cannot be null")
        frame["score_type"] = frame["score_type"].astype(str).str.strip()
        if frame["score_type"].eq("").any():
            raise ValueError(f"{metric}: score_type cannot be empty")
        if frame.duplicated(["month", "score_type"]).any():
            raise ValueError(f"{metric}: filter to one population/version per month and score_type")
        frame[metric] = pd.to_numeric(frame.pop("value"), errors="raise")
        if (frame[metric].notna() & ~frame[metric].between(*bounds)).any():
            raise ValueError(f"{metric} must be in {bounds}; convert percentages to fractions")
        for suffix in ("min", "max", "source_rows"):
            column = f"{metric}_{suffix}"
            if column in source:
                frame[column] = pd.to_numeric(source[column], errors="raise")
                if suffix != "source_rows":
                    invalid = frame[column].notna() & ~frame[column].between(*bounds)
                    if invalid.any():
                        raise ValueError(f"{column} must be in {bounds}")
        frames.append(frame)
    monthly = frames[0].merge(frames[1], on=["month", "score_type"], how="outer")
    if monthly.empty:
        return {"as_of": str(today.date()), "policy": vars(policy), "latest": [], "monthly": []}
    if (monthly["month"] > today.to_period("M")).any():
        raise ValueError("Evaluation month cannot be in the future")
    months = pd.period_range(monthly["month"].min(), monthly["month"].max(), freq="M")
    index = pd.MultiIndex.from_product(
        [months, sorted(monthly["score_type"].unique())], names=["month", "score_type"]
    )
    monthly = monthly.set_index(["month", "score_type"]).reindex(index).reset_index()
    monthly = monthly.sort_values(["score_type", "month"])
    monthly["roc_auc"] = (monthly["gini"] + 1) / 2
    for metric in ("gini", "ks"):
        monthly[f"{metric}_change"] = monthly.groupby("score_type")[metric].diff()
    latest = []
    for score_type, history in monthly.groupby("score_type", sort=True):
        row = history.iloc[-1]
        reasons = []
        observed = history[history[["gini", "ks"]].notna().all(axis=1)]
        age = None if observed.empty else max(
            0, (today - observed.iloc[-1]["month"].end_time.normalize()).days
        )
        if row["month"] == today.to_period("M"):
            reasons.append("Open month: provisional and potentially immature labels")
        if pd.isna(row["gini"]) or pd.isna(row["ks"]):
            reasons.append("Missing Gini or KS evaluation")
        if age is None or age > policy.max_age_days:
            reasons.append("Stale or unavailable paired evaluation")
        for metric in ("gini", "ks"):
            if row.get(f"{metric}_source_rows", 1) > 1:
                reasons.append(f"Ambiguous {metric.upper()}: multiple source evaluations")
            value = row[metric]
            if pd.notna(value) and value < getattr(policy, f"{metric}_weak"):
                reasons.append(f"{metric.upper()} below weak threshold")
            upper = row.get(f"{metric}_max")
            if pd.isna(value) and pd.notna(upper) and upper < getattr(policy, f"{metric}_weak"):
                reasons.append(f"All observed {metric.upper()} values below weak reference")
            change = row[f"{metric}_change"]
            if pd.notna(change) and change <= -policy.decline_alert:
                reasons.append(f"{metric.upper()} declined by at least {policy.decline_alert:.2f}")
        if reasons:
            status = "alert"
        elif row["gini"] >= policy.gini_good and row["ks"] >= policy.ks_good:
            status = "within_reference"
        else:
            status = "watch"
        latest.append({"score_type": score_type, "status": status, "reasons": reasons,
                       "last_complete_month": None if observed.empty else str(observed.iloc[-1]["month"]),
                       "age_days": age})
    monthly["month"] = monthly["month"].astype(str)
    records = monthly.astype(object).where(monthly.notna(), None).to_dict("records")
    return {"as_of": str(today.date()), "policy": vars(policy), "latest": latest, "monthly": records}