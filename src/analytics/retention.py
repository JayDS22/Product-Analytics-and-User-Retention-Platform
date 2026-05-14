"""Retention curves and N-day retention."""
from __future__ import annotations

import numpy as np
import pandas as pd


def retention_curve(
    events: pd.DataFrame,
    users: pd.DataFrame,
    max_day: int = 30,
) -> pd.DataFrame:
    """Probability of returning N days after signup, by signup cohort.

    Returns a long-form frame with columns: day, retained_rate, n_users.
    """
    e = events[["user_id", "event_time"]].copy()
    e["event_date"] = e["event_time"].dt.normalize()
    u = users[["user_id", "signup_date"]].copy()
    u["signup_date"] = u["signup_date"].dt.normalize()

    df = e.merge(u, on="user_id", how="inner")
    df["day_index"] = (df["event_date"] - df["signup_date"]).dt.days
    df = df[(df["day_index"] >= 0) & (df["day_index"] <= max_day)]

    total_users = u["user_id"].nunique()
    returned = df.groupby("day_index")["user_id"].nunique()
    full_index = pd.RangeIndex(0, max_day + 1, name="day_index")
    returned = returned.reindex(full_index, fill_value=0)

    out = pd.DataFrame(
        {
            "day": returned.index,
            "n_users": returned.values,
            "retained_rate": returned.values / max(total_users, 1),
        }
    )
    return out


def nth_day_retention(events: pd.DataFrame, users: pd.DataFrame, n: int) -> float:
    curve = retention_curve(events, users, max_day=n)
    return float(curve.loc[curve["day"] == n, "retained_rate"].iloc[0])
