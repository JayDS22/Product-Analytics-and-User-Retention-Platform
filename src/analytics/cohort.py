"""Cohort analysis: triangular retention and revenue matrices."""
from __future__ import annotations

import pandas as pd


def cohort_matrix(
    events: pd.DataFrame,
    users: pd.DataFrame,
    max_period: int = 12,
    freq: str = "W",
) -> pd.DataFrame:
    """Cohort retention matrix indexed by signup period and period offset."""
    u = users[["user_id", "signup_date"]].copy()
    u["cohort"] = u["signup_date"].dt.to_period(freq).dt.start_time

    e = events[["user_id", "event_time"]].copy()
    e["event_period"] = e["event_time"].dt.to_period(freq).dt.start_time

    df = e.merge(u, on="user_id", how="inner")
    df["period_offset"] = ((df["event_period"] - df["cohort"]) / pd.Timedelta(weeks=1 if freq == "W" else 30)).astype(int)
    df = df[df["period_offset"].between(0, max_period)]

    counts = df.groupby(["cohort", "period_offset"])["user_id"].nunique().unstack(fill_value=0)
    cohort_sizes = u.groupby("cohort")["user_id"].nunique()
    matrix = counts.div(cohort_sizes, axis=0).fillna(0.0)
    matrix.index = matrix.index.strftime("%Y-%m-%d")
    matrix.columns = [f"P{c}" for c in matrix.columns]
    return matrix


def cohort_revenue(
    events: pd.DataFrame,
    users: pd.DataFrame,
    max_period: int = 12,
    freq: str = "W",
) -> pd.DataFrame:
    """Cumulative revenue per user per cohort over time."""
    u = users[["user_id", "signup_date"]].copy()
    u["cohort"] = u["signup_date"].dt.to_period(freq).dt.start_time

    e = events[events["event_type"] == "purchase"][["user_id", "event_time", "revenue"]].copy()
    e["event_period"] = e["event_time"].dt.to_period(freq).dt.start_time

    df = e.merge(u, on="user_id", how="inner")
    df["period_offset"] = ((df["event_period"] - df["cohort"]) / pd.Timedelta(weeks=1 if freq == "W" else 30)).astype(int)
    df = df[df["period_offset"].between(0, max_period)]

    rev = df.groupby(["cohort", "period_offset"])["revenue"].sum().unstack(fill_value=0.0)
    sizes = u.groupby("cohort")["user_id"].nunique()
    arpu = rev.div(sizes, axis=0).cumsum(axis=1).fillna(0.0)
    arpu.index = arpu.index.strftime("%Y-%m-%d")
    arpu.columns = [f"P{c}" for c in arpu.columns]
    return arpu
