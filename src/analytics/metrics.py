"""Active-user metrics: DAU, WAU, MAU, stickiness."""
from __future__ import annotations

import pandas as pd


def _normalize_dates(events: pd.DataFrame) -> pd.DataFrame:
    df = events[["user_id", "event_time"]].copy()
    df["date"] = df["event_time"].dt.normalize()
    return df


def dau(events: pd.DataFrame) -> pd.DataFrame:
    """Daily active user counts by calendar date."""
    df = _normalize_dates(events)
    return (
        df.groupby("date")["user_id"].nunique().rename("dau").reset_index().sort_values("date")
    )


def wau(events: pd.DataFrame) -> pd.DataFrame:
    """Weekly active users (ISO week)."""
    df = _normalize_dates(events)
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    return (
        df.groupby("week")["user_id"].nunique().rename("wau").reset_index().sort_values("week")
    )


def mau(events: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_dates(events)
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    return (
        df.groupby("month")["user_id"].nunique().rename("mau").reset_index().sort_values("month")
    )


def rolling_active_users(events: pd.DataFrame, window: int = 28) -> pd.DataFrame:
    """Rolling unique-user count over a trailing window of `window` days."""
    df = _normalize_dates(events)
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    daily_users = df.groupby("date")["user_id"].apply(set)

    counts = []
    for d in all_dates:
        start = d - pd.Timedelta(days=window - 1)
        in_window = daily_users.loc[(daily_users.index >= start) & (daily_users.index <= d)]
        unique = set().union(*in_window.values) if len(in_window) else set()
        counts.append({"date": d, "rolling_users": len(unique)})
    return pd.DataFrame(counts)


def dau_mau_ratio(events: pd.DataFrame) -> pd.DataFrame:
    """Stickiness ratio (DAU / 30-day rolling unique users) by date."""
    df = _normalize_dates(events)
    dau_series = df.groupby("date")["user_id"].nunique().rename("dau")
    rolling = rolling_active_users(events, window=30).set_index("date")["rolling_users"]
    out = pd.concat([dau_series, rolling], axis=1).dropna()
    out["stickiness"] = out["dau"] / out["rolling_users"].replace(0, pd.NA)
    return out.reset_index().rename(columns={"index": "date"})
