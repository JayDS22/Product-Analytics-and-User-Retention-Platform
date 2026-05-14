"""Funnel and journey analytics."""
from __future__ import annotations

from typing import Sequence

import pandas as pd


def funnel_conversion(events: pd.DataFrame, steps: Sequence[str]) -> pd.DataFrame:
    """Per-step user counts and conversion rates for an ordered funnel.

    A user is counted at step k if they ever performed the event for step k
    after performing step k-1.
    """
    df = events.sort_values(["user_id", "event_time"])[["user_id", "event_time", "event_type"]]

    reached: set[str] = set(df["user_id"].unique())
    rows = []
    prev_time: dict[str, pd.Timestamp] = {u: pd.Timestamp.min for u in reached}

    for i, step in enumerate(steps):
        step_users: set[str] = set()
        sub = df[df["event_type"] == step]
        for user, time in zip(sub["user_id"].to_numpy(), sub["event_time"].to_numpy()):
            if user in reached and time > prev_time.get(user, pd.Timestamp.min.to_datetime64()):
                step_users.add(user)
                prev_time[user] = time
        reached = step_users
        total = max(len(set(df["user_id"].unique())), 1)
        prev_count = rows[-1]["users"] if rows else total
        rows.append(
            {
                "step_index": i,
                "step": step,
                "users": len(reached),
                "share_of_total": len(reached) / total,
                "step_conversion": (len(reached) / prev_count) if prev_count > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def journey_paths(events: pd.DataFrame, max_steps: int = 5, top_n: int = 15) -> pd.DataFrame:
    """Most frequent prefix event sequences (length up to max_steps)."""
    df = events.sort_values(["user_id", "event_time"])
    user_seq = df.groupby("user_id")["event_type"].apply(lambda s: tuple(s.iloc[:max_steps]))
    counts = user_seq.value_counts().head(top_n).reset_index()
    counts.columns = ["path", "users"]
    counts["path"] = counts["path"].apply(lambda t: " > ".join(t))
    return counts
