"""Synthetic event generator that simulates a realistic SaaS product.

Events are drawn from a non-homogeneous Poisson process where the per-user rate
declines over time according to a latent engagement score, with weekly seasonality
and plan-tier multipliers. Churn label is computed downstream from event recency.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CHANNELS,
    COUNTRIES,
    DEVICES,
    EVENT_TYPES,
    EVENTS_FILE,
    FEATURE_NAMES,
    PLANS,
    RANDOM_SEED,
    USERS_FILE,
)


PLAN_BASE_RATE = {"free": 0.6, "starter": 1.2, "pro": 2.1, "enterprise": 3.4}
PLAN_REVENUE = {"free": 0.0, "starter": 19.0, "pro": 79.0, "enterprise": 299.0}
EVENT_WEIGHTS = np.array([0.18, 0.34, 0.18, 0.08, 0.06, 0.04, 0.03, 0.02, 0.04, 0.03])
EVENT_REVENUE_FACTOR = {
    "purchase": 1.0,
    "checkout": 0.0,
    "add_to_cart": 0.0,
}


def generate_users(
    n_users: int,
    start_date: str,
    end_date: str,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a user table with signup dates and acquisition attributes."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    span_days = (end - start).days

    signup_offsets = rng.integers(0, span_days, size=n_users)
    signup_dates = pd.to_datetime([start + timedelta(days=int(d)) for d in signup_offsets])

    plan_probs = [0.55, 0.25, 0.15, 0.05]
    plans = rng.choice(PLANS, size=n_users, p=plan_probs)

    channels = rng.choice(CHANNELS, size=n_users, p=[0.28, 0.22, 0.14, 0.16, 0.12, 0.08])
    countries = rng.choice(COUNTRIES, size=n_users)
    devices = rng.choice(DEVICES, size=n_users, p=[0.62, 0.30, 0.08])

    engagement = rng.beta(2.0, 2.5, size=n_users)
    onboarding_completed = rng.binomial(1, 0.5 + 0.3 * engagement).astype(bool)

    return pd.DataFrame(
        {
            "user_id": [f"u_{i:07d}" for i in range(n_users)],
            "signup_date": signup_dates,
            "plan": plans,
            "channel": channels,
            "country": countries,
            "device": devices,
            "engagement_score": engagement.round(4),
            "onboarding_completed": onboarding_completed,
        }
    )


def _user_event_count(
    rng: np.random.Generator,
    plan: str,
    engagement: float,
    active_days: int,
) -> int:
    """Expected event volume across the active period for a single user."""
    base = PLAN_BASE_RATE[plan]
    daily_rate = base * (0.4 + 1.6 * engagement)
    expected = daily_rate * max(active_days, 1)
    return int(rng.poisson(expected))


def _sample_event_times(
    rng: np.random.Generator,
    signup: pd.Timestamp,
    end: pd.Timestamp,
    engagement: float,
    n_events: int,
) -> np.ndarray:
    """Draw event timestamps with a decay envelope and weekly seasonality."""
    if n_events == 0:
        return np.array([], dtype="datetime64[ns]")

    horizon_days = max((end - signup).days, 1)
    decay_lambda = 1.0 / (10.0 + 80.0 * engagement)
    days = rng.exponential(scale=1.0 / decay_lambda, size=n_events * 3)
    days = days[days <= horizon_days][:n_events]
    if days.size < n_events:
        extra = rng.uniform(0, horizon_days, size=n_events - days.size)
        days = np.concatenate([days, extra])

    weekday = ((signup.weekday() + days.astype(int)) % 7)
    weekly_boost = np.where(weekday >= 5, 0.7, 1.1)
    keep = rng.random(n_events) < (weekly_boost / weekly_boost.max())
    days = days[keep] if keep.sum() > 0 else days

    seconds = days * 86400 + rng.uniform(0, 86400, size=days.size)
    timestamps = signup.to_datetime64() + seconds.astype("timedelta64[s]")
    timestamps = np.sort(timestamps)
    return timestamps


def generate_events(
    users: pd.DataFrame,
    end_date: str,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate event-level data for all users up to end_date."""
    rng = np.random.default_rng(seed + 1)
    end = pd.Timestamp(end_date)

    rows: list[dict] = []
    for u in users.itertuples(index=False):
        active_days = (end - u.signup_date).days
        if active_days <= 0:
            continue

        n_events = _user_event_count(rng, u.plan, u.engagement_score, active_days)
        if n_events == 0:
            continue

        timestamps = _sample_event_times(rng, u.signup_date, end, u.engagement_score, n_events)
        if timestamps.size == 0:
            continue

        event_types = rng.choice(EVENT_TYPES, size=timestamps.size, p=EVENT_WEIGHTS)
        feature_targets = rng.choice(FEATURE_NAMES, size=timestamps.size)
        session_ids = np.cumsum(rng.random(timestamps.size) < 0.18) + (hash(u.user_id) & 0xFFFF)

        purchase_mask = event_types == "purchase"
        revenue = np.zeros(timestamps.size)
        if purchase_mask.any():
            revenue[purchase_mask] = rng.gamma(2.0, PLAN_REVENUE[u.plan] / 2.0 + 5.0, size=purchase_mask.sum())

        for ts, et, ft, sid, rev in zip(timestamps, event_types, feature_targets, session_ids, revenue):
            rows.append(
                {
                    "user_id": u.user_id,
                    "event_time": ts,
                    "event_type": et,
                    "feature": ft,
                    "session_id": int(sid),
                    "revenue": float(rev),
                    "plan": u.plan,
                    "device": u.device,
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events["event_time"] = pd.to_datetime(events["event_time"])
    events = events.sort_values("event_time").reset_index(drop=True)
    return events


def generate_dataset(
    n_users: int = 8000,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    seed: int = RANDOM_SEED,
    out_users: Path | None = None,
    out_events: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate users and events and persist to parquet."""
    users = generate_users(n_users, start_date, end_date, seed=seed)
    events = generate_events(users, end_date=end_date, seed=seed)

    out_users = out_users or USERS_FILE
    out_events = out_events or EVENTS_FILE
    users.to_parquet(out_users, index=False)
    events.to_parquet(out_events, index=False)
    return users, events


if __name__ == "__main__":
    u, e = generate_dataset()
    print(f"Users: {len(u):,}  Events: {len(e):,}")
    print(f"Date range: {e['event_time'].min()} -> {e['event_time'].max()}")
