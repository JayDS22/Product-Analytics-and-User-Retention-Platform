"""Feature engineering pipeline that builds 120+ user-level signals.

Groups:
    Recency, Frequency, Monetary, Diversity, Trend, Temporal,
    Engagement, Per-feature, Per-event-type, Profile.

All aggregations are vectorized via groupby and operate on a fixed
observation window ending at `reference_date`. The churn label is computed
from a forward-looking horizon disjoint from the observation window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import (
    CHURN_HORIZON_DAYS,
    EVENT_TYPES,
    FEATURE_NAMES,
    OBSERVATION_WINDOW_DAYS,
    PLANS,
)


@dataclass
class FeatureSpec:
    reference_date: pd.Timestamp
    observation_days: int = OBSERVATION_WINDOW_DAYS
    horizon_days: int = CHURN_HORIZON_DAYS

    @property
    def window_start(self) -> pd.Timestamp:
        return self.reference_date - pd.Timedelta(days=self.observation_days)

    @property
    def horizon_end(self) -> pd.Timestamp:
        return self.reference_date + pd.Timedelta(days=self.horizon_days)


def _shannon_entropy(series: pd.Series) -> float:
    counts = series.value_counts()
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.where(den == 0, num / den.replace(0, np.nan)).fillna(0.0)


def _filter_window(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (events["event_time"] >= start) & (events["event_time"] < end)
    return events.loc[mask].copy()


def _recency_features(events: pd.DataFrame, users: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """Days since last action across event subtypes."""
    out = users[["user_id"]].copy()

    def _days_since_last(df: pd.DataFrame, name: str) -> pd.Series:
        last = df.groupby("user_id")["event_time"].max()
        delta = (ref - last).dt.total_seconds() / 86400.0
        delta.name = name
        return delta

    out = out.merge(_days_since_last(events, "days_since_last_event").reset_index(), on="user_id", how="left")
    for et in ["purchase", "session_start", "feature_used", "support_ticket"]:
        sub = events[events["event_type"] == et]
        out = out.merge(_days_since_last(sub, f"days_since_last_{et}").reset_index(), on="user_id", how="left")

    first_purchase = events[events["event_type"] == "purchase"].groupby("user_id")["event_time"].min()
    days_since_first_purchase = (ref - first_purchase).dt.total_seconds() / 86400.0
    out = out.merge(days_since_first_purchase.rename("days_since_first_purchase").reset_index(), on="user_id", how="left")

    last_event = events.groupby("user_id")["event_time"].max()
    out = out.merge(((ref - last_event).dt.total_seconds() / 3600.0).rename("hours_since_last_event").reset_index(), on="user_id", how="left")

    signup_lookup = users.set_index("user_id")["signup_date"]
    out["days_since_signup"] = ((ref - out["user_id"].map(signup_lookup)).dt.total_seconds() / 86400.0)

    fill_cols = [c for c in out.columns if c != "user_id"]
    out[fill_cols] = out[fill_cols].fillna(9999.0)
    return out


def _windowed_frequency(events: pd.DataFrame, ref: pd.Timestamp, days: int) -> pd.DataFrame:
    """Activity counts within the trailing `days` window."""
    start = ref - pd.Timedelta(days=days)
    sub = _filter_window(events, start, ref)
    suffix = f"_w{days}"

    g = sub.groupby("user_id")
    out = g.size().rename(f"n_events{suffix}").to_frame()
    out[f"n_sessions{suffix}"] = g["session_id"].nunique()
    out[f"n_active_days{suffix}"] = g["event_time"].apply(lambda s: s.dt.normalize().nunique())
    out[f"unique_features{suffix}"] = g["feature"].nunique()
    out[f"revenue{suffix}"] = g["revenue"].sum()

    for et in ["purchase", "page_view", "feature_used", "search", "support_ticket", "add_to_cart", "checkout"]:
        out[f"n_{et}{suffix}"] = sub[sub["event_type"] == et].groupby("user_id").size()

    daily = sub.assign(day=sub["event_time"].dt.normalize()).groupby(["user_id", "day"]).size()
    out[f"max_events_in_day{suffix}"] = daily.groupby("user_id").max()
    out[f"events_per_active_day{suffix}"] = out[f"n_events{suffix}"] / out[f"n_active_days{suffix}"].replace(0, np.nan)

    weekend_mask = sub["event_time"].dt.weekday >= 5
    out[f"n_weekend_events{suffix}"] = sub[weekend_mask].groupby("user_id").size()

    return out.fillna(0.0)


def _monetary_features(events: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    purchases = events[events["event_type"] == "purchase"]
    g = purchases.groupby("user_id")["revenue"]
    out = pd.DataFrame(
        {
            "total_revenue": g.sum(),
            "avg_purchase_value": g.mean(),
            "max_purchase_value": g.max(),
            "min_purchase_value": g.min(),
            "purchase_count_total": g.count(),
            "revenue_std": g.std(),
        }
    ).fillna(0.0)

    all_events_g = events.groupby("user_id")
    active_days = all_events_g["event_time"].apply(lambda s: s.dt.normalize().nunique())
    sessions = all_events_g["session_id"].nunique()
    out["revenue_per_active_day"] = (out["total_revenue"] / active_days.reindex(out.index).replace(0, np.nan)).fillna(0.0)
    out["revenue_per_session"] = (out["total_revenue"] / sessions.reindex(out.index).replace(0, np.nan)).fillna(0.0)
    return out


def _diversity_features(events: pd.DataFrame) -> pd.DataFrame:
    g = events.groupby("user_id")
    out = pd.DataFrame(
        {
            "unique_event_types": g["event_type"].nunique(),
            "unique_features_used": g["feature"].nunique(),
            "unique_sessions": g["session_id"].nunique(),
            "lifetime_events": g.size(),
            "event_type_entropy": g["event_type"].apply(_shannon_entropy),
            "feature_entropy": g["feature"].apply(_shannon_entropy),
        }
    )

    def _dominant_share(s: pd.Series) -> float:
        counts = s.value_counts(normalize=True)
        return float(counts.iloc[0]) if not counts.empty else 0.0

    out["dominant_event_share"] = g["event_type"].apply(_dominant_share)
    out["dominant_feature_share"] = g["feature"].apply(_dominant_share)
    out["feature_diversity_ratio"] = out["unique_features_used"] / len(FEATURE_NAMES)
    out["event_diversity_ratio"] = out["unique_event_types"] / len(EVENT_TYPES)
    return out.fillna(0.0)


def _trend_features(events: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    """Velocity and acceleration of activity over time."""
    recent = _filter_window(events, ref - pd.Timedelta(days=7), ref)
    prev = _filter_window(events, ref - pd.Timedelta(days=14), ref - pd.Timedelta(days=7))
    long_win = _filter_window(events, ref - pd.Timedelta(days=28), ref)

    recent_n = recent.groupby("user_id").size()
    prev_n = prev.groupby("user_id").size()
    long_n = long_win.groupby("user_id").size()

    user_index = events["user_id"].unique()
    recent_n = recent_n.reindex(user_index, fill_value=0)
    prev_n = prev_n.reindex(user_index, fill_value=0)
    long_n = long_n.reindex(user_index, fill_value=0)

    out = pd.DataFrame(index=user_index)
    out["events_change_w7_vs_prev"] = recent_n - prev_n
    out["events_ratio_w7_to_prev"] = (recent_n / prev_n.replace(0, np.nan)).fillna(0.0)
    out["events_ratio_w7_to_w28"] = (recent_n / long_n.replace(0, np.nan)).fillna(0.0)
    out["momentum_index"] = (recent_n - prev_n) / (recent_n + prev_n).replace(0, np.nan)
    out["momentum_index"] = out["momentum_index"].fillna(0.0)

    def _linear_slope(s: pd.Series) -> float:
        if len(s) < 2:
            return 0.0
        x = np.arange(len(s))
        y = s.values
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)

    daily = long_win.assign(day=long_win["event_time"].dt.normalize()).groupby(["user_id", "day"]).size()
    daily_unstacked = daily.unstack(fill_value=0).reindex(user_index, fill_value=0)
    if not daily_unstacked.empty:
        out["linear_trend_events"] = daily_unstacked.apply(_linear_slope, axis=1)
        out["activity_std"] = daily_unstacked.std(axis=1).fillna(0.0)
        zero_days = (daily_unstacked == 0).sum(axis=1)
        out["days_with_zero_activity_w28"] = zero_days

        def _longest_run(row: np.ndarray) -> int:
            best = curr = 0
            for v in row:
                if v == 1:
                    curr += 1
                    best = max(best, curr)
                else:
                    curr = 0
            return best

        is_zero = (daily_unstacked == 0).astype(int).values
        streaks = pd.Series(
            [int(_longest_run(r)) for r in is_zero],
            index=daily_unstacked.index,
        )
        out["longest_inactive_streak_w28"] = streaks
    else:
        out["linear_trend_events"] = 0.0
        out["activity_std"] = 0.0
        out["days_with_zero_activity_w28"] = 0
        out["longest_inactive_streak_w28"] = 0

    sessions_recent = recent.groupby("user_id")["session_id"].nunique().reindex(user_index, fill_value=0)
    sessions_prev = prev.groupby("user_id")["session_id"].nunique().reindex(user_index, fill_value=0)
    out["sessions_change_w7_vs_prev"] = sessions_recent - sessions_prev

    revenue_recent = recent.groupby("user_id")["revenue"].sum().reindex(user_index, fill_value=0)
    revenue_prev = prev.groupby("user_id")["revenue"].sum().reindex(user_index, fill_value=0)
    out["revenue_change_w7_vs_prev"] = revenue_recent - revenue_prev

    out["acceleration"] = out["events_change_w7_vs_prev"] - (long_n - 2 * recent_n)
    out.index.name = "user_id"
    return out.fillna(0.0).reset_index().set_index("user_id")


def _temporal_features(events: pd.DataFrame) -> pd.DataFrame:
    df = events.assign(
        hour=events["event_time"].dt.hour,
        dow=events["event_time"].dt.weekday,
    )

    def _bucket_share(s: pd.Series, lo: int, hi: int) -> float:
        if len(s) == 0:
            return 0.0
        return float(((s >= lo) & (s < hi)).mean())

    g = df.groupby("user_id")
    out = pd.DataFrame(
        {
            "share_morning_events": g["hour"].apply(lambda s: _bucket_share(s, 6, 12)),
            "share_afternoon_events": g["hour"].apply(lambda s: _bucket_share(s, 12, 18)),
            "share_evening_events": g["hour"].apply(lambda s: _bucket_share(s, 18, 23)),
            "share_night_events": g["hour"].apply(lambda s: ((s < 6) | (s >= 23)).mean()),
            "share_weekend_events": g["dow"].apply(lambda s: (s >= 5).mean()),
            "share_business_hours": g["hour"].apply(lambda s: _bucket_share(s, 9, 17)),
            "hour_entropy": g["hour"].apply(_shannon_entropy),
            "dow_entropy": g["dow"].apply(_shannon_entropy),
            "most_active_hour": g["hour"].apply(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else 0),
            "most_active_dow": g["dow"].apply(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else 0),
        }
    )
    out["night_owl_score"] = out["share_night_events"] - out["share_morning_events"]
    out["weekend_warrior_score"] = out["share_weekend_events"] / 0.2857 - 1.0
    return out.fillna(0.0)


def _engagement_features(events: pd.DataFrame) -> pd.DataFrame:
    """Session shape and conversion ratios."""
    g_sess = events.groupby(["user_id", "session_id"])
    session_lengths = g_sess.size().rename("session_len")
    session_durations = (g_sess["event_time"].max() - g_sess["event_time"].min()).dt.total_seconds().rename("session_dur")
    session_df = pd.concat([session_lengths, session_durations], axis=1).reset_index()

    sg = session_df.groupby("user_id")
    out = pd.DataFrame(
        {
            "avg_session_length_events": sg["session_len"].mean(),
            "max_session_length": sg["session_len"].max(),
            "median_session_length": sg["session_len"].median(),
            "avg_session_duration_seconds": sg["session_dur"].mean(),
            "max_session_duration_seconds": sg["session_dur"].max(),
            "bounce_rate": sg["session_len"].apply(lambda s: float((s == 1).mean())),
            "long_session_share": sg["session_len"].apply(lambda s: float((s >= 10).mean())),
        }
    )

    active_days = events.groupby("user_id")["event_time"].apply(lambda s: s.dt.normalize().nunique())
    sessions = events.groupby("user_id")["session_id"].nunique()
    out["sessions_per_active_day"] = (sessions / active_days.replace(0, np.nan)).fillna(0.0)

    type_counts = events.groupby(["user_id", "event_type"]).size().unstack(fill_value=0)
    for et in EVENT_TYPES:
        if et not in type_counts.columns:
            type_counts[et] = 0
    out["feature_to_pageview_ratio"] = _safe_div(type_counts["feature_used"], type_counts["page_view"])
    out["search_to_purchase_ratio"] = _safe_div(type_counts["search"], type_counts["purchase"])
    out["cart_to_purchase_ratio"] = _safe_div(type_counts["add_to_cart"], type_counts["purchase"])
    out["checkout_abandonment_rate"] = _safe_div(
        (type_counts["checkout"] - type_counts["purchase"]).clip(lower=0),
        type_counts["checkout"],
    )
    return out.fillna(0.0)


def _per_feature_counts(events: pd.DataFrame) -> pd.DataFrame:
    counts = (
        events.groupby(["user_id", "feature"]).size().unstack(fill_value=0)
    )
    for f in FEATURE_NAMES:
        if f not in counts.columns:
            counts[f] = 0
    counts = counts[FEATURE_NAMES]
    counts.columns = [f"count_feature_{c}" for c in counts.columns]
    return counts


def _per_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    counts = (
        events.groupby(["user_id", "event_type"]).size().unstack(fill_value=0)
    )
    for et in EVENT_TYPES:
        if et not in counts.columns:
            counts[et] = 0
    counts = counts[EVENT_TYPES]
    counts.columns = [f"count_event_{c}" for c in counts.columns]
    return counts


def _profile_features(users: pd.DataFrame, ref: pd.Timestamp) -> pd.DataFrame:
    plan_scores = {p: i for i, p in enumerate(PLANS)}
    plan_revenue = {"free": 0, "starter": 19, "pro": 79, "enterprise": 299}
    out = users.copy()
    out["plan_score"] = out["plan"].map(plan_scores)
    out["plan_revenue_potential"] = out["plan"].map(plan_revenue)
    out["is_paid_plan"] = (out["plan"] != "free").astype(int)
    out["tenure_days"] = (ref - out["signup_date"]).dt.total_seconds() / 86400.0
    out["onboarding_completed_int"] = out["onboarding_completed"].astype(int)

    channel_q = {"organic": 0.8, "paid_search": 0.6, "referral": 0.85, "social": 0.5, "direct": 0.7, "email": 0.75}
    out["channel_quality_score"] = out["channel"].map(channel_q)

    for d in ("desktop", "mobile", "tablet"):
        out[f"is_device_{d}"] = (out["device"] == d).astype(int)

    keep = [
        "user_id",
        "plan_score",
        "plan_revenue_potential",
        "is_paid_plan",
        "tenure_days",
        "onboarding_completed_int",
        "channel_quality_score",
        "engagement_score",
        "is_device_desktop",
        "is_device_mobile",
        "is_device_tablet",
    ]
    return out[keep]


def build_features(events: pd.DataFrame, users: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Compute the full feature frame keyed on user_id."""
    window = _filter_window(events, spec.window_start, spec.reference_date)

    base = _profile_features(users, spec.reference_date)

    blocks: list[pd.DataFrame] = [base.set_index("user_id")]
    if not window.empty:
        blocks.append(_recency_features(window, users, spec.reference_date).set_index("user_id"))
        blocks.append(_windowed_frequency(window, spec.reference_date, 7))
        blocks.append(_windowed_frequency(window, spec.reference_date, 28))
        blocks.append(_monetary_features(window, spec.reference_date))
        blocks.append(_diversity_features(window))
        blocks.append(_trend_features(window, spec.reference_date))
        blocks.append(_temporal_features(window))
        blocks.append(_engagement_features(window))
        blocks.append(_per_feature_counts(window))
        blocks.append(_per_event_counts(window))

    features = blocks[0]
    for b in blocks[1:]:
        features = features.join(b, how="left")

    features = features.fillna(0.0).reset_index()
    return features


def label_churn(events: pd.DataFrame, users: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    """Label 1 if the user has zero events in [reference_date, reference_date + horizon)."""
    future = _filter_window(events, spec.reference_date, spec.horizon_end)
    active_users = set(future["user_id"].unique())
    labels = users["user_id"].apply(lambda u: 0 if u in active_users else 1)
    labels.index = users["user_id"]
    labels.name = "churned"
    return labels


def feature_columns(features: pd.DataFrame) -> Iterable[str]:
    """All numeric columns excluding the join key."""
    return [c for c in features.columns if c != "user_id"]
