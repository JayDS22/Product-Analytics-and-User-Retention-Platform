import numpy as np
import pandas as pd

from src.features.engineering import FeatureSpec, build_features, label_churn


def _spec(events: pd.DataFrame) -> FeatureSpec:
    ref = events["event_time"].max().normalize() - pd.Timedelta(days=14)
    return FeatureSpec(reference_date=ref, observation_days=56, horizon_days=14)


def test_feature_count_exceeds_120(users_small, events_small):
    spec = _spec(events_small)
    eligible = users_small[users_small["signup_date"] <= spec.window_start]
    feats = build_features(events_small, eligible, spec)
    n_features = feats.shape[1] - 1
    assert n_features >= 120, f"expected 120+ features, got {n_features}"


def test_features_have_no_nans(users_small, events_small):
    spec = _spec(events_small)
    eligible = users_small[users_small["signup_date"] <= spec.window_start]
    feats = build_features(events_small, eligible, spec)
    assert not feats.isna().any().any()


def test_label_distribution(users_small, events_small):
    spec = _spec(events_small)
    eligible = users_small[users_small["signup_date"] <= spec.window_start]
    labels = label_churn(events_small, eligible, spec)
    assert labels.isin([0, 1]).all()
    rate = labels.mean()
    assert 0.05 < rate < 0.95


def test_recency_within_window(users_small, events_small):
    spec = _spec(events_small)
    eligible = users_small[users_small["signup_date"] <= spec.window_start]
    feats = build_features(events_small, eligible, spec)
    assert (feats["days_since_last_event"] >= 0).all()
