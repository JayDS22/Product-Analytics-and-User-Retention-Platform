import numpy as np
import pandas as pd

from src.features.engineering import FeatureSpec, build_features, label_churn
from src.models.evaluator import evaluate, lift_table
from src.models.glm import ChurnGLM


def _build(users_small, events_small):
    ref = events_small["event_time"].max().normalize() - pd.Timedelta(days=14)
    spec = FeatureSpec(reference_date=ref, observation_days=56, horizon_days=14)
    eligible = users_small[users_small["signup_date"] <= spec.window_start]
    feats = build_features(events_small, eligible, spec)
    labels = label_churn(events_small, eligible, spec).reindex(feats["user_id"]).fillna(1).astype(int)
    return feats.drop(columns=["user_id"]), labels


def test_model_fits_and_predicts(users_small, events_small):
    X, y = _build(users_small, events_small)
    model = ChurnGLM().fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_model_better_than_random(users_small, events_small):
    X, y = _build(users_small, events_small)
    model = ChurnGLM().fit(X, y)
    proba = model.predict_proba(X)
    rpt = evaluate(y, proba)
    assert rpt.roc_auc > 0.70


def test_lift_table_sorted_descending(users_small, events_small):
    X, y = _build(users_small, events_small)
    model = ChurnGLM().fit(X, y)
    proba = model.predict_proba(X)
    lift = lift_table(y, proba, n_bins=10)
    assert lift["lift"].iloc[0] >= lift["lift"].iloc[-1]
