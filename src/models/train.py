"""Training pipeline: split, fit, evaluate, persist artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CHURN_HORIZON_DAYS,
    FEATURES_FILE,
    METRICS_FILE,
    MODEL_FILE,
    OBSERVATION_WINDOW_DAYS,
    RANDOM_SEED,
)
from src.data.loader import load_events, load_users
from src.features.engineering import FeatureSpec, build_features, label_churn

from .evaluator import calibration_table, evaluate, lift_table
from .glm import ChurnGLM


@dataclass
class TrainResult:
    model: ChurnGLM
    features: pd.DataFrame
    labels: pd.Series
    metrics: dict
    lift: pd.DataFrame
    calibration: pd.DataFrame


def _select_reference_date(events: pd.DataFrame) -> pd.Timestamp:
    """Reference date sits one churn-horizon before the last observed event."""
    last = events["event_time"].max()
    return (last - pd.Timedelta(days=CHURN_HORIZON_DAYS)).normalize()


def train_pipeline(
    events: pd.DataFrame | None = None,
    users: pd.DataFrame | None = None,
    test_size: float = 0.25,
    alpha: float = 0.05,
    persist: bool = True,
    seed: int = RANDOM_SEED,
) -> TrainResult:
    events = events if events is not None else load_events()
    users = users if users is not None else load_users()

    ref = _select_reference_date(events)
    spec = FeatureSpec(reference_date=ref, observation_days=OBSERVATION_WINDOW_DAYS, horizon_days=CHURN_HORIZON_DAYS)

    eligible_users = users[users["signup_date"] <= spec.window_start].copy()
    features = build_features(events, eligible_users, spec)
    labels = label_churn(events, eligible_users, spec)

    X = features.drop(columns=["user_id"])
    y = labels.reindex(features["user_id"]).fillna(1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    model = ChurnGLM().fit(X_train, y_train, alpha=alpha)

    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    train_report = evaluate(y_train, train_proba)
    test_report = evaluate(y_test, test_proba)
    lift = lift_table(y_test, test_proba)
    calib = calibration_table(y_test, test_proba)

    metrics = {
        "train": train_report.to_dict(),
        "test": test_report.to_dict(),
        "feature_count": int(X.shape[1]),
        "training_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "reference_date": ref.isoformat(),
        "observation_days": OBSERVATION_WINDOW_DAYS,
        "horizon_days": CHURN_HORIZON_DAYS,
    }

    if persist:
        model.save(MODEL_FILE)
        features.to_parquet(FEATURES_FILE, index=False)
        METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str))

    return TrainResult(
        model=model,
        features=features,
        labels=labels,
        metrics=metrics,
        lift=lift,
        calibration=calib,
    )


if __name__ == "__main__":
    result = train_pipeline()
    test = result.metrics["test"]
    print(f"Test accuracy: {test['accuracy']:.4f}")
    print(f"Test ROC AUC : {test['roc_auc']:.4f}")
    print(f"Test PR AUC  : {test['pr_auc']:.4f}")
    print(f"Features used: {result.metrics['feature_count']}")
