"""Loaders for raw and processed datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import EVENTS_FILE, FEATURES_FILE, USERS_FILE


def load_events(path: Path | None = None) -> pd.DataFrame:
    path = path or EVENTS_FILE
    df = pd.read_parquet(path)
    df["event_time"] = pd.to_datetime(df["event_time"])
    return df


def load_users(path: Path | None = None) -> pd.DataFrame:
    path = path or USERS_FILE
    df = pd.read_parquet(path)
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


def load_features(path: Path | None = None) -> pd.DataFrame:
    path = path or FEATURES_FILE
    return pd.read_parquet(path)
