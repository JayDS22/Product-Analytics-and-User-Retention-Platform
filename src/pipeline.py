"""End-to-end orchestration: generate -> features -> train -> persist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import EVENTS_FILE, METRICS_FILE, USERS_FILE
from src.data.synthetic_generator import generate_dataset
from src.models.train import train_pipeline


def run(n_users: int = 8000, regenerate: bool = False) -> dict:
    if regenerate or not (EVENTS_FILE.exists() and USERS_FILE.exists()):
        print(f"[pipeline] generating synthetic dataset for {n_users:,} users")
        users, events = generate_dataset(n_users=n_users)
    else:
        print(f"[pipeline] reusing cached dataset at {EVENTS_FILE.parent}")

    print("[pipeline] training churn GLM")
    result = train_pipeline()
    metrics = result.metrics
    print(f"[pipeline] feature_count={metrics['feature_count']} "
          f"test_accuracy={metrics['test']['accuracy']:.4f} "
          f"test_auc={metrics['test']['roc_auc']:.4f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retention-platform pipeline.")
    parser.add_argument("--n-users", type=int, default=8000)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic data")
    args = parser.parse_args()
    run(n_users=args.n_users, regenerate=args.regenerate)


if __name__ == "__main__":
    main()
