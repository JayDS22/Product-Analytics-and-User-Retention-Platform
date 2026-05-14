"""Shared fixtures: a small synthetic dataset used across tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import generate_users, generate_events


@pytest.fixture(scope="session")
def users_small():
    return generate_users(n_users=400, start_date="2025-01-01", end_date="2025-06-30", seed=7)


@pytest.fixture(scope="session")
def events_small(users_small):
    return generate_events(users_small, end_date="2025-06-30", seed=7)
