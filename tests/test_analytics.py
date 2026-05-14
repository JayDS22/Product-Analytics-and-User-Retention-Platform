import pandas as pd

from src.analytics.cohort import cohort_matrix, cohort_revenue
from src.analytics.funnel import funnel_conversion, journey_paths
from src.analytics.metrics import dau, mau, wau
from src.analytics.retention import retention_curve


def test_dau_monotonic_dates(events_small):
    df = dau(events_small)
    assert df["dau"].min() >= 0
    assert df["date"].is_monotonic_increasing


def test_wau_mau_have_data(events_small):
    assert len(wau(events_small)) > 0
    assert len(mau(events_small)) > 0


def test_retention_curve_decreases(events_small, users_small):
    curve = retention_curve(events_small, users_small, max_day=30)
    assert curve.loc[curve["day"] == 0, "retained_rate"].iloc[0] >= curve.loc[curve["day"] == 30, "retained_rate"].iloc[0]


def test_funnel_non_increasing(events_small):
    steps = ["session_start", "page_view", "add_to_cart", "checkout", "purchase"]
    fn = funnel_conversion(events_small, steps)
    assert (fn["users"].diff().dropna() <= 0).all()


def test_journey_paths_non_empty(events_small):
    paths = journey_paths(events_small, max_steps=4, top_n=10)
    assert len(paths) > 0
    assert "users" in paths.columns


def test_cohort_matrix_rates_in_range(events_small, users_small):
    m = cohort_matrix(events_small, users_small, max_period=8, freq="W")
    assert ((m >= 0) & (m <= 1)).all().all()
