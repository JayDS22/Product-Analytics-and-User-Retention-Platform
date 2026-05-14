import pandas as pd

from src.config import EVENT_TYPES, PLANS


def test_user_table_shape(users_small):
    assert len(users_small) == 400
    assert set(["user_id", "signup_date", "plan", "channel", "country", "device"]).issubset(users_small.columns)
    assert users_small["user_id"].is_unique
    assert set(users_small["plan"]).issubset(set(PLANS))


def test_events_after_signup(users_small, events_small):
    signups = users_small.set_index("user_id")["signup_date"]
    sample = events_small.head(5000)
    joined = sample.join(signups.rename("signup_date"), on="user_id")
    assert (joined["event_time"] >= joined["signup_date"]).all()


def test_event_types_within_catalog(events_small):
    assert set(events_small["event_type"]).issubset(set(EVENT_TYPES))


def test_revenue_only_on_purchases(events_small):
    non_purchase = events_small[events_small["event_type"] != "purchase"]
    assert (non_purchase["revenue"] == 0).all()
