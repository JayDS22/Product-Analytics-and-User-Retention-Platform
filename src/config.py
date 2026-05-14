"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EVENTS_FILE = RAW_DIR / "events.parquet"
USERS_FILE = RAW_DIR / "users.parquet"
FEATURES_FILE = PROCESSED_DIR / "features.parquet"
MODEL_FILE = ARTIFACTS_DIR / "glm_model.joblib"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"

# Observation window for feature aggregation, in days.
OBSERVATION_WINDOW_DAYS = 56
# Prediction horizon used to label churn (no activity within N days).
CHURN_HORIZON_DAYS = 28
# Random seed for reproducibility across data generation and modeling.
RANDOM_SEED = 42

EVENT_TYPES = [
    "session_start",
    "page_view",
    "feature_used",
    "search",
    "add_to_cart",
    "checkout",
    "purchase",
    "support_ticket",
    "share",
    "review",
]

FEATURE_NAMES = [
    "dashboard",
    "reports",
    "exports",
    "integrations",
    "api",
    "billing",
    "team",
    "notifications",
    "settings",
    "help_center",
]

PLANS = ["free", "starter", "pro", "enterprise"]
CHANNELS = ["organic", "paid_search", "referral", "social", "direct", "email"]
DEVICES = ["desktop", "mobile", "tablet"]
COUNTRIES = ["US", "GB", "DE", "FR", "IN", "BR", "JP", "AU", "CA", "SG"]
