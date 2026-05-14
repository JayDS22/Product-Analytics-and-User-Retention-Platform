"""Model evaluation: classification report, lift, calibration."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    log_loss: float
    brier: float
    positive_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(y_true, y_proba, threshold: float = 0.5) -> EvalReport:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    return EvalReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        pr_auc=float(average_precision_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        log_loss=float(log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6))),
        brier=float(brier_score_loss(y_true, y_proba)),
        positive_rate=float(np.mean(y_true)),
    )


def lift_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Decile lift table sorted by descending predicted probability."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    proba_sorted = y_proba[order]

    base_rate = y_true.mean() if y_true.mean() > 0 else np.nan
    chunks = np.array_split(np.arange(len(y_true)), n_bins)
    rows = []
    for i, idx in enumerate(chunks):
        actual = y_sorted[idx].mean()
        rows.append(
            {
                "decile": i + 1,
                "n": int(len(idx)),
                "avg_predicted": float(proba_sorted[idx].mean()),
                "observed_rate": float(actual),
                "lift": float(actual / base_rate) if base_rate else 0.0,
                "cumulative_positives": int(y_sorted[: idx.max() + 1].sum()),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y_true, y_proba, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: predicted vs observed by bucket."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_proba, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "predicted": float(y_proba[mask].mean()),
                "observed": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)
