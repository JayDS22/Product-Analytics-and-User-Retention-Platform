"""Regularized logistic GLM with standardization and feature variance filtering.

statsmodels is used for the binomial GLM fit so coefficients, standard errors,
and p-values are directly available for downstream interpretation. An L2 penalty
is applied through a regularized fit when the design matrix is wide relative to
the sample size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


@dataclass
class ChurnGLM:
    feature_names: list[str] = field(default_factory=list)
    scaler: StandardScaler | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0
    feature_importance: pd.DataFrame | None = None
    alpha: float = 0.05

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names].astype(float).fillna(0.0)
        X = X.replace([np.inf, -np.inf], 0.0)
        Z = self.scaler.transform(X)
        return np.clip(Z, -10, 10)

    def fit(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.05) -> "ChurnGLM":
        self.alpha = alpha
        variance = X.var(numeric_only=True)
        keep = variance[variance > 1e-8].index.tolist()
        X = X[keep].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        self.feature_names = list(X.columns)
        self.scaler = StandardScaler()
        Z = self.scaler.fit_transform(X)
        Z = np.clip(Z, -10, 10)

        Z_design = sm.add_constant(Z, has_constant="add")
        model = sm.GLM(y.values.astype(float), Z_design, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=alpha, L1_wt=0.0, refit=False)

        params = np.asarray(result.params)
        self.intercept = float(params[0])
        self.coefficients = params[1:]

        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": self.coefficients,
                "abs_coefficient": np.abs(self.coefficients),
            }
        ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
        self.feature_importance = importance
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._prepare(X)
        logits = self.intercept + Z @ self.coefficients
        logits = np.clip(logits, -30, 30)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ChurnGLM":
        return joblib.load(path)
