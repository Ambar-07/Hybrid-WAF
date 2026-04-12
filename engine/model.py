"""
Simple Isolation Forest wrapper for project compatibility.

predict(features) returns 1 for normal and -1 for anomaly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestModel:
    def __init__(self, contamination: float = 0.05, n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, features: np.ndarray) -> "IsolationForestModel":
        self.model.fit(features)
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("IsolationForestModel is not fitted. Call fit() first.")
        return self.model.predict(features)

    def predict_one(self, feature_vector: Iterable[float]) -> int:
        arr = np.array([list(feature_vector)], dtype=np.float32)
        return int(self.predict(arr)[0])
