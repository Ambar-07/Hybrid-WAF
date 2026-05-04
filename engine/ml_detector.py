"""
ML Detector
-----------
Uses a Random Forest classifier when labeled traffic is available.
The output is normalized into an attack/anomaly score so the fusion layer can
use the same contract for supervised and legacy unsupervised models.

Output: anomaly_score (0.0 = normal, 1.0 = highly anomalous)
"""

import numpy as np
import pandas as pd
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from engine.feature_extractor import FlowFeatures


@dataclass
class MLResult:
    anomaly_score: float    # 0.0 (normal) → 1.0 (anomaly)
    is_anomaly: bool
    raw_score: float        # model score before fusion
    confidence: float       # model certainty for the predicted class
    predicted_class: Optional[object] = None
    predicted_label: str = ""
    class_probabilities: Optional[Dict[str, float]] = None


class MLDetector:
    """
    Usage
    -----
    detector = MLDetector()
    detector.train(flows, labels)           # supervised Random Forest
    result = detector.predict(flow)         # single FlowFeatures
    results = detector.predict_batch(flows) # list of FlowFeatures

    detector.save("models/wireshark_random_forest.pkl")
    detector.load("models/wireshark_random_forest.pkl")
    """

    def __init__(
        self,
        contamination: float = 0.05,   # expected % of outliers in training data
        n_estimators: int = 100,
        threshold: float = 0.55,       # anomaly_score threshold to flag
        random_state: int = 42,
        normal_class_label=0,
    ):
        self.contamination = contamination
        self.n_estimators   = n_estimators
        self.threshold      = threshold
        self.random_state   = random_state
        self.normal_class_label = normal_class_label
        self.model = None
        self.trained = False
        self.feature_state = None
        self.feature_names = None
        self.expects_raw_features = False
        self.class_names = {
            0: "Benign",
            1: "DDoS/DoS",
            2: "PortScan",
            3: "BruteForce",
            4: "Botnet",
            5: "WebAttack",
            6: "Infiltration",
        }

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, flows: List[FlowFeatures], labels: Optional[List[object]] = None) -> "MLDetector":
        """Train a supervised Random Forest from labeled flows."""
        if labels is None:
            labels = [flow.label for flow in flows]

        X = np.array([f.normalized for f in flows])
        y = np.array([self._encode_label(label) for label in labels], dtype=int)

        if len(X) == 0:
            raise ValueError("Cannot train Random Forest with no flows.")
        if len(set(y)) < 2:
            raise ValueError("Random Forest training needs at least two classes, e.g. BENIGN and ATTACK.")

        print(f"[MLDetector] Training Random Forest on {len(X)} labeled flows...")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        )
        self.model.fit(X, y)
        self.trained = True
        self.threshold = 0.35
        print("[MLDetector] Training complete.")
        return self

    def train_isolation_forest(self, normal_flows: List[FlowFeatures]) -> "MLDetector":
        """Legacy fallback for benign-only datasets."""
        X = np.array([f.normalized for f in normal_flows])
        print(f"[MLDetector] Training Isolation Forest on {len(X)} normal flows...")

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X)
        self.trained = True
        print("[MLDetector] Training complete.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, ff: FlowFeatures) -> MLResult:
        """Score a single flow."""
        self._check_trained()
        if self.expects_raw_features:
            X_raw = self._raw_feature_frame([ff])
            if hasattr(self.model, "predict_proba"):
                return self._proba_result(self.model.predict_proba(X_raw)[0])

        X = ff.normalized.reshape(1, -1)

        if hasattr(self.model, "predict_proba"):
            return self._proba_result(self.model.predict_proba(X)[0])

        anomaly_score, raw_score = self._score(X)
        is_anomaly = anomaly_score >= self.threshold

        return MLResult(
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            raw_score=raw_score,
            confidence=anomaly_score,
            predicted_label="Anomaly" if is_anomaly else "Benign",
        )

    def predict_batch(self, flows: List[FlowFeatures]) -> List[MLResult]:
        """Score multiple flows efficiently."""
        self._check_trained()
        if self.expects_raw_features:
            X_raw = self._raw_feature_frame(flows)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_raw)
                return [
                    self._proba_result(p)
                    for p in proba
                ]

        X = np.array([f.normalized for f in flows])

        if hasattr(self.model, "score_samples"):
            raw_scores = self.model.score_samples(X)
            return [
                MLResult(
                    anomaly_score=(score := self._scale_score(float(s))),
                    is_anomaly=score >= self.threshold,
                    raw_score=float(s),
                    confidence=score,
                    predicted_label="Anomaly" if score >= self.threshold else "Benign",
                )
                for s in raw_scores
            ]

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            return [
                self._proba_result(p)
                for p in proba
            ]

        return [
            MLResult(
                anomaly_score=0.0,
                is_anomaly=False,
                raw_score=0.0,
                confidence=0.0,
            )
            for _ in flows
        ]

    def _score(self, X: np.ndarray) -> tuple[float, float]:
        if hasattr(self.model, "score_samples"):
            raw_score = float(self.model.score_samples(X)[0])
            return self._scale_score(raw_score), raw_score

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            score = self._proba_anomaly_score(proba)
            return score, score

        raise RuntimeError("Unsupported model type for scoring.")

    def _proba_anomaly_score(self, proba: np.ndarray) -> float:
        """For supervised models: anomaly_score = 1 - P(normal class). Falls back to max(proba) if no normal class."""
        classes = getattr(self.model, "classes_", None)
        if classes is not None and len(classes) > 1:
            try:
                normal_index = list(classes).index(self.normal_class_label)
                normal_proba = float(proba[normal_index])
                return round(1.0 - normal_proba, 4)
            except ValueError:
                pass
        # Fallback: highest non-normal proba or max
        return float(np.max(proba[1:] if len(proba) > 1 else proba))

    def _proba_result(self, proba: np.ndarray) -> MLResult:
        score = self._proba_anomaly_score(proba)
        classes = list(getattr(self.model, "classes_", range(len(proba))))
        predicted_index = int(np.argmax(proba))
        predicted_class = classes[predicted_index]
        confidence = float(proba[predicted_index])
        predicted_label = self._class_label(predicted_class)
        class_probabilities = {
            self._class_label(cls): round(float(prob), 4)
            for cls, prob in zip(classes, proba)
        }
        return MLResult(
            anomaly_score=score,
            is_anomaly=predicted_class != self.normal_class_label or score >= self.threshold,
            raw_score=score,
            confidence=round(confidence, 4),
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            class_probabilities=class_probabilities,
        )

    def _class_label(self, cls) -> str:
        return self.class_names.get(cls, str(cls))

    def _get_anomaly_class_index(self) -> int:
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            return 1 if self.model.n_classes_ > 1 else 0

        classes = list(classes)
        if 1 in classes:
            return classes.index(1)
        if "anomaly" in classes:
            return classes.index("anomaly")
        return len(classes) - 1

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = "models/wireshark_random_forest.pkl", feature_state=None, feature_names=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.threshold,
                "feature_state": feature_state,
                "feature_names": feature_names,
                "class_names": self.class_names,
                "normal_class_label": self.normal_class_label,
            }, f)
        print(f"[MLDetector] Model saved to {path}")

    def load(self, path: str = "models/wireshark_random_forest.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self.threshold = data.get("threshold", self.threshold)
            self.feature_state = data.get("feature_state")
            self.feature_names = data.get("feature_names")
            self.expects_raw_features = bool(data.get("expects_raw_features", False))
            self.class_names.update(data.get("class_names", {}))
            self.normal_class_label = data.get("normal_class_label", self.normal_class_label)
        else:
            self.model = data

        self.trained = True
        print(f"[MLDetector] Model loaded from {path}")
        return self

    # ── Helpers ───────────────────────────────────────────────────────────
    def _scale_score(self, raw: float) -> float:
        """Map raw IF score to [0,1] where 1=anomaly."""
        # IF scores roughly range from -0.6 to +0.1
        normalized = (raw + 0.6) / 0.7          # shift to ~[0, 1]
        normalized = float(np.clip(normalized, 0, 1))
        return round(1.0 - normalized, 4)        # invert: high = anomaly

    def _check_trained(self):
        if not self.trained or self.model is None:
            raise RuntimeError("MLDetector is not trained. Call .train() or .load() first.")

    def _raw_feature_frame(self, flows: List[FlowFeatures]) -> pd.DataFrame:
        feature_names = self.feature_names
        if not feature_names:
            raise RuntimeError("Raw-feature model is missing feature_names metadata.")

        rows = []
        for flow in flows:
            rows.append({
                feature: self._to_float(flow.raw.get(feature, 0))
                for feature in feature_names
            })
        return pd.DataFrame(rows, columns=feature_names)

    def _to_float(self, value) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _encode_label(self, value) -> int:
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)

        key = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        label_map = {
            "BENIGN": 0,
            "NORMAL": 0,
            "LEGIT": 0,
            "ATTACK": 1,
            "ANOMALY": 1,
            "MALICIOUS": 1,
            "DDOS": 1,
            "DOS": 1,
            "SYN_FLOOD": 1,
            "PORTSCAN": 2,
            "PORT_SCAN": 2,
            "SCAN": 2,
            "BRUTEFORCE": 3,
            "BRUTE_FORCE": 3,
            "SSH_BRUTE_FORCE": 3,
            "FTP_BRUTE_FORCE": 3,
            "BOTNET": 4,
            "BOT": 4,
            "WEBATTACK": 5,
            "WEB_ATTACK": 5,
            "XSS": 5,
            "SQLI": 5,
            "SQL_INJECTION": 5,
            "INFILTRATION": 6,
        }
        return label_map.get(key, 1)
