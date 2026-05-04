"""
Attack specialist model
-----------------------
Secondary model trained on attack-only log/packet CSVs. It does not decide
benign vs attack by itself; it classifies known attack families and exposes a
confidence score that can boost weak main-model detections.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ATTACK_MODEL_PATH = "models/attack_specialist.pkl"
ATTACK_LABEL_COLUMN = "Attack Type"


@dataclass
class AttackSpecialistResult:
    attack_type: str = "Unknown"
    confidence: float = 0.0
    class_probabilities: Optional[Dict[str, float]] = None


class AttackSpecialist:
    def __init__(self):
        self.model = None
        self.trained = False

    def train(self, df: pd.DataFrame) -> "AttackSpecialist":
        if ATTACK_LABEL_COLUMN not in df.columns:
            raise ValueError(f"Missing required label column: {ATTACK_LABEL_COLUMN}")

        df = self._prepare_df(df)
        y = df[ATTACK_LABEL_COLUMN].astype(str)
        X = df.drop(columns=[ATTACK_LABEL_COLUMN])

        if y.nunique() < 2:
            raise ValueError("Attack specialist needs at least two attack types.")

        numeric_features = [
            "Source Port",
            "Destination Port",
            "Packet Length",
            "Anomaly Scores",
        ]
        categorical_features = [
            "Protocol",
            "Packet Type",
            "Traffic Type",
            "Malware Indicators",
            "Alerts/Warnings",
            "Attack Signature",
            "Action Taken",
            "Severity Level",
            "Network Segment",
            "Log Source",
            "IDS/IPS Alerts",
        ]
        text_features = [
            "Payload Data",
            "Firewall Logs",
        ]

        numeric_features = [col for col in numeric_features if col in X.columns]
        categorical_features = [col for col in categorical_features if col in X.columns]
        text_features = [col for col in text_features if col in X.columns]

        transformers = []
        if numeric_features:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ))
        if categorical_features:
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ))
        for col in text_features:
            transformers.append((
                f"text_{self._safe_name(col)}",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                    ("flatten", _TextFlattener()),
                    ("tfidf", TfidfVectorizer(max_features=1500, ngram_range=(1, 2))),
                ]),
                [col],
            ))

        if not transformers:
            raise ValueError("No usable columns found for attack specialist training.")

        self.model = Pipeline([
            ("features", ColumnTransformer(transformers=transformers, remainder="drop")),
            ("classifier", RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict_row(self, row: Dict) -> AttackSpecialistResult:
        if not self.trained or self.model is None:
            return AttackSpecialistResult()

        df = self._prepare_prediction_df(pd.DataFrame([row]))
        probabilities = self.model.predict_proba(df)[0]
        classes = list(self.model.named_steps["classifier"].classes_)
        best_idx = int(probabilities.argmax())
        class_probabilities = {
            str(cls): round(float(prob), 4)
            for cls, prob in zip(classes, probabilities)
        }
        return AttackSpecialistResult(
            attack_type=str(classes[best_idx]),
            confidence=round(float(probabilities[best_idx]), 4),
            class_probabilities=class_probabilities,
        )

    def save(self, path: str = DEFAULT_ATTACK_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model}, f)

    def load(self, path: str = DEFAULT_ATTACK_MODEL_PATH) -> "AttackSpecialist":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"] if isinstance(data, dict) else data
        self.trained = True
        return self

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("").astype(str)
        return df

    def _prepare_prediction_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._prepare_df(df)
        alias_map = {
            "Source Port": "src_port",
            "Destination Port": "dst_port",
            "Protocol": "protocol_type",
            "Packet Length": "avg_pkt_size",
            "Anomaly Scores": "anomaly_score",
            "Payload Data": "payload_data",
            "Firewall Logs": "firewall_logs",
        }

        for expected, internal in alias_map.items():
            if expected not in df.columns and internal in df.columns:
                df[expected] = df[internal]

        expected_columns = self._expected_columns()
        numeric_defaults = {
            "Source Port",
            "Destination Port",
            "Packet Length",
            "Anomaly Scores",
        }
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0 if col in numeric_defaults else ""

        return df

    def _safe_name(self, value: str) -> str:
        return value.lower().replace("/", "_").replace(" ", "_")

    def _expected_columns(self) -> list[str]:
        features = self.model.named_steps.get("features") if hasattr(self.model, "named_steps") else None
        if features is None:
            return []
        return list(getattr(features, "feature_names_in_", []))


class _TextFlattener:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, 0].fillna("").astype(str).to_numpy()
        return pd.Series(X.ravel()).fillna("").astype(str).to_numpy()
