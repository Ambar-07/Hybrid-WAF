"""
Train a dashboard-ready CICIDS2017 Random Forest pipeline.

The saved artifact contains the selected feature list, imputer, scaler, and
Random Forest model together, so dashboard inference uses the same preprocessing
that training used.

Usage:
  python3 train_cicids_pipeline.py --data /path/to/CICIDS2017_combined.csv
"""

import argparse
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from engine.feature_extractor import CICIDS_COLUMN_MAP, FeatureExtractor, SELECTED_FEATURES


DEFAULT_MODEL_OUT = "models/cicids_rf_pipeline.pkl"

LABEL_MAP: Dict[str, int] = {
    "BENIGN": 0,
    "DDOS": 1,
    "DOS HULK": 1,
    "DOS GOLDENEYE": 1,
    "DOS SLOWLORIS": 1,
    "DOS SLOWHTTPTEST": 1,
    "HEARTBLEED": 1,
    "PORTSCAN": 2,
    "FTP-PATATOR": 3,
    "SSH-PATATOR": 3,
    "BOT": 4,
    "WEB ATTACK BRUTE FORCE": 5,
    "WEB ATTACK XSS": 5,
    "WEB ATTACK SQL INJECTION": 5,
    "INFILTRATION": 6,
}

CLASS_NAMES = {
    0: "Benign",
    1: "DDoS/DoS",
    2: "PortScan",
    3: "BruteForce",
    4: "Botnet",
    5: "WebAttack",
    6: "Infiltration",
}


def normalize_label(value: str) -> str:
    text = str(value).strip().upper()
    for bad in ["\u0096", "\ufffd", "\u2013", "\u2014"]:
        text = text.replace(bad, " ")
    return " ".join(text.replace("-", " ").split())


def encode_labels(labels: pd.Series) -> np.ndarray:
    return np.array([LABEL_MAP.get(normalize_label(label), 0) for label in labels], dtype=int)


def read_training_csv(path: str) -> pd.DataFrame:
    selected_cicids_cols = [
        source
        for source, target in CICIDS_COLUMN_MAP.items()
        if target in SELECTED_FEATURES or target == "label"
    ]
    usecols = lambda col: str(col).strip() in selected_cicids_cols or str(col).strip().lower() == "label"
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def cap_rows_per_class(df: pd.DataFrame, y: np.ndarray, max_per_class: int) -> tuple[pd.DataFrame, np.ndarray]:
    if max_per_class <= 0:
        return df, y

    rng = np.random.default_rng(42)
    selected = []
    for cls in sorted(set(y)):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        selected.append(idx)

    keep = np.concatenate(selected)
    rng.shuffle(keep)
    return df.iloc[keep].reset_index(drop=True), y[keep]


def main():
    parser = argparse.ArgumentParser(description="Train CICIDS2017 RF preprocessing pipeline")
    parser.add_argument("--data", required=True, help="CICIDS2017 combined CSV")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT, help="Output artifact path")
    parser.add_argument("--trees", type=int, default=300, help="Number of Random Forest trees")
    parser.add_argument("--max-per-class", type=int, default=250000, help="Cap large classes before training")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split")
    args = parser.parse_args()

    raw_df = read_training_csv(args.data)
    extractor = FeatureExtractor()
    prepared = extractor._prepare_df(raw_df)

    label_col = next((col for col in prepared.columns if str(col).lower() == "label"), None)
    if label_col is None:
        raise ValueError("Label column not found.")

    missing = [feature for feature in SELECTED_FEATURES if feature not in prepared.columns]
    if missing:
        raise ValueError(f"Missing selected feature columns after preparation: {missing}")

    y = encode_labels(prepared[label_col])
    prepared, y = cap_rows_per_class(prepared, y, args.max_per_class)
    X = prepared[SELECTED_FEATURES].replace([np.inf, -np.inf], np.nan)

    print("\nClass distribution:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {cls} [{CLASS_NAMES.get(int(cls), '?'):<12}] {count:>8,}")

    stratify = y if len(set(y)) > 1 and min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=args.trees,
            class_weight="balanced",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    labels = sorted(set(y_test) | set(preds))
    print("\nValidation report:")
    print(classification_report(
        y_test,
        preds,
        labels=labels,
        target_names=[CLASS_NAMES.get(label, str(label)) for label in labels],
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds, labels=labels))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump({
            "model": pipeline,
            "model_kind": "cicids_rf_pipeline",
            "expects_raw_features": True,
            "feature_names": list(SELECTED_FEATURES),
            "threshold": 0.35,
            "class_names": CLASS_NAMES,
            "normal_class_label": 0,
        }, f)

    print(f"\nSaved CICIDS RF pipeline to {args.model_out}")


if __name__ == "__main__":
    main()
