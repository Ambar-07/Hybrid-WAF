"""
Train a dashboard-compatible ML model from labeled Wireshark/flow CSV data.

Examples
--------
  python3 train_wireshark_model.py --data training_flows.csv
  python3 train_wireshark_model.py --data wireshark_packets.csv --model-out models/wireshark_random_forest.pkl

Input requirements
------------------
The CSV must contain a label/class column. For Wireshark packet exports, include
columns such as ip.src, ip.dst, tcp.srcport, tcp.dstport, frame.time_relative,
frame.len, ip.proto, tcp.flags.* or Info. The existing FeatureExtractor will
aggregate packets into bidirectional flows before training.
"""

import argparse
import os
import pickle
import re
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from engine.feature_extractor import CICIDS_COLUMN_MAP, FeatureExtractor, RAW_FEATURES, SELECTED_FEATURES


DEFAULT_MODEL_OUT = "models/wireshark_random_forest.pkl"

LABEL_MAP: Dict[str, int] = {
    "BENIGN": 0,
    "NORMAL": 0,
    "LEGIT": 0,
    "ATTACK": 1,
    "ANOMALY": 1,
    "MALICIOUS": 1,
    "DDOS": 1,
    "DOS": 1,
    "DOS_HULK": 1,
    "DOS_GOLDENEYE": 1,
    "DOS_SLOWLORIS": 1,
    "DOS_SLOWHTTPTEST": 1,
    "HEARTBLEED": 1,
    "SYN_FLOOD": 1,
    "PORTSCAN": 2,
    "PORT_SCAN": 2,
    "SCAN": 2,
    "BRUTEFORCE": 3,
    "BRUTE_FORCE": 3,
    "FTP_PATATOR": 3,
    "SSH_PATATOR": 3,
    "SSH_BRUTE_FORCE": 3,
    "FTP_BRUTE_FORCE": 3,
    "BOTNET": 4,
    "BOT": 4,
    "WEBATTACK": 5,
    "WEB_ATTACK": 5,
    "WEB_ATTACK_BRUTE_FORCE": 5,
    "WEB_ATTACK_XSS": 5,
    "WEB_ATTACK_SQL_INJECTION": 5,
    "XSS": 5,
    "SQLI": 5,
    "SQL_INJECTION": 5,
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


def find_label_col(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() in {"label", "class", "target", "attack", "attack_type"}:
            return col
    raise ValueError("No label column found. Add a Label column with BENIGN/ATTACK/etc.")


def encode_labels(labels: pd.Series) -> np.ndarray:
    encoded = []
    for value in labels.astype(str):
        key = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
        encoded.append(LABEL_MAP.get(key, 0))
    return np.array(encoded, dtype=int)


def read_training_csv(path: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    label_col = next(
        (col for col in header if str(col).strip().lower() in {"label", "class", "target", "attack", "attack_type"}),
        None,
    )
    if label_col is None:
        raise ValueError("No label column found. Add a Label column with BENIGN/ATTACK/etc.")

    needed_targets = set(RAW_FEATURES) | set(SELECTED_FEATURES)
    usecols = [label_col]
    for col in header:
        target = CICIDS_COLUMN_MAP.get(str(col).strip())
        if target in needed_targets:
            usecols.append(col)

    if len(usecols) > 1:
        return pd.read_csv(path, usecols=sorted(set(usecols)), low_memory=False)

    return pd.read_csv(path, low_memory=False)


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


def upsample_minority(X: np.ndarray, y: np.ndarray, min_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    if min_per_class <= 0:
        return X, y

    rng = np.random.default_rng(42)
    X_parts = [X]
    y_parts = [y]
    for cls in sorted(set(y)):
        idx = np.flatnonzero(y == cls)
        if len(idx) == 0 or len(idx) >= min_per_class:
            continue
        extra = rng.choice(idx, size=min_per_class - len(idx), replace=True)
        X_parts.append(X[extra])
        y_parts.append(y[extra])

    X_bal = np.concatenate(X_parts)
    y_bal = np.concatenate(y_parts)
    order = rng.permutation(len(y_bal))
    return X_bal[order], y_bal[order]


def main():
    parser = argparse.ArgumentParser(description="Train Wireshark-compatible Random Forest model")
    parser.add_argument("--data", required=True, help="Labeled Wireshark packet CSV or flow CSV")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_OUT, help="Output model artifact path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--trees", type=int, default=300, help="Number of Random Forest trees")
    parser.add_argument("--max-per-class", type=int, default=250000, help="Cap very large classes before training")
    parser.add_argument("--min-train-per-class", type=int, default=25000, help="Oversample small classes in train split")
    args = parser.parse_args()

    df = read_training_csv(args.data)
    label_col = find_label_col(df)

    extractor = FeatureExtractor()
    prepared = extractor._prepare_df(df)
    if label_col not in prepared.columns:
        prepared["label"] = df[label_col].astype(str).values[: len(prepared)]

    y = encode_labels(prepared["label"])
    prepared, y = cap_rows_per_class(prepared, y, args.max_per_class)
    print("\nClass distribution after mapping/capping:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {cls} [{CLASS_NAMES.get(int(cls), '?'):<12}] {count:>8,}")

    extractor.fit(prepared)
    flows = extractor.transform_df(prepared)
    X = np.array([flow.normalized for flow in flows])

    stratify = y if len(set(y)) > 1 and min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )
    X_train, y_train = upsample_minority(X_train, y_train, args.min_train_per_class)
    print("\nTraining distribution after minority upsampling:")
    for cls, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"  {cls} [{CLASS_NAMES.get(int(cls), '?'):<12}] {count:>8,}")

    model = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
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
            "model": model,
            "threshold": 0.35,
            "feature_state": extractor.get_state(),
            "feature_names": list(SELECTED_FEATURES),
            "class_names": CLASS_NAMES,
            "normal_class_label": 0,
            "source": "train_wireshark_model.py",
        }, f)

    print(f"\nSaved model: {args.model_out}")
    print("Use this same path in main.py and ui/dashboard.py.")


if __name__ == "__main__":
    main()
