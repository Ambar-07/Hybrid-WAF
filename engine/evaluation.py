"""
Evaluation helpers for ThreatShield predictions.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _normalize_label(label: str) -> str:
    text = str(label).strip().upper()

    if "BENIGN" in text or text in {"NORMAL", "ALLOW", "ALLOWED"}:
        return "BENIGN"
    if "SUSP" in text or "ANOM" in text or text in {"ALERT", "SUSPICIOUS"}:
        return "SUSPICIOUS"
    if (
        "MAL" in text
        or "ATTACK" in text
        or "INTRUSION" in text
        or "BLOCK" in text
        or "SQL" in text
        or "XSS" in text
    ):
        return "MALICIOUS"

    return "BENIGN"


def normalize_truth_labels(labels: Iterable[str]) -> List[str]:
    return [_normalize_label(v) for v in labels]


def actions_to_labels(actions: Iterable[str]) -> List[str]:
    mapped = []
    for action in actions:
        action_text = str(action).strip().upper()
        if action_text == "BLOCK":
            mapped.append("MALICIOUS")
        elif action_text == "ALERT":
            mapped.append("SUSPICIOUS")
        else:
            mapped.append("BENIGN")
    return mapped


def evaluate_predictions(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, float]:
    y_true_norm = normalize_truth_labels(y_true)
    y_pred_norm = normalize_truth_labels(y_pred)

    acc = accuracy_score(y_true_norm, y_pred_norm)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_norm,
        y_pred_norm,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
