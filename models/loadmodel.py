"""
Load Model Utility
-------------------
Loads the trained ML detector model for the Hybrid IDS.
This module can be imported by other scripts or run standalone to verify
that a trained model exists and is loadable.

Usage:
  from models.loadmodel import load_ids_model
  detector = load_ids_model()
"""

import os
import sys

# Ensure project root is on the path so engine imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engine.ml_detector import MLDetector

MODEL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "models", "best_model.pkl"),
    os.path.join(PROJECT_ROOT, "models", "cicids_rf_pipeline.pkl"),
    os.path.join(PROJECT_ROOT, "models", "wireshark_random_forest.pkl"),
    os.path.join(PROJECT_ROOT, "models", "isolation_forest.pkl"),
]


def load_ids_model(model_path: str = None) -> MLDetector:
    """
    Load a trained MLDetector.

    Parameters
    ----------
    model_path : str, optional
        Path to the saved .pkl model file.
        Defaults to the first available model in the project model priority list.

    Returns
    -------
    MLDetector
        A trained MLDetector instance ready for prediction.
    """
    path = model_path or next((candidate for candidate in MODEL_CANDIDATES if os.path.exists(candidate)), None)

    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "No trained model found.\n"
            "Please train first:\n"
            "  • CLI:       python main.py --train <your_training.csv>\n"
            "  • Dashboard: streamlit run ui/dashboard.py  →  ⚙️ Train Model"
        )

    detector = MLDetector()
    detector.load(path)
    print(f"[loadmodel] ✅ ML detector model loaded from {path}")
    print(f"[loadmodel]    Attack threshold: {detector.threshold}")
    return detector


# ── Run standalone to verify ──────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        detector = load_ids_model()
        print("[loadmodel] Model is ready for predictions.")
    except FileNotFoundError as e:
        print(f"[loadmodel] ⚠️  {e}")
