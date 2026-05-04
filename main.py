"""
Hybrid IDS — Main Pipeline
--------------------------
Ties together:  FeatureExtractor → RuleEngine → MLDetector → FusionLayer

Usage
-----
  python main.py --train data/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
  python main.py --analyze data/cicids2017/test.csv
"""

import argparse
import pandas as pd
import sys
import os
from engine.feature_extractor import FeatureExtractor
from engine.rule_engine import RuleEngine
from engine.ml_detector import MLDetector
from engine.fusion import FusionLayer, FusionDecision

MODEL_PATH = "models/best_model.pkl"
CICIDS_PIPELINE_MODEL_PATH = "models/cicids_rf_pipeline.pkl"
TRAINED_MODEL_PATH = "models/wireshark_random_forest.pkl"
RULES_PATH = "config/rules.yaml"
MODEL_CANDIDATES = [
    MODEL_PATH,
    CICIDS_PIPELINE_MODEL_PATH,
    TRAINED_MODEL_PATH,
    "models/isolation_forest.pkl",
]


class HybridIDS:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.rule_engine = RuleEngine(rules_path=RULES_PATH)
        self.ml_detector = MLDetector()
        self.fusion = FusionLayer()

    # ── Train on labeled traffic ──────────────────────────────────────────────
    def train(self, csv_path: str):
        print(f"\n[IDS] Loading training data from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"[IDS] Total rows: {len(df)}")

        label_col = self._find_label_col(df)
        if not label_col:
            raise ValueError("Random Forest training requires a label/class column.")

        self.extractor.fit(df)
        flows = self.extractor.transform_df(df)
        labels = [flow.label for flow in flows]

        self.ml_detector.train(flows, labels)
        self.ml_detector.save(TRAINED_MODEL_PATH, feature_state=self.extractor.get_state())
        print(f"\n[IDS] ✅ Random Forest model saved to {TRAINED_MODEL_PATH}")

    # ── Analyze traffic ───────────────────────────────────────────────────────
    def analyze(self, csv_path: str, max_rows: int = 500) -> list:
        print(f"\n[IDS] Analyzing: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False).head(max_rows)
        print(f"[IDS] Analyzing {len(df)} flows...")

        if not self.ml_detector.trained:
            model_path = next(
                (
                    path
                    for path in MODEL_CANDIDATES
                    if os.path.exists(path)
                ),
                None,
            )
            if model_path:
                self.ml_detector.load(model_path)
                if self.ml_detector.feature_state:
                    self.extractor.set_state(self.ml_detector.feature_state)
            else:
                print("[IDS] ⚠️  No trained model found. Run --train first.")
                sys.exit(1)

        if not self.extractor.fitted:
            self.extractor.fit(df)  # fallback for legacy models without saved feature stats
        flows = self.extractor.transform_df(df)

        results = []
        block_count = alert_count = allow_count = 0

        for i, ff in enumerate(flows):
            rule_out = self.rule_engine.evaluate(ff)
            ml_out   = self.ml_detector.predict(ff)
            decision = self.fusion.decide(rule_out, ml_out)

            results.append({
                "flow_idx":      i,
                "src_ip":        ff.src_ip,
                "dst_ip":        ff.dst_ip,
                "src_port":      ff.src_port,
                "dst_port":      ff.dst_port,
                "label":         ff.label,
                "action":        decision.action,
                "risk_score":    decision.risk_score,
                "rule_matched":  decision.rule_matched,
                "matched_rules": ", ".join(decision.matched_rule_names),
                "ml_score":      decision.ml_anomaly_score,
                "rule_aware_ml_score": decision.rule_aware_ml_score,
                "ml_prediction": ml_out.predicted_label or ("Anomaly" if ml_out.is_anomaly else "Benign"),
                "ml_confidence": ml_out.confidence,
                "reasoning":     decision.reasoning,
            })

            if decision.action == "BLOCK":   block_count += 1
            elif decision.action == "ALERT": alert_count += 1
            else:                            allow_count += 1

        print(f"\n[IDS] Results:")
        print(f"  🚫 BLOCK : {block_count}")
        print(f"  ⚠️  ALERT : {alert_count}")
        print(f"  ✅ ALLOW : {allow_count}")

        return results

    def _find_label_col(self, df: pd.DataFrame):
        for col in df.columns:
            if col.strip().lower() in {"label", "class", "target", "attack", "attack_type"}:
                return col
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid IDS")
    parser.add_argument("--train",   type=str, help="CSV path for training")
    parser.add_argument("--analyze", type=str, help="CSV path for analysis")
    parser.add_argument("--rows",    type=int, default=500, help="Max rows to analyze")
    args = parser.parse_args()

    ids = HybridIDS()

    if args.train:
        ids.train(args.train)
    elif args.analyze:
        ids.analyze(args.analyze, max_rows=args.rows)
    else:
        parser.print_help()
