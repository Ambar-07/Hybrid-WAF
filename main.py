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
from engine.preprocessing import Preprocessor
from engine.evaluation import evaluate_predictions, actions_to_labels

MODEL_PATH = "models/isolation_forest.pkl"
RULES_PATH = "config/rules.yaml"


class HybridIDS:
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.preprocessor = Preprocessor()
        self.rule_engine = RuleEngine(rules_path=RULES_PATH)
        self.ml_detector = MLDetector()
        self.fusion = FusionLayer()

    # ── Train on normal traffic ───────────────────────────────────────────────
    def train(self, csv_path: str):
        print(f"\n[IDS] Loading training data from: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        df = self.preprocessor.fit_transform(df)
        print(f"[IDS] Total rows: {len(df)}")

        # Keep only BENIGN flows for training the ML model
        label_col = self._find_label_col(df)
        if label_col:
            normal_df = df[df[label_col].str.upper().str.contains("BENIGN")]
            print(f"[IDS] Normal flows for ML training: {len(normal_df)}")
        else:
            normal_df = df
            print("[IDS] No label column found — using all rows for training")

        # Fit feature extractor on ALL data (for normalization stats)
        self.extractor.fit(df)

        # Transform normal flows for ML
        normal_flows = self.extractor.transform_df(normal_df)

        # Train ML model
        self.ml_detector.train(normal_flows)
        self.ml_detector.save(MODEL_PATH)
        print(f"\n[IDS] ✅ Model saved to {MODEL_PATH}")

    # ── Analyze traffic ───────────────────────────────────────────────────────
    def analyze(self, csv_path: str, max_rows: int = 500) -> list:
        print(f"\n[IDS] Analyzing: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False).head(max_rows)
        df = self.preprocessor.fit_transform(df)
        print(f"[IDS] Analyzing {len(df)} flows...")

        if not self.ml_detector.trained:
            if os.path.exists(MODEL_PATH):
                self.ml_detector.load(MODEL_PATH)
            else:
                print("[IDS] ⚠️  No trained model found. Run --train first.")
                sys.exit(1)

        self.extractor.fit(df)  # fit on current batch for normalization
        flows = self.extractor.transform_df(df)

        results = []
        block_count = alert_count = allow_count = 0

        for i, ff in enumerate(flows):
            rule_out = self.rule_engine.evaluate(ff)
            ml_out   = self.ml_detector.predict(ff)
            decision = self.fusion.decide(rule_out, ml_out)
            predicted_label = actions_to_labels([decision.action])[0]

            results.append({
                "flow_idx":      i,
                "src_ip":        ff.src_ip,
                "dst_ip":        ff.dst_ip,
                "src_port":      ff.src_port,
                "dst_port":      ff.dst_port,
                "label":         ff.label,
                "action":        decision.action,
                "predicted_label": predicted_label,
                "risk_score":    decision.risk_score,
                "rule_matched":  decision.rule_matched,
                "matched_rules": ", ".join(decision.matched_rule_names),
                "ml_score":      decision.ml_anomaly_score,
                "reasoning":     decision.reasoning,
            })

            if decision.action == "BLOCK":   block_count += 1
            elif decision.action == "ALERT": alert_count += 1
            else:                            allow_count += 1

        print(f"\n[IDS] Results:")
        print(f"  🚫 BLOCK : {block_count}")
        print(f"  ⚠️  ALERT : {alert_count}")
        print(f"  ✅ ALLOW : {allow_count}")

        label_col = self._find_label_col(df)
        if label_col:
            metrics = evaluate_predictions(df[label_col].tolist(), [r["predicted_label"] for r in results])
            print("\n[IDS] Evaluation Metrics (vs ground truth):")
            print(f"  Accuracy : {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall   : {metrics['recall']:.4f}")
            print(f"  F1 Score : {metrics['f1_score']:.4f}")

        return results

    def _find_label_col(self, df: pd.DataFrame):
        for col in df.columns:
            if col.strip().lower() == "label":
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
