"""
Train the attack-only specialist model.

Usage:
  python3 train_attack_specialist.py --data data/cybersecurity_attacks.csv
"""

import argparse

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from engine.attack_specialist import AttackSpecialist, ATTACK_LABEL_COLUMN, DEFAULT_ATTACK_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Train attack-only specialist model")
    parser.add_argument("--data", required=True, help="Attack-only CSV with an 'Attack Type' column")
    parser.add_argument("--model-out", default=DEFAULT_ATTACK_MODEL_PATH, help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split")
    args = parser.parse_args()

    df = pd.read_csv(args.data, low_memory=False)
    if ATTACK_LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {ATTACK_LABEL_COLUMN}")

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=42,
        stratify=df[ATTACK_LABEL_COLUMN],
    )

    specialist = AttackSpecialist().train(train_df)
    preds = [
        specialist.predict_row(row).attack_type
        for row in test_df.drop(columns=[ATTACK_LABEL_COLUMN]).to_dict(orient="records")
    ]
    print(classification_report(test_df[ATTACK_LABEL_COLUMN].astype(str), preds, zero_division=0))

    specialist.save(args.model_out)
    print(f"Saved attack specialist model to {args.model_out}")


if __name__ == "__main__":
    main()
