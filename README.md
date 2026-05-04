#  Hybrid Intrusion Detection System

Combines Rule-Based detection + ML Anomaly Detection + Risk Fusion.

## Setup

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run ui/dashboard.py
```

## Train the ML Model (CLI)

```bash
python main.py --train data/your_cicids_file.csv
```

## Analyze Traffic (CLI)

```bash
python main.py --analyze data/test.csv --rows 500
```

## Project Structure

```
hybrid-ids/
├── engine/
│   ├── feature_extractor.py   # Extracts + normalizes selected CICIDS features
│   ├── rule_engine.py         # YAML-based signature rules
│   ├── ml_detector.py         # Random Forest / legacy anomaly detection
│   └── fusion.py              # Weighted risk score + decision
├── config/
│   └── rules.yaml             # All signature rules (edit here!)
├── models/
│   ├── best_model.pkl         # Primary Random Forest model
│   ├── cicids_rf_pipeline.pkl # Fallback transferred model
│   └── attack_specialist.pkl  # Secondary attack-family specialist
├── ui/
│   └── dashboard.py           # Streamlit UI
├── main.py                    # CLI entry point
└── requirements.txt
```

## Dataset

Download CICIDS2017 from:  
https://www.unb.ca/cic/datasets/ids-2017.html


## Team Tasks

| Person | File(s) to own |
|--------|---------------|
| CS Person 1  | fusion.py, rules.yaml, integration |
| CS Person 2  | feature_extractor.py, dataset loading |
| CS Person 3  | testing, evaluation metrics, attack simulation |
| DS/ML Person | ml_detector.py, model tuning |
