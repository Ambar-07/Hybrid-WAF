# Hybrid-WAF: Hybrid Web Application Firewall and Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-1.24+-013243?logo=numpy&logoColor=white)
![PyYAML](https://img.shields.io/badge/PyYAML-6.0+-CB171E?logo=yaml&logoColor=white)
![Scapy](https://img.shields.io/badge/Scapy-2.5+-CC0000?logo=python&logoColor=white)
![Requests](https://img.shields.io/badge/requests-2.31+-2CA5E0?logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20WAF%20%2B%20IDS-007ACC)
![ML Model](https://img.shields.io/badge/ML%20Model-Isolation%20Forest-28a745)
![Rule Engine](https://img.shields.io/badge/Rule%20Engine-YAML%20Signatures-6f42c1)
![Benchmark](https://img.shields.io/badge/Benchmark-CIC--IDS2017-d93f0b)
![License](https://img.shields.io/badge/License-MIT-yellow)

A dual-engine cybersecurity defense platform that integrates deterministic signature matching, unsupervised machine learning anomaly detection, and a dynamic risk fusion layer for real-time web application security and network intrusion monitoring.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Detection Pipeline](#core-detection-pipeline)
  - [1. Feature Extractor](#1-feature-extractor)
  - [2. Signature Rule Engine](#2-signature-rule-engine)
  - [3. Machine Learning Anomaly Detector](#3-machine-learning-anomaly-detector)
  - [4. Dynamic Risk Fusion Engine](#4-dynamic-risk-fusion-engine)
- [Interactive Streamlit Web Dashboard](#interactive-streamlit-web-dashboard)
- [Branch Enhancements (Ambar Gairola Branches)](#branch-enhancements-ambar-gairola-branches)
  - [Safe Localhost Traffic Generator](#safe-localhost-traffic-generator)
  - [Real-Time WAF Simulation and Stream Replay](#real-time-waf-simulation-and-stream-replay)
  - [Automated Evaluation and Performance Metrics](#automated-evaluation-and-performance-metrics)
  - [Traffic Capture and Packet Sniffing](#traffic-capture-and-packet-sniffing)
  - [Tabular Preprocessing and Feature Alignment](#tabular-preprocessing-and-feature-alignment)
- [Repository Branch Structure](#repository-branch-structure)
- [Project Directory Structure](#project-directory-structure)
- [Installation and Setup](#installation-and-setup)
- [Execution Guide](#execution-guide)
  - [Launch Web Dashboard](#launch-web-dashboard)
  - [CLI Model Training](#cli-model-training)
  - [CLI Traffic Analysis](#cli-traffic-analysis)
  - [Verify Model Integrity](#verify-model-integrity)
- [Risk Scoring and Decision Matrix](#risk-scoring-and-decision-matrix)
- [Attack Coverage and Detection Categories](#attack-coverage-and-detection-categories)
- [Configuration Reference](#configuration-reference)
- [License and Disclaimers](#license-and-disclaimers)

---

## Overview

Modern web applications face an evolving spectrum of cyber threats, ranging from well-documented exploit patterns (SQL Injection, Cross-Site Scripting, directory reconnaissance) to zero-day anomalous behaviors (distributed denial-of-service surges, port scanning, abnormal session timings).

Traditional signature-only Web Application Firewalls (WAFs) fail against novel evasion techniques and parameter permutations, while pure machine learning systems frequently produce unacceptable false positive rates on benign operational shifts.

Hybrid-WAF resolves this fundamental trade-off through a coordinated multi-layered architecture:

1. **Deterministic Rule Engine**: Instantly flags known exploit signatures, abnormal headers, malicious payloads, and malicious threshold crossings with high precision.
2. **Isolation Forest Anomaly Detector**: Unsupervised statistical modeling trained on benign traffic baselines to flag anomalous deviations and unknown threat vectors.
3. **Risk Fusion Layer**: Blends rule confidence, rule severity weighting, and statistical anomaly scores into a single normalized risk index, enforcing definitive actions: `ALLOW`, `ALERT`, or `BLOCK`.

---

## System Architecture

```
                                  INCOMING TRAFFIC
             (HTTP Requests / PCAP Network Flows / CSV Ingestion)
                                         |
                                         v
                         +-------------------------------+
                         |   Feature Extraction Engine   |
                         |   (19 Network Flow Features)  |
                         +---------------+---------------+
                                         |
                    +--------------------+--------------------+
                    |                                         |
                    v                                         v
     +-----------------------------+           +-----------------------------+
     |     Signature Rule Engine   |           |    ML Anomaly Detector      |
     |   (YAML Pattern Matcher)    |           |     (Isolation Forest)      |
     +--------------+--------------+           +--------------+--------------+
                    |                                         |
         Severity & Confidence                         Anomaly Score
             (0.0 - 1.0)                                (0.0 - 1.0)
                    |                                         |
                    +--------------------+--------------------+
                                         |
                                         v
                         +-------------------------------+
                         |      Risk Fusion Engine       |
                         |  Weighted Scoring & Threshold |
                         +---------------+---------------+
                                         |
                      +------------------+------------------+
                      |                  |                  |
                      v                  v                  v
                  [ ALLOW ]          [ ALERT ]          [ BLOCK ]
```

---

## Core Detection Pipeline

### 1. Feature Extractor
Located in `engine/feature_extractor.py`.

Extracts and normalizes 19 core network flow features compatible with enterprise benchmarks (such as the CIC-IDS2017 dataset format):
- Flow Metrics: `flow_duration`, `total_fwd_packets`, `total_bwd_packets`, `total_len_fwd_packets`, `total_len_bwd_packets`
- Packet Length Statistics: `fwd_packet_length_max`, `fwd_packet_length_min`, `fwd_packet_length_mean`, `bwd_packet_length_mean`
- Inter-Arrival Time (IAT): `flow_iat_mean`, `flow_iat_std`, `flow_iat_max`, `fwd_iat_total`
- Header and Protocol Flags: `fwd_header_length`, `bwd_header_length`, `fwd_packets_s`, `bwd_packets_s`, `syn_flag_count`, `ack_flag_count`

The extractor maintains internal mean/std deviation statistics to reliably transform raw flows into normalized feature vectors bounded for ML inference.

### 2. Signature Rule Engine
Located in `engine/rule_engine.py` with rules configured via `config/rules.yaml`.

Evaluates traffic against declarative, structured security rules across HTTP payloads, port configurations, and flow counters. Supports rich condition operators:
- Numeric comparisons: `>`, `<`, `>=`, `<=`, `==`, `!=`
- String and payload operators: `contains`, `startswith`, `endswith`, `in`

Outputs a structured `RuleEngineOutput` dataclass containing:
- Boolean detection status (`any_match`, `rule_detected`)
- List of triggered `RuleResult` items with rule IDs, names, severities, and triggered conditions
- Calculated maximum confidence rating
- Highest severity level (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`)

### 3. Machine Learning Anomaly Detector
Located in `engine/ml_detector.py`.

Employs an unsupervised **Isolation Forest** (`sklearn.ensemble.IsolationForest`) architecture:
- Designed to train exclusively on clean/benign network flows, learning nominal behavioral boundaries.
- Generates continuous anomaly scores mapped between `0.0` (normal) and `1.0` (severe anomaly).
- Incorporates dynamic feature width alignment: automatically pads or trims incoming vectors if dataset schemas vary between training and inference runs, guaranteeing backward compatibility.
- Configurable anomaly decision thresholds (defaulting to `0.55`).

### 4. Dynamic Risk Fusion Engine
Located in `engine/fusion.py`.

Merges deterministic signatures and probabilistic machine learning inference into an actionable risk assessment.

#### Fusion Mathematical Model

```
Rule Contribution  = Severity Weight * Rule Max Confidence
ML Contribution    = ML Anomaly Score
Composite Risk     = (w_rule * Rule Contribution) + (w_ml * ML Contribution)
```

Where:
- Default `w_rule` = 0.60
- Default `w_ml` = 0.40
- Severity Weights:
  - `CRITICAL` = 1.00
  - `HIGH`     = 0.85
  - `MEDIUM`   = 0.50
  - `LOW`      = 0.25
  - `NONE`     = 0.00

#### Action Decision Thresholds

| Action | Condition | Response |
|---|---|---|
| **BLOCK** | Risk Score >= 0.75 OR Rule Severity == CRITICAL | Packet/flow dropped, request rejected, alert logged |
| **ALERT** | Risk Score >= 0.45 OR Rule Severity == HIGH | Passed with warning flag, anomaly tagged in audit log |
| **ALLOW** | Risk Score < 0.45 | Normal clean traffic routed to destination |

---

## Interactive Streamlit Web Dashboard

The primary user interface is built with Streamlit (`ui/dashboard.py`) and organized into functional operational views:

### Dashboard Overview
- High-level KPI metric cards: Total Flows Processed, Blocks Enforced, Warnings Issued, Clean Flows Allowed.
- Visual charts: Action breakdown distribution, threat category distribution, and risk score frequency histograms.
- Real-time engine status monitor (ML model availability, threshold index, active rule count).

### Analyze Traffic
- Upload standard network traffic captures or flow CSV files (e.g., CIC-IDS2017 extracts).
- Run full hybrid analysis across thousands of flow records in seconds.
- Interactive table inspection with custom status badges, rule attribution, anomaly scores, and explanatory rationale.
- Data export of flagged threats for SIEM or incident response ingestion.

### Train Model
- Retrain or fine-tune the Isolation Forest model directly from the browser.
- Select local training datasets or generated capture files.
- Automated BENIGN flow filtering to ensure model integrity.
- Hyperparameter controls: Contamination factor (0.01 - 0.20) and number of estimators (50 - 300).
- One-click model serialization to `models/isolation_forest.pkl`.

### Rules Viewer
- Interactive inspection of all rules declared in `config/rules.yaml`.
- Search by rule name, rule ID, or attack category.
- Filter by severity grade (CRITICAL, HIGH, MEDIUM, LOW).
- View exact matching logic, condition operators, and detection parameters.

---

## Branch Enhancements (Ambar Gairola Branches)

In addition to the core pipeline on `main`, extended capabilities have been developed and tested in branches maintained by **Ambar Gairola** (`test-branch` and `test-branch-2`):

### Safe Localhost Traffic Generator
Implemented in `traffic_generator.py`:
- **Strict Sandbox Safety Guard**: Enforces execution exclusively against `127.0.0.1` and `localhost`. Requests targeting remote IP addresses or external hostnames are strictly blocked by safety assertion guards.
- **Dedicated Attack Simulators**:
  - SQL Injection Generator: Emits payloads including `' OR 1=1`, `admin'--`, and `UNION SELECT`.
  - Cross-Site Scripting (XSS) Generator: Emits `<script>`, `onerror=`, and `javascript:` vector injections.
  - Path Fuzzing and Reconnaissance: Generates directory traversal probes (`../`, `/.env`, `/wp-admin`, `/admin`).
  - Login Burst Generator: High-frequency authentication request bursts with credential fuzzing.
  - TCP Port Scanner: Sequential probe of target localhost ports to mimic port reconnaissance.
  - Connection Burst Generator: High-frequency TCP socket connect/close cycles to simulate DoS pressure.
  - Weighted Mixed Traffic Simulation: Allows users to configure exact ratios of Normal, Suspicious, and Malicious events to test pipeline resilience.

### Real-Time WAF Simulation and Stream Replay
Implemented in `ui/dashboard.py` (`test-branch-2`):
- **Live Stream Mode**: Replays simulated traffic step-by-step through the detection pipeline with configurable inter-request delays.
- **Dynamic Counters**: Live animated counters for Allowed, Alerted, and Blocked events.
- **Rule Explainability Panel**: Deep forensics display revealing triggered rule ID, rule name, severity level, exact regex/substring pattern, and match location (URI, headers, or payload).
- **ML Anomaly Score Tracking**: Interactive time-series plot comparing real-time anomaly scores against the threshold boundary.
- **Scenario History and Replay**: Saves previous simulation runs in session storage, enabling one-click replay of specific threat scenarios.

### Automated Evaluation and Performance Metrics
Implemented in `engine/evaluation.py`:
- Comprehensive quantitative model validation:
  - Accuracy Score
  - Precision Score
  - Recall Score
  - False Positive Count
  - Global Detection Rate
- Granular breakdown reporting detection efficiency grouped by specific attack classification.

### Traffic Capture and Packet Sniffing
Implemented in `capture/traffic_capture.py`:
- Localhost synthetic traffic capture converting simulated runs into structured CSV datasets (`capture/generated_traffic.csv`).
- Optional live packet sniffing powered by Scapy (`scapy.all.sniff`), extracting raw IP/TCP/UDP packet headers when loopback capture drivers (Npcap) are installed.

### Tabular Preprocessing and Feature Alignment
Implemented in `engine/preprocessing.py`:
- Missing value imputation and safe numeric conversion.
- Time-delta derivation (`epoch_seconds`, `hour`, `minute`, `time_since_start`) from raw timestamp streams.
- Optional min-max feature normalization.
- Backward compatibility layer in `engine/ml_detector.py` to reconcile differing column matrices automatically.

---

## Repository Branch Structure

The repository maintains the following branch layout:

| Branch Name | Primary Contributor | Focus / Contents |
|---|---|---|
| `main` | Ambar Gairola | Stable production baseline. Core hybrid IDS engine, CLI runner, models, rules, and Streamlit dashboard. |
| `test-branch-2` | Ambar Gairola | Active development branch. Adds Weighted Mixed Traffic Simulator, Real-Time Replay Simulation, Stream Mode, Metrics Panel, Rule Forensics, and Localhost Demo Rules. |
| `test-branch` | Ambar Gairola | Initial prototype branch for localhost traffic generation, capture module, and Streamlit UI refresh. |

*Note: For the latest live attack simulation and real-time decision replay features, switch to `test-branch-2`.*

---

## Project Directory Structure

```
Hybrid-WAF/
|-- .streamlit/
|   `-- config.toml             # Streamlit server and theme configuration
|-- capture/
|   |-- generated_traffic.csv   # Persisted localhost attack/normal event logs
|   `-- traffic_capture.py      # Localhost traffic logging and Scapy packet sniffer
|-- config/
|   `-- rules.yaml              # Declarative YAML signature rules dictionary
|-- engine/
|   |-- __init__.py
|   |-- evaluation.py           # Precision, recall, and detection metrics engine
|   |-- feature_extractor.py    # 19-dimensional flow feature extraction engine
|   |-- fusion.py               # Weighted risk fusion and decision arbiter
|   |-- ml_detector.py          # Isolation Forest anomaly scoring model
|   |-- model.py                # Isolation Forest wrapper class
|   |-- preprocessing.py        # Tabular data cleaner and normalizer
|   `-- rule_engine.py          # Rule parsing and pattern evaluation engine
|-- models/
|   |-- isolation_forest.pkl    # Serialized Isolation Forest model binary
|   `-- loadmodel.py            # Model loading and health-check verification utility
|-- ui/
|   `-- dashboard.py            # Streamlit multi-page web dashboard
|-- main.py                     # Command-line interface for training and batch analysis
|-- requirements.txt            # Python package dependencies
`-- README.md                   # Project documentation
```

---

## Installation and Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Ambar-07/Hybrid-WAF.git
cd Hybrid-WAF
```

### 2. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*For branches utilizing Scapy and requests (`test-branch-2`), dependencies include:*
```bash
pip install pandas numpy scikit-learn streamlit pyyaml requests scapy
```

---

## Execution Guide

### Launch Web Dashboard
Run the Streamlit application from the project root:
```bash
streamlit run ui/dashboard.py
```
The interface will automatically open in your default browser at `http://localhost:8501`.

### CLI Model Training
Train the Isolation Forest model on normal/benign network traffic records:
```bash
python main.py --train data/cicids2017/normal_traffic.csv
```
The script will filter for benign flows, normalize feature representations, train the estimator, and persist the weights to `models/isolation_forest.pkl`.

### CLI Traffic Analysis
Execute the hybrid inspection pipeline on an unlabelled or mixed traffic batch:
```bash
python main.py --analyze data/test_traffic.csv --rows 500
```
Console output displays flow-by-flow assessments along with aggregate totals:
```text
[IDS] Results:
  BLOCK : 42
  ALERT : 18
  ALLOW : 440
```

### Verify Model Integrity
Confirm that the trained model file is present, structurally intact, and loadable:
```bash
python models/loadmodel.py
```

---

## Risk Scoring and Decision Matrix

Hybrid-WAF combines signature certainty with statistical variance using a multi-factor risk weighting model:

```
+-------------------------------------------------------------------------+
| Risk Score = (0.60 * Rule Severity Weight * Confidence) + (0.40 * ML)   |
+-------------------------------------------------------------------------+
```

### Decision Matrix

| Rule Match Status | ML Anomaly Score | Calculated Risk | Output Action | Explanation |
|---|---|---|---|---|
| Critical Signature Match | Low (< 0.40) | >= 0.75 (Override) | **BLOCK** | Explicit threat detected; blocked regardless of ML |
| High Signature Match | Moderate (0.40 - 0.60) | 0.65 - 0.85 | **BLOCK / ALERT** | High-severity rule combined with abnormal flow characteristics |
| No Rule Match | High (> 0.80) | 0.45 - 0.65 | **ALERT** | Novel anomaly detected; flagged for review without signature |
| No Rule Match | Low (< 0.30) | < 0.30 | **ALLOW** | Clean baseline behavior; unobstructed passage |

---

## Attack Coverage and Detection Categories

The rule definitions in `config/rules.yaml` provide coverage across key web and network attack classifications:

| Category | Example Rule IDs | Trigger Conditions / Signatures | Severity |
|---|---|---|---|
| **SQL Injection** | `SQLI-UNION-001`, `DEMO-SQLI-001` | String contains `' OR 1=1`, `UNION SELECT`, `admin'--` | HIGH / CRITICAL |
| **Cross-Site Scripting** | `XSS-SCRIPT-001`, `DEMO-XSS-001` | Pattern contains `<script>`, `onerror=`, `javascript:` | HIGH |
| **Reconnaissance / Probe** | `DEMO-PROBE-001`, `PORT-SCAN-001` | Unique destination ports > 12 within short duration | MEDIUM |
| **DoS / Flooding** | `DEMO-BURST-001`, `DOS-SYN-001` | High connection bursts, packet rate > 20 pkts/s | HIGH |
| **Brute Force** | `BRUTE-SSH-001`, `BRUTE-HTTP-001` | Rapid repeat requests targeting authentication endpoints | HIGH |
| **Web Shell / Traversal** | `TRAVERSAL-001` | Path indicators `../`, `/wp-admin`, `/.env` access attempts | HIGH |
| **Heartbleed / Protocol Exploit** | `HEARTBLEED-001` | Payload size anomaly on TLS heartbeat port 443 | CRITICAL |

---

## Configuration Reference

Customization is managed through two primary configuration files:

1. **`config/rules.yaml`**:
   Add, modify, or disable signature rules. Each rule requires:
   - `id`: Unique alphanumeric identifier (e.g., `CUSTOM-001`)
   - `name`: Human-readable label
   - `category`: Attack classification
   - `severity`: `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`
   - `confidence`: Floating-point scalar from `0.0` to `1.0`
   - `conditions`: List of field expressions (`field`, `op`, `value`)

2. **`ui/dashboard.py` & `main.py`**:
   Tune fusion weightings (`w_rule`, `w_ml`), anomaly decision thresholds (`threshold`), and action limits (`block_threshold`, `alert_threshold`).

---

## Evaluation Benchmark

The system can be validated against the Canadian Institute for Cybersecurity **CIC-IDS2017** benchmark dataset:
- Benchmark source: [CIC-IDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- Recommended training subset: `Monday-WorkingHours.pcap_ISCX.csv` (100% Benign baseline flows)
- Recommended testing subsets: `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` and `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`

---

## License and Disclaimers

This software is developed for educational, defensive, and research purposes. Traffic generation features in `traffic_generator.py` are strictly bounded to localhost (`127.0.0.1`) and must never be directed toward unauthorized external targets.
