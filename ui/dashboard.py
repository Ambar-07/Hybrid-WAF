"""
Hybrid WAF — Streamlit Dashboard
Run: streamlit run ui/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Ensure working directory is project root so relative paths work
os.chdir(PROJECT_ROOT)

from engine.feature_extractor import FeatureExtractor
from engine.rule_engine import RuleEngine
from engine.ml_detector import MLDetector
from engine.fusion import FusionLayer
from engine.preprocessing import Preprocessor
from engine.evaluation import evaluate_predictions, actions_to_labels
from capture.traffic_capture import logs_to_structured_csv
import traffic_generator as tg

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid WAF",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme_css_vars = """
    --site-context-theme-color: #121315;
    --site-context-ink: #ece7de;
    --site-context-metainfo-color: #b2aca2;
    --site-context-border-soft: #2a2d31;
    --site-context-border-strong: rgba(236, 231, 222, 0.38);
    --site-context-ink-83: rgba(236, 231, 222, 0.83);
    --site-context-ink-82: rgba(236, 231, 222, 0.82);
    --site-context-ink-04: rgba(236, 231, 222, 0.06);
    --site-context-ink-03: rgba(236, 231, 222, 0.04);
    --site-context-focus-shadow: rgba(0, 0, 0, 0.35) 0px 4px 12px;
    --site-context-ring: rgba(96, 165, 250, 0.55);
    --sidebar-bg: #15171b;
    --surface-2: #1c2026;
    --button-bg: #ece7de;
    --button-text: #141414;
    --panel-bg: rgba(21, 24, 30, 0.62);
    --panel-border: rgba(236, 231, 222, 0.10);
"""

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600&family=IBM+Plex+Mono:wght@400;600&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,300,0,0&display=swap');

:root {
    __THEME_VARS__
    --font-brand: 'Camera Plain Variable', 'Manrope', ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
    --danger: #ef4444;
    --warning: #f59e0b;
    --success: #22c55e;
    --surface-soft: rgba(21, 24, 30, 0.58);
    --surface-strong: rgba(21, 24, 30, 0.78);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    font-family: var(--font-brand);
    line-height: 1.5;
}
code, .stCode, .stTextArea textarea, .stTextInput input {
    font-family: 'IBM Plex Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% -10%, rgba(96, 165, 250, 0.08), transparent 42%),
        radial-gradient(circle at 95% 0%, rgba(250, 204, 21, 0.04), transparent 36%),
        var(--site-context-theme-color);
    color: var(--site-context-ink);
}
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1240px;
    padding-top: 1.5rem;
    padding-bottom: 3.25rem;
}
[data-testid="stHeader"] {
    background: color-mix(in srgb, var(--site-context-theme-color) 88%, transparent);
}
[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--site-context-border-soft);
}
.sidebar-title {
    font-size: 1.28rem;
    font-weight: 600;
    letter-spacing: -0.4px;
    color: var(--site-context-ink);
    margin: 2px 0 12px;
    line-height: 1.25;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 2px 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    position: relative;
    border: none;
    border-radius: 12px;
    background: transparent;
    padding: 9px 11px 9px 40px;
    min-height: 40px;
    overflow: hidden;
    transition: background 0.2s ease, transform 0.2s ease, color 0.2s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: linear-gradient(90deg, rgba(96, 165, 250, 0.10) 0%, rgba(96, 165, 250, 0.02) 100%);
    transform: translateX(2px);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(96, 165, 250, 0.2) 0%, rgba(96, 165, 250, 0.04) 100%);
    box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.22) inset, 0 8px 18px rgba(59, 130, 246, 0.14);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label::before {
    position: absolute;
    left: 12px;
    top: 10px;
    font-family: 'Material Symbols Outlined';
    font-size: 18px;
    font-weight: 300;
    line-height: 1;
    color: rgba(176, 208, 255, 0.82);
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label::after {
    content: "";
    position: absolute;
    left: 0;
    top: 8px;
    width: 2px;
    height: calc(100% - 16px);
    background: linear-gradient(180deg, rgba(96, 165, 250, 0.0) 0%, rgba(140, 186, 255, 0.95) 40%, rgba(96, 165, 250, 0.0) 100%);
    opacity: 0;
    transition: opacity 0.2s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked)::after {
    opacity: 1;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(1)::before {
    content: "dashboard";
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(2)::before {
    content: "lan";
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(3)::before {
    content: "monitoring";
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(4)::before {
    content: "model_training";
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:nth-child(5)::before {
    content: "gavel";
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label p {
    color: rgba(236, 231, 222, 0.88);
    font-weight: 400;
    font-size: 0.92rem;
    letter-spacing: 0;
    text-transform: none;
    margin: 0;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
    color: rgba(236, 231, 222, 1);
}
.sidebar-panel {
    margin: 10px 0 14px;
    padding: 12px 12px 11px;
    background: linear-gradient(180deg, rgba(23, 28, 36, 0.72) 0%, rgba(18, 21, 28, 0.78) 100%);
    border: 1px solid rgba(236, 231, 222, 0.10);
    border-radius: 0;
}
.sidebar-panel-title {
    margin: 0 0 9px;
    color: var(--site-context-metainfo-color);
    font-size: 0.76rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.sidebar-status-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}
.sidebar-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    margin-right: 7px;
}
.sidebar-status {
    display: inline-flex;
    align-items: center;
    color: var(--site-context-ink);
    font-size: 0.9rem;
}
.sidebar-status.is-online .sidebar-status-dot {
    background: rgba(74, 222, 128, 0.95);
    box-shadow: 0 0 0 5px rgba(74, 222, 128, 0.14);
}
.sidebar-status.is-offline .sidebar-status-dot {
    background: rgba(245, 158, 11, 0.95);
    box-shadow: 0 0 0 5px rgba(245, 158, 11, 0.14);
}
.sidebar-rules {
    color: var(--site-context-metainfo-color);
    font-size: 0.82rem;
}
.sidebar-rules strong {
    color: var(--site-context-ink);
    font-weight: 600;
}

.hero-shell {
    position: relative;
    width: 100%;
    min-height: 292px;
    padding: 58px 52px;
    margin: 8px 0 30px;
    color: var(--site-context-ink);
    background:
        radial-gradient(circle at 14% 24%, rgba(255, 188, 145, 0.25) 0%, rgba(255, 188, 145, 0) 42%),
        radial-gradient(circle at 86% 8%, rgba(167, 194, 255, 0.24) 0%, rgba(167, 194, 255, 0) 40%),
        radial-gradient(circle at 50% 88%, rgba(246, 181, 177, 0.2) 0%, rgba(246, 181, 177, 0) 45%),
        var(--site-context-theme-color);
    border: 1px solid rgba(236, 231, 222, 0.10);
    border-radius: 0;
}
.hero-kicker {
    font-size: 0.95rem;
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
    color: var(--site-context-metainfo-color);
    margin: 0 0 8px 0;
    line-height: 1.5;
}
.hero-title {
    max-width: 14ch;
    font-size: clamp(2.4rem, 5vw, 3.85rem);
    font-weight: 600;
    letter-spacing: -1.3px;
    text-transform: none;
    line-height: 1.05;
    margin: 0;
}
.hero-sub {
    margin-top: 14px;
    max-width: 64ch;
    font-size: 1.05rem;
    line-height: 1.5;
    color: rgba(236, 231, 222, 0.8);
}

.light-shell {
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 10px 0 8px;
    margin: 0;
}
.section-kicker {
    color: var(--site-context-metainfo-color);
    margin: 0;
    font-size: 0.9rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    line-height: 1.5;
}
.section-title {
    margin: 6px 0 6px 0;
    color: var(--site-context-ink);
    font-size: clamp(1.85rem, 4.2vw, 2.55rem);
    font-weight: 600;
    letter-spacing: -1.2px;
    line-height: 1.05;
}
.section-body {
    margin: 0;
    color: var(--site-context-metainfo-color);
    font-size: 1rem;
    line-height: 1.5;
    max-width: 70ch;
}

.metric-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 18px 16px;
    text-align: left;
    min-height: 138px;
}
.metric-card-title {
    color: var(--site-context-ink);
    font-size: 1.05rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    line-height: 1.25;
    margin-bottom: 10px;
}
.metric-card-desc {
    color: var(--site-context-metainfo-color);
    font-size: 0.95rem;
    line-height: 1.5;
}

.architecture-shell {
    margin: 16px 0 30px;
    padding: 14px 2px 12px;
    overflow-x: auto;
    overflow-y: hidden;
    scrollbar-width: thin;
}
.architecture-track {
    width: fit-content;
    min-width: 100%;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
}
.architecture-connector {
    width: 56px;
    min-width: 56px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.architecture-connector-line {
    position: relative;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, rgba(236, 231, 222, 0.10) 0%, rgba(140, 186, 255, 0.45) 50%, rgba(236, 231, 222, 0.10) 100%);
    animation: flowPulse 3.2s ease-in-out infinite;
}
.architecture-connector-line::after {
    content: "";
    position: absolute;
    right: -1px;
    top: -4px;
    width: 0;
    height: 0;
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    border-left: 7px solid rgba(167, 202, 255, 0.72);
}
.architecture-step {
    width: 228px;
    min-width: 228px;
    min-height: 186px;
    padding: 18px;
    background: linear-gradient(180deg, rgba(27, 33, 43, 0.58) 0%, rgba(17, 20, 28, 0.84) 100%);
    border: 1px solid rgba(236, 231, 222, 0.09);
    border-radius: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
    transition: transform 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease;
}
.architecture-step:hover {
    transform: translateY(-3px) scale(1.015);
    border-color: rgba(140, 186, 255, 0.4);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28), inset 0 0 0 1px rgba(140, 186, 255, 0.12);
}
.architecture-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
}
.architecture-icon {
    width: 36px;
    height: 36px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: rgba(172, 209, 255, 0.92);
    background: rgba(96, 165, 250, 0.1);
    border: 1px solid rgba(96, 165, 250, 0.25);
}
.architecture-icon .material-symbols-outlined {
    font-size: 19px;
    font-variation-settings: 'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24;
    line-height: 1;
}
.architecture-index {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: rgba(236, 231, 222, 0.7);
    letter-spacing: 0.04em;
}
.architecture-title {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 600;
    color: var(--site-context-ink);
    line-height: 1.25;
}
.architecture-desc {
    margin: 0;
    color: var(--site-context-metainfo-color);
    font-size: 0.9rem;
    line-height: 1.45;
    max-width: 24ch;
}
@keyframes flowPulse {
    0%, 100% { opacity: 0.72; }
    50% { opacity: 1; }
}
.block-badge {
    background: var(--danger);
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 400;
    font-size: 13px;
}
.alert-badge {
    background: var(--warning);
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 400;
    font-size: 13px;
}
.allow-badge {
    background: var(--success);
    color: white;
    padding: 4px 12px;
    border-radius: 9999px;
    font-weight: 400;
    font-size: 13px;
}
.risk-bar-bg {
    background: var(--site-context-ink-04);
    border-radius: 9999px;
    height: 10px;
    width: 100%;
}
div[data-testid="stMetric"] {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    padding: 15px;
    box-shadow: none;
}
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
    color: var(--site-context-ink);
}
[data-testid="stButton"] button {
    background: var(--button-bg);
    color: var(--button-text);
    border: 1px solid rgba(0, 0, 0, 0);
    border-radius: 6px;
    width: auto;
    min-height: 40px;
    padding: 8px 16px;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    line-height: 1.5;
    font-size: 0.95rem;
    box-shadow:
        rgba(255, 255, 255, 0.2) 0px 0.5px 0px 0px inset,
        rgba(0, 0, 0, 0.2) 0px 0px 0px 0.5px inset,
        rgba(0, 0, 0, 0.05) 0px 1px 2px 0px;
}
[data-testid="stButton"] button p,
[data-testid="stButton"] button span {
    color: var(--button-text) !important;
    font-size: 0.95rem !important;
}
[data-testid="stButton"] button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow:
        rgba(255, 255, 255, 0.2) 0px 0.5px 0px 0px inset,
        rgba(0, 0, 0, 0.2) 0px 0px 0px 0.5px inset,
        rgba(0, 0, 0, 0.1) 0px 4px 12px;
}
[data-testid="stAlert"] {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: rgba(21, 24, 30, 0.55);
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div,
[data-testid="stFileUploader"] section,
[data-baseweb="select"] > div {
    background: rgba(21, 24, 30, 0.55) !important;
    border: 1px solid var(--panel-border) !important;
    border-radius: 6px !important;
    color: var(--site-context-ink) !important;
    box-shadow: none !important;
}
[data-testid="stNumberInput"] button {
    background: rgba(21, 24, 30, 0.75) !important;
    border: 1px solid var(--panel-border) !important;
    color: var(--site-context-ink) !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--site-context-ink-04) !important;
    border-color: var(--site-context-border-strong) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--site-context-ink);
    border: none;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #3b82f6;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus,
[data-baseweb="select"] > div:focus-within {
    outline: none !important;
    border-color: rgba(59, 130, 246, 0.5) !important;
    box-shadow: 0 0 0 2px var(--site-context-ring), var(--site-context-focus-shadow) !important;
}
[data-testid="stFileUploader"] section {
    background: rgba(28, 28, 28, 0.02) !important;
    border-style: dashed !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
}
[data-testid="stExpander"] {
    border: 1px solid var(--panel-border);
    border-radius: 12px;
    background: rgba(21, 24, 30, 0.50);
}
[data-testid="stMarkdownContainer"] p {
    color: var(--site-context-ink-83);
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] label {
    color: var(--site-context-ink);
}
a {
    color: var(--site-context-ink);
    text-decoration: underline;
}
a:hover {
    color: var(--site-context-ink);
    text-decoration: underline;
}
[data-testid="stSidebar"] hr,
.stMarkdown hr {
    border-color: var(--panel-border);
}
@media (max-width: 920px) {
    .hero-shell {
        min-height: 240px;
        padding: 40px 22px;
    }
    .section-title {
        font-size: clamp(2rem, 9vw, 2.25rem);
        letter-spacing: -0.9px;
    }
}
@media (max-width: 1024px) {
    .architecture-track {
        justify-content: flex-start;
    }
}
@media (max-width: 860px) {
    .architecture-shell {
        margin-bottom: 20px;
        padding-bottom: 14px;
        overflow: hidden;
    }
    .architecture-track {
        width: 100%;
        min-width: 100%;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        gap: 10px;
    }
    .architecture-step {
        width: min(100%, 420px);
        min-width: 0;
        min-height: 160px;
    }
    .architecture-connector {
        width: 2px;
        min-width: 2px;
        height: 26px;
    }
    .architecture-connector-line {
        width: 2px;
        height: 26px;
        background: linear-gradient(180deg, rgba(236, 231, 222, 0.10) 0%, rgba(140, 186, 255, 0.55) 50%, rgba(236, 231, 222, 0.10) 100%);
    }
    .architecture-connector-line::after {
        right: -2px;
        top: auto;
        bottom: -6px;
        border-top: 7px solid rgba(167, 202, 255, 0.72);
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-bottom: 0;
    }
}
</style>
""".replace("__THEME_VARS__", theme_css_vars), unsafe_allow_html=True)


# ── Cache models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_components():
    extractor    = FeatureExtractor()
    preprocessor = Preprocessor()
    rule_engine  = RuleEngine(rules_path="config/rules.yaml")
    ml_detector  = MLDetector()
    fusion       = FusionLayer()

    model_path = "models/isolation_forest.pkl"
    if os.path.exists(model_path):
        ml_detector.load(model_path)

    return extractor, preprocessor, rule_engine, ml_detector, fusion


extractor, preprocessor, rule_engine, ml_detector, fusion = load_components()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Hybrid WAF</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Dashboard", "Traffic Generator", "Analyze Traffic", "Train Model", "Rules Viewer", "Wireshark PCAP Capture"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    status_class = "is-online" if ml_detector.trained else "is-offline"
    status_text = "ML model loaded" if ml_detector.trained else "Model unavailable"
    status_hint = "Ready for anomaly scoring" if ml_detector.trained else "Train model to enable ML scoring"
    st.markdown(
        f'<div class="sidebar-panel">'
        f'<p class="sidebar-panel-title">Model Status</p>'
        f'<div class="sidebar-status-row">'
        f'<span class="sidebar-status {status_class}">'
        f'<span class="sidebar-status-dot"></span>{status_text}'
        f'</span>'
        f'</div>'
        f'<div class="sidebar-rules">{status_hint}<br/>Rules loaded: <strong>{len(rule_engine.rules)}</strong></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Fusion weights
    st.markdown("**Fusion Weights**")
    rule_w = st.slider("Rule Weight", 0.0, 1.0, 0.55, 0.05)
    ml_w   = 1.0 - rule_w
    st.caption(f"ML Weight: {ml_w:.2f}")
    fusion.rule_weight = rule_w
    fusion.ml_weight   = ml_w

    st.markdown("**Decision Thresholds**")
    fusion.allow_threshold = st.slider("Alert Threshold", 0.1, 0.9, 0.35, 0.05)
    fusion.block_threshold = st.slider("Block Threshold", 0.1, 1.0, 0.70, 0.05)


# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(action):
    cls = {"BLOCK": "block-badge", "ALERT": "alert-badge", "ALLOW": "allow-badge"}.get(action, "")
    return f'<span class="{cls}">{action}</span>'


def safe_ml_predict(ff, anomaly_hint: bool = False):
    """Predict with graceful fallback when a stale model schema is loaded."""
    from engine.ml_detector import MLResult

    if not ml_detector.trained:
        return MLResult(anomaly_score=0.0, is_anomaly=False, raw_score=0.0, confidence=0.0)

    try:
        return ml_detector.predict(ff)
    except ValueError as exc:
        msg = str(exc).lower()
        if "feature" in msg and "expect" in msg:
            if not st.session_state.get("ml_feature_schema_warning_shown", False):
                st.session_state["ml_feature_schema_warning_shown"] = True
                st.warning(
                    "Loaded ML model uses an older feature schema. "
                    "Using fallback ML scoring for now. Retrain from Train Model to fully fix this."
                )
            return MLResult(
                anomaly_score=0.7 if anomaly_hint else 0.0,
                is_anomaly=anomaly_hint,
                raw_score=0.0,
                confidence=0.5 if anomaly_hint else 0.0,
            )
        raise


def analyze_df(df: pd.DataFrame, max_rows=300):
    df = df.head(max_rows).copy()
    processed_df = preprocessor.fit_transform(df)
    extractor.fit(processed_df)
    flows = extractor.transform_df(processed_df)

    rows = []
    for i, ff in enumerate(flows):
        rule_out = rule_engine.evaluate(ff)
        ml_out = safe_ml_predict(ff)
        decision = fusion.decide(rule_out, ml_out)
        rule_detected = bool(getattr(rule_out, "rule_detected", getattr(rule_out, "any_match", False)))
        final_label = "Malicious" if rule_detected else ("Suspicious" if ml_out.is_anomaly else "Benign")

        attack_type = str(getattr(rule_out, "attack_type", "NONE"))

        rows.append({
            "Action":        decision.action,
            "Final Label":   final_label,
            "Risk Score":    decision.risk_score,
            "Rule Hit":      "Yes" if decision.rule_matched else "No",
            "Attack Type":   attack_type,
            "Matched Rules": ", ".join(decision.matched_rule_names) or "None",
            "Rule Severity": decision.rule_severity,
            "ML Score":      round(decision.ml_anomaly_score, 3),
            "Dst Port":      ff.dst_port,
            "Label":         ff.label,
            "Reasoning":     decision.reasoning,
        })

    return pd.DataFrame(rows)


def generate_demo_csv(mode: str, **kwargs):
    generated_logs = []
    if mode == "normal":
        generated_logs = tg.generate_normal_http(
            target_url=kwargs.get("target_url", "http://127.0.0.1:8000"),
            n=kwargs.get("n", 30),
            delay=kwargs.get("delay", 0.05),
        )
    elif mode == "payload":
        generated_logs = tg.generate_payload_http(
            target_url=kwargs.get("target_url", "http://127.0.0.1:8000"),
            payloads=kwargs.get("payloads", []),
            delay=kwargs.get("delay", 0.05),
        )
    elif mode == "probe":
        generated_logs = tg.generate_port_probe(
            target_ip=kwargs.get("target_ip", "127.0.0.1"),
            start_port=kwargs.get("start_port", 8000),
            end_port=kwargs.get("end_port", 8020),
            delay=kwargs.get("delay", 0.02),
        )
    elif mode == "burst":
        generated_logs = tg.generate_connection_burst(
            target_ip=kwargs.get("target_ip", "127.0.0.1"),
            port=kwargs.get("port", 8000),
            count=kwargs.get("count", 40),
            concurrency=kwargs.get("concurrency", 5),
            delay=kwargs.get("delay", 0.01),
        )
    elif mode == "path_fuzz":
        path_fuzz_fn = getattr(tg, "generate_path_fuzz_http", None)
        if path_fuzz_fn is None:
            raise ValueError("Path fuzz generator is unavailable. Please restart Streamlit to load latest traffic_generator.py")
        generated_logs = path_fuzz_fn(
            target_url=kwargs.get("target_url", "http://127.0.0.1:8000"),
            paths=kwargs.get("paths", []),
            delay=kwargs.get("delay", 0.04),
        )
    elif mode == "login_burst":
        login_burst_fn = getattr(tg, "generate_login_burst_http", None)
        if login_burst_fn is None:
            raise ValueError("Login burst generator is unavailable. Please restart Streamlit to load latest traffic_generator.py")
        generated_logs = login_burst_fn(
            target_url=kwargs.get("target_url", "http://127.0.0.1:8000"),
            usernames=kwargs.get("usernames", []),
            passwords=kwargs.get("passwords", []),
            attempts=kwargs.get("attempts", 30),
            delay=kwargs.get("delay", 0.03),
        )
    elif mode == "random_mix":
        random_mix_fn = getattr(tg, "generate_random_mixed_traffic", None)
        if random_mix_fn is None:
            raise ValueError("Random mixed traffic generator is unavailable. Please restart Streamlit to load latest traffic_generator.py")
        generated_logs = random_mix_fn(
            target_url=kwargs.get("target_url", "http://127.0.0.1:8000"),
            target_ip=kwargs.get("target_ip", "127.0.0.1"),
            total_events=kwargs.get("total_events", 100),
            malicious_ratio=kwargs.get("malicious_ratio", 0.35),
            delay=kwargs.get("delay", 0.02),
        )

    combined = st.session_state.get("generated_logs", []) + generated_logs
    st.session_state["generated_logs"] = combined
    df_generated = logs_to_structured_csv(combined, out_csv="capture/generated_traffic.csv")
    return df_generated


def render_page_header(title: str, subtitle: str, eyebrow: str = "Operations"):
    st.markdown(
        f"""
        <div class="light-shell">
            <p class="section-kicker">{eyebrow}</p>
            <h2 class="section-title">{title}</h2>
            <p class="section-body">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Dashboard ─────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown(
        """
        <section class="hero-shell">
            <p class="hero-kicker">Hybrid WAF Command Interface</p>
            <h1 class="hero-title">Hybrid Web Application Firewall</h1>
            <p class="hero-sub">Signature rules and anomaly intelligence fused into a real-time policy engine. Tune thresholds, simulate hostile web traffic, and inspect risk scoring with operational clarity.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_page_header(
        "System Architecture",
        "Five-stage pipeline engineered for deterministic signatures and adaptive anomaly detection.",
        eyebrow="Engineering Blueprint",
    )

    architecture_steps = [
        {"idx": "01", "icon": "traffic", "title": "Traffic Input", "desc": "Ingest live flows or replayed PCAP/CSV streams."},
        {"idx": "02", "icon": "schema", "title": "Feature Extraction", "desc": "Derive 35 normalized features per session."},
        {"idx": "03", "icon": "gavel", "title": "Rule Engine", "desc": "Run deterministic signature checks across rule sets."},
        {"idx": "04", "icon": "psychology", "title": "ML Detection", "desc": "Score anomalies with Isolation Forest."},
        {"idx": "05", "icon": "shield", "title": "Risk Fusion", "desc": "Merge scores into ALLOW, ALERT, or BLOCK."},
    ]

    architecture_markup = ['<section class="architecture-shell"><div class="architecture-track">']
    for idx, step in enumerate(architecture_steps):
        architecture_markup.append(
            f'<article class="architecture-step">'
            f'<div class="architecture-head">'
            f'<span class="architecture-icon"><span class="material-symbols-outlined">{step["icon"]}</span></span>'
            f'<span class="architecture-index">STAGE {step["idx"]}</span>'
            f'</div>'
            f'<h3 class="architecture-title">{step["title"]}</h3>'
            f'<p class="architecture-desc">{step["desc"]}</p>'
            f'</article>'
        )
        if idx < len(architecture_steps) - 1:
            architecture_markup.append(
                '<div class="architecture-connector"><span class="architecture-connector-line"></span></div>'
            )
    architecture_markup.append('</div></section>')
    st.markdown(''.join(architecture_markup), unsafe_allow_html=True)

    render_page_header(
        "Quick Demo",
        "Run a simulated flow and inspect how rule logic and anomaly scoring converge on action.",
        eyebrow="Live Simulation",
    )

    # Generate a demo flow
    if st.button("Simulate Random Traffic Flow"):
        scenario = np.random.choice(["Normal", "Port Scan", "SYN Flood", "Brute Force SSH", "Unknown Anomaly"])

        demo_flows = {
            "Normal": {
                "Flow Duration": 50000, "Protocol": 6, "Source Port": 55234,
                "Destination Port": 443, "Total Fwd Packets": 12,
                "Total Length of Fwd Packets": 2400, "Flow Bytes/s": 480,
                "Flow Packets/s": 8, "Fwd Packet Length Mean": 200,
                "Bwd Packet Length Mean": 180, "SYN Flag Count": 1,
                "ACK Flag Count": 10, "FIN Flag Count": 1, "RST Flag Count": 0,
                "PSH Flag Count": 3, "URG Flag Count": 0,
                "Average Packet Size": 200, "Active Mean": 1000, "Idle Mean": 500,
            },
            "Port Scan": {
                "Flow Duration": 100, "Protocol": 6, "Source Port": 45678,
                "Destination Port": 22, "Total Fwd Packets": 50,
                "Total Length of Fwd Packets": 300, "Flow Bytes/s": 3000,
                "Flow Packets/s": 500, "Fwd Packet Length Mean": 6,
                "Bwd Packet Length Mean": 0, "SYN Flag Count": 48,
                "ACK Flag Count": 0, "FIN Flag Count": 2, "RST Flag Count": 1,
                "PSH Flag Count": 0, "URG Flag Count": 0,
                "Average Packet Size": 6, "Active Mean": 50, "Idle Mean": 10,
            },
            "SYN Flood": {
                "Flow Duration": 500, "Protocol": 6, "Source Port": 12345,
                "Destination Port": 80, "Total Fwd Packets": 5000,
                "Total Length of Fwd Packets": 30000, "Flow Bytes/s": 60000,
                "Flow Packets/s": 10000, "Fwd Packet Length Mean": 6,
                "Bwd Packet Length Mean": 0, "SYN Flag Count": 4990,
                "ACK Flag Count": 2, "FIN Flag Count": 0, "RST Flag Count": 0,
                "PSH Flag Count": 0, "URG Flag Count": 0,
                "Average Packet Size": 6, "Active Mean": 100, "Idle Mean": 0,
            },
            "Brute Force SSH": {
                "Flow Duration": 20000, "Protocol": 6, "Source Port": 34567,
                "Destination Port": 22, "Total Fwd Packets": 200,
                "Total Length of Fwd Packets": 8000, "Flow Bytes/s": 400,
                "Flow Packets/s": 10, "Fwd Packet Length Mean": 40,
                "Bwd Packet Length Mean": 60, "SYN Flag Count": 5,
                "ACK Flag Count": 190, "FIN Flag Count": 2, "RST Flag Count": 40,
                "PSH Flag Count": 50, "URG Flag Count": 0,
                "Average Packet Size": 40, "Active Mean": 500, "Idle Mean": 200,
            },
            "Unknown Anomaly": {
                "Flow Duration": 99999, "Protocol": 17, "Source Port": 44444,
                "Destination Port": 9999, "Total Fwd Packets": 1500,
                "Total Length of Fwd Packets": 999999, "Flow Bytes/s": 9999,
                "Flow Packets/s": 15, "Fwd Packet Length Mean": 666,
                "Bwd Packet Length Mean": 999, "SYN Flag Count": 3,
                "ACK Flag Count": 1400, "FIN Flag Count": 0, "RST Flag Count": 0,
                "PSH Flag Count": 200, "URG Flag Count": 15,
                "Average Packet Size": 666, "Active Mean": 99999, "Idle Mean": 0,
            },
        }

        raw = demo_flows[scenario]

        # Process
        extractor.fit(pd.DataFrame([raw]))
        ff = extractor.transform(raw)
        rule_out = rule_engine.evaluate(ff)
        ml_out = safe_ml_predict(ff, anomaly_hint=scenario != "Normal")
        decision = fusion.decide(rule_out, ml_out)

        # Display
        st.markdown(f"**Simulated Scenario:** `{scenario}`")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Action",     decision.action)
        c2.metric("Risk Score", f"{decision.risk_score:.2f}")
        c3.metric("Rule Score", f"{decision.rule_score:.2f}")
        c4.metric("ML Score",   f"{decision.ml_score:.2f}")

        # Risk gauge bar
        pct = int(decision.risk_score * 100)
        color = {"ALLOW": "#22c55e", "ALERT": "#f59e0b", "BLOCK": "#ef4444"}.get(decision.action, "#94a3b8")
        st.markdown(f"""
        <div style="margin:15px 0">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-weight:700">Risk Score</span>
                <span style="color:{color};font-weight:700">{pct}%</span>
            </div>
            <div style="background:rgba(236,231,222,0.14);border-radius:14px;height:14px;border:1px solid rgba(236,231,222,0.18)">
                <div style="background:{color};width:{pct}%;height:14px;border-radius:14px;transition:width 0.5s"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Reasoning:** {decision.reasoning}")

        if rule_out.any_match:
            st.markdown("**Matched Signature Rules:**")
            for r in rule_out.matched_rules:
                st.code(f"[{r.rule_id}] {r.rule_name} | Severity: {r.severity} | Confidence: {r.confidence}")


# ── Analyze Traffic ───────────────────────────────────────────────────────────
elif page == "Traffic Generator":
    render_page_header(
        "Traffic Generator",
        "Generate controlled localhost activity, attack simulations, or randomized mixed traffic and append events to capture/generated_traffic.csv.",
        eyebrow="Synthetic Input",
    )

    target_url = st.text_input("Local HTTP target", value="http://127.0.0.1:8000")
    base_delay = st.slider("Request delay (seconds)", 0.0, 0.3, 0.05, 0.01)

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.markdown("#### Normal Traffic")
        normal_n = st.number_input("Normal request count", min_value=1, max_value=500, value=30, step=1)
        if st.button("Run Normal Traffic"):
            try:
                df_gen = generate_demo_csv("normal", target_url=target_url, n=int(normal_n), delay=float(base_delay))
                st.success(f"Generated normal traffic rows: {len(df_gen)}")
            except Exception as exc:
                st.error(str(exc))

    with row1_col2:
        st.markdown("#### Payload Tests")
        payload_text = st.text_area(
            "Payload strings (one per line)",
            value="' OR 1=1\n<script>alert(1)</script>\nadmin'--",
            height=130,
        )
        if st.button("Run Payload Tests"):
            payloads = [p.strip() for p in payload_text.splitlines() if p.strip()]
            try:
                df_gen = generate_demo_csv("payload", target_url=target_url, payloads=payloads, delay=float(base_delay))
                st.success(f"Generated payload traffic rows: {len(df_gen)}")
            except Exception as exc:
                st.error(str(exc))

    with row2_col1:
        st.markdown("#### Port Probe")
        start_port = st.number_input("Probe start port", min_value=1, max_value=65535, value=8000, step=1)
        end_port = st.number_input("Probe end port", min_value=1, max_value=65535, value=8015, step=1)
        if st.button("Run Probe Test"):
            try:
                df_gen = generate_demo_csv(
                    "probe",
                    target_ip="127.0.0.1",
                    start_port=int(start_port),
                    end_port=int(end_port),
                    delay=float(base_delay),
                )
                st.success(f"Generated probe traffic rows: {len(df_gen)}")
            except Exception as exc:
                st.error(str(exc))

    with row2_col2:
        st.markdown("#### Connection Burst")
        burst_port = st.number_input("Burst target port", min_value=1, max_value=65535, value=8000, step=1)
        burst_count = st.number_input("Burst connection count", min_value=1, max_value=500, value=40, step=1)
        burst_concurrency = st.number_input("Burst concurrency", min_value=1, max_value=30, value=5, step=1)
        if st.button("Run Connection Burst"):
            try:
                df_gen = generate_demo_csv(
                    "burst",
                    target_ip="127.0.0.1",
                    port=int(burst_port),
                    count=int(burst_count),
                    concurrency=int(burst_concurrency),
                    delay=float(base_delay),
                )
                st.success(f"Generated burst traffic rows: {len(df_gen)}")
            except Exception as exc:
                st.error(str(exc))

    st.markdown("---")
    ext_col1, ext_col2 = st.columns(2)

    with ext_col1:
        st.markdown("#### Path Fuzzing")
        path_text = st.text_area(
            "Suspicious paths (one per line)",
            value="/admin\n/.env\n/../../etc/passwd\n/wp-admin",
            height=120,
            key="path_fuzz_input",
        )
        if st.button("Run Path Fuzzing"):
            paths = [p.strip() for p in path_text.splitlines() if p.strip()]
            try:
                df_gen = generate_demo_csv("path_fuzz", target_url=target_url, paths=paths, delay=float(base_delay))
                st.success(f"Generated path fuzz traffic rows: {len(df_gen)}")
            except Exception as exc:
                st.error(str(exc))

    with ext_col2:
        st.markdown("#### Login Burst Simulation")
        usernames_text = st.text_area(
            "Usernames (one per line)",
            value="admin\nroot\nguest\ntest",
            height=84,
            key="login_usernames_input",
        )
        passwords_text = st.text_area(
            "Passwords (one per line)",
            value="123456\npassword\nadmin123\nqwerty",
            height=84,
            key="login_passwords_input",
        )
        login_attempts = st.number_input("Login attempt count", min_value=1, max_value=500, value=40, step=1)
        if st.button("Run Login Burst"):
            usernames = [u.strip() for u in usernames_text.splitlines() if u.strip()]
            passwords = [p.strip() for p in passwords_text.splitlines() if p.strip()]
            try:
                df_gen = generate_demo_csv(
                    "login_burst",
                    target_url=target_url,
                    usernames=usernames,
                    passwords=passwords,
                    attempts=int(login_attempts),
                    delay=float(base_delay),
                )
                st.success(f"Generated login burst traffic rows: {len(df_gen)}")
            except Exception as exc:
                st.error(str(exc))

    st.markdown("---")
    st.markdown("#### Random Mixed Traffic")
    mix_col1, mix_col2, mix_col3 = st.columns(3)
    with mix_col1:
        mix_total = st.number_input("Total mixed events", min_value=10, max_value=2000, value=120, step=10)
    with mix_col2:
        mix_ratio = st.slider("Attack-like ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
    with mix_col3:
        mix_target_ip = st.text_input("Local target IP", value="127.0.0.1")

    if st.button("Run Random Mixed Traffic"):
        try:
            df_gen = generate_demo_csv(
                "random_mix",
                target_url=target_url,
                target_ip=mix_target_ip,
                total_events=int(mix_total),
                malicious_ratio=float(mix_ratio),
                delay=float(base_delay),
            )
            st.success(f"Generated mixed traffic rows: {len(df_gen)}")
        except Exception as exc:
            st.error(str(exc))

    current = st.session_state.get("generated_logs", [])
    st.info(f"Current generated events in session: {len(current)}")
    if st.button("Clear Generated Session Logs"):
        st.session_state["generated_logs"] = []
        logs_to_structured_csv([], out_csv="capture/generated_traffic.csv")
        st.success("Cleared session logs and reset capture/generated_traffic.csv")


elif page == "Analyze Traffic":
    render_page_header(
        "Analyze Traffic",
        "Upload a dataset or load generated localhost events for full flow-by-flow risk analysis.",
        eyebrow="Detection Review",
    )

    source = st.radio("Data source", ["Upload File (CSV/PCAP)", "Generated Dataset"], horizontal=True)

    uploaded = None
    df_raw = None
    source_name = ""

    if source == "Upload File (CSV/PCAP)":
        uploaded = st.file_uploader("Upload Network Traffic File", type=["csv", "pcap", "pcapng"])
        if uploaded is not None:
            if uploaded.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded, low_memory=False)
                source_name = "uploaded CSV"
            else:
                # Handle PCAP files using Scapy
                import tempfile
                from capture.traffic_capture import _extract_packet_row
                # Optional dependency checking
                try:
                    from scapy.all import rdpcap
                except ImportError:
                    st.error("Scapy is not installed. Please run `pip install scapy` to analyze PCAP files.")
                    rdpcap = None
                
                if rdpcap:
                    with st.spinner("Extracting flow data from PCAP file..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
                            tmp.write(uploaded.getbuffer())
                            tmp_path = tmp.name
                            
                        try:
                            packets = rdpcap(tmp_path)
                            rows = []
                            for pkt in packets:
                                row = _extract_packet_row(pkt)
                                if row is not None:
                                    rows.append(row)
                            
                            if len(rows) > 0:
                                df_raw = pd.DataFrame(rows)
                                source_name = "uploaded PCAP"
                            else:
                                st.warning("PCAP file processed but contained zero valid IP packets to analyze. Make sure your capture was on the right interface or had readable traffic.")
                        except Exception as e:
                            st.error(f"Failed to read PCAP: {e}")
                        finally:
                            os.remove(tmp_path)
    else:
        generated_path = "capture/generated_traffic.csv"
        if os.path.exists(generated_path):
            df_raw = pd.read_csv(generated_path, low_memory=False)
            source_name = generated_path
            st.info(f"Loaded generated dataset from {generated_path}")
        else:
            st.warning("Generated dataset not found. Use the Traffic Generator page first.")

    if df_raw is not None:
        max_rows = st.slider("Max rows to analyze", 50, 2000, 300, 50)

        if st.button("Run Analysis"):
            with st.spinner("Analyzing traffic..."):
                st.info(f"Loaded {len(df_raw)} rows from {source_name}. Analyzing first {max_rows}...")
                results_df = analyze_df(df_raw, max_rows=max_rows)

            # Summary
            st.markdown("---")
            st.markdown("### Results Summary")
            c1, c2, c3, c4 = st.columns(4)
            total = len(results_df)
            blocks = (results_df["Action"] == "BLOCK").sum()
            alerts = (results_df["Action"] == "ALERT").sum()
            allows = (results_df["Action"] == "ALLOW").sum()

            c1.metric("Total Flows",  total)
            c2.metric("Blocked",  blocks, delta=f"{blocks/total*100:.1f}%")
            c3.metric("Alerts",   alerts, delta=f"{alerts/total*100:.1f}%")
            c4.metric("Allowed",  allows, delta=f"{allows/total*100:.1f}%")

            if "Label" in results_df.columns:
                truth = results_df["Label"].astype(str).tolist()
                pred = actions_to_labels(results_df["Action"].astype(str).tolist())
                metrics = evaluate_predictions(truth, pred)

                st.markdown("### Evaluation Metrics")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                m2.metric("Precision", f"{metrics['precision']:.3f}")
                m3.metric("Recall", f"{metrics['recall']:.3f}")
                m4.metric("F1 Score", f"{metrics['f1_score']:.3f}")

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                action_counts = results_df["Action"].value_counts()
                st.bar_chart(action_counts)

            with col2:
                st.markdown("**Risk Score Distribution**")
                hist_data = pd.DataFrame({"Risk Score": results_df["Risk Score"]})
                st.bar_chart(hist_data["Risk Score"].value_counts(bins=10, sort=False))

            # Table
            st.markdown("### Flow-by-Flow Results")
            st.dataframe(
                results_df.drop(columns=["Reasoning"]),
                use_container_width=True,
                height=400,
            )

            # Download
            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                csv_out,
                "waf_results.csv",
                "text/csv",
            )


# ── Train Model ───────────────────────────────────────────────────────────────
elif page == "Train Model":
    render_page_header(
        "Train ML Model",
        "Upload BENIGN flow data and retrain Isolation Forest with configurable contamination and tree depth.",
        eyebrow="Model Operations",
    )

    train_file = st.file_uploader("Upload Training CSV", type=["csv"], key="train")
    contamination = st.slider("Contamination (expected anomaly % in training data)", 0.01, 0.20, 0.05, 0.01)
    n_estimators  = st.slider("Number of Trees", 50, 300, 100, 50)

    if train_file and st.button("Train Model"):
        with st.spinner("Training..."):
            df_train = pd.read_csv(train_file, low_memory=False)

            # Filter to benign
            label_col = None
            for col in df_train.columns:
                if col.strip().lower() == "label":
                    label_col = col
                    break

            if label_col:
                normal_df = df_train[df_train[label_col].str.upper().str.contains("BENIGN")]
                st.info(f"Found {len(normal_df)} BENIGN flows out of {len(df_train)} total.")
            else:
                normal_df = df_train
                st.info(f"No 'Label' column found. Using all {len(df_train)} rows.")

            extractor.fit(df_train)
            normal_flows = extractor.transform_df(normal_df)

            ml_detector.contamination = contamination
            ml_detector.n_estimators  = n_estimators
            ml_detector.train(normal_flows)
            ml_detector.save("models/isolation_forest.pkl")

        st.success(f"Model trained on {len(normal_flows)} flows and saved.")


# ── Rules Viewer ──────────────────────────────────────────────────────────────
elif page == "Rules Viewer":
    render_page_header(
        "Signature Rules",
        f"{len(rule_engine.rules)} rules loaded from config/rules.yaml.",
        eyebrow="Rule Catalog",
    )

    # Filter
    categories = list(set(r.get("category", "Misc") for r in rule_engine.rules))
    selected_cats = st.multiselect("Filter by Category", categories, default=categories)

    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    selected_sev = st.multiselect("Filter by Severity", severities, default=severities)

    filtered = [
        r for r in rule_engine.rules
        if r.get("category", "Misc") in selected_cats
        and r.get("severity", "LOW") in selected_sev
    ]

    SEV_COLORS = {
        "CRITICAL": "#ef4444",
        "HIGH":     "#f97316",
        "MEDIUM":   "#f59e0b",
        "LOW":      "#22c55e",
    }

    for rule in filtered:
        sev   = rule.get("severity", "LOW")
        color = SEV_COLORS.get(sev, "#888")
        with st.expander(f"[{rule['id']}] {rule['name']}  |  {sev}  |  {rule.get('category')}"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**ID:** `{rule['id']}`")
                st.markdown(f"**Category:** `{rule.get('category')}`")
                st.markdown(f"**Severity:** <span style='color:{color};font-weight:700'>{sev}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** `{rule.get('confidence', 0.8)}`")
            with col2:
                st.markdown("**Conditions (ALL must be true):**")
                for cond in rule.get("conditions", []):
                    st.code(f"{cond['field']}  {cond['op']}  {cond['value']}")

# ── Wireshark PCAP Capture ──────────────────────────────────────────────────
elif page == "Wireshark PCAP Capture":
    import subprocess
    import threading
    import requests
    import shutil
    
    render_page_header(
        "Wireshark PCAP Capture",
        "Generate traffic to a target URL while Wireshark silently tracks it in the background, outputting a raw PCAP file for analysis.",
        eyebrow="Live Sniffer",
    )
    
    c1, c2 = st.columns([2, 1])
    with c1:
        target_url = st.text_input("Target URL to generate traffic specifically to", "http://www.google.com")
    with c2:
        duration = st.number_input("Capture Duration (Seconds)", min_value=5, max_value=60, value=10, step=5)
        
    interface = st.text_input("Wireshark Interface Number (Try '1', 'Wi-Fi', 'Ethernet')", "")
    st.markdown("*Hint: If captures are empty, tshark doesn't know which adapter to use. Open Wireshark and look at the interface names or try '1', '2', '3'.*")
    
    # Locate tshark
    tshark_path = shutil.which("tshark.exe")
    if not tshark_path:
        # fallback Windows check
        if os.path.exists(r"C:\Program Files\Wireshark\tshark.exe"):
            tshark_path = r"C:\Program Files\Wireshark\tshark.exe"
            
    if not tshark_path:
        st.error("❌ **tshark.exe not found.** Wireshark must be installed and added to your system PATH to use this feature.")
    else:
        if st.button("Start Wireshark Capture", type="primary"):
            pcap_file = "capture_output.pcapng"
            
            # Remove old file if exists
            if os.path.exists(pcap_file):
                os.remove(pcap_file)
                
            cmd = [tshark_path, "-a", f"duration:{duration}", "-w", pcap_file, "-Q"]
            if interface:
                cmd.extend(["-i", interface])
                
            st.info(f"🚀 Started {duration}-second capture. Generating traffic on {target_url}...")
            
            with st.spinner(f"Wireshark is listening... ({duration} seconds)"):
                
                # Function to generate background traffic
                def ping_target():
                    for _ in range(duration):
                        try:
                            # Use timeout to not permanently hang, send request
                            requests.get(target_url, timeout=2.0)
                        except Exception:
                            pass
                        time.sleep(1)
                
                # Start network spawner threaded so it doesn't block tshark subprocess
                traffic_thread = threading.Thread(target=ping_target)
                traffic_thread.start()
                
                try:
                    # Run tshark blocking (it self-terminates after defined duration)
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    st.error(f"Capturing failed: {e}")
                
            traffic_thread.join(timeout=1.0)
            
            if os.path.exists(pcap_file) and os.path.getsize(pcap_file) > 100:
                st.success(f"✅ Capture Complete! Generated {os.path.getsize(pcap_file)} bytes.")
                with open(pcap_file, "rb") as f:
                    file_bytes = f.read()
                
                st.download_button(
                    label="📥 Download .pcapng file (Open with Wireshark)",
                    data=file_bytes,
                    file_name=f"Traffic_Capture_{int(time.time())}.pcapng",
                    mime="application/vnd.tcpdump.pcap"
                )
            else:
                st.warning("⚠️ Capture finished, but the PCAP file seems empty or missing. Check your 'Network Interface' settings or run Streamlit as Administrator.")
