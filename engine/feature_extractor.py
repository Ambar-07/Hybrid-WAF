"""
Feature Extractor
-----------------
Takes raw network flow data (CSV row or dict) and returns
a normalized feature vector for the rule engine + ML model.

Compatible with CICIDS2017 column names out of the box.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


# ── The features we actually use ──────────────────────────────────────────────
SELECTED_FEATURES = [
    "duration",
    "protocol_type",      # will be encoded
    "src_port",
    "dst_port",
    "pkt_count",
    "byte_count",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "urg_flag_count",
    "avg_pkt_size",
    "active_mean",
    "idle_mean",
    "packet_rate",
    "unique_ports",
    "connection_count",
    "avg_packet_length",
    "flow_duration",
    "flags_count",
]

# CICIDS2017 → our internal name mapping
CICIDS_COLUMN_MAP = {
    "Flow Duration":             "duration",
    "Protocol":                  "protocol_type",
    "Source Port":               "src_port",
    "Destination Port":          "dst_port",
    "Total Fwd Packets":         "pkt_count",
    "Total Length of Fwd Packets": "byte_count",
    "Flow Bytes/s":              "flow_bytes_per_sec",
    "Flow Packets/s":            "flow_pkts_per_sec",
    "Fwd Packet Length Mean":    "fwd_pkt_len_mean",
    "Bwd Packet Length Mean":    "bwd_pkt_len_mean",
    "SYN Flag Count":            "syn_flag_count",
    "ACK Flag Count":            "ack_flag_count",
    "FIN Flag Count":            "fin_flag_count",
    "RST Flag Count":            "rst_flag_count",
    "PSH Flag Count":            "psh_flag_count",
    "URG Flag Count":            "urg_flag_count",
    "Average Packet Size":       "avg_pkt_size",
    "Active Mean":               "active_mean",
    "Idle Mean":                 "idle_mean",
    "Label":                     "label",
    "Packet Length":             "packet_length",
}

PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1, 6: 6, 17: 17, 1: 1}


@dataclass
class FlowFeatures:
    """Holds extracted features for one network flow."""
    raw: Dict[str, Any] = field(default_factory=dict)
    normalized: np.ndarray = field(default_factory=lambda: np.array([]))
    label: str = "UNKNOWN"          # ground truth if available
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: str = ""


class FeatureExtractor:
    """
    Usage
    -----
    extractor = FeatureExtractor()
    extractor.fit(df_train)          # learns min/max for normalization
    features  = extractor.transform(row_dict)   # single flow
    df_out    = extractor.transform_df(df)      # whole dataframe
    """

    def __init__(self):
        self._min: Dict[str, float] = {}
        self._max: Dict[str, float] = {}
        self.fitted = False

    # ── Fit (learn normalization stats from training data) ────────────────────
    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        df = self._rename_columns(df)
        df = self._clean(df)
        df = self._augment_features(df)
        for feat in SELECTED_FEATURES:
            if feat in df.columns and feat != "protocol_type":
                self._min[feat] = float(df[feat].min())
                self._max[feat] = float(df[feat].max())
        self.fitted = True
        print(f"[FeatureExtractor] Fitted on {len(df)} rows.")
        return self

    # ── Transform a single flow dict ──────────────────────────────────────────
    def transform(self, row: Dict[str, Any]) -> FlowFeatures:
        row = self._rename_row(row)
        row = self._augment_row(row)
        row = self._fill_missing(row)

        ff = FlowFeatures(
            raw=row,
            label=str(row.get("label", "UNKNOWN")),
            src_ip=str(row.get("src_ip", "")),
            dst_ip=str(row.get("dst_ip", "")),
            src_port=int(row.get("src_port", 0)),
            dst_port=int(row.get("dst_port", 0)),
            protocol=str(row.get("protocol_type", "")),
        )

        vec = []
        for feat in SELECTED_FEATURES:
            val = float(row.get(feat, 0))
            if feat == "protocol_type":
                val = float(PROTOCOL_MAP.get(val, 0))
            else:
                val = self._normalize(feat, val)
            vec.append(val)

        ff.normalized = np.array(vec, dtype=np.float32)
        return ff

    # ── Transform a whole DataFrame ───────────────────────────────────────────
    def transform_df(self, df: pd.DataFrame):
        df = self._rename_columns(df)
        df = self._clean(df)
        df = self._augment_features(df)
        return [self.transform(row) for row in df.to_dict(orient="records")]

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]
        return df.rename(columns=CICIDS_COLUMN_MAP)

    def _rename_row(self, row: Dict) -> Dict:
        return {CICIDS_COLUMN_MAP.get(k.strip(), k.strip()): v for k, v in row.items()}

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df

    def _fill_missing(self, row: Dict) -> Dict:
        for feat in SELECTED_FEATURES:
            if feat not in row:
                row[feat] = 0
        return row

    def _augment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "packet_length" in df.columns and "avg_packet_length" not in df.columns:
            df["avg_packet_length"] = df["packet_length"]

        if "duration" not in df.columns:
            df["duration"] = 1.0
        df["flow_duration"] = pd.to_numeric(df.get("duration", 1.0), errors="coerce").fillna(1.0).clip(lower=0.001)

        if "pkt_count" not in df.columns:
            df["pkt_count"] = 1
        pkt = pd.to_numeric(df.get("pkt_count", 1), errors="coerce").fillna(1.0)
        dur = pd.to_numeric(df.get("flow_duration", 1.0), errors="coerce").fillna(1.0).clip(lower=0.001)
        df["packet_rate"] = pkt / dur

        if "src_ip" in df.columns:
            if "dst_port" in df.columns:
                df["unique_ports"] = df.groupby("src_ip")["dst_port"].transform("nunique")
            else:
                df["unique_ports"] = 1
            df["connection_count"] = df.groupby("src_ip")["src_ip"].transform("count")
        else:
            df["unique_ports"] = 1
            df["connection_count"] = 1

        flag_cols = [
            "syn_flag_count",
            "ack_flag_count",
            "fin_flag_count",
            "rst_flag_count",
            "psh_flag_count",
            "urg_flag_count",
        ]
        for c in flag_cols:
            if c not in df.columns:
                df[c] = 0

        if "flags_count" not in df.columns:
            df["flags_count"] = df[flag_cols].sum(axis=1)

        if "avg_pkt_size" not in df.columns:
            df["avg_pkt_size"] = pd.to_numeric(df.get("avg_packet_length", 0), errors="coerce").fillna(0.0)

        if "byte_count" not in df.columns and "packet_length" in df.columns:
            df["byte_count"] = pd.to_numeric(df["packet_length"], errors="coerce").fillna(0.0) * pkt

        if "flow_pkts_per_sec" not in df.columns:
            df["flow_pkts_per_sec"] = df["packet_rate"]
        if "flow_bytes_per_sec" not in df.columns and "byte_count" in df.columns:
            bc = pd.to_numeric(df["byte_count"], errors="coerce").fillna(0.0)
            df["flow_bytes_per_sec"] = bc / dur

        return df

    def _augment_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(row)
        duration = float(row.get("duration", 1.0) or 1.0)
        duration = max(duration, 0.001)
        pkt_count = float(row.get("pkt_count", 1.0) or 1.0)

        row["flow_duration"] = float(row.get("flow_duration", duration))
        row["packet_rate"] = float(row.get("packet_rate", pkt_count / duration))
        row["unique_ports"] = float(row.get("unique_ports", 1))
        row["connection_count"] = float(row.get("connection_count", 1))
        row["avg_packet_length"] = float(row.get("avg_packet_length", row.get("packet_length", 0)))

        if "flags_count" not in row:
            row["flags_count"] = float(
                row.get("syn_flag_count", 0)
                + row.get("ack_flag_count", 0)
                + row.get("fin_flag_count", 0)
                + row.get("rst_flag_count", 0)
                + row.get("psh_flag_count", 0)
                + row.get("urg_flag_count", 0)
            )

        if "avg_pkt_size" not in row:
            row["avg_pkt_size"] = float(row["avg_packet_length"])
        if "flow_pkts_per_sec" not in row:
            row["flow_pkts_per_sec"] = float(row["packet_rate"])

        byte_count = row.get("byte_count", None)
        if byte_count in (None, ""):
            byte_count = float(row.get("packet_length", 0) or 0) * pkt_count
            row["byte_count"] = float(byte_count)
        if "flow_bytes_per_sec" not in row:
            row["flow_bytes_per_sec"] = float(row["byte_count"]) / duration

        return row

    def _normalize(self, feat: str, val: float) -> float:
        mn = self._min.get(feat, 0)
        mx = self._max.get(feat, 1)
        if mx == mn:
            return 0.0
        return float(np.clip((val - mn) / (mx - mn), 0, 1))
