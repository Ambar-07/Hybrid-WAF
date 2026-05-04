"""
Feature Extractor
-----------------
Takes raw network flow data (CSV row or dict) and returns
a normalized feature vector for the rule engine + ML model.

Compatible with CICIDS2017 column names out of the box.
"""

import pandas as pd
import numpy as np
import re
from dataclasses import dataclass, field
from typing import Dict, Any


# ── The features we actually use ──────────────────────────────────────────────
# Features used by models/best_model.pkl.
# Keep train and inference on this same order.
SELECTED_FEATURES = [
    "dst_port",                    # Destination Port
    "total_length_of_fwd_packets", # Total Length of Fwd Packets
    "total_length_of_bwd_packets", # Total Length of Bwd Packets
    "bwd_pkt_len_max",             # Bwd Packet Length Max
    "bwd_pkt_len_mean",            # Bwd Packet Length Mean
    "max_pkt_length",              # Max Packet Length
    "pkt_len_mean",                # Packet Length Mean
    "pkt_len_std",                 # Packet Length Std
    "pkt_len_var",                 # Packet Length Variance
    "avg_pkt_size",                # Average Packet Size
    "avg_bwd_segment_size",        # Avg Bwd Segment Size
    "subflow_fwd_bytes",           # Subflow Fwd Bytes
    "subflow_bwd_bytes",           # Subflow Bwd Bytes
    "init_win_bytes_forward",      # Init_Win_bytes_forward
    "init_win_bytes_backward",     # Init_Win_bytes_backward
]

# CICIDS2017 → our internal name mapping
CICIDS_COLUMN_MAP = {
    "Flow Duration":             "duration",
    "Protocol":                  "protocol_type",
    "Source Port":               "src_port",
    "Destination Port":          "dst_port",
    "Total Fwd Packets":         "fwd_pkt_count",
    "Total Bwd Packets":         "bwd_pkt_count",
    "Total Backward Packets":    "bwd_pkt_count",
    "Total Length of Fwd Packets": "total_length_of_fwd_packets",
    "Total Length of Bwd Packets": "total_length_of_bwd_packets",
    "Flow Bytes/s":              "flow_bytes_per_sec",
    "Flow Packets/s":            "flow_pkts_per_sec",
    "Fwd Packet Length Mean":    "fwd_pkt_len_mean",
    "Bwd Packet Length Mean":    "bwd_pkt_len_mean",
    "Packet Length Mean":        "pkt_len_mean",
    "Packet Length Std":         "pkt_len_std",
    "Packet Length Variance":    "pkt_len_var",
    "Max Packet Length":         "max_pkt_length",
    "Subflow Bwd Bytes":         "subflow_bwd_bytes",
    "Avg Bwd Segment Size":      "avg_bwd_segment_size",
    "Bwd Packet Length Max":     "bwd_pkt_len_max",
    "Init_Win_bytes_backward":   "init_win_bytes_backward",
    "Subflow Fwd Bytes":         "subflow_fwd_bytes",
    "Init_Win_bytes_forward":    "init_win_bytes_forward",
    "Average Packet Size":       "avg_pkt_size",
    "Active Mean":               "active_mean",
    "Idle Mean":                 "idle_mean",
    "Window Size":               "window_size",
    "Label":                     "label",
    "SYN Flag Count":            "syn_flag_count",
    "ACK Flag Count":            "ack_flag_count",
    "FIN Flag Count":            "fin_flag_count",
    "RST Flag Count":            "rst_flag_count",
    "PSH Flag Count":            "psh_flag_count",
    "URG Flag Count":            "urg_flag_count",
}

AUTO_COLUMN_ALIASES = {
    "duration": ["duration", "flow_duration", "flowduration", "flow_time", "elapsed", "elapsed_time", "time_delta"],
    "protocol_type": ["protocol", "proto", "protocol_type", "ip_proto", "l4_protocol", "transport_protocol"],
    "src_ip": ["src_ip", "source_ip", "source_address", "src_addr", "source", "ip_src", "ip.src", "ipv6_src", "ipv6.src"],
    "dst_ip": ["dst_ip", "destination_ip", "dest_ip", "dst_addr", "destination_address", "destination", "ip_dst", "ip.dst", "ipv6_dst", "ipv6.dst"],
    "src_port": ["src_port", "source_port", "sport", "srcport", "sourceport", "source_port_number", "tcp_srcport", "udp_srcport", "tcp.srcport", "udp.srcport"],
    "dst_port": ["dst_port", "destination_port", "dest_port", "dport", "dstport", "destinationport", "destination_port_number", "tcp_dstport", "udp_dstport", "tcp.dstport", "udp.dstport"],
    "pkt_count": ["pkt_count", "packet_count", "packets", "total_packets", "flow_packets", "packet_total", "tot_pkts", "total_pkts"],
    "fwd_pkt_count": ["fwd_pkt_count", "forward_packets", "fwd_packets", "total_fwd_packets", "tot_fwd_pkts", "total_fwd_pkts", "fwd_pkts"],
    "bwd_pkt_count": ["bwd_pkt_count", "backward_packets", "bwd_packets", "total_backward_packets", "total_bwd_packets", "tot_bwd_pkts", "total_bwd_pkts", "bwd_pkts"],
    "byte_count": ["byte_count", "bytes", "total_bytes", "flow_bytes", "bytes_total"],
    "total_length_of_fwd_packets": ["fwd_bytes", "forward_bytes", "total_fwd_bytes", "total_length_of_fwd_packets", "totlen_fwd_pkts", "tot_len_fwd_pkts", "fwd_pkt_bytes"],
    "total_length_of_bwd_packets": ["bwd_bytes", "backward_bytes", "total_bwd_bytes", "total_length_of_bwd_packets", "totlen_bwd_pkts", "tot_len_bwd_pkts", "bwd_pkt_bytes"],
    "flow_bytes_per_sec": ["flow_bytes_per_sec", "flow_bytes_s", "flow_bytes_sec", "bytes_per_sec", "bytes_sec", "bps"],
    "flow_pkts_per_sec": ["flow_pkts_per_sec", "flow_packets_per_sec", "flow_pkts_s", "flow_pkts_sec", "packets_per_sec", "pkts_per_sec", "pps"],
    "fwd_pkt_len_mean": ["fwd_pkt_len_mean", "fwd_packet_length_mean", "forward_packet_length_mean", "avg_fwd_pkt_size"],
    "bwd_pkt_len_mean": ["bwd_pkt_len_mean", "bwd_packet_length_mean", "backward_packet_length_mean", "bwd_pkt_length_mean", "avg_bwd_pkt_size"],
    "bwd_pkt_len_max": ["bwd_pkt_len_max", "bwd_packet_length_max", "backward_packet_length_max", "bwd_pkt_length_max", "max_bwd_pkt_size"],
    "pkt_len_mean": ["pkt_len_mean", "packet_length_mean", "pkt_length_mean", "mean_packet_length", "avg_packet_length"],
    "pkt_len_std": ["pkt_len_std", "packet_length_std", "pkt_length_std", "std_packet_length"],
    "pkt_len_var": ["pkt_len_var", "packet_length_variance", "pkt_length_variance", "packet_length_var", "pkt_length_var", "variance_packet_length"],
    "max_pkt_length": ["max_pkt_length", "packet_length_max", "pkt_length_max", "max_packet_length", "max_packet_size"],
    "avg_pkt_size": ["avg_pkt_size", "average_packet_size", "avg_packet_size", "packet_size_avg"],
    "subflow_fwd_bytes": ["subflow_fwd_bytes", "subflow_forward_bytes", "subflow_fwd_byts"],
    "subflow_bwd_bytes": ["subflow_bwd_bytes", "subflow_backward_bytes", "subflow_bwd_byts"],
    "avg_bwd_segment_size": ["avg_bwd_segment_size", "average_bwd_segment_size", "avg_bwd_segment", "avg_bwd_pkt_size"],
    "init_win_bytes_forward": ["init_win_bytes_forward", "init_win_bytes_fwd", "fwd_init_window_bytes", "init_win_bytes_fwd"],
    "init_win_bytes_backward": ["init_win_bytes_backward", "init_win_bytes_bwd", "bwd_init_window_bytes", "init_win_bytes_bwd"],
    "syn_flag_count": ["syn_flag_count", "syn_count", "syn", "tcp_syn", "tcp.flags.syn"],
    "ack_flag_count": ["ack_flag_count", "ack_count", "ack", "tcp_ack", "tcp.flags.ack"],
    "fin_flag_count": ["fin_flag_count", "fin_count", "fin", "tcp_fin", "tcp.flags.fin"],
    "rst_flag_count": ["rst_flag_count", "rst_count", "rst", "tcp_rst", "tcp.flags.reset", "tcp.flags.rst"],
    "psh_flag_count": ["psh_flag_count", "psh_count", "psh", "tcp_psh", "tcp.flags.push", "tcp.flags.psh"],
    "urg_flag_count": ["urg_flag_count", "urg_count", "urg", "tcp_urg", "tcp.flags.urg"],
    "frame_time": ["time", "timestamp", "frame_time", "frame.time", "frame.time_relative", "frame.time_epoch"],
    "frame_len": ["length", "len", "frame_len", "frame.len", "frame_length", "packet_length", "pkt_length"],
    "info": ["info", "packet_info", "_ws.col.info"],
    "active_mean": ["active_mean", "mean_active"],
    "idle_mean": ["idle_mean", "mean_idle"],
    "window_size": ["window_size", "tcp_window_size", "tcp.window_size", "tcp.window_size_value", "win_size"],
    "label": ["label", "class", "target", "attack", "attack_type"],
}

DERIVED_FEATURES = [
    "pkt_count", "byte_count", "flow_bytes_per_sec", "flow_pkts_per_sec",
    "avg_pkt_size", "pkt_len_mean", "subflow_fwd_bytes", "subflow_bwd_bytes",
    "avg_bwd_segment_size",
]

RAW_FEATURES = sorted(
    set(SELECTED_FEATURES)
    | set(AUTO_COLUMN_ALIASES)
    | set(CICIDS_COLUMN_MAP.values())
    | set(DERIVED_FEATURES)
)

PROTOCOL_MAP = {"TCP": 6, "UDP": 17, "ICMP": 1, 6: 6, 17: 17, 1: 1, "6": 6, "17": 17, "1": 1}


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
        df = self._prepare_df(df)
        for feat in SELECTED_FEATURES:
            if feat in df.columns and feat != "protocol_type":
                values = self._numeric_series(df[feat])
                self._min[feat] = float(values.min())
                self._max[feat] = float(values.max())
        self.fitted = True
        print(f"[FeatureExtractor] Fitted on {len(df)} rows.")
        return self

    def get_state(self) -> Dict[str, Any]:
        return {
            "min": dict(self._min),
            "max": dict(self._max),
            "features": list(SELECTED_FEATURES),
        }

    def set_state(self, state: Dict[str, Any]) -> "FeatureExtractor":
        self._min = {str(k): float(v) for k, v in state.get("min", {}).items()}
        self._max = {str(k): float(v) for k, v in state.get("max", {}).items()}
        self.fitted = bool(self._min and self._max)
        return self

    # ── Transform a single flow dict ──────────────────────────────────────────
    def transform(self, row: Dict[str, Any]) -> FlowFeatures:
        row = self._rename_row(row)
        row = self._derive_features_row(row)
        row = self._fill_missing(row)

        ff = FlowFeatures(
            raw=row,
            label=str(row.get("label", "UNKNOWN")),
            src_ip=str(row.get("src_ip", "")),
            dst_ip=str(row.get("dst_ip", "")),
            src_port=self._to_int(row.get("src_port", 0)),
            dst_port=self._to_int(row.get("dst_port", 0)),
            protocol=str(self._protocol_number(row.get("protocol_type", ""))),
        )

        vec = []
        for feat in SELECTED_FEATURES:
            if feat == "protocol_type":
                val = float(self._protocol_number(row.get(feat, 0)))
            else:
                val = self._to_float(row.get(feat, 0))
                val = self._normalize(feat, val)
            vec.append(val)

        ff.normalized = np.array(vec, dtype=np.float32)
        return ff

    # ── Transform a whole DataFrame ───────────────────────────────────────────
    def transform_df(self, df: pd.DataFrame):
        df = self._prepare_df(df)
        return [self.transform(row) for row in df.to_dict(orient="records")]

    # ── Internal helpers ──────────────────────────────────────────────────────
    def infer_column_mapping(self, columns) -> Dict[str, str]:
        """Return the auto-detected CSV column -> internal feature mapping."""
        return self._build_column_map(columns)

    def infer_available_features(self, df: pd.DataFrame) -> set[str]:
        """Return features available after auto-detection and derivation."""
        preview = self._prepare_df(df.head(500))
        return set(preview.columns)

    def infer_input_type(self, df: pd.DataFrame) -> str:
        """Return packet_csv for Wireshark-style packets, otherwise flow_csv."""
        renamed = self._rename_columns(df.head(20))
        return "packet_csv" if self._looks_like_packet_df(renamed) else "flow_csv"

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._rename_columns(df)
        if self._looks_like_packet_df(df):
            df = self._aggregate_packet_df(df)
        df = self._derive_features_df(df)
        return self._clean(df)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df = df.rename(columns=self._build_column_map(df.columns))
        return self._coalesce_alias_columns(df)

    def _rename_row(self, row: Dict) -> Dict:
        mapped = {}
        column_map = self._build_column_map(row.keys())
        for k, v in row.items():
            key = str(k).strip()
            mapped_key = column_map.get(key, key)
            mapped[mapped_key] = v

        return mapped

    def _build_column_map(self, columns) -> Dict[str, str]:
        available = {str(col).strip(): self._normalize_name(col) for col in columns}
        mapping = {}
        used_targets = set()

        normalized_cicids = {
            self._normalize_name(source): target
            for source, target in CICIDS_COLUMN_MAP.items()
        }

        for original in available:
            target = CICIDS_COLUMN_MAP.get(original) or normalized_cicids.get(available[original])
            if target:
                if target not in used_targets:
                    mapping[original] = target
                    used_targets.add(target)

        alias_lookup = {}
        for target, aliases in AUTO_COLUMN_ALIASES.items():
            for alias in [target, *aliases]:
                alias_lookup[self._normalize_name(alias)] = target

        for original, canonical in available.items():
            target = alias_lookup.get(canonical)
            if target and target not in used_targets:
                mapping[original] = target
                used_targets.add(target)

        return mapping

    def _coalesce_alias_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for target, aliases in AUTO_COLUMN_ALIASES.items():
            alias_names = {self._normalize_name(target)}
            alias_names.update(self._normalize_name(alias) for alias in aliases)
            candidate_indices = [
                idx for idx, col in enumerate(df.columns)
                if col == target or self._normalize_name(col) in alias_names
            ]
            if len(candidate_indices) <= 1:
                continue

            result = df.iloc[:, candidate_indices[0]]
            for idx in candidate_indices[1:]:
                result = self._combine_series(result, df.iloc[:, idx])

            keep_mask = [
                idx not in candidate_indices
                for idx in range(len(df.columns))
            ]
            df = df.loc[:, keep_mask]
            df[target] = result
        return df

    def _combine_series(self, primary: pd.Series, fallback: pd.Series) -> pd.Series:
        primary_text = primary.astype(str).str.strip()
        missing = primary.isna() | primary_text.eq("") | primary_text.eq("0") | primary_text.str.lower().eq("nan")
        return primary.where(~missing, fallback)

    def _derive_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "pkt_count" not in df and {"fwd_pkt_count", "bwd_pkt_count"}.issubset(df.columns):
            df["pkt_count"] = self._numeric_series(df["fwd_pkt_count"]) + self._numeric_series(df["bwd_pkt_count"])
        elif "pkt_count" not in df and "fwd_pkt_count" in df:
            df["pkt_count"] = self._numeric_series(df["fwd_pkt_count"])

        if "byte_count" not in df and {"total_length_of_fwd_packets", "total_length_of_bwd_packets"}.issubset(df.columns):
            df["byte_count"] = self._numeric_series(df["total_length_of_fwd_packets"]) + self._numeric_series(df["total_length_of_bwd_packets"])
        elif "byte_count" not in df and "total_length_of_fwd_packets" in df:
            df["byte_count"] = self._numeric_series(df["total_length_of_fwd_packets"])

        duration = self._numeric_series(df["duration"]) if "duration" in df else None
        if "flow_bytes_per_sec" not in df and duration is not None and "byte_count" in df:
            df["flow_bytes_per_sec"] = self._safe_rate(self._numeric_series(df["byte_count"]), duration)
        if "flow_pkts_per_sec" not in df and duration is not None and "pkt_count" in df:
            df["flow_pkts_per_sec"] = self._safe_rate(self._numeric_series(df["pkt_count"]), duration)
        if "avg_pkt_size" not in df and {"byte_count", "pkt_count"}.issubset(df.columns):
            packets = self._numeric_series(df["pkt_count"]).replace(0, np.nan)
            df["avg_pkt_size"] = (self._numeric_series(df["byte_count"]) / packets).fillna(0)
        if "pkt_len_mean" not in df and "avg_pkt_size" in df:
            df["pkt_len_mean"] = df["avg_pkt_size"]
        if "subflow_fwd_bytes" not in df and "total_length_of_fwd_packets" in df:
            df["subflow_fwd_bytes"] = df["total_length_of_fwd_packets"]
        if "subflow_bwd_bytes" not in df and "total_length_of_bwd_packets" in df:
            df["subflow_bwd_bytes"] = df["total_length_of_bwd_packets"]
        if "avg_bwd_segment_size" not in df and "bwd_pkt_len_mean" in df:
            df["avg_bwd_segment_size"] = df["bwd_pkt_len_mean"]
        if "bwd_pkt_len_mean" not in df and {"total_length_of_bwd_packets", "bwd_pkt_count"}.issubset(df.columns):
            bwd_packets = self._numeric_series(df["bwd_pkt_count"]).replace(0, np.nan)
            df["bwd_pkt_len_mean"] = (self._numeric_series(df["total_length_of_bwd_packets"]) / bwd_packets).fillna(0)
        if "avg_bwd_segment_size" not in df and "bwd_pkt_len_mean" in df:
            df["avg_bwd_segment_size"] = df["bwd_pkt_len_mean"]
        return df

    def _looks_like_packet_df(self, df: pd.DataFrame) -> bool:
        has_packet_shape = {"src_ip", "dst_ip", "protocol_type"}.issubset(df.columns)
        has_packet_metric = "frame_len" in df.columns or "frame_time" in df.columns or "info" in df.columns
        has_flow_metrics = any(
            col in df.columns
            for col in ["pkt_count", "fwd_pkt_count", "bwd_pkt_count", "total_length_of_fwd_packets"]
        )
        return has_packet_shape and has_packet_metric and not has_flow_metrics

    def _aggregate_packet_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self._extract_ports_from_info(df)
        self._extract_flags_from_info(df)

        if "frame_len" not in df.columns:
            df["frame_len"] = 0
        if "frame_time" not in df.columns:
            df["frame_time"] = np.arange(len(df), dtype=float)

        df["src_port"] = self._numeric_series(df.get("src_port", pd.Series(0, index=df.index))).astype(int)
        df["dst_port"] = self._numeric_series(df.get("dst_port", pd.Series(0, index=df.index))).astype(int)
        df["frame_len"] = self._numeric_series(df["frame_len"])
        df["frame_time"] = self._numeric_series(df["frame_time"])
        df["protocol_type"] = df["protocol_type"].apply(self._protocol_number)

        group_keys = df.apply(self._flow_key, axis=1)
        rows = []
        for _, group in df.groupby(group_keys, sort=False):
            group = group.sort_values("frame_time")
            first = group.iloc[0]
            fwd_mask = (
                (group["src_ip"].astype(str) == str(first["src_ip"]))
                & (group["dst_ip"].astype(str) == str(first["dst_ip"]))
                & (group["src_port"] == int(first["src_port"]))
                & (group["dst_port"] == int(first["dst_port"]))
            )
            fwd = group[fwd_mask]
            bwd = group[~fwd_mask]
            lengths = self._numeric_series(group["frame_len"])
            fwd_lengths = self._numeric_series(fwd["frame_len"])
            bwd_lengths = self._numeric_series(bwd["frame_len"])
            duration = max(float(group["frame_time"].max() - group["frame_time"].min()), 0.0)

            row = {
                "duration": duration,
                "protocol_type": int(first["protocol_type"]),
                "src_ip": str(first["src_ip"]),
                "dst_ip": str(first["dst_ip"]),
                "src_port": int(first["src_port"]),
                "dst_port": int(first["dst_port"]),
                "fwd_pkt_count": int(len(fwd)),
                "bwd_pkt_count": int(len(bwd)),
                "total_length_of_fwd_packets": float(fwd_lengths.sum()),
                "total_length_of_bwd_packets": float(bwd_lengths.sum()),
                "pkt_len_mean": float(lengths.mean()) if len(lengths) else 0.0,
                "pkt_len_std": float(lengths.std(ddof=0)) if len(lengths) else 0.0,
                "pkt_len_var": float(lengths.var(ddof=0)) if len(lengths) else 0.0,
                "max_pkt_length": float(lengths.max()) if len(lengths) else 0.0,
                "bwd_pkt_len_mean": float(bwd_lengths.mean()) if len(bwd_lengths) else 0.0,
                "bwd_pkt_len_max": float(bwd_lengths.max()) if len(bwd_lengths) else 0.0,
                "init_win_bytes_forward": self._first_numeric(fwd, "window_size"),
                "init_win_bytes_backward": self._first_numeric(bwd, "window_size"),
                "label": str(first.get("label", "UNKNOWN")),
            }

            for flag in ["syn", "ack", "fin", "rst", "psh", "urg"]:
                col = f"{flag}_flag_count"
                row[col] = int(self._numeric_series(group.get(col, pd.Series(0, index=group.index))).sum())

            rows.append(row)

        return pd.DataFrame(rows)

    def _extract_ports_from_info(self, df: pd.DataFrame) -> None:
        if "info" not in df.columns:
            return
        info = df["info"].astype(str)
        ports = info.str.extract(r"(?P<src>\d{1,5})\s*(?:->|>|→)\s*(?P<dst>\d{1,5})")
        if "src_port" not in df.columns:
            df["src_port"] = pd.to_numeric(ports["src"], errors="coerce").fillna(0)
        if "dst_port" not in df.columns:
            df["dst_port"] = pd.to_numeric(ports["dst"], errors="coerce").fillna(0)

    def _extract_flags_from_info(self, df: pd.DataFrame) -> None:
        if "info" not in df.columns:
            return
        info = df["info"].astype(str).str.upper()
        flag_patterns = {
            "syn_flag_count": r"\bSYN\b|\[S\b",
            "ack_flag_count": r"\bACK\b|\[\.|\[.*\bA\b",
            "fin_flag_count": r"\bFIN\b|\[F\b|\[.*\bF\b",
            "rst_flag_count": r"\bRST\b|\[R\b|\[.*\bR\b",
            "psh_flag_count": r"\bPSH\b|\[P\b|\[.*\bP\b",
            "urg_flag_count": r"\bURG\b|\[U\b|\[.*\bU\b",
        }
        for col, pattern in flag_patterns.items():
            if col not in df.columns:
                df[col] = info.str.contains(pattern, regex=True).astype(int)

    def _flow_key(self, row: pd.Series):
        left = (str(row.get("src_ip", "")), self._to_int(row.get("src_port", 0)))
        right = (str(row.get("dst_ip", "")), self._to_int(row.get("dst_port", 0)))
        endpoints = tuple(sorted([left, right]))
        return (self._protocol_number(row.get("protocol_type", 0)), endpoints)

    def _first_numeric(self, df: pd.DataFrame, col: str) -> float:
        if col not in df.columns or df.empty:
            return 0.0
        values = self._numeric_series(df[col])
        return float(values.iloc[0]) if len(values) else 0.0

    def _derive_features_row(self, row: Dict) -> Dict:
        row = dict(row)
        if "pkt_count" not in row:
            row["pkt_count"] = self._to_float(row.get("fwd_pkt_count", 0)) + self._to_float(row.get("bwd_pkt_count", 0))
        if "byte_count" not in row:
            row["byte_count"] = (
                self._to_float(row.get("total_length_of_fwd_packets", 0))
                + self._to_float(row.get("total_length_of_bwd_packets", 0))
            )

        duration = self._to_float(row.get("duration", 0))
        duration_seconds = self._duration_seconds(duration)
        if "flow_bytes_per_sec" not in row and duration_seconds > 0:
            row["flow_bytes_per_sec"] = self._to_float(row.get("byte_count", 0)) / duration_seconds
        if "flow_pkts_per_sec" not in row and duration_seconds > 0:
            row["flow_pkts_per_sec"] = self._to_float(row.get("pkt_count", 0)) / duration_seconds
        if "avg_pkt_size" not in row and self._to_float(row.get("pkt_count", 0)) > 0:
            row["avg_pkt_size"] = self._to_float(row.get("byte_count", 0)) / self._to_float(row.get("pkt_count", 0))
        if "pkt_len_mean" not in row and "avg_pkt_size" in row:
            row["pkt_len_mean"] = row["avg_pkt_size"]
        if "subflow_fwd_bytes" not in row and "total_length_of_fwd_packets" in row:
            row["subflow_fwd_bytes"] = row["total_length_of_fwd_packets"]
        if "subflow_bwd_bytes" not in row and "total_length_of_bwd_packets" in row:
            row["subflow_bwd_bytes"] = row["total_length_of_bwd_packets"]
        if "avg_bwd_segment_size" not in row and "bwd_pkt_len_mean" in row:
            row["avg_bwd_segment_size"] = row["bwd_pkt_len_mean"]
        if "bwd_pkt_len_mean" not in row and self._to_float(row.get("bwd_pkt_count", 0)) > 0:
            row["bwd_pkt_len_mean"] = (
                self._to_float(row.get("total_length_of_bwd_packets", 0))
                / self._to_float(row.get("bwd_pkt_count", 0))
            )
        if "avg_bwd_segment_size" not in row and "bwd_pkt_len_mean" in row:
            row["avg_bwd_segment_size"] = row["bwd_pkt_len_mean"]
        return row

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df

    def _fill_missing(self, row: Dict) -> Dict:
        defaults = {
            "src_ip": "",
            "dst_ip": "",
            "label": "UNKNOWN",
            "info": "",
        }
        for feat in RAW_FEATURES:
            if feat not in row:
                row[feat] = defaults.get(feat, 0)
        return row

    def _normalize(self, feat: str, val: float) -> float:
        mn = self._min.get(feat, 0)
        mx = self._max.get(feat, 1)
        if mx == mn:
            return 0.0
        return float(np.clip((val - mn) / (mx - mn), 0, 1))

    def _normalize_name(self, value) -> str:
        value = str(value).strip().lower()
        value = value.replace("/s", " per sec").replace("%", " percent ")
        return re.sub(r"[^a-z0-9]+", "", value)

    def _numeric_series(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    def _safe_rate(self, numerator: pd.Series, duration: pd.Series) -> pd.Series:
        seconds = duration.apply(self._duration_seconds).replace(0, np.nan)
        return (numerator / seconds).replace([np.inf, -np.inf], np.nan).fillna(0)

    def _duration_seconds(self, duration: float) -> float:
        duration = self._to_float(duration)
        if duration <= 0:
            return 0.0
        return duration / 1_000_000 if duration > 10_000 else duration

    def _protocol_number(self, value) -> int:
        if isinstance(value, str):
            value = value.strip().upper()
        return int(PROTOCOL_MAP.get(value, self._to_int(value)))

    def _to_float(self, value) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _to_int(self, value) -> int:
        return int(self._to_float(value))
