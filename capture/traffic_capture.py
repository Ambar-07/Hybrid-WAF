"""
Traffic capture and controlled localhost traffic generation.

This module supports two modes:
1) Simulated localhost traffic (recommended for student projects)
2) Optional Scapy sniffing on localhost if Scapy/Npcap is available
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import random
import socket
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    from scapy.all import IP, IPv6, TCP, UDP, sniff  # type: ignore
except Exception:  # pragma: no cover - optional dependency runtime guard
    IP = IPv6 = TCP = UDP = sniff = None


PROTOCOL_TO_ID = {"TCP": 6, "UDP": 17, "ICMP": 1}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_packet_length(pkt) -> int:
    try:
        return int(len(pkt))
    except Exception:
        return 0


def _extract_packet_row(pkt) -> Optional[Dict[str, object]]:
    """Extract a minimal row from a Scapy packet."""
    has_ipv4 = IP is not None and IP in pkt
    has_ipv6 = IPv6 is not None and IPv6 in pkt
    
    if not (has_ipv4 or has_ipv6):
        return None

    src_port = 0
    dst_port = 0
    protocol = "OTHER"

    if TCP is not None and TCP in pkt:
        protocol = "TCP"
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)
    elif UDP is not None and UDP in pkt:
        protocol = "UDP"
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)
        
    src_ip = str(pkt[IP].src) if has_ipv4 else str(pkt[IPv6].src)
    dst_ip = str(pkt[IP].dst) if has_ipv4 else str(pkt[IPv6].dst)

    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "protocol": protocol,
        "protocol_type": PROTOCOL_TO_ID.get(protocol, 0),
        "src_port": src_port,
        "dst_port": dst_port,
        "packet_length": _safe_packet_length(pkt),
        "timestamp": _utc_now_iso(),
        "label": "BENIGN",
    }


def capture_localhost_packets(
    duration_seconds: int = 10,
    max_packets: int = 2000,
    out_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Capture localhost packets with Scapy and convert to a DataFrame.

    Note: On Windows, loopback capture requires Npcap support.
    """
    if sniff is None:
        raise RuntimeError("Scapy is not available. Install 'scapy' or use simulation mode.")

    captured_rows: List[Dict[str, object]] = []

    def _on_packet(pkt):
        row = _extract_packet_row(pkt)
        if row is not None:
            captured_rows.append(row)

    sniff(
        filter="host 127.0.0.1",
        prn=_on_packet,
        timeout=duration_seconds,
        count=max_packets,
        store=False,
    )

    df = pd.DataFrame(captured_rows)
    if out_csv:
        save_csv(df, out_csv)
    return df


def generate_controlled_localhost_traffic(
    num_rows: int = 1000,
    malicious_ratio: float = 0.15,
    out_csv: Optional[str] = "capture/generated_traffic.csv",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate structured localhost flow rows for safe, controlled testing.

    The generated labels are:
    - BENIGN
    - MALICIOUS (rule-like pattern)
    - SUSPICIOUS (anomaly-like pattern)
    """
    random.seed(seed)

    rows: List[Dict[str, object]] = []
    benign_ratio = max(0.0, 1.0 - malicious_ratio)

    for _ in range(num_rows):
        r = random.random()
        if r < benign_ratio:
            profile = "BENIGN"
        elif r < benign_ratio + (malicious_ratio * 0.65):
            profile = "MALICIOUS"
        else:
            profile = "SUSPICIOUS"

        if profile == "BENIGN":
            protocol = random.choice(["TCP", "UDP"])
            row = {
                "src_ip": "127.0.0.1",
                "dst_ip": "127.0.0.1",
                "protocol": protocol,
                "protocol_type": PROTOCOL_TO_ID[protocol],
                "src_port": random.randint(20000, 65000),
                "dst_port": random.choice([80, 443, 8080]),
                "packet_length": random.randint(80, 1200),
                "pkt_count": random.randint(5, 60),
                "duration": random.randint(100, 5000),
                "flow_pkts_per_sec": random.uniform(1.0, 25.0),
                "syn_flag_count": random.randint(0, 2),
                "ack_flag_count": random.randint(1, 25),
                "fin_flag_count": random.randint(0, 2),
                "rst_flag_count": random.randint(0, 1),
                "label": "BENIGN",
            }
        elif profile == "MALICIOUS":
            row = {
                "src_ip": "127.0.0.1",
                "dst_ip": "127.0.0.1",
                "protocol": "TCP",
                "protocol_type": 6,
                "src_port": random.randint(30000, 65000),
                "dst_port": random.choice([22, 3306, 8080, 8443]),
                "packet_length": random.randint(40, 250),
                "pkt_count": random.randint(90, 3000),
                "duration": random.randint(10, 700),
                "flow_pkts_per_sec": random.uniform(120.0, 2200.0),
                "syn_flag_count": random.randint(40, 500),
                "ack_flag_count": random.randint(0, 3),
                "fin_flag_count": random.randint(0, 1),
                "rst_flag_count": random.randint(0, 3),
                "label": "MALICIOUS",
            }
        else:
            protocol = random.choice(["TCP", "UDP"])
            row = {
                "src_ip": "127.0.0.1",
                "dst_ip": "127.0.0.1",
                "protocol": protocol,
                "protocol_type": PROTOCOL_TO_ID[protocol],
                "src_port": random.randint(20000, 65000),
                "dst_port": random.choice([53, 123, 5353, 9000]),
                "packet_length": random.randint(60, 1500),
                "pkt_count": random.randint(60, 500),
                "duration": random.randint(50, 8000),
                "flow_pkts_per_sec": random.uniform(20.0, 250.0),
                "syn_flag_count": random.randint(0, 10),
                "ack_flag_count": random.randint(0, 60),
                "fin_flag_count": random.randint(0, 6),
                "rst_flag_count": random.randint(0, 10),
                "label": "SUSPICIOUS",
            }

        row["timestamp"] = _utc_now_iso()
        rows.append(row)

    df = pd.DataFrame(rows)
    if out_csv:
        save_csv(df, out_csv)
    return df


def save_csv(df: pd.DataFrame, out_csv: str) -> str:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def logs_to_structured_csv(
    logs: Iterable[Dict[str, object]],
    out_csv: str = "capture/generated_traffic.csv",
) -> pd.DataFrame:
    """
    Convert generator logs into a structured CSV that the pipeline can use.
    """
    columns = [
        "timestamp",
        "src_ip",
        "dst_ip",
        "protocol",
        "protocol_type",
        "src_port",
        "dst_port",
        "packet_length",
        "request_payload",
        "payload_length",
        "pkt_count",
        "duration",
        "flow_pkts_per_sec",
        "byte_count",
        "flow_bytes_per_sec",
        "syn_flag_count",
        "ack_flag_count",
        "fin_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "urg_flag_count",
        "flags_count",
        "status",
        "message",
        "label",
    ]

    rows: List[Dict[str, object]] = []
    for row in logs:
        src_ip = str(row.get("src_ip", "127.0.0.1"))
        dst_ip = str(row.get("dst_ip", "127.0.0.1"))
        protocol = str(row.get("protocol", "TCP")).upper()
        src_port = int(row.get("src_port", 0) or 0)
        dst_port = int(row.get("dst_port", 0) or 0)
        packet_length = int(row.get("packet_length", 0) or 0)
        payload = str(row.get("request_payload", ""))

        pkt_count = int(row.get("pkt_count", 1) or 1)
        duration = float(row.get("duration", 1.0) or 1.0)
        duration = max(duration, 0.001)

        flags_count = int(row.get("flags_count", 1) or 1)
        syn_count = int(row.get("syn_flag_count", 0) or 0)
        ack_count = int(row.get("ack_flag_count", 0) or 0)
        fin_count = int(row.get("fin_flag_count", 0) or 0)
        rst_count = int(row.get("rst_flag_count", 0) or 0)

        mapped = {
            "timestamp": str(row.get("timestamp", _utc_now_iso())),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "protocol": protocol,
            "protocol_type": PROTOCOL_TO_ID.get(protocol, int(row.get("protocol_type", 0) or 0)),
            "src_port": src_port,
            "dst_port": dst_port,
            "packet_length": packet_length,
            "request_payload": payload,
            "payload_length": int(row.get("payload_length", len(payload))),
            "pkt_count": pkt_count,
            "duration": duration,
            "flow_pkts_per_sec": float(row.get("flow_pkts_per_sec", pkt_count / duration)),
            "byte_count": int(row.get("byte_count", packet_length * max(pkt_count, 1))),
            "flow_bytes_per_sec": float(row.get("flow_bytes_per_sec", (packet_length * max(pkt_count, 1)) / duration)),
            "syn_flag_count": syn_count,
            "ack_flag_count": ack_count,
            "fin_flag_count": fin_count,
            "rst_flag_count": rst_count,
            "psh_flag_count": int(row.get("psh_flag_count", 0) or 0),
            "urg_flag_count": int(row.get("urg_flag_count", 0) or 0),
            "flags_count": flags_count,
            "status": str(row.get("status", "ok")),
            "message": str(row.get("message", "generated")),
            "label": str(row.get("label", "BENIGN")),
        }
        rows.append(mapped)

    df = pd.DataFrame(rows, columns=columns)
    save_csv(df, out_csv)
    return df


def hit_localhost_endpoint(url: str, requests_count: int = 25, timeout: float = 2.0) -> None:
    """
    Optional helper to generate harmless localhost HTTP traffic.
    Requires the 'requests' package.
    """
    try:
        import requests  # local import to keep dependency optional
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("'requests' is required for hit_localhost_endpoint") from exc

    for _ in range(requests_count):
        try:
            requests.get(url, timeout=timeout)
        except Exception:
            # Ignore failed requests in test mode.
            pass
        time.sleep(0.02)


if __name__ == "__main__":
    generated = generate_controlled_localhost_traffic(num_rows=500)
    print(f"Generated {len(generated)} rows at capture/generated_traffic.csv")
