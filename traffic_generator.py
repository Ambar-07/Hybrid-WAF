"""
Safe localhost traffic generator for ThreatShield demos.

This module generates controlled, non-destructive traffic patterns only for
127.0.0.1 so students can test detection pipelines safely.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import random
import socket
import time
from typing import Dict, Iterable, List
from urllib.parse import urlencode, urlparse

import requests

LOCALHOST_SET = {"127.0.0.1", "localhost"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_localhost_host(host: str) -> bool:
    return str(host).strip().lower() in LOCALHOST_SET


def _assert_localhost_url(target_url: str) -> None:
    parsed = urlparse(target_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP/HTTPS URLs are supported.")
    if not _is_localhost_host(parsed.hostname or ""):
        raise ValueError("Only localhost URLs are allowed (127.0.0.1 or localhost).")


def _assert_localhost_ip(target_ip: str) -> None:
    if not _is_localhost_host(target_ip):
        raise ValueError("Only localhost target IP is allowed (127.0.0.1 or localhost).")


def _payload_attack_type(payload: str) -> str:
    text = str(payload or "").lower()
    if any(sig in text for sig in ["' or 1=1", "union select", "admin'--"]):
        return "SQL Injection"
    if any(sig in text for sig in ["<script", "javascript:", "onerror="]):
        return "XSS"
    if any(sig in text for sig in ["../", "%2f", "/.env", "/wp-admin", "/admin"]):
        return "Recon/Scan"
    return "Suspicious"


def _http_log_row(
    target_url: str,
    payload: str,
    label: str,
    status: str,
    latency_ms: float,
    message: str,
    attack_type: str,
) -> Dict[str, object]:
    parsed = urlparse(target_url)
    src_port = random.randint(20000, 65000)
    dst_port = parsed.port or (443 if parsed.scheme == "https" else 80)

    return {
        "timestamp": _now_iso(),
        "src_ip": "127.0.0.1",
        "dst_ip": "127.0.0.1",
        "protocol": "TCP",
        "protocol_type": 6,
        "src_port": src_port,
        "dst_port": dst_port,
        "packet_length": max(64, len(payload) + 120),
        "request_payload": payload,
        "payload_length": len(payload),
        "latency_ms": round(latency_ms, 3),
        "status": status,
        "message": message,
        "label": label,
        "attack_type": attack_type,
    }


def generate_normal_http(target_url: str, n: int = 25, delay: float = 0.05) -> List[Dict[str, object]]:
    """Send normal GET requests to localhost and return labeled traffic logs."""
    _assert_localhost_url(target_url)

    logs: List[Dict[str, object]] = []
    for i in range(max(0, int(n))):
        start = time.perf_counter()
        status = "ok"
        message = "normal_request"
        try:
            response = requests.get(target_url, timeout=2.0)
            message = f"http_{response.status_code}"
        except Exception as exc:
            status = "error"
            message = f"request_failed:{type(exc).__name__}"

        latency_ms = (time.perf_counter() - start) * 1000.0
        logs.append(
            _http_log_row(
                target_url,
                payload="",
                label="BENIGN",
                status=status,
                latency_ms=latency_ms,
                message=message,
                attack_type="Normal",
            )
        )
        if delay > 0:
            time.sleep(delay)

    return logs


def generate_payload_http(target_url: str, payloads: Iterable[str], delay: float = 0.05) -> List[Dict[str, object]]:
    """Send localhost requests containing test payload strings (SQLi/XSS-like)."""
    _assert_localhost_url(target_url)

    logs: List[Dict[str, object]] = []
    for payload in payloads:
        payload = str(payload)
        start = time.perf_counter()
        status = "ok"
        message = "payload_request"
        try:
            params = urlencode({"q": payload})
            separator = "&" if "?" in target_url else "?"
            req_url = f"{target_url}{separator}{params}"
            response = requests.get(req_url, timeout=2.0)
            message = f"http_{response.status_code}"
        except Exception as exc:
            status = "error"
            message = f"request_failed:{type(exc).__name__}"

        latency_ms = (time.perf_counter() - start) * 1000.0
        logs.append(
            _http_log_row(
                target_url,
                payload=payload,
                label="MALICIOUS",
                status=status,
                latency_ms=latency_ms,
                message=message,
                attack_type=_payload_attack_type(payload),
            )
        )
        if delay > 0:
            time.sleep(delay)

    return logs


def generate_path_fuzz_http(target_url: str, paths: Iterable[str], delay: float = 0.04) -> List[Dict[str, object]]:
    """Send localhost requests with unusual paths to emulate URL probing safely."""
    _assert_localhost_url(target_url)

    logs: List[Dict[str, object]] = []
    base = target_url.rstrip("/")
    for path in paths:
        raw_path = str(path or "").strip()
        if not raw_path:
            continue
        if not raw_path.startswith("/"):
            raw_path = f"/{raw_path}"

        req_url = f"{base}{raw_path}"
        start = time.perf_counter()
        status = "ok"
        message = "path_fuzz"
        try:
            response = requests.get(req_url, timeout=2.0)
            message = f"http_{response.status_code}"
        except Exception as exc:
            status = "error"
            message = f"request_failed:{type(exc).__name__}"

        latency_ms = (time.perf_counter() - start) * 1000.0
        logs.append(
            _http_log_row(
                target_url,
                payload=raw_path,
                label="MALICIOUS",
                status=status,
                latency_ms=latency_ms,
                message=message,
                attack_type="Recon/Scan",
            )
        )
        if delay > 0:
            time.sleep(delay)

    return logs


def generate_login_burst_http(
    target_url: str,
    usernames: Iterable[str],
    passwords: Iterable[str],
    attempts: int = 30,
    delay: float = 0.03,
) -> List[Dict[str, object]]:
    """Create safe localhost login-like request bursts with credential permutations."""
    _assert_localhost_url(target_url)

    user_list = [u.strip() for u in usernames if str(u).strip()]
    pass_list = [p.strip() for p in passwords if str(p).strip()]
    if not user_list:
        user_list = ["admin", "user", "guest"]
    if not pass_list:
        pass_list = ["123456", "password", "admin123"]

    total = max(0, int(attempts))
    logs: List[Dict[str, object]] = []
    for _ in range(total):
        username = random.choice(user_list)
        password = random.choice(pass_list)
        payload = f"username={username}&password={password}"

        start = time.perf_counter()
        status = "ok"
        message = "login_burst"
        try:
            params = urlencode({"username": username, "password": password})
            separator = "&" if "?" in target_url else "?"
            req_url = f"{target_url}{separator}{params}"
            response = requests.get(req_url, timeout=2.0)
            message = f"http_{response.status_code}"
        except Exception as exc:
            status = "error"
            message = f"request_failed:{type(exc).__name__}"

        latency_ms = (time.perf_counter() - start) * 1000.0
        logs.append(
            _http_log_row(
                target_url,
                payload=payload,
                label="SUSPICIOUS",
                status=status,
                latency_ms=latency_ms,
                message=message,
                attack_type="Brute Force",
            )
        )
        if delay > 0:
            time.sleep(delay)

    return logs


def generate_port_probe(target_ip: str, start_port: int, end_port: int, delay: float = 0.03) -> List[Dict[str, object]]:
    """Attempt sequential localhost TCP connects to mimic safe probe-like behavior."""
    _assert_localhost_ip(target_ip)

    s_port = int(min(start_port, end_port))
    e_port = int(max(start_port, end_port))
    logs: List[Dict[str, object]] = []

    for dst_port in range(s_port, e_port + 1):
        src_port = random.randint(20000, 65000)
        start = time.perf_counter()
        status = "closed"
        message = "probe"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                result = sock.connect_ex(("127.0.0.1", int(dst_port)))
                status = "open" if result == 0 else "closed"
        except Exception as exc:
            status = "error"
            message = f"probe_failed:{type(exc).__name__}"

        latency_ms = (time.perf_counter() - start) * 1000.0
        logs.append(
            {
                "timestamp": _now_iso(),
                "src_ip": "127.0.0.1",
                "dst_ip": "127.0.0.1",
                "protocol": "TCP",
                "protocol_type": 6,
                "src_port": src_port,
                "dst_port": int(dst_port),
                "packet_length": 64,
                "request_payload": "",
                "payload_length": 0,
                "latency_ms": round(latency_ms, 3),
                "status": status,
                "message": message,
                "label": "SUSPICIOUS",
                "attack_type": "Recon/Scan",
            }
        )
        if delay > 0:
            time.sleep(delay)

    return logs


def generate_connection_burst(
    target_ip: str,
    port: int,
    count: int = 50,
    concurrency: int = 5,
    delay: float = 0.02,
) -> List[Dict[str, object]]:
    """Create short-lived, rate-limited localhost connection bursts."""
    _assert_localhost_ip(target_ip)

    total = max(0, int(count))
    pool_size = max(1, int(concurrency))

    def _connect_once(_: int) -> Dict[str, object]:
        src_port = random.randint(20000, 65000)
        start = time.perf_counter()
        status = "closed"
        message = "burst_connect"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.25)
                result = sock.connect_ex(("127.0.0.1", int(port)))
                status = "open" if result == 0 else "closed"
        except Exception as exc:
            status = "error"
            message = f"burst_failed:{type(exc).__name__}"

        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "timestamp": _now_iso(),
            "src_ip": "127.0.0.1",
            "dst_ip": "127.0.0.1",
            "protocol": "TCP",
            "protocol_type": 6,
            "src_port": src_port,
            "dst_port": int(port),
            "packet_length": 72,
            "request_payload": "",
            "payload_length": 0,
            "latency_ms": round(latency_ms, 3),
            "status": status,
            "message": message,
            "label": "SUSPICIOUS",
            "attack_type": "Burst/DoS",
        }

    logs: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [executor.submit(_connect_once, i) for i in range(total)]
        for future in as_completed(futures):
            logs.append(future.result())
            if delay > 0:
                time.sleep(delay)

    logs.sort(key=lambda row: str(row["timestamp"]))
    return logs


def generate_random_mixed_traffic(
    target_url: str,
    target_ip: str,
    total_events: int = 100,
    malicious_ratio: float = 0.35,
    delay: float = 0.02,
) -> List[Dict[str, object]]:
    """Generate randomized localhost traffic mixing normal and attack-like patterns."""
    _assert_localhost_url(target_url)
    _assert_localhost_ip(target_ip)

    total = max(0, int(total_events))
    if total == 0:
        return []

    ratio = float(max(0.0, min(1.0, malicious_ratio)))
    malicious_target = int(round(total * ratio))
    benign_target = total - malicious_target

    payload_bank = [
        "' OR 1=1",
        "<script>alert(1)</script>",
        "../../etc/passwd",
        "UNION SELECT username,password FROM users",
        "..%2f..%2fwindows%2fwin.ini",
    ]
    path_bank = [
        "/admin",
        "/debug",
        "/.env",
        "/wp-admin",
        "/api/internal",
        "/../../etc/passwd",
    ]

    logs: List[Dict[str, object]] = []

    if benign_target > 0:
        logs.extend(generate_normal_http(target_url=target_url, n=benign_target, delay=delay))

    remaining = malicious_target
    while remaining > 0:
        mode = random.choice(["payload", "path", "probe", "burst", "login"])
        batch = max(1, min(remaining, random.randint(2, 8)))

        if mode == "payload":
            sampled = random.sample(payload_bank, k=min(batch, len(payload_bank)))
            batch_logs = generate_payload_http(target_url=target_url, payloads=sampled, delay=delay)
        elif mode == "path":
            sampled = random.sample(path_bank, k=min(batch, len(path_bank)))
            batch_logs = generate_path_fuzz_http(target_url=target_url, paths=sampled, delay=delay)
        elif mode == "probe":
            start_port = random.choice([8000, 8080, 5000, 3000])
            end_port = start_port + max(1, batch - 1)
            batch_logs = generate_port_probe(target_ip=target_ip, start_port=start_port, end_port=end_port, delay=delay)
        elif mode == "login":
            batch_logs = generate_login_burst_http(
                target_url=target_url,
                usernames=["admin", "root", "test", "guest"],
                passwords=["123456", "password", "admin123", "qwerty"],
                attempts=batch,
                delay=delay,
            )
        else:
            batch_logs = generate_connection_burst(
                target_ip=target_ip,
                port=random.choice([8000, 8080, 5000, 3000]),
                count=batch,
                concurrency=min(6, batch),
                delay=delay,
            )

        logs.extend(batch_logs[:batch])
        remaining -= min(batch, len(batch_logs))

    logs.sort(key=lambda row: str(row["timestamp"]))
    return logs[:total]


def generate_weighted_mixed_traffic(
    target_url: str,
    target_ip: str,
    total_events: int = 120,
    normal_ratio: float = 0.50,
    suspicious_ratio: float = 0.25,
    malicious_ratio: float = 0.25,
    delay: float = 0.02,
) -> List[Dict[str, object]]:
    """Generate localhost traffic with explicit normal/suspicious/malicious ratio control."""
    _assert_localhost_url(target_url)
    _assert_localhost_ip(target_ip)

    total = max(0, int(total_events))
    if total == 0:
        return []

    nr = max(0.0, float(normal_ratio))
    sr = max(0.0, float(suspicious_ratio))
    mr = max(0.0, float(malicious_ratio))
    ratio_sum = nr + sr + mr
    if ratio_sum <= 0:
        nr, sr, mr = 1.0, 0.0, 0.0
    else:
        nr, sr, mr = nr / ratio_sum, sr / ratio_sum, mr / ratio_sum

    normal_count = int(round(total * nr))
    suspicious_count = int(round(total * sr))
    malicious_count = max(0, total - normal_count - suspicious_count)

    logs: List[Dict[str, object]] = []
    if normal_count > 0:
        logs.extend(generate_normal_http(target_url=target_url, n=normal_count, delay=delay))

    while suspicious_count > 0:
        mode = random.choice(["probe", "burst", "login"])
        batch = max(1, min(suspicious_count, random.randint(2, 6)))
        if mode == "probe":
            start_port = random.choice([8000, 8080, 5000])
            batch_logs = generate_port_probe(target_ip=target_ip, start_port=start_port, end_port=start_port + batch, delay=delay)
        elif mode == "login":
            batch_logs = generate_login_burst_http(
                target_url=target_url,
                usernames=["admin", "root", "guest", "test"],
                passwords=["123456", "password", "admin123", "qwerty"],
                attempts=batch,
                delay=delay,
            )
        else:
            batch_logs = generate_connection_burst(
                target_ip=target_ip,
                port=random.choice([8000, 8080, 5000]),
                count=batch,
                concurrency=min(6, batch),
                delay=delay,
            )
        logs.extend(batch_logs[:batch])
        suspicious_count -= min(batch, len(batch_logs))

    while malicious_count > 0:
        mode = random.choice(["payload", "path"])
        batch = max(1, min(malicious_count, random.randint(2, 6)))
        if mode == "payload":
            payload_bank = [
                "' OR 1=1",
                "UNION SELECT username,password FROM users",
                "<script>alert(1)</script>",
                "admin'--",
            ]
            sampled = random.sample(payload_bank, k=min(batch, len(payload_bank)))
            batch_logs = generate_payload_http(target_url=target_url, payloads=sampled, delay=delay)
        else:
            path_bank = ["/admin", "/.env", "/wp-admin", "/../../etc/passwd", "/debug"]
            sampled = random.sample(path_bank, k=min(batch, len(path_bank)))
            batch_logs = generate_path_fuzz_http(target_url=target_url, paths=sampled, delay=delay)
        logs.extend(batch_logs[:batch])
        malicious_count -= min(batch, len(batch_logs))

    logs.sort(key=lambda row: str(row.get("timestamp", "")))
    return logs[:total]
