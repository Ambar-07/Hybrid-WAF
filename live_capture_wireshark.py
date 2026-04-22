import subprocess
import argparse
import sys
import shutil
# (Removed RegexMLDetector import as it's not on this branch)

def start_wireshark_capture(interface=None):
    print("[WiresharkMonitor] Checking for tshark.exe...")
    tshark_path = shutil.which("tshark.exe")
    if not tshark_path:
        # Check default Windows install path
        default_path = r"C:\Program Files\Wireshark\tshark.exe"
        import os
        if os.path.exists(default_path):
            tshark_path = default_path
        else:
            print("[WiresharkMonitor] ❌ tshark.exe not found! Make sure Wireshark is installed and in PATH.")
            sys.exit(1)

    print(f"[WiresharkMonitor] Found tshark at: {tshark_path}")
    print("[WiresharkMonitor] (Note: RegexMLDetector is not available on this branch, using basic live monitoring)")

    # Build tshark command
    # -l: flush standard output after each packet
    # -T fields -e tcp.payload: we want the raw HEX of the payload
    # -Y http: only HTTP packets
    cmd = [
        tshark_path,
        "-l",
        "-T", "fields",
        "-e", "tcp.payload",
        "-Y", "http",
    ]
    if interface:
        cmd.extend(["-i", interface])

    print(f"[WiresharkMonitor] Starting live capture on interface: {interface if interface else 'default'}")
    print(f"[WiresharkMonitor] Running: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            
            line = line.strip()
            if not line:
                continue
                
            # tshark outputs the tcp.payload as hex like "47:45:54:20..." or "47455420..."
            try:
                # Remove colons if present and decode hex
                hex_str = line.replace(":", "")
                raw_payload = bytes.fromhex(hex_str).decode("utf-8", errors="ignore")
                
                # Check if it looks like an HTTP request or response
                if raw_payload.startswith("GET ") or raw_payload.startswith("POST ") or raw_payload.startswith("HTTP/"):
                    first_line = raw_payload.split("\r\n")[0]
                    
                    # Basic check for common attack strings for demonstration
                    is_suspicious = any(attack in raw_payload.lower() for attack in ["union select", "<script>", "etc/passwd", "cmd.exe", "eval("])
                    
                    if is_suspicious:
                        print(f"\n[🚨 ALERT] Suspicious traffic detected via Wireshark!")
                        print(f"  Summary    : {first_line}")
                        print(f"  Length     : {len(raw_payload)} bytes")
                    else:
                        print(f"[📡 LIVE] {first_line}")
            except Exception as e:
                pass

    except KeyboardInterrupt:
        print("\n[WiresharkMonitor] Stopping capture...")
        process.terminate()
    except Exception as e:
        print(f"\n[WiresharkMonitor] ❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Web Traffic Analyzer (Wireshark + Regex + ML)")
    parser.add_argument("-i", "--interface", help="Network interface to sniff on (e.g., eth0, wlan0, 1 for Windows adaptors)")
    args = parser.parse_args()
    
    start_wireshark_capture(args.interface)
