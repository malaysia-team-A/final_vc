import subprocess
import requests
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FILE = PROJECT_ROOT / "data" / "reports" / "stress_test_report_latest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "reports"
API_BASE = os.getenv("QA_API_BASE", "http://localhost:8000").rstrip("/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, filename):
    try:
        result = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode('utf-8', errors='ignore')
        with open(filename, "w", encoding="utf-8") as f:
            f.write(result)
    except Exception as e:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Error: {e}")

# 1. Check Python Processes
run_cmd("tasklist | findstr python", str(OUTPUT_DIR / "diag_process.txt"))

# 2. Check File Existence
if REPORT_FILE.exists():
    try:
        with open(REPORT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(OUTPUT_DIR / "diag_file.txt", "w", encoding="utf-8") as f:
            f.write(f"Report file exists. Lines: {len(lines)}\nLast line: {lines[-1] if lines else 'Empty'}")
    except Exception as e:
        with open(OUTPUT_DIR / "diag_file.txt", "w", encoding="utf-8") as f:
            f.write(f"Error reading report: {e}")
else:
    with open(OUTPUT_DIR / "diag_file.txt", "w", encoding="utf-8") as f:
        f.write("Report file does NOT exist.")

# 3. Check Server Response
try:
    resp = requests.get(f"{API_BASE}/", timeout=3)
    with open(OUTPUT_DIR / "diag_net.txt", "w", encoding="utf-8") as f:
        f.write(f"Server Status: {resp.status_code}\nContent: {resp.text[:100]}")
except Exception as e:
    with open(OUTPUT_DIR / "diag_net.txt", "w", encoding="utf-8") as f:
        f.write(f"Server Connection Failed: {e}")
