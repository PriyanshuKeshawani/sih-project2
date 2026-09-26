"""
Phase 7: Visual QA Screenshot Capture and UI Performance Benchmark.
SIH 2026 Problem Statement 26057.
"""
import os
import time
import subprocess
import urllib.request
import urllib.parse
import json

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "phase7")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://localhost:8000/"

def capture_screenshot(window_size, output_filename, delay_s=2):
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--disable-gpu",
        f"--window-size={window_size}",
        f"--virtual-time-budget={int(delay_s * 1000)}",
        f"--screenshot={out_path}",
        BASE_URL
    ]
    print(f"Capturing {output_filename} ({window_size})...")
    subprocess.run(cmd, check=True)
    if os.path.exists(out_path):
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  [OK] Saved {output_filename} ({size_kb:.1f} KB)")
    else:
        print(f"  [FAIL] Failed to save {output_filename}")

def benchmark_ui_performance():
    print("\n--- UI & API Rendering Benchmarks ---")
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)

    latencies = {
        "page_load_ms": [],
        "scan_rendering_ms": [],
        "track_update_ms": []
    }

    # 1. Measure initial page load (HTML + CSS + JS)
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.get("/")
        latencies["page_load_ms"].append((time.perf_counter() - t0) * 1000)

    # 2. Measure /api/scan endpoint latency
    sample_path = os.path.join(BASE_DIR, "data", "samples", "synth_ghost_net_00001.png")
    if os.path.exists(sample_path):
        for _ in range(10):
            t0 = time.perf_counter()
            with open(sample_path, "rb") as f:
                client.post("/api/scan", data={"conf_threshold": "0.45", "altitude": "12.0"}, files={"file": ("sample.png", f, "image/png")})
            latencies["scan_rendering_ms"].append((time.perf_counter() - t0) * 1000)

    # 3. Measure track update latency
    for _ in range(10):
        t0 = time.perf_counter()
        req_data = {
            "scan_id": "BENCH_SCAN",
            "observations": [{"class": "ghost_net", "confidence": 0.92, "box": {"x": 50, "y": 50, "w": 30, "h": 30}}]
        }
        client.post("/api/surveys/BENCH_SURVEY/observations", json=req_data)
        latencies["track_update_ms"].append((time.perf_counter() - t0) * 1000)

    stats = {}
    for k, vals in latencies.items():
        if vals:
            stats[k] = {
                "min": round(min(vals), 2),
                "median": round(sorted(vals)[len(vals)//2], 2),
                "mean": round(sum(vals)/len(vals), 2),
                "p95": round(sorted(vals)[int(len(vals)*0.95)], 2)
            }
            print(f"  {k}: median={stats[k]['median']} ms, mean={stats[k]['mean']} ms, min={stats[k]['min']} ms")

    summary_file = os.path.join(OUTPUT_DIR, "ui_performance_summary.json")
    with open(summary_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved performance summary to {summary_file}")

if __name__ == "__main__":
    capture_screenshot("1600,1050", "operator_cockpit_desktop.png", delay_s=3)
    capture_screenshot("820,1180", "operator_cockpit_tablet.png", delay_s=3)
    capture_screenshot("390,844", "operator_cockpit_mobile.png", delay_s=3)
    benchmark_ui_performance()
