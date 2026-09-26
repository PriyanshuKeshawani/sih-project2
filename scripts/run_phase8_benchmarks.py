"""
scripts/run_phase8_benchmarks.py
Executes Phase 8 Mission Replays across Scenarios A-G, computes component-level
statistical distributions (mean, median, p95, max), and writes summary reports.
"""

import os
import sys
import json
import time
import numpy as np

# Ensure root directory is on PYTHONPATH
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv()

from engine.mission_replay import MissionReplayEngine

def run_benchmarks(num_runs_per_scenario: int = 3):
    print("=" * 70)
    print("SAMUDRA-AI: PHASE 8 END-TO-END MISSION REPLAY & BENCHMARK SUITE")
    print("=" * 70)
    
    engine = MissionReplayEngine()
    scenarios = [
        "Scenario A",
        "Scenario B",
        "Scenario C",
        "Scenario D",
        "Scenario E",
        "Scenario F",
        "Scenario G"
    ]
    
    scenario_results = {}
    metric_samples = {
        "image_loading_ms": [],
        "tiled_inference_ms": [],
        "yolo_cpu_inference_ms": [],
        "physics_geo_ms": [],
        "laya_system1_ms": [],
        "temporal_tracking_ms": [],
        "system2_network_ms": [],
        "system2_fallback_ms": [],
        "pdf_generation_ms": [],
        "total_mission_ms": []
    }
    
    print("\n--- 1. Executing Full Scenario Replays (A through G) ---")
    for scen in scenarios:
        print(f"\n[RUNNING] {scen}...")
        res = engine.run_scenario(scen, conf_threshold=0.45)
        scenario_results[scen] = {
            "status": res["status"],
            "dataset_name": res["dataset_name"],
            "license": res["license"],
            "image_filename": res.get("image_filename"),
            "detections_count": len(res.get("detections", [])),
            "primary_class": res.get("detections", [{}])[0].get("class") if res.get("detections") else "clean_seabed",
            "system1_primitive": res.get("system1", {}).get("decision_primitive"),
            "system1_hazard": res.get("system1", {}).get("hazard_score"),
            "system2_type": res.get("timings", {}).get("system2_type"),
            "system2_priority": res.get("system2", {}).get("recovery_priority"),
            "total_ms": res.get("timings", {}).get("total_mission_ms")
        }
        print(f"  Status: {res['status']} | Dataset: {res['dataset_name']}")
        print(f"  Detections: {scenario_results[scen]['detections_count']} | S1: {scenario_results[scen]['system1_primitive']} | Total: {scenario_results[scen]['total_ms']}ms")
    
    print("\n--- 2. Executing Multi-Ping Temporal Transition Replay ---")
    multi_res = engine.run_multi_ping_replay("Scenario B", num_pings=3)
    print(f"  Multi-ping survey: {multi_res['survey_id']}")
    for ping in multi_res.get("ping_history", []):
        obs = ping["observations"]
        st = obs[0].get("persistence_status") if obs else "NO_TARGET"
        oc = obs[0].get("observation_count") if obs else 0
        print(f"  Ping #{ping['ping_idx']} -> Status: {st}, Count: {oc}")
    
    print(f"\n--- 3. Collecting Latency Distribution Over {num_runs_per_scenario} Iterations ---")
    for r in range(num_runs_per_scenario):
        for scen in ["Scenario A", "Scenario B", "Scenario C", "Scenario D"]:
            res = engine.run_scenario(scen, generate_pdf=True)
            t = res.get("timings", {})
            for k in ["image_loading_ms", "tiled_inference_ms", "yolo_cpu_inference_ms", "physics_geo_ms", "laya_system1_ms", "temporal_tracking_ms", "pdf_generation_ms", "total_mission_ms"]:
                if k in t:
                    metric_samples[k].append(t[k])
            
            s2_type = t.get("system2_type", "fallback")
            if s2_type == "network_groq":
                metric_samples["system2_network_ms"].append(t.get("system2_total_ms", 0.0))
            else:
                metric_samples["system2_fallback_ms"].append(t.get("system2_total_ms", 0.0))
    
    # Compute Statistics
    stats_summary = {}
    for m, vals in metric_samples.items():
        if vals:
            stats_summary[m] = {
                "count": len(vals),
                "mean": round(float(np.mean(vals)), 2),
                "median": round(float(np.median(vals)), 2),
                "p95": round(float(np.percentile(vals, 95)), 2),
                "max": round(float(np.max(vals)), 2),
                "min": round(float(np.min(vals)), 2)
            }
        else:
            stats_summary[m] = {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (LATENCY IN MILLISECONDS)")
    print("=" * 70)
    print(f"{'Component Metric':<28} | {'Mean':<8} | {'Median':<8} | {'P95':<8} | {'Max':<8}")
    print("-" * 70)
    for m, st in stats_summary.items():
        print(f"{m:<28} | {st['mean']:<8.2f} | {st['median']:<8.2f} | {st['p95']:<8.2f} | {st['max']:<8.2f}")
    
    out_path = os.path.join(BASE_DIR, "reports", "phase8", "benchmarks_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "scenarios": scenario_results,
            "multi_ping_validation": multi_res,
            "latency_benchmarks": stats_summary
        }, f, indent=2)
    print(f"\nSaved full benchmark summary to {out_path}")

if __name__ == "__main__":
    run_benchmarks(num_runs_per_scenario=2)
