import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import time
import numpy as np
from engine.detector import SonarDetector

import engine.physics as physics
from engine.reflex import System1ReflexEngine
from engine.laya_adapter import LayaDecisionEngine, System1InputState


def benchmark():
    print("=" * 60)
    print("PHASE 5 SYSTEM 1 BENCHMARK: REAL LAYA ON LOCAL CPU")
    print("=" * 60)

    # 1. Initialize Engines
    t_start = time.perf_counter()
    detector = SonarDetector(model_path="models/best_detector.onnx")
    laya_engine = LayaDecisionEngine()
    cold_start_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"Cold Start Time (Engine Load + Checkpoint): {cold_start_ms:.2f} ms")
    print(f"Laya Engine Status: {laya_engine.get_status()}")

    # 2. Benchmark Laya Decision Latency over 20 iterations
    states = [
        System1InputState("ghost_net", 0.92, "STANDARD_CLASS", True, 1.8, 1.05, "DERIVED", "DERIVED", "STRONG", 12.0, 45.0),
        System1InputState("mine_cylinder", 0.86, "STANDARD_CLASS", True, 1.2, 0.70, "DERIVED", "DERIVED", "STRONG", 12.0, 45.0),
        System1InputState("submarine_pipeline", 0.81, "STANDARD_CLASS", True, 0.9, 0.35, "DERIVED", "UNAVAILABLE", "MODERATE", 12.0, 45.0),
        System1InputState("shipwreck", 0.88, "CLOSED_SET_SURROGATE", True, 2.5, 1.5, "DERIVED", "DERIVED", "STRONG", 12.0, 45.0),
        System1InputState("crab_pot", 0.65, "STANDARD_CLASS", False, None, None, "UNAVAILABLE", "UNAVAILABLE", "MODERATE", 12.0, 45.0)
    ]

    laya_latencies = []
    print("\nWarming up and running 20 Laya System 1 inference iterations...")
    for i in range(20):
        st = states[i % len(states)]
        t0 = time.perf_counter()
        dec = laya_engine.decide(st)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        laya_latencies.append(dt_ms)
        print(f"  Iter {i+1:02d}: {st.contact_class:20s} -> {dec.decision_primitive:22s} in {dt_ms:.2f} ms (conf: {dec.laya_confidence})")

    avg_laya = np.mean(laya_latencies)
    med_laya = np.median(laya_latencies)
    p95_laya = np.percentile(laya_latencies, 95)
    min_laya = np.min(laya_latencies)
    max_laya = np.max(laya_latencies)

    print("\n--- LAYA INFERENCE SUMMARY (CPU) ---")
    print(f"Min:    {min_laya:.2f} ms")
    print(f"Median: {med_laya:.2f} ms")
    print(f"Mean:   {avg_laya:.2f} ms")
    print(f"P95:    {p95_laya:.2f} ms")
    print(f"Max:    {max_laya:.2f} ms")

    # 3. Benchmark End-to-End Pipeline on real sample image
    sample_img_path = "samples/synth_ghost_net_00001.png"
    if not os.path.exists(sample_img_path):
        sample_img_path = "samples/mine_0001_2015.jpg"

    if os.path.exists(sample_img_path):
        print(f"\n--- END-TO-END PIPELINE TIMING BREAKDOWN ({sample_img_path}) ---")
        img_bgr = cv2.imread(sample_img_path)
        h, w = img_bgr.shape[:2]

        # Stage 1: YOLO Detector (Tiled)
        t_det0 = time.perf_counter()
        detections, _, debug_info = detector.detect(img_bgr, conf_threshold=0.45, return_debug=True)
        det_ms = (time.perf_counter() - t_det0) * 1000.0

        # Stage 2: Acoustic Shadow Physics
        t_phys0 = time.perf_counter()
        physics_results = []
        for d in detections:
            elev = physics.calculate_elevation(d['box'], w, h, altitude=12.0)
            geo = physics.georeference(d['box'], w, h)
            physics_results.append((elev, geo))
        phys_ms = (time.perf_counter() - t_phys0) * 1000.0

        # Stage 3: System 1 Laya Decision
        t_s1_0 = time.perf_counter()
        reflex_engine = System1ReflexEngine()
        reflex_results = []
        for d, (elev, geo) in zip(detections, physics_results):
            ref = reflex_engine.process_reflex(d, physics={"elevation_m": elev, "shadow_detected": bool(elev and elev > 0)}, geo=geo, metadata={"sonar_altitude_m": 12.0, "heading_deg": 45.0})
            reflex_results.append(ref)
        s1_ms = (time.perf_counter() - t_s1_0) * 1000.0

        total_ms = det_ms + phys_ms + s1_ms

        print(f"  Detector (YOLO Tiled):  {det_ms:8.2f} ms  ({(det_ms / total_ms)*100:5.1f}%)")
        print(f"  Physics (Shadow & Geo): {phys_ms:8.2f} ms  ({(phys_ms / total_ms)*100:5.1f}%)")
        print(f"  System 1 (Real Laya):   {s1_ms:8.2f} ms  ({(s1_ms / total_ms)*100:5.1f}%)")
        print(f"  Total End-to-End:       {total_ms:8.2f} ms")
        print(f"  Detections processed:   {len(detections)}")
        if reflex_results:
            print(f"  Primary Decision:       {reflex_results[0]['decision_primitive']} (hazard_score: {reflex_results[0]['hazard_score']})")

if __name__ == "__main__":
    benchmark()
