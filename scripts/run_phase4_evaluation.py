import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.metadata import SurveyMetadata, ProvenanceStatus
from engine.reflex import System1ReflexEngine, DecisionPrimitive, AlertSeverity


def generate_phase4_visualizations_and_benchmark():
    os.makedirs("reports/phase4", exist_ok=True)
    model_path = os.path.join("models", "best_detector.onnx")
    detector = SonarDetector(model_path)
    physics_engine = SonarPhysicsEngine()
    reflex_engine = System1ReflexEngine()

    sample_images = [
        ("ghost_net", os.path.join("data", "samples", "synth_ghost_net_00001.png"), 0.05, 12.0),
        ("mine_cylinder", os.path.join("data", "samples", "mine_0001_2015.jpg"), 0.04, 10.0),
        ("shipwreck", os.path.join("data", "samples", "wreckR_ship-081_png.rf.6cf386b75ddb8ead86c0453021279296.jpg"), None, None)
    ]

    # Benchmarking fine-grained latency components over 500 iterations
    lat_input_parsing = []
    lat_risk_calc = []
    lat_decision_gen = []
    lat_event_creation = []
    lat_laya_check = []
    lat_total_reflex = []

    test_det = {"class": "ghost_net", "confidence": 0.92, "bbox_xyxy": [114, 210, 229, 319]}
    test_phys = {"shadow_detected": True, "shadow_length_m": 1.90, "elevation_m": 1.04}
    test_geo = {"status": "DERIVED"}
    test_meta = {"heading_deg": 90.0, "sonar_altitude_m": 12.0, "survey_id": "BENCH_SURVEY"}

    for _ in range(500):
        # 1. Input parsing
        t0 = time.perf_counter()
        cls_name = test_det["class"]
        conf = test_det["confidence"]
        bbox = test_det["bbox_xyxy"]
        elev = test_phys.get("elevation_m")
        t_parse = (time.perf_counter() - t0) * 1000.0
        lat_input_parsing.append(t_parse)

        # 2. Evidence & Risk Calculation
        t1 = time.perf_counter()
        eq = reflex_engine.evaluate_evidence_quality(conf, test_phys, test_geo)
        risk = reflex_engine.compute_deterministic_risk(cls_name, conf, elev, eq)
        t_risk = (time.perf_counter() - t1) * 1000.0
        lat_risk_calc.append(t_risk)

        # 3. Decision Generation
        t2 = time.perf_counter()
        prim, sev, reasons = reflex_engine.decide_primitive(cls_name, conf, risk, eq, elev)
        nav = reflex_engine.generate_navigation_recommendation(prim, cls_name, test_meta)
        t_dec = (time.perf_counter() - t2) * 1000.0
        lat_decision_gen.append(t_dec)

        # 4. Optional Laya Adapter Overhead
        t3 = time.perf_counter()
        _ = reflex_engine.laya_adapter.predict({"context": "benchmark"})
        t_laya = (time.perf_counter() - t3) * 1000.0
        lat_laya_check.append(t_laya)

        # 5. Full End-to-End Reflex Call (including event creation)
        t_tot0 = time.perf_counter()
        out = reflex_engine.process_reflex(test_det, physics=test_phys, geo=test_geo, metadata=test_meta)
        t_tot = (time.perf_counter() - t_tot0) * 1000.0
        lat_total_reflex.append(t_tot)
        lat_event_creation.append(t_tot - (t_parse + t_risk + t_dec + t_laya))

    # Generate Visual Debug Images
    color_map = {
        DecisionPrimitive.EMERGENCY_PROP_HAZARD.value: (0, 0, 255),      # Red alert
        DecisionPrimitive.LOITER_AND_RESCAN.value: (0, 165, 255),        # Orange warning
        DecisionPrimitive.PASSIVE_LOG.value: (0, 255, 0)                 # Green nominal
    }

    generated_files = []

    for name, path, m_per_px, alt_m in sample_images:
        if not os.path.exists(path):
            continue

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]
        meta = SurveyMetadata(
            survey_id=f"SURVEY_{name.upper()}",
            platform_lat=13.082 if m_per_px else None,
            platform_lon=80.270 if m_per_px else None,
            sonar_altitude_m=alt_m,
            meters_per_pixel=m_per_px,
            heading_deg=45.0 if m_per_px else None,
            source_type="RECORDED_SURVEY" if m_per_px else "UNKNOWN"
        )

        detections, _ = detector.detect(img_bgr, conf_threshold=0.25, tiling=True)
        scan_res = physics_engine.process_scan(img_bgr, detections, metadata=meta, enable_reflex=True)

        vis_img = img_bgr.copy()

        for d in scan_res["detections"]:
            gx1, gy1, gx2, gy2 = d["bbox_xyxy"]
            cls_name = d["class"]
            conf = d["confidence"]
            ref = d.get("reflex", {})
            decision = ref.get("decision_primitive", "UNKNOWN")
            risk = ref.get("hazard_score", 1.0)
            eq = ref.get("evidence_quality", "UNKNOWN")
            box_color = color_map.get(decision, (255, 255, 255))

            # Draw clean target box
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), box_color, 2)

            # Draw decision banner
            banner_text = f"[{decision}] RISK: {risk:.1f}/10"
            (bw, bh), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis_img, (gx1, max(0, gy1 - 22)), (gx1 + bw + 6, max(22, gy1)), box_color, -1)
            cv2.putText(vis_img, banner_text, (gx1 + 3, max(16, gy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw secondary detail line
            detail_text = f"{cls_name} ({conf*100:.0f}%) | EV: {eq}"
            if d["physics"]["elevation_m"] is not None:
                detail_text += f" | Elev: {d['physics']['elevation_m']:.2f}m"
            cv2.putText(vis_img, detail_text, (gx1, min(h - 8, gy2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join("reports", "phase4", f"phase4_reflex_{name}.png")
        cv2.imwrite(out_path, vis_img)
        generated_files.append((name, out_path, len(scan_res["detections"])))

    # Compute Statistics
    stats = {
        "mean": np.mean(lat_total_reflex),
        "median": np.median(lat_total_reflex),
        "p95": np.percentile(lat_total_reflex, 95),
        "max": np.max(lat_total_reflex)
    }

    print("============================================================")
    print("PHASE 4 SYSTEM 1 REFLEX BENCHMARK & EVALUATION RESULTS")
    print("============================================================")
    print(f"Sample size:                  500 iterations")
    print(f"Mean Latency:                 {stats['mean']:.4f} ms")
    print(f"Median Latency:               {stats['median']:.4f} ms")
    print(f"P95 Latency:                  {stats['p95']:.4f} ms")
    print(f"Max Latency:                  {stats['max']:.4f} ms")
    print("------------------------------------------------------------")
    print(f"Input Parsing (mean):         {np.mean(lat_input_parsing):.4f} ms")
    print(f"Risk Calculation (mean):      {np.mean(lat_risk_calc):.4f} ms")
    print(f"Decision Generation (mean):   {np.mean(lat_decision_gen):.4f} ms")
    print(f"Laya Check Overhead (mean):   {np.mean(lat_laya_check):.4f} ms")
    print(f"Event Creation (mean):        {np.mean(lat_event_creation):.4f} ms")
    print("------------------------------------------------------------")
    print("Generated Phase 4 Debug Images:")
    for n, p, cnt in generated_files:
        print(f" - [{n}] ({cnt} contacts): {p}")
    print("============================================================")


if __name__ == "__main__":
    generate_phase4_visualizations_and_benchmark()
