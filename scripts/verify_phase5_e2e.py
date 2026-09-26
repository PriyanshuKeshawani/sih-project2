import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import json
import numpy as np
from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.reflex import System1ReflexEngine
from engine.laya_adapter import System1InputState

def run_e2e_verification():
    print("=" * 70)
    print("PHASE 5: END-TO-END PIPELINE VERIFICATION ACROSS 5 TARGET SCENARIOS")
    print("=" * 70)

    # 1. Initialize Engines
    detector = SonarDetector(model_path="models/best_detector.onnx")
    physics_engine = SonarPhysicsEngine()
    reflex_engine = System1ReflexEngine(enable_laya=True)
    status = reflex_engine.system1_manager.get_status()
    print(f"System 1 Engine: {status.get('active_engine')} | Status: {status.get('status')}")
    print(f"Model Checkpoint: {status.get('checkpoint')}")
    print("-" * 70)

    # 2. Test Targets
    scenarios = [
        {"name": "ghost_net", "path": "sample_sonar_data/positives/test/images/synth_ghost_net_00001.png"},
        {"name": "shipwreck", "path": "sample_sonar_data/positives/test/images/wreckA_Artificial_Reef_06_y1280_x0.jpg"},
        {"name": "pipeline", "path": "sample_sonar_data/positives/test/images/pipe_1693569383.780_x3500.jpg"},
        {"name": "mine_cylinder", "path": "sample_sonar_data/positives/test/images/mine_0001_2015.jpg"},
        {"name": "background", "path": "sample_sonar_data/test/images/bg_1693569243.750_x2500.jpg"}
    ]

    # Create synthetic clean background if it doesn't exist
    bg_path = "samples/clean_seabed_bg.png"
    if not os.path.exists(bg_path):
        bg = np.full((640, 640, 3), 40, dtype=np.uint8)
        cv2.imwrite(bg_path, bg)

    results = []

    for sc in scenarios:
        name = sc["name"]
        path = sc["path"]
        print(f"\n[SCENARIO: {name.upper()}] Image: {path}")

        if not os.path.exists(path):
            print(f"  Warning: Image {path} not found on disk, skipping.")
            continue

        img_bgr = cv2.imread(path)
        h, w = img_bgr.shape[:2]

        # Stage 1: Detector
        detections, _, debug_info = detector.detect(img_bgr, conf_threshold=0.45, return_debug=True)
        print(f"  Stage 1 (Perception): {len(detections)} contacts detected (NMS final)")

        if not detections:
            print("  Stage 2-4: Nominal clear seafloor (0 contacts)")
            print("  System 1 Decision: PASSIVE_LOG (NOMINAL_CRUISE) | Hazard Score: 1.0")
            results.append({
                "scenario": name,
                "contacts": 0,
                "primary_class": "clean_seabed",
                "decision": "PASSIVE_LOG",
                "hazard_score": 1.0,
                "operator_review": False
            })
            continue

        # Stage 2: Physics
        primary_det = detections[0]
        elev = physics_engine.calculate_elevation(primary_det['box'], w, h, altitude=12.0)
        geo = physics_engine.georeference(primary_det['box'], w, h)
        shadow_det = bool(elev is not None and elev > 0.0)

        # Stage 3: System 1 Laya Decision
        ref = reflex_engine.process_reflex(
            detection=primary_det,
            physics={"elevation_m": elev, "shadow_detected": shadow_det, "provenance": "DERIVED"},
            geo=geo,
            metadata={"sonar_altitude_m": 12.0, "heading_deg": 45.0, "survey_id": f"SURVEY_{name.upper()}"}
        )

        s1 = ref["system1"]
        print(f"  Stage 2 (Physics): elevation={elev}m, shadow={shadow_det}, depth={geo.get('depth_m')}m")
        print(f"  Stage 3 (System 1 Real Laya):")
        print(f"    - Decision Primitive:    {s1['decision_primitive']}")
        print(f"    - Hazard Severity Score: {s1['hazard_score']} / 10.0")
        print(f"    - Needs Operator Review: {s1['needs_operator_review']}")
        print(f"    - Laya Confidence:       {s1['laya_confidence']}")
        print(f"    - Detector Confidence:   {s1['detector_confidence']}")
        print(f"    - Guardrail Applied:     {s1['guardrail_applied']}")
        print(f"    - Advisory Maneuver:     {ref['recommended_maneuver']}")

        results.append({
            "scenario": name,
            "contacts": len(detections),
            "primary_class": primary_det['class'],
            "decision": s1['decision_primitive'],
            "hazard_score": s1['hazard_score'],
            "operator_review": s1['needs_operator_review'],
            "guardrail_applied": s1['guardrail_applied'],
            "laya_conf": s1['laya_confidence']
        })

    print("\n" + "=" * 70)
    print("PHASE 5 E2E SCENARIO MATRIX SUMMARY:")
    print("=" * 70)
    print(f"{'SCENARIO':15s} | {'CONTACTS':8s} | {'PRIMARY CLASS':18s} | {'DECISION':22s} | {'HAZARD':6s} | {'REVIEW':6s}")
    print("-" * 75)
    for r in results:
        print(f"{r['scenario']:15s} | {r['contacts']:8d} | {r['primary_class']:18s} | {r['decision']:22s} | {r['hazard_score']:6.1f} | {str(r['operator_review']):6s}")
    print("=" * 70)

if __name__ == "__main__":
    run_e2e_verification()
