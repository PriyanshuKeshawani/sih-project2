"""
PHASE 4.6 — Model Class-Conflict and Unknown-Object Diagnosis Engine.
Evaluates closed-set classification behavior on known vs unknown marine objects.
SIH 2026 Problem Statement 26057.
"""
import os
import sys
import json
import cv2
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.detector import SonarDetector, CLASSES, CLASS_COLORS
from engine.preprocessing import PreprocessConfig, SonarPreprocessor
from engine.tiling import SonarTiler

def run_phase4_6_diagnosis():
    print("=" * 75)
    print("PHASE 4.6: MODEL CLASS-CONFLICT & UNKNOWN-OBJECT DIAGNOSIS")
    print("=" * 75)

    detector = SonarDetector("models/best_detector.onnx")
    panels_dir = os.path.join("reports", "phase4_6", "panels")
    os.makedirs(panels_dir, exist_ok=True)

    # 1. Panel metadata definition
    panels_meta = [
        (1, "shipwreck_large", "Shipwreck (Large)", "shipwreck", True),
        (2, "shipwreck_medium", "Shipwreck (Medium)", "shipwreck", True),
        (3, "shipwreck_broken", "Shipwreck (Broken)", "shipwreck", True),
        (4, "pipeline", "Pipe / Pipeline", "submarine_pipeline", True),
        (5, "cylinder_debris", "Cylinder (Debris)", "mine_cylinder", True),
        (6, "ghost_net", "Ghost Net (Fishing Net)", "ghost_net", True),
        (7, "manmade_debris", "Man-made Debris", "UNKNOWN / NOT_IN_MODEL_TAXONOMY", False),
        (8, "anchor_chain", "Anchor / Chain", "UNKNOWN / NOT_IN_MODEL_TAXONOMY", False),
        (9, "tire_rubber", "Tire / Rubber Debris", "UNKNOWN / NOT_IN_MODEL_TAXONOMY", False),
        (10, "natural_rock", "Natural Rock (Negative)", "UNKNOWN / BACKGROUND", False),
        (11, "sand_ripples", "Sand Ripples (Negative)", "UNKNOWN / BACKGROUND", False),
        (12, "mixed_scene", "Mixed Scene (Multiple)", "mixed", True)
    ]

    print("\n1. EVALUATING INDIVIDUAL ISOLATED PANELS (DIRECT ONNX RESIZE 640x640)")
    print("-" * 75)
    panel_results = []

    for p_num, name, label, expected_tax, in_taxonomy in panels_meta:
        fname = f"panel_{p_num:02d}_{name}.png"
        fp = os.path.join(panels_dir, fname)
        if not os.path.exists(fp):
            continue
        img = cv2.imread(fp)

        # Direct inference (tiling=False)
        dets_direct, _ = detector.detect(img, conf_threshold=0.25, tiling=False)
        
        # Also run ONNX directly to extract all 5 class scores for primary object
        img_640 = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)
        outputs = detector.session.run([detector.output_name], {detector.input_name: tensor})[0]
        preds = np.transpose(outputs[0])  # (8400, 9)
        class_probs = preds[:, 4:]  # (8400, 5)

        max_idx = np.argmax(np.max(class_probs, axis=1))
        top_score = float(np.max(class_probs[max_idx]))
        top_class_id = int(np.argmax(class_probs[max_idx]))
        top_class = CLASSES[top_class_id]

        scores_5 = {CLASSES[i]: round(float(class_probs[max_idx, i]), 5) for i in range(5)}

        pred_summary = f"{top_class} ({top_score:.3f})" if top_score >= 0.25 else "BACKGROUND (< 0.25)"

        panel_results.append({
            "panel_id": p_num,
            "panel_name": name,
            "label": label,
            "in_taxonomy": in_taxonomy,
            "expected_taxonomy": expected_tax,
            "primary_prediction": top_class if top_score >= 0.25 else "NONE",
            "confidence": round(top_score, 3),
            "all_5_scores": scores_5,
            "detections_count_025": len(dets_direct)
        })

        print(f"Panel {p_num:02d} | {label:<25} | In-Tax: {str(in_taxonomy):<5} | Primary Pred: {pred_summary}")
        print(f"     Class vector: {scores_5}")

    # 2. Unknown-object analysis & assignment rate
    print("\n2. UNKNOWN-OBJECT ANALYSIS & SHIPWRECK ASSIGNMENT RATE")
    print("-" * 75)
    unknown_panels = [p for p in panel_results if not p["in_taxonomy"]]
    unknown_count = len(unknown_panels)
    unknown_assigned_shipwreck = [p for p in unknown_panels if p["primary_prediction"] == "shipwreck"]
    unknown_shipwreck_rate = len(unknown_assigned_shipwreck) / unknown_count if unknown_count > 0 else 0.0

    print(f"Total evaluated non-taxonomy/unknown panels: {unknown_count}")
    print(f"Panels assigned to 'shipwreck': {len(unknown_assigned_shipwreck)} / {unknown_count}")
    print(f"UNKNOWN-OBJECT SHIPWRECK ASSIGNMENT RATE: {unknown_shipwreck_rate * 100:.1f}%")

    print("\nBreakdown of Unknown / Negative Objects:")
    for u in unknown_panels:
        print(f"  - {u['label']:<28} -> Predicted: {u['primary_prediction']:<15} (Conf: {u['confidence']:.3f})")

    # 3. High-Confidence False Positive Analysis
    print("\n3. HIGH-CONFIDENCE NON-TAXONOMY DETECTIONS TEST")
    print("-" * 75)
    high_conf_thresholds = [0.45, 0.50, 0.60, 0.70]
    for thresh in high_conf_thresholds:
        surviving = [p for p in unknown_panels if p["confidence"] >= thresh and p["primary_prediction"] == "shipwreck"]
        print(f"Unknown objects surviving as SHIPWRECK at conf >= {thresh:.2f}: {len(surviving)}")
        for s in surviving:
            print(f"    * {s['label']} (Confidence: {s['confidence']:.3f})")

    # 4. Compare Modes on Image B Mosaic: Mode A vs Mode B vs Mode C
    print("\n4. THREE-MODE COMPARISON ON COMPLETE IMAGE B MOSAIC")
    print("-" * 75)
    pB = os.path.join("reports", "phase4_6", "media_1790419786656.jpg")
    if not os.path.exists(pB):
        # copy or use user artifact
        src = r"C:\Users\santo\.gemini\antigravity-ide\brain\b7f7897e-dc1a-47cf-b616-cab1cf4f562a\.user_uploaded\media_1790419786656.jpg"
        if os.path.exists(src):
            import shutil
            shutil.copy(src, pB)

    imB = cv2.imread(pB)
    mode_results = {}

    for mode_key, mode_name, prep, tile in [
        ("mode_a", "Mode A: RAW + Tiling", False, True),
        ("mode_b", "Mode B: CLAHE + Bilateral + Tiling", True, True),
        ("mode_c_notile", "Mode C: Full Image Direct (No Tiling)", True, False)
    ]:
        dets, _ = detector.detect(imB, conf_threshold=0.45, tiling=tile, preprocess=prep)
        class_dist = Counter([d["class"] for d in dets])
        mode_results[mode_key] = {
            "name": mode_name,
            "total_detections": len(dets),
            "class_distribution": dict(class_dist)
        }
        print(f"{mode_name:<38} | Total: {len(dets):2d} | Distribution: {dict(class_dist)}")

    # 5. Class Conflict Matrix (Actual vs Predicted)
    print("\n5. CLASS-CONFLICT MATRIX (QUALITATIVE REFERENCE)")
    print("-" * 75)
    header = f"{'Actual / Reference Category':<30} | {'Shipwreck':<10} | {'Pipeline':<10} | {'GhostNet':<10} | {'Mine':<10} | {'CrabPot':<10} | {'None (<0.25)':<12}"
    print(header)
    print("-" * len(header))

    matrix_rows = []
    for p in panel_results:
        pred = p["primary_prediction"]
        row = {
            "category": p["label"],
            "shipwreck": 1 if pred == "shipwreck" else 0,
            "pipeline": 1 if pred == "submarine_pipeline" else 0,
            "ghost_net": 1 if pred == "ghost_net" else 0,
            "mine_cylinder": 1 if pred == "mine_cylinder" else 0,
            "crab_pot": 1 if pred == "crab_pot" else 0,
            "none": 1 if pred == "NONE" else 0
        }
        matrix_rows.append(row)
        print(f"{p['label']:<30} | {row['shipwreck']:<10} | {row['pipeline']:<10} | {row['ghost_net']:<10} | {row['mine_cylinder']:<10} | {row['crab_pot']:<10} | {row['none']:<12}")

    # 6. Save JSON audit payload
    out_json = os.path.join("reports", "phase4_6", "phase4_6_metrics.json")
    with open(out_json, "w") as f:
        json.dump({
            "panel_results": panel_results,
            "unknown_object_shipwreck_assignment_rate": unknown_shipwreck_rate,
            "mode_comparison": mode_results,
            "matrix_rows": matrix_rows
        }, f, indent=2)
    print(f"\nSaved diagnosis telemetry to {out_json}")

if __name__ == "__main__":
    run_phase4_6_diagnosis()
