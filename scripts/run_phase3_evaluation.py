import os
import sys
import time
import cv2
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.metadata import SurveyMetadata, ProvenanceStatus
from engine.shadow_analysis import AcousticShadowAnalyzer, ShadowConfig


def generate_visual_artifacts_and_benchmark():
    os.makedirs("reports/phase3", exist_ok=True)
    model_path = os.path.join("models", "best_detector.onnx")
    detector = SonarDetector(model_path)
    physics_engine = SonarPhysicsEngine()

    sample_paths = [
        ("ghost_net", os.path.join("data", "samples", "synth_ghost_net_00001.png"), 0.05),
        ("mine_cylinder", os.path.join("data", "samples", "mine_0001_2015.jpg"), 0.04),
        ("shipwreck", os.path.join("data", "samples", "wreckR_ship-081_png.rf.6cf386b75ddb8ead86c0453021279296.jpg"), None)
    ]

    # Also generate a controlled synthetic benchmark image with guaranteed acoustic shadow
    syn_h, syn_w = 400, 600
    syn_img = np.full((syn_h, syn_w, 3), 110, dtype=np.uint8)
    # Bright target
    syn_img[180:230, 320:370] = 235
    # Dark acoustic shadow
    syn_img[180:230, 370:435] = 12
    syn_path = os.path.join("reports", "phase3", "synthetic_shadow_source.png")
    cv2.imwrite(syn_path, syn_img)
    sample_paths.append(("synthetic_target", syn_path, 0.05))

    timings = {
        "detection_ms": [],
        "shadow_analysis_ms": [],
        "physics_ms": [],
        "geo_ms": [],
        "total_phase3_overhead_ms": []
    }

    processed_samples = []

    for name, path, m_per_px in sample_paths:
        if not os.path.exists(path):
            continue

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        h, w = img_bgr.shape[:2]

        # 1. Detection Timing
        t0 = time.perf_counter()
        if name == "synthetic_target":
            # Direct synthetic detection box
            detections = [{
                "class_id": 4,
                "class": "mine_cylinder",
                "confidence": 0.95,
                "box": {"x": 320, "y": 180, "w": 50, "h": 50},
                "bbox_xyxy": [320, 180, 370, 230],
                "tile_id": "syn_tile",
                "source_width": w,
                "source_height": h
            }]
            t_det = 0.0
        else:
            detections, _ = detector.detect(img_bgr, conf_threshold=0.25, tiling=True)
            t_det = (time.perf_counter() - t0) * 1000.0
            timings["detection_ms"].append(t_det)

        # Survey metadata
        meta = SurveyMetadata(
            survey_id=f"SURVEY_{name.upper()}",
            platform_lat=12.981 if m_per_px else None,
            platform_lon=80.252 if m_per_px else None,
            sonar_altitude_m=12.0 if m_per_px else None,
            slant_range_m=20.0 if m_per_px else None,
            vehicle_depth_m=25.0 if m_per_px else None,
            meters_per_pixel=m_per_px,
            source_type="RECORDED_SURVEY" if m_per_px else "UNKNOWN"
        )

        vis_img = img_bgr.copy()

        for d in detections:
            bbox = d["bbox_xyxy"]

            # 2. Shadow Analysis Timing
            t_sh0 = time.perf_counter()
            shadow_res = physics_engine.shadow_analyzer.analyze(img_bgr, bbox, meters_per_pixel=m_per_px)
            t_sh = (time.perf_counter() - t_sh0) * 1000.0
            timings["shadow_analysis_ms"].append(t_sh)

            # 3. Physics Timing
            t_ph0 = time.perf_counter()
            phys_res = physics_engine.analyze_target_physics(img_bgr, bbox, meta)
            t_ph = (time.perf_counter() - t_ph0) * 1000.0
            timings["physics_ms"].append(t_ph)

            # 4. Georeferencing Timing
            t_g0 = time.perf_counter()
            geo_res = physics_engine.georeference_target(bbox, w, h, meta, use_demo_fallback=False)
            t_g = (time.perf_counter() - t_g0) * 1000.0
            timings["geo_ms"].append(t_g)

            timings["total_phase3_overhead_ms"].append(t_sh + t_ph + t_g)

            # Draw visual debug annotations
            gx1, gy1, gx2, gy2 = bbox
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 255, 255), 2)
            cv2.putText(vis_img, f"{d['class']} ({d['confidence']:.2f})", (gx1, max(15, gy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            if shadow_res.detected and shadow_res.start_point and shadow_res.end_point:
                sp = tuple(shadow_res.start_point)
                ep = tuple(shadow_res.end_point)
                # Shadow ray line
                cv2.line(vis_img, sp, ep, (0, 0, 255), 2)
                cv2.circle(vis_img, sp, 4, (0, 255, 0), -1)  # Green = shadow start
                cv2.circle(vis_img, ep, 4, (0, 0, 255), -1)  # Red = shadow termination

                label_shadow = f"SHADOW: {shadow_res.shadow_length_px:.1f}px [{shadow_res.status}]"
                if shadow_res.shadow_length_m is not None:
                    label_shadow += f" | {shadow_res.shadow_length_m:.2f}m [DERIVED]"
                else:
                    label_shadow += " | Metric [UNAVAILABLE]"

                cv2.putText(vis_img, label_shadow, (gx1, min(h - 10, gy2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(vis_img, f"SHADOW: UNAVAILABLE [NO_DROPOUT]", (gx1, min(h - 10, gy2 + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

            # Physics elevation label
            elev = phys_res["elevation"]
            if elev["elevation_m"] is not None:
                elev_lbl = f"ELEVATION: {elev['elevation_m']:.2f}m [{elev['status']}]"
            else:
                elev_lbl = f"ELEVATION: null [UNAVAILABLE]"
            cv2.putText(vis_img, elev_lbl, (gx1, min(h - 10, gy2 + 34)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)

            # Geo label
            if geo_res.latitude is not None:
                geo_lbl = f"GEO: {geo_res.latitude:.5f}, {geo_res.longitude:.5f} [{geo_res.status}] ({geo_res.position_type})"
            else:
                geo_lbl = f"GEO: null [{geo_res.status}]"
            cv2.putText(vis_img, geo_lbl, (gx1, min(h - 10, gy2 + 50)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        out_vis_path = os.path.join("reports", "phase3", f"phase3_debug_{name}.png")
        cv2.imwrite(out_vis_path, vis_img)
        processed_samples.append((name, out_vis_path, len(detections)))

    # Compute benchmark metrics
    avg_det = np.mean(timings["detection_ms"]) if timings["detection_ms"] else 0.0
    avg_sh = np.mean(timings["shadow_analysis_ms"]) if timings["shadow_analysis_ms"] else 0.0
    avg_ph = np.mean(timings["physics_ms"]) if timings["physics_ms"] else 0.0
    avg_geo = np.mean(timings["geo_ms"]) if timings["geo_ms"] else 0.0
    avg_overhead = np.mean(timings["total_phase3_overhead_ms"]) if timings["total_phase3_overhead_ms"] else 0.0

    print("============================================================")
    print("PHASE 3 BENCHMARK & VISUAL EVALUATION RESULTS")
    print("============================================================")
    print(f"Shadow Analysis Latency (mean):     {avg_sh:.3f} ms")
    print(f"Physics Engine Latency (mean):      {avg_ph:.3f} ms")
    print(f"Georeferencing Latency (mean):      {avg_geo:.3f} ms")
    print(f"Total Phase 3 Overhead per target:  {avg_overhead:.3f} ms")
    print("------------------------------------------------------------")
    print("Generated Debug Images:")
    for name, p, count in processed_samples:
        print(f" - [{name}] ({count} targets): {p}")
    print("============================================================")


if __name__ == "__main__":
    generate_visual_artifacts_and_benchmark()
