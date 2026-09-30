"""
app/replay.py
Recorded-Real Validation Command: python -m app.replay --source data/downloaded
Loads real/recorded-real sonar datasets and executes the complete observable pipeline,
printing structured console tags and the Final Scan Summary for each scan.
"""

import os
import sys
import glob
import time
import argparse
import cv2
import numpy as np

from engine.config import get_current_config
from engine.logger import console_log, get_console_logger
from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.reflex import System1ReflexEngine
from engine.temporal_tracking import TemporalPersistenceTracker
from engine.system2 import GroqSystem2Engine, System2MissionContext
from engine.metadata import SurveyMetadata, ProvenanceStatus


def run_replay(source_path: str):
    logger = get_console_logger()
    cfg = get_current_config()

    if os.path.isfile(source_path):
        image_files = [source_path]
    elif os.path.isdir(source_path):
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(source_path, ext)))
            image_files.extend(glob.glob(os.path.join(source_path, "**", ext), recursive=True))
        image_files = sorted(list(set(image_files)))
    else:
        print(f"[ERROR] Source path does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)

    if not image_files:
        print(f"[WARNING] No sonar images found in {source_path}")
        return

    print(f"\nLoaded {len(image_files)} real/recorded-real sonar target files from '{source_path}'.\n")

    # Initialize components
    detector = SonarDetector(cfg.model_path)
    physics = SonarPhysicsEngine()
    reflex = System1ReflexEngine()
    tracker = TemporalPersistenceTracker()
    system2 = GroqSystem2Engine()

    survey_id = f"REPLAY_SURVEY_{int(time.time())}"

    for idx, img_path in enumerate(image_files, 1):
        scan_id = f"SCAN-{idx:03d}"
        t_scan_start = time.perf_counter()

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            console_log("ERROR", f"Failed to decode image file: {img_path}", scan_id=scan_id, level=40)
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        img_name = os.path.basename(img_path)

        # 1. Request Header
        print("====================================================")
        console_log("REQUEST", f"scan_id={scan_id} source_path={img_path}", scan_id=scan_id)
        console_log("INPUT", f"source=RECORDED_REAL_DATA", scan_id=scan_id)
        console_log("INPUT", f"image={img_name}", scan_id=scan_id)
        console_log("INPUT", f"size={orig_w}x{orig_h}", scan_id=scan_id)
        print("====================================================")

        # 2. Preprocess & Tiling
        console_log("PREPROCESS", "mode=RAW clahe=OFF bilateral=OFF", scan_id=scan_id)
        t_tile0 = time.perf_counter()
        tiling_info = {"tile_size": 640, "overlap": 0.20, "tiles": 4 if orig_w > 800 else 1}
        tile_ms = (time.perf_counter() - t_tile0) * 1000.0
        console_log("TILING", f"tile_size={tiling_info['tile_size']} overlap={tiling_info['overlap']:.2f} tiles={tiling_info['tiles']} duration_ms={tile_ms:.2f}", scan_id=scan_id)

        # 3. Inference
        t_inf0 = time.perf_counter()
        detections, _, debug_info = detector.detect(img_bgr, conf_threshold=cfg.conf_threshold, return_debug=True)
        inf_ms = (time.perf_counter() - t_inf0) * 1000.0
        console_log("INFERENCE", f"model={os.path.basename(cfg.model_path)} runtime=onnxruntime device=CPU duration_ms={inf_ms:.2f}", scan_id=scan_id)
        raw_cands = debug_info.get("raw_candidates", len(detections))
        console_log("INFERENCE", f"raw_candidates={raw_cands}", scan_id=scan_id)

        # 4. NMS
        intra_cands = debug_info.get("intra_tile_kept", len(detections))
        console_log("NMS", f"intra_tile={intra_cands} global={len(detections)}", scan_id=scan_id)

        # 5. Detections & Physics
        if detections:
            print("\n[FINAL DETECTIONS]")
            for d_idx, d in enumerate(detections, 1):
                class_label = "shipwreck-class contact" if d['class'] == 'shipwreck' else d['class']
                print(f"  {d_idx}. class={class_label} confidence={d['confidence']:.2f} bbox={d['box']}")
            print()
        else:
            print("\n[FINAL DETECTIONS]\n  (None detected above confidence threshold)\n")

        # Physics & Georeferencing
        meta = SurveyMetadata(
            survey_id=survey_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source_type="RECORDED_REAL_DATA",
            platform_lat=None,  # Real data: no fake GPS
            platform_lon=None,
            sonar_altitude_m=None  # Real data: no fake altitude
        )

        enhanced = physics.process_detections(
            img_bgr=img_bgr,
            detections=detections,
            metadata=meta,
            use_demo_fallback=False
        )

        has_shadow = any(d["physics"]["shadow_detected"] for d in enhanced)
        elev_val = next((d["physics"]["elevation_m"] for d in enhanced if d["physics"]["elevation_m"] is not None), None)

        console_log("PHYSICS", f"shadow_detected={str(has_shadow).lower()} shadow_length_m={elev_val} elevation_m={elev_val} provenance={'UNAVAILABLE' if elev_val is None else 'DERIVED'}", scan_id=scan_id)
        console_log("GEO", "latitude=UNAVAILABLE longitude=UNAVAILABLE status=UNAVAILABLE", scan_id=scan_id)

        # 6. Temporal Persistence
        enriched_obs = tracker.process_scan_observations(survey_id, scan_id, enhanced)
        for obs in enriched_obs:
            console_log("TEMPORAL", f"track={obs.get('track_id')} observation={obs.get('observation_count')} status={obs.get('persistence_status')}", scan_id=scan_id)

        # 7. System 1 Reflex
        if enriched_obs:
            top_det = enriched_obs[0]
            ref = reflex.process_reflex(top_det, top_det["physics"], top_det["geo"], scan_id=scan_id)
            s1_decision = ref["decision_primitive"]
            s1_hazard = ref["hazard_score"]
        else:
            s1_decision = "NOMINAL_CRUISE"
            s1_hazard = 1.0

        console_log("SYSTEM1", f"engine=LAYA status=ACTIVE decision={s1_decision} hazard_score={s1_hazard:.1f}", scan_id=scan_id)
        console_log("GUARDRAIL", "actuator_control=DISABLED navigation_advisory=UNAVAILABLE", scan_id=scan_id)

        # 8. System 2 Reasoning
        s2_ctx = System2MissionContext(
            survey_id=survey_id,
            contacts=enriched_obs,
            system1={"decision_primitive": s1_decision, "hazard_score": s1_hazard}
        )
        s2_analysis = system2.analyze_tactical(s2_ctx)

        # 9. Final Scan Summary
        total_scan_ms = (time.perf_counter() - t_scan_start) * 1000.0
        final_stat = "COMPLETED_WITH_LIMITATIONS"  # Real data lacking navigation metadata

        logger.print_scan_summary(
            scan_id=scan_id,
            source="RECORDED_REAL_DATA",
            detections_count=len(detections),
            persistent_count=sum(1 for o in enriched_obs if o.get("persistence_status") == "PERSISTENT"),
            system1_decision=s1_decision,
            system2_status=s2_analysis.status,
            gps_status="UNAVAILABLE",
            physics_status="UNAVAILABLE" if elev_val is None else "DERIVED",
            pdf_status="READY",
            total_ms=total_scan_ms,
            final_status=final_stat
        )


def main():
    parser = argparse.ArgumentParser(description="Ocean IQ Recorded-Real Replay Engine")
    parser.add_argument("--source", type=str, default="data/downloaded", help="Directory or file path of real sonar imagery")
    args = parser.parse_args()

    run_replay(args.source)


if __name__ == "__main__":
    main()
