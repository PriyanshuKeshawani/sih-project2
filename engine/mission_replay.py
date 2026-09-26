"""
engine/mission_replay.py
End-to-End Mission Replay & Benchmarking Engine.

SIH 2026 Problem Statement 26057 — MoES / NIOT Chennai
Full-system mission pipeline orchestrator:
IMAGE -> PREPROCESS -> TILING -> YOLO -> NMS -> PHYSICS -> GEO
  -> LAYA SYSTEM 1 -> TEMPORAL TRACKING -> SYSTEM 2 / FALLBACK
  -> MISSION EVENT -> UI PAYLOAD -> PDF DISPATCH
"""

import os
import time
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from engine.dataset_loader import SonarDatasetLoader, SonarSampleRecord
from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.reflex import System1ReflexEngine
from engine.temporal_tracking import TemporalPersistenceTracker
from engine.system2 import System2Queue, System2MissionContext, GroqSystem2Engine, DeterministicSystem2Fallback
from engine.mission import MissionReportGenerator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_detector.onnx")
REPLAYS_DIR = os.path.join(BASE_DIR, "reports", "phase8", "replays")


class MissionReplayEngine:
    """
    Executes reproducible, end-to-end mission replays against public sonar benchmarks.
    Measures component-level latency and stores complete auditable telemetry.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.loader = SonarDatasetLoader()
        self.detector = SonarDetector(model_path)
        self.physics = SonarPhysicsEngine()
        self.reflex = System1ReflexEngine()
        self.system2_engine = GroqSystem2Engine()
        os.makedirs(REPLAYS_DIR, exist_ok=True)

    def run_scenario(
        self,
        scenario_id: str,
        survey_id: Optional[str] = None,
        conf_threshold: float = 0.45,
        altitude: float = 12.0,
        generate_pdf: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a complete single-scan mission replay for a designated scenario.
        """
        replay_id = f"REPLAY_{scenario_id.replace(' ', '_').upper()}_{uuid.uuid4().hex[:6].upper()}"
        survey_id = survey_id or f"SURVEY_{uuid.uuid4().hex[:6].upper()}"
        scan_id = f"SCAN_{uuid.uuid4().hex[:6].upper()}"
        
        timings: Dict[str, float] = {}
        t_start_total = time.perf_counter()

        # 1. Image Loading & Provenance
        t0 = time.perf_counter()
        img_bgr, record = self.loader.load_scenario(scenario_id)
        timings["image_loading_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        if img_bgr is None:
            return {
                "replay_id": replay_id,
                "scenario_id": scenario_id,
                "status": "NOT AVAILABLE",
                "dataset_name": record.dataset_name,
                "source_url": record.source_url,
                "license": record.license,
                "provenance": record.provenance,
                "timings": timings
            }

        orig_h, orig_w = img_bgr.shape[:2]

        # 2. Preprocessing & Tiled Inference (YOLO + NMS)
        t0 = time.perf_counter()
        detections, annotated_bgr, debug_info = self.detector.detect(
            img_bgr,
            conf_threshold=conf_threshold,
            draw_tiles=True,
            return_debug=True
        )
        timings["tiled_inference_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        timings["yolo_cpu_inference_ms"] = debug_info.get("inference_time_ms", timings["tiled_inference_ms"])
        timings["tiling_overhead_ms"] = debug_info.get("tiling_time_ms", 0.0)
        timings["nms_time_ms"] = debug_info.get("nms_time_ms", 0.0)

        # 3. Physics & Georeferencing
        t0 = time.perf_counter()
        raw_observations = []
        for i, d in enumerate(detections):
            elev = self.physics.calculate_elevation(d['box'], orig_w, orig_h, altitude=altitude)
            geo = self.physics.georeference(d['box'], orig_w, orig_h)
            
            raw_observations.append({
                "detection_id": d.get("detection_id", f"det_{i+1:03d}"),
                "class": d["class"],
                "display_label": d.get("display_label", f"{d['class'].upper()}-CLASS CONTACT"),
                "confidence": d["confidence"],
                "box": d["box"],
                "bbox_xyxy": d["bbox_xyxy"],
                "tile_id": d.get("tile_id", "tile_000"),
                "elevation_m": elev,
                "elevation_provenance": "DERIVED" if elev is not None else "UNAVAILABLE",
                "geo": geo,
                "geo_provenance": "DEMO" if (geo and geo.get("lat")) else "UNAVAILABLE"
            })
        timings["physics_geo_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 4. Temporal Multi-Ping Tracking
        t0 = time.perf_counter()
        tracker = TemporalPersistenceTracker()
        enriched_obs = tracker.process_scan_observations(survey_id, scan_id, raw_observations)
        timings["temporal_tracking_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # 5. System 1 Edge Reflex
        t0 = time.perf_counter()
        results = []
        for obs in enriched_obs:
            temporal_ctx = {
                "track_id": obs.get("track_id"),
                "persistence_status": obs.get("persistence_status"),
                "observation_count": obs.get("observation_count"),
                "track_age_s": obs.get("track_age_s")
            }
            ref = self.reflex.process_reflex(
                detection=obs,
                physics={"elevation_m": obs.get("elevation_m"), "shadow_detected": bool(obs.get("elevation_m"))},
                geo=obs.get("geo"),
                temporal=temporal_ctx
            )
            obs["reflex"] = ref
            results.append(obs)
        timings["laya_system1_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        s1_status = self.reflex.system1_manager.get_status()
        if results and "system1" in results[0]["reflex"]:
            primary_s1 = results[0]["reflex"]["system1"]
        else:
            primary_s1 = {
                "engine": s1_status.get("active_engine", "laya"),
                "decision_primitive": "PASSIVE_LOG",
                "hazard_score": 1.0,
                "needs_operator_review": False,
                "confidence": None,
                "latency_ms": timings["laya_system1_ms"],
                "status": s1_status.get("status", "ACTIVE")
            }

        # 6. System 2 Tactical Reasoning
        t0 = time.perf_counter()
        s2_temporal = {
            "persistence_status": results[0].get("persistence_status", "NEW_CONTACT"),
            "observation_count": results[0].get("observation_count", 1),
            "track_age_s": results[0].get("track_age_s", 0.0)
        } if results else None

        s2_context = System2MissionContext(
            survey_id=survey_id,
            contacts=results,
            geo=results[0]["geo"] if results else None,
            system1=primary_s1,
            metadata={"dataset": record.dataset_name, "license": record.license, "altitude_m": altitude},
            temporal=s2_temporal
        )
        
        system2_analysis = self.system2_engine.analyze_tactical(s2_context)
        timings["system2_total_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        timings["system2_type"] = "network_groq" if system2_analysis.status == "ACTIVE" else "deterministic_fallback"

        # 7. Mission Events Timeline
        timestamp_now = datetime.utcnow().strftime("%H:%M:%S")
        events = []
        if not results:
            events.append({"time": timestamp_now, "type": "NOMINAL_CRUISE", "text": "Clean seafloor verified. No anomalies detected."})
        else:
            primary = results[0]
            events.append({"time": timestamp_now, "type": "NEW_CONTACT", "text": f"{primary['display_label']} ({primary['confidence']*100:.1f}%) [Track {primary.get('track_id')}]"})
            events.append({"time": timestamp_now, "type": "SYSTEM_1", "text": f"System 1 -> {primary_s1.get('decision_primitive')} (Hazard {primary_s1.get('hazard_score')}/10)"})
            events.append({"time": timestamp_now, "type": "SYSTEM_2", "text": f"System 2 -> {system2_analysis.incident_summary[:60]}..."})

        # 8. PDF Generation
        pdf_bytes = None
        timings["pdf_generation_ms"] = 0.0
        if generate_pdf:
            t0 = time.perf_counter()
            primary_contact = results[0] if results else {}
            pdf_payload = {
                "incident_id": replay_id,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                "dataset_name": f"{record.dataset_name} ({record.license})",
                "is_demo_mode": True,
                "class": primary_contact.get("class", "CLEAN_SEABED"),
                "confidence": primary_contact.get("confidence", 0.0),
                "elevation_m": primary_contact.get("elevation_m"),
                "elevation_provenance": primary_contact.get("elevation_provenance", "UNAVAILABLE"),
                "lat": primary_contact.get("geo", {}).get("lat") if primary_contact.get("geo") else None,
                "lon": primary_contact.get("geo", {}).get("lon") if primary_contact.get("geo") else None,
                "track_id": primary_contact.get("track_id", "N/A"),
                "persistence_status": primary_contact.get("persistence_status", "N/A"),
                "observation_count": primary_contact.get("observation_count", 0),
                "track_age_s": primary_contact.get("track_age_s", 0.0),
                "system1": primary_s1,
                "system2": system2_analysis.model_dump()
            }
            pdf_bytes = MissionReportGenerator.generate_pdf(pdf_payload)
            timings["pdf_generation_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        timings["total_mission_ms"] = round((time.perf_counter() - t_start_total) * 1000.0, 2)

        # Store Replay Record
        replay_record = {
            "replay_id": replay_id,
            "scenario_id": scenario_id,
            "status": "COMPLETED",
            "dataset_name": record.dataset_name,
            "source_url": record.source_url,
            "license": record.license,
            "image_filename": record.image_filename,
            "provenance": record.provenance,
            "detections": results,
            "tracks": [t.to_dict() for t in tracker.get_tracks_for_survey(survey_id)],
            "system1": primary_s1,
            "system2": system2_analysis.model_dump(),
            "mission_events": events,
            "timings": timings,
            "has_pdf": pdf_bytes is not None and len(pdf_bytes) > 0
        }

        out_file = os.path.join(REPLAYS_DIR, f"{replay_id}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(replay_record, f, indent=2)

        return replay_record

    def run_multi_ping_replay(self, scenario_id: str, num_pings: int = 3) -> Dict[str, Any]:
        """
        Replays repeated observations of a contact to verify state transitions:
        Ping 1: NEW_CONTACT
        Ping 2+: PERSISTENT
        Simulated scan omission: TRANSIENT / UNCERTAIN
        """
        survey_id = f"SURVEY_MULTI_{uuid.uuid4().hex[:6].upper()}"
        tracker = TemporalPersistenceTracker()
        
        img_bgr, record = self.loader.load_scenario(scenario_id)
        if img_bgr is None:
            return {"status": "NOT AVAILABLE", "scenario_id": scenario_id}

        orig_h, orig_w = img_bgr.shape[:2]
        detections, _ = self.detector.detect(img_bgr, conf_threshold=0.45)

        ping_history = []
        for ping_idx in range(1, num_pings + 1):
            scan_id = f"SCAN_{ping_idx:02d}_{uuid.uuid4().hex[:4].upper()}"
            obs_batch = []
            for d in detections:
                elev = self.physics.calculate_elevation(d['box'], orig_w, orig_h, altitude=12.0)
                geo = self.physics.georeference(d['box'], orig_w, orig_h)
                obs_batch.append({
                    "class": d["class"],
                    "confidence": d["confidence"],
                    "box": d["box"],
                    "bbox_xyxy": d["bbox_xyxy"],
                    "elevation_m": elev,
                    "geo": geo
                })
            enriched = tracker.process_scan_observations(survey_id, scan_id, obs_batch)
            ping_history.append({
                "ping_idx": ping_idx,
                "scan_id": scan_id,
                "observations": enriched
            })

        # Add simulated omission scan to test TRANSIENT / UNCERTAIN detection if requested
        scan_id_omitted = f"SCAN_OMIT_{uuid.uuid4().hex[:4].upper()}"
        tracker.process_scan_observations(survey_id, scan_id_omitted, [])

        tracks = [t.to_dict() for t in tracker.get_tracks_for_survey(survey_id)]

        return {
            "survey_id": survey_id,
            "scenario_id": scenario_id,
            "dataset_name": record.dataset_name,
            "total_pings": num_pings,
            "ping_history": ping_history,
            "tracks": tracks
        }
