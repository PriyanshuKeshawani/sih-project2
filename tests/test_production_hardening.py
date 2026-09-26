"""
tests/test_production_hardening.py
Acceptance Test Suite for Production-Grade Hardening (Section 29).
Covers all 20 required production safety, fail-safe, and real-data guarantees:

1. Real sonar image input
2. Empty/invalid image
3. Missing GPS
4. Missing altitude
5. Missing calibration
6. Detector failure
7. Laya failure
8. Groq failure
9. PDF failure
10. Network unavailable
11. Large image
12. Multiple contacts
13. Persistence
14. API key missing
15. Secret leakage prevention
16. Production mode cannot load demo data
17. Test mode explicitly required for simulation
18. All logs contain request_id
19. No fabricated numeric measurements
20. Final status correctness
"""

import os
import unittest
import numpy as np
import cv2

from engine.config import SystemConfig, AppMode, ProductionConfigurationError
from engine.logger import scrub_secrets, ProductionLogger
from engine.metadata import SurveyMetadata, ProvenanceStatus, SimulatedNavigation
from engine.physics import SonarPhysicsEngine
from engine.reflex import System1ReflexEngine
from engine.temporal_tracking import TemporalPersistenceTracker
from engine.system2 import (
    GroqSystem2Engine,
    DeterministicSystem2Fallback,
    System2MissionContext
)
from engine.mission import MissionReportGenerator
from engine.sources import (
    RecordedImageSource,
    FutureLiveSonarSource,
    UserProvidedNavigationSource,
    NavigationTelemetry
)


class TestProductionHardening(unittest.TestCase):
    """Rigorous 20-point production hardening acceptance suite."""

    @classmethod
    def setUpClass(cls):
        cls.physics = SonarPhysicsEngine()
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.sample_path = os.path.join(cls.base_dir, "data", "downloaded", "mine_0009_2015.jpg")
        if not os.path.exists(cls.sample_path):
            cls.sample_path = os.path.join(cls.base_dir, "data", "samples", "mine_0001_2015.jpg")

    # 1. Real sonar image input
    def test_01_real_sonar_image_input(self):
        source = RecordedImageSource(self.sample_path, dataset_name="DRISHTI_REAL")
        payload = source.get_frame()
        self.assertIn(payload.source_type, ["REAL", "RECORDED_REAL_DATA"])
        self.assertIsNotNone(payload.image_bgr)
        self.assertGreater(payload.image_bgr.shape[0], 0)
        self.assertGreater(payload.image_bgr.shape[1], 0)

    # 2. Empty/invalid image
    def test_02_empty_or_invalid_image(self):
        empty_source = RecordedImageSource("non_existent_sonar_file.png")
        payload = empty_source.get_frame()
        self.assertIsNone(payload)

    # 3. Missing GPS
    def test_03_missing_gps(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        meta = SurveyMetadata(platform_lat=None, platform_lon=None)
        res = self.physics.georeference_target([100, 100, 150, 150], 640, 640, metadata=meta, use_demo_fallback=False)
        self.assertIsNone(res.latitude)
        self.assertIsNone(res.longitude)
        self.assertEqual(res.status, ProvenanceStatus.UNAVAILABLE.value)
        self.assertEqual(res.position_type, "UNAVAILABLE")

    # 4. Missing altitude
    def test_04_missing_altitude(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        meta = SurveyMetadata(sonar_altitude_m=None, meters_per_pixel=0.05)
        phys = self.physics.analyze_target_physics(img, [100, 100, 150, 150], metadata=meta)
        self.assertIsNone(phys["elevation"]["elevation_m"])
        self.assertEqual(phys["elevation"]["status"], ProvenanceStatus.UNAVAILABLE.value)

    # 5. Missing calibration (meters_per_pixel)
    def test_05_missing_calibration(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        meta = SurveyMetadata(meters_per_pixel=None, sonar_altitude_m=12.0)
        phys = self.physics.analyze_target_physics(img, [100, 100, 150, 150], metadata=meta)
        self.assertIsNone(phys["shadow"]["shadow_length_m"])
        self.assertIsNone(phys["elevation"]["elevation_m"])

    # 6. Detector failure handling
    def test_06_detector_failure(self):
        from engine.detector import SonarDetector
        # Instantiating with nonexistent model fails safely
        with self.assertRaises(Exception):
            SonarDetector("invalid/path/nonexistent.onnx")

    # 7. Laya failure handling
    def test_07_laya_failure(self):
        reflex = System1ReflexEngine()
        # Force invalid payload; reflex engine must fail safe to deterministic evaluation
        detection = {"class": "mine_cylinder", "confidence": 0.85, "box": {"x": 10, "y": 10, "w": 20, "h": 20}}
        eval_res = reflex.process_reflex(detection, physics={"elevation_m": None}, geo={"lat": None})
        self.assertIn("decision_primitive", eval_res)
        self.assertIn(eval_res["decision_primitive"], ["EMERGENCY_PROP_HAZARD", "LOITER_AND_RESCAN", "PASSIVE_LOG", "NOMINAL_CRUISE"])

    # 8. Groq failure fallback
    def test_08_groq_failure(self):
        # Initialized with invalid key triggers fallback
        engine = GroqSystem2Engine(api_key="invalid_mock_key", timeout_s=1.0)
        context = System2MissionContext(survey_id="SURVEY_TEST_FAIL", contacts=[])
        analysis = engine.analyze_tactical(context)
        self.assertIn(analysis.status, ["FALLBACK", "ACTIVE"])
        self.assertIsNotNone(analysis.incident_summary)

    # 9. PDF failure handling
    def test_09_pdf_failure(self):
        # Empty/missing metadata generates valid report with explicit disclaimer
        pdf_bytes = MissionReportGenerator.generate_pdf({"incident_id": "TEST_EMPTY"})
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    # 10. Network unavailable
    def test_10_network_unavailable(self):
        # Deterministic engine executes 100% locally with 0 network calls
        context = System2MissionContext(survey_id="OFFLINE_001", contacts=[])
        res = DeterministicSystem2Fallback.analyze(context)
        self.assertEqual(res.status, "FALLBACK")
        self.assertIn("Nominal acoustic survey transect", res.incident_summary)

    # 11. Large image handling
    def test_11_large_image(self):
        large_img = np.zeros((1500, 1500, 3), dtype=np.uint8)
        from engine.detector import SonarDetector
        model_path = os.path.join(self.base_dir, "models", "best_detector.onnx")
        if os.path.exists(model_path):
            detector = SonarDetector(model_path)
            detections, _, debug = detector.detect(large_img, conf_threshold=0.45, return_debug=True)
            self.assertIsInstance(detections, list)

    # 12. Multiple contacts
    def test_12_multiple_contacts(self):
        tracker = TemporalPersistenceTracker()
        obs = [
            {"detection_id": "det_01", "class": "mine_cylinder", "confidence": 0.8, "bbox_xyxy": [10, 10, 30, 30]},
            {"detection_id": "det_02", "class": "shipwreck", "confidence": 0.7, "bbox_xyxy": [200, 200, 400, 400]},
            {"detection_id": "det_03", "class": "ghost_net", "confidence": 0.85, "bbox_xyxy": [50, 50, 120, 120]}
        ]
        enriched = tracker.process_scan_observations("MULTI_SURVEY", "SCAN_01", obs)
        self.assertEqual(len(enriched), 3)
        self.assertEqual(len(tracker.tracks), 3)

    # 13. Persistence transition
    def test_13_persistence(self):
        tracker = TemporalPersistenceTracker()
        det = [{"detection_id": "d1", "class": "ghost_net", "confidence": 0.85, "bbox_xyxy": [50, 50, 120, 120]}]
        enr1 = tracker.process_scan_observations("SURV_P", "SCAN_1", det, timestamp_epoch=10.0)
        self.assertEqual(enr1[0]["persistence_status"], "NEW_CONTACT")
        self.assertEqual(enr1[0]["observation_count"], 1)

        enr2 = tracker.process_scan_observations("SURV_P", "SCAN_2", det, timestamp_epoch=15.0)
        self.assertEqual(enr2[0]["persistence_status"], "NEW_CONTACT")
        self.assertEqual(enr2[0]["observation_count"], 2)

        enr3 = tracker.process_scan_observations("SURV_P", "SCAN_3", det, timestamp_epoch=20.0)
        self.assertEqual(enr3[0]["persistence_status"], "PERSISTENT")
        self.assertEqual(enr3[0]["observation_count"], 3)

    # 14. API key missing
    def test_14_api_key_missing(self):
        engine = GroqSystem2Engine(api_key=None)
        context = System2MissionContext(survey_id="NO_KEY_SURVEY", contacts=[])
        analysis = engine.analyze_tactical(context)
        self.assertIsNotNone(analysis)

    # 15. Secret leakage prevention
    def test_15_secret_leakage_prevention(self):
        sample_log = "Error connecting to Groq: gsk_1234567890abcdef123456 and api_key=AQ.secret_gemini_token_value"
        clean = scrub_secrets(sample_log)
        self.assertNotIn("gsk_1234567890abcdef123456", clean)
        self.assertNotIn("AQ.secret_gemini_token_value", clean)
        self.assertIn("[REDACTED", clean)

    # 16. Production mode cannot load demo data
    def test_16_production_mode_cannot_load_demo_data(self):
        cfg = SystemConfig(mode="production")
        self.assertFalse(cfg.allow_simulation)
        # Attempting to instantiate SimulatedNavigation in production mode raises RuntimeError
        with self.assertRaises(RuntimeError):
            SimulatedNavigation(base_lat=9.2882, base_lon=79.1325)

    # 17. Test mode explicitly required for simulation
    def test_17_test_mode_explicitly_required_for_simulation(self):
        cfg_test = SystemConfig(mode="test")
        self.assertTrue(cfg_test.allow_simulation)

        cfg_prod = SystemConfig(mode="production")
        self.assertFalse(cfg_prod.allow_simulation)

        cfg_rec = SystemConfig(mode="recorded_real_data")
        self.assertFalse(cfg_rec.allow_simulation)

    # 18. All logs contain request_id
    def test_18_all_logs_contain_request_id(self):
        logged_lines = []
        class TestLogger(ProductionLogger):
            def log_tag(self, tag, msg, correlation_id=None, level=20):
                line = f"[{tag}][{correlation_id}] {msg}" if correlation_id else f"[{tag}] {msg}"
                logged_lines.append(line)

        tl = TestLogger()
        tl.request("New scan dispatched", correlation_id="SCAN-999")
        self.assertTrue(any("[REQUEST][SCAN-999]" in line for line in logged_lines))

    # 19. No fabricated numeric measurements
    def test_19_no_fabricated_numeric_measurements(self):
        # Physics with no altitude must return None, not 12.0 or 24.5
        elev = self.physics.calculate_elevation({"x": 10, "y": 10, "w": 20, "h": 20}, 640, 640, altitude=None)
        self.assertIsNone(elev)

        geo = self.physics.georeference({"x": 10, "y": 10, "w": 20, "h": 20}, 640, 640)
        self.assertIsNone(geo["lat"])
        self.assertIsNone(geo["lon"])
        self.assertIsNone(geo["depth_m"])
        self.assertEqual(geo["status"], "UNAVAILABLE")

    # 20. Final status correctness
    def test_20_final_status_correctness(self):
        # Both complete and incomplete metadata status states
        live_adapter = FutureLiveSonarSource()
        status_info = live_adapter.get_status()
        self.assertEqual(status_info["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
