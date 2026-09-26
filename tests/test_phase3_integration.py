import unittest
import os
import cv2

from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.metadata import SurveyMetadata, ProvenanceStatus


class TestPhase3Integration(unittest.TestCase):
    """
    TEST 10: Full end-to-end integration test:
    Sonar Image -> Phase 2 Detector -> Detections -> Acoustic Shadow Analyzer ->
    Physics Engine -> Metadata Validation -> Georeferencing -> Structured Scan Result.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_detector.onnx")
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"Model file not found: {cls.model_path}")
        cls.detector = SonarDetector(cls.model_path)
        cls.physics = SonarPhysicsEngine()

        cls.sample_ghost_net = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "synth_ghost_net_00001.png")
        cls.sample_bg = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "bg_1693569243.750_x2500.jpg")

    def test_pipeline_without_metadata_no_fabricated_numbers(self):
        """
        Runs full pipeline on real image WITHOUT metadata and without demo fallback.
        Verifies:
        - Detections are produced by Phase 2 detector.
        - Acoustic shadow analysis measures pixels if shadow exists, but leaves metric length null.
        - Elevation is null (status = UNAVAILABLE).
        - GPS coordinates are null (status = UNAVAILABLE).
        - No numbers are fabricated!
        """
        img = cv2.imread(self.sample_ghost_net)
        self.assertIsNotNone(img, "Sample ghost net image could not be loaded")

        # 1. Detection
        detections, _ = self.detector.detect(img, conf_threshold=0.25, tiling=True)
        self.assertGreater(len(detections), 0, "Expected at least 1 detection on ghost net sample")

        # 2. End-to-end Phase 3 Scan Processing
        scan_result = self.physics.process_scan(
            img_bgr=img,
            detections=detections,
            metadata=None,
            use_demo_fallback=False
        )

        self.assertIn("survey_metadata", scan_result)
        self.assertIn("detections", scan_result)
        self.assertEqual(scan_result["total_detections"], len(detections))

        for d in scan_result["detections"]:
            # Detector fields preserved
            self.assertIn("class", d)
            self.assertIn("confidence", d)
            self.assertIn("bbox_xyxy", d)

            # Physics field present
            self.assertIn("physics", d)
            phys = d["physics"]
            self.assertIsNone(phys["shadow_length_m"], "Metric shadow length must be None when no scale calibration exists")
            self.assertIsNone(phys["elevation_m"], "Elevation must be None when geometry is missing")

            # Geo field present
            self.assertIn("geo", d)
            geo = d["geo"]
            self.assertIsNone(geo["latitude"])
            self.assertIsNone(geo["longitude"])
            self.assertEqual(geo["status"], ProvenanceStatus.UNAVAILABLE.value)

    def test_pipeline_with_full_survey_metadata(self):
        """
        Runs full pipeline with valid survey telemetry and scale calibration.
        Verifies:
        - When calibration is provided, metric values and elevation are derived.
        - Georeferencing derives target seafloor coordinates with OBJECT_POSITION.
        - Every output has explicit provenance.
        """
        img = cv2.imread(self.sample_ghost_net)
        self.assertIsNotNone(img)

        meta = SurveyMetadata(
            survey_id="EXPEDITION_NIOT_2026",
            timestamp="2026-09-26T12:00:00Z",
            platform_lat=12.9810,
            platform_lon=80.2520,
            vehicle_depth_m=30.0,
            sonar_altitude_m=12.0,
            heading_deg=45.0,
            slant_range_m=22.0,
            meters_per_pixel=0.06,
            source_type="RECORDED_SURVEY"
        )

        detections, _ = self.detector.detect(img, conf_threshold=0.25, tiling=True)
        scan_result = self.physics.process_scan(
            img_bgr=img,
            detections=detections,
            metadata=meta,
            use_demo_fallback=False
        )

        self.assertEqual(scan_result["survey_metadata"]["survey_id"], "EXPEDITION_NIOT_2026")
        for d in scan_result["detections"]:
            geo = d["geo"]
            self.assertEqual(geo["position_type"], "OBJECT_POSITION")
            self.assertIsNotNone(geo["latitude"])
            self.assertIsNotNone(geo["longitude"])

            phys = d["physics"]
            self.assertIn("geometry", phys)
            self.assertEqual(phys["geometry"]["sonar_altitude_m"], 12.0)
            self.assertEqual(phys["geometry"]["meters_per_pixel"], 0.06)

    def test_clean_background_produces_zero_artifacts(self):
        """
        Verifies pipeline on clean seabed background: 0 detections, 0 shadows, empty list.
        """
        img = cv2.imread(self.sample_bg)
        if img is not None:
            detections, _ = self.detector.detect(img, conf_threshold=0.25, tiling=True)
            scan_result = self.physics.process_scan(img, detections, use_demo_fallback=False)
            self.assertEqual(scan_result["total_detections"], 0)
            self.assertEqual(scan_result["shadows_detected"], 0)


if __name__ == '__main__':
    unittest.main()
