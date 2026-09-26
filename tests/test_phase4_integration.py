import unittest
import os
import cv2

from engine.detector import SonarDetector
from engine.physics import SonarPhysicsEngine
from engine.metadata import SurveyMetadata, ProvenanceStatus
from engine.reflex import System1ReflexEngine, DecisionPrimitive


class TestPhase4PipelineIntegration(unittest.TestCase):
    """
    End-to-End Pipeline Integration Test (Section 20):
    Sonar Image -> Phase 2 Tiled Detector -> Phase 3 Physics & Georeferencing ->
    Phase 4 System 1 Reflex Engine -> Final Structured Scan Result.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_detector.onnx")
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"Model not found: {cls.model_path}")
        cls.detector = SonarDetector(cls.model_path)
        cls.physics = SonarPhysicsEngine()
        cls.sample_ghost_net = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "synth_ghost_net_00001.png")

    def test_full_pipeline_flow_to_structured_result(self):
        """
        Verifies that every detection passing through the full pipeline contains:
        detection, physics, geo, risk, decision, reasons, event.
        """
        img = cv2.imread(self.sample_ghost_net)
        self.assertIsNotNone(img)

        meta = SurveyMetadata(
            survey_id="SURVEY_BAY_OF_BENGAL_01",
            platform_lat=13.0827,
            platform_lon=80.2707,
            vehicle_depth_m=20.0,
            sonar_altitude_m=10.0,
            heading_deg=90.0,
            meters_per_pixel=0.05,
            source_type="RECORDED_SURVEY"
        )

        # 1. Phase 2: Detection
        detections, _ = self.detector.detect(img, conf_threshold=0.25, tiling=True)
        self.assertGreater(len(detections), 0)

        # 2. Phase 3 & 4: Integrated Physics, Geo, and Reflex Processing
        scan_result = self.physics.process_scan(
            img_bgr=img,
            detections=detections,
            metadata=meta,
            use_demo_fallback=False,
            enable_reflex=True
        )

        self.assertIn("survey_metadata", scan_result)
        self.assertIn("detections", scan_result)
        self.assertIn("critical_alerts", scan_result)

        for d in scan_result["detections"]:
            # Detection fields
            self.assertIn("class", d)
            self.assertIn("confidence", d)
            self.assertIn("bbox_xyxy", d)

            # Physics fields
            self.assertIn("physics", d)
            self.assertIn("shadow_detected", d["physics"])

            # Geo fields
            self.assertIn("geo", d)
            self.assertIn("position_type", d["geo"])

            # Reflex & Decision fields (Phase 4 requirements)
            self.assertIn("risk", d)
            self.assertIn("decision", d)
            self.assertIn("reasons", d)
            self.assertIn("event", d)
            self.assertIn("reflex", d)

            self.assertIsInstance(d["risk"], float)
            self.assertIn(d["decision"], [
                DecisionPrimitive.EMERGENCY_PROP_HAZARD.value,
                DecisionPrimitive.LOITER_AND_RESCAN.value,
                DecisionPrimitive.PASSIVE_LOG.value
            ])
            self.assertIsInstance(d["reasons"], list)
            self.assertGreater(len(d["reasons"]), 0)


if __name__ == '__main__':
    unittest.main()
