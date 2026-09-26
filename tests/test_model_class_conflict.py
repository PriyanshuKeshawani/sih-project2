"""
Phase 4.6 Unit and Integration Tests:
Model Class-Conflict, Closed-Set Unknown-Object Diagnosis, and MVP-Safe Labeling.
SIH 2026 Problem Statement 26057.
"""
import os
import unittest
import cv2
import numpy as np

from engine.detector import SonarDetector, CLASSES
from engine.preprocessing import PreprocessConfig, SonarPreprocessor

class TestModelClassConflict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = SonarDetector("models/best_detector.onnx")
        cls.panels_dir = os.path.join("reports", "phase4_6", "panels")
        cls.samples_dir = os.path.join("data", "samples")

    def test_known_class_prediction(self):
        """Known taxonomy samples should be detected with valid taxonomy items."""
        sample_path = os.path.join(self.samples_dir, "synth_ghost_net_00001.png")
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample not found: {sample_path}")

        img = cv2.imread(sample_path)
        dets, _ = self.detector.detect(img, conf_threshold=0.25)
        self.assertGreater(len(dets), 0, "Known ghost net sample must produce at least one detection")
        classes = [d["class"] for d in dets]
        self.assertIn("ghost_net", classes)

    def test_unknown_object_handling_and_taxonomy_mismatch(self):
        """Unknown non-taxonomy objects assigned to shipwreck must carry MVP-safe surrogate label."""
        rock_path = os.path.join(self.panels_dir, "panel_10_natural_rock.png")
        if not os.path.exists(rock_path):
            self.skipTest(f"Panel not found: {rock_path}")

        img = cv2.imread(rock_path)
        dets, _ = self.detector.detect(img, conf_threshold=0.25, tiling=False)
        self.assertGreater(len(dets), 0)

        for d in dets:
            if d["class"] == "shipwreck":
                self.assertEqual(d["taxonomy_status"], "CLOSED_SET_SURROGATE")
                self.assertEqual(d["display_label"], "SHIPWRECK-CLASS CONTACT")

    def test_confidence_handling_rejects_sand_ripples(self):
        """Weak seabed texture like sand ripples must produce confidence < 0.25 and no false alarms."""
        ripples_path = os.path.join(self.panels_dir, "panel_11_sand_ripples.png")
        if not os.path.exists(ripples_path):
            self.skipTest(f"Panel not found: {ripples_path}")

        img = cv2.imread(ripples_path)
        dets, _ = self.detector.detect(img, conf_threshold=0.25, tiling=False)
        self.assertEqual(len(dets), 0, "Sand ripples must not produce false detections at conf >= 0.25")

    def test_top_k_class_output_format(self):
        """Raw ONNX predictions must contain all 5 classes with non-negative probabilities."""
        tire_path = os.path.join(self.panels_dir, "panel_09_tire_rubber.png")
        if not os.path.exists(tire_path):
            self.skipTest(f"Panel not found: {tire_path}")

        img = cv2.imread(tire_path)
        img_640 = cv2.resize(img, (640, 640))
        boxes, scores, class_ids = self.detector._infer_tile_raw(img_640)

        self.assertEqual(boxes.shape[1], 4)
        self.assertEqual(scores.shape[0], boxes.shape[0])
        self.assertEqual(class_ids.shape[0], boxes.shape[0])
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        self.assertTrue(np.all((class_ids >= 0) & (class_ids < 5)))

    def test_preprocessing_comparison(self):
        """Test with preprocessing enabled vs disabled runs cleanly without crash."""
        sample_path = os.path.join(self.samples_dir, "mine_0001_2015.jpg")
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample not found: {sample_path}")

        img = cv2.imread(sample_path)
        dets_raw, _ = self.detector.detect(img, conf_threshold=0.45, preprocess=False)
        dets_prep, _ = self.detector.detect(img, conf_threshold=0.45, preprocess=True)

        self.assertIsInstance(dets_raw, list)
        self.assertIsInstance(dets_prep, list)

    def test_tiling_comparison(self):
        """Tiling vs direct resize both execute cleanly and produce bounded coordinates."""
        sample_path = os.path.join(self.samples_dir, "pipe_1693569383.780_x3500.jpg")
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample not found: {sample_path}")

        img = cv2.imread(sample_path)
        h, w = img.shape[:2]

        dets_tiled, _ = self.detector.detect(img, conf_threshold=0.45, tiling=True)
        dets_direct, _ = self.detector.detect(img, conf_threshold=0.45, tiling=False)

        for d in dets_tiled + dets_direct:
            b = d["bbox_xyxy"]
            self.assertGreaterEqual(b[0], 0)
            self.assertGreaterEqual(b[1], 0)
            self.assertLessEqual(b[2], w)
            self.assertLessEqual(b[3], h)


if __name__ == "__main__":
    unittest.main()
