"""
Regression Tests: API Detection Consistency and Coordinate Integrity.
SIH 2026 Problem Statement 26057.
"""
import os
import unittest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app, detector

class TestApiDetectionConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_img_path = os.path.join("reports", "display_audit", "reconstructed_test_input.png")
        if not os.path.exists(cls.test_img_path):
            # Fallback to sample image if audit artifact is missing
            cls.test_img_path = os.path.join("data", "samples", "synth_ghost_net_00001.png")
        cls.img_bgr = cv2.imread(cls.test_img_path)
        cls.orig_h, cls.orig_w = cls.img_bgr.shape[:2]

    def test_api_detection_count_matches_backend(self):
        """API detection count must exactly match direct backend detector output."""
        # 1. Direct detector call
        backend_dets, _, debug_info = detector.detect(
            self.img_bgr,
            conf_threshold=0.45,
            return_debug=True
        )

        # 2. Call /api/scan with uploaded file
        with open(self.test_img_path, "rb") as f:
            resp = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.45", "altitude": "12.0"},
                files={"file": ("test.png", f, "image/png")}
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        api_dets = data["detections"]

        self.assertEqual(len(api_dets), len(backend_dets),
                         f"API count {len(api_dets)} must match backend count {len(backend_dets)}")
        self.assertEqual(len(api_dets), data["debug"]["final_detection_count"])

    def test_bbox_coordinates_remain_inside_image(self):
        """BBox coordinates must remain strictly within original image boundaries."""
        with open(self.test_img_path, "rb") as f:
            resp = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.25"},
                files={"file": ("test.png", f, "image/png")}
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        for d in data["detections"]:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            self.assertGreaterEqual(x1, 0, f"x1 {x1} out of bounds")
            self.assertGreaterEqual(y1, 0, f"y1 {y1} out of bounds")
            self.assertLessEqual(x2, self.orig_w, f"x2 {x2} exceeds width {self.orig_w}")
            self.assertLessEqual(y2, self.orig_h, f"y2 {y2} exceeds height {self.orig_h}")
            self.assertGreater(x2, x1, f"x2 {x2} must be > x1 {x1}")
            self.assertGreater(y2, y1, f"y2 {y2} must be > y1 {y1}")

    def test_tile_offsets_not_added_twice(self):
        """Coordinates must correspond to global image space without double-offset addition."""
        # For a tile at offset (384, 0), remapped box x2 cannot exceed orig_w (1024)
        with open(self.test_img_path, "rb") as f:
            resp = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.25"},
                files={"file": ("test.png", f, "image/png")}
            )
        data = resp.json()
        for d in data["detections"]:
            b = d["bbox_xyxy"]
            self.assertLessEqual(b[2], self.orig_w, "Box x2 exceeded image width: tile offset added twice!")
            self.assertLessEqual(b[3], self.orig_h, "Box y2 exceeded image height: tile offset added twice!")

    def test_confidence_threshold_consistency(self):
        """Higher threshold must produce <= detections than lower threshold."""
        with open(self.test_img_path, "rb") as f:
            resp_low = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.25"},
                files={"file": ("test.png", f, "image/png")}
            )
        with open(self.test_img_path, "rb") as f:
            resp_high = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.45"},
                files={"file": ("test.png", f, "image/png")}
            )

        data_low = resp_low.json()
        data_high = resp_high.json()

        self.assertGreaterEqual(len(data_low["detections"]), len(data_high["detections"]),
                                "Higher threshold should filter out low-confidence clutter")
        self.assertEqual(data_low["debug"]["confidence_threshold"], 0.25)
        self.assertEqual(data_high["debug"]["confidence_threshold"], 0.45)

    def test_tile_grid_disabled_by_default(self):
        """Tile grid overlay must be disabled by default."""
        with open(self.test_img_path, "rb") as f:
            resp = self.client.post(
                "/api/scan",
                data={},
                files={"file": ("test.png", f, "image/png")}
            )
        data = resp.json()
        self.assertFalse(data["debug"]["tile_grid_enabled"], "Tile grid must be OFF by default")


if __name__ == "__main__":
    unittest.main()
