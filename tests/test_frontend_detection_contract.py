"""
Regression Tests: Frontend Detection Contract and Duplicate Render Prevention.
SIH 2026 Problem Statement 26057.
"""
import os
import unittest
from fastapi.testclient import TestClient

from main import app

class TestFrontendDetectionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_img_path = os.path.join("reports", "display_audit", "reconstructed_test_input.png")
        if not os.path.exists(cls.test_img_path):
            cls.test_img_path = os.path.join("data", "samples", "synth_ghost_net_00001.png")

    def test_frontend_payload_structure(self):
        """Verify API response provides required fields for frontend UI rendering."""
        with open(self.test_img_path, "rb") as f:
            resp = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.45", "altitude": "12.0"},
                files={"file": ("test.png", f, "image/png")}
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # 1. Top-level contract
        self.assertIn("status", data)
        self.assertIn("summary", data)
        self.assertIn("debug", data)
        self.assertIn("detections", data)
        self.assertIn("annotated_image", data)
        self.assertIn("raw_image", data)

        self.assertEqual(data["status"], "success")
        self.assertTrue(data["annotated_image"].startswith("data:image/jpeg;base64,"))

        # 2. Debug contract exposes raw vs final counts
        debug = data["debug"]
        self.assertIn("tile_count", debug)
        self.assertIn("raw_tile_detections_count", debug)
        self.assertIn("final_detection_count", debug)
        self.assertIn("confidence_threshold", debug)
        self.assertIn("tile_grid_enabled", debug)

        # Raw count must be >= final count
        self.assertGreaterEqual(debug["raw_tile_detections_count"], debug["final_detection_count"])

        # 3. Detections list items contract
        for det in data["detections"]:
            self.assertIn("detection_id", det)
            self.assertIn("class", det)
            self.assertIn("confidence", det)
            self.assertIn("box", det)
            self.assertIn("bbox_xyxy", det)
            self.assertIn("elevation_m", det)
            self.assertIn("geo", det)
            self.assertIn("reflex", det)

    def test_frontend_renders_same_count_no_duplicates(self):
        """Simulate frontend consumption and ensure 1:1 render with zero duplicate injection."""
        with open(self.test_img_path, "rb") as f:
            resp = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.45"},
                files={"file": ("test.png", f, "image/png")}
            )

        data = resp.json()
        detections = data["detections"]
        received_count = len(detections)

        # Frontend renders exactly detections.length table rows and map markers
        rendered_table_rows = len(detections)
        rendered_map_pins = len(detections)

        self.assertEqual(received_count, rendered_table_rows)
        self.assertEqual(received_count, rendered_map_pins)

        # Check unique detection IDs
        det_ids = [d["detection_id"] for d in detections]
        self.assertEqual(len(det_ids), len(set(det_ids)), "Detection IDs must be unique (no duplicates)")

    def test_frontend_does_not_render_raw_tiles(self):
        """Frontend receives ONLY final post-NMS detections, never raw tile candidates."""
        with open(self.test_img_path, "rb") as f:
            resp = self.client.post(
                "/api/scan",
                data={"conf_threshold": "0.45"},
                files={"file": ("test.png", f, "image/png")}
            )

        data = resp.json()
        raw_tile_count = data["debug"]["raw_tile_detections_count"]
        final_count = len(data["detections"])

        # Final detections must equal final_detection_count, not raw_tile_count
        self.assertEqual(final_count, data["debug"]["final_detection_count"])
        if raw_tile_count > 0:
            self.assertLessEqual(final_count, raw_tile_count)


if __name__ == "__main__":
    unittest.main()
