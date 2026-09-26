import unittest
import os
import time
import cv2
from engine.detector import SonarDetector
from engine.preprocessing import PreprocessConfig
from engine.tiling import SonarTiler


class TestDetectorPhase2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join("models", "best_detector.onnx")
        cls.detector = SonarDetector(cls.model_path)
        cls.reports_dir = os.path.join("reports", "phase2")
        os.makedirs(cls.reports_dir, exist_ok=True)

    def test_background_suppression(self):
        """Verify clear background seabed produces 0 false positives."""
        path = os.path.join("data", "samples", "bg_1693569243.750_x2500.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img, f"Missing test image: {path}")

        detections, annotated = self.detector.detect(img, conf_threshold=0.25, tiling=True)
        self.assertEqual(len(detections), 0)

        # Save debug visual
        out_path = os.path.join(self.reports_dir, "test_background_result.jpg")
        SonarDetector.save_annotated_result(img, detections, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_ghost_net_detection(self):
        """Verify ghost net target detection with tiling enabled."""
        path = os.path.join("data", "samples", "synth_ghost_net_00001.png")
        img = cv2.imread(path)
        self.assertIsNotNone(img, f"Missing test image: {path}")

        detections, annotated = self.detector.detect(img, conf_threshold=0.25, tiling=True)
        self.assertGreaterEqual(len(detections), 1)
        classes = [d['class'] for d in detections]
        self.assertIn('ghost_net', classes)

        # Save debug visual
        out_path = os.path.join(self.reports_dir, "test_ghost_net_result.jpg")
        SonarDetector.save_annotated_result(img, detections, out_path)
        self.assertTrue(os.path.exists(out_path))

    def test_large_image_tiling_1024x1024(self):
        """Verify 1024x1024 image is tiled (4 tiles), processed without squashing, and detections mapped."""
        path = os.path.join("data", "samples", "mine_0001_2015.jpg")
        img = cv2.imread(path)
        self.assertIsNotNone(img, f"Missing test image: {path}")
        h, w = img.shape[:2]
        self.assertEqual((w, h), (1024, 1024))

        t0 = time.perf_counter()
        detections, annotated = self.detector.detect(img, conf_threshold=0.30, tiling=True)
        t_total_ms = (time.perf_counter() - t0) * 1000

        self.assertGreaterEqual(len(detections), 1)
        classes = [d['class'] for d in detections]
        self.assertIn('mine_cylinder', classes)

        # Verify all bounding boxes are within original 1024x1024 bounds
        for d in detections:
            x1, y1, x2, y2 = d['bbox_xyxy']
            self.assertTrue(0 <= x1 < 1024)
            self.assertTrue(0 <= y1 < 1024)
            self.assertTrue(0 < x2 <= 1024)
            self.assertTrue(0 < y2 <= 1024)

        # Save debug visual with tile boundaries
        tiler = SonarTiler(tile_size=640, overlap=0.20)
        tiles = tiler.split_into_tiles(img)
        self.assertEqual(len(tiles), 4)

        out_path = os.path.join(self.reports_dir, "test_large_mine_1024_tiled.jpg")
        SonarDetector.save_annotated_result(img, detections, out_path, draw_tiles=True, tiles=tiles)
        self.assertTrue(os.path.exists(out_path))
        print(f"\n[Test Large 1024x1024 Mine] Tiles: {len(tiles)} | Latency: {t_total_ms:.1f}ms | Detections: {len(detections)}")


if __name__ == '__main__':
    unittest.main()
