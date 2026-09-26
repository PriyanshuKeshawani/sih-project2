import unittest
import os
import cv2

from engine.detector import SonarDetector
from engine.evaluator import ObjectDetectionEvaluator


class TestValidationPipeline(unittest.TestCase):
    """
    Unit tests integrating the SonarDetector with the ObjectDetectionEvaluator.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_detector.onnx")
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"Model file not found: {cls.model_path}")
        cls.detector = SonarDetector(cls.model_path)

        cls.sample_bg = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "bg_1693569243.750_x2500.jpg")
        cls.sample_net = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "synth_ghost_net_00001.png")

    def test_background_image_evaluation_zero_fps(self):
        """
        Background image evaluation: GT = []
        Detector should output 0 detections, resulting in 0 TP, 0 FP, 0 FN.
        """
        img = cv2.imread(self.sample_bg)
        self.assertIsNotNone(img)

        dets, _ = self.detector.detect(img, conf_threshold=0.25, tiling=True)
        res = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=[dets],
            dataset_ground_truths=[[]],
            iou_threshold=0.50,
            confidence_threshold=0.25
        )

        self.assertEqual(res.total_tp, 0)
        self.assertEqual(res.total_fp, 0)
        self.assertEqual(res.total_fn, 0)

    def test_ghost_net_image_evaluation_matches_gt(self):
        """
        Ghost net test image evaluation:
        Ground truth box [114, 210, 229, 319] of class ghost_net.
        Detector should find a matching box with IoU >= 0.50 -> 1 TP, 0 FP, 0 FN.
        """
        img = cv2.imread(self.sample_net)
        self.assertIsNotNone(img)

        gt_box = [{"class": "ghost_net", "bbox_xyxy": [114, 210, 229, 319]}]
        dets, _ = self.detector.detect(img, conf_threshold=0.25, tiling=True)

        res = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=[dets],
            dataset_ground_truths=[gt_box],
            iou_threshold=0.50,
            confidence_threshold=0.25
        )

        self.assertGreaterEqual(res.total_tp, 1)
        ghost_metric = res.per_class.get("ghost_net")
        self.assertIsNotNone(ghost_metric)
        self.assertEqual(ghost_metric.tp, 1)
        self.assertEqual(ghost_metric.fn, 0)


if __name__ == '__main__':
    unittest.main()
