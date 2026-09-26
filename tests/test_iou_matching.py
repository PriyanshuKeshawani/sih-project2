import unittest

from engine.evaluator import ObjectDetectionEvaluator, MatchItem


class TestIoUMatching(unittest.TestCase):
    """
    Unit tests for IoU computation and greedy prediction-to-ground-truth matching.
    """

    def test_perfect_prediction_matching(self):
        """
        TEST 1: Perfect prediction matching identical GT box.
        Expected: IoU = 1.0, status = TP.
        """
        preds = [{"class": "ghost_net", "confidence": 0.95, "bbox_xyxy": [100, 100, 200, 200]}]
        gts = [{"class": "ghost_net", "bbox_xyxy": [100, 100, 200, 200]}]

        matches = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.50)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "TP")
        self.assertAlmostEqual(matches[0].iou, 1.0, places=3)
        self.assertEqual(matches[0].class_name, "ghost_net")

    def test_wrong_class_not_matched(self):
        """
        TEST 2: Perfectly overlapping box but DIFFERENT class name.
        Expected: Prediction is FP, GT is FN. Class consistency enforced!
        """
        preds = [{"class": "shipwreck", "confidence": 0.80, "bbox_xyxy": [100, 100, 200, 200]}]
        gts = [{"class": "ghost_net", "bbox_xyxy": [100, 100, 200, 200]}]

        matches = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.50)
        self.assertEqual(len(matches), 2)
        statuses = {m.status for m in matches}
        self.assertIn("FP", statuses)
        self.assertIn("FN", statuses)

    def test_unmatched_prediction_counts_as_fp(self):
        """
        TEST 3: Prediction with no corresponding GT object.
        Expected: status = FP, iou = 0.0.
        """
        preds = [{"class": "mine_cylinder", "confidence": 0.70, "bbox_xyxy": [300, 300, 350, 350]}]
        gts = []

        matches = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.50)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "FP")
        self.assertEqual(matches[0].iou, 0.0)

    def test_missed_gt_counts_as_fn(self):
        """
        TEST 4: Ground truth object with no predictions.
        Expected: status = FN.
        """
        preds = []
        gts = [{"class": "submarine_pipeline", "bbox_xyxy": [50, 100, 400, 150]}]

        matches = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.50)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].status, "FN")

    def test_duplicate_prediction_single_gt_matched_once(self):
        """
        TEST 5: Two predictions overlapping a single GT object.
        Expected: Highest confidence prediction gets TP; second prediction is FP.
        Prevents one GT object from being counted multiple times!
        """
        preds = [
            {"class": "ghost_net", "confidence": 0.90, "bbox_xyxy": [100, 100, 200, 200]},
            {"class": "ghost_net", "confidence": 0.75, "bbox_xyxy": [105, 105, 195, 195]}
        ]
        gts = [{"class": "ghost_net", "bbox_xyxy": [100, 100, 200, 200]}]

        matches = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.50)
        self.assertEqual(len(matches), 2)
        tp_matches = [m for m in matches if m.status == "TP"]
        fp_matches = [m for m in matches if m.status == "FP"]

        self.assertEqual(len(tp_matches), 1)
        self.assertEqual(len(fp_matches), 1)
        self.assertEqual(tp_matches[0].confidence, 0.90)
        self.assertEqual(fp_matches[0].confidence, 0.75)

    def test_multiple_gt_objects_matched_one_to_one(self):
        """
        TEST 6: Multiple GT objects in image matched independently.
        """
        preds = [
            {"class": "mine_cylinder", "confidence": 0.85, "bbox_xyxy": [50, 50, 80, 80]},
            {"class": "mine_cylinder", "confidence": 0.80, "bbox_xyxy": [200, 200, 240, 240]}
        ]
        gts = [
            {"class": "mine_cylinder", "bbox_xyxy": [50, 50, 80, 80]},
            {"class": "mine_cylinder", "bbox_xyxy": [200, 200, 240, 240]}
        ]

        matches = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.50)
        tp_matches = [m for m in matches if m.status == "TP"]
        self.assertEqual(len(tp_matches), 2)

    def test_iou_thresholds_scaling(self):
        """
        TEST 7: Prediction with partial overlap (~0.35 IoU).
        Expected:
        - At IoU threshold 0.25: TP (0.35 >= 0.25)
        - At IoU threshold 0.50: FP (0.35 < 0.50)
        - At IoU threshold 0.75: FP (0.35 < 0.75)
        """
        boxA = [100.0, 100.0, 200.0, 200.0]  # Area = 10000
        boxB = [140.0, 100.0, 240.0, 200.0]  # Inter = 60*100 = 6000, Union = 20000 - 6000 = 14000. IoU = 6000/14000 = 0.4285
        actual_iou = ObjectDetectionEvaluator.calculate_iou(boxA, boxB)
        self.assertAlmostEqual(actual_iou, 0.4285, places=3)

        preds = [{"class": "shipwreck", "confidence": 0.85, "bbox_xyxy": boxA}]
        gts = [{"class": "shipwreck", "bbox_xyxy": boxB}]

        # At 0.25 -> TP
        m_25 = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.25)
        self.assertEqual(m_25[0].status, "TP")

        # At 0.50 -> FP & FN
        m_50 = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.50)
        statuses_50 = {m.status for m in m_50}
        self.assertEqual(statuses_50, {"FP", "FN"})

        # At 0.75 -> FP & FN
        m_75 = ObjectDetectionEvaluator.match_image_detections(preds, gts, iou_threshold=0.75)
        statuses_75 = {m.status for m in m_75}
        self.assertEqual(statuses_75, {"FP", "FN"})


if __name__ == '__main__':
    unittest.main()
