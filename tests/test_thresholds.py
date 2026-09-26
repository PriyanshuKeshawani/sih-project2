import unittest

from engine.evaluator import ObjectDetectionEvaluator


class TestConfidenceThresholds(unittest.TestCase):
    """
    Unit tests for multi-threshold confidence sweeps and filtering behavior.
    """

    def setUp(self):
        # Synthetic predictions across confidence levels
        self.preds = [[
            {"class": "mine_cylinder", "confidence": 0.85, "bbox_xyxy": [100, 100, 150, 150]},
            {"class": "mine_cylinder", "confidence": 0.55, "bbox_xyxy": [200, 200, 250, 250]},
            {"class": "shipwreck", "confidence": 0.35, "bbox_xyxy": [300, 300, 350, 350]},      # Low-conf clutter
            {"class": "shipwreck", "confidence": 0.20, "bbox_xyxy": [400, 400, 450, 450]}       # Low-conf clutter
        ]]
        self.gts = [[
            {"class": "mine_cylinder", "bbox_xyxy": [100, 100, 150, 150]},
            {"class": "mine_cylinder", "bbox_xyxy": [200, 200, 250, 250]}
        ]]

    def test_threshold_filtering_monotonicity(self):
        """
        Total predictions must monotonically non-increase as confidence threshold increases.
        """
        thresholds = [0.15, 0.25, 0.40, 0.60, 0.90]
        prev_dets = 99999

        for conf in thresholds:
            res = ObjectDetectionEvaluator.evaluate_dataset(
                dataset_predictions=self.preds,
                dataset_ground_truths=self.gts,
                iou_threshold=0.50,
                confidence_threshold=conf
            )
            total_active_preds = res.total_tp + res.total_fp
            self.assertLessEqual(total_active_preds, prev_dets)
            prev_dets = total_active_preds

    def test_low_confidence_false_positives_eliminated_at_higher_threshold(self):
        """
        At conf=0.15: both shipwreck FPs are included (FP = 2).
        At conf=0.40: both shipwreck FPs are filtered out (FP = 0).
        Precision improves from 2/4 (0.50) to 2/2 (1.00).
        """
        res_low = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=self.preds,
            dataset_ground_truths=self.gts,
            iou_threshold=0.50,
            confidence_threshold=0.15
        )
        self.assertEqual(res_low.total_fp, 2)
        self.assertAlmostEqual(res_low.macro_precision, 0.50, places=2)

        res_high = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=self.preds,
            dataset_ground_truths=self.gts,
            iou_threshold=0.50,
            confidence_threshold=0.40
        )
        self.assertEqual(res_high.total_fp, 0)
        self.assertEqual(res_high.total_tp, 2)
        self.assertAlmostEqual(res_high.macro_precision, 1.0, places=2)


if __name__ == '__main__':
    unittest.main()
