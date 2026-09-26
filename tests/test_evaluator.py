import unittest

from engine.evaluator import ObjectDetectionEvaluator, EvaluationSummary
from engine.detector import CLASSES


class TestEvaluator(unittest.TestCase):
    """
    Unit tests for dataset-level metric calculations, empty sets, and class mapping.
    """

    def test_class_mapping_verified(self):
        """
        TEST 10: Verify explicit class mapping between model index and class names.
        0 = crab_pot
        1 = submarine_pipeline
        2 = shipwreck
        3 = ghost_net
        4 = mine_cylinder
        """
        expected_classes = ['crab_pot', 'submarine_pipeline', 'shipwreck', 'ghost_net', 'mine_cylinder']
        self.assertEqual(CLASSES, expected_classes)
        self.assertEqual(CLASSES[0], 'crab_pot')
        self.assertEqual(CLASSES[1], 'submarine_pipeline')
        self.assertEqual(CLASSES[2], 'shipwreck')
        self.assertEqual(CLASSES[3], 'ghost_net')
        self.assertEqual(CLASSES[4], 'mine_cylinder')

    def test_empty_gt_image_clear_background(self):
        """
        TEST 8: Empty GT image (background seabed).
        If detector outputs 0 predictions -> 0 TP, 0 FP, 0 FN. Perfect background rejection!
        If detector outputs 1 false prediction -> 0 TP, 1 FP, 0 FN.
        """
        # Case A: Clean rejection
        res_clean = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=[[]],
            dataset_ground_truths=[[]],
            iou_threshold=0.50
        )
        self.assertEqual(res_clean.total_tp, 0)
        self.assertEqual(res_clean.total_fp, 0)
        self.assertEqual(res_clean.total_fn, 0)

        # Case B: False detection on background
        preds_false = [[{"class": "shipwreck", "confidence": 0.40, "bbox_xyxy": [100, 100, 200, 200]}]]
        res_false = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=preds_false,
            dataset_ground_truths=[[]],
            iou_threshold=0.50
        )
        self.assertEqual(res_false.total_tp, 0)
        self.assertEqual(res_false.total_fp, 1)
        self.assertEqual(res_false.total_fn, 0)
        self.assertEqual(res_false.per_class["shipwreck"].fp, 1)

    def test_empty_prediction_missed_all(self):
        """
        TEST 9: Model predicts nothing on an image that has 2 GT objects.
        Expected: 0 TP, 0 FP, 2 FN, Recall = 0.0.
        """
        gts = [[
            {"class": "ghost_net", "bbox_xyxy": [50, 50, 150, 150]},
            {"class": "ghost_net", "bbox_xyxy": [200, 200, 300, 300]}
        ]]
        preds = [[]]

        res = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=preds,
            dataset_ground_truths=gts,
            iou_threshold=0.50
        )
        self.assertEqual(res.total_tp, 0)
        self.assertEqual(res.total_fp, 0)
        self.assertEqual(res.total_fn, 2)
        self.assertEqual(res.macro_recall, 0.0)

    def test_metric_calculation_accuracy(self):
        """
        Verifies mathematical precision of Precision, Recall, and F1 calculations.
        Scenario: 3 TP, 1 FP, 1 FN
        Precision = 3 / (3 + 1) = 0.75
        Recall = 3 / (3 + 1) = 0.75
        F1 = 0.75
        """
        preds = [[
            {"class": "submarine_pipeline", "confidence": 0.90, "bbox_xyxy": [0, 0, 100, 100]}, # TP
            {"class": "submarine_pipeline", "confidence": 0.85, "bbox_xyxy": [150, 150, 250, 250]}, # TP
            {"class": "submarine_pipeline", "confidence": 0.80, "bbox_xyxy": [300, 300, 400, 400]}, # TP
            {"class": "submarine_pipeline", "confidence": 0.70, "bbox_xyxy": [500, 500, 550, 550]}  # FP
        ]]
        gts = [[
            {"class": "submarine_pipeline", "bbox_xyxy": [0, 0, 100, 100]},
            {"class": "submarine_pipeline", "bbox_xyxy": [150, 150, 250, 250]},
            {"class": "submarine_pipeline", "bbox_xyxy": [300, 300, 400, 400]},
            {"class": "submarine_pipeline", "bbox_xyxy": [600, 600, 700, 700]} # FN
        ]]

        res = ObjectDetectionEvaluator.evaluate_dataset(
            dataset_predictions=preds,
            dataset_ground_truths=gts,
            iou_threshold=0.50
        )

        pipe_metric = res.per_class["submarine_pipeline"]
        self.assertEqual(pipe_metric.tp, 3)
        self.assertEqual(pipe_metric.fp, 1)
        self.assertEqual(pipe_metric.fn, 1)
        self.assertAlmostEqual(pipe_metric.precision, 0.75, places=4)
        self.assertAlmostEqual(pipe_metric.recall, 0.75, places=4)
        self.assertAlmostEqual(pipe_metric.f1, 0.75, places=4)


if __name__ == '__main__':
    unittest.main()
