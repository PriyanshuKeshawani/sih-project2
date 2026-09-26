import unittest
from engine.tiling import SonarTiler


class TestClassAwareNMS(unittest.TestCase):

    def test_same_class_overlapping_suppression(self):
        """TEST 5: Two highly overlapping detections of SAME class are suppressed, keeping the highest score."""
        boxes = [
            [100, 100, 200, 200],  # Box A
            [105, 102, 203, 198]   # Box B (IoU > 0.85)
        ]
        scores = [0.92, 0.85]
        class_ids = [3, 3]  # Both ghost_net

        keep_idx = SonarTiler.class_aware_nms(boxes, scores, class_ids, iou_threshold=0.45)
        self.assertEqual(len(keep_idx), 1)
        self.assertEqual(keep_idx[0], 0)  # Box A preserved because score 0.92 > 0.85

    def test_different_classes_overlapping_preservation(self):
        """TEST 6: Two overlapping detections of DIFFERENT classes are NOT suppressed."""
        boxes = [
            [100, 100, 200, 200],  # Box A (ghost_net)
            [105, 102, 203, 198]   # Box B (submarine_pipeline)
        ]
        scores = [0.92, 0.85]
        class_ids = [3, 1]  # Class 3 vs Class 1

        keep_idx = SonarTiler.class_aware_nms(boxes, scores, class_ids, iou_threshold=0.45)
        self.assertEqual(len(keep_idx), 2)
        self.assertIn(0, keep_idx)
        self.assertIn(1, keep_idx)

    def test_empty_boxes(self):
        """Verify NMS safely handles empty detection lists without crashing."""
        keep_idx = SonarTiler.class_aware_nms([], [], [], iou_threshold=0.45)
        self.assertEqual(keep_idx, [])


if __name__ == '__main__':
    unittest.main()
