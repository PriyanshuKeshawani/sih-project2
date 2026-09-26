import unittest
import numpy as np
import cv2

from engine.shadow_analysis import AcousticShadowAnalyzer, ShadowConfig, ShadowAnalysis


class TestAcousticShadowAnalysis(unittest.TestCase):
    """
    Unit tests for Acoustic Shadow Intensity Segmentation.
    Verifies image-evidence based acoustic dropout detection down-range of highlights.
    """

    def setUp(self):
        self.analyzer = AcousticShadowAnalyzer(
            config=ShadowConfig(
                search_distance_px=150,
                smoothing_window=3,
                relative_drop_factor=0.50,
                absolute_shadow_max=55.0,
                minimum_shadow_run=4
            )
        )

    def _create_synthetic_sonar_image(
        self,
        width: int = 600,
        height: int = 400,
        seabed_intensity: int = 120,
        target_box: tuple = (350, 180, 390, 220), # Starboard side
        target_intensity: int = 230,
        shadow_extent_px: int = 40,
        shadow_intensity: int = 15
    ) -> np.ndarray:
        """
        Creates synthetic side-scan sonar image with seabed reverberation,
        a bright highlight target, and a dark down-range acoustic dropout (shadow).
        """
        img = np.full((height, width), seabed_intensity, dtype=np.uint8)

        # Draw bright acoustic highlight
        x1, y1, x2, y2 = target_box
        img[y1:y2, x1:x2] = target_intensity

        # Draw down-range acoustic shadow (starboard side casts to +X)
        if shadow_extent_px > 0:
            shadow_x1 = x2
            shadow_x2 = min(width, x2 + shadow_extent_px)
            img[y1:y2, shadow_x1:shadow_x2] = shadow_intensity

        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def test_synthetic_target_with_shadow_detected(self):
        """
        TEST 1: Synthetic image with an obvious bright object followed by a dark region.
        Expected: shadow analyzer identifies the shadow region.
        """
        target_box = (350, 180, 390, 220)
        img = self._create_synthetic_sonar_image(
            target_box=target_box,
            shadow_extent_px=45,
            shadow_intensity=10
        )

        res: ShadowAnalysis = self.analyzer.analyze(
            img_bgr=img,
            bbox_xyxy=list(target_box)
        )

        self.assertTrue(res.detected, "Shadow should be detected behind synthetic target")
        self.assertIsNotNone(res.shadow_length_px)
        self.assertGreaterEqual(res.shadow_length_px, 35)
        self.assertLessEqual(res.shadow_length_px, 50)
        self.assertEqual(res.search_direction, "DOWN_RANGE_RIGHT")
        self.assertIsNotNone(res.start_point)
        self.assertIsNotNone(res.end_point)
        self.assertEqual(res.start_point[0], target_box[2])
        self.assertEqual(res.end_point[0], target_box[2] + int(res.shadow_length_px))

    def test_target_without_shadow_not_detected(self):
        """
        TEST 2: Detection with no detectable shadow (uniform seafloor behind object).
        Expected: shadow detection = false, shadow_length_px = None, status = UNAVAILABLE.
        """
        target_box = (350, 180, 390, 220)
        # 0 shadow extent: normal seabed continues immediately behind object
        img = self._create_synthetic_sonar_image(
            target_box=target_box,
            shadow_extent_px=0
        )

        res: ShadowAnalysis = self.analyzer.analyze(
            img_bgr=img,
            bbox_xyxy=list(target_box)
        )

        self.assertFalse(res.detected, "No shadow should be detected when seabed intensity is uniform")
        self.assertIsNone(res.shadow_length_px)
        self.assertIsNone(res.shadow_length_m)
        self.assertEqual(res.status, "UNAVAILABLE")

    def test_shadow_detected_without_pixel_calibration(self):
        """
        TEST 3: Shadow detected but no pixel-to-meter calibration provided.
        Expected:
        - shadow_length_px is MEASURED
        - shadow_length_m is null (UNAVAILABLE)
        - never assumes 1 pixel = 1 meter!
        """
        target_box = (350, 180, 390, 220)
        img = self._create_synthetic_sonar_image(
            target_box=target_box,
            shadow_extent_px=30,
            shadow_intensity=8
        )

        res: ShadowAnalysis = self.analyzer.analyze(
            img_bgr=img,
            bbox_xyxy=list(target_box),
            meters_per_pixel=None  # No calibration available
        )

        self.assertTrue(res.detected)
        self.assertIsNotNone(res.shadow_length_px)
        self.assertIsNone(res.shadow_length_m, "Metric shadow length MUST be None when calibration is missing")
        self.assertEqual(res.status, "MEASURED")

    def test_shadow_detected_with_pixel_calibration(self):
        """
        Validates that when meters_per_pixel is provided, metric shadow length is derived.
        """
        target_box = (350, 180, 390, 220)
        img = self._create_synthetic_sonar_image(
            target_box=target_box,
            shadow_extent_px=40,
            shadow_intensity=8
        )

        m_per_px = 0.05  # 5 cm per pixel
        res: ShadowAnalysis = self.analyzer.analyze(
            img_bgr=img,
            bbox_xyxy=list(target_box),
            meters_per_pixel=m_per_px
        )

        self.assertTrue(res.detected)
        self.assertIsNotNone(res.shadow_length_px)
        self.assertIsNotNone(res.shadow_length_m)
        expected_m = round(res.shadow_length_px * m_per_px, 3)
        self.assertEqual(res.shadow_length_m, expected_m)
        self.assertEqual(res.status, "DERIVED")

    def test_port_side_shadow_direction(self):
        """
        Port side targets (X < nadir_x) should cast shadows towards the left (-X).
        """
        width = 800
        nadir_x = 400
        # Port target at x=[150..190]
        target_box = (150, 150, 190, 190)
        img = np.full((400, width), 130, dtype=np.uint8)
        img[150:190, 150:190] = 240  # Target highlight
        img[150:190, 110:150] = 10   # Port shadow (to the left of target)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        res: ShadowAnalysis = self.analyzer.analyze(
            img_bgr=img_bgr,
            bbox_xyxy=list(target_box)
        )

        self.assertTrue(res.detected)
        self.assertEqual(res.search_direction, "DOWN_RANGE_LEFT")
        self.assertIsNotNone(res.shadow_length_px)
        self.assertGreaterEqual(res.shadow_length_px, 35)
        self.assertEqual(res.end_point[0], target_box[0] - int(res.shadow_length_px))


if __name__ == '__main__':
    unittest.main()
