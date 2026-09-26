import unittest
import numpy as np
import cv2

from engine.physics import SonarPhysicsEngine
from engine.metadata import SurveyMetadata, ProvenanceStatus


class TestPhysicsPhase3(unittest.TestCase):
    """
    Unit tests for rigorous acoustic physics, elevation trigonometry, and depth stratification.
    """

    def setUp(self):
        self.physics = SonarPhysicsEngine()

    def test_rigorous_elevation_calculation_all_geometry_provided(self):
        """
        TEST 4: All required geometric inputs provided.
        H_alt = 15.0m, L_shadow = 5.0m, R_slant = 20.0m
        Expected:
        H = (15.0 * 5.0) / (20.0 + 5.0) = 75.0 / 25.0 = 3.0m
        Status = DERIVED
        """
        res = self.physics.calculate_elevation_rigorous(
            altitude_m=15.0,
            shadow_length_m=5.0,
            slant_range_m=20.0
        )

        self.assertIsNotNone(res["elevation_m"])
        self.assertEqual(res["elevation_m"], 3.0)
        self.assertEqual(res["status"], ProvenanceStatus.DERIVED.value)
        self.assertIn("H = (H_alt * L_shadow) / (R_slant + L_shadow)", res["formula"])
        self.assertEqual(res["inputs"]["altitude_m"], 15.0)
        self.assertEqual(res["inputs"]["shadow_length_m"], 5.0)
        self.assertEqual(res["inputs"]["slant_range_m"], 20.0)

    def test_elevation_missing_altitude_returns_unavailable(self):
        """
        TEST 5: Missing altitude.
        Expected: elevation_m = None, status = UNAVAILABLE. Never fabricates an altitude!
        """
        res = self.physics.calculate_elevation_rigorous(
            altitude_m=None,
            shadow_length_m=5.0,
            slant_range_m=20.0
        )

        self.assertIsNone(res["elevation_m"])
        self.assertEqual(res["status"], ProvenanceStatus.UNAVAILABLE.value)
        self.assertIn("Missing", res["reason"])

    def test_elevation_missing_shadow_length_returns_unavailable(self):
        """
        Missing shadow length: cannot compute elevation.
        Expected: elevation_m = None, status = UNAVAILABLE.
        """
        res = self.physics.calculate_elevation_rigorous(
            altitude_m=12.0,
            shadow_length_m=None,
            slant_range_m=25.0
        )

        self.assertIsNone(res["elevation_m"])
        self.assertEqual(res["status"], ProvenanceStatus.UNAVAILABLE.value)

    def test_elevation_missing_slant_range_returns_unavailable(self):
        """
        Missing slant range: cannot compute elevation.
        Expected: elevation_m = None, status = UNAVAILABLE.
        """
        res = self.physics.calculate_elevation_rigorous(
            altitude_m=12.0,
            shadow_length_m=4.5,
            slant_range_m=None
        )

        self.assertIsNone(res["elevation_m"])
        self.assertEqual(res["status"], ProvenanceStatus.UNAVAILABLE.value)

    def test_depth_stratification_separate_from_vehicle_depth(self):
        """
        Separates vehicle depth, sonar altitude, seabed depth, and target elevation.
        Does NOT confuse vehicle depth with object depth!
        Given:
        - vehicle_depth_m = 25.0
        - sonar_altitude_m = 10.0
        -> seabed_depth_m = 35.0
        - elevation_m = 2.0
        -> target_depth_m = 33.0
        """
        # Create synthetic image with shadow
        img = np.full((300, 500, 3), 120, dtype=np.uint8)
        img[100:140, 260:300] = 230  # Highlight
        img[100:140, 300:340] = 10   # Shadow (40px)

        meta = SurveyMetadata(
            vehicle_depth_m=25.0,
            sonar_altitude_m=10.0,
            slant_range_m=30.0,
            meters_per_pixel=0.05  # 40px * 0.05 = 2.0m shadow
        )

        phys_res = self.physics.analyze_target_physics(
            img_bgr=img,
            bbox_xyxy=[260, 100, 300, 140],
            metadata=meta
        )

        depth = phys_res["depth"]
        elev = phys_res["elevation"]

        self.assertEqual(depth["vehicle_depth_m"], 25.0)
        self.assertEqual(depth["sonar_altitude_m"], 10.0)
        self.assertEqual(depth["seabed_depth_m"], 35.0)
        self.assertIsNotNone(elev["elevation_m"])
        # Target depth = seabed (35.0) - elevation
        expected_target_depth = round(35.0 - elev["elevation_m"], 2)
        self.assertEqual(depth["target_depth_m"], expected_target_depth)
        self.assertNotEqual(depth["target_depth_m"], depth["vehicle_depth_m"])


if __name__ == '__main__':
    unittest.main()
