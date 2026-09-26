import unittest

from engine.physics import SonarPhysicsEngine
from engine.metadata import SurveyMetadata, ProvenanceStatus, GeoResult


class TestGeoreferencing(unittest.TestCase):
    """
    Unit tests for geospatial coordinate handling, platform vs object position distinction,
    and demo vs real provenance.
    """

    def setUp(self):
        self.physics = SonarPhysicsEngine()

    def test_demo_coordinates_explicitly_flagged(self):
        """
        TEST 6: Demo coordinates.
        Expected: status = DEMO, source mentions demo/hardcoded. Never disguised as LIVE GPS!
        """
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"APP_MODE": "test"}):
            geo: GeoResult = self.physics.georeference_target(
                bbox_xyxy=[100, 100, 150, 150],
                img_w=640,
                img_h=640,
                metadata=None,
                use_demo_fallback=True
            )

            self.assertEqual(geo.status, ProvenanceStatus.DEMO.value)
            self.assertIn("demo", geo.source.lower())
            self.assertIn("DEMO", geo.zone_label)
            self.assertAlmostEqual(geo.latitude, 9.2882, delta=0.01)
            self.assertAlmostEqual(geo.longitude, 79.1325, delta=0.01)

    def test_no_coordinates_returns_null(self):
        """
        TEST 7: No coordinates provided and demo fallback disabled.
        Expected: latitude = None, longitude = None, status = UNAVAILABLE.
        """
        geo: GeoResult = self.physics.georeference_target(
            bbox_xyxy=[100, 100, 150, 150],
            img_w=640,
            img_h=640,
            metadata=None,
            use_demo_fallback=False
        )

        self.assertIsNone(geo.latitude)
        self.assertIsNone(geo.longitude)
        self.assertEqual(geo.status, ProvenanceStatus.UNAVAILABLE.value)
        self.assertEqual(geo.position_type, "UNAVAILABLE")

    def test_platform_position_distinguished_from_object_position(self):
        """
        TEST 8: Vehicle GPS is available, but meters_per_pixel is NOT available.
        Cannot calculate target cross-track offset accurately.
        Expected:
        - Output coordinates are the vehicle's position.
        - position_type = "PLATFORM_POSITION", NOT "OBJECT_POSITION".
        """
        meta = SurveyMetadata(
            platform_lat=13.0827,
            platform_lon=80.2707,
            meters_per_pixel=None,  # No scale calibration
            source_type="REAL_SENSOR"
        )

        geo: GeoResult = self.physics.georeference_target(
            bbox_xyxy=[100, 100, 150, 150],
            img_w=640,
            img_h=640,
            metadata=meta,
            use_demo_fallback=False
        )

        self.assertEqual(geo.latitude, 13.0827)
        self.assertEqual(geo.longitude, 80.2707)
        self.assertEqual(geo.position_type, "PLATFORM_POSITION")
        self.assertNotEqual(geo.position_type, "OBJECT_POSITION")

    def test_object_position_derived_when_scale_and_heading_available(self):
        """
        When platform GPS, heading, and meters_per_pixel are all available,
        the exact target seafloor location is derived and labeled OBJECT_POSITION.
        """
        meta = SurveyMetadata(
            platform_lat=13.0827,
            platform_lon=80.2707,
            heading_deg=0.0,        # North
            meters_per_pixel=0.10,  # 10 cm / pixel
            source_type="REAL_SENSOR"
        )

        # Starboard object: center is at x=420 in 640px wide image (cross-track = +100px = +10m East)
        geo: GeoResult = self.physics.georeference_target(
            bbox_xyxy=[400, 200, 440, 240],
            img_w=640,
            img_h=640,
            metadata=meta,
            use_demo_fallback=False
        )

        self.assertEqual(geo.position_type, "OBJECT_POSITION")
        self.assertEqual(geo.status, ProvenanceStatus.DERIVED.value)
        self.assertIsNotNone(geo.latitude)
        self.assertIsNotNone(geo.longitude)
        # Because offset is to the right (East), longitude should shift slightly positive
        self.assertGreater(geo.longitude, 80.2707)


if __name__ == '__main__':
    unittest.main()
