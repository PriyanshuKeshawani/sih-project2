import unittest

from engine.metadata import (
    ProvenanceStatus,
    PhysicsMeasurement,
    SonarGeometry,
    SurveyMetadata,
    GeoResult,
    UserMetadata,
    RecordedNavigation,
    NMEAAdapter,
    SimulatedNavigation
)


class TestMetadataAndProvenance(unittest.TestCase):
    """
    Unit tests for SurveyMetadata, Provenance statuses, and navigation adapters.
    """

    def test_empty_metadata_has_nulls_no_fabricated_defaults(self):
        """
        Missing fields must remain None, never populated with magic numbers.
        """
        meta = SurveyMetadata()
        self.assertIsNone(meta.survey_id)
        self.assertIsNone(meta.platform_lat)
        self.assertIsNone(meta.platform_lon)
        self.assertIsNone(meta.sonar_altitude_m)
        self.assertIsNone(meta.slant_range_m)
        self.assertIsNone(meta.meters_per_pixel)
        self.assertEqual(meta.source_type, "UNKNOWN")

    def test_provenance_enum_values(self):
        """
        TEST 9: Provenance values must strictly adhere to the required scientific states:
        MEASURED, DERIVED, ASSUMED, SIMULATED, DEMO, UNAVAILABLE.
        """
        valid_statuses = {"MEASURED", "DERIVED", "ASSUMED", "SIMULATED", "DEMO", "UNAVAILABLE"}
        for status in valid_statuses:
            self.assertEqual(ProvenanceStatus(status).value, status)

        # Test PhysicsMeasurement with explicit provenance
        meas = PhysicsMeasurement(
            value=3.25,
            unit="m",
            status=ProvenanceStatus.MEASURED.value,
            method="1d_downrange_intensity_profiling"
        )
        self.assertEqual(meas.status, "MEASURED")
        self.assertEqual(meas.value, 3.25)

    def test_user_metadata_adapter(self):
        """
        UserMetadata properly maps raw dictionary to SurveyMetadata.
        """
        payload = {
            "survey_id": "EXPEDITION_2026_01",
            "latitude": 12.834,
            "longitude": 80.231,
            "sonar_altitude_m": 14.5,
            "meters_per_pixel": 0.08
        }
        adapter = UserMetadata(payload)
        meta = adapter.get_metadata()

        self.assertEqual(meta.survey_id, "EXPEDITION_2026_01")
        self.assertEqual(meta.platform_lat, 12.834)
        self.assertEqual(meta.platform_lon, 80.231)
        self.assertEqual(meta.sonar_altitude_m, 14.5)
        self.assertEqual(meta.meters_per_pixel, 0.08)
        self.assertEqual(meta.source_type, "USER_PROVIDED")

    def test_recorded_navigation_adapter(self):
        """
        RecordedNavigation reads telemetry log structure.
        """
        record = {
            "survey_id": "NIOT_SAGAR_NIDHI_LOG_42",
            "lat": 10.45,
            "lon": 79.92,
            "depth_m": 45.0,
            "alt_m": 8.0,
            "m_per_px": 0.04
        }
        adapter = RecordedNavigation(record)
        meta = adapter.get_metadata()

        self.assertEqual(meta.survey_id, "NIOT_SAGAR_NIDHI_LOG_42")
        self.assertEqual(meta.platform_lat, 10.45)
        self.assertEqual(meta.sonar_altitude_m, 8.0)
        self.assertEqual(meta.source_type, "RECORDED_SURVEY")

    def test_nmea_adapter_parsing(self):
        """
        NMEAAdapter parses standard GPGGA sentence.
        Example: $GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47
        Lat: 48 deg 07.038 min N = 48 + 7.038/60 = 48.1173
        Lon: 11 deg 31.000 min E = 11 + 31.000/60 = 11.516667
        """
        sentences = [
            "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A",
            "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
        ]
        adapter = NMEAAdapter(sentences)
        meta = adapter.get_metadata()

        self.assertIsNotNone(meta.platform_lat)
        self.assertIsNotNone(meta.platform_lon)
        self.assertAlmostEqual(meta.platform_lat, 48.1173, places=3)
        self.assertAlmostEqual(meta.platform_lon, 11.516667, places=3)
        self.assertEqual(meta.source_type, "REAL_SENSOR")

    def test_simulated_navigation_adapter_flagged(self):
        """
        Simulated navigation is clearly tagged as SIMULATED, never disguised as real.
        """
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"APP_MODE": "test"}):
            adapter = SimulatedNavigation()
            meta = adapter.get_metadata()

            self.assertEqual(meta.source_type, "SIMULATED")
            self.assertIsNotNone(meta.platform_lat)
            self.assertIsNotNone(meta.platform_lon)


if __name__ == '__main__':
    unittest.main()
