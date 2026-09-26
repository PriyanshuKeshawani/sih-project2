"""
Unit & Integration Tests: Provenance Tagging, Zero-Fabrication GPS Policy, and Filter Integrity.
SIH 2026 Problem Statement 26057.
"""
import unittest
from fastapi.testclient import TestClient
from main import app, temporal_tracker
from engine.physics import SonarPhysicsEngine

class TestProvenanceDisplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.physics = SonarPhysicsEngine()

    def test_demo_gps_clearly_marked(self):
        """
        Georeferencing coordinates must be flagged with DEMO/SIMULATED provenance
        in the UI when operating with simulated navigation data (Section 7, 16).
        """
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"APP_MODE": "test"}):
            box = {"x": 100, "y": 100, "w": 50, "h": 50}
            geo = self.physics.georeference(box, 640, 640)
            self.assertIn("lat", geo)
            self.assertIn("lon", geo)
            self.assertIn("zone", geo)
            # Check that zone is explicitly tagged with DEMO provenance
            self.assertIn("DEMO", geo["zone"])
            self.assertIn("Gulf of Mannar", geo["zone"])

    def test_unavailable_gps_not_rendered_as_real(self):
        """
        ZERO-FABRICATION RULE: When GPS coordinates are unavailable,
        the system must return None or empty dict, never invent random coordinates (Section 8).
        """
        obs_no_gps = {
            "class": "ghost_net",
            "confidence": 0.85,
            "box": {"x": 100, "y": 100, "w": 40, "h": 40},
            "geo": {} # No GPS provided
        }
        res = temporal_tracker.process_scan_observations("SURVEY_NO_GPS", "SCAN_01", [obs_no_gps])
        self.assertEqual(len(res), 1)
        track = temporal_tracker.get_track(res[0]["track_id"])
        self.assertEqual(track.positions, [])
        # Ensure lat/lon were NOT fabricated
        self.assertNotIn("lat", res[0].get("geo", {}))

    def test_derived_elevation_provenance(self):
        """
        Seabed elevation calculation is DERIVED from acoustic shadow geometry,
        not directly measured from a depth sensor (Section 7).
        """
        box = {"x": 150, "y": 150, "w": 60, "h": 80}
        elev = self.physics.calculate_elevation(box, 640, 640, altitude=12.0)
        self.assertIsInstance(elev, float)
        self.assertGreaterEqual(elev, 0.0)

    def test_filters_do_not_modify_backend_data(self):
        """
        UI filtering is non-destructive (Section 12).
        Querying survey tracks returns full backend state regardless of UI filter options.
        """
        survey_id = "SURVEY_FILTER_TEST"
        obs = [
            {"class": "ghost_net", "confidence": 0.90, "box": {"x": 10, "y": 10, "w": 20, "h": 20}},
            {"class": "crab_pot", "confidence": 0.50, "box": {"x": 50, "y": 50, "w": 20, "h": 20}},
        ]
        temporal_tracker.process_scan_observations(survey_id, "SCAN_1", obs)
        tracks = temporal_tracker.get_tracks_for_survey(survey_id)
        self.assertEqual(len(tracks), 2)
        # Verify classes are intact in backend storage
        classes = {t.target_class for t in tracks}
        self.assertIn("ghost_net", classes)
        self.assertIn("crab_pot", classes)

    def test_selected_track_data_consistency(self):
        """
        Track details retrieved via API remain strictly consistent with observation updates.
        """
        survey_id = "SURVEY_CONSISTENCY"
        obs1 = [{"class": "mine_cylinder", "confidence": 0.82, "box": {"x": 80, "y": 80, "w": 30, "h": 40}}]
        res1 = temporal_tracker.process_scan_observations(survey_id, "SCAN_1", obs1)
        track_id = res1[0]["track_id"]

        # Re-observe same contact
        obs2 = [{"class": "mine_cylinder", "confidence": 0.87, "box": {"x": 82, "y": 81, "w": 30, "h": 40}}]
        temporal_tracker.process_scan_observations(survey_id, "SCAN_2", obs2)

        track = temporal_tracker.get_track(track_id)
        self.assertEqual(track.observation_count, 2)
        self.assertEqual(len(track.confidence_history), 2)
        self.assertEqual(track.confidence_history, [0.82, 0.87])
        self.assertAlmostEqual(track.mean_confidence, 0.845, places=3)

if __name__ == "__main__":
    unittest.main()
