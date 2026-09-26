"""
tests/test_temporal_tracking.py
Unit tests for Temporal Persistence Tracker data model, single-observation rules,
and missing sensor handling.
"""

import unittest
from engine.temporal_tracking import (
    TemporalPersistenceTracker,
    TrackingConfig,
    PersistenceStatus,
    TemporalTrack
)

class TestTemporalTracking(unittest.TestCase):
    """
    Validates core temporal tracking rules and edge case handling.
    """

    def setUp(self):
        self.tracker = TemporalPersistenceTracker(TrackingConfig(min_observations_for_persistent=3))

    def test_single_observation_is_never_persistent(self):
        """
        SINGLE OBSERVATION RULE: One observation MUST NOT be called PERSISTENT.
        Observation count = 1 must be NEW_CONTACT or UNCERTAIN.
        """
        obs = [{"class": "ghost_net", "confidence": 0.92, "box": {"x": 100, "y": 100, "w": 50, "h": 50}}]
        enriched = self.tracker.process_scan_observations("SURVEY_01", "SCAN_01", obs)

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0]["observation_count"], 1)
        self.assertEqual(enriched[0]["persistence_status"], PersistenceStatus.NEW_CONTACT.value)
        self.assertNotEqual(enriched[0]["persistence_status"], PersistenceStatus.PERSISTENT.value)

    def test_single_observation_low_confidence_is_uncertain(self):
        """Low confidence single observation initializes as UNCERTAIN."""
        obs = [{"class": "ghost_net", "confidence": 0.25, "box": {"x": 100, "y": 100, "w": 50, "h": 50}}]
        enriched = self.tracker.process_scan_observations("SURVEY_01", "SCAN_01", obs)

        self.assertEqual(enriched[0]["persistence_status"], PersistenceStatus.UNCERTAIN.value)

    def test_missing_gps_falls_back_to_image_coordinates(self):
        """When GPS is unavailable, tracker uses bbox/image coordinates and does not invent GPS."""
        obs = [{"class": "shipwreck", "confidence": 0.85, "box": {"x": 200, "y": 200, "w": 100, "h": 60}, "geo": {}}]
        enriched = self.tracker.process_scan_observations("SURVEY_NO_GPS", "SCAN_01", obs)

        track_id = enriched[0]["track_id"]
        track = self.tracker.get_track(track_id)
        self.assertIsNotNone(track)
        self.assertEqual(len(track.bbox_history), 1)
        self.assertEqual(len(track.positions), 0)  # No fake GPS fabricated

    def test_missing_timestamp_uses_valid_iso_default(self):
        """Missing explicit timestamp defaults gracefully without error."""
        obs = [{"class": "mine_cylinder", "confidence": 0.80, "box": {"x": 50, "y": 50, "w": 30, "h": 30}}]
        enriched = self.tracker.process_scan_observations("SURVEY_TIME", "SCAN_01", obs, timestamp_str=None)

        track = self.tracker.get_track(enriched[0]["track_id"])
        self.assertTrue(len(track.first_seen) > 0)
        self.assertIn("T", track.first_seen)  # Standard ISO format

    def test_simultaneous_multiple_contacts_tracked_separately(self):
        """Two distinct contacts in the same scan receive distinct track IDs."""
        obs = [
            {"class": "ghost_net", "confidence": 0.90, "box": {"x": 100, "y": 100, "w": 50, "h": 50}},
            {"class": "submarine_pipeline", "confidence": 0.85, "box": {"x": 400, "y": 400, "w": 150, "h": 30}}
        ]
        enriched = self.tracker.process_scan_observations("SURVEY_MULTI", "SCAN_01", obs)

        self.assertEqual(len(enriched), 2)
        self.assertNotEqual(enriched[0]["track_id"], enriched[1]["track_id"])
        self.assertEqual(len(self.tracker.get_tracks_for_survey("SURVEY_MULTI")), 2)


if __name__ == "__main__":
    unittest.main()
