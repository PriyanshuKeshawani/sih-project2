"""
tests/test_track_matching.py
Unit tests for spatial, temporal, and class-aware track association logic.
"""

import unittest
from engine.temporal_tracking import TemporalPersistenceTracker, TrackingConfig, PersistenceStatus

class TestTrackMatching(unittest.TestCase):
    """
    Validates association heuristics: class compatibility, metric distance,
    pixel distance, time windows, and repeatability.
    """

    def setUp(self):
        self.config = TrackingConfig(
            max_position_distance_m=15.0,
            max_image_distance_px=100.0,
            min_iou=0.15,
            max_time_gap_s=30.0,
            min_observations_for_persistent=3
        )
        self.tracker = TemporalPersistenceTracker(self.config)

    def test_repeated_same_contact_associates_to_same_track(self):
        """Sequential observations of the same contact associate to the same track_id."""
        obs1 = [{"class": "ghost_net", "confidence": 0.88, "box": {"x": 100, "y": 100, "w": 60, "h": 50}}]
        obs2 = [{"class": "ghost_net", "confidence": 0.91, "box": {"x": 105, "y": 102, "w": 58, "h": 52}}]

        enr1 = self.tracker.process_scan_observations("SURVEY_01", "SCAN_01", obs1, timestamp_epoch=100.0)
        enr2 = self.tracker.process_scan_observations("SURVEY_01", "SCAN_02", obs2, timestamp_epoch=105.0)

        self.assertEqual(enr1[0]["track_id"], enr2[0]["track_id"])
        self.assertEqual(enr2[0]["observation_count"], 2)

    def test_different_class_does_not_associate(self):
        """Co-located detections with different classes must create separate tracks."""
        obs1 = [{"class": "ghost_net", "confidence": 0.88, "box": {"x": 100, "y": 100, "w": 60, "h": 50}}]
        obs2 = [{"class": "shipwreck", "confidence": 0.85, "box": {"x": 100, "y": 100, "w": 60, "h": 50}}]

        enr1 = self.tracker.process_scan_observations("SURVEY_01", "SCAN_01", obs1, timestamp_epoch=100.0)
        enr2 = self.tracker.process_scan_observations("SURVEY_01", "SCAN_02", obs2, timestamp_epoch=105.0)

        self.assertNotEqual(enr1[0]["track_id"], enr2[0]["track_id"])

    def test_spatial_distance_exceeded_creates_new_track(self):
        """Contact appearing far outside the spatial threshold must not match."""
        # 1. GPS distance test (> 15m)
        obs1 = [{"class": "mine_cylinder", "confidence": 0.85, "box": {"x": 100, "y": 100, "w": 30, "h": 30}, "geo": {"lat": 9.28820, "lon": 79.13250}}]
        obs2 = [{"class": "mine_cylinder", "confidence": 0.85, "box": {"x": 100, "y": 100, "w": 30, "h": 30}, "geo": {"lat": 9.28950, "lon": 79.13250}}] # ~144 meters away

        enr1 = self.tracker.process_scan_observations("SURVEY_GPS", "SCAN_01", obs1, timestamp_epoch=100.0)
        enr2 = self.tracker.process_scan_observations("SURVEY_GPS", "SCAN_02", obs2, timestamp_epoch=105.0)

        self.assertNotEqual(enr1[0]["track_id"], enr2[0]["track_id"])

        # 2. Pixel distance test (> 100px) when no GPS
        obs3 = [{"class": "ghost_net", "confidence": 0.85, "box": {"x": 50, "y": 50, "w": 40, "h": 40}}]
        obs4 = [{"class": "ghost_net", "confidence": 0.85, "box": {"x": 300, "y": 300, "w": 40, "h": 40}}]

        enr3 = self.tracker.process_scan_observations("SURVEY_PX", "SCAN_01", obs3, timestamp_epoch=100.0)
        enr4 = self.tracker.process_scan_observations("SURVEY_PX", "SCAN_02", obs4, timestamp_epoch=105.0)

        self.assertNotEqual(enr3[0]["track_id"], enr4[0]["track_id"])

    def test_time_gap_exceeded_creates_new_track(self):
        """If elapsed time between scans exceeds max_time_gap_s (30s), starts new track."""
        obs1 = [{"class": "ghost_net", "confidence": 0.90, "box": {"x": 100, "y": 100, "w": 50, "h": 50}}]
        obs2 = [{"class": "ghost_net", "confidence": 0.90, "box": {"x": 102, "y": 101, "w": 50, "h": 50}}]

        enr1 = self.tracker.process_scan_observations("SURVEY_TIME", "SCAN_01", obs1, timestamp_epoch=100.0)
        enr2 = self.tracker.process_scan_observations("SURVEY_TIME", "SCAN_02", obs2, timestamp_epoch=150.0) # 50s gap (> 30s)

        self.assertNotEqual(enr1[0]["track_id"], enr2[0]["track_id"])

    def test_deterministic_matching_repeatability(self):
        """Identical sequences of scan observations must yield strictly identical track associations."""
        tracker1 = TemporalPersistenceTracker(self.config)
        tracker2 = TemporalPersistenceTracker(self.config)

        scans = [
            [{"class": "mine_cylinder", "confidence": 0.80, "box": {"x": 100, "y": 100, "w": 30, "h": 30}}],
            [{"class": "mine_cylinder", "confidence": 0.82, "box": {"x": 104, "y": 102, "w": 32, "h": 31}}],
            [{"class": "mine_cylinder", "confidence": 0.85, "box": {"x": 106, "y": 103, "w": 30, "h": 30}}]
        ]

        tracks1 = []
        for i, s in enumerate(scans):
            e = tracker1.process_scan_observations("SURV_REP", f"S_{i}", s, timestamp_epoch=float(i * 5))
            tracks1.append(e[0]["observation_count"])

        tracks2 = []
        for i, s in enumerate(scans):
            e = tracker2.process_scan_observations("SURV_REP", f"S_{i}", s, timestamp_epoch=float(i * 5))
            tracks2.append(e[0]["observation_count"])

        self.assertEqual(tracks1, tracks2)
        self.assertEqual(tracks1, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
