"""
tests/test_persistence_status.py
Unit tests for persistence status transitions (NEW_CONTACT -> PERSISTENT -> TRANSIENT)
and integration with System 1 Edge Reflex Engine.
"""

import unittest
from engine.temporal_tracking import TemporalPersistenceTracker, TrackingConfig, PersistenceStatus
from engine.reflex import System1ReflexEngine, EvidenceQuality

class TestPersistenceStatus(unittest.TestCase):
    """
    Validates state machine transitions for acoustic persistence and System 1 enrichment.
    """

    def setUp(self):
        self.config = TrackingConfig(
            min_observations_for_persistent=3,
            transient_miss_threshold=2
        )
        self.tracker = TemporalPersistenceTracker(self.config)
        self.reflex_engine = System1ReflexEngine(enable_laya=False)

    def test_persistence_status_progression(self):
        """Contact transitions from NEW_CONTACT -> NEW_CONTACT -> PERSISTENT on 3rd observation."""
        obs = [{"class": "ghost_net", "confidence": 0.90, "box": {"x": 100, "y": 100, "w": 50, "h": 50}}]

        # Scan 1: First seen -> NEW_CONTACT
        enr1 = self.tracker.process_scan_observations("SURV_P", "SCAN_1", obs, timestamp_epoch=10.0)
        self.assertEqual(enr1[0]["persistence_status"], PersistenceStatus.NEW_CONTACT.value)
        self.assertEqual(enr1[0]["observation_count"], 1)

        # Scan 2: Second seen -> still NEW_CONTACT (requires >= 3)
        enr2 = self.tracker.process_scan_observations("SURV_P", "SCAN_2", obs, timestamp_epoch=15.0)
        self.assertEqual(enr2[0]["persistence_status"], PersistenceStatus.NEW_CONTACT.value)
        self.assertEqual(enr2[0]["observation_count"], 2)

        # Scan 3: Third seen -> PERSISTENT
        enr3 = self.tracker.process_scan_observations("SURV_P", "SCAN_3", obs, timestamp_epoch=20.0)
        self.assertEqual(enr3[0]["persistence_status"], PersistenceStatus.PERSISTENT.value)
        self.assertEqual(enr3[0]["observation_count"], 3)

    def test_transient_status_on_disappearance(self):
        """Contact seen once and missed in next 2 scans is classified as TRANSIENT."""
        obs = [{"class": "shipwreck", "confidence": 0.85, "box": {"x": 100, "y": 100, "w": 60, "h": 60}}]

        # Scan 1: Contact detected
        enr1 = self.tracker.process_scan_observations("SURV_TR", "SCAN_1", obs, timestamp_epoch=10.0)
        track_id = enr1[0]["track_id"]

        # Scan 2: Clear seafloor (miss 1)
        self.tracker.process_scan_observations("SURV_TR", "SCAN_2", [], timestamp_epoch=15.0)
        track = self.tracker.get_track(track_id)
        self.assertEqual(track.missed_scans, 1)
        self.assertEqual(track.persistence_status, PersistenceStatus.NEW_CONTACT.value)

        # Scan 3: Clear seafloor (miss 2 -> reaches transient_miss_threshold)
        self.tracker.process_scan_observations("SURV_TR", "SCAN_3", [], timestamp_epoch=20.0)
        track = self.tracker.get_track(track_id)
        self.assertEqual(track.missed_scans, 2)
        self.assertEqual(track.persistence_status, PersistenceStatus.TRANSIENT.value)

    def test_simultaneous_contacts_persistent_and_transient(self):
        """
        Multiple contacts in same survey:
        Scan 1: ghost_net + shipwreck
        Scan 2: ghost_net + shipwreck
        Scan 3: ghost_net only
        Scan 4: ghost_net only
        Result: ghost_net -> PERSISTENT; shipwreck -> TRANSIENT
        """
        net_obs = {"class": "ghost_net", "confidence": 0.92, "box": {"x": 100, "y": 100, "w": 50, "h": 50}}
        wreck_obs = {"class": "shipwreck", "confidence": 0.85, "box": {"x": 350, "y": 350, "w": 80, "h": 70}}

        # Scan 1
        self.tracker.process_scan_observations("SURV_SIMUL", "S1", [net_obs, wreck_obs], timestamp_epoch=0.0)
        # Scan 2
        self.tracker.process_scan_observations("SURV_SIMUL", "S2", [net_obs, wreck_obs], timestamp_epoch=5.0)
        # Scan 3 (shipwreck absent)
        self.tracker.process_scan_observations("SURV_SIMUL", "S3", [net_obs], timestamp_epoch=10.0)
        # Scan 4 (shipwreck absent)
        self.tracker.process_scan_observations("SURV_SIMUL", "S4", [net_obs], timestamp_epoch=15.0)

        tracks = self.tracker.get_tracks_for_survey("SURV_SIMUL")
        net_track = next(t for t in tracks if t.target_class == "ghost_net")
        wreck_track = next(t for t in tracks if t.target_class == "shipwreck")

        self.assertEqual(net_track.persistence_status, PersistenceStatus.PERSISTENT.value)
        self.assertEqual(net_track.observation_count, 4)

        self.assertEqual(wreck_track.persistence_status, PersistenceStatus.TRANSIENT.value)
        self.assertEqual(wreck_track.observation_count, 2)

    def test_system1_reflex_enrichment_from_persistence(self):
        """System 1 Reflex evaluates higher evidence quality when temporal persistence is confirmed."""
        temporal_info = {
            "persistence_status": PersistenceStatus.PERSISTENT.value,
            "observation_count": 4,
            "track_age_s": 15.5
        }
        res = self.reflex_engine.process_reflex(
            detection={"class": "ghost_net", "confidence": 0.90, "box": {"x": 100, "y": 100, "w": 50, "h": 50}},
            physics={"elevation_m": 1.5, "shadow_detected": True},
            temporal=temporal_info
        )

        self.assertEqual(res["evidence_quality"], EvidenceQuality.STRONG.value)
        reasons_text = " ".join(res["reasons"])
        self.assertIn("Multi-ping persistence confirmed", reasons_text)


if __name__ == "__main__":
    unittest.main()
