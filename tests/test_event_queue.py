import unittest
import uuid
import time

from engine.reflex import MissionEventQueue, MissionEvent, DecisionPrimitive, AlertSeverity


class TestEventQueue(unittest.TestCase):
    """
    Unit tests for in-memory MissionEventQueue and same-scan debouncing.
    """

    def setUp(self):
        self.queue = MissionEventQueue(max_capacity=10)

    def _create_sample_event(self, scan_id: str = "scan_01", cls_name: str = "ghost_net", box: tuple = (100, 100, 150, 150), severity: str = "CRITICAL") -> MissionEvent:
        fp_x1, fp_y1, fp_x2, fp_y2 = [v // 10 for v in box]
        fingerprint = f"{scan_id}_{cls_name}_{fp_x1}_{fp_y1}_{fp_x2}_{fp_y2}"
        return MissionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            survey_id=scan_id,
            decision_primitive=DecisionPrimitive.EMERGENCY_PROP_HAZARD.value if severity == "CRITICAL" else DecisionPrimitive.PASSIVE_LOG.value,
            hazard_score=8.5 if severity == "CRITICAL" else 3.0,
            alert_severity=severity,
            contact={"class": cls_name, "confidence": 0.90, "bbox_xyxy": list(box)},
            evidence_quality="STRONG",
            reasons=["test reason"],
            navigation={"maneuver": None, "status": "UNAVAILABLE"},
            fingerprint=fingerprint
        )

    def test_same_scan_duplicate_suppressed(self):
        """
        TEST 9: Same scan duplicate returns False and is not duplicated in queue.
        """
        ev1 = self._create_sample_event(scan_id="scan_A", box=(100, 100, 150, 150))
        ev2 = self._create_sample_event(scan_id="scan_A", box=(102, 104, 151, 153))  # Overlapping box in same 10px grid

        added1 = self.queue.add_event(ev1)
        added2 = self.queue.add_event(ev2)

        self.assertTrue(added1, "First event should be added")
        self.assertFalse(added2, "Duplicate event in same scan should be suppressed")
        self.assertEqual(len(self.queue), 1)

    def test_different_scans_produce_separate_events(self):
        """
        TEST 10: Different scans or distinct objects produce separate event IDs.
        """
        ev_scan1 = self._create_sample_event(scan_id="scan_01", box=(100, 100, 150, 150))
        ev_scan2 = self._create_sample_event(scan_id="scan_02", box=(100, 100, 150, 150))

        added1 = self.queue.add_event(ev_scan1)
        added2 = self.queue.add_event(ev_scan2)

        self.assertTrue(added1)
        self.assertTrue(added2)
        self.assertEqual(len(self.queue), 2)
        self.assertNotEqual(ev_scan1.event_id, ev_scan2.event_id)

    def test_critical_events_filtering(self):
        """
        get_critical_events filters only CRITICAL severity events.
        """
        ev_crit = self._create_sample_event(scan_id="scan_A", box=(10, 10, 50, 50), severity="CRITICAL")
        ev_info = self._create_sample_event(scan_id="scan_B", box=(200, 200, 250, 250), severity="INFO")

        self.queue.add_event(ev_crit)
        self.queue.add_event(ev_info)

        crit_list = self.queue.get_critical_events()
        self.assertEqual(len(crit_list), 1)
        self.assertEqual(crit_list[0]["alert_severity"], "CRITICAL")
        self.assertEqual(len(self.queue.get_recent_events()), 2)


if __name__ == '__main__':
    unittest.main()
