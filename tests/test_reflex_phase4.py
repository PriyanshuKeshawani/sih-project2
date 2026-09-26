import unittest
import numpy as np
import time

from engine.reflex import (
    System1ReflexEngine,
    DecisionPrimitive,
    AlertSeverity,
    EvidenceQuality
)


class TestReflexPhase4(unittest.TestCase):
    """
    Unit tests for System 1 Edge Reflex Engine decisions, determinism, and performance.
    """

    def setUp(self):
        self.engine = System1ReflexEngine(enable_laya=False)


    def test_high_confidence_ghost_net_strong_evidence(self):
        """
        TEST 1: High-confidence ghost_net + strong evidence.
        Expected: EMERGENCY_PROP_HAZARD primitive, CRITICAL severity alert.
        """
        det = {
            "class": "ghost_net",
            "confidence": 0.92,
            "bbox_xyxy": [100, 100, 150, 150]
        }
        phys = {
            "shadow_detected": True,
            "elevation_m": 1.20,
            "shadow_length_m": 2.5
        }
        geo = {"status": "DERIVED"}
        meta = {"heading_deg": 45.0, "sonar_altitude_m": 12.0}

        res = self.engine.process_reflex(det, physics=phys, geo=geo, metadata=meta)

        self.assertEqual(res["decision_primitive"], DecisionPrimitive.EMERGENCY_PROP_HAZARD.value)
        self.assertEqual(res["alert_severity"], AlertSeverity.CRITICAL.value)
        self.assertGreaterEqual(res["hazard_score"], 8.0)
        self.assertEqual(res["evidence_quality"], EvidenceQuality.STRONG.value)
        self.assertIn("propeller fouling hazard", " ".join(res["reasons"]).lower())

    def test_low_confidence_ghost_net_triggers_rescan(self):
        """
        TEST 2: Low-confidence ghost_net.
        Expected: Not automatically emergency; triggers LOITER_AND_RESCAN for verification.
        """
        det = {
            "class": "ghost_net",
            "confidence": 0.38,
            "bbox_xyxy": [100, 100, 150, 150]
        }
        phys = {"shadow_detected": False, "elevation_m": None}

        res = self.engine.process_reflex(det, physics=phys, geo=None, metadata=None)

        self.assertEqual(res["decision_primitive"], DecisionPrimitive.LOITER_AND_RESCAN.value)
        self.assertEqual(res["alert_severity"], AlertSeverity.WARNING.value)
        self.assertIn("secondary observation recommended", " ".join(res["reasons"]).lower())

    def test_high_confidence_shipwreck_deterministic_decision(self):
        """
        TEST 3: High-confidence shipwreck.
        Expected: Deterministic logging / navigation structural contact classification.
        """
        det = {
            "class": "shipwreck",
            "confidence": 0.88,
            "bbox_xyxy": [200, 150, 350, 300]
        }
        phys = {"shadow_detected": True, "elevation_m": 0.4}

        res = self.engine.process_reflex(det, physics=phys, geo=None, metadata=None)

        self.assertIn(res["decision_primitive"], [DecisionPrimitive.PASSIVE_LOG.value, DecisionPrimitive.LOITER_AND_RESCAN.value])
        self.assertIn("shipwreck", " ".join(res["reasons"]))
        self.assertIn("navigation hazard", " ".join(res["reasons"]))

    def test_pipeline_contact_infrastructure_classification(self):
        """
        TEST 4: Pipeline contact.
        Expected: Classified as critical infrastructure / benthic asset.
        """
        det = {
            "class": "submarine_pipeline",
            "confidence": 0.85,
            "bbox_xyxy": [50, 200, 500, 230]
        }
        res = self.engine.process_reflex(det, physics=None, geo=None, metadata=None)

        self.assertIn("critical infrastructure", " ".join(res["reasons"]).lower())

    def test_missing_navigation_metadata_returns_unavailable(self):
        """
        TEST 8: Missing navigation data.
        Expected: maneuver = UNAVAILABLE, no fabricated 'turn 20 degrees' command!
        """
        det = {"class": "ghost_net", "confidence": 0.90, "bbox_xyxy": [100, 100, 150, 150]}
        res = self.engine.process_reflex(det, physics=None, geo=None, metadata=None)

        self.assertEqual(res["navigation"]["status"], "UNAVAILABLE")
        self.assertIsNone(res["navigation"]["maneuver"])

    def test_determinism_100_runs(self):
        """
        TEST 11: Determinism. Run identical input 100 times.
        Expected: Exactly the same decision_primitive, hazard_score, alert_severity,
        evidence_quality, and reasons across every single run.
        """
        det = {"class": "ghost_net", "confidence": 0.88, "bbox_xyxy": [100, 100, 150, 150]}
        phys = {"shadow_detected": True, "elevation_m": 1.1}
        meta = {"heading_deg": 90.0, "sonar_altitude_m": 10.0}

        baseline = self.engine.process_reflex(det, physics=phys, metadata=meta)

        for _ in range(100):
            run_res = self.engine.process_reflex(det, physics=phys, metadata=meta)
            self.assertEqual(run_res["decision_primitive"], baseline["decision_primitive"])
            self.assertEqual(run_res["hazard_score"], baseline["hazard_score"])
            self.assertEqual(run_res["alert_severity"], baseline["alert_severity"])
            self.assertEqual(run_res["evidence_quality"], baseline["evidence_quality"])
            self.assertEqual(run_res["reasons"], baseline["reasons"])

    def test_performance_latency_benchmarking(self):
        """
        TEST 12: Performance latency benchmarking over 200 iterations.
        Verifies sub-millisecond execution on edge CPU without artificial delays.
        """
        det = {"class": "ghost_net", "confidence": 0.90, "bbox_xyxy": [120, 120, 180, 180]}
        phys = {"shadow_detected": True, "elevation_m": 1.0}
        meta = {"heading_deg": 180.0, "sonar_altitude_m": 15.0}

        latencies = []
        for _ in range(200):
            t0 = time.perf_counter()
            self.engine.process_reflex(det, physics=phys, metadata=meta)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = np.mean(latencies)
        p95_lat = np.percentile(latencies, 95)

        # Confirm actual fast execution (< 2ms per target)
        self.assertLess(mean_lat, 2.0, f"Mean latency {mean_lat:.3f}ms exceeds 2.0ms threshold")
        self.assertLess(p95_lat, 5.0, f"P95 latency {p95_lat:.3f}ms exceeds 5.0ms threshold")


if __name__ == '__main__':
    unittest.main()
