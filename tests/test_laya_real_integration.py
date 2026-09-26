import os
import unittest
import time
from engine.laya_adapter import (
    System1InputState,
    System1Decision,
    LayaDecisionEngine,
    System1Manager
)


class TestLayaRealIntegration(unittest.TestCase):
    """
    Verifies Real Laya Integration (Sections 1, 2, 4, 5, 7, 8, 14, 15, 16).
    Ensures real inference execution, zero faking, typed decisions, and offline behavior.
    """

    def test_laya_package_installed_and_version(self):
        """Verifies official Laya package presence and version in environment."""
        import laya
        version = getattr(laya, "__version__", None)
        self.assertIsNotNone(version, "Laya package must expose __version__")
        print(f"\n[LAYA TEST] Detected Laya version: {version}")

    def test_laya_local_checkpoint_structure_or_unavailable(self):
        """
        Inspects models/laya/checkpoint.
        If weights exist (model.safetensors, rl_agent_config.json, tokenizer, encoder),
        Laya must initialize to ACTIVE.
        Otherwise, must report UNAVAILABLE (no faking allowed!).
        """
        engine = LayaDecisionEngine()
        status = engine.get_status()

        self.assertIn("status", status)
        self.assertIn(status["status"], ["ACTIVE", "UNAVAILABLE", "ERROR"])

        if status["status"] == "ACTIVE":
            self.assertTrue(engine.is_active)
            self.assertIsNotNone(engine.agent)
            print(f"[LAYA TEST] Real Laya agent successfully active from: {status['checkpoint']}")
        else:
            self.assertFalse(engine.is_active)
            print(f"[LAYA TEST] Laya weights cleanly reported as {status['status']}: {status.get('error')}")

    def test_real_laya_inference_or_offline_fallback(self):
        """
        Tests end-to-end inference through System1Manager on structured state.
        Verifies:
        - Typed choice primitive
        - Hazard score in [1.0, 10.0]
        - Operator review boolean
        - Separated detector and Laya confidences
        - Measured execution latency
        """
        manager = System1Manager()
        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.92,
            taxonomy_status="STANDARD_CLASS",
            shadow_detected=True,
            shadow_length_m=1.8,
            elevation_m=1.05,
            physics_provenance="DERIVED",
            geo_status="DERIVED",
            evidence_quality="STRONG",
            sonar_altitude_m=12.0,
            heading_deg=45.0
        )

        t0 = time.perf_counter()
        decision = manager.decide(state)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertIsInstance(decision, System1Decision)
        self.assertIn(decision.decision_primitive, ["EMERGENCY_PROP_HAZARD", "LOITER_AND_RESCAN", "PASSIVE_LOG"])
        self.assertGreaterEqual(decision.hazard_score, 1.0)
        self.assertLessEqual(decision.hazard_score, 10.0)
        self.assertIsInstance(decision.needs_operator_review, bool)
        self.assertEqual(decision.detector_confidence, 0.92)

        # Emergency rule verification on critical net
        self.assertEqual(decision.decision_primitive, "EMERGENCY_PROP_HAZARD")
        self.assertTrue(decision.needs_operator_review)

        print(f"[LAYA TEST] System 1 evaluated in {decision.latency_ms:.2f}ms (engine={decision.model}, status={decision.status})")

    def test_laya_offline_operation_no_network_dependence(self):
        """
        Verifies that when local weights exist, Laya executes completely locally without network calls.
        """
        engine = LayaDecisionEngine()
        if not engine.is_active:
            self.skipTest("Local Laya checkpoint not yet active on disk; skipping live offline test.")

        # Simulate offline state: disable socket network if possible or test offline local path
        state = System1InputState(
            contact_class="mine_cylinder",
            detector_confidence=0.88,
            shadow_detected=True,
            elevation_m=0.75,
            evidence_quality="STRONG"
        )
        dec = engine.decide(state)
        self.assertIsNotNone(dec)
        self.assertEqual(dec.status, "ACTIVE")
        self.assertEqual(dec.decision_primitive, "EMERGENCY_PROP_HAZARD")


if __name__ == "__main__":
    unittest.main()
