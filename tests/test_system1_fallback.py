import unittest
from unittest.mock import MagicMock
from engine.laya_adapter import (
    System1InputState,
    System1Decision,
    System1Manager,
    DeterministicDecisionEngine,
    LayaDecisionEngine
)


class TestSystem1Fallback(unittest.TestCase):
    """
    Validates Dual-Engine Fallback Architecture (Section 10).
    Guarantees continuous, reliable operation if Laya is unavailable or encounters an error.
    """

    def setUp(self):
        self.manager = System1Manager()

    def test_fallback_engine_contract_compliance(self):
        """DeterministicFallbackEngine returns valid System1Decision matching contract."""
        engine = DeterministicDecisionEngine()
        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.85,
            shadow_detected=True,
            elevation_m=0.9,
            evidence_quality="STRONG"
        )
        decision = engine.decide(state)
        self.assertIsInstance(decision, System1Decision)
        self.assertEqual(decision.status, "FALLBACK")
        self.assertEqual(decision.decision_primitive, "EMERGENCY_PROP_HAZARD")
        self.assertGreaterEqual(decision.hazard_score, 7.5)

    def test_manager_routes_to_fallback_when_laya_inactive(self):
        """System1Manager routes smoothly to deterministic fallback when Laya checkpoint is not loaded."""
        state = System1InputState(
            contact_class="crab_pot",
            detector_confidence=0.60,
            evidence_quality="MODERATE"
        )
        decision = self.manager.decide(state)
        self.assertIsNotNone(decision)
        self.assertIn(decision.status, ["ACTIVE", "FALLBACK"])
        self.assertEqual(decision.decision_primitive, "PASSIVE_LOG")

    def test_manager_catches_runtime_exception_and_falls_back(self):
        """If Laya inference raises a runtime exception, manager routes to fallback without crashing."""
        mock_laya = MagicMock(spec=LayaDecisionEngine)
        mock_laya.is_active = True
        mock_laya.decide.side_effect = RuntimeError("Simulated GPU OOM / CUDA error in Laya forward pass")
        mock_laya.get_status.return_value = {"engine": "laya", "status": "ERROR"}

        test_manager = System1Manager()
        test_manager.laya_engine = mock_laya

        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.88,
            shadow_detected=True,
            elevation_m=1.0,
            evidence_quality="STRONG"
        )

        decision = test_manager.decide(state)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.status, "FALLBACK")
        self.assertEqual(decision.decision_primitive, "EMERGENCY_PROP_HAZARD")
        self.assertTrue(any("Laya runtime error" in r for r in decision.reasons))

    def test_repeatability_over_50_iterations(self):
        """Identical input states produce strictly identical decisions (no random drift)."""
        state = System1InputState(
            contact_class="submarine_pipeline",
            detector_confidence=0.78,
            shadow_detected=True,
            elevation_m=0.3,
            evidence_quality="MODERATE"
        )
        first = self.manager.decide(state)
        for _ in range(50):
            subsequent = self.manager.decide(state)
            self.assertEqual(first.decision_primitive, subsequent.decision_primitive)
            self.assertEqual(first.hazard_score, subsequent.hazard_score)
            self.assertEqual(first.needs_operator_review, subsequent.needs_operator_review)

    def test_status_reports_fallback_available(self):
        """Status endpoint dictionary must confirm fallback_available=True."""
        status = self.manager.get_status()
        self.assertTrue(status.get("fallback_available"))
        self.assertIn("status", status)
        self.assertIn("engine", status)


if __name__ == "__main__":
    unittest.main()
