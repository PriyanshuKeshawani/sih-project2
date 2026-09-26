import unittest
from engine.laya_adapter import (
    System1InputState,
    System1Decision,
    System1SafetyGuardrails
)


class TestSystem1Guardrails(unittest.TestCase):
    """
    Validates Deterministic Safety Guardrails operating on System 1 decisions (Section 9).
    Guarantees Laya cannot suppress critical hazards or violate marine safety limits.
    """

    def test_guardrail_escalates_passive_log_on_elevated_ghost_net(self):
        """If an engine attempts PASSIVE_LOG on an elevated net obstacle, guardrail must force EMERGENCY."""
        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.88,
            shadow_detected=True,
            elevation_m=1.20,
            evidence_quality="STRONG"
        )
        unsafe_decision = System1Decision(
            decision_primitive="PASSIVE_LOG",
            hazard_score=3.0,
            needs_operator_review=False,
            laya_confidence=0.90,
            detector_confidence=0.88,
            evidence_quality="STRONG",
            physics_provenance="DERIVED",
            geo_status="DERIVED",
            model="convaiinnovations/laya",
            latency_ms=30.0,
            status="ACTIVE",
            reasons=["Raw engine decision"]
        )

        safe_decision = System1SafetyGuardrails.apply(unsafe_decision, state)

        self.assertEqual(safe_decision.decision_primitive, "EMERGENCY_PROP_HAZARD")
        self.assertGreaterEqual(safe_decision.hazard_score, 8.5)
        self.assertTrue(safe_decision.needs_operator_review)
        self.assertTrue(safe_decision.guardrail_applied)
        self.assertIn("SAFETY GUARDRAIL OVERRIDE", " ".join(safe_decision.reasons))

    def test_guardrail_escalates_critical_mine_cylinder(self):
        """High-confidence mine_cylinder cannot be demoted to PASSIVE_LOG."""
        state = System1InputState(
            contact_class="mine_cylinder",
            detector_confidence=0.85,
            shadow_detected=True,
            evidence_quality="STRONG"
        )
        unsafe_decision = System1Decision(
            decision_primitive="PASSIVE_LOG",
            hazard_score=4.0,
            needs_operator_review=False,
            laya_confidence=0.85,
            detector_confidence=0.85,
            evidence_quality="STRONG",
            physics_provenance="DERIVED",
            geo_status="UNAVAILABLE",
            model="convaiinnovations/laya",
            latency_ms=25.0,
            status="ACTIVE"
        )

        safe_decision = System1SafetyGuardrails.apply(unsafe_decision, state)

        self.assertEqual(safe_decision.decision_primitive, "EMERGENCY_PROP_HAZARD")
        self.assertTrue(safe_decision.needs_operator_review)
        self.assertTrue(safe_decision.guardrail_applied)

    def test_guardrail_clamps_out_of_bound_scores(self):
        """Scores outside [1.0, 10.0] are strictly clamped."""
        state = System1InputState(
            contact_class="crab_pot",
            detector_confidence=0.50,
            evidence_quality="MODERATE"
        )
        over_decision = System1Decision(
            decision_primitive="PASSIVE_LOG",
            hazard_score=14.5,
            needs_operator_review=False,
            laya_confidence=0.6,
            detector_confidence=0.50,
            evidence_quality="MODERATE",
            physics_provenance="UNAVAILABLE",
            geo_status="UNAVAILABLE",
            model="test-engine",
            latency_ms=10.0,
            status="ACTIVE"
        )
        clamped = System1SafetyGuardrails.apply(over_decision, state)
        self.assertEqual(clamped.hazard_score, 10.0)

        under_decision = System1Decision(
            decision_primitive="PASSIVE_LOG",
            hazard_score=-2.5,
            needs_operator_review=False,
            laya_confidence=0.6,
            detector_confidence=0.50,
            evidence_quality="MODERATE",
            physics_provenance="UNAVAILABLE",
            geo_status="UNAVAILABLE",
            model="test-engine",
            latency_ms=10.0,
            status="ACTIVE"
        )
        clamped_low = System1SafetyGuardrails.apply(under_decision, state)
        self.assertEqual(clamped_low.hazard_score, 1.0)

    def test_emergency_decision_mandates_operator_review(self):
        """Any EMERGENCY_PROP_HAZARD decision must enforce needs_operator_review=True."""
        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.92,
            evidence_quality="STRONG"
        )
        decision = System1Decision(
            decision_primitive="EMERGENCY_PROP_HAZARD",
            hazard_score=9.0,
            needs_operator_review=False, # Invalid state for emergency
            laya_confidence=0.95,
            detector_confidence=0.92,
            evidence_quality="STRONG",
            physics_provenance="DERIVED",
            geo_status="DERIVED",
            model="convaiinnovations/laya",
            latency_ms=28.0,
            status="ACTIVE"
        )
        guarded = System1SafetyGuardrails.apply(decision, state)
        self.assertTrue(guarded.needs_operator_review)
        self.assertTrue(guarded.guardrail_applied)


if __name__ == "__main__":
    unittest.main()
