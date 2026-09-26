import unittest
from engine.laya_adapter import (
    System1InputState,
    System1Decision,
    DeterministicDecisionEngine,
    LAYA_DECISION_QUESTIONS
)


class TestSystem1Contract(unittest.TestCase):
    """
    Validates System 1 Typed Decision Contracts and Input/Output Schemas (Sections 6, 7, 8).
    Ensures no invented facts, strictly typed choice primitives, score bounds, and noul flags.
    """

    def setUp(self):
        self.engine = DeterministicDecisionEngine()

    def test_input_state_builder_structure(self):
        """Validates that System1InputState faithfully structures facts without raw imagery."""
        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.92,
            taxonomy_status="STANDARD_CLASS",
            shadow_detected=True,
            shadow_length_m=1.85,
            elevation_m=0.95,
            physics_provenance="DERIVED",
            geo_status="DERIVED",
            evidence_quality="STRONG",
            sonar_altitude_m=12.0,
            heading_deg=45.0
        )
        data = state.to_dict()
        self.assertEqual(data["contact"]["class"], "ghost_net")
        self.assertEqual(data["contact"]["confidence"], 0.92)
        self.assertTrue(data["physics"]["shadow_detected"])
        self.assertEqual(data["physics"]["elevation_m"], 0.95)
        self.assertEqual(data["evidence"]["quality"], "STRONG")

        prompt = state.to_laya_prompt()
        self.assertIn("ghost_net", prompt)
        self.assertIn("0.92", prompt)
        self.assertIn("0.95m", prompt)

    def test_output_contract_fields_and_types(self):
        """Verifies System1Decision fields, value ranges, and preservation of perception facts."""
        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.91,
            shadow_detected=True,
            elevation_m=1.1,
            evidence_quality="STRONG"
        )
        decision = self.engine.decide(state)

        # 1. Decision primitive strictly one of the 3 allowed options (Section 7)
        self.assertIn(
            decision.decision_primitive,
            ["EMERGENCY_PROP_HAZARD", "LOITER_AND_RESCAN", "PASSIVE_LOG"]
        )
        # 2. Hazard score bounded in [1.0, 10.0]
        self.assertGreaterEqual(decision.hazard_score, 1.0)
        self.assertLessEqual(decision.hazard_score, 10.0)
        # 3. Operator review is boolean
        self.assertIsInstance(decision.needs_operator_review, bool)
        # 4. Latency measured in ms
        self.assertGreater(decision.latency_ms, 0.0)
        # 5. Status is valid
        self.assertIn(decision.status, ["ACTIVE", "FALLBACK", "UNAVAILABLE"])
        # 6. Preserves original perception facts
        self.assertEqual(decision.detector_confidence, 0.91)
        self.assertEqual(decision.evidence_quality, "STRONG")

    def test_missing_physics_handled_gracefully(self):
        """Missing acoustic shadow and elevation must not crash System 1."""
        state = System1InputState(
            contact_class="submarine_pipeline",
            detector_confidence=0.82,
            shadow_detected=False,
            elevation_m=None,
            physics_provenance="UNAVAILABLE",
            evidence_quality="MODERATE"
        )
        decision = self.engine.decide(state)
        self.assertIsNotNone(decision)
        self.assertIn(decision.decision_primitive, ["LOITER_AND_RESCAN", "PASSIVE_LOG"])
        self.assertEqual(decision.physics_provenance, "UNAVAILABLE")

    def test_missing_geo_handled_gracefully(self):
        """Missing georeferencing telemetry must be preserved as UNAVAILABLE."""
        state = System1InputState(
            contact_class="crab_pot",
            detector_confidence=0.75,
            geo_status="UNAVAILABLE",
            evidence_quality="MODERATE"
        )
        decision = self.engine.decide(state)
        self.assertEqual(decision.geo_status, "UNAVAILABLE")
        self.assertEqual(decision.decision_primitive, "PASSIVE_LOG")

    def test_no_direct_actuator_control_generated(self):
        """System 1 decisions must remain advisory decision primitives, never direct thruster commands."""
        state = System1InputState(
            contact_class="ghost_net",
            detector_confidence=0.95,
            shadow_detected=True,
            elevation_m=1.4,
            evidence_quality="STRONG"
        )
        decision = self.engine.decide(state)
        d_dict = decision.to_dict()

        # Must not contain direct PWM / thruster rpm / rudder angles
        for key in ["pwm", "thruster_rpm", "rudder_angle_deg", "motor_voltage"]:
            self.assertNotIn(key, d_dict)

    def test_question_schema_compliance(self):
        """Verifies that LAYA_DECISION_QUESTIONS conforms to official Laya schema specs."""
        self.assertIn("action", LAYA_DECISION_QUESTIONS)
        self.assertEqual(LAYA_DECISION_QUESTIONS["action"]["type"], "choice")
        criteria = LAYA_DECISION_QUESTIONS["action"]["criteria"]
        self.assertIn("EMERGENCY_PROP_HAZARD", criteria)
        self.assertIn("LOITER_AND_RESCAN", criteria)
        self.assertIn("PASSIVE_LOG", criteria)

        self.assertIn("hazard_score", LAYA_DECISION_QUESTIONS)
        self.assertEqual(LAYA_DECISION_QUESTIONS["hazard_score"]["type"], "score")

        self.assertIn("needs_operator_review", LAYA_DECISION_QUESTIONS)
        self.assertEqual(LAYA_DECISION_QUESTIONS["needs_operator_review"]["type"], "noul")


if __name__ == "__main__":
    unittest.main()
