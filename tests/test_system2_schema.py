"""
tests/test_system2_schema.py
Unit tests for System 2 Pydantic schema validation, output contracts,
and JSON repair capabilities.
"""

import unittest
from pydantic import ValidationError
from engine.system2 import TacticalAnalysis, OperatorQueryResponse, System2MissionContext, GroqSystem2Engine

class TestSystem2Schema(unittest.TestCase):
    """
    Validates Pydantic schema compliance for System 2 TacticalAnalysis and OperatorQueryResponse.
    """

    def test_tactical_analysis_valid_schema(self):
        """Verifies that TacticalAnalysis initializes correctly with valid types."""
        analysis = TacticalAnalysis(
            incident_summary="Ghost net entanglement hazard detected.",
            observed_evidence=["Contact: ghost_net (92%)", "Elevation: 1.5m"],
            uncertainties=["Net mesh extent unconfirmed"],
            risk_interpretation="System 1 evaluated EMERGENCY_PROP_HAZARD (score 8.5/10.0)",
            operator_action="Increase altitude by +3.0m",
            recovery_priority="CRITICAL",
            questions_for_operator=["Is recovery grapple available?"],
            model="llama-3.3-70b-versatile",
            status="ACTIVE",
            latency_ms=1240.5
        )
        self.assertEqual(analysis.recovery_priority, "CRITICAL")
        self.assertEqual(len(analysis.observed_evidence), 2)
        self.assertEqual(analysis.status, "ACTIVE")
        self.assertIsInstance(analysis.model_dump(), dict)

    def test_tactical_analysis_missing_required_fields_raises(self):
        """Missing mandatory fields must raise pydantic.ValidationError."""
        with self.assertRaises(ValidationError):
            TacticalAnalysis(
                incident_summary="Incomplete summary"
                # Missing observed_evidence, uncertainties, risk_interpretation, operator_action, recovery_priority, questions_for_operator
            )

    def test_operator_query_response_schema(self):
        """Verifies OperatorQueryResponse output contract."""
        resp = OperatorQueryResponse(
            answer="The primary obstacle is a ghost net.",
            evidence_used=["Contact class: ghost_net", "Hazard score: 8.5"],
            uncertainties=["Single pass acoustic backscatter"],
            model="deterministic_fallback",
            latency_ms=0.4,
            status="FALLBACK"
        )
        self.assertTrue(resp.answer.startswith("The primary"))
        self.assertEqual(len(resp.evidence_used), 2)
        self.assertEqual(resp.status, "FALLBACK")

    def test_mission_context_builder(self):
        """Verifies System2MissionContext validation and default handling."""
        ctx = System2MissionContext(
            survey_id="SURVEY_TEST_01",
            contacts=[{"class": "mine_cylinder", "confidence": 0.88}],
            system1={"decision_primitive": "LOITER_AND_RESCAN", "hazard_score": 7.0}
        )
        self.assertEqual(ctx.survey_id, "SURVEY_TEST_01")
        self.assertEqual(len(ctx.contacts), 1)
        self.assertEqual(ctx.system1["decision_primitive"], "LOITER_AND_RESCAN")
        self.assertIsNone(ctx.geo)


if __name__ == "__main__":
    unittest.main()
