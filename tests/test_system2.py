"""
tests/test_system2.py
Core integration tests for System 2 Tactical Reasoning Engine (Groq & Fallback).
Verifies closed-set detector handling, uncertainty preservation, zero sensor fabrication,
multi-contact reasoning, and grounded operator Q&A.
"""

import unittest
from unittest.mock import MagicMock
import json

from engine.system2 import (
    GroqSystem2Engine,
    System2MissionContext,
    TacticalAnalysis,
    OperatorQueryResponse
)

class TestSystem2(unittest.TestCase):
    """
    Validates System 2 reasoning behavior, closed-set limitations, and grounding integrity.
    """

    def setUp(self):
        self.engine = GroqSystem2Engine(api_key="gsk_mock_test_key_xyz")
        # Setup mock client returning structured JSON
        self.mock_client = MagicMock()
        self.engine.client = self.mock_client

    def test_valid_groq_response_parsing(self):
        """Verifies that a valid Groq JSON completion is parsed into TacticalAnalysis."""
        mock_payload = {
            "incident_summary": "Sonar detector identified an acoustic contact classified as ghost_net.",
            "observed_evidence": [
                "Detector confidence: 91.5%",
                "Derived elevation: 1.6m above seafloor via acoustic shadow trigonometry"
            ],
            "uncertainties": [
                "Single-aspect acoustic backscatter cannot resolve monofilament net tensile integrity"
            ],
            "risk_interpretation": "System 1 issued EMERGENCY_PROP_HAZARD (hazard score 8.5/10.0) due to propeller fouling risk.",
            "operator_action": "Elevate AUV altitude by +3.0m to avoid prop entanglement.",
            "recovery_priority": "CRITICAL",
            "questions_for_operator": [
                "Is specialized salvage vessel available in Sector 4?"
            ]
        }
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_payload)
        self.mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        context = System2MissionContext(
            survey_id="SURVEY_GROQ_01",
            contacts=[{"class": "ghost_net", "confidence": 0.915, "elevation_m": 1.6}],
            system1={"decision_primitive": "EMERGENCY_PROP_HAZARD", "hazard_score": 8.5}
        )

        analysis = self.engine.analyze_tactical(context)
        self.assertIsInstance(analysis, TacticalAnalysis)
        self.assertEqual(analysis.status, "ACTIVE")
        self.assertEqual(analysis.recovery_priority, "CRITICAL")
        self.assertEqual(len(analysis.observed_evidence), 2)
        self.assertIn("ghost_net", analysis.incident_summary)

    def test_closed_set_limitation_preserved_for_shipwreck(self):
        """
        Closed-set detector limitation: Shipwreck-class contact must be treated
        as a model prediction / surrogate, NOT as a physically confirmed shipwreck.
        """
        fallback_engine = GroqSystem2Engine(api_key=None)
        context = System2MissionContext(
            survey_id="SURVEY_WRECK",
            contacts=[{"class": "shipwreck", "confidence": 0.81, "elevation_m": 3.4}],
            geo={"status": "UNAVAILABLE"},
            system1={"decision_primitive": "PASSIVE_LOG", "hazard_score": 5.6}
        )
        analysis = fallback_engine.analyze_tactical(context)

        # Fallback or prompt must clearly state surrogate / unconfirmed status
        text = (analysis.incident_summary + " " + " ".join(analysis.uncertainties)).lower()
        self.assertTrue(
            "surrogate" in text or "closed-set" in text or "insufficient" in text or "reef" in text or "clutter" in text,
            f"Analysis must communicate closed-set surrogate uncertainty, got: {text}"
        )

    def test_no_fabricated_sensor_information_when_missing(self):
        """
        When coordinates and depth are UNAVAILABLE, System 2 must preserve that status
        and NEVER invent GPS coordinates or water depth.
        """
        fallback_engine = GroqSystem2Engine(api_key=None)
        context = System2MissionContext(
            survey_id="SURVEY_NO_GEO",
            contacts=[{"class": "submarine_pipeline", "confidence": 0.75}],
            geo={"status": "UNAVAILABLE", "latitude": None, "longitude": None},
            system1={"decision_primitive": "PASSIVE_LOG", "hazard_score": 4.5}
        )
        analysis = fallback_engine.analyze_tactical(context)
        evidence_text = " ".join(analysis.observed_evidence)
        self.assertIn("UNAVAILABLE", evidence_text)

    def test_multi_contact_reasoning(self):
        """
        Multi-contact scenario: 3 mine cylinders + 1 pipeline + 1 ghost net
        must be aggregated into a coherent multi-target operational summary.
        """
        fallback_engine = GroqSystem2Engine(api_key=None)
        contacts = [
            {"class": "ghost_net", "confidence": 0.92, "elevation_m": 1.7},
            {"class": "mine_cylinder", "confidence": 0.78, "elevation_m": 0.4},
            {"class": "mine_cylinder", "confidence": 0.65, "elevation_m": 0.3},
            {"class": "mine_cylinder", "confidence": 0.61, "elevation_m": 0.3},
            {"class": "submarine_pipeline", "confidence": 0.85, "elevation_m": 0.0}
        ]
        context = System2MissionContext(
            survey_id="SURVEY_MULTI_01",
            contacts=contacts,
            system1={"decision_primitive": "EMERGENCY_PROP_HAZARD", "hazard_score": 8.8}
        )
        analysis = fallback_engine.analyze_tactical(context)

        self.assertIsInstance(analysis, TacticalAnalysis)
        summary_text = analysis.incident_summary.lower()
        evidence_text = " ".join(analysis.observed_evidence).lower()

        # Must mention multi-target or total count
        self.assertTrue("5" in evidence_text or "multi-target" in summary_text or "contacts" in summary_text)

    def test_grounded_operator_qa(self):
        """
        Operator Q&A answers questions strictly using mission context,
        and refuses to hallucinate facts outside the context.
        """
        fallback_engine = GroqSystem2Engine(api_key=None)
        context = System2MissionContext(
            survey_id="SURVEY_QA_01",
            contacts=[{"class": "ghost_net", "confidence": 0.93, "elevation_m": 1.8}],
            system1={"decision_primitive": "EMERGENCY_PROP_HAZARD", "hazard_score": 8.5}
        )

        # 1. Grounded query
        resp_grounded = fallback_engine.query_operator("What hazard is threatening the vehicle?", context)
        ans_lower = resp_grounded.answer.lower()
        self.assertTrue("ghost net" in ans_lower or "ghost_net" in ans_lower, f"Expected ghost net contact in answer, got: {ans_lower}")

        # 2. Query for non-existent telemetry (salinity)
        resp_missing = fallback_engine.query_operator("What is the water salinity and temperature?", context)
        self.assertIsInstance(resp_missing, OperatorQueryResponse)
        self.assertIn("not available", resp_missing.answer.lower())


if __name__ == "__main__":
    unittest.main()
