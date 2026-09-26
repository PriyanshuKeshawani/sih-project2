"""
tests/test_system2_fallback.py
Unit tests for System 2 Deterministic Fallback Engine when Groq is unavailable,
network is offline, or API timeouts occur.
"""

import unittest
from unittest.mock import MagicMock, patch
import time

from engine.system2 import (
    GroqSystem2Engine,
    DeterministicSystem2Fallback,
    System2MissionContext,
    TacticalAnalysis,
    OperatorQueryResponse
)

class TestSystem2Fallback(unittest.TestCase):
    """
    Validates transparent, safe fallback execution of System 2.
    """

    def setUp(self):
        self.context = System2MissionContext(
            survey_id="SURVEY_FALLBACK_TEST",
            contacts=[
                {
                    "class": "ghost_net",
                    "confidence": 0.94,
                    "elevation_m": 1.85,
                    "box": {"x": 100, "y": 100, "w": 80, "h": 60}
                }
            ],
            geo={"status": "UNAVAILABLE", "latitude": None, "longitude": None},
            system1={
                "decision_primitive": "EMERGENCY_PROP_HAZARD",
                "hazard_score": 8.5,
                "evidence_quality": "STRONG"
            }
        )

    def test_missing_api_key_activates_fallback(self):
        """Engine with no API key must report not available and route to fallback."""
        import os
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "GEMINI_API_KEY": ""}):
            engine = GroqSystem2Engine(api_key=None)
            self.assertFalse(engine.is_available())

            analysis = engine.analyze_tactical(self.context)
            self.assertIsInstance(analysis, TacticalAnalysis)
            self.assertEqual(analysis.status, "FALLBACK")
            self.assertEqual(analysis.model, "deterministic_system2_fallback")
            self.assertIn("ghost net", analysis.incident_summary.lower())
            self.assertEqual(analysis.recovery_priority, "CRITICAL")
            self.assertTrue(len(analysis.observed_evidence) > 0)
            self.assertTrue(len(analysis.uncertainties) > 0)

    def test_api_network_exception_falls_back_without_crashing(self):
        """If Groq raises a network timeout or connection error, engine falls back cleanly."""
        import os
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            engine = GroqSystem2Engine(api_key="gsk_mock_test_key_12345")
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = TimeoutError("Groq API connection timed out")
            engine.client = mock_client
            engine.gemini_client = None

            analysis = engine.analyze_tactical(self.context)
            self.assertIsInstance(analysis, TacticalAnalysis)
            self.assertEqual(analysis.status, "FALLBACK")
            self.assertEqual(analysis.model, "deterministic_system2_fallback")
            self.assertIn("ghost net", analysis.incident_summary.lower())

    def test_fallback_latency_is_sub_millisecond(self):
        """Deterministic fallback must execute in less than 5 milliseconds."""
        t0 = time.perf_counter()
        analysis = DeterministicSystem2Fallback.analyze(self.context)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(elapsed_ms, 5.0, f"Fallback took {elapsed_ms}ms, expected < 5ms")
        self.assertIsInstance(analysis, TacticalAnalysis)

    def test_fallback_query_unsupported_data_rejection(self):
        """Operator queries for unrecorded telemetry (e.g. salinity) must be explicitly noted as missing."""
        resp = DeterministicSystem2Fallback.query("What was the water salinity in this sector?", self.context)
        self.assertIsInstance(resp, OperatorQueryResponse)
        self.assertIn("salinity", resp.answer.lower())
        self.assertIn("not available", resp.answer.lower())

    def test_clean_seabed_fallback_analysis(self):
        """Empty contacts list yields nominal seabed analysis."""
        clean_context = System2MissionContext(survey_id="SURVEY_CLEAR", contacts=[])
        analysis = DeterministicSystem2Fallback.analyze(clean_context)

        self.assertEqual(analysis.recovery_priority, "ROUTINE")
        self.assertIn("nominal", analysis.incident_summary.lower())
        self.assertEqual(analysis.status, "FALLBACK")


if __name__ == "__main__":
    unittest.main()
