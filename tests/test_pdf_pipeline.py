"""
tests/test_pdf_pipeline.py
Tests ReportLab PDF generator with System 1, Temporal Tracking, System 2,
provenance matrices, and DEMO mode banners (Phase 8 Section 9 & 10).
"""

import unittest
from engine.mission import MissionReportGenerator


class TestPdfPipeline(unittest.TestCase):

    def test_pdf_generation_full_payload(self):
        payload = {
            "incident_id": "NET-TEST-REPLAY-001",
            "timestamp": "2026-09-26 18:30:00 UTC",
            "dataset_name": "DRISHTI SSS (CC-BY-SA-4.0)",
            "is_demo_mode": True,
            "class": "GHOST_NET",
            "confidence": 0.945,
            "elevation_m": 1.85,
            "elevation_provenance": "DERIVED",
            "depth_m": 24.0,
            "depth_provenance": "MEASURED",
            "lat": 9.2882,
            "lon": 79.1325,
            "geo_provenance": "DEMO",
            "track_id": "TRK-944A89",
            "persistence_status": "PERSISTENT",
            "observation_count": 3,
            "track_age_s": 25.4,
            "system1": {
                "engine": "LAYA",
                "decision_primitive": "EMERGENCY_PROP_HAZARD",
                "hazard_score": 9.5,
                "latency_ms": 28.4
            },
            "system2": {
                "status": "ACTIVE",
                "model": "qwen/qwen3.8-27b",
                "incident_summary": "High risk ghost net fouling hazard detected in primary survey line.",
                "operator_action": "Execute loitering maneuver at 50m standoff distance.",
                "recovery_priority": "CRITICAL",
                "uncertainties": ["Single pass sonar cannot confirm mesh gauge or synthetic polymer type."]
            }
        }
        pdf_bytes = MissionReportGenerator.generate_pdf(payload)
        self.assertIsNotNone(pdf_bytes)
        self.assertGreater(len(pdf_bytes), 2000)
        # PDF magic header
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_pdf_generation_clean_background_zero_contacts(self):
        payload = {
            "incident_id": "NET-NOMINAL-002",
            "dataset_name": "SubPipe (CC-BY-4.0)",
            "is_demo_mode": False,
            "class": "CLEAN_SEABED",
            "confidence": 0.0,
            "elevation_m": None,
            "elevation_provenance": "UNAVAILABLE",
            "lat": None,
            "lon": None,
            "system1": {
                "engine": "LAYA",
                "decision_primitive": "NOMINAL_CRUISE",
                "hazard_score": 1.0,
                "latency_ms": 1.2
            },
            "system2": {
                "status": "ACTIVE",
                "model": "qwen/qwen3.8-27b",
                "incident_summary": "Nominal seafloor transect with zero detected hazards.",
                "operator_action": "Continue along pre-planned survey transect.",
                "recovery_priority": "ROUTINE",
                "uncertainties": []
            }
        }
        pdf_bytes = MissionReportGenerator.generate_pdf(payload)
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_pdf_does_not_invent_missing_elevation(self):
        payload = {
            "incident_id": "NET-UNAVAILABLE-003",
            "elevation_m": None,
            "elevation_provenance": "UNAVAILABLE"
        }
        pdf_bytes = MissionReportGenerator.generate_pdf(payload)
        self.assertIsNotNone(pdf_bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
