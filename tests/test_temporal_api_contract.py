"""
Unit & Contract Tests: Temporal Persistence and Track Management API Endpoints.
SIH 2026 Problem Statement 26057.
"""
import unittest
from fastapi.testclient import TestClient
from main import app, temporal_tracker

class TestTemporalApiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_system1_status_api(self):
        """GET /api/system1/status returns active engine status."""
        resp = self.client.get("/api/system1/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn(data["status"], ["ACTIVE", "READY", "STANDBY"])

    def test_system2_status_api(self):
        """GET /api/system2/status returns queue and engine status."""
        resp = self.client.get("/api/system2/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("engine", data)

    def test_survey_observations_and_tracks_lifecycle(self):
        """
        Tests end-to-end lifecycle:
        1. Submitting observations to POST /api/surveys/{survey_id}/observations
        2. Querying tracks via GET /api/surveys/{survey_id}/tracks
        3. Querying specific track via GET /api/tracks/{track_id}
        """
        survey_id = "SURVEY_API_TEST_01"
        obs_payload = {
            "scan_id": "SCAN_001",
            "timestamp": "2026-09-26T12:00:00Z",
            "observations": [
                {
                    "class": "ghost_net",
                    "confidence": 0.88,
                    "box": {"x": 100, "y": 100, "w": 50, "h": 50},
                    "geo": {"lat": 9.2882, "lon": 79.1325, "depth_m": 22.0}
                }
            ]
        }

        # 1. Post observation
        resp1 = self.client.post(f"/api/surveys/{survey_id}/observations", json=obs_payload)
        self.assertEqual(resp1.status_code, 200)
        body1 = resp1.json()
        self.assertEqual(body1["survey_id"], survey_id)
        self.assertEqual(len(body1["enriched_observations"]), 1)

        det = body1["enriched_observations"][0]
        track_id = det["track_id"]
        self.assertTrue(track_id.startswith("TRK-"))
        self.assertEqual(det["persistence_status"], "NEW_CONTACT")
        self.assertEqual(det["observation_count"], 1)

        # 2. Get survey tracks
        resp2 = self.client.get(f"/api/surveys/{survey_id}/tracks")
        self.assertEqual(resp2.status_code, 200)
        body2 = resp2.json()
        self.assertEqual(body2["track_count"], 1)
        self.assertEqual(body2["tracks"][0]["track_id"], track_id)

        # 3. Get track detail
        resp3 = self.client.get(f"/api/tracks/{track_id}")
        self.assertEqual(resp3.status_code, 200)
        body3 = resp3.json()
        self.assertEqual(body3["track_id"], track_id)
        self.assertEqual(body3["class"], "ghost_net")
        self.assertEqual(body3["confidence_history"], [0.88])

    def test_nonexistent_track_returns_404(self):
        """Querying a non-existent track returns 404 Not Found."""
        resp = self.client.get("/api/tracks/TRK-NONEXISTENT")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("not found", resp.json()["error"].lower())

    def test_operator_track_confirmation_and_dismissal(self):
        """
        Tests operator control endpoints:
        - POST /api/tracks/{track_id}/confirm
        - POST /api/tracks/{track_id}/dismiss
        - POST /api/tracks/{track_id}/visibility
        Verifies audit trail logging.
        """
        survey_id = "SURVEY_API_TEST_02"
        obs_payload = {
            "scan_id": "SCAN_001",
            "observations": [
                {
                    "class": "shipwreck",
                    "confidence": 0.75,
                    "box": {"x": 200, "y": 200, "w": 80, "h": 60}
                }
            ]
        }
        resp = self.client.post(f"/api/surveys/{survey_id}/observations", json=obs_payload)
        track_id = resp.json()["enriched_observations"][0]["track_id"]

        # Confirm track
        confirm_resp = self.client.post(f"/api/tracks/{track_id}/confirm", json={
            "note": "Operator verified acoustic highlight matches wreck geometry."
        })
        self.assertEqual(confirm_resp.status_code, 200)
        data_c = confirm_resp.json()
        self.assertEqual(data_c["track"]["operator_status"], "CONFIRMED_FOR_MISSION")
        self.assertGreaterEqual(len(data_c["track"]["audit_log"]), 1)
        self.assertEqual(data_c["track"]["audit_log"][0]["action"], "CONFIRM_CONTACT")

        # Toggle visibility
        vis_resp = self.client.post(f"/api/tracks/{track_id}/visibility")
        self.assertEqual(vis_resp.status_code, 200)
        self.assertTrue(vis_resp.json()["hidden"])

        # Dismiss track
        dismiss_resp = self.client.post(f"/api/tracks/{track_id}/dismiss", json={
            "note": "Operator dismissed contact after high-resolution review."
        })
        self.assertEqual(dismiss_resp.status_code, 200)
        data_d = dismiss_resp.json()
        self.assertEqual(data_d["track"]["operator_status"], "DISMISSED")
        self.assertGreaterEqual(len(data_d["track"]["audit_log"]), 2)

    def test_system2_query_endpoint(self):
        """POST /api/system2/query answers operator forensic questions without error."""
        resp = self.client.post("/api/system2/query", json={
            "question": "What is the primary risk of this ghost net contact?",
            "mission_context": {
                "contacts": [{"class": "ghost_net", "confidence": 0.92, "elevation_m": 1.2}],
                "temporal": {"persistence_status": "PERSISTENT", "observation_count": 4}
            }
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertIsInstance(data["answer"], str)
        self.assertGreater(len(data["answer"]), 10)

if __name__ == "__main__":
    unittest.main()
