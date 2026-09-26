"""
tests/test_health_endpoint.py
Contract tests for GET /api/health and GET /api/system/status (Phase 8 Section 12 & 13).
"""

import unittest
from fastapi.testclient import TestClient
from main import app


class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_startup_health_returns_required_contract(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("detector", data)
        self.assertIn("system1", data)
        self.assertIn("system2", data)
        self.assertIn("temporal_tracking", data)
        self.assertEqual(data.get("version"), "2.0.0")

    def test_startup_health_does_not_expose_secrets(self):
        res = self.client.get("/api/health")
        raw_text = res.text
        self.assertNotIn("gsk_", raw_text)
        self.assertNotIn("API_KEY", raw_text)
        self.assertNotIn("secret", raw_text.lower())

    def test_system_status_returns_verified_states(self):
        res = self.client.get("/api/system/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        valid_states = {"ACTIVE", "FALLBACK", "UNAVAILABLE", "ERROR"}
        
        required_keys = ["detector", "physics", "laya", "system2", "groq", "temporal", "pdf"]
        for key in required_keys:
            self.assertIn(key, data, f"Key {key} must be present in /api/system/status")
            self.assertIn(data[key], valid_states, f"State for {key} must be one of {valid_states}")


if __name__ == "__main__":
    unittest.main()
