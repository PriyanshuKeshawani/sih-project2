"""
tests/test_deployment_runtime.py
Validates deployment configuration, runtime constraints, health endpoints,
and environment isolation for Phase 8.5.
"""

import unittest
import os
from fastapi.testclient import TestClient
from main import app

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDeploymentRuntime(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_dockerfile_contains_opencv_dependencies(self):
        dockerfile = os.path.join(BASE_DIR, "Dockerfile")
        self.assertTrue(os.path.exists(dockerfile), f"Missing {dockerfile}")
        with open(dockerfile, "r", encoding="utf-8") as f:
            content = f.read()
        
        self.assertIn("libgl1", content, "Dockerfile must install libgl1 for headless OpenCV.")
        self.assertIn("libglib2.0-0", content, "Dockerfile must install libglib2.0-0.")
        self.assertIn("PORT", content, "Dockerfile must support dynamic PORT assignment.")

    def test_render_yaml_documents_memory_tier_constraints(self):
        render_yaml = os.path.join(BASE_DIR, "render.yaml")
        self.assertTrue(os.path.exists(render_yaml), f"Missing {render_yaml}")
        with open(render_yaml, "r", encoding="utf-8") as f:
            content = f.read()
        
        self.assertIn("plan:", content)
        self.assertIn("512MB", content, "render.yaml must document 512MB free tier memory constraints.")

    def test_env_example_contains_no_secrets(self):
        env_example = os.path.join(BASE_DIR, ".env.example")
        self.assertTrue(os.path.exists(env_example), f"Missing {env_example}")
        with open(env_example, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Verify placeholder values only
        self.assertTrue(
            "your_groq_api_key_here" in content or "gsk_your_groq_api_key_here" in content,
            "Placeholder API key expected in .env.example"
        )
        self.assertNotIn("gsk_xW", content, "Real Groq API keys must NEVER be present in .env.example")

    def test_gitignore_protects_env_and_safetensors(self):
        gitignore = os.path.join(BASE_DIR, ".gitignore")
        self.assertTrue(os.path.exists(gitignore), f"Missing {gitignore}")
        with open(gitignore, "r", encoding="utf-8") as f:
            content = f.read()
        
        self.assertIn(".env", content)
        self.assertIn("*.safetensors", content, ".gitignore must ignore large safetensors files (>100MB).")

    def test_health_endpoint_response_structure(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("detector", data)
        self.assertIn("system1", data)
        self.assertIn("system2", data)
        self.assertIn("version", data)

    def test_system_status_endpoint_returns_verified_enums(self):
        res = self.client.get("/api/system/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        valid_states = {"ACTIVE", "FALLBACK", "UNAVAILABLE", "ERROR"}
        
        for comp in ["detector", "physics", "laya", "system2", "groq", "temporal", "pdf"]:
            self.assertIn(comp, data, f"Missing component in /api/system/status: {comp}")
            self.assertIn(data[comp], valid_states, f"Invalid state enum for {comp}: {data[comp]}")


if __name__ == "__main__":
    unittest.main()
