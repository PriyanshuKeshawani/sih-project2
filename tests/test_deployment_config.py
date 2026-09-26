"""
tests/test_deployment_config.py
Deployment configuration tests (Phase 8 Section 11).
Verifies Dockerfile, render.yaml, .gitignore, .env.example, and secret protection.
"""

import unittest
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDeploymentConfig(unittest.TestCase):

    def test_gitignore_exists_and_protects_secrets(self):
        gitignore_path = os.path.join(BASE_DIR, ".gitignore")
        self.assertTrue(os.path.exists(gitignore_path), ".gitignore must exist")
        with open(gitignore_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn(".env", content)
        self.assertIn("*.safetensors", content)

    def test_env_example_has_clean_templates(self):
        env_ex_path = os.path.join(BASE_DIR, ".env.example")
        self.assertTrue(os.path.exists(env_ex_path), ".env.example must exist")
        with open(env_ex_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("GROQ_API_KEY", content)
        # Ensure no real groq key is in .env.example
        self.assertNotIn("gsk_", content)

    def test_dockerfile_exists_and_valid(self):
        dockerfile_path = os.path.join(BASE_DIR, "Dockerfile")
        self.assertTrue(os.path.exists(dockerfile_path), "Dockerfile must exist")
        with open(dockerfile_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("FROM python", content)
        self.assertIn("requirements.txt", content)
        self.assertIn("uvicorn", content)

    def test_render_yaml_valid(self):
        render_path = os.path.join(BASE_DIR, "render.yaml")
        self.assertTrue(os.path.exists(render_path), "render.yaml must exist")
        with open(render_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("samudra-sonar-ai", content)
        self.assertIn("uvicorn main:app", content)

    def test_requirements_contains_all_dependencies(self):
        req_path = os.path.join(BASE_DIR, "requirements.txt")
        self.assertTrue(os.path.exists(req_path), "requirements.txt must exist")
        with open(req_path, "r", encoding="utf-8") as f:
            content = f.read()
        required_pkgs = ["fastapi", "uvicorn", "onnxruntime", "reportlab", "laya", "groq", "python-dotenv"]
        for pkg in required_pkgs:
            self.assertIn(pkg, content, f"Package {pkg} missing in requirements.txt")


if __name__ == "__main__":
    unittest.main()
