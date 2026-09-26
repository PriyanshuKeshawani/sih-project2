"""
Unit & Contract Tests: Phase 7 UI HTML/CSS/JS Specification Integrity.
SIH 2026 Problem Statement 26057.
"""
import os
import unittest
from fastapi.testclient import TestClient
from main import app

class TestUIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.html_path = os.path.join(cls.base_dir, "static", "index.html")
        cls.css_path = os.path.join(cls.base_dir, "static", "style.css")
        cls.js_path = os.path.join(cls.base_dir, "static", "app.js")

        with open(cls.html_path, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

        with open(cls.css_path, "r", encoding="utf-8") as f:
            cls.css_content = f.read()

        with open(cls.js_path, "r", encoding="utf-8") as f:
            cls.js_content = f.read()

    def test_serve_index_endpoint(self):
        """Root endpoint '/' serves 200 OK with SAMUDRA-AI HTML."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("SAMUDRA-AI", resp.text)
        self.assertIn("v2.0 DUAL-PROCESS", resp.text)

    def test_demo_mode_badge_clearly_indicated(self):
        """Mandatory DEMO MODE indicator must be visible in HTML (Section 16)."""
        self.assertIn("DEMO / SIMULATION", self.html_content)
        self.assertIn("badge-demo", self.html_content)

    def test_detection_inspector_and_sparkline_containers(self):
        """Inspector panel, sparkline SVG container, and provenance section must exist."""
        self.assertIn("panel-inspector", self.html_content)
        self.assertIn("confidence-sparkline", self.html_content)
        self.assertIn("MEASUREMENT PROVENANCE BREAKDOWN", self.html_content)
        self.assertIn("prov-elevation", self.html_content)
        self.assertIn("prov-gps", self.html_content)

    def test_operator_track_controls_exist(self):
        """Operator action buttons for track confirmation, dismissal, and visibility exist."""
        self.assertIn("btn-confirm-contact", self.html_content)
        self.assertIn("btn-dismiss-contact", self.html_content)
        self.assertIn("btn-toggle-vis", self.html_content)
        self.assertIn("track-audit-log", self.html_content)

    def test_mission_timeline_panel_exists(self):
        """Mission event timeline container exists in HTML."""
        self.assertIn("panel-timeline", self.html_content)
        self.assertIn("timeline-list", self.html_content)

    def test_system1_advisory_disclaimer(self):
        """System 1 must clearly display advisory disclaimer."""
        self.assertIn("Advisory only — no direct vehicle control.", self.html_content)

    def test_mobile_responsive_stack_order(self):
        """
        CSS must declare exact mobile stack order (Section 13):
        1. sonar viewer (panel-sonar)
        2. selected detection (panel-inspector)
        3. System 1 (reflex-box)
        4. System 2 (system2-box)
        5. timeline (panel-timeline)
        6. map (map-box)
        7. specs table (dispatch-box)
        """
        self.assertIn(".panel-sonar { order: 1; }", self.css_content)
        self.assertIn(".panel-inspector { order: 2; }", self.css_content)
        self.assertIn(".reflex-box { order: 3; }", self.css_content)
        self.assertIn(".system2-box { order: 4; }", self.css_content)
        self.assertIn(".panel-timeline { order: 5; }", self.css_content)
        self.assertIn(".map-box { order: 6; }", self.css_content)
        self.assertIn(".dispatch-box { order: 7; }", self.css_content)
        self.assertIn("overflow-x: hidden", self.css_content)

    def test_cautious_terminology_in_app_js(self):
        """JS must enforce cautious terminology (e.g. SHIPWRECK-CLASS CONTACT, not CONFIRMED SHIPWRECK)."""
        self.assertIn("SHIPWRECK-CLASS CONTACT", self.js_content)
        self.assertNotIn("CONFIRMED SHIPWRECK", self.js_content)

    def test_persistence_badge_accessibility_icons(self):
        """JS must provide icons and distinct visual shapes so persistence does not rely solely on color."""
        self.assertIn("getPersistenceIcon", self.js_content)
        self.assertIn("🛡️", self.js_content) # PERSISTENT
        self.assertIn("⚡", self.js_content) # NEW_CONTACT
        self.assertIn("⏳", self.js_content) # TRANSIENT
        self.assertIn("❓", self.js_content) # UNCERTAIN

if __name__ == "__main__":
    unittest.main()
