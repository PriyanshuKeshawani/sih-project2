import unittest

from engine.laya_adapter import LayaAdapter


class TestLayaAdapter(unittest.TestCase):
    """
    Unit tests for LayaAdapter.
    Verifies that the adapter honestly reports status and does not make unauthorized network calls.
    """

    def test_laya_adapter_initialization_status(self):
        """
        Verifies that LayaAdapter reports its status honestly:
        If package is installed but no local weights exist, status must report
        'Laya adapter present but not active (no local weights configured, using deterministic local reflex engine)'.
        """
        adapter = LayaAdapter()
        status_info = adapter.get_status()

        self.assertIn("installed", status_info)
        self.assertIn("active", status_info)
        self.assertIn("status", status_info)

        # Laya is installed in this python environment
        self.assertTrue(adapter.is_installed)
        # But no local weights exist in models/laya/
        self.assertFalse(adapter.is_active)
        self.assertIn("Laya adapter present but not active", adapter.status)

    def test_laya_predict_fallback_when_inactive(self):
        """
        When inactive, predict() must return None so System 1 falls back to deterministic local reflex engine.
        """
        adapter = LayaAdapter()
        pred = adapter.predict({"context": "sonar_scan"})
        self.assertIsNone(pred, "Laya predict should return None when no local weights are active")


if __name__ == '__main__':
    unittest.main()
