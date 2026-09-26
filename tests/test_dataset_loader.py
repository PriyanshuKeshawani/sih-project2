"""
tests/test_dataset_loader.py
Unit & contract tests for engine/dataset_loader.py (Phase 8).
Validates dataset manifest loading, provenance tracking, scenario mapping,
and missing file handling without fabrication.
"""

import unittest
import os
from engine.dataset_loader import SonarDatasetLoader, SonarSampleRecord


class TestDatasetLoader(unittest.TestCase):

    def setUp(self):
        self.loader = SonarDatasetLoader()

    def test_manifest_loads_all_datasets(self):
        datasets = self.loader.get_datasets()
        self.assertGreaterEqual(len(datasets), 3)
        sources = [d["source"] for d in datasets]
        self.assertTrue(any("DRISHTI SSS" in s for s in sources))
        self.assertTrue(any("SubPipe" in s for s in sources))
        self.assertTrue(any("AI4Shipwrecks" in s for s in sources))

    def test_scenarios_have_full_provenance_and_license(self):
        catalog = self.loader.get_scenario_catalog()
        self.assertIn("Scenario A", catalog)
        self.assertIn("Scenario B", catalog)
        self.assertIn("Scenario C", catalog)
        self.assertIn("Scenario D", catalog)
        self.assertIn("Scenario E", catalog)
        self.assertIn("Scenario F", catalog)
        self.assertIn("Scenario G", catalog)

        for s_id, s_spec in catalog.items():
            self.assertTrue(bool(s_spec.get("dataset_name")), f"{s_id} missing dataset_name")
            self.assertTrue(bool(s_spec.get("license")), f"{s_id} missing license")
            self.assertTrue(bool(s_spec.get("provenance")), f"{s_id} missing provenance")
            self.assertTrue(bool(s_spec.get("source_url")), f"{s_id} missing source_url")

    def test_scenario_a_loads_clean_seabed_image(self):
        img, record = self.loader.load_scenario("Scenario A")
        self.assertIsNotNone(img, "Scenario A background image must load")
        self.assertEqual(len(img.shape), 3)
        self.assertEqual(record.target_class, "clean_seabed")
        self.assertIn("SubPipe", record.dataset_name)

    def test_scenario_b_loads_ghost_net(self):
        img, record = self.loader.load_scenario("Scenario B")
        self.assertIsNotNone(img, "Scenario B ghost net must load")
        self.assertEqual(record.target_class, "ghost_net")
        self.assertIn("DRISHTI", record.dataset_name)

    def test_nonexistent_scenario_marked_not_available(self):
        img, record = self.loader.load_scenario("Scenario Z_NONEXISTENT")
        self.assertIsNone(img)
        self.assertEqual(record.dataset_name, "NOT AVAILABLE")
        self.assertFalse(record.annotations_available)

    def test_list_all_samples_records_metadata(self):
        samples = self.loader.list_all_samples()
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertTrue(os.path.exists(s.image_path), f"File {s.image_path} must exist on disk")
            self.assertNotEqual(s.license, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
