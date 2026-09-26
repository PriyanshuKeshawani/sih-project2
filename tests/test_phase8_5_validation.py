"""
tests/test_phase8_5_validation.py
Validates the corrected Phase 8.5 validation matrix, dataset manifests,
and truth-in-validation rules.
"""

import unittest
import os
import csv
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_CSV = os.path.join(BASE_DIR, "reports", "phase8_5", "validation_matrix.csv")
MANIFEST_JSON = os.path.join(BASE_DIR, "dataset_manifest.json")


class TestPhase85Validation(unittest.TestCase):

    def test_validation_matrix_exists_and_has_all_scenarios(self):
        self.assertTrue(os.path.exists(VALIDATION_CSV), f"Missing {VALIDATION_CSV}")
        with open(VALIDATION_CSV, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        self.assertGreaterEqual(len(reader), 7, "Validation matrix must contain at least 7 scenarios (A-G)")
        
        expected_cols = [
            "dataset", "image", "ground_truth_available", "target",
            "detected", "correct_class", "correct_bbox", "system1",
            "system2", "pdf", "overall_status"
        ]
        for col in expected_cols:
            self.assertIn(col, reader[0], f"Missing expected column: {col}")

    def test_scenario_a_and_b_are_pass(self):
        with open(VALIDATION_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        row_a = rows[0]
        self.assertEqual(row_a["target"], "clean_seabed")
        self.assertEqual(row_a["overall_status"], "PASS")

        row_b = rows[1]
        self.assertEqual(row_b["target"], "ghost_net")
        self.assertEqual(row_b["overall_status"], "PASS")

    def test_scenario_f_unknown_object_is_not_labeled_pass(self):
        with open(VALIDATION_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        row_f = next(r for r in rows if "noaa_fig2" in r["image"])
        self.assertNotEqual(
            row_f["overall_status"], "PASS",
            "Scenario F (natural geology) must NOT be labeled PASS because 5-class detector hallucinated classes."
        )
        self.assertIn(row_f["overall_status"], ["FAIL", "PARTIAL", "NOT_EVALUATED"])

    def test_absent_ground_truth_marked_not_evaluated(self):
        with open(VALIDATION_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        for r in rows:
            if r["ground_truth_available"] == "FALSE":
                self.assertIn(
                    "NOT_EVALUATED", r["correct_bbox"],
                    f"Row {r['target']} without GT must have correct_bbox = NOT_EVALUATED"
                )

    def test_dataset_manifest_categorization(self):
        self.assertTrue(os.path.exists(MANIFEST_JSON), f"Missing {MANIFEST_JSON}")
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.assertIn("categories_summary", data)
        categories = [d["validation_category"] for d in data["datasets"]]
        self.assertIn("A_DOWNLOADED_AND_TESTED", categories)
        self.assertIn("B_REFERENCED_NOT_TESTED", categories)
        self.assertIn("C_RESTRICTED_UNAVAILABLE", categories)


if __name__ == "__main__":
    unittest.main()
