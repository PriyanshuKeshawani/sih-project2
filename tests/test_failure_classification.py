"""
tests/test_failure_classification.py
Validates the completeness and rigor of the Phase 8.5 failure matrix.
"""

import unittest
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURE_CSV = os.path.join(BASE_DIR, "reports", "phase8_5", "failure_matrix.csv")


class TestFailureClassification(unittest.TestCase):

    def test_failure_matrix_structure_and_columns(self):
        self.assertTrue(os.path.exists(FAILURE_CSV), f"Missing {FAILURE_CSV}")
        with open(FAILURE_CSV, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        self.assertGreaterEqual(len(reader), 5, "Failure matrix must document all key failure modes.")
        required_cols = ["component", "scenario", "expected", "actual", "status", "root_cause", "severity", "next_action"]
        for col in required_cols:
            self.assertIn(col, reader[0], f"Missing required column: {col}")

    def test_open_set_rejection_failure_documented(self):
        with open(FAILURE_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        open_set_row = next((r for r in rows if "OpenSet" in r["component"] or "Scenario F" in r["scenario"]), None)
        self.assertIsNotNone(open_set_row, "Failure matrix must document open-set rejection failure.")
        self.assertEqual(open_set_row["status"], "FAIL")
        self.assertIn("closed-set", open_set_row["root_cause"].lower())

    def test_laya_runtime_resource_limitation_documented(self):
        with open(FAILURE_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        laya_row = next((r for r in rows if "Laya" in r["component"] or "Deployment_S1" in r["scenario"]), None)
        self.assertIsNotNone(laya_row, "Failure matrix must document Laya neural model runtime limitations.")
        self.assertIn("804mb", laya_row["root_cause"].lower())
        self.assertIn("deterministic", laya_row["next_action"].lower())

    def test_groq_rate_limit_mitigation_documented(self):
        with open(FAILURE_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        
        groq_row = next((r for r in rows if "Groq" in r["component"] or "System2_Groq" in r["scenario"]), None)
        self.assertIsNotNone(groq_row, "Failure matrix must document Groq cloud rate-limiting behavior.")
        self.assertEqual(groq_row["status"], "MITIGATED")
        self.assertIn("429", groq_row["actual"])


if __name__ == "__main__":
    unittest.main()
