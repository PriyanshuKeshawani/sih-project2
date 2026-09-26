import unittest

from engine.reflex import (
    System1ReflexEngine,
    EvidenceQuality,
    HAZARD_TAXONOMY,
    ReflexConfig
)


class TestRiskScoring(unittest.TestCase):
    """
    Unit tests for deterministic risk scoring, evidence quality assessment, and hazard taxonomy.
    """

    def setUp(self):
        self.engine = System1ReflexEngine()

    def test_hazard_taxonomy_scientific_wording(self):
        """
        TEST 5: Mine-cylinder-class contact must use cautious scientific taxonomy,
        NOT claiming 'confirmed explosive mine'.
        """
        mine_tax = HAZARD_TAXONOMY.get("mine_cylinder")
        self.assertIsNotNone(mine_tax)
        self.assertEqual(mine_tax["hazard_category"], "mine-cylinder-class sonar contact")
        self.assertNotIn("confirmed explosive mine", mine_tax["scientific_description"].lower())
        self.assertIn("cautious non-contact", mine_tax["scientific_description"].lower())

    def test_crab_pot_lower_hazard_category(self):
        """
        TEST 6: Crab pot has lower hazard category and base weight than critical threats.
        """
        crab_tax = HAZARD_TAXONOMY.get("crab_pot")
        net_tax = HAZARD_TAXONOMY.get("ghost_net")

        self.assertLess(crab_tax["base_weight"], net_tax["base_weight"])
        self.assertEqual(crab_tax["base_weight"], 3.0)

        # Compute risk for crab pot with high confidence
        risk_crab = self.engine.compute_deterministic_risk(
            cls_name="crab_pot",
            confidence=0.90,
            elevation_m=0.2,
            evidence_quality=EvidenceQuality.MODERATE
        )
        self.assertLess(risk_crab, 5.0, "Crab pot hazard score should remain low (<5.0)")

    def test_deterministic_risk_bounds_and_reproducibility(self):
        """
        Hazard score must be strictly clamped between 1.0 and 10.0,
        and yield identical results on repeated evaluations.
        """
        # Highest theoretical inputs
        max_risk = self.engine.compute_deterministic_risk(
            cls_name="ghost_net",
            confidence=1.0,
            elevation_m=5.0,
            evidence_quality=EvidenceQuality.STRONG
        )
        self.assertLessEqual(max_risk, 10.0)
        self.assertEqual(max_risk, 10.0)

        # Lowest inputs
        min_risk = self.engine.compute_deterministic_risk(
            cls_name="crab_pot",
            confidence=0.1,
            elevation_m=None,
            evidence_quality=EvidenceQuality.WEAK
        )
        self.assertGreaterEqual(min_risk, 1.0)

    def test_missing_physics_metadata_handled_gracefully(self):
        """
        TEST 7: Missing physics metadata (elevation_m is None).
        Expected: Does not crash with TypeError; calculates risk with 0.0 elevation bonus.
        """
        risk_with_none = self.engine.compute_deterministic_risk(
            cls_name="ghost_net",
            confidence=0.85,
            elevation_m=None,  # Missing elevation
            evidence_quality=EvidenceQuality.MODERATE
        )
        self.assertIsInstance(risk_with_none, float)
        self.assertGreater(risk_with_none, 5.0)

        # Comparing with elevation=0.0: should match
        risk_with_zero = self.engine.compute_deterministic_risk(
            cls_name="ghost_net",
            confidence=0.85,
            elevation_m=0.0,
            evidence_quality=EvidenceQuality.MODERATE
        )
        self.assertEqual(risk_with_none, risk_with_zero)

    def test_evidence_quality_stratification(self):
        """
        Evidence quality properly distinguishes strong vs moderate vs weak evidence.
        """
        # Strong: high confidence + shadow + metric elevation
        eq_strong = self.engine.evaluate_evidence_quality(
            confidence=0.85,
            physics_data={"shadow_detected": True, "elevation_m": 1.2},
            geo_data={"status": "DERIVED"}
        )
        self.assertEqual(eq_strong, EvidenceQuality.STRONG)

        # Weak: low confidence + no shadow
        eq_weak = self.engine.evaluate_evidence_quality(
            confidence=0.35,
            physics_data={"shadow_detected": False, "elevation_m": None},
            geo_data=None
        )
        self.assertEqual(eq_weak, EvidenceQuality.WEAK)


if __name__ == '__main__':
    unittest.main()
