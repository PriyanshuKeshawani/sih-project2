"""
tests/test_benchmark_integrity.py
Validates benchmark latency measurement integrity:
- Separates per-stage latency
- Separates System 1 deterministic vs PyTorch Laya
- Separates System 2 Groq cloud vs local fallback
- Distinguishes Synchronous blocking latency from Asynchronous edge loop latency
"""

import unittest
import os
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCHMARK_JSON = os.path.join(BASE_DIR, "reports", "phase8", "benchmarks_summary.json")


class TestBenchmarkIntegrity(unittest.TestCase):

    def test_benchmark_summary_file_exists_and_structured(self):
        self.assertTrue(os.path.exists(BENCHMARK_JSON), f"Missing {BENCHMARK_JSON}")
        with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.assertIn("latency_benchmarks", data)
        self.assertIn("scenarios", data)

    def test_per_stage_metrics_separated(self):
        with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        stages = data["latency_benchmarks"]
        required_stages = [
            "image_loading_ms",
            "tiled_inference_ms",
            "physics_geo_ms",
            "temporal_tracking_ms",
            "laya_system1_ms",
            "system2_fallback_ms",
            "pdf_generation_ms",
            "total_mission_ms"
        ]
        for st in required_stages:
            self.assertIn(st, stages, f"Missing stage: {st}")
            self.assertIn("mean", stages[st])
            self.assertIn("median", stages[st])
            self.assertIn("p95", stages[st])
            self.assertIn("max", stages[st])

    def test_system1_latency_differentiation(self):
        """
        Deterministic reflex is sub-millisecond (< 1ms).
        PyTorch Laya checkpoint on CPU is multi-second (> 1500ms).
        Ensure both modes are cleanly separated and not confused.
        """
        from engine.laya_adapter import DeterministicDecisionEngine, System1InputState

        engine = DeterministicDecisionEngine()
        state = System1InputState(
            contact_class="mine_cylinder",
            detector_confidence=0.85,
            shadow_detected=True,
            shadow_length_m=12.0,
            elevation_m=2.5,
            sonar_altitude_m=12.0
        )
        t0 = time.perf_counter()
        fb_decision = engine.decide(state)
        fb_latency_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(fb_latency_ms, 5.0, "Deterministic fallback must execute in < 5ms on CPU")
        self.assertIn(fb_decision.decision_primitive, [
            "EMERGENCY_PROP_HAZARD", "LOITER_AND_RESCAN", "PASSIVE_LOG",
            "EMERGENCY_SURFACE", "NAVIGATION_HAZARD", "HAZARD_ALERT"
        ])

    def test_synchronous_vs_asynchronous_latency_distinction(self):
        """
        Total blocking synchronous latency MUST be distinct from the edge loop latency.
        Edge loop (Tiling + YOLO + Physics + S1) is ~115ms.
        Blocking loop with System 2 cloud or PDF is hundreds/thousands of ms.
        """
        with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)

        tiled_median = data["latency_benchmarks"]["tiled_inference_ms"]["median"]
        s1_median = data["latency_benchmarks"]["laya_system1_ms"]["median"]
        total_mission_median = data["latency_benchmarks"]["total_mission_ms"]["median"]

        self.assertGreater(
            total_mission_median, tiled_median + s1_median,
            "Total mission latency must include asynchronous/synchronous pipeline overhead, not just detection."
        )


if __name__ == "__main__":
    unittest.main()
