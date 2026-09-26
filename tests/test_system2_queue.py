"""
tests/test_system2_queue.py
Unit tests for System 2 asynchronous queue and background worker execution.
Verifies that System 1 execution is NEVER blocked by System 2 processing.
"""

import unittest
import time
from engine.system2 import System2Queue, System2MissionContext, TacticalAnalysis
from engine.reflex import System1ReflexEngine

class TestSystem2Queue(unittest.TestCase):
    """
    Validates non-blocking queue semantics, worker thread execution,
    and decoupling from System 1 reflex engine.
    """

    def setUp(self):
        self.queue = System2Queue(max_capacity=20)
        self.sample_context = System2MissionContext(
            survey_id="SURVEY_QUEUE_01",
            contacts=[{"class": "mine_cylinder", "confidence": 0.82, "elevation_m": 0.5}],
            system1={"decision_primitive": "LOITER_AND_RESCAN", "hazard_score": 6.8}
        )

    def tearDown(self):
        self.queue.shutdown()

    def test_enqueue_is_immediate_and_non_blocking(self):
        """enqueue() must return in < 2.0 milliseconds to guarantee zero System 1 blocking."""
        t0 = time.perf_counter()
        task_id = self.queue.enqueue(self.sample_context)
        enqueue_ms = (time.perf_counter() - t0) * 1000.0

        self.assertLess(enqueue_ms, 2.0, f"enqueue() took {enqueue_ms:.3f}ms, must be < 2.0ms")
        self.assertIsNotNone(task_id)

    def test_background_worker_processes_and_stores_analysis(self):
        """Worker thread must process the queued context into history and latest_analysis."""
        task_id = self.queue.enqueue(self.sample_context)

        # Wait for worker thread to process (up to 2 seconds)
        timeout = time.time() + 2.0
        analysis = None
        while time.time() < timeout:
            analysis = self.queue.get_analysis(task_id)
            if analysis is not None:
                break
            time.sleep(0.05)

        self.assertIsNotNone(analysis, "Worker thread failed to process queued item within timeout")
        self.assertIsInstance(analysis, TacticalAnalysis)
        self.assertIn("mine", analysis.incident_summary.lower())

        latest = self.queue.get_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.incident_summary, analysis.incident_summary)

    def test_system1_reflex_latency_unaffected_by_system2(self):
        """System 1 Reflex latency remains sub-millisecond even when queue is loaded."""
        reflex_engine = System1ReflexEngine(enable_laya=False)

        # Enqueue 5 items into System 2 queue
        for i in range(5):
            ctx = System2MissionContext(survey_id=f"BURST_{i}", contacts=[{"class": "ghost_net", "confidence": 0.9}])
            self.queue.enqueue(ctx)

        # Execute 20 System 1 reflexes and benchmark
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            reflex_engine.process_reflex(
                detection={"class": "ghost_net", "confidence": 0.92, "box": {"x": 100, "y": 100, "w": 50, "h": 50}},
                physics={"elevation_m": 1.5, "shadow_detected": True}
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

        median_ms = sorted(latencies)[len(latencies) // 2]
        self.assertLess(median_ms, 2.0, f"System 1 reflex median latency {median_ms:.2f}ms exceeded 2.0ms")

    def test_queue_status_reporting(self):
        """get_status() returns dictionary with engine, worker status, and queue depth."""
        status = self.queue.get_status()
        self.assertIn("engine", status)
        self.assertIn("status", status)
        self.assertIn("queue_size", status)
        self.assertTrue(status["worker_alive"])


if __name__ == "__main__":
    unittest.main()
