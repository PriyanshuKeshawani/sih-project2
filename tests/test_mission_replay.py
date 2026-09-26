"""
tests/test_mission_replay.py
End-to-end tests for MissionReplayEngine (Phase 8 Section 3, 4, 5, 6).
"""

import unittest
import os
import json
from engine.mission_replay import MissionReplayEngine, REPLAYS_DIR


class TestMissionReplay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = MissionReplayEngine()

    def test_scenario_a_clean_background_replay(self):
        res = self.engine.run_scenario("Scenario A", generate_pdf=True)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertEqual(res["dataset_name"], "SubPipe / DRISHTI SSS")
        self.assertEqual(len(res["detections"]), 0, "Scenario A background must have zero detections")
        self.assertEqual(res["system1"]["decision_primitive"], "PASSIVE_LOG")
        self.assertTrue(res["has_pdf"])
        self.assertIn("total_mission_ms", res["timings"])
        
        # Verify JSON record was saved on disk
        saved_file = os.path.join(REPLAYS_DIR, f"{res['replay_id']}.json")
        self.assertTrue(os.path.exists(saved_file), f"Saved replay {saved_file} must exist")
        with open(saved_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["replay_id"], res["replay_id"])

    def test_scenario_b_ghost_net_replay(self):
        res = self.engine.run_scenario("Scenario B", generate_pdf=True)
        self.assertEqual(res["status"], "COMPLETED")
        self.assertGreater(len(res["detections"]), 0, "Scenario B must detect ghost net")
        det = res["detections"][0]
        self.assertEqual(det["class"], "ghost_net")
        self.assertEqual(res["system1"]["decision_primitive"], "EMERGENCY_PROP_HAZARD")
        self.assertTrue(res["has_pdf"])
        self.assertIn("system2", res)

    def test_multi_ping_persistence_transition(self):
        multi_res = self.engine.run_multi_ping_replay("Scenario B", num_pings=3)
        self.assertEqual(multi_res.get("total_pings"), 3)
        history = multi_res.get("ping_history", [])
        self.assertEqual(len(history), 3)

        # Ping 1: single observation -> must be NEW_CONTACT (not PERSISTENT)
        obs_ping1 = history[0]["observations"]
        self.assertGreater(len(obs_ping1), 0)
        self.assertEqual(obs_ping1[0]["persistence_status"], "NEW_CONTACT")
        self.assertEqual(obs_ping1[0]["observation_count"], 1)

        # Ping 3: 3 observations -> must be PERSISTENT
        obs_ping3 = history[2]["observations"]
        self.assertGreater(len(obs_ping3), 0)
        self.assertEqual(obs_ping3[0]["persistence_status"], "PERSISTENT")
        self.assertGreaterEqual(obs_ping3[0]["observation_count"], 3)

    def test_unavailable_scenario_returns_not_available(self):
        res = self.engine.run_scenario("Scenario Z_UNKNOWN")
        self.assertEqual(res["status"], "NOT AVAILABLE")
        self.assertEqual(res["dataset_name"], "NOT AVAILABLE")


if __name__ == "__main__":
    unittest.main()
