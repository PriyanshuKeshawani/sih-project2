"""
scripts/verify_phase6_system2.py
End-to-End Verification and Benchmarking of System 2 Tactical Reasoning Engine (Groq).

SIH 2026 Problem Statement 26057 — MoES / NIOT Chennai
Evaluates System 2 execution across realistic sonar contact scenarios,
measuring queue time, request time, parsing time, and end-to-end latency.
"""

import os
import sys
import time
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.system2 import System2Queue, System2MissionContext, TacticalAnalysis, OperatorQueryResponse

def run_phase6_verification():
    print("=" * 75)
    print("PHASE 6: SYSTEM 2 TACTICAL REASONING ENGINE (GROQ) VERIFICATION")
    print("=" * 75)

    queue = System2Queue(max_capacity=50)
    status = queue.get_status()
    print(f"System 2 Engine:  {status['engine']} (Status: {status['status']})")
    print(f"Configured Model: {status['model']}")
    print(f"API Key Present:  {status['api_key_configured']}")
    print(f"Worker Alive:     {status['worker_alive']}")
    print("-" * 75)

    test_scenarios = [
        {
            "name": "Ghost Net (Critical Propeller Hazard)",
            "context": System2MissionContext(
                survey_id="SURVEY_NET_001",
                contacts=[
                    {"class": "ghost_net", "confidence": 0.935, "elevation_m": 1.8, "box": {"x": 120, "y": 140, "w": 65, "h": 50}}
                ],
                geo={"status": "AVAILABLE", "latitude": 9.2882, "longitude": 79.1325, "depth_m": 24.5, "zone": "Gulf of Mannar Sector 4"},
                system1={"decision_primitive": "EMERGENCY_PROP_HAZARD", "hazard_score": 8.5, "evidence_quality": "STRONG"},
                metadata={"sonar_altitude_m": 12.0, "swath_width_m": 50.0}
            )
        },
        {
            "name": "Shipwreck Closed-Set Contact (Surrogate Anomaly)",
            "context": System2MissionContext(
                survey_id="SURVEY_WRECK_002",
                contacts=[
                    {"class": "shipwreck", "confidence": 0.812, "elevation_m": 4.6, "box": {"x": 300, "y": 250, "w": 180, "h": 90}}
                ],
                geo={"status": "UNAVAILABLE", "latitude": None, "longitude": None, "depth_m": 24.5},
                system1={"decision_primitive": "PASSIVE_LOG", "hazard_score": 5.6, "evidence_quality": "STRONG"},
                metadata={"sonar_altitude_m": 12.0, "swath_width_m": 50.0}
            )
        },
        {
            "name": "Multiple Mine Cylinders (Ordnance Standoff)",
            "context": System2MissionContext(
                survey_id="SURVEY_MINE_CLUSTER_003",
                contacts=[
                    {"class": "mine_cylinder", "confidence": 0.78, "elevation_m": 0.52},
                    {"class": "mine_cylinder", "confidence": 0.65, "elevation_m": 0.40},
                    {"class": "mine_cylinder", "confidence": 0.58, "elevation_m": 0.35}
                ],
                geo={"status": "AVAILABLE", "latitude": 9.2915, "longitude": 79.1402, "depth_m": 22.0, "zone": "Palk Strait Approach"},
                system1={"decision_primitive": "LOITER_AND_RESCAN", "hazard_score": 7.2, "evidence_quality": "STRONG"},
                metadata={"sonar_altitude_m": 10.0, "swath_width_m": 45.0}
            )
        },
        {
            "name": "Submarine Pipeline (Linear Infrastructure)",
            "context": System2MissionContext(
                survey_id="SURVEY_PIPE_004",
                contacts=[
                    {"class": "submarine_pipeline", "confidence": 0.88, "elevation_m": 0.20}
                ],
                geo={"status": "AVAILABLE", "latitude": 9.2750, "longitude": 79.1100, "depth_m": 28.0, "zone": "Rameswaram Corridor"},
                system1={"decision_primitive": "PASSIVE_LOG", "hazard_score": 5.2, "evidence_quality": "STRONG"},
                metadata={"sonar_altitude_m": 12.0, "swath_width_m": 50.0}
            )
        },
        {
            "name": "Nominal Seafloor (Zero Contacts)",
            "context": System2MissionContext(
                survey_id="SURVEY_CLEAR_005",
                contacts=[],
                geo={"status": "UNAVAILABLE"},
                system1={"decision_primitive": "NOMINAL_CRUISE", "hazard_score": 1.0, "evidence_quality": "STRONG"},
                metadata={"sonar_altitude_m": 12.0, "swath_width_m": 50.0}
            )
        }
    ]

    print("\nExecuting Tactical Analyses across 5 Scenarios...")
    for sc in test_scenarios:
        t0 = time.perf_counter()
        analysis = queue.analyze_sync(sc["context"])
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        print(f"\n[SCENARIO: {sc['name']}]")
        print(f"  Incident Summary:      {analysis.incident_summary}")
        print(f"  Recovery Priority:     {analysis.recovery_priority}")
        print(f"  Operator Action:       {analysis.operator_action}")
        print(f"  Observed Evidence:     {len(analysis.observed_evidence)} items verified")
        print(f"  Active Uncertainties:  {len(analysis.uncertainties)} logged")
        print(f"  Engine / Status:       {analysis.model} ({analysis.status})")
        print(f"  Execution Latency:     {total_time_ms:.2f} ms")

    # Interactive Q&A Verification
    print("\n" + "-" * 75)
    print("OPERATOR INTERACTIVE FORENSIC Q&A VERIFICATION:")
    print("-" * 75)

    test_queries = [
        ("What hazard was detected on the seabed?", test_scenarios[0]["context"]),
        ("Is the shipwreck classification verified?", test_scenarios[1]["context"]),
        ("What is the water temperature and ocean current?", test_scenarios[0]["context"])
    ]

    for q, ctx in test_queries:
        t0 = time.perf_counter()
        resp = queue.query_sync(q, ctx)
        q_time_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[Q]: {q}")
        print(f" [A]: {resp.answer}")
        print(f" [Evidence Used]: {resp.evidence_used}")
        print(f" [Latency]:       {q_time_ms:.2f} ms ({resp.status})")

    queue.shutdown()
    print("\n" + "=" * 75)
    print("PHASE 6 SYSTEM 2 VERIFICATION COMPLETE.")
    print("=" * 75)

if __name__ == "__main__":
    run_phase6_verification()
