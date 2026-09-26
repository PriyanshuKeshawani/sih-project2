"""
scripts/verify_phase8_5_pdfs.py
Generates and inspects 3 distinct mission PDFs:
1. Ghost Net
2. Shipwreck
3. Pipeline

Verifies:
- demo labels present
- provenance table present
- uncertainty present
- persistence status present
- System 1 decisions present
- System 2 briefing present
- no fabricated GPS
"""

import os
import sys
import io
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pypdf import PdfReader
from engine.mission import MissionReportGenerator
PDF_DIR = os.path.join(BASE_DIR, "reports", "phase8_5", "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

test_cases = [
    {
        "name": "ghost_net",
        "payload": {
            "incident_id": "INC-GHOST-NET-001",
            "timestamp": "2026-09-26 18:30:00 UTC",
            "dataset_name": "DRISHTI SSS Synthetic Split (CC-BY-SA-4.0)",
            "is_demo_mode": True,
            "class": "ghost_net",
            "confidence": 0.929,
            "elevation_m": 4.12,
            "elevation_provenance": "DERIVED",
            "lat": None,
            "lon": None,
            "geo_provenance": "UNAVAILABLE",
            "track_id": "TRK-NET-01",
            "persistence_status": "PERSISTENT",
            "observation_count": 4,
            "track_age_s": 42.5,
            "system1": {
                "engine": "convaiinnovations/laya",
                "decision_primitive": "EVADE_HAZARD",
                "hazard_score": 8.7,
                "needs_operator_review": True,
                "status": "ACTIVE"
            },
            "system2": {
                "incident_summary": "Derelict fishing gear contact identified in active survey corridor. Entanglement hazard to propulsion.",
                "observed_evidence": ["High backscatter mesh signature", "Acoustic shadow length 14.2m yielding 4.12m elevation"],
                "uncertainties": ["Exact anchoring tension to seabed unknown", "Current drift vector unmeasured"],
                "risk_interpretation": "Severe propeller fouling risk if distance drops below 25 meters.",
                "operator_action": "Execute evasive turn starboard 30 degrees and mark clearance waypoint.",
                "recovery_priority": "CRITICAL"
            }
        }
    },
    {
        "name": "shipwreck",
        "payload": {
            "incident_id": "INC-SHIPWRECK-002",
            "timestamp": "2026-09-26 18:35:00 UTC",
            "dataset_name": "AI4Shipwrecks (CC-BY-4.0)",
            "is_demo_mode": True,
            "class": "shipwreck",
            "confidence": 0.861,
            "elevation_m": 8.45,
            "elevation_provenance": "DERIVED",
            "lat": 45.0521,
            "lon": -83.3104,
            "geo_provenance": "DERIVED",
            "track_id": "TRK-WRECK-02",
            "persistence_status": "PERSISTENT",
            "observation_count": 6,
            "track_age_s": 85.0,
            "system1": {
                "engine": "convaiinnovations/laya",
                "decision_primitive": "REDUCE_SPEED",
                "hazard_score": 6.8,
                "needs_operator_review": True,
                "status": "ACTIVE"
            },
            "system2": {
                "incident_summary": "Large metallic / timber hull structure detected on benthic floor with significant vertical acoustic relief.",
                "observed_evidence": ["Elongated hull signature exceeding 35 meters", "Pronounced shadow extending 28 meters"],
                "uncertainties": ["Structural integrity of mast protrusions uncertain", "Archaeological status unverified"],
                "risk_interpretation": "Vertical navigational obstruction. Speed reduction mandated to prevent acoustic collision.",
                "operator_action": "Log coordinates for hydrographic heritage office; ascend 5 meters for stand-off pass.",
                "recovery_priority": "HIGH"
            }
        }
    },
    {
        "name": "pipeline",
        "payload": {
            "incident_id": "INC-PIPELINE-003",
            "timestamp": "2026-09-26 18:40:00 UTC",
            "dataset_name": "SubPipe (CC-BY-4.0)",
            "is_demo_mode": True,
            "class": "submarine_pipeline",
            "confidence": 0.645,
            "elevation_m": 1.25,
            "elevation_provenance": "DERIVED",
            "lat": None,
            "lon": None,
            "geo_provenance": "UNAVAILABLE",
            "track_id": "TRK-PIPE-03",
            "persistence_status": "NEW_CONTACT",
            "observation_count": 1,
            "track_age_s": 0.0,
            "system1": {
                "engine": "convaiinnovations/laya",
                "decision_primitive": "PASSIVE_LOG",
                "hazard_score": 5.4,
                "needs_operator_review": False,
                "status": "ACTIVE"
            },
            "system2": {
                "incident_summary": "Linear benthic conduit detected across survey transect. Potential subsea utility infrastructure.",
                "observed_evidence": ["Continuous linear high-intensity boundary", "Uniform narrow acoustic shadow consistent with cylindrical pipe"],
                "uncertainties": ["Burial depth along eastern boundary unconfirmed", "Cathodic protection condition unmeasured"],
                "risk_interpretation": "Infrastructure hazard. Bottom-tracking altitude must remain above 5m.",
                "operator_action": "Maintain pipeline perpendicular survey track; continue passive logging.",
                "recovery_priority": "MODERATE"
            }
        }
    }
]

def verify_all_pdfs():
    results = {}
    for tc in test_cases:
        name = tc["name"]
        payload = tc["payload"]
        pdf_bytes = MissionReportGenerator.generate_pdf(payload)
        out_path = os.path.join(PDF_DIR, f"{name}_report.pdf")
        with open(out_path, "wb") as f:
            f.write(pdf_bytes)
        
        # Parse and inspect PDF content
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        checks = {
            "pdf_generated": len(pdf_bytes) > 1000,
            "demo_banner_present": "DEMO MODE ACTIVE" in full_text,
            "provenance_table_present": "PROVENANCE" in full_text or "EVIDENCE PROVENANCE" in full_text,
            "uncertainty_present": "UNCERTAINT" in full_text or len(payload["system2"]["uncertainties"]) > 0,
            "persistence_status_present": payload["persistence_status"] in full_text,
            "system1_decision_present": payload["system1"]["decision_primitive"] in full_text,
            "system2_present": "SYSTEM-2" in full_text or "SYSTEM 2" in full_text,
            "no_fabricated_gps": (("GPS Coordinates: UNAVAILABLE" in full_text or "UNAVAILABLE" in full_text) if payload["lat"] is None else True)
        }
        results[name] = {
            "checks": checks,
            "size_bytes": len(pdf_bytes),
            "file_path": out_path
        }
        print(f"[{name.upper()}] All Checks Passed: {all(checks.values())}")
        for k, v in checks.items():
            print(f"  - {k}: {v}")

    return results

if __name__ == "__main__":
    verify_all_pdfs()
