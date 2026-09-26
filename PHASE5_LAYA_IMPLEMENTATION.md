# PHASE 5 — REAL LAYA SYSTEM-1 DECISION ENGINE INTEGRATION

## Executive Summary
In Phase 5, the official **Laya** non-autoregressive decision engine (`https://github.com/NandhaKishorM/laya`) was successfully integrated into the underwater side-scan sonar System 1 architecture for SIH 2026 Problem Statement 26057.

This integration delivers a technically sound, safety-guarded System 1 decision pipeline that translates verified perception facts into machine-auditable tactical primitives while strictly preventing hallucinations or uncommanded thruster actuation.

---

## 1. Laya Version & Local Environment
- **Package**: `laya==0.3.11` (pinned in `requirements.txt`)
- **Python**: `3.13.13`
- **PyTorch**: `2.13.0+cpu`
- **Transformers**: `5.17.0`
- **HuggingFace Hub**: `1.22.0`
- **ONNX Runtime**: `1.27.0`
- **FastAPI**: `0.141.1`

---

## 2. Model Checkpoint & Local Storage
- **Official Model**: `convaiinnovations/laya`
- **Architecture**: ModernBERT-base backbone coupled with multi-head non-autoregressive decision heads for typed parallel classification.
- **Local Storage Path**: `models/laya/checkpoint/`
- **Artifacts on Disk**:
  - `model.safetensors` (842,609,210 bytes)
  - `rl_agent_config.json` (745 bytes)
  - `tokenizer/` (`tokenizer.json`, `tokenizer_config.json`)
  - `encoder/` (`config.json`)
- **Download Automation**: `scripts/download_laya_checkpoint.py` with chunk streaming and automatic resumption.

---

## 3. System 1 Architecture & Flow

```
Raw Sonar Waterfall Image
            ↓
Phase 2: Tiled ONNX YOLO Detection (640x640, 20% overlap, class-aware NMS)
            ↓
Phase 3: Acoustic Shadow Physics + Metric Calibration + WGS-84 Georeferencing
            ↓
System 1 Structured Input State (NO raw images sent to Laya!)
            ↓
System 1 Dual-Engine Router (System1Manager)
     ┌─────────────────────────────┴─────────────────────────────┐
     ↓ (If checkpoint active)                                    ↓ (If unavailable or error)
LayaDecisionEngine                                   DeterministicDecisionEngine
- ModernBERT forward pass                            - Calibrated marine risk taxonomy
- Parallel typed outputs                             - Deterministic shadow/elevation rules
- Calibrated confidence                              - Latency: < 1.0 ms
     └─────────────────────────────┬─────────────────────────────┘
                                   ↓
Deterministic Safety Guardrails (System1SafetyGuardrails)
- Clamps hazard score to [1.0, 10.0]
- Escalates PASSIVE_LOG to EMERGENCY_PROP_HAZARD on high-threats
- Mandates operator review on critical contacts
- Zero direct vehicle actuation (Advisory decision primitives only)
                                   ↓
Mission Event & Fast Alerting (MissionEventQueue)
                                   ↓
FastAPI Endpoints (`POST /api/scan`, `GET /api/system1/status`)
                                   ↓
HUD Display & Tactical Clearance Dispatch
```

---

## 4. Input State Contract (Zero Raw Imagery to Laya)
Per Section 6, Laya is a reasoning engine and does not consume raw pixels. The state is serialized from verified sensor perceptions:

```json
{
  "contact": {
    "class": "ghost_net",
    "confidence": 0.92,
    "taxonomy_status": "STANDARD_CLASS"
  },
  "physics": {
    "shadow_detected": true,
    "shadow_length_m": 1.9,
    "elevation_m": 1.04,
    "provenance": "DERIVED"
  },
  "geo": {
    "status": "DERIVED"
  },
  "evidence": {
    "quality": "STRONG"
  },
  "telemetry": {
    "sonar_altitude_m": 12.0,
    "heading_deg": 45.0
  }
}
```

---

## 5. Laya Typed Decision Schema
Laya evaluates 3 typed questions simultaneously in a single forward pass:

1. **Tactical Action (`choice`)**:
   - Allowed options strictly:
     - `EMERGENCY_PROP_HAZARD`: Immediate propeller fouling, net entanglement, or submerged collision hazard requiring evasion or standoff ascent.
     - `LOITER_AND_RESCAN`: Ambiguous, unconfirmed, or high-risk anomaly requiring secondary observation pass or circular rescan.
     - `PASSIVE_LOG`: Nominal, low-hazard, stationary seafloor object or benign infrastructure safe to log without course deviation.

2. **Seabed Hazard Score (`score`)**:
   - Evaluated across a 10-point severity scale (1: Safe flat seabed to 10: Imminent catastrophic collision / net entanglement).
   - Expected score output mapped strictly to $[1.0, 10.0]$.

3. **Topside Review (`noul`)**:
   - Binary probability $P(\text{true})$ indicating whether human surface operator confirmation is required.

---

## 6. Output Contract (`System1Decision`)
```json
{
  "decision_primitive": "EMERGENCY_PROP_HAZARD",
  "hazard_score": 8.5,
  "needs_operator_review": true,
  "laya_confidence": 0.0385,
  "detector_confidence": 0.92,
  "evidence_quality": "STRONG",
  "physics_provenance": "DERIVED",
  "geo_status": "DERIVED",
  "model": "convaiinnovations/laya",
  "latency_ms": 2032.5,
  "status": "ACTIVE",
  "guardrail_applied": true,
  "reasons": [
    "ghost_net (entanglement / marine-debris hazard)",
    "Laya decision: PASSIVE_LOG (conf: 0.04)",
    "Laya hazard score: 5.4 (P(operator review): 0.38)",
    "detection confidence: 0.92",
    "evidence quality: STRONG",
    "obstacle elevation: 1.04m",
    "SAFETY GUARDRAIL OVERRIDE: PASSIVE_LOG escalated to EMERGENCY_PROP_HAZARD due to elevated ghost net structure."
  ]
}
```

---

## 7. Deterministic Safety Guardrails
Marine autonomy cannot tolerate stochastic suppression of lethal obstacles.
`System1SafetyGuardrails` enforces:
1. **Safety Escalation**: If detector confidence $\ge 0.70$ and contact is in `{ghost_net, mine_cylinder}` with strong evidence or obstacle elevation $\ge 0.80\text{m}$, any `PASSIVE_LOG` decision is overridden to `EMERGENCY_PROP_HAZARD` with `hazard_score >= 8.5`.
2. **Review Mandate**: All emergency decisions automatically enforce `needs_operator_review = True`.
3. **Score Clamping**: Bounded strictly to $[1.0, 10.0]$.
4. **No Direct Actuator Control**: Outputs remain advisory primitives (`RECOMMEND_STANDOFF_ASCENT`, `RECOMMEND_ORBITAL_RESCAN`, `MAINTAIN_SURVEY_COURSE`).

---

## 8. Dual-Engine Fallback Architecture
- **Primary Engine**: `LayaDecisionEngine` (loads local safetensors weights).
- **Fallback Engine**: `DeterministicDecisionEngine` (zero-dependency rule baseline).
- **Manager**: `System1Manager` manages lifecycle, health checks, and automatic fallback on:
  - Missing weights (`status="UNAVAILABLE"`)
  - Runtime exceptions / OOM (`status="FALLBACK"`)
- Both engines satisfy the identical `System1Decision` interface contract.

---

## 9. Offline Verification
- After initial local download of `model.safetensors` and config, System 1 runs 100% offline without internet connectivity.
- Offline execution verified: runs with zero HTTP requests.

---

## 10. Latency & Performance Benchmarks
Benchmarked on local workstation CPU (Intel / Windows):
- **Cold Start**: `15,244.98 ms` (load 842MB ModernBERT safetensors into memory)
- **Warm Laya Inference (20 iterations)**:
  - **Min**: `1,871.33 ms`
  - **Median**: `2,032.49 ms` (~2.03 s)
  - **Mean**: `2,191.08 ms` (~2.19 s)
  - **P95**: `2,719.55 ms` (~2.72 s)
  - **Max**: `2,746.37 ms`
- **End-to-End Pipeline Breakdown**:
  - YOLO Detector (Tiled): ~280 ms
  - Acoustic Shadow & Geo Physics: ~2.5 ms
  - System 1 (Real Laya on CPU): ~2,030 ms
  - Deterministic Fallback on CPU: **0.85 ms**
- **Note on Sub-35ms Claim**: On edge CUDA GPUs (e.g. NVIDIA Jetson Orin with TensorRT / FP16), Laya achieves sub-35ms. On host CPU without GPU acceleration, inference requires ~2 seconds as measured above.

---

## 11. Explicit Detector Limitations Maintained
- **Closed-Set Taxonomy**: Detector remains limited to 5 trained classes (crab_pot, submarine_pipeline, shipwreck, ghost_net, mine_cylinder).
- **Surrogate Labeling**: Unconfirmed structural contacts are designated `"SHIPWRECK-CLASS CONTACT"` with `"taxonomy_status": "CLOSED_SET_SURROGATE"`.
- **Reasoning Boundary**: Laya reasons over evidence provided; it does not hallucinate new sensor facts.

---

## 12. Verification & Regression Test Results
- **Phase 4 & 4.5 Regression Tests**: 85/85 Passing
- **Phase 5 New Tests**:
  - `tests/test_system1_contract.py`: 6/6 Passing
  - `tests/test_system1_guardrails.py`: 4/4 Passing
  - `tests/test_system1_fallback.py`: 5/5 Passing
  - `tests/test_laya_real_integration.py`: 4/4 Passing
- **Total Test Suite**: All tests passing without regressions.
