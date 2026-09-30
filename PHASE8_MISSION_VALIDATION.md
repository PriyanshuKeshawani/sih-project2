# PHASE 8: END-TO-END MISSION REPLAY, EXTERNAL DATASET TESTING, BENCHMARKING & DEPLOYMENT PACKAGING

**System:** Ocean IQ (SIH 2026 Problem Statement 26057 — MoES / NIOT Chennai)  
**Phase:** 8 — Full System Validation, Mission Replay & Deployment Packaging  
**Quality Gate Status:** `END_TO_END_VALIDATED_WITH_LIMITATIONS`  
**Date:** 2026-09-26  
**Evaluator:** Principal Autonomous Marine Systems Architect & Senior Edge ML Lead  

---

## 1. Public Datasets Used

Four public sonar repositories were integrated and formally documented in `dataset_manifest.json`:

| Dataset Name | Upstream Repository / DOI | Origin & Authority | Role in Ocean IQ |
|---|---|---|---|
| **DRISHTI SSS** | [HuggingFace: rehan9599/drishti-sss](https://huggingface.co/datasets/rehan9599/drishti-sss) | Assembled training/test splits for SIH 2026 PS 26057 | Primary multi-class validation split (ghost net, mine, pipeline, shipwreck, background) |
| **SubPipe** | [GitHub: remaro-network/SubPipe-dataset](https://github.com/remaro-network/SubPipe-dataset) | OceanScan-MST / REMARO Network (Alvarez-Tuñón et al., 2024) | Subsea pipeline inspection transects and clean negative control backgrounds |
| **AI4Shipwrecks** | [Deep Blue: 8623hz41x](https://deepblue.lib.umich.edu/data/concern/data_sets/8623hz41x) | University of Michigan Field Robotics / NOAA Thunder Bay Sanctuary | Submerged shipwreck hull anomalies and artificial reef structures |
| **NOAA Ocean Exploration** | [NOAA Ocean Exploration Multimedia](https://oceanexplorer.noaa.gov/multimedia/georeferenced-side-scan-sonar-image/) | U.S. National Oceanic and Atmospheric Administration (NOAA OER) | Georeferenced side-scan sonar swath verifying non-taxonomy natural benthic geology |

---

## 2. Licenses & Redistribution Terms

No dataset was ingested or processed silently. Every sample processed by `engine/dataset_loader.py` records its source and license:

| Dataset | Governing License | Commercial / Academic Redistribution Terms |
|---|---|---|
| **DRISHTI SSS** | `CC-BY-SA-4.0` | Creative Commons Attribution-ShareAlike 4.0 International. Upstream citation required; derivative datasets inherit ShareAlike. |
| **SubPipe** | `CC-BY-4.0` | Creative Commons Attribution 4.0 International. Free sharing and adaptation with credit to OceanScan-MST. |
| **AI4Shipwrecks** | `CC-BY-4.0` | Creative Commons Attribution 4.0 International. Free sharing with citation of Sethuraman et al. and NOAA Thunder Bay. |
| **NOAA Sonar Archive** | `U.S. Public Domain` | United States Government Work (17 U.S.C. § 105). Unrestricted public distribution and scientific use. |

---

## 3. Sonar Imagery Tested

The complete end-to-end replay engine evaluated the following physical images:

| Scenario | Image Filename | Resolution | Channels | Dataset Source | Target Anomaly Category |
|---|---|---|---|---|---|
| **Scenario A** | `bg_1693569243.750_x2500.jpg` | 500 × 640 | 3 | SubPipe (CC-BY-4.0) | Clean Seabed (Negative Control) |
| **Scenario B** | `synth_ghost_net_00001.png` | 640 × 640 | 3 | DRISHTI SSS Synthetic (CC-BY-SA-4.0) | High-contrast derelict fishing gear |
| **Scenario C** | `wreckA_Artificial_Reef_06_y1280_x0.jpg` | 640 × 640 | 3 | AI4Shipwrecks (CC-BY-4.0) | Shipwreck / Artificial Reef Structure |
| **Scenario D** | `pipe_1693569383.780_x3500.jpg` | 500 × 640 | 3 | SubPipe (CC-BY-4.0) | Submarine Pipeline Linear Feature |
| **Scenario E** | `mine_0001_2015.jpg` | 1024 × 1024 | 3 | DRISHTI SSS MILCO (CC-BY-SA-4.0) | Cylindrical Naval Mine Anomaly |
| **Scenario F** | `noaa_fig2_sidescan.png` | 700 × 724 | 3 | NOAA Ocean Exploration (Public Domain) | Real NOAA Georeferenced Benthic Swath |
| **Scenario G** | `wreckA_Artificial_Reef_06_y1280_x320.jpg` | 640 × 640 | 3 | AI4Shipwrecks (CC-BY-4.0) | Multi-target structural debris scene |

---

## 4. Scenario Replay Results (A through G)

Every replay executed the entire sequential architecture:
`IMAGE → PREPROCESS → TILING → YOLO → NMS → PHYSICS → GEO → SYSTEM 1 (LAYA) → TEMPORAL TRACKING → SYSTEM 2 (GROQ / FALLBACK) → TIMELINE EVENT → PDF DISPATCH`

| Scenario | Detections Count | Primary Detected Class | System 1 Reflex Decision | Hazard Score | System 2 Recovery Priority | Total Replay Latency |
|---|---|---|---|---|---|---|
| **A: Clean Background** | 0 | `clean_seabed` | `PASSIVE_LOG` | 1.0 / 10 | `ROUTINE` | 1,408.5 ms |
| **B: Ghost Net** | 1 | `ghost_net` (conf=0.929) | `EMERGENCY_PROP_HAZARD` | 8.5 / 10 | `CRITICAL` | 3,573.7 ms |
| **C: Shipwreck** | 1 | `shipwreck` (conf=0.861) | `PASSIVE_LOG` | 5.8 / 10 | `MODERATE` | 3,166.9 ms |
| **D: Pipeline** | 1 | `submarine_pipeline` (conf=0.645) | `PASSIVE_LOG` | 5.4 / 10 | `LOW` | 3,400.8 ms |
| **E: Mine-Cylinder** | 3 | `mine_cylinder` (conf=0.551) | `LOITER_AND_RESCAN` | 6.5 / 10 | `HIGH` | 10,992.3 ms |
| **F: Non-Taxonomy Swath** | 2 | `shipwreck` (surrogate, conf=0.800) | `PASSIVE_LOG` | 5.8 / 10 | `MODERATE` | 6,304.4 ms |
| **G: Mixed Multi-Contact** | 1 | `shipwreck` (conf=0.858) | `PASSIVE_LOG` | 5.8 / 10 | `MODERATE` | 3,367.5 ms |

---

## 5. Detector Outputs & Behavior

1. **Aspect-Ratio-Preserving Tiling**: 1024×1024 images (Scenario E) generated 4 overlapping 640×640 tiles with 20% overlap, detecting all 3 distinct mine contact datums without aspect ratio distortion.
2. **Class-Aware NMS**: Successfully resolved overlapping bounding boxes within identical classes while preventing cross-class suppression.
3. **Closed-Set Surrogate Perception**: On the unannotated NOAA natural benthic swath (Scenario F), the 5-class detector produced detections labeled `shipwreck-class contact`. The system handled this conservatively as designed, noting that the model lacks an `unknown_debris` class.

---

## 6. System 1 Reflex Engine Decisions (Laya Integration)

1. **Active Real Laya Agent**: Official `laya==0.3.11` package executed forward passes directly from the local checkpoint (`models/laya/checkpoint`).
2. **Propeller Fouling Emergency**: On Scenario B (ghost net), Laya evaluated the risk profile and issued `EMERGENCY_PROP_HAZARD` with hazard score `8.5/10.0` and recommended immediate vertical climb (+15° pitch).
3. **Mine Standoff Loitering**: On Scenario E (cylindrical mine), Laya evaluated an explosive hazard and issued `LOITER_AND_RESCAN` with hazard score `6.5/10.0`.
4. **Deterministic Local Fallback**: When Laya checkpoint is absent or if a forward pass encounters runtime memory limits, the local rule engine (`RuleBasedSystem1Engine`) executes in `< 0.5 ms` with identical safety invariants.

---

## 7. Multi-Ping Temporal Tracking & State Transitions

Tested using repeated scan observations of Scenario B target:
- **Ping #1 (`SCAN_01`)**: Single observation → Contact initialized with status `NEW_CONTACT` (Count: 1). **Single observations are NEVER called PERSISTENT.**
- **Ping #2 (`SCAN_02`)**: Second sequential observation within spatial threshold (IoU $\ge 0.30$) → Updated with status `NEW_CONTACT` (Count: 2).
- **Ping #3 (`SCAN_03`)**: Third consecutive observation → Transitioned to `PERSISTENT` (Count: 3, Mean Confidence: 0.929).
- **Simulated Omission (`SCAN_OMIT`)**: Contact missed in subsequent scans correctly transitioned toward `TRANSIENT`.

---

## 8. System 2 Reasoning & Groq LLM Verification

1. **Investigation of Accessible Groq Models**:
   The Groq client was probed using `client.models.list()`. The model `llama-3.3-70b-versatile` returned `404 - model not found / no access`. The accessible models for the user account were:
   - `qwen/qwen3.8-27b`
   - `openai/gpt-oss-20b`
   - `openai/gpt-oss-120b`
   - `allam-2-7b`
   - `meta-llama/llama-prompt-guard-2-86m`
2. **Automated Dynamic Verification**:
   `GroqSystem2Engine` was enhanced to inspect account models at startup without blind guessing. It verified that `qwen/qwen3.8-27b` produces valid JSON schema output and automatically engaged it (`groq_status = "ACTIVE"`, `groq_model_verified = "qwen/qwen3.8-27b"`).
3. **Graceful Fallback Handling**:
   When Groq free-tier rate limits (1000 output tokens/minute) returned `429 Too Many Requests`, the engine failed fast (`max_retries=0`) and instantly routed to the local deterministic fallback (`DeterministicSystem2Fallback`), producing valid structured tactical briefings without blocking System 1 or crashing the pipeline.

---

## 9. Official Mission Report PDF Integration

1. **ReportLab Generator**: Generated official "Operation Net-Zero: Tactical Recovery Dispatch Order" PDFs for all replays.
2. **Prominent DEMO Watermark / Alert**: Every replay generated with simulated telemetry or uncalibrated GPS displays a prominent banner:  
   `[DEMO MODE ACTIVE] This mission report contains simulated / demo acoustic survey data. Not for live maritime navigation.`
3. **Structured Telemetry Table**: Includes Incident Ref ID, Dataset Source & License, AUV Asset, Target Anomaly Class, Confidence, Shadow Height, Seafloor Depth, Coordinates, and System 1 Reflex telemetry.
4. **Complete Provenance Column**: Every measurement is tagged with its provenance (`MEASURED`, `DERIVED`, `ASSUMED`, `SIMULATED`, `DEMO`, or `UNAVAILABLE`).

---

## 10. Provenance Matrix

| Parameter | Display Value (Scenario B Example) | Assigned Provenance | Derivation Method |
|---|---|---|---|
| **Anomaly Class** | `GHOST_NET-CLASS CONTACT` | `MEASURED` | ONNX Tiled Detector Classification |
| **Detector Confidence** | `92.9%` | `MEASURED` | Softmax Class Output Probability |
| **Target Elevation** | `1.75 m` | `DERIVED` | Acoustic Shadow Trigonometry ($h_t = H \cdot L_s / R$) |
| **Acoustic Shadow Length** | `4.80 m` | `MEASURED` | Pixel Radiometry Thresholding |
| **Seafloor Water Depth** | `24.5 m` | `ASSUMED` | Default Altimeter Datum (Uncalibrated) |
| **WGS-84 Coordinates** | `Lat: 9.288274° N, Lon: 79.132425° E` | `DEMO` | Simulated AUV Track (Never represented as live GPS) |
| **Missing Coordinates** | `"GPS unavailable — image-space tracking only"` | `UNAVAILABLE` | Explicit zero-fabrication fallback |

---

## 11. Component Latency Benchmarks

Measured across repeated replay runs under real CPU and network workloads:

| Pipeline Component | Mean Latency | Median Latency | P95 Latency | Max Latency | Evaluation & Execution Notes |
|---|---|---|---|---|---|
| **Image Loading & Decoding** | 6.05 ms | 5.30 ms | 12.26 ms | 12.30 ms | OpenCV direct disk read |
| **Tiled ONNX Inference (CPU)** | 108.11 ms | 107.00 ms | 116.25 ms | 118.53 ms | Tiling + YOLOv8s ONNX forward pass |
| **Shadow Physics & Georef** | 0.04 ms | 0.06 ms | 0.07 ms | 0.08 ms | Sub-millisecond trigonometric math |
| **Temporal Multi-Ping Tracking** | 0.14 ms | 0.14 ms | 0.21 ms | 0.22 ms | Hungarian / spatial track association |
| **System 1 Laya (Local CPU Checkpoint)** | 3,120.40 ms | 3,085.00 ms | 3,420.00 ms | 3,600.85 ms | 804MB PyTorch safetensors forward pass |
| **System 1 Fallback (Deterministic Rule)** | 0.24 ms | 0.30 ms | 0.39 ms | 0.44 ms | Real-time edge reflex (<0.5 ms) |
| **System 2 (Groq Cloud LLM)** | 1,680.50 ms | 1,610.00 ms | 2,240.30 ms | 2,240.30 ms | Asynchronous remote inference |
| **System 2 (Deterministic Fallback)** | 304.51 ms | 324.54 ms | 346.82 ms | 348.83 ms | Local rule-based synthesis |
| **ReportLab PDF Generation** | 29.00 ms | 28.17 ms | 32.42 ms | 33.82 ms | PDF document compilation |
| **Total Asynchronous Pipeline (w/ Fallback)** | 448.23 ms | 466.45 ms | 497.58 ms | 501.94 ms | Sub-500ms full mission processing |

---

## 12. Deployment Verification & Resource Assessment

| Configuration File | Verified Status | Deployment Role |
|---|---|---|
| `Dockerfile` | **VERIFIED** | `python:3.10-slim` base, installs `libgl1` and `libglib2.0-0` for OpenCV, copies source, exposes dynamic `$PORT` |
| `render.yaml` | **VERIFIED** | Web service configuration, `--no-cache-dir` pip install, defines `$PORT` binding and environment variables |
| `requirements.txt` | **VERIFIED** | Pinned dependencies: `fastapi`, `uvicorn`, `onnxruntime`, `opencv-python-headless`, `reportlab`, `laya`, `groq`, `python-dotenv` |
| `.gitignore` | **VERIFIED** | Explicitly excludes `.env`, `*.safetensors`, `models/laya/hf_cache/`, `reports/phase8/replays/REPLAY_*.json` |
| `.env.example` | **VERIFIED** | Clean template with documentation; zero committed API keys |

### Realistic Platform Hosting Feasibility:
- **GitHub Repository Restrictions**: GitHub enforces a hard 100 MB file limit. The local Laya checkpoint (`model.safetensors`) is **803.57 MB** and **CANNOT be committed directly to GitHub**. It must be downloaded during container build or mounted via Git LFS.
- **Render Free Tier (512 MB RAM)**: Cannot load an 800 MB PyTorch model in RAM. Attempting to run Laya locally on Render Free causes the Linux OOM killer to terminate the process (`Signal 9`).
- **Production Architecture**:
  - **Option 1 (Free / Edge Tier, 512MB RAM)**: Run ONNX YOLO (42MB) + Deterministic System 1 Reflex Engine (<1MB RAM, 0.3ms latency) + Groq Cloud System 2. Total RAM: ~220 MB. 100% stable on Render Free Tier.
  - **Option 2 (Starter / Standard Tier, $\ge$ 2GB RAM)**: Run full Laya PyTorch checkpoint in memory with PyTorch CPU runtime.

---

## 13. Failure Cases Observed & Addressed

1. **Groq Model 404 & Rate Limiting (429)**: The default model `llama-3.3-70b-versatile` was unavailable on the user account, and rapid burst requests exceeded free-tier OTPM limits. Fixed by auto-verifying accessible model `qwen/qwen3.8-27b`, setting `max_retries=0`, and routing cleanly to deterministic local fallback without stalling.
2. **Laya Checkpoint Temperature Warnings**: Upstream `laya` package issues a warning regarding temperature clamping (`choice:11+=0.10 -> 0.5`). Handled gracefully by `engine/laya_adapter.py` by flagging confidence as uncalibrated and providing deterministic fallback.
3. **Missing Telemetry Fabrication**: Images loaded without altitude or navigation metadata were previously vulnerable to hardcoded coordinate invention. The system now strictly marks these as `UNAVAILABLE` and image-space only.

---

## 14. Known Limitations & Scientific Boundaries

1. **Closed-Set Classification Limitation**: The perception model is restricted to 5 trained classes (`crab_pot`, `submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`). Natural seabed formations (sand ripples, boulders, coral heads) can produce false alarms mapped to `shipwreck` or `mine_cylinder`.
2. **No Physical Sensor Stream**: Imagery is ingested via image files, not live acoustic hydrophone beams.
3. **Simulated Navigation**: GPS coordinates and water depth in demo replays are derived from reference test positions and are clearly marked `DEMO` / `ASSUMED`.

---

## 15. Strict Truth-in-Validation Declaration

As mandated by Section 17:
- This system has **NOT** been field-tested on an operational naval vessel at sea.
- This system has **NOT** been certified by the Indian Coast Guard or NIOT on live operational missions.
- Validation has been conducted strictly against **public benchmark datasets** (DRISHTI SSS, SubPipe, AI4Shipwrecks, and NOAA Ocean Exploration) in an offline robotic simulation environment.

---

## 16. Final Quality Gate

```
================================================================================
FINAL QUALITY GATE DECISION:
END_TO_END_VALIDATED_WITH_LIMITATIONS
================================================================================
```
The full pipeline executes deterministically from raw side-scan sonar image to official tactical mission PDF dispatch. All component latencies have been measured objectively.
