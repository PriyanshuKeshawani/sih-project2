# SAMUDRA-AI: COMPREHENSIVE PROJECT AUDIT (PHASE 1)
**Project:** SIH 2026 Problem Statement 26057  
**System:** AI-Powered Automated Underwater Marine Debris and Anomaly Detection System using Side-Scan Sonar Imagery  
**Sponsoring Body:** Ministry of Earth Sciences (MoES) / National Institute of Ocean Technology (NIOT, Chennai)  
**Audit Date:** 2026-09-26  
**Auditor:** Principal ML & Marine Robotics Software Engineer  

---

## 1. Executive Summary & Verification Matrix

| Component | Status | Verified Real Behavior |
| :--- | :--- | :--- |
| **ONNX Runtime Inference** | **Working** | Local CPU inference verified in 74ms–112ms per 640x640 tile. |
| **Model Weights & Metadata** | **Working** | Verified Ultralytics YOLOv8s (opset 12, stride 32, input: [1, 3, 640, 640], output: [1, 9, 8400]). |
| **Class Detection Accuracy** | **Working** | Tested on 5 real/synthetic sonar images across all 5 classes (`crab_pot`, `submarine_pipeline`, `shipwreck`, `ghost_net`, `mine_cylinder`). |
| **False-Positive Suppression** | **Working** | Tested on background seabed (`bg_1693569243.750_x2500.jpg`) -> 0 detections above 0.25 conf. |
| **Large-Image Tiling / Slicing** | **BROKEN / MISSING** | Detector directly runs `cv2.resize(img, (640, 640))`. Large sonar waterfall strips (e.g. 10,000x1,000 px) get squashed and small nets are destroyed. |
| **Acoustic Preprocessing** | **MISSING** | No configurable CLAHE, Bilateral, Lee filter, or Raw vs Processed toggle. |
| **Shadow Physics Rigor** | **PARTIALLY WORKING (HEURISTIC)** | Shadow length is currently estimated from bounding box height ($H_{box} \times 15.0$) rather than segmented from acoustic dark patches. No provenance tagging (`measured` vs `assumed`). |
| **Geospatial Rigor** | **PARTIALLY WORKING (SIMULATED)** | Hardcoded to Gulf of Mannar ($9.2882^\circ N, 79.1325^\circ E$). Lacks explicit `DEMO / SIMULATED` metadata tags. |
| **System 1 Reflex Engine** | **WORKING (DETERMINISTIC)** | Sub-5ms deterministic rule-based autopilot decisions (`EMERGENCY_PROP_HAZARD`, `RECORD_INFRASTRUCTURE`, `LOITER_AND_RESCAN`). Laya library installed, but direct ModernBERT model adapter not yet hooked. |
| **System 2 Tactical Commander** | **MISSING** | No Groq/LPU layer exists in the codebase yet. |
| **Temporal Persistence** | **MISSING** | No multi-pass survey comparison or stationary vs transient tracking. |
| **FastAPI Data Contracts** | **PARTIALLY WORKING** | Functional endpoints exist (`/api/scan`, `/api/samples`, `/api/export-pdf`), but use raw dictionaries instead of typed Pydantic models. |
| **Automated Test Suite** | **MISSING** | No `tests/` directory with pytest coverage. |
| **Security & Upload Hardening**| **PARTIALLY WORKING** | No file-size limits, magic-byte MIME validation, or path-traversal guards on uploads. |

---

## 2. Multi-Class Inference Test Execution Log (Actual Measured Results)

Executed on local CPU (Windows, Python 3.12, ONNX Runtime 1.20):

```
=== MULTI-CLASS INFERENCE VERIFICATION ===

1. Background (Clear Seabed) [data/samples/bg_1693569243.750_x2500.jpg]
   - Original Dimensions: 640x500 px
   - Total Latency: 93.4 ms (Inference: 90.6 ms)
   - Detections Count: 0 (Correct negative response)

2. Ghost Net (Target) [data/samples/synth_ghost_net_00001.png]
   - Original Dimensions: 640x640 px
   - Total Latency: 126.4 ms (Inference: 112.4 ms)
   - Detections:
     * Class: ghost_net | Confidence: 92.9% | Box: [x=113, y=210, w=114, h=109]
     * Elevation: 1.75m | Geo: 9.288275 N, 79.132424 E | Reflex: EMERGENCY_PROP_HAZARD

3. Shipwreck [data/samples/wreckR_ship-081_png.rf.6cf386b75ddb8ead86c0453021279296.jpg]
   - Original Dimensions: 489x525 px
   - Total Latency: 78.2 ms (Inference: 74.6 ms)
   - Detections:
     * Class: shipwreck | Confidence: 81.6% | Box: [x=7, y=29, w=273, h=455]
     * Elevation: 8.46m | Geo: 9.288266 N, 79.132433 E | Reflex: RECORD_INFRASTRUCTURE

4. Submarine Pipeline [data/samples/pipe_1693569383.780_x3500.jpg]
   - Original Dimensions: 640x500 px
   - Total Latency: 78.7 ms (Inference: 75.3 ms)
   - Detections:
     * Class: submarine_pipeline | Confidence: 71.8% | Box: [x=84, y=0, w=454, h=255]
     * Elevation: 7.26m | Geo: 9.288204 N, 79.132495 E | Reflex: RECORD_INFRASTRUCTURE

5. Mine / Cylinder [data/samples/mine_0001_2015.jpg]
   - Original Dimensions: 1024x1024 px
   - Total Latency: 85.6 ms (Inference: 76.6 ms)
   - Detections:
     * Class: mine_cylinder | Confidence: 81.3% | Box: [x=87, y=213, w=47, h=37] | Elevation: 0.57m
     * Class: mine_cylinder | Confidence: 72.3% | Box: [x=908, y=963, w=57, h=40] | Elevation: 0.19m
     * Class: mine_cylinder | Confidence: 42.2% | Box: [x=406, y=425, w=54, h=56] | Elevation: 0.52m (Reflex: LOITER_AND_RESCAN)
```

---

## 3. Deep Architectural Analysis & Deficiencies

### A. What Currently Works
1. **Model Weight Loading & Execution:** `models/best_detector.onnx` (42.68 MB) loads reliably via ONNX Runtime CPUExecutionProvider without PyTorch or CUDA dependencies.
2. **Output Parsing:** Correctly transposes output shape `(1, 9, 8400)` to `(8400, 9)` and decodes center-based coordinates $(c_x, c_y, w, h)$.
3. **Basic NMS & Rescaling:** OpenCV `cv2.dnn.NMSBoxes` collapses redundant boxes and maps bounding boxes back to original image dimensions.
4. **Mission PDF Pipeline:** ReportLab generates valid multi-page binary PDF dispatch sheets.
5. **Basic Web Frontend:** Custom dark naval cockpit (FastAPI + HTML/CSS/JS + Leaflet.js) loads and communicates with backend endpoints.

### B. What Is Broken or Flawed
1. **The Aspect Ratio Squashing Bug:**
   - In `engine/detector.py:21`: `img_resized = cv2.resize(img_rgb, (640, 640))`.
   - Side-scan sonar waterfall data comes in long continuous swaths (e.g., $10,000 \times 1,000$ pixels).
   - Squashing a 10:1 ratio strip into 1:1 distorts acoustic geometry completely, blurring acoustic shadows and making small nets undetectable.
2. **Class-Agnostic NMS Suppression:**
   - Currently, `NMSBoxes` runs on all candidate boxes together. If a submarine pipeline (high confidence) overlaps or sits adjacent to a ghost net, the pipeline box can suppress the ghost net box. NMS must be **class-aware**.

### C. Scientific & Physics Assumptions (Scientific Honesty Deficiencies)
1. **Shadow Length Heuristic:**
   - In `engine/physics.py`, shadow length is approximated as `norm_h * 15.0`. This is a placeholder heuristic rather than a measurement.
   - *Fix Needed:* Add an adaptive acoustic shadow segmentation routine (Otsu/adaptive thresholding in the down-range acoustic shadow zone behind the highlight) and return provenance: `"source": "derived|estimated|assumed"`.
2. **Hardcoded Georeferencing:**
   - Coordinates default to Gulf of Mannar without indicating whether the coordinates came from real embedded NMEA EXIF metadata or were simulated.
   - *Fix Needed:* Clear `"source": "simulated_demo" | "nmea_metadata"` tags.

### D. Missing Modules
1. **Large-Image Slicing / Tiling Window (SAHI style):** Needed to process long sonar strips with configurable overlap (e.g. 20%) and coordinate reconciliation across tiles.
2. **Acoustic Preprocessing Pipeline:** CLAHE contrast stretching + Bilateral speckle filter with toggle.
3. **System 2 Tactical Commander:** Groq / LPU integration with safe local fallback.
4. **Temporal Persistence Tracker:** Data structure to compare repeated passes over same geofenced coordinates to flag transient fish vs persistent debris.
5. **Typed Pydantic Data Contracts:** Structured schemas for `DetectionResult`, `PhysicsResult`, `ReflexDecision`, `ScanResponse`.
6. **Automated Pytest Suite:** End-to-end regression tests.

---

## 4. Security & Deployment Audit

1. **Upload Hardening:**
   - Current `/api/scan` accepts `UploadFile` without file-size cap or image type check. A 2GB malicious payload could exhaust server memory.
   - *Fix Needed:* Max file size limit (e.g., 25MB), magic-byte MIME validation, and sanitization.
2. **Path Traversal Guard:**
   - `sample_id` from request parameters must be strictly checked to prevent directory traversal (`../../`).
3. **Secrets Management:**
   - Groq API keys must be loaded from `.env` via `python-dotenv`, never committed to source.

---

## 5. Recommended Implementation Order (Phase by Phase)

* **PHASE 1 (Completed):** Audit repository, verify ONNX model, execute 5-class baseline tests, identify root bottlenecks.
* **PHASE 2 (Completed):** Implemented Image Preprocessing (CLAHE, Bilateral despeckle) + Sliding Window / Tiling Inference with Class-Aware NMS.
* **PHASE 3:** Refactor Acoustic Shadow Physics & Georeferencing with explicit provenance (`measured` vs `assumed` vs `simulated`).
* **PHASE 4:** Hardened System 1 Edge Reflex Engine with deterministic decision primitives and Laya adapter.
* **PHASE 5:** System 2 Tactical Commander (Groq API + zero-crash local offline fallback).
* **PHASE 6:** Temporal Persistence Layer (Multi-pass survey comparison).
* **PHASE 7:** Pydantic Data Contracts & FastAPI Endpoint Hardening (Validation, rate limits, security).
* **PHASE 8:** Frontend Cockpit Enhancements (Raw vs Preprocessed comparison, Tiling view, Leaflet simulated tags).
* **PHASE 9:** Comprehensive Pytest Suite (`tests/`).
* **PHASE 10:** Docker, Deployment Hardening, and `THIRD_PARTY_NOTICES.md`.

---

## 6. PHASE 2 IMPLEMENTATION

### Files Created & Changed
1. **`engine/preprocessing.py` (New):** Modular `PreprocessConfig` and `SonarPreprocessor` for grayscale normalization, CLAHE, and bilateral filtering.
2. **`engine/tiling.py` (New):** `SonarTiler` and `SonarTile` for aspect-ratio-preserving sliding-window slicing, interval calculation, edge padding, coordinate remapping, and `class_aware_nms`.
3. **`engine/detector.py` (Refactored):** Unified inference pipeline integrating preprocessing, tiling, intra-tile NMS, coordinate remapping, and global cross-tile NMS. Backward-compatible with existing callers.
4. **`tests/test_preprocessing.py` (New):** 4 unit tests verifying dimension preservation, dtype, and grayscale handling.
5. **`tests/test_tiling.py` (New):** 5 unit tests verifying 640x640 single tile, 1024x1024 tiling, 10000x1000 panoramic coverage, smaller-than-640 padding, and boundary clipping.
6. **`tests/test_nms.py` (New):** 3 unit tests verifying independent suppression per class and preservation of overlapping different classes.
7. **`tests/test_detector_phase2.py` (New):** 3 integration tests verifying real ONNX model inference, background suppression, and large 1024x1024 tiling.
8. **`run_phase2_benchmark.py` (New):** Measures microsecond timings and saves structured metrics and visual verification images.

### Design Decisions
- **Tile Size & Stride:** Fixed `tile_size=640` with `overlap=0.20`, resulting in a `stride=512`. This aligns with the ONNX model's native training resolution without interpolation distortion.
- **Edge Padding Policy:** For dimensions smaller than 640, constant-value padding is applied to the right and bottom. Detections in the padded zone are discarded, and valid boxes are shifted by tile offsets $[x_{offset}, y_{offset}]$.
- **Two-Stage Class-Aware NMS:** First stage suppresses redundant candidate boxes inside each tile. Second stage merges duplicate proposals across overlapping tile seams without cross-class interference.

### Measured Timings (CPU Baseline)
- **Single 640x640 Tile:** ~65 ms inference, ~10 ms preprocessing, ~190 ms total.
- **Large 1024x1024 (4 Tiles):** ~260 ms total inference (65.2 ms/tile), ~553 ms end-to-end total.

### Known Limitations
- Sequential CPU execution for tiles (no dynamic batching yet).
- Boundary-crossing targets spanning >2 tiles produce multiple bounding boxes if IoU < threshold.

