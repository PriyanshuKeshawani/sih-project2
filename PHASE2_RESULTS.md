# PHASE 2 IMPLEMENTATION & BENCHMARK VERIFICATION RESULTS
**System:** SAMUDRA-AI (SIH 2026 Problem Statement 26057)  
**Verification Date:** 2026-09-26  
**Auditor/Engineer:** Principal ML & Marine Robotics Software Engineer  

---

## 1. Overview of Phase 2 Implementation

In Phase 2, we completely eliminated the unsafe direct-resizing path that squashed arbitrary sonar images to $640 \times 640$. We implemented a robust, modular pipeline:

1. **`engine/preprocessing.py`:**
   - Single-channel / Grayscale normalization to 3-channel BGR without color distortion.
   - Configurable `PreprocessConfig`:
     * CLAHE (Contrast-Limited Adaptive Histogram Equalization) with adaptive grid ($8 \times 8$) and clip limit ($2.0$) on Luminance channel.
     * Bilateral filtering ($d=5$, $\sigma=50$) to suppress high-frequency speckle noise while preserving sharp acoustic shadow edges.
   - `compare_raw_vs_processed()` for visual diagnostics.

2. **`engine/tiling.py`:**
   - Configurable sliding-window tiler: `tile_size=640`, `overlap=0.20`, `stride=512`.
   - Exact interval math guaranteeing full coverage of arbitrary dimensions (e.g. $10000 \times 1000$ waterfall swaths without pixel loss).
   - Safe right- and bottom-padding for edge tiles smaller than 640.
   - Exact coordinate restoration: maps tile bounding boxes $[tx_1, ty_1, tx_2, ty_2]$ to global original coordinates $[gx_1, gy_1, gx_2, gy_2]$, with strict clamping to $[0, W_{orig}]$ and $[0, H_{orig}]$.
   - Discards boxes falling into padded zones.

3. **`engine/detector.py`:**
   - Persistent ONNX session reuse on CPU.
   - Intra-tile class-aware NMS to eliminate local duplicate proposals.
   - Global cross-tile class-aware NMS to merge detections overlapping across tile seams while preserving distinct co-located classes (e.g., ghost net entangled on a pipeline).
   - Structured detection output with `class_id`, `class`, `confidence`, `bbox_xyxy`, `box` (backward-compatible), and `tile_id`.
   - Visual verification tool: `save_annotated_result()` with optional tile grid rendering.

---

## 2. Automated Test Suite Results

Ran complete automated test suite (`tests/`):
```
Ran 15 tests in 0.905s — OK (100% Passed)
```

| Test Name | File | Verified Behavior | Status |
| :--- | :--- | :--- | :--- |
| `test_640x640_image_single_tile` | `tests/test_tiling.py` | Exactly 1 tile produced for 640x640 input. | **PASSED** |
| `test_1024x1024_image_multiple_tiles` | `tests/test_tiling.py` | 4 overlapping tiles with 25% overlap. | **PASSED** |
| `test_large_panoramic_waterfall_image` | `tests/test_tiling.py` | 10000x1000 strip produces 40 tiles covering full width/height without pixel loss. | **PASSED** |
| `test_image_smaller_than_640` | `tests/test_tiling.py` | 500x400 image padded correctly; padded zone boxes safely discarded. | **PASSED** |
| `test_boundary_detection_clipped_to_original` | `tests/test_tiling.py` | BBoxes strictly clamped to $[0, W]$ and $[0, H]$. | **PASSED** |
| `test_same_class_overlapping_suppression` | `tests/test_nms.py` | Overlapping duplicate boxes of SAME class merged to highest score. | **PASSED** |
| `test_different_classes_overlapping_preservation` | `tests/test_nms.py` | Overlapping boxes of DIFFERENT classes (ghost_net vs pipeline) both preserved. | **PASSED** |
| `test_empty_boxes` | `tests/test_nms.py` | NMS safely handles empty arrays without crashing. | **PASSED** |
| `test_raw_preprocessing_valid_input` | `tests/test_preprocessing.py` | RAW mode leaves pixel values unaltered. | **PASSED** |
| `test_clahe_and_bilateral_preserves_dimensions` | `tests/test_preprocessing.py` | Processed output has identical spatial dimensions and dtype. | **PASSED** |
| `test_grayscale_conversion` | `tests/test_preprocessing.py` | 2D and 3D single-channel inputs converted to 3-channel BGR. | **PASSED** |
| `test_background_suppression` | `tests/test_detector_phase2.py` | Clear seabed produces 0 false detections above threshold. | **PASSED** |
| `test_ghost_net_detection` | `tests/test_detector_phase2.py` | Ghost net target detected with high confidence ($>90\%$). | **PASSED** |
| `test_large_image_tiling_1024x1024` | `tests/test_detector_phase2.py` | 1024x1024 mine image tiled, detections remapped globally. | **PASSED** |

---

## 3. Real Model Verification & Measured Timings

Executed via [run_phase2_benchmark.py](file:///d:/CODE%20JAANI%20CODE/hackathorns/sih%20ka%20project%202/run_phase2_benchmark.py) on local CPU (Windows, Python 3.12, ONNX Runtime 1.20):

| Test Image | True Target | Image Size | Tiles | Preprocess (ms) | Inference (ms) | NMS & Remap (ms) | Total (ms) | Detections Found |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bg_1693569243.750_x2500.jpg` | Background Seabed | 640x500 px | 1 | 119.37 | 74.40 | 86.22 | 280.37 | **0** (Clean pass) |
| `synth_ghost_net_00001.png` | Ghost Net | 640x640 px | 1 | 10.43 | 65.61 | 117.42 | 193.51 | **1** (`ghost_net`: 92.9%) |
| `wreckR_ship-081_png...jpg` | Shipwreck | 489x525 px | 1 | 5.61 | 67.67 | 75.04 | 148.69 | **1** (`shipwreck`: 84.5%) |
| `pipe_1693569383.780_x3500.jpg`| Submarine Pipeline| 640x500 px | 1 | 10.75 | 64.23 | 74.96 | 150.35 | **1** (`submarine_pipeline`: 64.5%) |
| `mine_0001_2015.jpg` | Mine / Cylinders | 1024x1024 px | **4** | 13.23 | 260.86 (65.2 ms/tile) | 279.56 | 553.71 | **4** (2 Mines, 2 Wrecks) |

All measured metrics exported to: [reports/phase2/phase2_benchmark_metrics.json](file:///d:/CODE%20JAANI%20CODE/hackathorns/sih%20ka%20project%202/reports/phase2/phase2_benchmark_metrics.json)

---

## 4. Example Structured Detection Output

```json
{
  "class_id": 3,
  "class": "ghost_net",
  "confidence": 0.929,
  "box": {
    "x": 114,
    "y": 210,
    "w": 115,
    "h": 109
  },
  "bbox_xyxy": [114, 210, 229, 319],
  "tile_id": "tile_000",
  "source_width": 640,
  "source_height": 640
}
```

---

## 5. Visual Artifacts Generated

Debug visualizations showing tile boundaries and mapped detections have been saved to:
- `reports/phase2/verified_bg_1693569243_750_x2500_jpg.jpg`
- `reports/phase2/verified_synth_ghost_net_00001_png.jpg`
- `reports/phase2/verified_wreckR_ship-081_png_rf_...jpg`
- `reports/phase2/verified_pipe_1693569383_780_x3500_jpg.jpg`
- `reports/phase2/verified_mine_0001_2015_jpg.jpg`
- `reports/phase2/test_large_mine_1024_tiled.jpg`

---

## 6. Known Limitations of Phase 2

1. **Sequential CPU Inference:** Tiles are currently processed sequentially in a loop. For a 10,000x1,000 waterfall strip (40 tiles), total inference at 65ms/tile will take ~2.6 seconds. This is acceptable for survey analysis, but could later be batched if ONNX dynamic batching is compiled.
2. **Boundary Object Splitting:** If a massive target (e.g., an 80-meter shipwreck) spans across 3 tiles, each tile detects a partial segment. Global NMS merges them if IoU exceeds the threshold, but polygon/mask union (segmentation) is outside the scope of bounding-box YOLO.
3. **No Physics/Geospatial in Detector:** Detection objects strictly report image pixel coordinates $[x_1, y_1, x_2, y_2]$, confidence, and class. Physics and Georeferencing are kept decoupled for Phase 3.

---

## 7. Exact Recommendation for Phase 3

In **PHASE 3 (Acoustic Shadow Physics & Georeferencing)**:
1. Replace heuristic shadow calculation with adaptive segmentation of down-range acoustic shadows behind detected highlights.
2. Explicitly label provenance: `measured`, `derived`, or `assumed`.
3. Support optional NMEA metadata parsing (Lat/Long, vehicle depth, heading, slant range) and explicitly mark coordinates as `DEMO / SIMULATED POSITION` whenever true navigation telemetry is unavailable.
