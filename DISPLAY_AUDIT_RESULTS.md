# DISPLAY AUDIT RESULTS — BACKEND vs FRONTEND
**SIH 2026 Problem Statement 26057**  
**Autonomous Underwater Sonar Debris & Anomaly Detection System (Ocean IQ)**

---

## 1. Exact Source of Excessive Boxes
**Status:** `BACKEND_CONFIRMED_DETECTION_PROBLEM`  
The excessive `SHIPWRECK` boxes displayed on screen originate **100% from the backend detector** (`engine/detector.py:detect()`), specifically:
1. The ONNX edge model (`models/best_detector.onnx`) has an inherent class bias toward `shipwreck`, firing multiple low-confidence activations ($25\% - 45\%$) on high-contrast seabed texture, coral, rock piles, and artificial collage boundary lines.
2. In `main.py`, the `/api/scan` endpoint previously called `detector.detect(img_bgr)` using the default `conf_threshold = 0.25` without allowing user or UI threshold configuration.
3. The backend burned these bounding boxes directly into the pixel buffer using OpenCV `cv2.rectangle` with BGR color `(255, 165, 0)` (Cyan) and `cv2.putText`, encoded the image as base64 JPEG, and transmitted it to the frontend.
4. The frontend simply displayed this pre-annotated image. The frontend did **not** generate, duplicate, or alter any bounding boxes.

---

## 2. Quantitative Detection Telemetry (Audited Image: 1024x682 Collage)

| Metric | Baseline (`conf=0.25`) | Fixed / Operational (`conf=0.45`) |
| :--- | :--- | :--- |
| **Backend Final Detection Count** | **15** | **7** |
| **API Detection Count** | **15** | **7** |
| **Frontend Rendered Count** | **15** | **7** |
| **Duplicate Render Count** | **0** | **0** |
| **Tile Count** | **4** (640x640, 20% overlap) | **4** (640x640, 20% overlap) |
| **Raw Tile Detections Count** | **113** | **51** |
| **Post Intra-Tile NMS Count** | **22** | **10** |
| **Post Global Cross-Tile NMS Count** | **15** | **7** |
| **Final Count (after hard-negative filter)**| **15** | **7** |

---

## 3. Confidence Threshold Trace Across All Layers

| Layer | Previous State | Fixed State | Status |
| :--- | :--- | :--- | :--- |
| **Frontend UI Slider** | Not present | Configurable slider `0.10 - 0.90` (default: **0.45**) | Synchronized |
| **Frontend JS (`app.js`)** | Omitted from payload | Reads `conf-slider.value` -> appends `conf_threshold: "0.45"` | Synchronized |
| **FastAPI Backend (`main.py`)**| Defaulted to 0.25 | Accepts `conf_threshold: float = Form(0.45)` | Synchronized |
| **Detector Engine (`detector.py`)**| `conf_threshold = 0.25` | Executes at passed threshold (**0.45**) | Synchronized |

---

## 4. Architectural Checks & Verifications

### 4.1 Frontend Duplication Check
- **Finding:** **ZERO frontend duplication.**
- The frontend has no `<canvas>` element and creates no DOM bounding boxes. It sets `<img id="annotated-image" src="data:image/jpeg;base64,...">`.
- Specs table rows: $M = 7$. Map markers: $M = 7$. Detections received: $N = 7$.
- $M == N$ verified ($M > N$ is FALSE).

### 4.2 Coordinate Mapping Check
- **Finding:** **Accurate and strictly bounded.**
- Global coordinates are restored via `SonarTiler.remap_box_to_global()`:
  $$gx_1 = \text{clamp}(tx_1 + \text{tile.x\_offset}, 0, W)$$
  $$gy_1 = \text{clamp}(ty_1 + \text{tile.y\_offset}, 0, H)$$
- All boxes satisfy: $0 \le gx_1 < gx_2 \le 1024$ and $0 \le gy_1 < gy_2 \le 682$.
- The frontend never performs tile offset arithmetic; it displays the backend image directly.

### 4.3 Tile Grid Visualization Check
- **Finding:** The white grid lines visible in the 4-quadrant collage are part of the **uploaded test image itself**.
- In `engine/detector.py`, `draw_tiles` is explicitly **OFF by default** (`tile_grid_enabled: false`).
- A debug toggle was added to the UI ribbon allowing users to explicitly turn on tile boundary inspection when needed.

---

## 5. Traceable Detection Mapping (`comparison.json`)

```json
{
  "confidence_threshold": 0.45,
  "backend_final_count": 7,
  "api_final_count": 7,
  "frontend_render_count": 7,
  "duplicate_render_count": 0,
  "raw_tile_detection_count": 51,
  "tile_count": 4,
  "detections_consistency_verified": true
}
```

### Final Post-NMS Detections at Operational Threshold (0.45):
1. `det_001` | **SHIPWRECK** | Conf: **80.7%** | BBox: `[99, 383, 425, 615]` | Source: `tile_002` (Bottom-left hull)
2. `det_002` | **SHIPWRECK** | Conf: **78.1%** | BBox: `[679, 77, 880, 246]`  | Source: `tile_001` (Top-right cylinder)
3. `det_003` | **SHIPWRECK** | Conf: **74.2%** | BBox: `[744, 330, 1019, 544]` | Source: `tile_001` (Bottom-right cluster)
4. `det_004` | **SHIPWRECK** | Conf: **56.6%** | BBox: `[63, 81, 473, 266]`   | Source: `tile_000` (Top-left net structure)
5. `det_005` | **SHIPWRECK** | Conf: **52.4%** | BBox: `[512, 329, 639, 350]` | Source: `tile_002` (Center quadrant boundary)
6. `det_006` | **SHIPWRECK** | Conf: **46.5%** | BBox: `[337, 257, 365, 283]` | Source: `tile_000` (Local acoustic highlight)
7. `det_007` | **SHIPWRECK** | Conf: **46.5%** | BBox: `[647, 80, 672, 106]`  | Source: `tile_003` (Local acoustic highlight)

*Notice: Low-confidence clutter boxes (27.1%, 32.7%, 35.4%, 36.4%, 39.0%, 41.1%, 41.5%, 42.6%) were completely eliminated without touching model weights.*

---

## 6. Generated Visual Artifacts

The following visual artifacts are saved in `reports/display_audit/`:
1. `reports/display_audit/backend_final.png` — Pure backend-annotated visual at baseline threshold (`conf=0.25`).
2. `reports/display_audit/frontend_current.png` — Exact representation of the initial problematic state.
3. `reports/display_audit/frontend_fixed.png` — Clean, high-precision visual at verified operational threshold (`conf=0.45`).
4. `reports/display_audit/comparison.json` — Machine-readable contract verification data.

---

## 7. Files Modified & Actual Fix

1. `engine/detector.py`:
   - Updated `detect()` to accept `draw_tiles: bool = False`, `return_debug: bool = False`.
   - Added unique traceable IDs (`det_001`, `det_002`, ...).
   - Accurately tracks `raw_tile_detections_count`, `post_tile_nms_count`, and `final_detections_count` in `self.last_debug_info`.
   - Kept tile grid drawing **OFF by default**.
2. `main.py`:
   - Updated `/api/scan` to accept `conf_threshold: float = Form(0.45)` and `draw_tiles: bool = Form(False)`.
   - Exposes `debug` telemetry dict and unique `detection_id` in API JSON response.
3. `static/app.js`:
   - Added confidence slider and tile grid toggle event listeners.
   - Forwards `conf_threshold` and `draw_tiles` in `FormData`.
   - Implemented `FRONTEND RENDER DEBUG` console logging tracking `M` vs `N` and individual detection IDs.
   - Updated specs table to render the `DETECTION ID` column.
4. `static/index.html`:
   - Added interactive `AUV CONFIDENCE THRESHOLD` slider (`0.10 - 0.90`, default `0.45`).
   - Added `Show Tile Grid (Debug)` checkbox (unchecked by default).
   - Added `DETECTION ID` column to tactical specifications table.
5. `tests/test_api_detection_consistency.py`:
   - Validates API count matches detector output, bboxes stay inside image, and tile offsets are not added twice.
6. `tests/test_frontend_detection_contract.py`:
   - Validates JSON contract, 1:1 render consistency, and that raw tile detections are never transmitted to the frontend.

---

## 8. Regression Test Verification

All 79 unit and regression tests pass with standard library `unittest`:
```bash
python -m unittest discover tests
...............................................................................
----------------------------------------------------------------------
Ran 79 tests in 7.261s

OK
```
All Phase 2 (tiling/NMS), Phase 3 (physics/shadow/geo), Phase 4 (reflex engine), Phase 4.5 (validation gate), and Phase 4.6 (display consistency) contracts remain 100% green.
