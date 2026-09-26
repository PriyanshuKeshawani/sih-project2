# PHASE 3 AUDIT: ACOUSTIC PHYSICS & GEOREFERENCING DEFICIENCIES
**System:** SAMUDRA-AI (SIH 2026 Problem Statement 26057)  
**Audit Target:** `engine/physics.py` and its callers in `main.py` and `test_phase1.py`  
**Date:** 2026-09-26  
**Auditor:** Principal ML & Marine Robotics Software Engineer  

---

## 1. Executive Assessment

The current physics implementation in `engine/physics.py` contains severe scientific deficiencies, arbitrary heuristics, and unvalidated spatial assumptions. While it served as a placeholder prototype in early demo tests, it cannot be defended in front of Ministry of Earth Sciences (MoES) / NIOT hydrographers and sonar scientists.

---

## 2. Detailed Flaw Inventory

### A. Arbitrary Shadow Length Heuristic (Fake Geometry)
* **Code:** `engine/physics.py:31`:
  ```python
  shadow_len_m = max(0.5, norm_h * 15.0)
  ```
* **Flaw:** Shadow length is fabricated purely from bounding box height multiplied by $15.0$. It ignores the actual image pixel intensity data entirely.
* **Impact:** An object with a tall bounding box is falsely assigned a long shadow even if the seafloor behind it is bright seabed with zero acoustic shadow!
* **Correction:** Must inspect the actual image pixels down-range of the acoustic highlight, detect the dark acoustic dropout region, measure its run in pixels, and only convert to meters if valid spatial calibration exists.

### B. Arbitrary Slant Range Calculation
* **Code:** `engine/physics.py:32`:
  ```python
  slant_range_m = max(5.0, (norm_y + 0.1) * 35.0)
  ```
* **Flaw:** Assumes slant range scales with vertical pixel index $Y$ multiplied by $35.0$. In side-scan sonar, the range axis is perpendicular to the along-track vehicle path (usually the horizontal axis $X$ relative to the nadir centerline, or along the cross-track sweep).
* **Impact:** Completely incorrect geometric input to the elevation formula.

### C. Uncalibrated Pixel-to-Meter Scaling
* **Code:** Assumes meters directly without sensor metadata (`range_m`, `swath_width_m`, `meters_per_pixel`).
* **Correction:** If spatial scale is not provided in metadata, `shadow_length_m` must remain `null` with status `UNAVAILABLE`. Only `shadow_length_px` can be `MEASURED`.

### D. Hardcoded Navigation & False GPS Coordinates
* **Code:** `engine/physics.py:11, 56-59`:
  ```python
  self.base_lat = 9.2882
  self.base_lon = 79.1325
  self.water_depth = 24.5
  "zone": "Gulf of Mannar - Dugong Sanctuary Buffer Sector 4"
  ```
* **Flaw:** Coordinates are hardcoded to the Gulf of Mannar. Every uploaded image is falsely portrayed as located in Tamil Nadu waters with 24.5m water depth.
* **Correction:** Separate real GPS from demo coordinates. Explicitly label provenance: `status = "DEMO"` or `"SIMULATED"` or `"UNAVAILABLE"`.

### E. Conflation of Platform Position and Object Position
* **Flaw:** The system currently outputs a single coordinate without indicating whether it represents the AUV vessel position or the target on the seafloor.
* **Correction:** Distinguish `platform_position` from derived `target_position`. If range offset geometry is missing, return `platform_position` only.

### F. Conflation of Depth and Elevation Terminology
* **Flaw:** `depth_m = 24.5` is presented as target depth. In hydrography:
  - `vehicle_depth_m` = distance from sea surface to AUV.
  - `sonar_altitude_m` = distance from AUV to seabed.
  - `seabed_depth_m` = `vehicle_depth_m` + `sonar_altitude_m`.
  - `target_elevation_m` = vertical clearance of debris above the seabed.
  - `target_depth_m` = `seabed_depth_m` - `target_elevation_m`.
* **Correction:** Represent each physical parameter distinctly with provenance.

---

## 3. Required Phase 3 Architecture

```
Sonar Image + Metadata
        ↓
Target Detection (Phase 2)
        ↓
Acoustic Shadow Intensity Segmentation (engine/shadow_analysis.py)
        ↓
    [Shadow in Pixels: MEASURED]
        ↓
Scale Calibration (Metadata: meters_per_pixel / swath_width)
        ↓
    [Shadow in Meters: DERIVED or UNAVAILABLE]
        ↓
Acoustic Elevation Trigonometry (engine/physics.py)
        ↓
    [Elevation in Meters: DERIVED or UNAVAILABLE]
        ↓
Georeferencing & Navigation Engine (engine/georeferencing.py)
        ↓
    [Coordinates: REAL | DEMO | UNAVAILABLE]
```

---

## 4. Scientific Honesty Provenance States

Every physical value will strictly carry one of these states:
- `MEASURED`: Directly calculated from image pixel evidence or telemetry sensors.
- `DERIVED`: Computed via validated formulas from measured inputs.
- `ASSUMED`: Stated default values used for demonstration when metadata is missing.
- `SIMULATED`: Generated from synthetic or simulated tracks.
- `UNAVAILABLE`: Set to `null` because required prerequisites are missing.
