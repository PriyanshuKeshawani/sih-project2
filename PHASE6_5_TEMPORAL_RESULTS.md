# Phase 6.5 — Temporal / Multi-Ping Persistence Results

**System**: Underwater Sonar AI System (SIH 2026 Problem Statement 26057)  
**Phase**: 6.5 — Temporal Persistence Layer  
**Date**: September 2026  
**Status**: COMPLETE & VERIFIED (136/136 unit & integration tests passing)  

---

## 1. Executive Summary

Sonar waterfall imagery is inherently noisy: acoustic speckle, fleeting reverberation, boat wake, and seabed surface returns can trigger transient false alarms in single pings. Single-observation detections must never trigger high-consequence maritime maneuvers.

Phase 6.5 introduces an auditable, deterministic **Temporal Persistence Layer** (`engine/temporal_tracking.py`) that tracks, correlates, and classifies underwater contacts across repeated sonar survey passes.

### Core Status Categories
- **`NEW_CONTACT`**: A contact detected for the first time or observed fewer than `min_observations_for_persistent` times.
- **`PERSISTENT`**: A contact observed consistently across $\ge 3$ sequential observations with consistent spatial and temporal alignment.
- **`TRANSIENT`**: A contact that appeared briefly (1–2 pings) and failed to appear in subsequent scans (missed for $\ge 2$ consecutive passes).
- **`UNCERTAIN`**: Detections with conflicting data, low confidence ($< 0.40$), or insufficient spatial/temporal correlation.

---

## 2. Track Matching Strategy

Track matching solves the multi-target data association problem deterministically between incoming sonar detections and existing active tracks.

### Association Algorithm
1. **Survey Isolation**: Observations are strictly partitioned by `survey_id`. Tracks never cross survey boundaries.
2. **Class Compatibility**: Detections only associate with tracks of the exact same semantic class (`ghost_net`, `mine_cylinder`, `shipwreck`, `submarine_pipeline`, `crab_pot`).
3. **Temporal Ordering & Windowing**:
   $$\Delta t = t_{\text{obs}} - t_{\text{last\_seen}} \ge 0$$
   If $\Delta t > \text{max\_time\_gap\_s}$ ($60.0$ seconds), the contact is treated as a new event, and a new track is initialized.
4. **Spatial Proximity Matching**:
   - **GPS Primary Path (Metric Distance)**:
     When both observation and track possess valid GPS coordinates (`lat`, `lon`), distance is calculated via the Haversine great-circle formula:
     $$d_{\text{geo}} = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
     Association requires $d_{\text{geo}} \le \text{max\_position\_distance\_m}$ ($15.0$ meters).
   - **Image Fallback Path (Zero-GPS Scenarios)**:
     When GPS is missing or uncalibrated, Euclidean centroid pixel distance and Bounding Box Intersection-over-Union (IoU) are evaluated:
     $$d_{\text{pix}} = \sqrt{(c_{x1} - c_{x2})^2 + (c_{y1} - c_{y2})^2} \le 120\text{ px} \quad \text{OR} \quad \text{IoU}(B_1, B_2) \ge 0.15$$
   - **Zero Sensor Fabrication**: GPS coordinates are *never* invented or extrapolated from pixel space without acoustic georeferencing metadata.
5. **Greedy Minimum-Distance Assignment**: Matches are assigned in order of increasing distance; remaining unassociated detections spawn new tracks (`NEW_CONTACT`).

---

## 3. Configurable Parameters & Defaults

All thresholds are centralized in `TrackingConfig` and not hardcoded across files:

| Parameter | Default | Description |
|:---|:---:|:---|
| `max_position_distance_m` | `15.0 m` | Maximum allowable GPS drift between consecutive pings for association |
| `max_image_distance_px` | `120.0 px` | Centroid pixel threshold fallback when GPS coordinates are unavailable |
| `min_iou` | `0.15` | Minimum bounding-box IoU fallback for co-located pixel contacts |
| `max_time_gap_s` | `60.0 s` | Temporal window limit before resetting track correlation |
| `min_observations_for_persistent` | `3` | Minimum observations required before promoting to `PERSISTENT` |
| `min_confidence` | `0.40` | Minimum detector confidence required to avoid `UNCERTAIN` status |
| `transient_miss_threshold` | `2` | Number of missed consecutive scans before declaring contact `TRANSIENT` |
| `max_track_history` | `20` | Bounded ring-buffer length to prevent unbounded memory growth |

---

## 4. Persistence Rules & Edge Cases

### Rule 1: Single Observation Rule (Non-Persistence Guarantee)
> **One observation MUST NOT be called `PERSISTENT`.**
- If `observation_count == 1`:
  - `confidence >= min_confidence` $\rightarrow$ `NEW_CONTACT`
  - `confidence < min_confidence` $\rightarrow$ `UNCERTAIN`
- Prevents false-alarm sonar flashes from triggering emergency evasive maneuvers.

### Rule 2: Multi-Ping Persistence Rule
- If `observation_count >= 3` AND spatial drift is bounded AND time gap $\le 60$s:
  - Transition status to `PERSISTENT`.
  - Promotes evidence quality in System 1 Reflex Engine.

### Rule 3: Transient Detection & Graceful Retirement
- If a contact is observed 1 or 2 times and disappears from subsequent sweeps:
  - `missed_scans >= 2` $\rightarrow$ Transition status to `TRANSIENT`.
  - Track is preserved in historical buffer for operator forensics; not immediately deleted.

### Rule 4: System 1 Reflex Engine Enrichment
- When a detection maps to a `PERSISTENT` track with $\ge 3$ observations:
  - System 1 receives `{persistence_status: "PERSISTENT", observation_count: N, track_age_s: T}`.
  - Automatically reinforces `evidence_quality` to `EvidenceQuality.STRONG` with justification: `"Confirmed persistent sonar contact across N observations"`.

### Rule 5: System 2 Tactical Reasoning Enrichment
- System 2 receives full temporal context in `System2MissionContext.temporal`:
  - Summarized as: `"observed 5 times over 16.0s (PERSISTENT, mean conf=0.92)"`.
  - Prevents System 2 from hallucinating or fabricating persistence history.

---

## 5. Performance & Latency Benchmarks

Measured using 500 consecutive multi-contact track association and update iterations on real sonar waterfall scenarios (`scripts/generate_phase6_5_visuals_and_benchmark.py`):

| Metric | Measured Latency | Budget Target | Status |
|:---|:---:|:---:|:---:|
| **Min Latency** | **0.0181 ms** | $< 5.0$ ms | PASS |
| **Median Latency** | **0.0199 ms** | $< 5.0$ ms | PASS |
| **Mean Latency** | **0.0251 ms** | $< 5.0$ ms | PASS |
| **P95 Latency** | **0.0407 ms** | $< 5.0$ ms | PASS |
| **Max Latency** | **0.2521 ms** | $< 10.0$ ms | PASS |

Track matching adds **$< 0.03$ ms** overhead to the pipeline, completely negligible compared to image I/O or ONNX inference.

---

## 6. Verification & Automated Test Results

The suite includes **14 new tests** covering temporal tracking edge cases, plus **122 existing tests**:
$$\text{Total Test Suite: } 136 \text{ tests, } 0 \text{ failures, } 0 \text{ errors}$$

### New Test Suites
1. **`tests/test_temporal_tracking.py`**:
   - `test_single_observation_is_never_persistent`: Validates Rule 1.
   - `test_single_observation_low_confidence_is_uncertain`: Validates `UNCERTAIN` on low confidence.
   - `test_missing_gps_falls_back_to_image_coordinates`: Validates IoU/centroid pixel fallback.
   - `test_missing_timestamp_uses_valid_iso_default`: Validates timestamp fallback.
   - `test_simultaneous_multiple_contacts_tracked_separately`: Validates multi-target tracking.
2. **`tests/test_track_matching.py`**:
   - `test_repeated_same_contact_associates_to_same_track`: Validates sequential matching.
   - `test_different_class_does_not_associate`: Validates class separation.
   - `test_spatial_distance_exceeded_creates_new_track`: Validates max distance threshold.
   - `test_time_gap_exceeded_creates_new_track`: Validates max time gap threshold.
   - `test_deterministic_matching_repeatability`: Validates zero random drift.
3. **`tests/test_persistence_status.py`**:
   - `test_status_progression_from_new_to_persistent`: Validates progression `NEW_CONTACT` $\rightarrow$ `PERSISTENT`.
   - `test_transient_status_on_disappearance`: Validates transition to `TRANSIENT`.
   - `test_simultaneous_persistent_and_transient_contacts`: Validates mixed contact scenario.
   - `test_system1_reflex_enrichment_from_persistence`: Validates System 1 evidence quality promotion.

---

## 7. Visual Artifacts Generated

Stored under `reports/phase6_5/`:
- **`reports/phase6_5/temporal_tracking_timeline.svg`**: Vector timeline diagram illustrating ping-by-ping track progression and status transitions.
- **`reports/phase6_5/temporal_tracking_timeline.png`**: High-resolution raster visual showing track history, confidence trends, and persistence badges.
- **`reports/phase6_5/track_summary.json`**: Machine-readable JSON export of active tracks.

---

## 8. Known Limitations & Next Stabilization Phase

### Known Limitations
1. **Track Deletion / Pruning**: Currently, transient tracks remain in memory up to `max_track_history=20` records. A long-duration 24-hour survey will require a periodic garbage collector for tracks inactive for $> 1$ hour.
2. **Vehicle Motion Compensation**: Current pixel distance fallback does not integrate dead-reckoning DVL (Doppler Velocity Log) vectors when GPS drops out.
3. **Acoustic Occlusion**: If a persistent contact is occluded behind a seabed ridge for 3 pings, it will currently transition to `TRANSIENT` before re-associating.

### Recommended Next Stabilization Phase: Phase 7 — Operator UI & Real-Time Mission Cockpit
- Real-time visualization of `PERSISTENT` vs `TRANSIENT` contact trails on the sonar waterfall canvas.
- Display of confidence history sparklines and track IDs in the UI bounding box overlays.
- Integration of operator override commands for transient alerts.
