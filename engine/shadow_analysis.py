from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import cv2
import numpy as np


@dataclass
class ShadowConfig:
    """
    Configuration parameters for acoustic shadow segmentation.
    Controls the down-range search window and acoustic dropout thresholds.
    """
    search_distance_px: int = 150        # Maximum down-range search extent in pixels
    smoothing_window: int = 3            # 1D kernel size for smoothing intensity profile
    relative_drop_factor: float = 0.50   # Maximum fraction of ambient seabed intensity to qualify as acoustic shadow
    absolute_shadow_max: float = 55.0    # Absolute 8-bit ceiling for acoustic zero-return
    minimum_shadow_run: int = 4          # Contiguous low-intensity pixels required to confirm a shadow


@dataclass
class ShadowAnalysis:
    detected: bool
    shadow_length_px: Optional[float]
    shadow_length_m: Optional[float]
    start_point: Optional[List[int]]     # [x, y] start of acoustic shadow
    end_point: Optional[List[int]]       # [x, y] termination of acoustic shadow
    method: str
    status: str                          # MEASURED / UNAVAILABLE
    confidence: float
    search_direction: str                # DOWN_RANGE_RIGHT, DOWN_RANGE_LEFT, DOWN_RANGE_DOWN


class AcousticShadowAnalyzer:
    """
    Acoustic Shadow Intensity Segmentation Engine for Side-Scan Sonar.
    Detects physical acoustic dropouts (shadows) down-range of high-backscatter targets.
    Computes genuine pixel-level shadow lengths without arbitrary bounding box heuristics.
    """

    def __init__(self, config: ShadowConfig = None):
        self.config = config if config is not None else ShadowConfig()

    def _determine_downrange_direction(
        self,
        box: List[int],
        img_w: int,
        img_h: int,
        forced_direction: Optional[str] = None
    ) -> str:
        """
        In side-scan sonar, acoustic pulses travel outward from the central nadir line.
        Targets on the starboard (right) cast shadows to the right (+X).
        Targets on the port (left) cast shadows to the left (-X).
        """
        if forced_direction in ["DOWN_RANGE_RIGHT", "DOWN_RANGE_LEFT", "DOWN_RANGE_DOWN"]:
            return forced_direction

        cx = (box[0] + box[2]) / 2.0
        nadir_x = img_w / 2.0

        if cx >= nadir_x:
            return "DOWN_RANGE_RIGHT"
        else:
            return "DOWN_RANGE_LEFT"

    def analyze(
        self,
        img_bgr: np.ndarray,
        bbox_xyxy: List[int],
        meters_per_pixel: Optional[float] = None,
        forced_direction: Optional[str] = None
    ) -> ShadowAnalysis:
        """
        Analyzes the acoustic return down-range of a target bounding box [x1, y1, x2, y2].
        Returns ShadowAnalysis with measured pixel length, and metric length if calibration exists.
        """
        if img_bgr is None or img_bgr.size == 0:
            return ShadowAnalysis(
                detected=False,
                shadow_length_px=None,
                shadow_length_m=None,
                start_point=None,
                end_point=None,
                method="image_unavailable",
                status="UNAVAILABLE",
                confidence=0.0,
                search_direction="NONE"
            )

        # Convert to single-channel 8-bit grayscale
        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        img_h, img_w = gray.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]

        # Clamp box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            return ShadowAnalysis(
                detected=False,
                shadow_length_px=None,
                shadow_length_m=None,
                start_point=None,
                end_point=None,
                method="invalid_bounding_box",
                status="UNAVAILABLE",
                confidence=0.0,
                search_direction="NONE"
            )

        direction = self._determine_downrange_direction([x1, y1, x2, y2], img_w, img_h, forced_direction)
        center_y = (y1 + y2) // 2

        # Compute ambient seabed backscatter intensity from surroundings
        # Sample margins around the box
        margin = 30
        ambient_samples = []
        if y1 > margin:
            ambient_samples.append(gray[max(0, y1 - margin):y1, max(0, x1 - margin):min(img_w, x2 + margin)])
        if y2 + margin < img_h:
            ambient_samples.append(gray[y2:min(img_h, y2 + margin), max(0, x1 - margin):min(img_w, x2 + margin)])

        if ambient_samples and any(s.size > 0 for s in ambient_samples):
            all_ambient = np.concatenate([s.flatten() for s in ambient_samples if s.size > 0])
            ambient_level = float(np.median(all_ambient))
        else:
            ambient_level = float(np.median(gray))

        # Shadow cutoff threshold: must drop below fraction of ambient level and below absolute ceiling
        shadow_thresh = min(self.config.absolute_shadow_max, ambient_level * self.config.relative_drop_factor)
        shadow_thresh = max(15.0, shadow_thresh) # Reasonable floor for noise

        # Extract down-range search strip
        strip_y1 = max(0, center_y - max(2, (y2 - y1) // 4))
        strip_y2 = min(img_h, center_y + max(2, (y2 - y1) // 4))

        if direction == "DOWN_RANGE_RIGHT":
            start_x = x2
            end_x = min(img_w, x2 + self.config.search_distance_px)
            if end_x <= start_x:
                return self._no_shadow_result(direction, "edge_reached")
            strip = gray[strip_y1:strip_y2, start_x:end_x]
            profile = np.mean(strip, axis=0) # 1D along down-range X
            start_coord = [start_x, center_y]

        elif direction == "DOWN_RANGE_LEFT":
            start_x = x1
            end_x = max(0, x1 - self.config.search_distance_px)
            if start_x <= end_x:
                return self._no_shadow_result(direction, "edge_reached")
            strip = gray[strip_y1:strip_y2, end_x:start_x]
            # Flip so index 0 is nearest to the object
            profile = np.mean(strip, axis=0)[::-1]
            start_coord = [start_x, center_y]

        else: # DOWN_RANGE_DOWN
            start_y = y2
            end_y = min(img_h, y2 + self.config.search_distance_px)
            if end_y <= start_y:
                return self._no_shadow_result(direction, "edge_reached")
            strip = gray[start_y:end_y, x1:x2]
            profile = np.mean(strip, axis=1) # 1D along down-range Y
            start_coord = [(x1 + x2) // 2, start_y]

        # 1D smoothing
        if len(profile) >= self.config.smoothing_window:
            kernel = np.ones(self.config.smoothing_window) / self.config.smoothing_window
            profile = np.convolve(profile, kernel, mode='same')

        # Trace shadow run: consecutive low-intensity pixels
        shadow_length_px = 0
        in_shadow = False
        consecutive_shadow = 0

        # Allow slight initial boundary tolerance (first 2 pixels might be target transition)
        scan_offset = min(2, len(profile) - 1)
        for i in range(scan_offset, len(profile)):
            val = profile[i]
            if val <= shadow_thresh:
                consecutive_shadow += 1
                if consecutive_shadow >= self.config.minimum_shadow_run:
                    in_shadow = True
                shadow_length_px = i + 1
            else:
                if in_shadow:
                    # Shadow has terminated (intensity recovered back to seabed backscatter)
                    break
                else:
                    consecutive_shadow = 0

        # Verify shadow validity
        if not in_shadow or shadow_length_px < self.config.minimum_shadow_run:
            return self._no_shadow_result(direction, "no_acoustic_dropout_detected")

        # Compute termination point
        if direction == "DOWN_RANGE_RIGHT":
            end_coord = [start_coord[0] + shadow_length_px, center_y]
        elif direction == "DOWN_RANGE_LEFT":
            end_coord = [start_coord[0] - shadow_length_px, center_y]
        else:
            end_coord = [start_coord[0], start_coord[1] + shadow_length_px]

        # Metric length: ONLY if valid scale calibration provided
        if meters_per_pixel is not None and meters_per_pixel > 0:
            shadow_length_m = round(shadow_length_px * meters_per_pixel, 3)
            status = "DERIVED"
        else:
            shadow_length_m = None
            status = "MEASURED" # Measured in pixels, metric unavailable

        confidence = min(0.95, round(0.50 + (min(shadow_length_px, 50) / 100.0), 2))

        return ShadowAnalysis(
            detected=True,
            shadow_length_px=float(shadow_length_px),
            shadow_length_m=shadow_length_m,
            start_point=start_coord,
            end_point=end_coord,
            method="1d_downrange_intensity_profiling",
            status=status,
            confidence=confidence,
            search_direction=direction
        )

    def _no_shadow_result(self, direction: str, reason: str) -> ShadowAnalysis:
        return ShadowAnalysis(
            detected=False,
            shadow_length_px=None,
            shadow_length_m=None,
            start_point=None,
            end_point=None,
            method=f"acoustic_search_{reason}",
            status="UNAVAILABLE",
            confidence=0.0,
            search_direction=direction
        )
