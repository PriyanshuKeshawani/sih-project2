"""
engine/temporal_tracking.py
Temporal Multi-Ping Persistence & Track Association Engine.

SIH 2026 Problem Statement 26057 — MoES / NIOT Chennai
Tracks underwater sonar contacts across repeated waterfall pings and survey sweeps.
Classifies contacts into:
  - NEW_CONTACT: First observation in survey.
  - PERSISTENT: Confirmed across >= N spatially and temporally consistent scans.
  - TRANSIENT: Appeared once or twice and disappeared in subsequent pings.
  - UNCERTAIN: Conflicting evidence, uncalibrated drift, or marginal confidence.

Architectural Guarantees:
1. Strict Single-Observation Rule: 1 observation is NEVER PERSISTENT.
2. Zero Sensor Fabrication: Falls back to image bbox distance when GPS is unavailable.
3. Decoupled & Non-blocking: Sub-millisecond track association.
"""

import math
import time
import uuid
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("temporal_tracking")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [TRACK] %(message)s")


# =====================================================================
# 1. Enums and Configuration
# =====================================================================

class PersistenceStatus(str, Enum):
    NEW_CONTACT = "NEW_CONTACT"
    PERSISTENT = "PERSISTENT"
    TRANSIENT = "TRANSIENT"
    UNCERTAIN = "UNCERTAIN"


class TrackingConfig(BaseModel):
    """
    Configurable parameters for spatial-temporal multi-ping association.
    Not hardcoded; fully adjustable per survey or vehicle velocity.
    """
    max_position_distance_m: float = Field(default=15.0, description="Max GPS distance in meters for association.")
    max_image_distance_px: float = Field(default=120.0, description="Max pixel center distance if GPS unavailable.")
    min_iou: float = Field(default=0.15, description="Minimum box IoU for consecutive waterfall pings.")
    max_time_gap_s: float = Field(default=60.0, description="Max allowable time gap between scans in seconds.")
    min_observations_for_persistent: int = Field(default=3, description="Observations required to establish PERSISTENT status.")
    min_confidence: float = Field(default=0.40, description="Minimum confidence threshold for valid track initialization.")
    transient_miss_threshold: int = Field(default=2, description="Missed scans before unconfirmed contact marked TRANSIENT.")


# =====================================================================
# 2. Track Data Models
# =====================================================================

class TemporalTrack(BaseModel):
    """
    Stateful acoustic contact track maintained across successive survey sweeps.
    """
    track_id: str = Field(default_factory=lambda: f"TRK-{uuid.uuid4().hex[:6].upper()}")
    survey_id: str = Field(description="Survey session or transect identifier.")
    target_class: str = Field(description="Primary classified anomaly class name.")
    first_seen: str = Field(description="ISO timestamp of initial detection.")
    last_seen: str = Field(description="ISO timestamp of most recent observation.")
    first_seen_epoch: float = Field(default_factory=time.time)
    last_seen_epoch: float = Field(default_factory=time.time)
    observation_count: int = Field(default=1)
    missed_scans: int = Field(default=0)
    confidence_history: List[float] = Field(default_factory=list)
    positions: List[Dict[str, Any]] = Field(default_factory=list)
    bbox_history: List[Dict[str, Any]] = Field(default_factory=list)
    physics_history: List[Dict[str, Any]] = Field(default_factory=list)
    system1_history: List[Dict[str, Any]] = Field(default_factory=list)
    persistence_status: str = Field(default=PersistenceStatus.NEW_CONTACT.value)
    active: bool = Field(default=True)
    operator_status: str = Field(default="UNREVIEWED")
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)
    hidden: bool = Field(default=False)

    @property
    def track_age_s(self) -> float:
        return max(0.0, self.last_seen_epoch - self.first_seen_epoch)

    @property
    def mean_confidence(self) -> float:
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)

    def confirm(self, operator_id: str = "OPERATOR_1", note: str = ""):
        self.operator_status = "CONFIRMED_FOR_MISSION"
        self.audit_log.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": "CONFIRM_CONTACT",
            "operator": operator_id,
            "note": note or "Operator confirmed contact for mission workflow (not ground truth)."
        })

    def dismiss(self, operator_id: str = "OPERATOR_1", note: str = ""):
        self.operator_status = "DISMISSED"
        self.audit_log.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": "DISMISS_CONTACT",
            "operator": operator_id,
            "note": note or "Operator dismissed contact as transient/spurious acoustic return."
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "survey_id": self.survey_id,
            "class": self.target_class,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observation_count": self.observation_count,
            "missed_scans": self.missed_scans,
            "confidence_history": self.confidence_history,
            "positions": self.positions,
            "bbox_history": self.bbox_history,
            "physics_history": self.physics_history,
            "persistence_status": self.persistence_status,
            "track_age_s": round(self.track_age_s, 2),
            "mean_confidence": round(self.mean_confidence, 3),
            "active": self.active,
            "operator_status": self.operator_status,
            "audit_log": self.audit_log,
            "hidden": self.hidden
        }


# =====================================================================
# 3. Spatial Distance & IoU Helpers
# =====================================================================

def calculate_haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle metric distance in meters between two WGS-84 coordinates."""
    r_earth = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def calculate_bbox_iou(boxA: Dict[str, Any], boxB: Dict[str, Any]) -> float:
    """Computes Intersection-over-Union between two bounding boxes."""
    xA1 = boxA.get("x", 0)
    yA1 = boxA.get("y", 0)
    xA2 = xA1 + boxA.get("w", 0)
    yA2 = yA1 + boxA.get("h", 0)

    xB1 = boxB.get("x", 0)
    yB1 = boxB.get("y", 0)
    xB2 = xB1 + boxB.get("w", 0)
    yB2 = yB1 + boxB.get("h", 0)

    inter_x1 = max(xA1, xB1)
    inter_y1 = max(yA1, yB1)
    inter_x2 = min(xA2, xB2)
    inter_y2 = min(yA2, yB2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    boxA_area = max(1, (xA2 - xA1) * (yA2 - yA1))
    boxB_area = max(1, (xB2 - xB1) * (yB2 - yB1))
    union_area = boxA_area + boxB_area - inter_area

    return inter_area / float(union_area) if union_area > 0 else 0.0


def calculate_box_center_distance(boxA: Dict[str, Any], boxB: Dict[str, Any]) -> float:
    """Computes Euclidean pixel distance between the centers of two bounding boxes."""
    cA_x = boxA.get("x", 0) + boxA.get("w", 0) / 2.0
    cA_y = boxA.get("y", 0) + boxA.get("h", 0) / 2.0
    cB_x = boxB.get("x", 0) + boxB.get("w", 0) / 2.0
    cB_y = boxB.get("y", 0) + boxB.get("h", 0) / 2.0
    return math.hypot(cA_x - cB_x, cA_y - cB_y)


def _normalize_box(obs: Dict[str, Any]) -> Dict[str, Any]:
    if not obs:
        return {}
    if "box" in obs and isinstance(obs["box"], dict) and "x" in obs["box"]:
        return obs["box"]
    if "bbox_xyxy" in obs and isinstance(obs["bbox_xyxy"], (list, tuple)) and len(obs["bbox_xyxy"]) == 4:
        x1, y1, x2, y2 = obs["bbox_xyxy"]
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
    if "x" in obs and "y" in obs and "w" in obs and "h" in obs:
        return {"x": obs["x"], "y": obs["y"], "w": obs["w"], "h": obs["h"]}
    return {}


# =====================================================================
# 4. Temporal Multi-Ping Tracker Engine
# =====================================================================

class TemporalPersistenceTracker:
    """
    Maintains active track states for a survey and associates multi-ping detections.
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self.tracks: Dict[str, TemporalTrack] = {}  # track_id -> track

    def get_tracks_for_survey(self, survey_id: str) -> List[TemporalTrack]:
        return [t for t in self.tracks.values() if t.survey_id == survey_id]

    def get_track(self, track_id: str) -> Optional[TemporalTrack]:
        return self.tracks.get(track_id)

    def process_scan_observations(
        self,
        survey_id: str,
        scan_id: str,
        observations: List[Dict[str, Any]],
        timestamp_str: Optional[str] = None,
        timestamp_epoch: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Updates temporal track states given a list of detections from a single scan.
        Returns the original detections enriched with:
          - track_id
          - persistence_status
          - observation_count
          - track_age_s
        """
        t_str = timestamp_str or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t_epoch = timestamp_epoch if timestamp_epoch is not None else time.time()

        # Step 1: Filter active tracks for this survey
        active_tracks = [t for t in self.tracks.values() if t.survey_id == survey_id and t.active]

        # Step 2: Compute bipartite association matrix
        matches: List[Tuple[int, TemporalTrack]] = []
        matched_obs_indices = set()
        matched_track_ids = set()

        # Score candidate pairings (lower cost = better match)
        candidates = []
        for obs_idx, obs in enumerate(observations):
            obs_class = obs.get("class", "unknown").lower()
            obs_conf = float(obs.get("confidence", 0.0))
            obs_box = _normalize_box(obs)
            obs_geo = obs.get("geo", {})

            for track in active_tracks:
                if track.track_id in matched_track_ids:
                    continue

                # Class compatibility: must match
                if track.target_class.lower() != obs_class:
                    continue

                # Temporal validity: time gap check
                time_diff = abs(t_epoch - track.last_seen_epoch)
                if time_diff > self.config.max_time_gap_s:
                    continue

                # Spatial correlation
                cost = float("inf")
                is_match = False

                last_geo = track.positions[-1] if track.positions else None
                last_box = track.bbox_history[-1] if track.bbox_history else None

                # Metric GPS comparison if available
                has_gps = (
                    obs_geo and obs_geo.get("lat") is not None and obs_geo.get("lon") is not None
                    and last_geo and last_geo.get("lat") is not None and last_geo.get("lon") is not None
                )

                if has_gps:
                    dist_m = calculate_haversine_distance_m(
                        obs_geo["lat"], obs_geo["lon"], last_geo["lat"], last_geo["lon"]
                    )
                    if dist_m <= self.config.max_position_distance_m:
                        cost = dist_m
                        is_match = True
                elif obs_box and last_box:
                    # Fallback to image-space IoU and center distance
                    iou = calculate_bbox_iou(obs_box, last_box)
                    px_dist = calculate_box_center_distance(obs_box, last_box)
                    if iou >= self.config.min_iou or px_dist <= self.config.max_image_distance_px:
                        cost = px_dist / (iou + 0.1)
                        is_match = True

                if is_match:
                    candidates.append((cost, obs_idx, track))

        # Sort candidate matches by lowest cost (greedy assignment)
        candidates.sort(key=lambda x: x[0])
        for cost, obs_idx, track in candidates:
            if obs_idx not in matched_obs_indices and track.track_id not in matched_track_ids:
                matches.append((obs_idx, track))
                matched_obs_indices.add(obs_idx)
                matched_track_ids.add(track.track_id)

        enriched_results = [None] * len(observations)

        # Step 3: Update matched tracks
        for obs_idx, track in matches:
            obs = observations[obs_idx]
            obs_conf = float(obs.get("confidence", 0.0))
            obs_box = _normalize_box(obs)
            obs_geo = obs.get("geo", {})
            obs_phys = obs.get("physics", {})
            obs_s1 = obs.get("system1", {}) or obs.get("reflex", {})

            track.observation_count += 1
            track.last_seen = t_str
            track.last_seen_epoch = t_epoch
            track.missed_scans = 0
            track.confidence_history.append(round(obs_conf, 3))
            if obs_geo:
                track.positions.append(obs_geo)
            if obs_box:
                track.bbox_history.append(obs_box)
            if obs_phys:
                track.physics_history.append(obs_phys)
            if obs_s1:
                track.system1_history.append(obs_s1)

            # Evaluate persistence
            if track.observation_count >= self.config.min_observations_for_persistent:
                if track.mean_confidence >= self.config.min_confidence:
                    track.persistence_status = PersistenceStatus.PERSISTENT.value
                else:
                    track.persistence_status = PersistenceStatus.UNCERTAIN.value
            else:
                track.persistence_status = PersistenceStatus.NEW_CONTACT.value

            enriched = dict(obs)
            enriched["track_id"] = track.track_id
            enriched["persistence_status"] = track.persistence_status
            enriched["observation_count"] = track.observation_count
            enriched["track_age_s"] = round(track.track_age_s, 2)
            enriched_results[obs_idx] = enriched

        # Step 4: Initialize new tracks for unmatched observations
        for obs_idx, obs in enumerate(observations):
            if obs_idx in matched_obs_indices:
                continue

            obs_class = obs.get("class", "unknown").lower()
            obs_conf = float(obs.get("confidence", 0.0))
            obs_box = _normalize_box(obs)
            obs_geo = obs.get("geo", {})
            obs_phys = obs.get("physics", {})
            obs_s1 = obs.get("system1", {}) or obs.get("reflex", {})

            # Single observation rule: MUST NOT be PERSISTENT
            init_status = (
                PersistenceStatus.NEW_CONTACT.value
                if obs_conf >= self.config.min_confidence
                else PersistenceStatus.UNCERTAIN.value
            )

            new_track = TemporalTrack(
                survey_id=survey_id,
                target_class=obs_class,
                first_seen=t_str,
                last_seen=t_str,
                first_seen_epoch=t_epoch,
                last_seen_epoch=t_epoch,
                observation_count=1,
                missed_scans=0,
                confidence_history=[round(obs_conf, 3)],
                positions=[obs_geo] if obs_geo else [],
                bbox_history=[obs_box] if obs_box else [],
                physics_history=[obs_phys] if obs_phys else [],
                system1_history=[obs_s1] if obs_s1 else [],
                persistence_status=init_status
            )
            self.tracks[new_track.track_id] = new_track

            enriched = dict(obs)
            enriched["track_id"] = new_track.track_id
            enriched["persistence_status"] = new_track.persistence_status
            enriched["observation_count"] = 1
            enriched["track_age_s"] = 0.0
            enriched_results[obs_idx] = enriched

        # Step 5: Age unmatched existing tracks (transient detection)
        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                track.missed_scans += 1
                if track.observation_count < self.config.min_observations_for_persistent:
                    if track.missed_scans >= self.config.transient_miss_threshold:
                        track.persistence_status = PersistenceStatus.TRANSIENT.value
                else:
                    # Was persistent before, but now missing
                    if track.missed_scans >= (self.config.transient_miss_threshold * 2):
                        track.active = False  # Retire track from active matching

        return enriched_results
