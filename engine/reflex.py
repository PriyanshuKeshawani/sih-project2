import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from engine.laya_adapter import (
    System1InputState,
    System1Decision,
    System1Manager,
    System1SafetyGuardrails
)

logger = logging.getLogger("system1_reflex")


class DecisionPrimitive(str, Enum):
    EMERGENCY_PROP_HAZARD = "EMERGENCY_PROP_HAZARD"
    LOITER_AND_RESCAN = "LOITER_AND_RESCAN"
    PASSIVE_LOG = "PASSIVE_LOG"


class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class EvidenceQuality(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    UNKNOWN = "UNKNOWN"


# Scientific hazard taxonomy table (Section 5)
HAZARD_TAXONOMY = {
    "ghost_net": {
        "hazard_category": "entanglement / marine-debris hazard",
        "scientific_description": "Synthetic derelict netting posing immediate propeller foul & marine entrapment risk",
        "base_weight": 9.0
    },
    "mine_cylinder": {
        "hazard_category": "mine-cylinder-class sonar contact",
        "scientific_description": "Cylindrical metallic/composite acoustic signature requiring cautious non-contact classification",
        "base_weight": 8.5
    },
    "submarine_pipeline": {
        "hazard_category": "critical infrastructure / benthic hazard",
        "scientific_description": "Continuous underwater asset requiring standoff distance & seafloor burial monitoring",
        "base_weight": 7.0
    },
    "shipwreck": {
        "hazard_category": "submerged structure / navigation hazard",
        "scientific_description": "Large rigid seafloor structural anomaly presenting hull collision hazard",
        "base_weight": 6.5
    },
    "crab_pot": {
        "hazard_category": "fishing gear / marine-debris contact",
        "scientific_description": "Localized benthic trap with minor tethering hazard to low-altitude AUVs",
        "base_weight": 3.0
    }
}


@dataclass
class ReflexConfig:
    """Centralized thresholds for System 1 deterministic reflex decisions."""
    high_confidence_threshold: float = 0.70
    medium_confidence_threshold: float = 0.45
    emergency_risk_threshold: float = 7.5
    rescan_risk_threshold: float = 4.5
    elevation_warning_threshold_m: float = 0.80


@dataclass
class MissionEvent:
    """Structured mission log event representing a verified System 1 reflex decision."""
    event_id: str
    timestamp: str
    survey_id: Optional[str]
    decision_primitive: str
    hazard_score: float
    alert_severity: str
    contact: Dict[str, Any]
    evidence_quality: str
    reasons: List[str]
    navigation: Dict[str, Any]
    fingerprint: str
    system1_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "survey_id": self.survey_id,
            "decision_primitive": self.decision_primitive,
            "hazard_score": self.hazard_score,
            "alert_severity": self.alert_severity,
            "contact": self.contact,
            "evidence_quality": self.evidence_quality,
            "reasons": self.reasons,
            "navigation": self.navigation,
            "fingerprint": self.fingerprint
        }
        if self.system1_metadata:
            res["system1_metadata"] = self.system1_metadata
        return res


class MissionEventQueue:
    """In-memory event queue abstraction with same-scan deduplication support."""
    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self._events: List[MissionEvent] = []
        self._fingerprints: set = set()

    def add_event(self, event: MissionEvent) -> bool:
        """Adds event if not duplicate within the active scan / session."""
        if event.fingerprint in self._fingerprints:
            return False  # Debounced duplicate

        if len(self._events) >= self.max_capacity:
            oldest = self._events.pop(0)
            self._fingerprints.discard(oldest.fingerprint)

        self._events.append(event)
        self._fingerprints.add(event.fingerprint)
        return True

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events[-limit:]]

    def get_critical_events(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._events if e.alert_severity == AlertSeverity.CRITICAL.value]

    def clear(self):
        self._events.clear()
        self._fingerprints.clear()

    def __len__(self) -> int:
        return len(self._events)


class System1ReflexEngine:
    """
    Unified System 1 Edge Reflex & Decision Engine.
    Coordinates between the real Laya non-autoregressive decision engine and the
    calibrated deterministic fallback engine.
    Applies deterministic safety guardrails on all decisions.
    """

    def __init__(self, config: Optional[ReflexConfig] = None, local_laya_dir: Optional[str] = None, enable_laya: bool = True):
        self.config = config if config is not None else ReflexConfig()
        self.event_queue = MissionEventQueue()
        self.enable_laya = enable_laya
        self.system1_manager = System1Manager(local_model_dir=local_laya_dir, enable_laya=enable_laya)
        # Backward compatibility alias
        self.laya_adapter = self.system1_manager


    def evaluate_evidence_quality(
        self,
        confidence: float,
        physics_data: Optional[Dict[str, Any]],
        geo_data: Optional[Dict[str, Any]]
    ) -> EvidenceQuality:
        """
        Assesses evidence robustness based on detector confidence, acoustic shadow,
        metric calibration, and geographic provenance.
        """
        shadow_detected = False
        has_metric_physics = False
        has_valid_geo = False

        if physics_data:
            shadow_detected = bool(physics_data.get("shadow_detected", False))
            has_metric_physics = bool(
                physics_data.get("shadow_length_m") is not None or
                physics_data.get("elevation_m") is not None
            )

        if geo_data:
            has_valid_geo = geo_data.get("status") in ["MEASURED", "DERIVED", "REAL"]

        # Strong: high confidence + acoustic shadow confirmed + metric or geo data
        if confidence >= self.config.high_confidence_threshold and shadow_detected and (has_metric_physics or has_valid_geo):
            return EvidenceQuality.STRONG

        # Moderate: high confidence with partial acoustic evidence OR medium confidence with shadow
        if (confidence >= self.config.high_confidence_threshold) or (confidence >= self.config.medium_confidence_threshold and shadow_detected):
            return EvidenceQuality.MODERATE

        # Weak: low confidence and no shadow
        if confidence < self.config.medium_confidence_threshold and not shadow_detected:
            return EvidenceQuality.WEAK

        return EvidenceQuality.MODERATE

    def compute_deterministic_risk(
        self,
        cls_name: str,
        confidence: float,
        elevation_m: Optional[float],
        evidence_quality: EvidenceQuality
    ) -> float:
        """
        Computes deterministic hazard score on a 1.0 - 10.0 scale.
        """
        tax = HAZARD_TAXONOMY.get(cls_name, {
            "hazard_category": "unclassified sonar contact",
            "base_weight": 5.0
        })
        base_score = tax["base_weight"]
        conf_factor = 0.5 + 0.5 * max(0.0, min(1.0, confidence))

        if elevation_m is not None and elevation_m > 0:
            elevation_bonus = min(1.0, float(elevation_m) * 0.4)
        else:
            elevation_bonus = 0.0

        if evidence_quality == EvidenceQuality.STRONG:
            evidence_factor = 1.1
        elif evidence_quality == EvidenceQuality.MODERATE:
            evidence_factor = 1.0
        elif evidence_quality == EvidenceQuality.WEAK:
            evidence_factor = 0.85
        else:
            evidence_factor = 0.90

        raw_risk = (base_score * conf_factor + elevation_bonus) * evidence_factor
        return max(1.0, min(10.0, round(raw_risk, 1)))

    def decide_primitive(
        self,
        cls_name: str,
        confidence: float,
        hazard_score: float,
        evidence_quality: EvidenceQuality,
        elevation_m: Optional[float]
    ) -> tuple[DecisionPrimitive, AlertSeverity, List[str]]:
        """
        Evaluates deterministic decision primitive, alert severity, and concise machine-auditable reasons.
        """
        reasons = []
        tax = HAZARD_TAXONOMY.get(cls_name, {
            "hazard_category": "unclassified contact",
            "base_weight": 5.0
        })

        reasons.append(f"{cls_name} ({tax['hazard_category']})")
        reasons.append(f"detection confidence: {confidence:.2f}")
        reasons.append(f"evidence quality: {evidence_quality.value}")

        if elevation_m is not None:
            reasons.append(f"obstacle elevation: {elevation_m:.2f}m")
        else:
            reasons.append("elevation: unavailable")

        is_elevated_net = (cls_name == "ghost_net" and elevation_m is not None and elevation_m >= self.config.elevation_warning_threshold_m)
        is_high_threat = (hazard_score >= self.config.emergency_risk_threshold and evidence_quality in [EvidenceQuality.STRONG, EvidenceQuality.MODERATE])

        if is_elevated_net or (is_high_threat and confidence >= self.config.high_confidence_threshold):
            primitive = DecisionPrimitive.EMERGENCY_PROP_HAZARD
            severity = AlertSeverity.CRITICAL
            if is_elevated_net:
                reasons.append("propeller fouling hazard: elevated net structure detected above seabed")
            else:
                reasons.append(f"critical hazard threshold exceeded (score: {hazard_score:.1f} >= {self.config.emergency_risk_threshold})")
        elif (hazard_score >= self.config.rescan_risk_threshold) or (confidence < self.config.high_confidence_threshold and cls_name in ["ghost_net", "mine_cylinder"]):
            primitive = DecisionPrimitive.LOITER_AND_RESCAN
            severity = AlertSeverity.WARNING
            reasons.append("secondary observation recommended: contact requires verification look-angle")
        else:
            primitive = DecisionPrimitive.PASSIVE_LOG
            severity = AlertSeverity.INFO
            reasons.append("nominal contact recorded in mission log")

        return primitive, severity, reasons

    def generate_navigation_recommendation(
        self,
        primitive: DecisionPrimitive,
        cls_name: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Formulates navigation recommendation.
        CRITICAL SAFETY RULE: Never fabricate maneuvers.
        If vehicle heading, speed, or state is missing, returns status='UNAVAILABLE'.
        """
        if metadata is None:
            return {
                "maneuver": None,
                "status": "UNAVAILABLE",
                "reason": "missing_navigation_metadata"
            }

        heading = metadata.get("heading_deg")
        alt_m = metadata.get("sonar_altitude_m")

        if heading is None or alt_m is None:
            return {
                "maneuver": None,
                "status": "UNAVAILABLE",
                "reason": "heading_or_altitude_telemetry_missing"
            }

        if primitive == DecisionPrimitive.EMERGENCY_PROP_HAZARD:
            return {
                "maneuver": "RECOMMEND_STANDOFF_ASCENT",
                "status": "ADVISORY_ONLY",
                "advisory": f"Increase altitude by +3.0m above current {alt_m:.1f}m to clear fouling obstacle",
                "recommended_standoff_m": 15.0
            }
        elif primitive == DecisionPrimitive.LOITER_AND_RESCAN:
            return {
                "maneuver": "RECOMMEND_ORBITAL_RESCAN",
                "status": "ADVISORY_ONLY",
                "advisory": f"Plan 360-degree re-acquisition circle at 20m range from contact datum",
                "recommended_standoff_m": 20.0
            }
        else:
            return {
                "maneuver": "MAINTAIN_SURVEY_COURSE",
                "status": "NOMINAL",
                "advisory": f"Maintain planned transect on heading {heading:.1f} deg",
                "recommended_standoff_m": None
            }

    def process_reflex(
        self,
        detection: Dict[str, Any],
        physics: Optional[Dict[str, Any]] = None,
        geo: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scan_id: Optional[str] = None,
        temporal: Optional[Dict[str, Any]] = None,
        allow_laya: bool = True
    ) -> Dict[str, Any]:
        """
        Full System 1 Reflex evaluation for a single detection with physics & geo context.
        Routes through System1Manager (Laya primary -> Deterministic fallback) with guardrails.
        Enriches decision with temporal multi-ping persistence status when available.
        """
        t0 = time.perf_counter()

        cls_name = detection.get("class", "unclassified")
        confidence = float(detection.get("confidence", 0.0))
        bbox = detection.get("bbox_xyxy") or [0, 0, 0, 0]
        tax_status = detection.get("taxonomy_status", "STANDARD_CLASS")

        elevation_m = physics.get("elevation_m") if physics else None
        shadow_len_m = physics.get("shadow_length_m") if physics else None
        shadow_det = bool(physics.get("shadow_detected", False)) if physics else False
        phys_prov = physics.get("provenance", "UNAVAILABLE") if physics else "UNAVAILABLE"
        geo_status = geo.get("status", "UNAVAILABLE") if geo else "UNAVAILABLE"

        sonar_alt_m = metadata.get("sonar_altitude_m") if metadata else None
        heading_deg = metadata.get("heading_deg") if metadata else None

        # 1. Evaluate Evidence Quality (enriched by temporal persistence)
        evidence_quality = self.evaluate_evidence_quality(confidence, physics, geo)
        if temporal:
            persistence = temporal.get("persistence_status")
            obs_count = temporal.get("observation_count", 1)
            if persistence == "PERSISTENT" and obs_count >= 3:
                evidence_quality = EvidenceQuality.STRONG

        # 2. Build Structured System 1 Input State (Section 6: No raw images sent to Laya)
        input_state = System1InputState(
            contact_class=cls_name,
            detector_confidence=confidence,
            taxonomy_status=tax_status,
            shadow_detected=shadow_det,
            shadow_length_m=shadow_len_m,
            elevation_m=elevation_m,
            physics_provenance=phys_prov,
            geo_status=geo_status,
            evidence_quality=evidence_quality.value,
            sonar_altitude_m=sonar_alt_m,
            heading_deg=heading_deg
        )

        # 3. Route through System 1 Manager (Laya Primary if allowed, or sub-millisecond Fallback)
        if allow_laya:
            s1_decision: System1Decision = self.system1_manager.decide(input_state)
        else:
            s1_decision: System1Decision = self.system1_manager.fallback_engine.decide(input_state)

        # Append temporal persistence justification if confirmed
        if temporal and temporal.get("persistence_status") == "PERSISTENT":
            s1_decision.reasons.append(
                f"Multi-ping persistence confirmed across {temporal.get('observation_count')} observations ({temporal.get('track_age_s')}s)"
            )

        # 4. Parse Primitive & Severity for mission event
        try:
            primitive = DecisionPrimitive(s1_decision.decision_primitive)
        except ValueError:
            primitive = DecisionPrimitive.PASSIVE_LOG

        if primitive == DecisionPrimitive.EMERGENCY_PROP_HAZARD:
            severity = AlertSeverity.CRITICAL
        elif primitive == DecisionPrimitive.LOITER_AND_RESCAN:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        # 5. Advisory Navigation Recommendation (Strictly advisory, no direct thruster actuation)
        nav_advisory = self.generate_navigation_recommendation(primitive, cls_name, metadata)

        # 6. Debounce Fingerprint
        fp_x1 = int(bbox[0]) // 10
        fp_y1 = int(bbox[1]) // 10
        fp_x2 = int(bbox[2]) // 10
        fp_y2 = int(bbox[3]) // 10
        fingerprint = f"{scan_id or 'default'}_{cls_name}_{fp_x1}_{fp_y1}_{fp_x2}_{fp_y2}"

        # 7. Create Structured Mission Event
        survey_id = metadata.get("survey_id") if metadata else None
        event = MissionEvent(
            event_id=str(uuid.uuid4()),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            survey_id=survey_id,
            decision_primitive=primitive.value,
            hazard_score=s1_decision.hazard_score,
            alert_severity=severity.value,
            contact={
                "class": cls_name,
                "confidence": round(confidence, 3),
                "bbox_xyxy": bbox,
                "hazard_category": HAZARD_TAXONOMY.get(cls_name, {}).get("hazard_category", "unclassified")
            },
            evidence_quality=evidence_quality.value,
            reasons=s1_decision.reasons,
            navigation=nav_advisory,
            fingerprint=fingerprint,
            system1_metadata={
                "engine": s1_decision.model,
                "status": s1_decision.status,
                "laya_confidence": s1_decision.laya_confidence,
                "needs_operator_review": s1_decision.needs_operator_review,
                "guardrail_applied": s1_decision.guardrail_applied
            }
        )

        is_new_event = self.event_queue.add_event(event)
        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        # Log
        logger.info(
            f"[SYSTEM1] engine={s1_decision.model} class={cls_name} "
            f"conf={confidence:.3f} score={s1_decision.hazard_score:.1f} "
            f"primitive={primitive.value} latency={s1_decision.latency_ms:.2f}ms"
        )

        # Status text for edge engine
        status_info = self.system1_manager.get_status()
        edge_engine_desc = f"Laya System 1 ({status_info['status']})" if status_info.get("active_engine") == "laya" else f"Deterministic Rule Fallback ({status_info.get('status', 'ACTIVE')})"

        return {
            "decision_primitive": primitive.value,
            "hazard_score": s1_decision.hazard_score,
            "alert_severity": severity.value,
            "evidence_quality": evidence_quality.value,
            "reasons": s1_decision.reasons,
            "recommended_maneuver": nav_advisory.get("advisory") or "Navigation advisory unavailable",
            "navigation": nav_advisory,
            "event_id": event.event_id,
            "is_new_event": is_new_event,
            "latency_ms": s1_decision.latency_ms,
            "total_reflex_latency_ms": total_latency_ms,
            "edge_engine": edge_engine_desc,
            "needs_operator_review": s1_decision.needs_operator_review,
            "laya_confidence": s1_decision.laya_confidence,
            "guardrail_applied": s1_decision.guardrail_applied,
            "system1": {
                "engine": s1_decision.model,
                "decision_primitive": primitive.value,
                "hazard_score": s1_decision.hazard_score,
                "needs_operator_review": s1_decision.needs_operator_review,
                "laya_confidence": s1_decision.laya_confidence,
                "detector_confidence": confidence,
                "latency_ms": s1_decision.latency_ms,
                "status": s1_decision.status,
                "guardrail_applied": s1_decision.guardrail_applied
            }
        }

    def evaluate_reflex(self, detection: dict, elevation_m: Optional[float] = None) -> dict:
        """Backward-compatible wrapper for existing callers."""
        elev = float(elevation_m) if elevation_m is not None else None
        phys_proxy = {"elevation_m": elev} if elev is not None else None

        res = self.process_reflex(
            detection=detection,
            physics=phys_proxy,
            geo=None,
            metadata=None
        )

        return {
            "decision_primitive": res["decision_primitive"],
            "recommended_maneuver": res["recommended_maneuver"],
            "hazard_score": res["hazard_score"],
            "status": res["alert_severity"],
            "latency_ms": res["latency_ms"],
            "edge_engine": res["edge_engine"],
            "reasons": res["reasons"],
            "system1": res.get("system1", {})
        }
