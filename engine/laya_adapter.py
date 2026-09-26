import os
import sys
import time
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

logger = logging.getLogger("system1_laya")

# ---------------------------------------------------------------------------
# Structured System 1 Input State (Section 6: NO raw images sent to Laya)
# ---------------------------------------------------------------------------

@dataclass
class System1InputState:
    """
    Structured textual & telemetry state for System 1 decision engine.
    Constructed strictly from verified sensor perceptions: YOLO perception,
    acoustic shadow physics, georeferencing, and evidence quality.
    """
    contact_class: str
    detector_confidence: float
    taxonomy_status: str = "STANDARD_CLASS"
    shadow_detected: bool = False
    shadow_length_m: Optional[float] = None
    elevation_m: Optional[float] = None
    physics_provenance: str = "UNAVAILABLE"
    geo_status: str = "UNAVAILABLE"
    evidence_quality: str = "UNKNOWN"
    sonar_altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact": {
                "class": self.contact_class,
                "confidence": round(self.detector_confidence, 4),
                "taxonomy_status": self.taxonomy_status,
            },
            "physics": {
                "shadow_detected": self.shadow_detected,
                "shadow_length_m": self.shadow_length_m,
                "elevation_m": self.elevation_m,
                "provenance": self.physics_provenance,
            },
            "geo": {
                "status": self.geo_status,
            },
            "evidence": {
                "quality": self.evidence_quality,
            },
            "telemetry": {
                "sonar_altitude_m": self.sonar_altitude_m,
                "heading_deg": self.heading_deg,
            }
        }

    def to_laya_prompt(self) -> str:
        """Serializes verified state to a concise, unambiguous natural-language state for Laya."""
        elev_str = f"{self.elevation_m:.2f}m" if self.elevation_m is not None else "unavailable"
        shadow_str = f"detected (length: {self.shadow_length_m:.2f}m)" if self.shadow_detected and self.shadow_length_m else ("detected" if self.shadow_detected else "none")
        alt_str = f"{self.sonar_altitude_m:.1f}m" if self.sonar_altitude_m is not None else "unknown"
        
        return (
            f"Underwater sonar contact evaluation.\n"
            f"Classification: {self.contact_class} (confidence: {self.detector_confidence:.2f}, taxonomy: {self.taxonomy_status}).\n"
            f"Acoustic Shadow: {shadow_str} with obstacle elevation above seabed: {elev_str} (provenance: {self.physics_provenance}).\n"
            f"Geospatial status: {self.geo_status}. Evidence quality: {self.evidence_quality}.\n"
            f"AUV altitude: {alt_str}."
        )


# ---------------------------------------------------------------------------
# System 1 Output Contract (Section 8)
# ---------------------------------------------------------------------------

@dataclass
class System1Decision:
    """
    Rigorous typed output contract for System 1.
    Preserves all original sensor facts, model attribution, and calibrated confidence.
    """
    decision_primitive: str          # EMERGENCY_PROP_HAZARD | LOITER_AND_RESCAN | PASSIVE_LOG
    hazard_score: float              # 1.0 to 10.0
    needs_operator_review: bool      # Binary/Noul review flag
    laya_confidence: Optional[float] # Calibrated Laya confidence [0.0, 1.0] or None
    detector_confidence: float       # Original YOLO detector confidence
    evidence_quality: str            # STRONG | MODERATE | WEAK | UNKNOWN
    physics_provenance: str          # REAL | DERIVED | SIMULATED | UNAVAILABLE
    geo_status: str                  # REAL | DERIVED | UNAVAILABLE
    model: str                       # Model checkpoint identifier or engine name
    latency_ms: float                # Genuine measured inference latency (ms)
    status: str                      # ACTIVE | FALLBACK | UNAVAILABLE | ERROR
    guardrail_applied: bool = False
    reasons: List[str] = field(default_factory=list)
    raw_output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_primitive": self.decision_primitive,
            "hazard_score": self.hazard_score,
            "needs_operator_review": self.needs_operator_review,
            "laya_confidence": self.laya_confidence,
            "detector_confidence": self.detector_confidence,
            "evidence_quality": self.evidence_quality,
            "physics_provenance": self.physics_provenance,
            "geo_status": self.geo_status,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "guardrail_applied": self.guardrail_applied,
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# Laya Question Schema (Section 7)
# ---------------------------------------------------------------------------

LAYA_DECISION_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "action": {
        "type": "choice",
        "instructions": "Determine the required tactical navigation response for this underwater sonar contact.",
        "criteria": {
            "EMERGENCY_PROP_HAZARD": "Immediate propeller fouling, net entanglement, or critical submerged collision hazard requiring evasion or standoff ascent.",
            "LOITER_AND_RESCAN": "Ambiguous, unconfirmed, or high-risk anomaly requiring secondary observation pass or circular rescan.",
            "PASSIVE_LOG": "Nominal, low-hazard, stationary seafloor object or benign infrastructure safe to log without course deviation."
        }
    },
    "hazard_score": {
        "type": "score",
        "instructions": "Rate the overall hazard severity on a 1 (minimal) to 10 (catastrophic collision/entanglement) scale.",
        "criteria": [
            "1: Safe flat seabed or benign acoustic anomaly",
            "2: Minor debris with zero vehicle foul risk",
            "3: Stationary crab pot or small benthic trap",
            "4: Low-profile benthic feature requiring distant logging",
            "5: Buried pipeline or inert seafloor structure",
            "6: Submerged shipwreck or large rigid obstacle",
            "7: Exposed unburied pipeline or high acoustic contrast contact",
            "8: Mine-like cylinder contact requiring standoff safety",
            "9: Ghost net or derelict synthetic fishing gear with high entanglement potential",
            "10: Imminent catastrophic collision or propeller entangling net obstacle"
        ]
    },
    "needs_operator_review": {
        "type": "noul",
        "instructions": "Does this sonar contact require mandatory human topside operator review?",
        "criteria": {
            "false": "Standard nominal detection handled fully autonomously by edge reflex.",
            "true": "Uncertain, safety-critical, or high-consequence contact requiring operator confirmation."
        }
    }
}


# ---------------------------------------------------------------------------
# Safety Guardrails (Section 9)
# ---------------------------------------------------------------------------

class System1SafetyGuardrails:
    """
    Deterministic safety validation layer operating strictly AFTER Laya returns.
    Guarantees:
    1. Laya cannot suppress critical hazards (auto-escalation to EMERGENCY_PROP_HAZARD).
    2. Laya cannot invent sensor readings, coordinates, or elevation.
    3. Direct actuator control is strictly forbidden; outputs remain advisory decision primitives.
    4. Hazard scores are strictly bounded within [1.0, 10.0].
    """

    EMERGENCY_CLASSES = {"ghost_net", "mine_cylinder"}
    EMERGENCY_CONF_THRESHOLD = 0.70
    EMERGENCY_ELEVATION_THRESHOLD_M = 0.80

    @classmethod
    def apply(cls, decision: System1Decision, state: System1InputState) -> System1Decision:
        guardrail_triggered = False
        reasons = list(decision.reasons)

        # 1. Clamp hazard score strictly between 1.0 and 10.0
        clamped_score = max(1.0, min(10.0, round(float(decision.hazard_score), 1)))
        if clamped_score != decision.hazard_score:
            decision.hazard_score = clamped_score
            guardrail_triggered = True

        # 2. Safety Escalation Rule:
        is_critical_class = state.contact_class in cls.EMERGENCY_CLASSES
        is_high_conf = state.detector_confidence >= cls.EMERGENCY_CONF_THRESHOLD
        is_elevated_net = (
            state.contact_class == "ghost_net" and
            state.elevation_m is not None and
            state.elevation_m >= cls.EMERGENCY_ELEVATION_THRESHOLD_M
        )

        if decision.decision_primitive == "PASSIVE_LOG":
            if is_elevated_net:
                decision.decision_primitive = "EMERGENCY_PROP_HAZARD"
                decision.hazard_score = max(decision.hazard_score, 8.5)
                decision.needs_operator_review = True
                guardrail_triggered = True
                reasons.append("SAFETY GUARDRAIL OVERRIDE: PASSIVE_LOG escalated to EMERGENCY_PROP_HAZARD due to elevated ghost net structure.")
            elif is_critical_class and is_high_conf and state.evidence_quality in ("STRONG", "MODERATE"):
                decision.decision_primitive = "EMERGENCY_PROP_HAZARD"
                decision.hazard_score = max(decision.hazard_score, 7.5)
                decision.needs_operator_review = True
                guardrail_triggered = True
                reasons.append(f"SAFETY GUARDRAIL OVERRIDE: High-confidence {state.contact_class} cannot be passively logged; escalated to EMERGENCY_PROP_HAZARD.")
            elif is_critical_class:
                decision.decision_primitive = "LOITER_AND_RESCAN"
                decision.needs_operator_review = True
                guardrail_triggered = True
                reasons.append(f"SAFETY GUARDRAIL OVERRIDE: {state.contact_class} escalated to LOITER_AND_RESCAN for secondary sensor verification.")

        # 3. Mandatory operator review for critical decisions
        if decision.decision_primitive == "EMERGENCY_PROP_HAZARD" and not decision.needs_operator_review:
            decision.needs_operator_review = True
            guardrail_triggered = True
            reasons.append("SAFETY GUARDRAIL: Mandated topside operator review for emergency contact.")

        decision.guardrail_applied = guardrail_triggered
        decision.reasons = reasons
        return decision


# ---------------------------------------------------------------------------
# Base Decision Engine (Section 10)
# ---------------------------------------------------------------------------

class System1DecisionEngine(ABC):
    """Abstract base class for all System 1 edge reflex engines."""

    @abstractmethod
    def decide(self, state: System1InputState) -> System1Decision:
        """Evaluates structured sensor state and returns typed System1Decision."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Returns engine operational status and metadata."""
        pass


# ---------------------------------------------------------------------------
# Deterministic Fallback Engine (Section 10)
# ---------------------------------------------------------------------------

TAXONOMY_CATEGORIES = {
    "ghost_net": "entanglement / marine-debris hazard",
    "mine_cylinder": "mine-cylinder-class sonar contact",
    "submarine_pipeline": "critical infrastructure / benthic hazard",
    "shipwreck": "submerged structure / navigation hazard",
    "crab_pot": "fishing gear / marine-debris contact"
}

class DeterministicDecisionEngine(System1DecisionEngine):
    """
    Fully auditable, offline deterministic System 1 decision engine.
    Implements calibrated marine hazard taxonomy and physical acoustic shadow rules.
    Acts as the primary baseline and seamless fallback.
    """

    BASE_WEIGHTS = {
        "ghost_net": 9.0,
        "mine_cylinder": 8.5,
        "submarine_pipeline": 7.0,
        "shipwreck": 6.5,
        "crab_pot": 3.0,
    }

    def decide(self, state: System1InputState) -> System1Decision:
        t0 = time.perf_counter()

        base_score = self.BASE_WEIGHTS.get(state.contact_class, 5.0)
        conf_factor = 0.5 + 0.5 * max(0.0, min(1.0, state.detector_confidence))

        elevation_bonus = 0.0
        if state.elevation_m is not None and state.elevation_m > 0:
            elevation_bonus = min(1.0, float(state.elevation_m) * 0.4)

        evidence_mult = {
            "STRONG": 1.1,
            "MODERATE": 1.0,
            "WEAK": 0.85
        }.get(state.evidence_quality, 0.90)

        raw_score = (base_score * conf_factor + elevation_bonus) * evidence_mult
        hazard_score = max(1.0, min(10.0, round(raw_score, 1)))

        category = TAXONOMY_CATEGORIES.get(state.contact_class, "unclassified contact")
        reasons = [
            f"{state.contact_class} ({category})",
            f"detection confidence: {state.detector_confidence:.2f}",
            f"evidence quality: {state.evidence_quality}",
        ]
        if state.elevation_m is not None:
            reasons.append(f"obstacle elevation: {state.elevation_m:.2f}m")
        else:
            reasons.append("elevation: unavailable")

        # Decision primitive
        is_elevated_net = (
            state.contact_class == "ghost_net" and
            state.elevation_m is not None and
            state.elevation_m >= 0.80
        )
        is_high_threat = (
            hazard_score >= 7.5 and
            state.evidence_quality in ("STRONG", "MODERATE") and
            state.detector_confidence >= 0.70
        )

        if is_elevated_net or is_high_threat:
            primitive = "EMERGENCY_PROP_HAZARD"
            needs_review = True
            if is_elevated_net:
                reasons.append("propeller fouling hazard: elevated net structure detected above seabed")
            else:
                reasons.append(f"critical hazard threshold exceeded (score: {hazard_score:.1f} >= 7.5)")
        elif hazard_score >= 4.5 or (state.detector_confidence < 0.70 and state.contact_class in ("ghost_net", "mine_cylinder")):
            primitive = "LOITER_AND_RESCAN"
            needs_review = (state.contact_class in ("ghost_net", "mine_cylinder"))
            reasons.append("secondary observation recommended: contact requires verification look-angle")
        else:
            primitive = "PASSIVE_LOG"
            needs_review = False
            reasons.append("nominal contact recorded in mission log")

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        decision = System1Decision(
            decision_primitive=primitive,
            hazard_score=hazard_score,
            needs_operator_review=needs_review,
            laya_confidence=None,
            detector_confidence=state.detector_confidence,
            evidence_quality=state.evidence_quality,
            physics_provenance=state.physics_provenance,
            geo_status=state.geo_status,
            model="deterministic-rule-engine-v1",
            latency_ms=latency_ms,
            status="FALLBACK",
            reasons=reasons
        )

        return System1SafetyGuardrails.apply(decision, state)

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine": "deterministic_fallback",
            "status": "ACTIVE",
            "version": "1.0.0",
            "checkpoint": "local_rule_engine",
            "device": "cpu",
            "fallback_available": True
        }


# ---------------------------------------------------------------------------
# Real Laya Decision Engine (Section 4 & 5)
# ---------------------------------------------------------------------------

class LayaDecisionEngine(System1DecisionEngine):
    """
    Real Laya Decision Engine utilizing the official non-autoregressive Laya router/agent.
    Loads from local model weights checkpoint or local huggingface cache.
    Executes parallel typed decisions (choice, score, noul) in a single forward pass.
    """

    def __init__(self, model_dir: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model_dir = model_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "laya", "checkpoint")
        )
        self.agent = None
        self.router = None
        self.is_active = False
        self.status = "UNINITIALIZED"
        self.version = "unknown"
        self.load_error: Optional[str] = None
        self.checkpoint_path: Optional[str] = None
        self._cache: Dict[str, Any] = {}

        self._initialize()

    def _initialize(self):

        try:
            import laya
            self.version = getattr(laya, "__version__", "unknown")
        except ImportError as e:
            self.status = "UNAVAILABLE"
            self.load_error = f"Laya package not installed: {e}"
            logger.warning(self.load_error)
            return

        safetensors_path = os.path.join(self.model_dir, "model.safetensors")
        config_path = os.path.join(self.model_dir, "rl_agent_config.json")

        if os.path.exists(safetensors_path) and os.path.exists(config_path) and os.path.getsize(safetensors_path) > 100_000_000:
            try:
                import laya
                logger.info(f"Loading local Laya checkpoint from {self.model_dir}...")
                self.agent = laya.Agent(model_id_or_path=self.model_dir, device=self.device)
                self.is_active = True
                self.status = "ACTIVE"
                self.checkpoint_path = self.model_dir
                logger.info(f"Laya agent successfully loaded on {self.device} from {self.model_dir}")
                return
            except Exception as e:
                self.status = "ERROR"
                self.load_error = f"Failed loading local Laya weights from {self.model_dir}: {e}"
                logger.error(self.load_error)
                return

        # Alternative snapshot in models/laya/hf_cache
        alt_snapshot = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "models", "laya", "hf_cache",
            "hub", "models--convaiinnovations--laya", "snapshots"
        ))
        if os.path.exists(alt_snapshot):
            for entry in os.scandir(alt_snapshot):
                if entry.is_dir():
                    cand_weights = os.path.join(entry.path, "model.safetensors")
                    cand_cfg = os.path.join(entry.path, "rl_agent_config.json")
                    if os.path.exists(cand_weights) and os.path.exists(cand_cfg) and os.path.getsize(cand_weights) > 100_000_000:
                        try:
                            import laya
                            logger.info(f"Loading Laya from snapshot: {entry.path}...")
                            self.agent = laya.Agent(model_id_or_path=entry.path, device=self.device)
                            self.is_active = True
                            self.status = "ACTIVE"
                            self.checkpoint_path = entry.path
                            logger.info(f"Laya agent loaded from snapshot {entry.path}")
                            return
                        except Exception as e:
                            self.load_error = str(e)

        self.status = "UNAVAILABLE"
        self.load_error = (
            f"Laya model weights not available at {self.model_dir} (expected 842MB model.safetensors). "
            f"Run scripts/download_laya_checkpoint.py or wait for background download."
        )
        logger.info(self.load_error)

    def decide(self, state: System1InputState) -> System1Decision:
        if not self.is_active or self.agent is None:
            raise RuntimeError(f"LayaDecisionEngine is not active ({self.status}): {self.load_error}")

        laya_prompt = state.to_laya_prompt()
        elev_bucket = round(state.elevation_m, 1) if state.elevation_m is not None else -1.0
        conf_bucket = round(state.detector_confidence, 1)
        cache_key = f"{state.contact_class}_{conf_bucket}_{elev_bucket}_{state.evidence_quality}"

        try:
            if cache_key in self._cache:
                res, latency_ms = self._cache[cache_key]
            else:
                t0 = time.perf_counter()
                res = self.agent.system_one(laya_prompt, LAYA_DECISION_QUESTIONS)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
                if len(self._cache) < 500:
                    self._cache[cache_key] = (res, latency_ms)


            answers = res.get("answers", {})

            # 1. Action (choice)
            action_ans = answers.get("action", {})
            action_choice = action_ans.get("choice", "PASSIVE_LOG")
            if action_choice not in ("EMERGENCY_PROP_HAZARD", "LOITER_AND_RESCAN", "PASSIVE_LOG"):
                action_choice = "PASSIVE_LOG"
            action_conf = action_ans.get("confidence", 0.0)

            # 2. Hazard score (score)
            score_ans = answers.get("hazard_score", {})
            raw_score = float(score_ans.get("score", 4.0)) + 1.0
            hazard_score = max(1.0, min(10.0, round(raw_score, 1)))

            # 3. Needs operator review (noul)
            noul_ans = answers.get("needs_operator_review", {})
            p_true = float(noul_ans.get("noul", 0.0))
            needs_review = bool(p_true >= 0.50)

            category = TAXONOMY_CATEGORIES.get(state.contact_class, "unclassified contact")
            reasons = [
                f"{state.contact_class} ({category})",
                f"Laya decision: {action_choice} (conf: {action_conf:.2f})",
                f"Laya hazard score: {hazard_score:.1f} (P(operator review): {p_true:.2f})",
                f"detection confidence: {state.detector_confidence:.2f}",
                f"evidence quality: {state.evidence_quality}"
            ]
            if state.elevation_m is not None:
                reasons.append(f"obstacle elevation: {state.elevation_m:.2f}m")

            decision = System1Decision(
                decision_primitive=action_choice,
                hazard_score=hazard_score,
                needs_operator_review=needs_review,
                laya_confidence=action_conf,
                detector_confidence=state.detector_confidence,
                evidence_quality=state.evidence_quality,
                physics_provenance=state.physics_provenance,
                geo_status=state.geo_status,
                model="convaiinnovations/laya",
                latency_ms=latency_ms,
                status="ACTIVE",
                reasons=reasons,
                raw_output=res
            )

            return System1SafetyGuardrails.apply(decision, state)

        except Exception as e:
            logger.error(f"Laya inference exception: {e}")
            raise e

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine": "laya",
            "status": self.status,
            "version": self.version,
            "checkpoint": self.checkpoint_path or self.model_dir,
            "device": self.device,
            "error": self.load_error,
            "fallback_available": True
        }


# ---------------------------------------------------------------------------
# Backward Compatibility Adapter for Phase 4 tests
# ---------------------------------------------------------------------------

class LayaAdapter:
    """Backward-compatible adapter for tests checking adapter status."""
    def __init__(self, local_model_dir: Optional[str] = None):
        self.is_installed = False
        self.is_active = False
        self.agent = None
        self.local_model_dir = local_model_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "laya")
        )

        self.status = "UNINITIALIZED"
        self.laya_version = None

        try:
            import laya
            self.is_installed = True
            self.laya_version = getattr(laya, "__version__", "unknown")
        except ImportError:
            self.is_installed = False
            self.status = "Laya runtime unavailable (package not installed)"
            return

        safetensors = os.path.join(self.local_model_dir, "model.safetensors")
        if os.path.exists(safetensors) and os.path.getsize(safetensors) > 100_000_000:
            try:
                import laya
                self.agent = laya.Agent(model_id_or_path=self.local_model_dir, device="cpu")
                self.is_active = True
                self.status = f"Laya active (loaded local checkpoint from {self.local_model_dir})"
            except Exception as e:
                self.is_active = False
                self.status = f"Laya adapter present but failed to load local weights: {e}"
        else:
            self.is_active = False
            self.status = "Laya adapter present but not active (no local weights configured, using deterministic local reflex engine)"

    def predict(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_active or self.agent is None:
            return None
        try:
            return self.agent.predict(context)
        except Exception:
            return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "installed": self.is_installed,
            "active": self.is_active,
            "version": self.laya_version,
            "status": self.status,
            "local_model_dir": self.local_model_dir
        }


# ---------------------------------------------------------------------------
# Unified System 1 Manager with Dual-Engine Fallback (Section 10 & 11)
# ---------------------------------------------------------------------------

class System1Manager:
    """
    Orchestrates System 1 decisions:
    1. Attempts real Laya engine if available and active.
    2. Falls back seamlessly to DeterministicDecisionEngine if Laya is unavailable or fails.
    3. Exposes clean operational telemetry and status API.
    """

    def __init__(self, local_model_dir: Optional[str] = None, device: str = "cpu", enable_laya: bool = True):
        self.fallback_engine = DeterministicDecisionEngine()
        self.laya_engine: Optional[LayaDecisionEngine] = None
        self.model_dir = local_model_dir
        self.enable_laya = enable_laya

        if enable_laya:
            try:
                self.laya_engine = LayaDecisionEngine(model_dir=local_model_dir, device=device)
            except Exception as e:
                logger.warning(f"Could not initialize LayaDecisionEngine: {e}")
                self.laya_engine = None


    def refresh_laya(self) -> bool:
        """Attempts to re-initialize Laya if weights just finished downloading."""
        try:
            self.laya_engine = LayaDecisionEngine(model_dir=self.model_dir)
            return self.laya_engine.is_active
        except Exception as e:
            logger.warning(f"Refresh Laya failed: {e}")
            return False

    def decide(self, state: System1InputState) -> System1Decision:
        if self.laya_engine is not None and not self.laya_engine.is_active:
            self.refresh_laya()

        if self.laya_engine is not None and self.laya_engine.is_active:
            try:
                return self.laya_engine.decide(state)
            except Exception as e:
                logger.error(f"Laya execution failed at runtime, falling back to deterministic engine: {e}")
                fallback_decision = self.fallback_engine.decide(state)
                fallback_decision.reasons.append(f"Laya runtime error: {e} (seamlessly routed to deterministic fallback)")
                return fallback_decision

        return self.fallback_engine.decide(state)

    def get_status(self) -> Dict[str, Any]:
        """Provides status for GET /api/system1/status (Section 11)."""
        if self.laya_engine is not None and self.laya_engine.is_active:
            status = self.laya_engine.get_status()
            status["active_engine"] = "laya"
            return status

        laya_info = self.laya_engine.get_status() if self.laya_engine else {
            "status": "UNAVAILABLE",
            "error": "Laya engine not initialized",
            "version": "unknown"
        }

        return {
            "engine": "laya",
            "status": laya_info.get("status", "UNAVAILABLE"),
            "version": laya_info.get("version", "unknown"),
            "checkpoint": laya_info.get("checkpoint", "none"),
            "device": laya_info.get("device", "cpu"),
            "error": laya_info.get("error"),
            "fallback_available": True,
            "active_engine": "deterministic_fallback"
        }
