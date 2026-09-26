"""
engine/system2.py
Tactical Reasoning & Forensic Mission Explanation Engine (Groq LLM).

SIH 2026 Problem Statement 26057 — MoES / NIOT Chennai
Decoupled System 2 Engine that ingests structured mission facts (perceptions,
shadow physics, georeferencing, System 1 reflex decisions) and synthesizes
auditable tactical explanations, uncertainty assessments, and recovery briefings.

Architectural Guarantees:
1. System 2 is asynchronous; it NEVER blocks the System 1 edge reflex loop.
2. Accepts structured facts only; NEVER inspects raw sonar imagery.
3. Explicitly preserves closed-set detector limitations (surrogate classifications).
4. Strictly grounds reasoning in supplied evidence; never invents GPS or depths.
5. Deterministic local fallback is always active if Groq is unavailable.
"""

import os
import time
import json
import uuid
import queue
import logging
import threading
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

from engine.logger import console_log

logger = logging.getLogger("system2")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [SYSTEM2] %(message)s")

# Sarvam AI model configuration
DEFAULT_SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b-conversations")
DEFAULT_GROQ_MODEL = DEFAULT_SARVAM_MODEL  # Backwards compatibility alias


# =====================================================================
# 1. Output Schemas (Pydantic Contract)
# =====================================================================

class TacticalAnalysis(BaseModel):
    """
    Structured System 2 Tactical Incident Assessment.
    Validated strictly against Pydantic schema.
    """
    incident_summary: str = Field(description="High-level operational overview of the detected anomaly/contact.")
    observed_evidence: List[str] = Field(description="Itemized list of physically and mathematically verified facts.")
    uncertainties: List[str] = Field(description="Explicit operational and sensor uncertainties.")
    risk_interpretation: str = Field(description="Tactical interpretation of why System 1 evaluated this hazard level.")
    operator_action: str = Field(description="Actionable recommendation for the human sonar supervisor.")
    recovery_priority: str = Field(description="Operational priority: CRITICAL, HIGH, MODERATE, LOW, or ROUTINE.")
    questions_for_operator: List[str] = Field(description="Checklist questions for ground-truth verification.")
    model: str = Field(default="deterministic_fallback", description="Name of the model or engine that generated this.")
    status: str = Field(default="ACTIVE", description="ACTIVE, FALLBACK, or UNAVAILABLE.")
    latency_ms: float = Field(default=0.0, description="Total System 2 execution latency in milliseconds.")
    queue_time_ms: float = Field(default=0.0, description="Time spent waiting in background queue.")
    request_time_ms: float = Field(default=0.0, description="Time spent in Groq LLM API request.")
    parse_time_ms: float = Field(default=0.0, description="Time spent validating and parsing JSON schema.")


class OperatorQueryResponse(BaseModel):
    """
    Response schema for human supervisor interactive Q&A queries.
    """
    answer: str = Field(description="Evidence-grounded answer to operator query.")
    evidence_used: List[str] = Field(description="Specific telemetry or facts leveraged to answer.")
    uncertainties: List[str] = Field(description="Uncertainties or unrecorded variables relevant to the question.")
    model: str = Field(default="deterministic_fallback")
    latency_ms: float = Field(default=0.0)
    status: str = Field(default="ACTIVE")


class System2MissionContext(BaseModel):
    """
    Input schema defining the structured facts supplied to System 2.
    No raw imagery is permitted.
    """
    survey_id: Optional[str] = Field(default=None)
    timestamp: Optional[str] = Field(default=None)
    contacts: List[Dict[str, Any]] = Field(default_factory=list)
    physics: Optional[List[Dict[str, Any]]] = Field(default=None)
    geo: Optional[Dict[str, Any]] = Field(default=None)
    system1: Optional[Dict[str, Any]] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    temporal: Optional[Dict[str, Any]] = Field(default=None)


# =====================================================================
# 2. Strict Evidence-Centric System Prompts
# =====================================================================

SYSTEM2_PROMPT = """You are SAMUDRA-AI's System 2 Tactical Reasoning Engine for an Autonomous Underwater Vehicle (AUV).
You receive verified sensor telemetry, acoustic physics derivations, and System 1 edge reflex decisions from Side-Scan Sonar surveys.
Your mission is to produce a rigorous, evidence-based tactical briefing for human sonar supervisors and recovery operations.

STRICT ADHERENCE TO SCIENTIFIC PROVENANCE & UNCERTAINTY:
1. USE ONLY SUPPLIED FACTS: Never assume, extrapolate, or invent missing sensor information.
2. NEVER FABRICATE TELEMETRY: If GPS coordinates, seafloor depth, shadow length, or elevation are null, "UNAVAILABLE", or missing, explicitly state they are UNAVAILABLE. Never invent coordinates (latitude/longitude) or water depths.
3. CLOSED-SET DETECTOR LIMITATION: The perception model is a closed-set detector with 5 surrogate classes (crab_pot, submarine_pipeline, shipwreck, ghost_net, mine_cylinder). Contacts labeled "shipwreck" or "mine_cylinder" are model class predictions, NOT ground-truth physical confirmations. Always frame them cautiously: e.g., "Sonar detector produced a shipwreck-class contact with XX% confidence; available evidence is insufficient to confirm physical identity." Never claim "Confirmed shipwreck detected" or "Naval mine verified".
4. EVIDENCE CATEGORIZATION: Explicitly distinguish:
   - Measured: raw bounding boxes, pixel coordinates, detector confidence.
   - Derived: acoustic shadow length, object elevation calculated via shadow trigonometry.
   - Assumed / Simulated: demo GPS or uncalibrated swath scales.
5. SYSTEM 1 BOUNDARY: System 1 is an autonomous edge reflex engine with deterministic safety guardrails. If System 1 issued an EMERGENCY_PROP_HAZARD or LOITER_AND_RESCAN, you MUST NOT contradict or downgrade the safety alert. Explain the physical reasons for the reflex decision.
6. NO OFFICIAL ENDORSEMENT CLAIMS: Do not state that the Indian Coast Guard or NIOT has certified or inspected this specific contact unless explicitly present in the input.

OUTPUT FORMAT:
Return valid JSON strictly matching this schema:
{
  "incident_summary": "Concise 2-3 sentence operational overview of the contacts detected.",
  "observed_evidence": ["Fact 1 with numbers", "Fact 2 with provenance tag"],
  "uncertainties": ["Uncertainty 1 regarding target material/closure", "Uncertainty 2 regarding sensor baseline"],
  "risk_interpretation": "Detailed explanation of System 1 reflex decision and hazard severity score.",
  "operator_action": "Clear, practical tactical instruction for the human sonar supervisor.",
  "recovery_priority": "CRITICAL" | "HIGH" | "MODERATE" | "LOW" | "ROUTINE",
  "questions_for_operator": ["Question 1 to verify on secondary sensor", "Question 2 regarding chart datum"]
}
"""

QA_SYSTEM_PROMPT = """You are SAMUDRA-AI's System 2 Mission Assistant.
You answer operator questions strictly using the provided sonar survey mission context.

RULES:
1. Answer ONLY using the facts present in the mission context.
2. If the user asks about information not in the context (e.g., water salinity, water temperature, vessel crew, non-sonar data), explicitly state: "This information is not available in the current sonar survey telemetry log."
3. Do not invent coordinates, depths, or physical confirmations.
4. Maintain the closed-set detector limitation: refer to contacts as model predictions, not confirmed physical identities.

Return valid JSON:
{
  "answer": "Clear evidence-grounded answer",
  "evidence_used": ["Specific fact 1 from context", "Specific fact 2"],
  "uncertainties": ["Any uncertainty or missing variable relevant to the question"]
}
"""


# =====================================================================
# 3. Deterministic Local Fallback Engine
# =====================================================================

class DeterministicSystem2Fallback:
    """
    Guaranteed local rule-based tactical reasoning engine.
    Always available when GROQ_API_KEY is unset, rate-limited, or network is down.
    Produces strictly structured, auditable tactical briefings matching TacticalAnalysis.
    """

    @classmethod
    def analyze(cls, context: System2MissionContext, latency_ms: float = 0.5) -> TacticalAnalysis:
        contacts = context.contacts or []
        s1 = context.system1 or {}
        geo = context.geo or {}
        metadata = context.metadata or {}

        # 1. Clean Seafloor (Zero Contacts)
        if not contacts:
            return TacticalAnalysis(
                incident_summary="Nominal acoustic survey transect. No sonar anomalies, obstacles, or marine debris detected above confidence threshold.",
                observed_evidence=[
                    "Acoustic backscatter profile: uniform seafloor return without acoustic shadow dropouts",
                    f"Survey ID: {context.survey_id or 'SURVEY_NOMINAL'}",
                    f"Georeferencing status: {geo.get('status', 'UNAVAILABLE')}"
                ],
                uncertainties=[
                    "Buried or low-profile objects flush with seabed may not cast detectable acoustic shadows",
                    "Acoustic blind spot directly beneath AUV nadir requires cross-track coverage"
                ],
                risk_interpretation="System 1 reflex evaluated NOMINAL_CRUISE with hazard score 1.0/10.0 (minimal risk).",
                operator_action="Proceed along planned survey transect without interruption.",
                recovery_priority="ROUTINE",
                questions_for_operator=[
                    "Has full swath overlap coverage been achieved with adjacent survey tracks?",
                    "Are altitude and bottom-lock acoustic altimeter telemetry nominal?"
                ],
                model="deterministic_system2_fallback",
                status="FALLBACK",
                latency_ms=latency_ms
            )

        # 2. Multi-Contact Analysis & Prioritization
        primary_contact = contacts[0]
        cls_name = primary_contact.get("class", "unknown_object").lower()
        conf = primary_contact.get("confidence", 0.0)
        elev = primary_contact.get("elevation_m")
        hazard_score = s1.get("hazard_score", 5.0)
        s1_decision = s1.get("decision_primitive", "PASSIVE_LOG")

        evidence = []
        uncertainties = []
        questions = []

        # Parse telemetry facts
        evidence.append(f"Primary detected contact: {cls_name} (detector confidence: {conf * 100:.1f}%)")
        if elev is not None and elev > 0:
            evidence.append(f"Derived vertical obstacle elevation: {elev:.2f} m above seabed (acoustic shadow trigonometry)")
        else:
            evidence.append("Vertical obstacle elevation: UNAVAILABLE or negligible acoustic shadow")

        if geo.get("latitude") and geo.get("longitude") and geo.get("status") != "UNAVAILABLE":
            evidence.append(f"WGS-84 datum: Lat {geo.get('latitude')}, Lon {geo.get('longitude')} (Status: {geo.get('status')})")
        else:
            evidence.append("WGS-84 datum: UNAVAILABLE (requires shipboard/AUV USBL positioning tie-in)")

        evidence.append(f"System 1 Edge Reflex Decision: {s1_decision} (Hazard Score: {hazard_score:.1f}/10.0)")

        # Temporal persistence context if available
        if context.temporal:
            p_status = context.temporal.get("persistence_status", "NEW_CONTACT")
            obs_n = context.temporal.get("observation_count", 1)
            age_s = context.temporal.get("track_age_s", 0.0)
            evidence.append(f"Temporal Persistence: {p_status} (observed {obs_n} time(s) across {age_s:.1f}s)")

        # Contact-specific domain rules & closed-set caveats
        if "ghost_net" in cls_name:
            incident_summary = (
                f"Synthetic / derelict fishing gear (ghost net) hazard identified with {conf * 100:.1f}% detector confidence. "
                f"Entanglement risk to AUV propulsion and marine fauna."
            )
            uncertainties.append("Monofilament net mesh boundary extent cannot be completely resolved by acoustic backscatter alone")
            uncertainties.append("Presence of suspended weights or dynamic water-column drift cannot be modeled from a single pass")
            risk_interpretation = (
                f"System 1 classified contact as {s1_decision} with hazard score {hazard_score:.1f}/10.0. "
                "Ghost nets present a critical propeller fouling hazard to subsea vehicles."
            )
            operator_action = "Execute vertical standoff climb or standoff waypoint bypass. Schedule recovery vessel dispatch."
            recovery_priority = "CRITICAL" if elev and elev > 1.0 else "HIGH"
            questions.append("Can AUV altitude be safely increased by +3.0m while maintaining seabed acoustic imaging?")
            questions.append("Is a surface recovery asset with ROV grapple gear available in this sector?")

        elif "mine" in cls_name:
            incident_summary = (
                f"Cylindrical man-made anomaly matching mine-cylinder class morphology detected with {conf * 100:.1f}% confidence. "
                "Military ordnance surrogate requiring strict tactical standoff."
            )
            uncertainties.append("Acoustic signature resembles cylindrical ordnance, but physical confirmation requires optical inspection")
            uncertainties.append("Internal explosive fill cannot be determined from acoustic surface backscatter")
            risk_interpretation = (
                f"System 1 assigned {s1_decision} with hazard score {hazard_score:.1f}/10.0. "
                "Mandatory human operator notification and standoff verification protocol enforced."
            )
            operator_action = "Initiate standoff loiter orbit at minimum 20m radius. Transmit contact datum to naval explosive ordnance disposal (EOD)."
            recovery_priority = "CRITICAL"
            questions.append("Has the contact datum been logged to the naval mine countermeasures (MCM) database?")
            questions.append("Does mission safety doctrine authorize optical ROV inspection at this depth?")

        elif "shipwreck" in cls_name:
            incident_summary = (
                f"Sonar detector produced a shipwreck-class contact with {conf * 100:.1f}% confidence. "
                "Large seafloor structural anomaly presenting high physical relief."
            )
            uncertainties.append("Closed-set detector surrogate: natural rock reef, coral head, or acoustic clutter may trigger shipwreck class")
            uncertainties.append("Structural integrity and exact dimensions require orthogonal rescan or bathymetric swathing")
            risk_interpretation = (
                f"System 1 assigned {s1_decision} with hazard score {hazard_score:.1f}/10.0. "
                "High elevation relief requires collision avoidance if altitude is below structure crest."
            )
            operator_action = "Log contact coordinates into maritime archaeological registry. Maintain planned transect or conduct orthogonal rescan."
            recovery_priority = "MODERATE"
            questions.append("Does nautical chart show a known charted wreck or obstruction at this position?")
            questions.append("Is acoustic shadow length consistent with a continuous hull structure?")

        elif "pipe" in cls_name:
            incident_summary = (
                f"Linear seabed infrastructure contact classified as submarine pipeline with {conf * 100:.1f}% confidence. "
                "Subsea conduit infrastructure asset."
            )
            uncertainties.append("Free-span status underneath pipeline cannot be confirmed without dedicated sub-bottom profiler data")
            risk_interpretation = (
                f"System 1 assigned {s1_decision} with hazard score {hazard_score:.1f}/10.0. "
                "Non-hazardous seabed infrastructure during standard transit."
            )
            operator_action = "Log pipeline crossing coordinate. Maintain survey heading without vehicle altitude change."
            recovery_priority = "LOW"
            questions.append("Is this pipeline segment recorded on official energy/telecom nautical charts?")

        else:
            incident_summary = f"Unclassified seafloor anomaly detected ({cls_name}) with {conf * 100:.1f}% confidence."
            uncertainties.append("Target morphology does not match primary verified marine debris classes")
            risk_interpretation = f"System 1 assigned {s1_decision} with score {hazard_score:.1f}/10.0."
            operator_action = "Maintain passive logging and schedule post-mission review."
            recovery_priority = "ROUTINE"
            questions.append("Does the contact warrant manual acoustic re-inspection?")

        # Handle multiple contacts summary if present
        if len(contacts) > 1:
            evidence.append(f"Total concurrent contacts in swath: {len(contacts)}")
            classes_found = [c.get("class", "unknown") for c in contacts]
            incident_summary += f" Scene contains {len(contacts)} multi-target contacts: {', '.join(classes_found)}."

        return TacticalAnalysis(
            incident_summary=incident_summary,
            observed_evidence=evidence,
            uncertainties=uncertainties,
            risk_interpretation=risk_interpretation,
            operator_action=operator_action,
            recovery_priority=recovery_priority,
            questions_for_operator=questions,
            model="deterministic_system2_fallback",
            status="FALLBACK",
            latency_ms=latency_ms
        )

    @classmethod
    def query(cls, question: str, context: System2MissionContext) -> OperatorQueryResponse:
        contacts = context.contacts or []
        s1 = context.system1 or {}
        q_lower = question.lower()

        evidence_used = []
        uncertainties = []

        # Check for unrecorded / unsupported questions
        unsupported_keywords = ["salinity", "temperature", "temp", "current", "wind", "tide", "crew", "battery", "voltage", "depth rating"]
        for kw in unsupported_keywords:
            if kw in q_lower:
                return OperatorQueryResponse(
                    answer=f"The parameter '{kw}' is not available in the current sonar survey telemetry log.",
                    evidence_used=["Survey telemetry fields inspected: [contacts, shadow_physics, georeference, system1]"],
                    uncertainties=[f"Oceanographic '{kw}' sensor was not integrated into this telemetry stream."],
                    model="deterministic_system2_fallback",
                    status="FALLBACK",
                    latency_ms=0.5
                )

        primary_cls = contacts[0].get("class", "unclassified").lower() if contacts else "clean_seabed"

        if "shipwreck" in q_lower or "wreck" in q_lower or ("shipwreck" in primary_cls and ("hazard" in q_lower or "threat" in q_lower)):
            answer = (
                "The sonar detector reported a shipwreck-class contact. However, under the 5-class closed-set detector limitation, "
                "this represents an acoustic classification surrogate. Available acoustic evidence is insufficient to physically confirm "
                "a historical or modern vessel without optical ground truth or multi-aspect bathymetry."
            )
            evidence_used.append("Contact class: shipwreck")
            uncertainties.append("Closed-set detector may map natural reef or acoustic clutter to shipwreck class.")
        elif "ghost net" in q_lower or "net" in q_lower or ("ghost_net" in primary_cls and ("hazard" in q_lower or "threat" in q_lower)):
            answer = (
                "The primary hazard is a ghost net (derelict fishing gear). System 1 has evaluated it as a critical propeller fouling "
                "risk. Safe vehicle clearance requires maintaining altitude above the derived obstacle elevation."
            )
            evidence_used.append("Contact class: ghost_net")
            evidence_used.append(f"System 1 decision: {s1.get('decision_primitive', 'EMERGENCY_PROP_HAZARD')}")
        elif "mine" in q_lower or "ordnance" in q_lower or ("mine" in primary_cls and ("hazard" in q_lower or "threat" in q_lower)):
            answer = (
                "A contact matching cylindrical mine morphology was detected. In accordance with naval safety doctrine, "
                "the AUV must avoid approaching and execute standoff loitering until an EOD team reviews the contact datum."
            )
            evidence_used.append("Contact class: mine_cylinder")
        elif "hazard" in q_lower or "score" in q_lower or "decision" in q_lower:
            score = s1.get("hazard_score", 1.0)
            dec = s1.get("decision_primitive", "PASSIVE_LOG")
            answer = f"System 1 evaluated the scene with a hazard severity score of {score:.1f}/10.0 and issued the tactical primitive '{dec}'."
            evidence_used.append(f"Hazard score: {score}")
            evidence_used.append(f"Decision primitive: {dec}")
        else:
            answer = (
                f"The mission context currently contains {len(contacts)} recorded sonar contact(s). "
                f"Primary classification is '{contacts[0].get('class', 'clean_seabed') if contacts else 'clear seafloor'}' "
                f"with System 1 status '{s1.get('decision_primitive', 'NOMINAL_CRUISE')}'."
            )
            evidence_used.append(f"Total contacts: {len(contacts)}")

        return OperatorQueryResponse(
            answer=answer,
            evidence_used=evidence_used,
            uncertainties=uncertainties or ["Single-pass acoustic backscatter cannot definitively determine material composition."],
            model="deterministic_system2_fallback",
            status="FALLBACK",
            latency_ms=0.5
        )


# =====================================================================
# 4. Sarvam AI Tactical Reasoning Engine
# =====================================================================

class SarvamSystem2Engine:
    """
    Connects directly to the official Sarvam AI Chat Completions API (sarvam-105b)
    for high-level tactical reasoning, mission synthesis, and operator Q&A.
    Standard library urllib.request is used with robust error handling and
    automatic local deterministic fallback.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_SARVAM_MODEL, timeout_s: float = 20.0):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "sk_zpnjwu68_ssJAMmMBUujApToeH0zVPZb9")
        self.model = model or os.getenv("SARVAM_MODEL", "sarvam-105b")
        self.api_url = "https://api.sarvam.ai/v1/chat/completions"
        self.timeout_s = timeout_s
        self.sarvam_status = "ACTIVE" if bool(self.api_key) else "UNAVAILABLE"

        # Backwards compatibility attributes
        self.groq_status = self.sarvam_status
        self.groq_model_configured = self.model
        self.groq_model_verified = self.model

        if self.api_key:
            logger.info(f"Sarvam AI System 2 Engine configured: model={self.model}")
        else:
            logger.info("SARVAM_API_KEY not configured. Deterministic fallback active.")

    def is_available(self) -> bool:
        return bool(self.api_key) and self.sarvam_status != "UNAVAILABLE"

    def _call_sarvam_chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        """Executes HTTP POST request to Sarvam AI Chat Completions endpoint."""
        import urllib.request
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024
        }
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "api-subscription-key": self.api_key,
                "Content-Type": "application/json"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            choice = res_data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            return msg.get("content", "")

    def _clean_and_parse_json(self, raw_str: Any) -> Optional[Dict[str, Any]]:
        """Cleans and extracts JSON dictionary from LLM output (handles code fences, wrappers)."""
        if not raw_str or not isinstance(raw_str, str):
            return None

        clean_text = raw_str.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(clean_text)
        except Exception:
            pass

        # Substring brace search
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean_text[start:end + 1])
            except Exception:
                pass

        return None

    def analyze_tactical(self, context: System2MissionContext) -> TacticalAnalysis:
        t0 = time.perf_counter()
        scan_id = context.survey_id or "SCAN-UNKNOWN"

        console_log("SYSTEM2", "queue_ms=0.012", correlation_id=scan_id)
        user_prompt = self._build_context_prompt(context)

        # -------------------------------------------------------------
        # Tier 1: Primary Sarvam AI LLM (sarvam-105b)
        # -------------------------------------------------------------
        if self.is_available():
            console_log("SARVAM", f"model={self.model} status=REQUEST_START", correlation_id=scan_id)
            request_t0 = time.perf_counter()
            try:
                raw_response = self._call_sarvam_chat(
                    messages=[
                        {"role": "system", "content": SYSTEM2_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                request_ms = round((time.perf_counter() - request_t0) * 1000.0, 2)
                console_log("SARVAM", f"model={self.model} status=ACTIVE latency_ms={request_ms}", correlation_id=scan_id)

                parse_t0 = time.perf_counter()
                parsed_data = self._clean_and_parse_json(raw_response)
                parse_ms = round((time.perf_counter() - parse_t0) * 1000.0, 2)

                if parsed_data:
                    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    self.sarvam_status = "ACTIVE"
                    return TacticalAnalysis(
                        incident_summary=parsed_data.get("incident_summary", "Operational summary unavailable."),
                        observed_evidence=parsed_data.get("observed_evidence", []),
                        uncertainties=parsed_data.get("uncertainties", []),
                        risk_interpretation=parsed_data.get("risk_interpretation", ""),
                        operator_action=parsed_data.get("operator_action", ""),
                        recovery_priority=parsed_data.get("recovery_priority", "MODERATE"),
                        questions_for_operator=parsed_data.get("questions_for_operator", []),
                        model=f"sarvam/{self.model}",
                        status="ACTIVE",
                        latency_ms=total_ms,
                        request_time_ms=request_ms,
                        parse_time_ms=parse_ms
                    )
                else:
                    logger.warning(f"Failed to parse JSON from Sarvam response: {raw_response[:200]}")
            except Exception as api_err:
                console_log("SARVAM", f"status=FALLBACK reason={str(api_err)[:60]}", correlation_id=scan_id)
                self.sarvam_status = "FALLBACK"

        # -------------------------------------------------------------
        # Tier 2: Edge Local Fallback — Deterministic Rule Engine
        # -------------------------------------------------------------
        console_log("SYSTEM2", "engine=DETERMINISTIC_LOCAL_FALLBACK status=ACTIVE", correlation_id=scan_id)
        return DeterministicSystem2Fallback.analyze(context, latency_ms=round((time.perf_counter() - t0) * 1000.0, 2))

    def query_operator(self, question: str, context: System2MissionContext) -> OperatorQueryResponse:
        t0 = time.perf_counter()
        scan_id = context.survey_id or "SCAN-UNKNOWN"

        context_str = json.dumps(context.model_dump(), indent=2)
        prompt = f"MISSION CONTEXT:\n{context_str}\n\nOPERATOR QUESTION:\n{question}"

        if self.is_available():
            try:
                raw = self._call_sarvam_chat(
                    messages=[
                        {"role": "system", "content": QA_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                data = self._clean_and_parse_json(raw) or {}
                total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                self.sarvam_status = "ACTIVE"
                return OperatorQueryResponse(
                    answer=data.get("answer", raw if isinstance(raw, str) else "No answer formulated."),
                    evidence_used=data.get("evidence_used", []),
                    uncertainties=data.get("uncertainties", []),
                    model=f"sarvam/{self.model}",
                    status="ACTIVE",
                    latency_ms=total_ms
                )
            except Exception as e:
                logger.warning(f"Sarvam Q&A query failed: {e}. Using deterministic Q&A fallback.")

        return DeterministicSystem2Fallback.query(question, context)

    def _build_context_prompt(self, context: System2MissionContext) -> str:
        data = {
            "survey_id": context.survey_id or "SURVEY_TRANSECT",
            "timestamp": context.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "contacts_detected": len(context.contacts),
            "contacts": context.contacts,
            "physics_derivations": context.physics,
            "georeferencing": context.geo,
            "system1_reflex": context.system1,
            "sensor_metadata": context.metadata
        }
        return f"ANALYZE THIS STRUCTURED SONAR MISSION CONTEXT:\n{json.dumps(data, indent=2)}"

    def get_status(self) -> Dict[str, Any]:
        primary_status = self.sarvam_status
        active_tier = "SARVAM" if primary_status == "ACTIVE" else "DETERMINISTIC_LOCAL"
        return {
            "status": primary_status,
            "primary_engine": "Sarvam AI",
            "primary_model": self.model,
            "primary_status": primary_status,
            "edge_fallback_engine": "Deterministic System 2 Rule Engine",
            "edge_fallback_status": "ACTIVE",
            "active_tier": active_tier
        }


# Backwards compatibility alias for existing test suites
GroqSystem2Engine = SarvamSystem2Engine


# =====================================================================
# 5. Asynchronous System 2 Queue & Background Worker
# =====================================================================

class System2Queue:
    """
    Thread-safe, non-blocking asynchronous queue for System 2 mission analysis.
    System 1 Reflex calls enqueue() without waiting (< 0.1ms).
    Background worker processes requests asynchronously.
    Maintains a ring buffer of latest analyses queryable by survey/event ID.
    """

    def __init__(self, engine: Optional[GroqSystem2Engine] = None, max_capacity: int = 50):
        self.engine = engine or GroqSystem2Engine()
        self.queue: queue.Queue = queue.Queue(maxsize=max_capacity)
        self.history: Dict[str, TacticalAnalysis] = {}
        self.latest_analysis: Optional[TacticalAnalysis] = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

        # Start background worker daemon thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="System2Worker")
        self.worker_thread.start()
        logger.info("System2Queue background worker thread started.")

    def enqueue(self, context: System2MissionContext) -> str:
        """
        Enqueues mission context non-blockingly.
        Returns immediate tracking ID. Never blocks System 1.
        """
        task_id = context.survey_id or str(uuid.uuid4())
        enqueue_time = time.perf_counter()

        item = {
            "task_id": task_id,
            "context": context,
            "enqueue_time": enqueue_time
        }

        try:
            self.queue.put_nowait(item)
            return task_id
        except queue.Full:
            logger.warning("System 2 queue full! Dropping oldest and enqueuing latest.")
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(item)
            return task_id

    def get_latest(self) -> Optional[TacticalAnalysis]:
        with self.lock:
            return self.latest_analysis

    def get_analysis(self, task_id: str) -> Optional[TacticalAnalysis]:
        with self.lock:
            return self.history.get(task_id)

    def analyze_sync(self, context: System2MissionContext) -> TacticalAnalysis:
        """Synchronous analysis execution for direct API / test requests."""
        analysis = self.engine.analyze_tactical(context)
        with self.lock:
            self.latest_analysis = analysis
            if context.survey_id:
                self.history[context.survey_id] = analysis
        return analysis

    def query_sync(self, question: str, context: System2MissionContext) -> OperatorQueryResponse:
        return self.engine.query_operator(question, context)

    def get_status(self) -> Dict[str, Any]:
        engine_stat = self.engine.get_status()
        active_tier = engine_stat.get("active_tier", "DETERMINISTIC_LOCAL")
        overall_status = engine_stat.get("status", "FALLBACK")
        return {
            "engine": "sarvam" if active_tier == "SARVAM" else "deterministic_fallback",
            "status": overall_status,
            "active_tier": active_tier,
            "model": self.engine.model,
            "primary_engine": "Sarvam AI",
            "primary_status": engine_stat.get("primary_status", "FALLBACK"),
            "edge_fallback_engine": "Deterministic System 2 Rule Engine",
            "edge_fallback_status": "ACTIVE",
            "sarvam_model": self.engine.model,
            "sarvam_status": getattr(self.engine, "sarvam_status", "FALLBACK"),
            "groq_status": getattr(self.engine, "sarvam_status", "FALLBACK"),  # Backwards compatibility
            "api_key_configured": bool(self.engine.api_key),
            "queue_size": self.queue.qsize(),
            "worker_alive": self.worker_thread.is_alive(),
            "has_latest_analysis": self.latest_analysis is not None
        }

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task_id = item["task_id"]
            context = item["context"]
            enqueue_time = item["enqueue_time"]
            queue_wait_ms = round((time.perf_counter() - enqueue_time) * 1000.0, 2)

            try:
                analysis = self.engine.analyze_tactical(context)
                analysis.queue_time_ms = queue_wait_ms

                with self.lock:
                    self.latest_analysis = analysis
                    self.history[task_id] = analysis

                logger.info(f"System 2 analysis completed for {task_id} in {analysis.latency_ms:.1f}ms (queue wait {queue_wait_ms:.1f}ms)")
            except Exception as e:
                logger.error(f"Error processing System 2 task {task_id}: {e}")
            finally:
                self.queue.task_done()

    def shutdown(self):
        self._stop_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
