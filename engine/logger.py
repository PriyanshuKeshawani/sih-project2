"""
engine/logger.py
Production-Grade Console Observability & Structured Audit Logger for SAMUDRA-AI.
Enforces Section 8, 9, 10, 11, 27, 28 of Production-Grade Hardening.

Features:
- Standard bracketed domain tags: [STARTUP], [REQUEST], [INPUT], [PREPROCESS], [TILING],
  [INFERENCE], [NMS], [PHYSICS], [GEO], [TEMPORAL], [SYSTEM1], [LAYA], [GUARDRAIL],
  [SYSTEM2], [GROQ], [GEMINI], [PDF], [DEPLOYMENT], [ERROR], [FINAL].
- Request correlation across all logs: [COMPONENT][REQ-ID/SCAN-ID].
- Execution duration measurement via time.perf_counter().
- Secret suppression: automatically scrubs API keys and tokens.
- Boxed Final Scan Summary.
"""

import sys
import re
import time
import logging
from typing import Optional, Dict, Any, List

# Sensitive token scrubbers
SECRET_PATTERNS = [
    re.compile(r'gsk_[A-Za-z0-9_]{16,}', re.IGNORECASE),
    re.compile(r'AQ\.[A-Za-z0-9_-]{20,}', re.IGNORECASE),
    re.compile(r'(api_key|apikey|secret|token|password)\s*[:=]\s*["\']?([^"\'\s]+)["\']?', re.IGNORECASE)
]


def scrub_secrets(message: str) -> str:
    """Replaces sensitive credential strings with [REDACTED]."""
    if not isinstance(message, str):
        message = str(message)
    for pat in SECRET_PATTERNS:
        if 'api_key' in pat.pattern:
            message = pat.sub(r'\1=[REDACTED]', message)
        else:
            message = pat.sub('[REDACTED_SECRET]', message)
    return message


from collections import deque

_RECENT_LOGS = deque(maxlen=300)

class ProductionLogger:
    """
    Structured, human-readable console observer for real-time mission execution.
    """

    def __init__(self, log_level: str = "INFO"):
        self.level_name = log_level.upper()
        self.level = getattr(logging, self.level_name, logging.INFO)

    def _should_log(self, level: int) -> bool:
        return level >= self.level

    def log_tag(self, tag: str, message: str, correlation_id: Optional[str] = None, level: int = logging.INFO):
        if not self._should_log(level):
            return
        tag_str = f"[{tag.upper()}]"
        if correlation_id:
            tag_str += f"[{correlation_id}]"
        safe_msg = scrub_secrets(message)
        formatted = f"{tag_str} {safe_msg}"
        _RECENT_LOGS.append(formatted)
        print(formatted, flush=True)

    def get_recent_logs(self, limit: int = 100) -> List[str]:
        return list(_RECENT_LOGS)[-limit:]

    def startup(self, msg: str):
        self.log_tag("STARTUP", msg)

    def request(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("REQUEST", msg, correlation_id)

    def input(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("INPUT", msg, correlation_id)

    def preprocess(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("PREPROCESS", msg, correlation_id)

    def tiling(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("TILING", msg, correlation_id)

    def inference(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("INFERENCE", msg, correlation_id)

    def nms(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("NMS", msg, correlation_id)

    def physics(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("PHYSICS", msg, correlation_id)

    def geo(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("GEO", msg, correlation_id)

    def temporal(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("TEMPORAL", msg, correlation_id)

    def system1(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("SYSTEM1", msg, correlation_id)

    def laya(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("LAYA", msg, correlation_id)

    def guardrail(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("GUARDRAIL", msg, correlation_id)

    def system2(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("SYSTEM2", msg, correlation_id)

    def groq(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("GROQ", msg, correlation_id)

    def gemini(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("GEMINI", msg, correlation_id)

    def pdf(self, msg: str, correlation_id: Optional[str] = None):
        self.log_tag("PDF", msg, correlation_id)

    def deployment(self, msg: str):
        self.log_tag("DEPLOYMENT", msg)

    def error(self, component: str, msg: str, correlation_id: Optional[str] = None, exc: Optional[Exception] = None):
        tag = f"ERROR][{component.upper()}"
        detail = msg
        if exc is not None:
            detail += f" | {type(exc).__name__}: {str(exc)}"
        self.log_tag(tag, detail, correlation_id, level=logging.ERROR)

    def print_scan_summary(
        self,
        scan_id: str,
        source: str,
        detections_count: int,
        persistent_count: int,
        system1_decision: str,
        system2_status: str,
        gps_status: str,
        physics_status: str,
        pdf_status: str,
        total_ms: float,
        final_status: str
    ):
        """Prints the mandatory Section 28 Final Scan Summary box."""
        banner = (
            "\n"
            "====================================================\n"
            "FINAL SCAN SUMMARY\n"
            "====================================================\n"
            f"scan_id={scan_id}\n"
            f"source={source}\n"
            f"detections={detections_count}\n"
            f"persistent_contacts={persistent_count}\n"
            f"system1={system1_decision}\n"
            f"system2={system2_status}\n"
            f"gps={gps_status}\n"
            f"physics={physics_status}\n"
            f"pdf={pdf_status}\n"
            f"total_ms={total_ms:.2f}\n"
            f"status={final_status}\n"
            "====================================================\n"
        )
        for line in banner.strip().split("\n"):
            _RECENT_LOGS.append(line)
        print(banner, flush=True)


# Global logger instance
logger = ProductionLogger()


def get_console_logger() -> ProductionLogger:
    return logger


def get_recent_logs(limit: int = 100) -> List[str]:
    return logger.get_recent_logs(limit)


def console_log(tag: str, message: str, scan_id: Optional[str] = None, correlation_id: Optional[str] = None, level: int = logging.INFO):
    cid = scan_id or correlation_id
    logger.log_tag(tag, message, correlation_id=cid, level=level)
