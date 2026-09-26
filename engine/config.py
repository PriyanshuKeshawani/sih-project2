"""
engine/config.py
Production-Grade Configuration & Mode Validation for SAMUDRA-AI.
Enforces the Absolute Real-Data Rule (Phase 8.5+ Production Hardening).

Modes:
- PRODUCTION: Real hardware/live streams and real recorded data only. No simulation or demo fallbacks.
- RECORDED_REAL_DATA: Curated historical datasets with verified provenance.
- TEST: Automated test suites, isolated fixtures, and simulation.

Startup strictly fails if APP_MODE is unset or invalid.
"""

import os
import sys
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class AppMode(str, Enum):
    PRODUCTION = "production"
    RECORDED_REAL_DATA = "recorded_real_data"
    TEST = "test"


class ProductionConfigurationError(Exception):
    """Raised when production safety rules or required environment configurations are violated."""
    pass


class SystemConfig:
    """
    Central validated configuration for SAMUDRA-AI.
    Guarantees no demo/simulation leaks into the production path.
    """

    def __init__(self, mode: Optional[str] = None):
        raw_mode = (mode or os.getenv("APP_MODE", "")).strip().lower()

        # If running inside pytest or unittest, allow graceful test mode detection
        if not raw_mode and ("unittest" in sys.modules or "pytest" in sys.modules):
            raw_mode = "test"

        if not raw_mode:
            raise ProductionConfigurationError(
                "CRITICAL: APP_MODE environment variable is missing.\n"
                "SAMUDRA-AI requires an explicit operating mode to prevent accidental demo data execution.\n"
                "Please set APP_MODE to one of: 'production', 'recorded_real_data', 'test'."
            )

        valid_modes = [m.value for m in AppMode]
        if raw_mode not in valid_modes:
            raise ProductionConfigurationError(
                f"CRITICAL: Invalid APP_MODE '{raw_mode}'.\n"
                f"Must be explicitly one of: {valid_modes}"
            )

        self.mode = AppMode(raw_mode)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Models
        self.model_path = os.getenv("MODEL_PATH", os.path.join(self.base_dir, "models", "best_detector.onnx"))
        self.laya_model_path = os.getenv("LAYA_MODEL_PATH", os.path.join(self.base_dir, "models", "laya", "checkpoint"))
        self.groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # Thresholds
        self.conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.45"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        # API Keys (Protected — never log directly)
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

    @property
    def is_production(self) -> bool:
        return self.mode == AppMode.PRODUCTION

    @property
    def is_recorded_real(self) -> bool:
        return self.mode == AppMode.RECORDED_REAL_DATA

    @property
    def is_test(self) -> bool:
        return self.mode == AppMode.TEST

    @property
    def allow_simulation(self) -> bool:
        """Simulation and demo data are strictly prohibited in PRODUCTION and RECORDED_REAL_DATA modes."""
        return self.mode == AppMode.TEST

    def to_safe_dict(self) -> dict:
        """Returns non-sensitive configuration state for observability and diagnostics."""
        return {
            "mode": self.mode.value.upper(),
            "model_path": self.model_path,
            "laya_model_path": self.laya_model_path,
            "groq_model": self.groq_model,
            "gemini_model": self.gemini_model,
            "confidence_threshold": self.conf_threshold,
            "log_level": self.log_level,
            "groq_key_configured": bool(self.groq_api_key),
            "gemini_key_configured": bool(self.gemini_api_key),
            "simulation_allowed": self.allow_simulation
        }


# Global validated configuration singleton
try:
    current_config = SystemConfig()
except ProductionConfigurationError:
    # Set to test mode if imported during unit testing setup
    if "unittest" in sys.modules or "pytest" in sys.modules:
        current_config = SystemConfig(mode="test")
    else:
        raise


def get_current_config() -> SystemConfig:
    global current_config
    env_mode = os.getenv("APP_MODE")
    if current_config is None or (env_mode and env_mode.strip().lower() != current_config.mode.value):
        current_config = SystemConfig()
    return current_config


def validate_startup_environment() -> SystemConfig:
    global current_config
    current_config = SystemConfig()
    return current_config
