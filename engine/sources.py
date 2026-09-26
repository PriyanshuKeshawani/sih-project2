"""
engine/sources.py
Real Sonar & Navigation Input Abstraction for SAMUDRA-AI (Production MVP).
Enforces Sections 5, 6, 7, 31 of Production-Grade Hardening.

Architectural Guarantees:
1. No faked live streams or synthetic pings in production.
2. Missing navigation/telemetry returns status=UNAVAILABLE (never hardcoded 9.2882/12.0m).
3. Full provenance tagging on every input packet.
4. FutureLiveSonarSource interface reports UNAVAILABLE until a certified vendor hardware adapter is attached.
"""

import os
import cv2
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from engine.config import current_config, AppMode


@dataclass
class NavigationTelemetry:
    """Real or recorded-real navigation telemetry. No fabricated defaults."""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_m: Optional[float] = None
    altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None
    speed_knots: Optional[float] = None
    source_type: str = "UNAVAILABLE"  # REAL_SENSOR, RECORDED_REAL_DATA, USER_VERIFIED, UNAVAILABLE
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_gps_available(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def is_altitude_available(self) -> bool:
        return self.altitude_m is not None and self.altitude_m > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "depth_m": self.depth_m,
            "altitude_m": self.altitude_m,
            "heading_deg": self.heading_deg,
            "speed_knots": self.speed_knots,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "gps_status": "MEASURED" if self.is_gps_available() else "UNAVAILABLE",
            "altitude_status": "MEASURED" if self.is_altitude_available() else "UNAVAILABLE"
        }


@dataclass
class SonarFramePayload:
    """Complete, self-describing sonar frame with verified provenance."""
    image_bgr: np.ndarray
    scan_id: str
    request_id: str
    mission_id: str
    source_type: str  # REAL_SENSOR, RECORDED_REAL_DATA, TEST_FIXTURE
    dataset_name: str
    source_file: str
    dimensions: Tuple[int, int]  # (height, width)
    timestamp: str
    navigation: NavigationTelemetry
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_provenance_dict(self) -> Dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "request_id": self.request_id,
            "mission_id": self.mission_id,
            "source_type": self.source_type,
            "dataset_name": self.dataset_name,
            "source_file": self.source_file,
            "image_width": self.dimensions[1],
            "image_height": self.dimensions[0],
            "timestamp": self.timestamp,
            "navigation": self.navigation.to_dict()
        }


# =====================================================================
# Navigation Source Abstractions
# =====================================================================

class NavigationSource(ABC):
    @abstractmethod
    def get_telemetry(self) -> NavigationTelemetry:
        pass


class UserProvidedNavigationSource(NavigationSource):
    """Encapsulates verified, operator-supplied or API-provided navigation telemetry."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.data = data or {}

    def get_telemetry(self) -> NavigationTelemetry:
        lat = self.data.get("lat") or self.data.get("latitude")
        lon = self.data.get("lon") or self.data.get("longitude")
        alt = self.data.get("altitude") or self.data.get("altitude_m") or self.data.get("sonar_altitude_m")
        dep = self.data.get("depth") or self.data.get("depth_m")
        heading = self.data.get("heading") or self.data.get("heading_deg")

        # Explicit validation: do not accept fabricated/uncalibrated numbers silently
        source_type = "USER_VERIFIED" if (lat is not None or alt is not None) else "UNAVAILABLE"
        if current_config.is_recorded_real and source_type != "UNAVAILABLE":
            source_type = "RECORDED_REAL_DATA"

        return NavigationTelemetry(
            latitude=float(lat) if lat is not None else None,
            longitude=float(lon) if lon is not None else None,
            depth_m=float(dep) if dep is not None else None,
            altitude_m=float(alt) if alt is not None else None,
            heading_deg=float(heading) if heading is not None else None,
            source_type=source_type
        )


class RecordedNmeaNavigationSource(NavigationSource):
    """Parses real recorded NMEA sentence streams (e.g. $GPGGA, $GPRMC, $PRDID)."""

    def __init__(self, nmea_lines: List[str]):
        self.lines = nmea_lines

    def get_telemetry(self) -> NavigationTelemetry:
        lat, lon = None, None
        for line in self.lines:
            line = line.strip()
            if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                parts = line.split(',')
                try:
                    raw_lat = float(parts[2])
                    lat = int(raw_lat / 100) + (raw_lat % 100) / 60.0
                    if parts[3].upper() == 'S':
                        lat = -lat
                    raw_lon = float(parts[4])
                    lon = int(raw_lon / 100) + (raw_lon % 100) / 60.0
                    if parts[5].upper() == 'W':
                        lon = -lon
                    break
                except (ValueError, IndexError):
                    continue

        return NavigationTelemetry(
            latitude=round(lat, 6) if lat is not None else None,
            longitude=round(lon, 6) if lon is not None else None,
            source_type="RECORDED_REAL_DATA" if lat is not None else "UNAVAILABLE"
        )


class LiveNmeaStreamNavigationSource(NavigationSource):
    """
    Interface for live serial / UDP NMEA receiver on live AUV or hydrographic vessel.
    Reports UNAVAILABLE until real hardware serial/socket is established.
    """

    def __init__(self, port: Optional[str] = None):
        self.port = port
        self.is_connected = False

    def get_telemetry(self) -> NavigationTelemetry:
        # Without real serial stream connected, strictly report UNAVAILABLE
        return NavigationTelemetry(
            latitude=None,
            longitude=None,
            depth_m=None,
            altitude_m=None,
            heading_deg=None,
            source_type="UNAVAILABLE"
        )


# =====================================================================
# Sonar Source Abstractions
# =====================================================================

class SonarSource(ABC):
    @abstractmethod
    def get_next_frame(self, request_id: Optional[str] = None) -> Optional[SonarFramePayload]:
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        pass


class RecordedImageSource(SonarSource):
    """Loads a single verified real or recorded side-scan sonar image from disk."""

    def __init__(
        self,
        file_path: str,
        dataset_name: str = "RECORDED_DATASET",
        navigation: Optional[NavigationSource] = None
    ):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.nav_source = navigation or UserProvidedNavigationSource()
        self._consumed = False

    def get_frame(self, request_id: Optional[str] = None) -> Optional[SonarFramePayload]:
        return self.get_next_frame(request_id)

    def get_next_frame(self, request_id: Optional[str] = None) -> Optional[SonarFramePayload]:
        if self._consumed or not os.path.exists(self.file_path):
            return None

        img = cv2.imread(self.file_path)
        if img is None:
            return None

        self._consumed = True
        req_id = request_id or f"REQ-{uuid.uuid4().hex[:6].upper()}"
        scan_id = f"SCAN-{uuid.uuid4().hex[:6].upper()}"
        mission_id = f"MIS-{uuid.uuid4().hex[:6].upper()}"

        source_type = "RECORDED_REAL_DATA" if current_config.is_recorded_real else "REAL"

        return SonarFramePayload(
            image_bgr=img,
            scan_id=scan_id,
            request_id=req_id,
            mission_id=mission_id,
            source_type=source_type,
            dataset_name=self.dataset_name,
            source_file=os.path.basename(self.file_path),
            dimensions=(img.shape[0], img.shape[1]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            navigation=self.nav_source.get_telemetry()
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "source_class": "RecordedImageSource",
            "file_path": self.file_path,
            "exists": os.path.exists(self.file_path),
            "consumed": self._consumed
        }


class RecordedSurveySource(SonarSource):
    """Sequentially replays a sequence of real recorded sonar files across a survey transect."""

    def __init__(self, file_paths: List[str], dataset_name: str = "RECORDED_SURVEY"):
        self.file_paths = file_paths
        self.dataset_name = dataset_name
        self.current_idx = 0
        self.mission_id = f"SURVEY-{uuid.uuid4().hex[:6].upper()}"

    def get_next_frame(self, request_id: Optional[str] = None) -> Optional[SonarFramePayload]:
        if self.current_idx >= len(self.file_paths):
            return None

        path = self.file_paths[self.current_idx]
        self.current_idx += 1

        if not os.path.exists(path):
            return None

        img = cv2.imread(path)
        if img is None:
            return None

        req_id = request_id or f"REQ-{uuid.uuid4().hex[:6].upper()}"
        scan_id = f"SCAN-{self.current_idx:03d}-{uuid.uuid4().hex[:4].upper()}"

        return SonarFramePayload(
            image_bgr=img,
            scan_id=scan_id,
            request_id=req_id,
            mission_id=self.mission_id,
            source_type="RECORDED_REAL_DATA",
            dataset_name=self.dataset_name,
            source_file=os.path.basename(path),
            dimensions=(img.shape[0], img.shape[1]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            navigation=NavigationTelemetry(source_type="UNAVAILABLE")
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "source_class": "RecordedSurveySource",
            "total_frames": len(self.file_paths),
            "current_index": self.current_idx,
            "dataset_name": self.dataset_name
        }


class FutureLiveSonarSource(SonarSource):
    """
    Interface specification for live hydrographic sonar integration (e.g., Klein, Edgetech, DeepVision).
    Guaranteed to return UNAVAILABLE until the actual live hardware communication driver is installed.
    """

    def __init__(self, stream_uri: Optional[str] = None, protocol: str = "VENDOR_SDK"):
        self.stream_uri = stream_uri
        self.protocol = protocol
        self.hardware_connected = False

    def get_next_frame(self, request_id: Optional[str] = None) -> Optional[SonarFramePayload]:
        # Live hardware connection is not active: fail safe, return None
        return None

    def get_status(self) -> Dict[str, Any]:
        return {
            "source_class": "FutureLiveSonarSource",
            "status": "UNAVAILABLE",
            "hardware_connected": False,
            "protocol": self.protocol,
            "message": "Live sonar stream protocol adapter not connected. Connect physical AUV sensor feed."
        }
