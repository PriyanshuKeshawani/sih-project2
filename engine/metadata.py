from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ProvenanceStatus(str, Enum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    SIMULATED = "SIMULATED"
    DEMO = "DEMO"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class PhysicsMeasurement:
    value: Optional[float]
    unit: str
    status: str  # ProvenanceStatus value
    method: str
    confidence: Optional[float] = None


@dataclass
class SonarGeometry:
    vehicle_altitude_m: Optional[float] = None
    slant_range_m: Optional[float] = None
    shadow_length_m: Optional[float] = None
    vehicle_depth_m: Optional[float] = None
    meters_per_pixel: Optional[float] = None
    swath_range_m: Optional[float] = None
    source: str = "UNAVAILABLE"


@dataclass
class SurveyMetadata:
    survey_id: Optional[str] = None
    timestamp: Optional[str] = None
    platform_lat: Optional[float] = None
    platform_lon: Optional[float] = None
    vehicle_depth_m: Optional[float] = None
    sonar_altitude_m: Optional[float] = None
    heading_deg: Optional[float] = None
    slant_range_m: Optional[float] = None
    swath_range_m: Optional[float] = None
    meters_per_pixel: Optional[float] = None
    sonar_model: Optional[str] = None
    source_type: str = "UNKNOWN"  # REAL_SENSOR, RECORDED_SURVEY, USER_PROVIDED, SIMULATED, DEMO, UNKNOWN
    coordinate_reference_system: str = "EPSG:4326"


@dataclass
class GeoResult:
    latitude: Optional[float]
    longitude: Optional[float]
    status: str  # REAL, DEMO, SIMULATED, UNAVAILABLE
    source: str
    position_type: str = "PLATFORM_POSITION"  # PLATFORM_POSITION or OBJECT_POSITION
    accuracy_m: Optional[float] = None
    zone_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "status": self.status,
            "source": self.source,
            "position_type": self.position_type,
            "accuracy_m": self.accuracy_m,
            "zone_label": self.zone_label
        }


# =========================================================================
# Navigation Adapters (Requirement 10)
# =========================================================================
class NavigationSource:
    """Base class for all navigation/telemetry providers."""
    def get_metadata(self) -> SurveyMetadata:
        raise NotImplementedError


class UserMetadata(NavigationSource):
    """User-supplied survey parameters from REST API or UI forms."""
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def get_metadata(self) -> SurveyMetadata:
        return SurveyMetadata(
            survey_id=self.data.get("survey_id"),
            timestamp=self.data.get("timestamp"),
            platform_lat=self.data.get("latitude") or self.data.get("platform_lat"),
            platform_lon=self.data.get("longitude") or self.data.get("platform_lon"),
            vehicle_depth_m=self.data.get("vehicle_depth_m"),
            sonar_altitude_m=self.data.get("sonar_altitude_m") or self.data.get("altitude_m"),
            heading_deg=self.data.get("heading_deg"),
            slant_range_m=self.data.get("slant_range_m"),
            swath_range_m=self.data.get("swath_range_m"),
            meters_per_pixel=self.data.get("meters_per_pixel"),
            sonar_model=self.data.get("sonar_model"),
            source_type="USER_PROVIDED" if self.data.get("platform_lat") or self.data.get("latitude") else "UNKNOWN"
        )


class RecordedNavigation(NavigationSource):
    """Reads telemetry from recorded survey log dictionaries."""
    def __init__(self, record: Dict[str, Any]):
        self.record = record

    def get_metadata(self) -> SurveyMetadata:
        return SurveyMetadata(
            survey_id=self.record.get("survey_id", "RECORDED_LOG"),
            timestamp=self.record.get("timestamp"),
            platform_lat=self.record.get("lat"),
            platform_lon=self.record.get("lon"),
            vehicle_depth_m=self.record.get("depth_m"),
            sonar_altitude_m=self.record.get("alt_m"),
            heading_deg=self.record.get("heading_deg"),
            meters_per_pixel=self.record.get("m_per_px"),
            source_type="RECORDED_SURVEY"
        )


class NMEAAdapter(NavigationSource):
    """
    Parses standard recorded NMEA-0183 sentences (GPGGA, GPRMC).
    Does NOT pretend a live sensor stream exists; parses recorded sentence strings.
    """
    def __init__(self, nmea_sentences: List[str]):
        self.sentences = nmea_sentences

    def get_metadata(self) -> SurveyMetadata:
        lat, lon = None, None
        for s in self.sentences:
            parts = s.strip().split(',')
            if len(parts) > 6 and (parts[0] in ['$GPGGA', 'GPGGA', '$GNGGA', 'GNGGA']):
                # Raw NMEA: ddmm.mmmm
                try:
                    raw_lat = float(parts[2])
                    lat_deg = int(raw_lat / 100)
                    lat_min = raw_lat - (lat_deg * 100)
                    lat = lat_deg + (lat_min / 60.0)
                    if parts[3].upper() == 'S':
                        lat = -lat

                    raw_lon = float(parts[4])
                    lon_deg = int(raw_lon / 100)
                    lon_min = raw_lon - (lon_deg * 100)
                    lon = lon_deg + (lon_min / 60.0)
                    if parts[5].upper() == 'W':
                        lon = -lon
                    break
                except (ValueError, IndexError):
                    continue

        return SurveyMetadata(
            survey_id="NMEA_STREAM_RECORD",
            platform_lat=lat,
            platform_lon=lon,
            source_type="REAL_SENSOR" if lat is not None else "UNAVAILABLE"
        )


class SimulatedNavigation(NavigationSource):
    """Generates synthetic transect telemetry with explicit SIMULATED provenance.
    Strictly isolated: Prohibited from execution in PRODUCTION or RECORDED_REAL_DATA modes.
    """
    def __init__(self, base_lat: float = 9.2882, base_lon: float = 79.1325, altitude_m: float = 12.0):
        from engine.config import get_current_config
        cfg = get_current_config()
        if not cfg.allow_simulation:
            raise RuntimeError(
                f"SimulatedNavigation is strictly prohibited in {cfg.mode.value} mode. "
                "Simulation data is isolated under tests/fixtures/simulation."
            )
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.altitude_m = altitude_m

    def get_metadata(self) -> SurveyMetadata:
        return SurveyMetadata(
            survey_id="SIMULATED_TRANSECT_01",
            platform_lat=self.base_lat,
            platform_lon=self.base_lon,
            sonar_altitude_m=self.altitude_m,
            vehicle_depth_m=20.0,
            heading_deg=90.0,
            meters_per_pixel=0.05,
            source_type="SIMULATED"
        )


