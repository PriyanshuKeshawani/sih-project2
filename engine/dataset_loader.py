"""
engine/dataset_loader.py
Dataset Adapter & Loader for SAMUDRA-AI (SIH 2026 PS 26057).

Manages external and local sonar datasets:
- DRISHTI SSS (CC-BY-SA-4.0)
- SubPipe (CC-BY-4.0)
- AI4Shipwrecks (CC-BY-4.0)
- NOAA Ocean Exploration (Public Domain)

Guarantees:
1. Datasets are NEVER mixed silently.
2. Every loaded scan explicitly records:
   - dataset_name
   - source_url
   - license
   - image_filename
   - provenance
3. Scenarios that do not physically exist are marked 'NOT AVAILABLE'.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE_DIR, "dataset_manifest.json")


@dataclass
class SonarSampleRecord:
    dataset_name: str
    source_url: str
    license: str
    image_filename: str
    image_path: str
    target_class: str
    provenance: str
    scenario_id: Optional[str] = None
    annotations_available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SonarDatasetLoader:
    """
    Standardized dataset adapter for public side-scan sonar benchmarks.
    """

    def __init__(self, manifest_path: str = MANIFEST_PATH):
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"datasets": []}

    def get_datasets(self) -> List[Dict[str, Any]]:
        return self.manifest.get("datasets", [])

    def resolve_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(BASE_DIR, rel_path)

    def get_scenario_catalog(self) -> Dict[str, Dict[str, Any]]:
        """
        Defines canonical test scenarios A-G as specified in Phase 8.
        """
        catalog = {
            "Scenario A": {
                "name": "Clean Background Seabed",
                "rel_path": "data/samples/bg_1693569243.750_x2500.jpg",
                "dataset_name": "SubPipe / DRISHTI SSS",
                "source_url": "https://github.com/remaro-network/SubPipe-dataset",
                "license": "CC-BY-4.0",
                "target_class": "clean_seabed",
                "provenance": "Negative control swath from SubPipe autonomous survey. Ground truth = empty."
            },
            "Scenario B": {
                "name": "Ghost Net (Derelict Gear)",
                "rel_path": "data/samples/synth_ghost_net_00001.png",
                "dataset_name": "DRISHTI SSS Synthetic Split",
                "source_url": "https://huggingface.co/datasets/rehan9599/drishti-sss",
                "license": "CC-BY-SA-4.0",
                "target_class": "ghost_net",
                "provenance": "High-contrast synthetic derelict fishing net signature composited on Roboflow seabed canvas."
            },
            "Scenario C": {
                "name": "Shipwreck Hull / Artificial Reef",
                "rel_path": "data/samples/wreckA_Artificial_Reef_06_y1280_x0.jpg",
                "dataset_name": "AI4Shipwrecks",
                "source_url": "https://umfieldrobotics.github.io/ai4shipwrecks/",
                "license": "CC-BY-4.0",
                "target_class": "shipwreck",
                "provenance": "NOAA Thunder Bay National Marine Sanctuary side-scan survey."
            },
            "Scenario D": {
                "name": "Submarine Pipeline",
                "rel_path": "data/samples/pipe_1693569383.780_x3500.jpg",
                "dataset_name": "SubPipe",
                "source_url": "https://github.com/remaro-network/SubPipe-dataset",
                "license": "CC-BY-4.0",
                "target_class": "submarine_pipeline",
                "provenance": "Continuous linear pipeline benthic feature from OceanScan-MST survey."
            },
            "Scenario E": {
                "name": "Mine-Cylinder Contact",
                "rel_path": "data/samples/mine_0001_2015.jpg",
                "dataset_name": "DRISHTI SSS (MILCO)",
                "source_url": "https://huggingface.co/datasets/rehan9599/drishti-sss",
                "license": "CC-BY-SA-4.0",
                "target_class": "mine_cylinder",
                "provenance": "MILCO cylindrical bottom mine anomaly with acoustic shadow."
            },
            "Scenario F": {
                "name": "Unknown / Non-Taxonomy Object",
                "rel_path": "data/downloaded/noaa_fig2_sidescan.png",
                "dataset_name": "NOAA Ocean Exploration",
                "source_url": "https://oceanexplorer.noaa.gov/multimedia/georeferenced-side-scan-sonar-image/",
                "license": "U.S. Public Domain",
                "target_class": "non_taxonomy_geology",
                "provenance": "Real NOAA georeferenced side-scan swath containing natural benthic geology not present in 5-class detector training taxonomy."
            },
            "Scenario G": {
                "name": "Mixed Multi-Contact Scene",
                "rel_path": "sample_sonar_data/positives/test/images/wreckA_Artificial_Reef_06_y1280_x320.jpg",
                "dataset_name": "AI4Shipwrecks / DRISHTI SSS",
                "source_url": "https://umfieldrobotics.github.io/ai4shipwrecks/",
                "license": "CC-BY-4.0",
                "target_class": "multi_contact",
                "provenance": "Complex multi-target acoustic scene with structural wreck components and shadow boundaries."
            }
        }
        return catalog

    def load_scenario(self, scenario_id: str) -> Tuple[Optional[np.ndarray], SonarSampleRecord]:
        """
        Loads a scenario image and its complete provenance record.
        If file is absent or scenario does not exist, marks as NOT AVAILABLE.
        """
        catalog = self.get_scenario_catalog()
        if scenario_id not in catalog:
            rec = SonarSampleRecord(
                dataset_name="NOT AVAILABLE",
                source_url="NOT AVAILABLE",
                license="NOT AVAILABLE",
                image_filename="NOT AVAILABLE",
                image_path="NOT AVAILABLE",
                target_class="NOT AVAILABLE",
                provenance="Scenario not defined in catalog.",
                scenario_id=scenario_id,
                annotations_available=False
            )
            return None, rec

        spec = catalog[scenario_id]
        full_path = self.resolve_path(spec["rel_path"])

        if not os.path.exists(full_path):
            rec = SonarSampleRecord(
                dataset_name=spec["dataset_name"],
                source_url=spec["source_url"],
                license=spec["license"],
                image_filename=os.path.basename(spec["rel_path"]),
                image_path=full_path,
                target_class=spec["target_class"],
                provenance="NOT AVAILABLE (image file missing on disk)",
                scenario_id=scenario_id,
                annotations_available=False
            )
            return None, rec

        img = cv2.imread(full_path)
        if img is None:
            rec = SonarSampleRecord(
                dataset_name=spec["dataset_name"],
                source_url=spec["source_url"],
                license=spec["license"],
                image_filename=os.path.basename(spec["rel_path"]),
                image_path=full_path,
                target_class=spec["target_class"],
                provenance="NOT AVAILABLE (image decoding error)",
                scenario_id=scenario_id,
                annotations_available=False
            )
            return None, rec

        rec = SonarSampleRecord(
            dataset_name=spec["dataset_name"],
            source_url=spec["source_url"],
            license=spec["license"],
            image_filename=os.path.basename(spec["rel_path"]),
            image_path=full_path,
            target_class=spec["target_class"],
            provenance=spec["provenance"],
            scenario_id=scenario_id,
            annotations_available=True
        )
        return img, rec

    def list_all_samples(self) -> List[SonarSampleRecord]:
        """
        Enumerates all available samples across all tracked datasets.
        """
        records = []
        for d in self.get_datasets():
            dname = d.get("source", "UNKNOWN")
            url = d.get("url", "")
            lic = d.get("license", "UNKNOWN")
            prov = d.get("provenance", "")
            for sf in d.get("sample_files", []):
                full_path = self.resolve_path(sf)
                if os.path.exists(full_path):
                    records.append(SonarSampleRecord(
                        dataset_name=dname,
                        source_url=url,
                        license=lic,
                        image_filename=os.path.basename(sf),
                        image_path=full_path,
                        target_class="auto_detect",
                        provenance=prov
                    ))
        return records
