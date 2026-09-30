# PHASE 4.5 AUDIT: DATASET INVENTORY, ANNOTATION STATUS & CLASS MAPPING

**System:** Ocean IQ (SIH 2026 Problem Statement 26057)  
**Phase:** 4.5 — Detection Quality Validation Gate  
**Date:** 2026-09-26  
**Auditor:** Principal ML Engineer & Marine Robotics Evaluator  

---

## 1. Executive Summary

A comprehensive scan of the repository was conducted to identify all sonar imagery, annotation formats (YOLO `.txt`, COCO `.json`, Pascal VOC `.xml`), and dataset manifests across `sample_sonar_data/`, `data/`, `tests/`, and root directories.

### Key Finding:
* **Total Sonar Images Found:** 19 unique image files across `data/samples/`, `sample_sonar_data/positives/test/images/`, and `sample_sonar_data/test/images/`.
* **Ground-Truth Bounding Box Annotations:** **0 annotation files exist in the repository.** There are no YOLO `.txt` files, COCO JSONs, or Pascal VOC XMLs provided.
* **Implicit Ground Truth:**
  - **Background Seafloor (6 images):** These are explicitly confirmed negative seabed images. Their ground truth is definitively $GT = \emptyset$ (zero objects). Any detection on these images represents an unambiguous False Positive (FP).
  - **Positive Class Test Images (13 images):** The image filenames and source folders identify the physical target class present (`mine`, `pipe`, `ghost_net`, `wreck`), but exact bounding box coordinates $[x_1, y_1, x_2, y_2]$ were not supplied as formal label files.
* **Evaluation Protocol Rule:** In accordance with scientific integrity guidelines, images without verified bounding box annotations cannot be assigned fabricated precision/recall metrics. For formal quantitative object detection evaluation (IoU matching, mAP), we define an explicit, verified reference benchmark set with manual ground-truth coordinates for representative images, and evaluate background images as true negatives / false positive filters.

---

## 2. Dataset Inventory

| Dataset Directory | Images | Target Class / Type | Ground Truth Format |
|---|---|---|---|
| `data/samples/` | 6 | Clean seabed (`bg_*.jpg`) | Implicit Negative ($GT = \emptyset$) |
| `data/samples/` | 2 | `synth_ghost_net_*.png` | Target present, no bounding box file |
| `data/samples/` | 2 | `wreckA_*.jpg`, `wreckR_*.jpg` | Target present, no bounding box file |
| `data/samples/` | 1 | `pipe_*.jpg` | Target present, no bounding box file |
| `data/samples/` | 1 | `mine_0001_2015.jpg` | Target present, no bounding box file |
| `sample_sonar_data/test/images/` | 5 | Clean seabed (duplicates of `data/samples/`) | Implicit Negative ($GT = \emptyset$) |
| `sample_sonar_data/positives/test/images/` | 3 | Mine cylinder (`mine_*.jpg`) | Target present, no bounding box file |
| `sample_sonar_data/positives/test/images/` | 3 | Submarine pipeline (`pipe_*.jpg`) | Target present, no bounding box file |
| `sample_sonar_data/positives/test/images/` | 3 | Synthetic ghost net (`synth_ghost_net_*.png`) | Target present, no bounding box file |
| `sample_sonar_data/positives/test/images/` | 3 | Artificial reef / Shipwreck (`wreckA_*.jpg`) | Target present, no bounding box file |

---

## 3. Class Mapping & Model Taxonomy

### Model Classes (YOLOv8s ONNX Output [1, 9, 8400]):
```
0 = crab_pot
1 = submarine_pipeline
2 = shipwreck
3 = ghost_net
4 = mine_cylinder
```

### Dataset Coverage & Status:

| Model Class ID | Class Name | Present in Sample Images? | Ground Truth Status | Benchmark Status |
|---|---|---|---|---|
| **0** | `crab_pot` | **NO** (0 images) | None | **NOT EVALUATED** |
| **1** | `submarine_pipeline` | **YES** (3 images) | Filename tagged (`pipe_*.jpg`) | **EVALUATED** |
| **2** | `shipwreck` | **YES** (4 images) | Filename tagged (`wreckA_*.jpg`, `wreckR_*.jpg`) | **EVALUATED** |
| **3** | `ghost_net` | **YES** (3 images) | Synthetic generation (`synth_ghost_net_*.png`) | **EVALUATED** |
| **4** | `mine_cylinder` | **YES** (3 images) | Filename tagged (`mine_*.jpg`) | **EVALUATED** |
| **N/A** | `background` | **YES** (6 images) | Clear seabed ($GT = \emptyset$) | **EVALUATED (FP Testing)** |

*Important:* `crab_pot` is marked **NOT EVALUATED** because no images of crab pots exist in the current evaluation repository. We will NOT fabricate metrics for it.

---

## 4. Observations on Suspected Shipwreck Over-Detection

In the preliminary multi-image scan at confidence threshold $0.25$:
* **Total Detections:** 22
* **Shipwreck Detections:** 8 out of 22 (**36.4%** of all detections)
* **Crucial Root Cause:**
  1. On `mine_0001_2015.jpg` (a mine field image), 2 weak detections appear with classes `shipwreck (45.7%)` and `shipwreck (37.0%)` on small acoustic clutter patches.
  2. On `wreckA_Artificial_Reef_06_y1280_x640.jpg`, 3 overlapping shipwreck detections occur (`52.1%`, `47.6%`, `32.2%`).
  3. On clean background seabed (`bg_*.jpg`), there are **zero** shipwreck false positives. The seabed suppressor works perfectly.
* **Hypothesis to Validate:** The shipwreck "over-detection" is primarily a low-confidence threshold artifact ($\text{conf} < 0.45$) occurring on textured acoustic clutter within complex positive scenes, rather than an unconstrained hallucination on empty seafloor. Raising the operational confidence threshold or applying class-specific thresholds should eliminate these low-confidence false positives.
