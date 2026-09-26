from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class GroundTruthBox:
    bbox_xyxy: List[int]
    class_name: str


@dataclass
class PredictionBox:
    bbox_xyxy: List[int]
    class_name: str
    confidence: float


@dataclass
class MatchItem:
    pred_idx: Optional[int]
    gt_idx: Optional[int]
    class_name: str
    confidence: Optional[float]
    iou: float
    status: str  # TP, FP, FN
    pred_box: Optional[List[int]] = None
    gt_box: Optional[List[int]] = None


@dataclass
class ClassMetrics:
    class_name: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationSummary:
    iou_threshold: float
    confidence_threshold: float
    total_images: int
    total_tp: int
    total_fp: int
    total_fn: int
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, ClassMetrics]
    matches: List[MatchItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iou_threshold": self.iou_threshold,
            "confidence_threshold": self.confidence_threshold,
            "total_images": self.total_images,
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class": {
                k: {
                    "tp": v.tp,
                    "fp": v.fp,
                    "fn": v.fn,
                    "precision": round(v.precision, 4),
                    "recall": round(v.recall, 4),
                    "f1": round(v.f1, 4)
                } for k, v in self.per_class.items()
            }
        }


class ObjectDetectionEvaluator:
    """
    Standard Object Detection Evaluator for Side-Scan Sonar Imagery.
    Performs IoU-based matching between predicted boxes and ground-truth boxes.
    Enforces strict class consistency and prevents double-matching.
    """

    @staticmethod
    def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
        """
        Calculates Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2].
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        union_area = areaA + areaB - inter_area
        if union_area <= 0.0:
            return 0.0

        return float(inter_area / union_area)

    @classmethod
    def match_image_detections(
        cls,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        iou_threshold: float = 0.50
    ) -> List[MatchItem]:
        """
        Matches predictions against ground truth for a single image.
        1. Sorts predictions by descending confidence.
        2. Greedy matching to eligible, unmatched ground truth boxes with matching class.
        3. Marks matched predictions as TP, unmatched as FP, unmatched GT as FN.
        """
        # Parse inputs
        preds = []
        for i, p in enumerate(predictions):
            bbox = p.get("bbox_xyxy") or [
                p["box"]["x"], p["box"]["y"],
                p["box"]["x"] + p["box"]["w"], p["box"]["y"] + p["box"]["h"]
            ]
            preds.append({
                "idx": i,
                "box": [float(v) for v in bbox],
                "class": p.get("class") or p.get("class_name"),
                "confidence": float(p.get("confidence", 0.0))
            })

        # Sort descending by confidence
        preds.sort(key=lambda x: x["confidence"], reverse=True)

        gts = []
        for j, g in enumerate(ground_truths):
            bbox = g.get("bbox_xyxy") or [
                g["box"]["x"], g["box"]["y"],
                g["box"]["x"] + g["box"]["w"], g["box"]["y"] + g["box"]["h"]
            ]
            gts.append({
                "idx": j,
                "box": [float(v) for v in bbox],
                "class": g.get("class") or g.get("class_name"),
                "matched": False
            })

        matches: List[MatchItem] = []

        # Match predictions
        for p in preds:
            best_iou = 0.0
            best_gt_idx = None

            for g in gts:
                if g["matched"]:
                    continue
                if g["class"] != p["class"]:
                    continue

                iou = cls.calculate_iou(p["box"], g["box"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g["idx"]

            if best_gt_idx is not None:
                # Mark GT as consumed (prevent double match)
                for g in gts:
                    if g["idx"] == best_gt_idx:
                        g["matched"] = True
                        gt_box = g["box"]
                        break
                matches.append(MatchItem(
                    pred_idx=p["idx"],
                    gt_idx=best_gt_idx,
                    class_name=p["class"],
                    confidence=p["confidence"],
                    iou=round(best_iou, 4),
                    status="TP",
                    pred_box=[int(v) for v in p["box"]],
                    gt_box=[int(v) for v in gt_box]
                ))
            else:
                # Unmatched prediction -> FP
                matches.append(MatchItem(
                    pred_idx=p["idx"],
                    gt_idx=None,
                    class_name=p["class"],
                    confidence=p["confidence"],
                    iou=0.0,
                    status="FP",
                    pred_box=[int(v) for v in p["box"]],
                    gt_box=None
                ))

        # Unmatched ground truths -> FN
        for g in gts:
            if not g["matched"]:
                matches.append(MatchItem(
                    pred_idx=None,
                    gt_idx=g["idx"],
                    class_name=g["class"],
                    confidence=None,
                    iou=0.0,
                    status="FN",
                    pred_box=None,
                    gt_box=[int(v) for v in g["box"]]
                ))

        return matches

    @classmethod
    def evaluate_dataset(
        cls,
        dataset_predictions: List[List[Dict[str, Any]]],
        dataset_ground_truths: List[List[Dict[str, Any]]],
        iou_threshold: float = 0.50,
        confidence_threshold: float = 0.25,
        target_classes: Optional[List[str]] = None
    ) -> EvaluationSummary:
        """
        Evaluates a complete dataset across multiple images.
        Computes TP, FP, FN, Precision, Recall, and F1 per class and overall.
        """
        all_matches: List[MatchItem] = []
        num_images = len(dataset_ground_truths)

        for preds, gts in zip(dataset_predictions, dataset_ground_truths):
            # Filter predictions by confidence threshold
            filtered_preds = [p for p in preds if p.get("confidence", 0.0) >= confidence_threshold]
            matches = cls.match_image_detections(filtered_preds, gts, iou_threshold=iou_threshold)
            all_matches.extend(matches)

        # Collect evaluated classes
        classes = set()
        if target_classes:
            classes.update(target_classes)
        for m in all_matches:
            classes.add(m.class_name)

        per_class: Dict[str, ClassMetrics] = {}
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for c in sorted(classes):
            c_matches = [m for m in all_matches if m.class_name == c]
            tp = sum(1 for m in c_matches if m.status == "TP")
            fp = sum(1 for m in c_matches if m.status == "FP")
            fn = sum(1 for m in c_matches if m.status == "FN")

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class[c] = ClassMetrics(
                class_name=c,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Macro averages
        active_classes = [v for v in per_class.values() if (v.tp + v.fn) > 0 or v.fp > 0]
        if active_classes:
            macro_prec = sum(c.precision for c in active_classes) / len(active_classes)
            macro_rec = sum(c.recall for c in active_classes) / len(active_classes)
            macro_f1 = (2 * macro_prec * macro_rec) / (macro_prec + macro_rec) if (macro_prec + macro_rec) > 0 else 0.0
        else:
            macro_prec = 0.0
            macro_rec = 0.0
            macro_f1 = 0.0

        return EvaluationSummary(
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
            total_images=num_images,
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            macro_precision=macro_prec,
            macro_recall=macro_rec,
            macro_f1=macro_f1,
            per_class=per_class,
            matches=all_matches
        )
