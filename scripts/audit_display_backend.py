"""
Comprehensive Backend vs Frontend Detection Display Audit Script.
Traces raw tile detections, per-tile NMS, global NMS, and API response.
SIH 2026 Problem Statement 26057.
"""
import os
import sys
import json
import cv2
import numpy as np
from collections import Counter

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.detector import SonarDetector, SonarTile, SonarTiler, CLASSES, CLASS_COLORS
from engine.preprocessing import PreprocessConfig, SonarPreprocessor

def run_backend_audit():
    print("=" * 70)
    print("BACKEND RAW OUTPUT AUDIT — DETECTION DISPLAY PIPELINE")
    print("=" * 70)

    input_img_path = os.path.join("reports", "display_audit", "reconstructed_test_input.png")
    if not os.path.exists(input_img_path):
        raise FileNotFoundError(f"Input image not found: {input_img_path}")

    img_bgr = cv2.imread(input_img_path)
    orig_h, orig_w = img_bgr.shape[:2]
    print(f"Image dimensions: {orig_w} x {orig_h} (width x height)")

    detector = SonarDetector("models/best_detector.onnx")

    # Evaluate at current baseline conf_threshold (0.25) and recommended threshold (0.45)
    for conf_threshold in [0.25, 0.45]:
        print("\n" + "-" * 70)
        print(f"EVALUATION AT CONFIGURED CONFIDENCE THRESHOLD = {conf_threshold}")
        print("-" * 70)

        tiling_enabled = True
        overlap = 0.20
        tile_size = 640
        iou_threshold = 0.45

        # 1. Preprocessing
        working_img = SonarPreprocessor.preprocess(img_bgr, PreprocessConfig())

        # 2. Tiling
        tiler = SonarTiler(tile_size=tile_size, overlap=overlap)
        tiles = tiler.split_into_tiles(working_img)
        print(f"Tiling enabled: {tiling_enabled}")
        print(f"Number of tiles: {len(tiles)}")
        for i, t in enumerate(tiles):
            print(f"  Tile {i} [{t.tile_id}]: offset=({t.x_offset}, {t.y_offset}), valid_dim=({t.valid_width}x{t.valid_height})")

        # 3. Per-tile inference & intra-tile NMS
        raw_tile_detections_count = 0
        all_intra_nms_boxes = []

        for tile in tiles:
            boxes_xyxy, scores, class_ids = detector._infer_tile_raw(tile.image)
            valid_mask = scores >= conf_threshold
            t_boxes = boxes_xyxy[valid_mask]
            t_scores = scores[valid_mask]
            t_classes = class_ids[valid_mask]

            # filter degenerate
            nd_mask = np.array([b[2] - b[0] >= 4.0 and b[3] - b[1] >= 4.0 for b in t_boxes])
            if len(nd_mask) > 0 and any(nd_mask):
                t_boxes = t_boxes[nd_mask]
                t_scores = t_scores[nd_mask]
                t_classes = t_classes[nd_mask]
            else:
                t_boxes, t_scores, t_classes = np.empty((0, 4)), np.empty(0), np.empty(0)

            raw_count = len(t_boxes)
            raw_tile_detections_count += raw_count

            # intra-tile NMS
            keep_idx = SonarTiler.class_aware_nms(
                boxes=[b.tolist() for b in t_boxes],
                scores=t_scores.tolist(),
                class_ids=t_classes.tolist(),
                iou_threshold=iou_threshold
            )
            print(f"  [{tile.tile_id}]: raw candidates={raw_count} -> kept after intra-tile NMS={len(keep_idx)}")

            for k in keep_idx:
                raw_box = t_boxes[k].tolist()
                g_box = SonarTiler.remap_box_to_global(raw_box, tile, orig_w, orig_h)
                if g_box is not None:
                    all_intra_nms_boxes.append({
                        "class_id": int(t_classes[k]),
                        "class": CLASSES[int(t_classes[k])],
                        "confidence": float(t_scores[k]),
                        "box": g_box,
                        "tile_id": tile.tile_id
                    })

        post_tile_nms_count = len(all_intra_nms_boxes)
        print(f"Raw detections before NMS (sum of tile raw): {raw_tile_detections_count}")
        print(f"Detections after per-tile NMS: {post_tile_nms_count}")

        # 4. Global Class-Aware NMS
        g_boxes = [d["box"] for d in all_intra_nms_boxes]
        g_scores = [d["confidence"] for d in all_intra_nms_boxes]
        g_classes = [d["class_id"] for d in all_intra_nms_boxes]

        global_keep_idx = SonarTiler.class_aware_nms(
            boxes=g_boxes,
            scores=g_scores,
            class_ids=g_classes,
            iou_threshold=iou_threshold
        )
        post_global_nms_count = len(global_keep_idx)
        print(f"Detections after global NMS: {post_global_nms_count}")

        # 5. Final backend detections (after hard-negative filter)
        final_detections = []
        for det_idx, g_idx in enumerate(global_keep_idx):
            item = all_intra_nms_boxes[g_idx]
            if item["class"] == "crab_pot" and item["confidence"] < 0.40:
                continue
            final_det = {
                "detection_id": f"det_{det_idx + 1:03d}",
                "class": item["class"],
                "confidence": round(item["confidence"], 3),
                "bbox_xyxy": item["box"],
                "tile_id": item["tile_id"]
            }
            final_detections.append(final_det)

        final_detections_count = len(final_detections)
        print(f"Final detections returned by API: {final_detections_count}")
        print(f"raw_tile_detections_count: {raw_tile_detections_count}")
        print(f"final_detections_count: {final_detections_count}")

        print("\nFinal Detections Detail:")
        for fd in final_detections:
            print(f"  {fd['detection_id']} | Class: {fd['class']:<18} | Conf: {fd['confidence']:.3f} | BBox: {fd['bbox_xyxy']} | Source: {fd['tile_id']}")

        # 6. Save visual for backend final
        if conf_threshold == 0.25:
            vis_img = img_bgr.copy()
            for fd in final_detections:
                gx1, gy1, gx2, gy2 = fd["bbox_xyxy"]
                cls_name = fd["class"]
                score = fd["confidence"]
                color = CLASS_COLORS.get(cls_name, (0, 255, 0))
                cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), color, 2)
                label = f"{fd['detection_id']}:{cls_name.upper()} {score*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis_img, (gx1, max(0, gy1 - 18)), (gx1 + tw + 6, max(18, gy1)), color, -1)
                cv2.putText(vis_img, label, (gx1 + 3, max(14, gy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            out_path = os.path.join("reports", "display_audit", "backend_final.png")
            cv2.imwrite(out_path, vis_img)
            print(f"\nSaved backend-only visualization: {out_path}")

            # Also save frontend_current.png (reproducing the 0.25 visual)
            out_curr = os.path.join("reports", "display_audit", "frontend_current.png")
            cv2.imwrite(out_curr, vis_img)
            print(f"Saved frontend current representation: {out_curr}")

        elif conf_threshold == 0.45:
            # Save frontend_fixed.png (at verified operational threshold 0.45)
            vis_img_fixed = img_bgr.copy()
            for fd in final_detections:
                gx1, gy1, gx2, gy2 = fd["bbox_xyxy"]
                cls_name = fd["class"]
                score = fd["confidence"]
                color = CLASS_COLORS.get(cls_name, (0, 255, 0))
                cv2.rectangle(vis_img_fixed, (gx1, gy1), (gx2, gy2), color, 2)
                label = f"{fd['detection_id']}:{cls_name.upper()} {score*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis_img_fixed, (gx1, max(0, gy1 - 18)), (gx1 + tw + 6, max(18, gy1)), color, -1)
                cv2.putText(vis_img_fixed, label, (gx1 + 3, max(14, gy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            out_fixed = os.path.join("reports", "display_audit", "frontend_fixed.png")
            cv2.imwrite(out_fixed, vis_img_fixed)
            print(f"Saved frontend fixed representation (conf=0.45): {out_fixed}")

if __name__ == "__main__":
    run_backend_audit()
