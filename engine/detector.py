import os
import cv2
import numpy as np
import onnxruntime as ort
import base64
from typing import List, Dict, Any, Tuple, Optional

from engine.preprocessing import PreprocessConfig, SonarPreprocessor
from engine.tiling import SonarTile, SonarTiler

CLASSES = ['crab_pot', 'submarine_pipeline', 'shipwreck', 'ghost_net', 'mine_cylinder']
CLASS_COLORS = {
    'ghost_net': (0, 0, 255),           # Red (High ecological danger)
    'shipwreck': (255, 165, 0),         # Orange (Navigation hazard)
    'submarine_pipeline': (255, 255, 0),# Yellow (Critical infra)
    'mine_cylinder': (255, 0, 255),     # Magenta (Munition/drum hazard)
    'crab_pot': (128, 128, 128)         # Gray (Hard negative)
}


class SonarDetector:
    """
    Production-grade Side-Scan Sonar (SSS) detector supporting:
    - Preprocessing: Optional CLAHE contrast equalization and bilateral speckle filtering.
    - Sliding-Window Tiling: Preservation of aspect ratio and small targets across large sonar waterfall swaths.
    - Class-Aware NMS: Independent suppression per class (both intra-tile and global cross-tile).
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Reuse single persistent ONNX session on CPU
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.last_debug_info: Dict[str, Any] = {}

    def _infer_tile_raw(self, tile_img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a single 640x640 tile through the ONNX session.
        Returns:
            boxes_xyxy: (N, 4) in tile coordinates [x1, y1, x2, y2]
            scores: (N,) maximum class confidence
            class_ids: (N,) class index
        """
        # Ensure 3-channel RGB and normalization to [0, 1]
        img_rgb = cv2.cvtColor(tile_img_bgr, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]
        preds = np.transpose(outputs[0])  # Shape: (8400, 9)

        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        class_scores = preds[:, 4:]  # 5 classes

        max_scores = np.max(class_scores, axis=1)
        max_classes = np.argmax(class_scores, axis=1)

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes_xyxy = np.column_stack((x1, y1, x2, y2))
        return boxes_xyxy, max_scores, max_classes

    def detect(
        self,
        img: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tiling: bool = True,
        preprocess: bool = True,
        tile_size: int = 640,
        overlap: float = 0.20,
        preprocess_config: Optional[PreprocessConfig] = None,
        draw_tiles: bool = False,
        return_debug: bool = False
    ) -> Any:
        """
        End-to-end detection pipeline:
        1. Preprocessing (optional CLAHE & bilateral filtering).
        2. Large-image tiling into overlapping 640x640 windows.
        3. ONNX inference per tile.
        4. Tile-level class-aware NMS.
        5. Global coordinate remapping back to original image space.
        6. Global class-aware NMS to eliminate duplicates from tile overlaps.
        7. Returns structured detections and annotated image (and optional debug dict).
        """
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or invalid.")

        orig_bgr = SonarPreprocessor.ensure_3channel_bgr(img)
        orig_h, orig_w = orig_bgr.shape[:2]

        # 1. Acoustic Preprocessing
        if preprocess:
            cfg = preprocess_config if preprocess_config is not None else PreprocessConfig()
            working_img = SonarPreprocessor.preprocess(orig_bgr, cfg)
        else:
            working_img = orig_bgr

        # 2. Tiling or Direct Inference
        if tiling:
            tiler = SonarTiler(tile_size=tile_size, overlap=overlap)
            tiles = tiler.split_into_tiles(working_img)
        else:
            # Fallback: direct resize (not recommended for large imagery)
            direct_resized = cv2.resize(working_img, (tile_size, tile_size))
            tiles = [SonarTile(
                tile_id="tile_direct",
                image=direct_resized,
                x_offset=0,
                y_offset=0,
                valid_width=tile_size,
                valid_height=tile_size,
                tile_width=tile_size,
                tile_height=tile_size,
                pad_right=0,
                pad_bottom=0
            )]

        all_global_boxes: List[List[int]] = []
        all_scores: List[float] = []
        all_class_ids: List[int] = []
        all_tile_ids: List[str] = []
        raw_tile_detections_count: int = 0

        # 3. Process Each Tile
        for tile in tiles:
            boxes_xyxy, scores, class_ids = self._infer_tile_raw(tile.image)

            # Confidence filtering
            valid_mask = scores >= conf_threshold
            t_boxes = boxes_xyxy[valid_mask]
            t_scores = scores[valid_mask]
            t_classes = class_ids[valid_mask]

            if len(t_boxes) == 0:
                continue

            # Filter degenerate / flat thin boxes inside tile (e.g. razor lines)
            non_degenerate = []
            for b in t_boxes:
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                # Bounding box must have sensible minimum extent
                if bw >= 4.0 and bh >= 4.0:
                    non_degenerate.append(True)
                else:
                    non_degenerate.append(False)

            if not any(non_degenerate):
                continue

            nd_mask = np.array(non_degenerate)
            t_boxes = t_boxes[nd_mask]
            t_scores = t_scores[nd_mask]
            t_classes = t_classes[nd_mask]
            raw_tile_detections_count += len(t_boxes)

            # Intra-tile class-aware NMS
            tile_keep_idx = SonarTiler.class_aware_nms(
                boxes=[b.tolist() for b in t_boxes],
                scores=t_scores.tolist(),
                class_ids=t_classes.tolist(),
                iou_threshold=iou_threshold
            )

            # Remap kept boxes to global image space
            for k_idx in tile_keep_idx:
                raw_box = t_boxes[k_idx].tolist()
                global_box = SonarTiler.remap_box_to_global(raw_box, tile, orig_w, orig_h)
                if global_box is not None:
                    all_global_boxes.append(global_box)
                    all_scores.append(float(t_scores[k_idx]))
                    all_class_ids.append(int(t_classes[k_idx]))
                    all_tile_ids.append(tile.tile_id)

        # 4. Global Class-Aware NMS across tiles (merge overlap duplicates)
        final_detections: List[Dict[str, Any]] = []
        annotated_img = orig_bgr.copy()

        # Optional debug tile grid visualization (OFF by default)
        if draw_tiles and tiles:
            for t in tiles:
                tx1, ty1 = t.x_offset, t.y_offset
                tx2, ty2 = tx1 + t.valid_width, ty1 + t.valid_height
                cv2.rectangle(annotated_img, (tx1, ty1), (tx2, ty2), (50, 150, 250), 1)

        if len(all_global_boxes) > 0:
            global_keep_idx = SonarTiler.class_aware_nms(
                boxes=all_global_boxes,
                scores=all_scores,
                class_ids=all_class_ids,
                iou_threshold=iou_threshold
            )

            for g_idx in global_keep_idx:
                gx1, gy1, gx2, gy2 = all_global_boxes[g_idx]
                cls_id = all_class_ids[g_idx]
                cls_name = CLASSES[cls_id]
                score = all_scores[g_idx]
                tile_source = all_tile_ids[g_idx]

                # Hard negative filter for weak crab_pot
                if cls_name == 'crab_pot' and score < 0.40:
                    continue

                bw = gx2 - gx1
                bh = gy2 - gy1

                display_label = "SHIPWRECK-CLASS CONTACT" if cls_name == "shipwreck" else cls_name.upper()
                taxonomy_status = "CLOSED_SET_SURROGATE" if cls_name == "shipwreck" else "TAXONOMY_MATCH"

                # Structured detection item with unique traceable ID and MVP-safe taxonomy status
                detection_item = {
                    "detection_id": f"det_{len(final_detections) + 1:03d}",
                    "class_id": cls_id,
                    "class": cls_name,
                    "display_label": display_label,
                    "taxonomy_status": taxonomy_status,
                    "confidence": round(score, 3),
                    "box": {"x": gx1, "y": gy1, "w": bw, "h": bh}, # Backward compatible
                    "bbox_xyxy": [gx1, gy1, gx2, gy2],
                    "tile_id": tile_source,
                    "source_width": orig_w,
                    "source_height": orig_h
                }
                final_detections.append(detection_item)

                # Draw annotation
                color = CLASS_COLORS.get(cls_name, (0, 255, 0))
                cv2.rectangle(annotated_img, (gx1, gy1), (gx2, gy2), color, 2)
                short_tag = {
                    'shipwreck': 'WRECK',
                    'ghost_net': 'GHOST NET',
                    'mine_cylinder': 'MINE',
                    'submarine_pipeline': 'PIPE',
                    'crab_pot': 'CRAB POT'
                }.get(cls_name, cls_name.upper())
                label = f"{short_tag} {score*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                tag_y1 = max(0, gy1 - 18) if gy1 > 20 else gy2
                tag_y2 = tag_y1 + 18
                cv2.rectangle(annotated_img, (gx1, tag_y1), (gx1 + tw + 6, tag_y2), color, -1)
                cv2.putText(annotated_img, label, (gx1 + 3, tag_y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        self.last_debug_info = {
            "tile_count": len(tiles) if tiling else 1,
            "raw_tile_detections_count": raw_tile_detections_count,
            "raw_tile_detection_count": raw_tile_detections_count,
            "post_tile_nms_count": len(all_global_boxes),
            "final_detections_count": len(final_detections),
            "final_detection_count": len(final_detections),
            "confidence_threshold": conf_threshold,
            "tile_grid_enabled": draw_tiles
        }

        if return_debug:
            return final_detections, annotated_img, self.last_debug_info
        return final_detections, annotated_img

    @staticmethod
    def encode_image(img_bgr: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', img_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    @staticmethod
    def save_annotated_result(
        img_bgr: np.ndarray,
        detections: List[Dict[str, Any]],
        output_path: str,
        draw_tiles: bool = False,
        tiles: Optional[List[SonarTile]] = None
    ) -> str:
        """
        Saves annotated image with optional tile boundaries for visual verification.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        vis_img = img_bgr.copy()

        if draw_tiles and tiles:
            for t in tiles:
                tx1, ty1 = t.x_offset, t.y_offset
                tx2, ty2 = tx1 + t.valid_width, ty1 + t.valid_height
                cv2.rectangle(vis_img, (tx1, ty1), (tx2, ty2), (50, 150, 250), 1)

        for d in detections:
            gx1, gy1, gx2, gy2 = d['bbox_xyxy']
            cls_name = d['class']
            conf = d['confidence']
            color = CLASS_COLORS.get(cls_name, (0, 255, 0))
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), color, 2)
            cv2.putText(vis_img, f"{cls_name}: {conf:.2f}", (gx1, max(15, gy1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imwrite(output_path, vis_img)
        return output_path
