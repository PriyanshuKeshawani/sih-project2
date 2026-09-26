from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class SonarTile:
    tile_id: str
    image: np.ndarray       # Always (tile_height, tile_width, 3)
    x_offset: int          # X offset in original image
    y_offset: int          # Y offset in original image
    valid_width: int       # Width of valid pixel data from original image
    valid_height: int      # Height of valid pixel data from original image
    tile_width: int        # 640
    tile_height: int       # 640
    pad_right: int         # Right padding added (if any)
    pad_bottom: int        # Bottom padding added (if any)


class SonarTiler:
    """
    Sliding-window tiling engine for side-scan sonar (SSS) imagery.
    Prevents aspect-ratio squashing on large or panoramic waterfall imagery
    by slicing into overlapping 640x640 windows with edge padding and precise
    coordinate remapping.
    """

    def __init__(self, tile_size: int = 640, overlap: float = 0.20):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive.")
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in range [0.0, 1.0).")

        self.tile_size = int(tile_size)
        self.overlap = float(overlap)
        self.stride = max(1, int(self.tile_size * (1.0 - self.overlap)))

    def _compute_intervals(self, total_dim: int) -> List[Tuple[int, int]]:
        """
        Computes starting and ending pixel coordinates along one axis.
        Ensures full coverage of the dimension without dropping edge pixels.
        """
        if total_dim <= self.tile_size:
            return [(0, total_dim)]

        starts = list(range(0, total_dim - self.tile_size + 1, self.stride))
        if starts[-1] + self.tile_size < total_dim:
            starts.append(total_dim - self.tile_size)

        return [(s, s + self.tile_size) for s in starts]

    def split_into_tiles(self, img: np.ndarray) -> List[SonarTile]:
        """
        Splits an arbitrary-sized sonar image into overlapping SonarTiles of size (tile_size, tile_size, 3).
        If dimensions are smaller than tile_size, padding is applied to the right and bottom.
        """
        orig_h, orig_w = img.shape[:2]
        x_intervals = self._compute_intervals(orig_w)
        y_intervals = self._compute_intervals(orig_h)

        tiles: List[SonarTile] = []
        tile_index = 0

        for y_start, y_end in y_intervals:
            for x_start, x_end in x_intervals:
                crop = img[y_start:y_end, x_start:x_end]
                crop_h, crop_w = crop.shape[:2]

                pad_right = max(0, self.tile_size - crop_w)
                pad_bottom = max(0, self.tile_size - crop_h)

                if pad_right > 0 or pad_bottom > 0:
                    tile_img = np.zeros((self.tile_size, self.tile_size, 3), dtype=crop.dtype)
                    tile_img[0:crop_h, 0:crop_w] = crop
                else:
                    tile_img = crop

                tile = SonarTile(
                    tile_id=f"tile_{tile_index:03d}",
                    image=tile_img,
                    x_offset=x_start,
                    y_offset=y_start,
                    valid_width=crop_w,
                    valid_height=crop_h,
                    tile_width=self.tile_size,
                    tile_height=self.tile_size,
                    pad_right=pad_right,
                    pad_bottom=pad_bottom
                )
                tiles.append(tile)
                tile_index += 1

        return tiles

    @staticmethod
    def remap_box_to_global(
        tile_bbox: List[float],
        tile: SonarTile,
        orig_w: int,
        orig_h: int
    ) -> List[int]:
        """
        Remaps a bounding box [x1, y1, x2, y2] from tile coordinates to global original image coordinates.
        Clips to valid tile data boundaries and the original image boundary.
        Returns [gx1, gy1, gx2, gy2] or None if box falls in padded region.
        """
        tx1, ty1, tx2, ty2 = tile_bbox

        # Discard boxes whose center falls outside valid non-padded area
        cx = (tx1 + tx2) / 2.0
        cy = (ty1 + ty2) / 2.0
        if cx >= tile.valid_width or cy >= tile.valid_height:
            return None

        # Clamp box to valid unpadded tile area
        tx1 = max(0.0, min(float(tile.valid_width), tx1))
        ty1 = max(0.0, min(float(tile.valid_height), ty1))
        tx2 = max(0.0, min(float(tile.valid_width), tx2))
        ty2 = max(0.0, min(float(tile.valid_height), ty2))

        # Check degenerate dimensions
        if (tx2 - tx1) < 2.0 or (ty2 - ty1) < 2.0:
            return None

        # Shift by tile offset
        gx1 = int(round(tx1 + tile.x_offset))
        gy1 = int(round(ty1 + tile.y_offset))
        gx2 = int(round(tx2 + tile.x_offset))
        gy2 = int(round(ty2 + tile.y_offset))

        # Clamp to global image dimensions
        gx1 = max(0, min(orig_w, gx1))
        gy1 = max(0, min(orig_h, gy1))
        gx2 = max(0, min(orig_w, gx2))
        gy2 = max(0, min(orig_h, gy2))

        if (gx2 - gx1) < 2 or (gy2 - gy1) < 2:
            return None

        return [gx1, gy1, gx2, gy2]

    @staticmethod
    def _compute_iom(box_a: List[int], box_b: List[int]) -> float:
        """Computes Intersection over Minimum Area (Containment ratio)."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        min_area = min(area_a, area_b)
        if min_area <= 0:
            return 0.0
        return inter_area / float(min_area)

    @staticmethod
    def _compute_iou(box_a: List[int], box_b: List[int]) -> float:
        """Computes standard Intersection over Union (IoU)."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / float(union_area)

    @staticmethod
    def class_aware_nms(
        boxes: List[List[int]],
        scores: List[float],
        class_ids: List[int],
        iou_threshold: float = 0.35,
        iom_threshold: float = 0.35
    ) -> List[int]:
        """
        Performs Non-Maximum Suppression independently for each class.
        Includes both standard IoU NMS and Containment (IoM) suppression to eliminate
        redundant sub-region detections of the same large structural target.
        """
        if len(boxes) == 0:
            return []

        np_boxes = np.array(boxes, dtype=np.float32)
        np_scores = np.array(scores, dtype=np.float32)
        np_classes = np.array(class_ids, dtype=np.int32)

        keep_indices: List[int] = []
        unique_classes = np.unique(np_classes)

        for c in unique_classes:
            c_mask = np.where(np_classes == c)[0]
            c_boxes = np_boxes[c_mask]
            c_scores = np_scores[c_mask]

            # Convert [x1, y1, x2, y2] to OpenCV format [x, y, w, h]
            cv_boxes = []
            for b in c_boxes:
                x1, y1, x2, y2 = b
                cv_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

            indices = cv2.dnn.NMSBoxes(
                bboxes=cv_boxes,
                scores=c_scores.tolist(),
                score_threshold=0.0,
                nms_threshold=float(iou_threshold)
            )

            c_kept = []
            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    c_kept.append(int(c_mask[idx]))

            # Containment (IoM) and dual-IoU suppression for same class
            # Eliminates duplicate nested or dual-scale boxes (e.g. tile vs full-frame pass)
            c_kept_sorted = sorted(c_kept, key=lambda i: float(scores[i]), reverse=True)
            suppressed = set()
            for i in range(len(c_kept_sorted)):
                idx_a = c_kept_sorted[i]
                if idx_a in suppressed:
                    continue
                for j in range(i + 1, len(c_kept_sorted)):
                    idx_b = c_kept_sorted[j]
                    if idx_b in suppressed:
                        continue
                    iom = SonarTiler._compute_iom(boxes[idx_a], boxes[idx_b])
                    iou = SonarTiler._compute_iou(boxes[idx_a], boxes[idx_b])
                    if iom >= iom_threshold or iou >= iou_threshold:
                        suppressed.add(idx_b)

            for idx in c_kept_sorted:
                if idx not in suppressed:
                    keep_indices.append(idx)

        # Sort all kept indices across all classes by confidence score descending
        keep_indices = sorted(keep_indices, key=lambda i: float(scores[i]), reverse=True)
        return keep_indices

    @staticmethod
    def cross_class_nms(
        boxes: List[List[int]],
        scores: List[float],
        iou_threshold: float = 0.45,
        iom_threshold: float = 0.50
    ) -> List[int]:
        """
        Suppresses redundant overlapping bounding boxes across different classes
        when two distinct classes claim the exact same physical anomaly (IoU or IoM >= threshold).
        Resolves classification conflict by strictly keeping the candidate with higher confidence.
        """
        if len(boxes) <= 1:
            return list(range(len(boxes)))

        cv_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b
            cv_boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

        indices = cv2.dnn.NMSBoxes(
            bboxes=cv_boxes,
            scores=[float(s) for s in scores],
            score_threshold=0.0,
            nms_threshold=float(iou_threshold)
        )

        kept = [int(i) for i in np.array(indices).flatten()] if len(indices) > 0 else []
        kept_sorted = sorted(kept, key=lambda i: float(scores[i]), reverse=True)

        # Further suppress any cross-class nested containment (IoM >= iom_threshold)
        final_kept = []
        suppressed = set()
        for i in range(len(kept_sorted)):
            idx_a = kept_sorted[i]
            if idx_a in suppressed:
                continue
            final_kept.append(idx_a)
            for j in range(i + 1, len(kept_sorted)):
                idx_b = kept_sorted[j]
                if idx_b in suppressed:
                    continue
                iom = SonarTiler._compute_iom(boxes[idx_a], boxes[idx_b])
                if iom >= iom_threshold:
                    suppressed.add(idx_b)

        return final_kept
