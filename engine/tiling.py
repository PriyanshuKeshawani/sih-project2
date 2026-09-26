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
    def class_aware_nms(
        boxes: List[List[int]],
        scores: List[float],
        class_ids: List[int],
        iou_threshold: float = 0.45
    ) -> List[int]:
        """
        Performs Non-Maximum Suppression independently for each class.
        Guarantees that distinct classes (e.g. ghost_net and pipeline) never suppress each other.
        Returns list of kept original indices sorted by confidence descending.
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

            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    keep_indices.append(int(c_mask[idx]))

        # Sort all kept indices across all classes by confidence score descending
        keep_indices = sorted(keep_indices, key=lambda i: float(scores[i]), reverse=True)
        return keep_indices
