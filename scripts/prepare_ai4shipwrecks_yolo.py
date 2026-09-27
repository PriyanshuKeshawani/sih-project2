import os
import cv2
import numpy as np
import yaml
from pathlib import Path

def prepare_yolo_dataset():
    base_dir = Path("AI4Shipwrecks")
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    test_img_dir = base_dir / "test" / "images"

    out_dir = base_dir / "yolo_dataset"
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    tile_size = 640
    overlap = 0.20
    step = int(tile_size * (1 - overlap))

    label_files = sorted([f for f in os.listdir(train_lbl_dir) if f.endswith(".png")])
    print(f"Total training masks found: {len(label_files)}")

    # Split: 80% train, 20% val
    np.random.seed(42)
    shuffled = np.random.permutation(label_files)
    val_count = max(5, int(len(label_files) * 0.20))
    val_files = set(shuffled[:val_count])

    stats = {"train_tiles": 0, "val_tiles": 0, "positive_tiles": 0, "negative_tiles": 0}

    for idx, lbl_name in enumerate(label_files):
        img_name = lbl_name.replace(".png", ".jpg")
        img_path = train_img_dir / img_name
        if not img_path.exists():
            img_path = train_img_dir / lbl_name
        if not img_path.exists():
            continue

        lbl_path = train_lbl_dir / lbl_name
        mask = cv2.imread(str(lbl_path), 0)
        img = cv2.imread(str(img_path))
        if mask is None or img is None:
            continue

        h, w = mask.shape[:2]
        split = "val" if lbl_name in val_files else "train"

        # Find all shipwreck contours in this mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        global_boxes = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw >= 8 and bh >= 8:  # filter tiny speckles
                global_boxes.append((bx, by, bx + bw, by + bh))

        # If image has targets, extract positive tiles around objects + some background context
        has_positives = len(global_boxes) > 0

        # Slide over waterfall
        y_max = max(1, h - tile_size + 1)
        x_max = max(1, w - tile_size + 1)

        for y in range(0, h, step):
            for x in range(0, w, step):
                tx1 = min(x, w - tile_size)
                ty1 = min(y, h - tile_size)
                tx2 = tx1 + tile_size
                ty2 = ty1 + tile_size

                # Find boxes that fall within this tile
                tile_boxes = []
                for gx1, gy1, gx2, gy2 in global_boxes:
                    ix1 = max(tx1, gx1)
                    iy1 = max(ty1, gy1)
                    ix2 = min(tx2, gx2)
                    iy2 = min(ty2, gy2)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    box_area = (gx2 - gx1) * (gy2 - gy1)
                    inter_area = iw * ih

                    # Keep box if significant portion is inside tile
                    if inter_area > 0 and (inter_area / float(box_area) > 0.30 or inter_area > 400):
                        # Convert to normalized YOLO [class x_center y_center width height]
                        # Class 0: shipwreck
                        cx = (ix1 + ix2) / 2.0 - tx1
                        cy = (iy1 + iy2) / 2.0 - ty1
                        tile_boxes.append((0, cx / tile_size, cy / tile_size, iw / tile_size, ih / tile_size))

                # Background sampling: keep all positive tiles, but sample negatives to avoid imbalance
                is_positive = len(tile_boxes) > 0
                if not is_positive:
                    # Randomly sample 10% of negative seabed tiles
                    if np.random.rand() > 0.10:
                        continue

                tile_img = img[ty1:ty2, tx1:tx2]
                if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                    tile_img = cv2.resize(tile_img, (tile_size, tile_size))

                tile_id = f"{img_path.stem}_y{ty1}_x{tx1}"
                out_img_path = out_dir / "images" / split / f"{tile_id}.jpg"
                out_lbl_path = out_dir / "labels" / split / f"{tile_id}.txt"

                cv2.imwrite(str(out_img_path), tile_img)
                with open(out_lbl_path, "w") as lf:
                    for b in tile_boxes:
                        lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

                if split == "train":
                    stats["train_tiles"] += 1
                else:
                    stats["val_tiles"] += 1

                if is_positive:
                    stats["positive_tiles"] += 1
                else:
                    stats["negative_tiles"] += 1

        if (idx + 1) % 25 == 0 or idx == len(label_files) - 1:
            print(f"Processed {idx + 1}/{len(label_files)} waterfalls -> Generated {stats['train_tiles']} train, {stats['val_tiles']} val tiles")

    # Write data.yaml for YOLO
    yaml_content = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "shipwreck"
        }
    }
    yaml_path = out_dir / "shipwrecks.yaml"
    with open(yaml_path, "w") as yf:
        yaml.dump(yaml_content, yf, default_flow_style=False)

    print("\nDataset preparation complete!")
    print(f"Stats: {stats}")
    print(f"Config saved to: {yaml_path}")
    return stats

if __name__ == "__main__":
    prepare_yolo_dataset()
