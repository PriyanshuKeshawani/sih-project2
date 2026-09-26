import os
import time
import cv2
import json
from engine.detector import SonarDetector
from engine.preprocessing import PreprocessConfig, SonarPreprocessor
from engine.tiling import SonarTiler

def run_benchmark():
    model_path = os.path.join("models", "best_detector.onnx")
    detector = SonarDetector(model_path)
    reports_dir = os.path.join("reports", "phase2")
    os.makedirs(reports_dir, exist_ok=True)

    test_cases = [
        ("Background", "data/samples/bg_1693569243.750_x2500.jpg", 0.25),
        ("Ghost Net", "data/samples/synth_ghost_net_00001.png", 0.25),
        ("Shipwreck", "data/samples/wreckR_ship-081_png.rf.6cf386b75ddb8ead86c0453021279296.jpg", 0.25),
        ("Submarine Pipeline", "data/samples/pipe_1693569383.780_x3500.jpg", 0.25),
        ("Mine Cylinder (1024x1024 Large)", "data/samples/mine_0001_2015.jpg", 0.30)
    ]

    results = []

    print("=" * 80)
    print("PHASE 2 REAL BENCHMARK — MEASURED SYSTEM METRICS")
    print("=" * 80)

    for label, path, conf_thresh in test_cases:
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping {path} (not found)")
            continue
        h, w = img.shape[:2]

        # 1. Preprocessing Stage
        t0 = time.perf_counter()
        prep_cfg = PreprocessConfig(enabled=True, use_clahe=True, use_bilateral=True)
        prep_img = SonarPreprocessor.preprocess(img, prep_cfg)
        t_prep = (time.perf_counter() - t0) * 1000

        # 2. Tiling Stage
        t1 = time.perf_counter()
        tiler = SonarTiler(tile_size=640, overlap=0.20)
        tiles = tiler.split_into_tiles(prep_img)
        t_tile = (time.perf_counter() - t1) * 1000

        # 3. ONNX Tile Inference Stage
        t2 = time.perf_counter()
        tile_raw_preds = []
        for t in tiles:
            boxes_xyxy, scores, class_ids = detector._infer_tile_raw(t.image)
            tile_raw_preds.append((t, boxes_xyxy, scores, class_ids))
        t_inf = (time.perf_counter() - t2) * 1000

        # 4. NMS & Coordinate Remap Stage
        t3 = time.perf_counter()
        detections, annotated = detector.detect(
            img,
            conf_threshold=conf_thresh,
            iou_threshold=0.45,
            tiling=True,
            preprocess=True,
            preprocess_config=prep_cfg
        )
        t_nms = (time.perf_counter() - t3) * 1000

        t_total = t_prep + t_tile + t_inf + t_nms

        # Save Visual Verification Output
        clean_name = os.path.basename(path).replace('.', '_')
        out_vis_path = os.path.join(reports_dir, f"verified_{clean_name}.jpg")
        SonarDetector.save_annotated_result(img, detections, out_vis_path, draw_tiles=True, tiles=tiles)

        item = {
            "label": label,
            "file": path,
            "dimensions": f"{w}x{h}",
            "tile_count": len(tiles),
            "preprocess_ms": round(t_prep, 2),
            "inference_ms": round(t_inf, 2),
            "nms_postprocess_ms": round(t_nms, 2),
            "total_ms": round(t_total, 2),
            "detection_count": len(detections),
            "visual_artifact": out_vis_path,
            "detections": detections
        }
        results.append(item)

        print(f"\nIMAGE: {label} ({os.path.basename(path)})")
        print(f"SIZE: {w}x{h} px | TILES: {len(tiles)}")
        print(f"PREPROCESS: {t_prep:.2f} ms")
        print(f"INFERENCE: {t_inf:.2f} ms ({t_inf/len(tiles):.2f} ms/tile)")
        print(f"NMS & REMAP: {t_nms:.2f} ms")
        print(f"TOTAL: {t_total:.2f} ms")
        print(f"FINAL DETECTIONS ({len(detections)}):")
        for d in detections:
            print(f"  -> {d['class'].upper()} (conf: {d['confidence']*100:.1f}%) box: {d['bbox_xyxy']} from {d['tile_id']}")

    # Save JSON metrics report
    metrics_path = os.path.join(reports_dir, "phase2_benchmark_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured metrics to {metrics_path}")

if __name__ == '__main__':
    run_benchmark()
