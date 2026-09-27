import os
import cv2
import numpy as np
from pathlib import Path

# Fix Windows onnxruntime pybind webgpu mismatch
try:
    import onnxruntime as ort
    from onnxruntime.capi import _pybind_state
    if not hasattr(_pybind_state.InferenceSession, "is_webgpu_graph_capture_enabled"):
        _pybind_state.InferenceSession.is_webgpu_graph_capture_enabled = lambda self: False
except Exception:
    pass

from ultralytics import YOLO

def evaluate_on_test_set():
    print("=" * 60)
    print("EVALUATING MODEL ON AI4SHIPWRECKS TEST DATASET")
    print("=" * 60)

    test_img_dir = Path("AI4Shipwrecks/test/images")
    if not test_img_dir.exists():
        print(f"[ERROR] Test directory {test_img_dir} does not exist.")
        return

    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith((".jpg", ".png"))])
    print(f"Total test waterfalls found: {len(test_files)}")

    weights_path = Path("runs/detect/runs/ai4shipwrecks/rtx4050_train/weights/best.pt")
    if not weights_path.exists():
        weights_path = Path("models/best_shipwreck_detector.onnx")

    print(f"Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    results_summary = {"total_tested": 0, "detected_shipwrecks": 0, "high_conf_detections": 0}

    out_preview_dir = Path("reports/ai4shipwrecks_test_eval")
    out_preview_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate on a diverse sample of 15 test waterfalls
    sample_test_files = test_files[:15]

    for fname in sample_test_files:
        fpath = test_img_dir / fname
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        results = model.predict(source=img, conf=0.25, imgsz=640, verbose=False)
        det_count = len(results[0].boxes)
        results_summary["total_tested"] += 1
        results_summary["detected_shipwrecks"] += det_count

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf >= 0.50:
                results_summary["high_conf_detections"] += 1

        # Save annotated preview
        annotated_frame = results[0].plot()
        preview_path = out_preview_dir / f"eval_{fname}"
        cv2.imwrite(str(preview_path), annotated_frame)

    print("\nEvaluation Results on Test Subset:")
    print(f"  Waterfalls Evaluated: {results_summary['total_tested']}")
    print(f"  Total Shipwreck Detections: {results_summary['detected_shipwrecks']}")
    print(f"  High Confidence (>= 50%): {results_summary['high_conf_detections']}")
    print(f"  Preview images saved to: {out_preview_dir}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_on_test_set()
