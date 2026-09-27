import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=" * 60)
    print("AI4SHIPWRECKS GPU TRAINING PIPELINE (NVIDIA RTX 4050)")
    print("=" * 60)

    # 1. Hardware Verification
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_ok}")
    if not cuda_ok:
        print("[ERROR] CUDA is not available. Please ensure PyTorch with CUDA is installed.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Active GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")

    # 2. Dataset Configuration
    dataset_yaml = Path("AI4Shipwrecks/yolo_dataset/shipwrecks.yaml").resolve()
    if not dataset_yaml.exists():
        print(f"[ERROR] Dataset configuration not found at {dataset_yaml}")
        sys.exit(1)

    print(f"Dataset YAML: {dataset_yaml}")

    # 3. Model Initialization (YOLOv8 Small - optimized for speed & high mAP)
    # Using yolov8s.pt pre-trained weights for transfer learning
    model = YOLO("yolov8s.pt")

    # 4. Training on RTX 4050 (Batch=8 to safely fit 6GB VRAM without OOM)
    print("\nStarting GPU Training on NVIDIA RTX 4050...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,           # RTX 4050 GPU
        amp=True,           # FP16 mixed precision for max speed & low VRAM
        workers=2,          # Prevent Windows multiprocessing bottlenecks
        patience=10,        # Early stopping if no improvement
        project="runs/ai4shipwrecks",
        name="rtx4050_train",
        exist_ok=True,
        verbose=True
    )

    print("\nTraining completed successfully!")

    # 5. Validation on Hold-out Val Set
    print("\nRunning Validation Metrics...")
    val_metrics = model.val(data=str(dataset_yaml), split="val", device=0)
    print(f"mAP50: {val_metrics.box.map50:.4f}")
    print(f"mAP50-95: {val_metrics.box.map:.4f}")

    # 6. Export to ONNX for High-Speed Deployment
    best_pt = Path("runs/ai4shipwrecks/rtx4050_train/weights/best.pt")
    if best_pt.exists():
        print(f"\nExporting {best_pt} to ONNX format...")
        best_model = YOLO(str(best_pt))
        onnx_path = best_model.export(format="onnx", imgsz=640, dynamic=False)
        print(f"Exported ONNX model to: {onnx_path}")

        # Copy to models directory
        out_onnx = Path("models/best_shipwreck_detector.onnx")
        import shutil
        shutil.copy(onnx_path, out_onnx)
        print(f"Copied production model to: {out_onnx}")

    print("\n=" * 60)
    print("GPU TRAINING & ONNX EXPORT COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
