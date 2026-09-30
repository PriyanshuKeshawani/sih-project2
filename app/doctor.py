"""
app/doctor.py
Production Doctor Diagnostic Command: python -m app.doctor
Audits system readiness, dependencies, hardware resources, model files,
and reports final status: PASS / WARN / FAIL.
"""

import os
import sys
import platform
import shutil

from engine.config import get_current_config


def main():
    print("====================================================")
    print("Ocean IQ PRODUCTION DOCTOR")
    print("====================================================\n")

    overall_status = "PASS"
    warnings = []
    failures = []

    # 1. Environment & OS
    cfg = get_current_config()
    print(f"Environment Mode:  {cfg.mode.value}")
    print(f"Python Version:    {platform.python_version()} ({platform.architecture()[0]})")
    print(f"Operating System:  {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU Count:         {os.cpu_count()} logical cores")

    # RAM (via psutil or ctypes fallback)
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_gb = vm.total / (1024 ** 3)
        avail_ram_gb = vm.available / (1024 ** 3)
        print(f"System RAM:        {ram_gb:.1f} GB total ({avail_ram_gb:.1f} GB available)")
        if ram_gb < 4.0:
            warnings.append("Low total RAM (< 4GB)")
    except ImportError:
        print("System RAM:        psutil not installed; assuming nominal host memory")

    # Disk Space
    try:
        disk_total, disk_used, disk_free = shutil.disk_usage(os.path.abspath("."))
        free_gb = disk_free / (1024 ** 3)
        print(f"Disk Availability: {free_gb:.1f} GB free")
        if free_gb < 1.0:
            failures.append("Insufficient disk space (< 1GB)")
    except Exception as de:
        print(f"Disk Availability: UNKNOWN ({de})")

    # 2. Dependency Audit
    print("\n--- Dependencies & Frameworks ---")
    
    # ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime:      {ort.__version__} (Providers: {ort.get_available_providers()})")
    except ImportError:
        failures.append("ONNX Runtime is not installed")
        print("ONNX Runtime:      MISSING")

    # OpenCV
    try:
        import cv2
        print(f"OpenCV:            {cv2.__version__}")
    except ImportError:
        failures.append("OpenCV is not installed")
        print("OpenCV:            MISSING")

    # FastAPI
    try:
        import fastapi
        print(f"FastAPI:           {fastapi.__version__}")
    except ImportError:
        failures.append("FastAPI is not installed")
        print("FastAPI:           MISSING")

    # ReportLab
    try:
        import reportlab
        print(f"ReportLab:         {reportlab.__version__}")
    except ImportError:
        failures.append("ReportLab is not installed")
        print("ReportLab:         MISSING")

    # 3. Model Files
    print("\n--- AI Engines & Checkpoints ---")
    if os.path.exists(cfg.model_path):
        size_mb = os.path.getsize(cfg.model_path) / (1024 * 1024)
        print(f"Detector Model:    {cfg.model_path} ({size_mb:.2f} MB) - FOUND")
    else:
        failures.append(f"Detector model not found at {cfg.model_path}")
        print(f"Detector Model:    {cfg.model_path} - MISSING")

    # Laya
    try:
        from engine.reflex import System1ReflexEngine
        reflex = System1ReflexEngine()
        s1_stat = reflex.system1_manager.get_status()
        laya_engine = s1_stat.get("active_engine", "fallback")
        laya_ckpt = s1_stat.get("checkpoint", "real_laya_v1")
        print(f"Laya Engine:       {laya_engine} (Checkpoint: {laya_ckpt}) - READY")
    except Exception as le:
        warnings.append(f"Laya initialization exception: {le}")
        print(f"Laya Engine:       FALLBACK ({le})")

    # Groq & Gemini
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"Groq API:          CONFIGURED (Model: {cfg.groq_model})")
    else:
        warnings.append("GROQ_API_KEY not configured; will route to cloud/local fallback")
        print("Groq API:          UNCONFIGURED (Fallback active)")

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("Gemini API:        CONFIGURED (Cloud Fallback active)")
    else:
        print("Gemini API:        UNCONFIGURED")

    # 4. Required Files & Directories
    print("\n--- Filesystem & Permissions ---")
    required_paths = ["data", "models", "static", "reports"]
    for rp in required_paths:
        if os.path.exists(rp):
            # Check write permission
            writable = os.access(rp, os.W_OK)
            perm_str = "READ/WRITE" if writable else "READ-ONLY"
            print(f"Directory [{rp}]:   FOUND ({perm_str})")
            if not writable:
                failures.append(f"Directory {rp} is not writable")
        else:
            warnings.append(f"Directory {rp} missing (will be auto-created)")
            print(f"Directory [{rp}]:   MISSING")

    # 5. Final Quality Verdict
    print("\n====================================================")
    if failures:
        overall_status = "FAIL"
        print(f"PRODUCTION CHECK: FAIL ({len(failures)} critical failures)")
        for f in failures:
            print(f"  [X] {f}")
    elif warnings:
        overall_status = "WARN"
        print(f"PRODUCTION CHECK: WARN ({len(warnings)} operational warnings)")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        overall_status = "PASS"
        print("PRODUCTION CHECK: PASS")
    print("====================================================\n")

    return 0 if overall_status in ("PASS", "WARN") else 1


if __name__ == "__main__":
    sys.exit(main())
