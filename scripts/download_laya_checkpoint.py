import os
import sys
import shutil
import time
import requests

SNAPSHOT_DIR = os.path.abspath("models/laya/hf_cache/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
LOCAL_MODEL_DIR = os.path.abspath("models/laya/checkpoint")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Copy encoder, tokenizer, and config to LOCAL_MODEL_DIR
for item in ["encoder", "tokenizer", "rl_agent_config.json"]:
    src = os.path.join(SNAPSHOT_DIR, item)
    dst = os.path.join(LOCAL_MODEL_DIR, item)
    if os.path.exists(src) and not os.path.exists(dst):
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        print(f"Copied {item} to {LOCAL_MODEL_DIR}")

target_file = os.path.join(LOCAL_MODEL_DIR, "model.safetensors")
temp_file = target_file + ".downloading"

if os.path.exists(target_file) and os.path.getsize(target_file) == 842609210:
    print(f"Model weights already exist at {target_file} (842,609,210 bytes).")
    sys.exit(0)

existing_size = os.path.getsize(temp_file) if os.path.exists(temp_file) else 0
print(f"Resuming download from byte {existing_size} to {temp_file}...")

url = "https://huggingface.co/convaiinnovations/laya/resolve/main/model.safetensors"
headers = {}
if existing_size > 0:
    headers["Range"] = f"bytes={existing_size}-"

t0 = time.time()
with requests.get(url, headers=headers, stream=True, timeout=30) as r:
    if r.status_code not in (200, 206):
        print(f"Error: HTTP {r.status_code} - {r.text}")
        sys.exit(1)
    
    total_expected = 842609210
    mode = "ab" if existing_size > 0 else "wb"
    with open(temp_file, mode) as f:
        downloaded = existing_size
        last_report = time.time()
        for chunk in r.iter_content(chunk_size=1024 * 1024): # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_report > 5:
                    pct = (downloaded / total_expected) * 100
                    speed_mb = (downloaded - existing_size) / (time.time() - t0) / (1024 * 1024)
                    print(f"Downloaded {downloaded / (1024 * 1024):.1f} / {total_expected / (1024 * 1024):.1f} MB ({pct:.1f}%) @ {speed_mb:.2f} MB/s", flush=True)
                    last_report = time.time()

if os.path.getsize(temp_file) == total_expected:
    if os.path.exists(target_file):
        os.remove(target_file)
    os.rename(temp_file, target_file)
    print(f"Successfully downloaded model.safetensors to {target_file}")
else:
    print(f"Download incomplete: {os.path.getsize(temp_file)} / {total_expected}")
