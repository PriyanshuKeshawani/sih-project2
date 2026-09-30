"""Post-process Pocket TTS masters into final deliverables.

For each audio/<version>_<voice>/OceanIQ_<VER>_master_<voice>.wav it writes:
  FINAL_OceanIQ_<VER>_48k.wav   48 kHz, mono, 24-bit PCM, -16.7 LUFS, -1.5 dBTP
  FINAL_OceanIQ_<VER>.mp3       192 kbps stereo, same loudness

Run:
    python scripts/postprocess_narration.py --voice alba
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIO = HERE.parent / "audio"

TARGET_LUFS = -16.7
TARGET_TP = -1.5


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit("ffmpeg failed: " + " ".join(cmd))
    return p.stdout + p.stderr


def measure(src):
    out = run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
        "-af", "ebur128=peak=true", "-f", "null", "-",
    ])
    block = out[out.rfind("Integrated loudness:"):]
    i = re.search(r"I:\s*(-?[\d.]+)\s*LUFS", block)
    t = re.search(r"Peak:\s*(-?[\d.]+)\s*dBFS", block)
    return (float(i.group(1)) if i else None, float(t.group(1)) if t else None)


def loudnorm_json(src):
    out = run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
        "-af", "loudnorm=I={0}:TP={1}:LRA=11:print_format=json".format(
            TARGET_LUFS, TARGET_TP),
        "-f", "null", "-",
    ])
    start = out.rfind("{")
    end = out.rfind("}")
    return json.loads(out[start:end + 1])


def process(master, ver):
    d = master.parent
    wav_out = d / ("FINAL_OceanIQ_{0}_48k.wav".format(ver))
    mp3_out = d / ("FINAL_OceanIQ_{0}.mp3".format(ver))

    print("   measuring ...", flush=True)
    ln = loudnorm_json(master)
    filt = ("loudnorm=I={0}:TP={1}:LRA=11"
            ":measured_I={2}:measured_TP={3}:measured_LRA={4}"
            ":measured_thresh={5}:offset={6}:linear=true:print_format=summary"
            ).format(
        TARGET_LUFS, TARGET_TP,
        ln["input_i"], ln["input_tp"], ln["input_lra"],
        ln["input_thresh"], ln["target_offset"],
    )

    print("   wav 48k/24bit ...", flush=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(master), "-af", filt,
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s24le", str(wav_out),
    ])

    print("   mp3 192k ...", flush=True)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(wav_out), "-c:a", "libmp3lame", "-b:a", "192k", str(mp3_out),
    ])

    lufs, tp = measure(wav_out)
    print("      wav : {0:.2f} MB   {1} LUFS   {2} dBTP".format(
        wav_out.stat().st_size / 1048576.0, lufs, tp), flush=True)
    print("      mp3 : {0:.2f} MB".format(mp3_out.stat().st_size / 1048576.0),
          flush=True)
    return wav_out, mp3_out, lufs, tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--versions", default="v1,v2")
    args = ap.parse_args()

    if not AUDIO.is_dir():
        raise SystemExit("audio/ directory not found")

    for ver in args.versions.split(","):
        ver = ver.strip()
        d = AUDIO / (ver + "_" + args.voice)
        if not d.is_dir():
            print("skip (not found): " + str(d))
            continue
        masters = sorted(d.glob("OceanIQ_*_master_*.wav"))
        if not masters:
            print("skip (no master): " + str(d))
            continue
        print("== {0} / {1} ==".format(ver, args.voice), flush=True)
        process(masters[0], ver.upper())
        print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
