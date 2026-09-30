"""Generate the Ocean IQ narration with Kyutai Pocket TTS.

Produces, per version:
  - one WAV per section (for the editor)
  - one uncut master WAV (sections joined with short silences)

Run:
    python scripts/generate_narration.py --version both --voice charles
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from narration_scripts import V1, V2  # noqa: E402

SCRIPTS = {"v1": V1, "v2": V2}
GAP_SECONDS = 0.6


def output_dir(version, voice):
    d = HERE.parent / "audio" / (version + "_" + voice)
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_name(section):
    keep = []
    for ch in section:
        keep.append(ch if (ch.isalnum() or ch == " ") else "_")
    return "_".join("".join(keep).split())


def build_report(vkey, voice, rows, total, master_path):
    out = []
    out.append("=== " + vkey.upper() + " ===")
    out.append("voice: " + voice)
    for idx, sec, start, end, dur in rows:
        out.append(
            "{0}. {1}\n     start {2:7.2f}s   end {3:7.2f}s   dur {4:6.2f}s".format(
                idx, sec, start, end, dur
            )
        )
    mins = int(total // 60)
    secs = total % 60
    out.append("   TOTAL {0:.2f}s  ({1}m {2:.1f}s)".format(total, mins, secs))
    out.append("   master: " + str(master_path))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2", "both"], default="v1")
    ap.add_argument("--voice", default="charles")
    args = ap.parse_args()

    from pocket_tts import TTSModel

    print("[1/3] loading pocket-tts model (one-time, ~4 min) ...", flush=True)
    t0 = time.time()
    model = TTSModel.load_model()
    sr = model.sample_rate
    print("      loaded in {0:.1f}s   sample_rate={1}".format(time.time() - t0, sr), flush=True)

    print("[2/3] preparing voice '" + args.voice + "' ...", flush=True)
    voice_state = model.get_state_for_audio_prompt(args.voice)

    versions = ["v1", "v2"] if args.version == "both" else [args.version]

    for vkey in versions:
        script = SCRIPTS[vkey]
        out_dir = output_dir(vkey, args.voice)

        print("[3/3] generating " + vkey.upper() + " ...", flush=True)
        chunks = []
        rows = []
        cursor = 0.0

        for i, section in enumerate(script, 1):
            lines = script[section]
            text = " ".join(lines)

            t0 = time.time()
            audio = model.generate_audio(voice_state, text)
            gen_s = time.time() - t0
            samples = audio.numpy()
            dur = len(samples) / float(sr)

            sec_path = out_dir / ("sec{0:02d}_{1}.wav".format(i, safe_name(section)))
            scipy.io.wavfile.write(sec_path, sr, samples)

            end = cursor + dur
            rows.append((i, section, cursor, end, dur))
            print("   [{0}/5] {1:<22} {2:6.2f}s audio  ({3:5.1f}s gen)".format(
                i, section, dur, gen_s), flush=True)

            chunks.append(samples)

            if i < len(script):
                chunks.append(np.zeros(int(sr * GAP_SECONDS), dtype=samples.dtype))
                cursor = end + GAP_SECONDS
            else:
                cursor = end

        master = np.concatenate(chunks)
        master_path = out_dir / ("OceanIQ_{0}_master_{1}.wav".format(vkey.upper(), args.voice))
        scipy.io.wavfile.write(master_path, sr, master)

        total = len(master) / float(sr)
        report = build_report(vkey, args.voice, rows, total, master_path)
        (out_dir / "TIMING.txt").write_text(report, encoding="utf-8")

        print("")
        print(report)
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
