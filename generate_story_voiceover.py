#!/usr/bin/env python3
"""
Full-story line-by-line voiceover generator for Qwen3-TTS VoiceDesign.

- Loads the model once and keeps it in memory for the entire story.
- Every line is generated separately (as required).
- Resume support: skips lines that already have a valid WAV.
- Per-line config is fully supported.
"""

import argparse
import csv
import gc
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_story(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "lines" not in data or not isinstance(data["lines"], list):
        raise ValueError("Config must contain a non-empty 'lines' list.")
    if "speakers" not in data or not isinstance(data["speakers"], dict):
        raise ValueError("Config must contain a 'speakers' object.")
    return data


def get_line_seed(line: dict, speaker: dict, defaults: dict, line_index: int) -> int:
    # Explicit per-line seed wins
    if "seed" in line:
        return int(line["seed"])

    strategy = defaults.get("seed_strategy", "per_speaker")

    if strategy == "per_speaker":
        return int(speaker.get("seed", 38117))
    if strategy == "per_line":
        base = int(defaults.get("base_seed", 38117))
        step = int(defaults.get("seed_step", 7919))
        return base + (line_index * step)
    if strategy == "fixed":
        return int(defaults.get("base_seed", 38117))

    # fallback
    return int(speaker.get("seed", 38117))


def build_instruct(speaker: dict, line: dict) -> str:
    description = line.get("description") or speaker["description"]
    direction = line.get("direction", "").strip()
    if direction:
        return f"{description}\n\nFor this line: {direction}"
    return description


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate full-story voiceovers line-by-line with Qwen3-TTS VoiceDesign."
    )
    parser.add_argument(
        "--config",
        default="story_script.json",
        help="Path to the story JSON config (default: story_script.json)",
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parent
    CONFIG_PATH = Path(args.config).expanduser().resolve()

    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config not found: {CONFIG_PATH}")

    story = load_story(CONFIG_PATH)

    # Paths
    story_meta = story.get("story", {})
    output_rel = story_meta.get("output_dir", f"story-voiceovers/{CONFIG_PATH.stem}")
    OUTPUT_DIR = (ROOT / output_rel).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Model settings
    model_cfg = story.get("model", {})
    MODEL_DIR = Path(
        model_cfg.get("path", "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    )
    if not MODEL_DIR.is_absolute():
        MODEL_DIR = ROOT / MODEL_DIR

    device_map = model_cfg.get("device_map", "cpu")
    dtype_str = model_cfg.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    threads = int(model_cfg.get("threads", os.environ.get("QWEN_THREADS", "6")))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # Defaults
    defaults = story.get("defaults", {})
    TEMPERATURE = float(defaults.get("temperature", 0.80))
    TOP_P = float(defaults.get("top_p", 0.90))
    TOP_K = int(defaults.get("top_k", 40))
    MAX_NEW_TOKENS = int(defaults.get("max_new_tokens", 1024))
    LANGUAGE = story_meta.get("language", "English")

    speakers = story["speakers"]
    lines = story["lines"]

    # Copy config for reproducibility
    shutil.copy2(CONFIG_PATH, OUTPUT_DIR / "CONFIG_USED.json")

    # Catalog
    catalog_path = OUTPUT_DIR / "story_catalog.csv"

    def rebuild_catalog():
        with catalog_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "speaker", "seed", "duration_seconds",
                "wav", "direction", "text"
            ])
            for i, line in enumerate(lines):
                lid = str(line.get("id", f"{i+1:03d}"))
                speaker_key = line["speaker"]
                speaker = speakers[speaker_key]
                seed = get_line_seed(line, speaker, defaults, i)
                filename = f"{lid}_{slugify(speaker_key)}_seed_{seed}.wav"
                wav_path = OUTPUT_DIR / filename
                duration = ""
                if wav_path.exists():
                    try:
                        info = sf.info(wav_path)
                        duration = round(info.frames / info.samplerate, 2)
                    except Exception:
                        pass
                writer.writerow([
                    lid,
                    speaker.get("name", speaker_key),
                    seed,
                    duration,
                    str(wav_path.relative_to(ROOT)),
                    line.get("direction", ""),
                    line.get("text", ""),
                ])

    rebuild_catalog()

    # Count remaining work
    remaining = 0
    for i, line in enumerate(lines):
        lid = str(line.get("id", f"{i+1:03d}"))
        speaker_key = line["speaker"]
        speaker = speakers[speaker_key]
        seed = get_line_seed(line, speaker, defaults, i)
        filename = f"{lid}_{slugify(speaker_key)}_seed_{seed}.wav"
        out = OUTPUT_DIR / filename
        if not (out.exists() and out.stat().st_size > 10000):
            remaining += 1

    print()
    print("=" * 60)
    print("Qwen3-TTS VoiceDesign – Full Story Voiceover")
    print("=" * 60)
    print(f"Config:     {CONFIG_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Lines:      {len(lines)}")
    print(f"Remaining:  {remaining}")
    print(f"Threads:    {torch.get_num_threads()}")
    print()

    if remaining == 0:
        print("All lines already generated. Nothing to do.")
        return

    # ------------------------------------------------------------------
    # Load model ONCE and keep it for the whole story
    # ------------------------------------------------------------------
    print("Loading model (this happens only once)...")
    t0 = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR),
        device_map=device_map,
        dtype=dtype,
    )
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")
    print()

    # ------------------------------------------------------------------
    # Generate every line
    # ------------------------------------------------------------------
    try:
        for i, line in enumerate(lines):
            lid = str(line.get("id", f"{i+1:03d}"))
            speaker_key = line["speaker"]
            if speaker_key not in speakers:
                raise ValueError(f"Line {lid}: unknown speaker '{speaker_key}'")

            speaker = speakers[speaker_key]
            seed = get_line_seed(line, speaker, defaults, i)
            text = line["text"].strip()
            if not text:
                print(f"SKIP empty text on line {lid}")
                continue

            filename = f"{lid}_{slugify(speaker_key)}_seed_{seed}.wav"
            output_path = OUTPUT_DIR / filename

            # Resume
            if output_path.exists() and output_path.stat().st_size > 10000:
                print(f"SKIP existing: {filename}")
                continue

            # Per-line overrides
            temp = float(line.get("temperature", TEMPERATURE))
            top_p = float(line.get("top_p", TOP_P))
            top_k = int(line.get("top_k", TOP_K))
            max_tokens = int(line.get("max_new_tokens", MAX_NEW_TOKENS))

            instruct = build_instruct(speaker, line)

            # Sidecar for exact reproducibility
            sidecar = {
                "id": lid,
                "speaker_key": speaker_key,
                "speaker_name": speaker.get("name"),
                "seed": seed,
                "direction": line.get("direction", ""),
                "text": text,
                "instruct": instruct,
                "generation": {
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_new_tokens": max_tokens,
                    "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "dtype": dtype_str,
                },
            }
            output_path.with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            print("-" * 60)
            print(f"Line {lid}  |  {speaker.get('name', speaker_key)}")
            print(f"Seed: {seed}")
            print(f"Direction: {line.get('direction', '')}")
            print(f"Text: {text}")
            print("-" * 60)

            # Deterministic seed for this line
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            started = time.monotonic()
            try:
                with torch.inference_mode():
                    wavs, sample_rate = model.generate_voice_design(
                        text=text,
                        language=LANGUAGE,
                        instruct=instruct,
                        do_sample=True,
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                        max_new_tokens=max_tokens,
                    )

                wav = wavs[0]
                sf.write(output_path, wav, sample_rate)
                duration = len(wav) / float(sample_rate)
                elapsed = time.monotonic() - started

                print(f"DONE  {duration:.2f}s audio  |  {elapsed:.1f}s wall")
            except Exception as exc:
                print(f"FAILED: {type(exc).__name__}: {exc}")

            gc.collect()
            rebuild_catalog()

    finally:
        # Explicit cleanup (optional but nice)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("Story generation finished.")
    print(f"Catalog: {catalog_path}")


if __name__ == "__main__":
    main()
