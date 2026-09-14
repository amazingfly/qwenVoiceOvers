#!/usr/bin/env python3

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


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = (
    ROOT
    / "models"
    / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)


# ============================================================
# Command line
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Generate Qwen3-TTS VoiceDesign character auditions "
        "from JSON descriptions and per-character dialog styles."
    )
)

parser.add_argument(
    "--voices",
    default=str(ROOT / "voice-designs.json"),
    help=(
        "JSON configuration file. "
        "Default: voice-designs.json"
    ),
)

args = parser.parse_args()

VOICE_FILE = Path(
    args.voices
).expanduser().resolve()

if not VOICE_FILE.exists():
    raise SystemExit(
        f"Voice JSON file not found: {VOICE_FILE}"
    )


# ============================================================
# Output directory
#
# monster-voices.json becomes:
#
# voice-design-options/
#   monster-voices/
# ============================================================

CONFIG_NAME = VOICE_FILE.stem

OUTPUT_DIR = (
    ROOT
    / "voice-design-options"
    / CONFIG_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CPU configuration
# ============================================================

threads = int(
    os.environ.get(
        "QWEN_THREADS",
        "6",
    )
)

torch.set_num_threads(threads)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# Generation settings
#
# These are deliberately a little less stochastic than the
# previous audition generator.
#
# We want interesting designed voices, but right now the goal
# is finding CHARACTER IDENTITIES that generalize well to
# later cloning.
# ============================================================

TEMPERATURE = 0.80
TOP_P = 0.90
TOP_K = 40

MAX_NEW_TOKENS = 1024

# One seed per CHARACTER.
#
# Every dialog style for that character uses the same seed.
# That removes one unnecessary source of variation while we
# test whether the designed identity survives different
# deliveries.
BASE_SEED = 38117
SEED_STEP = 7919


# ============================================================
# Helpers
# ============================================================

def slugify(text):

    text = str(text).strip().lower()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def load_config(path):

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(data, dict):
        voices = data.get("voices")
    else:
        voices = data

    if not isinstance(voices, list):
        raise ValueError(
            'JSON must contain a "voices" list.'
        )

    if not voices:
        raise ValueError(
            "No voices were found in the JSON."
        )

    result = []

    for voice_number, voice in enumerate(
        voices,
        start=1,
    ):

        if not isinstance(voice, dict):
            raise ValueError(
                f"Voice #{voice_number} is not an object."
            )

        name = str(
            voice.get("name", "")
        ).strip()

        description = str(
            voice.get("description", "")
        ).strip()

        if not name:
            raise ValueError(
                f"Voice #{voice_number} has no name."
            )

        if not description:
            raise ValueError(
                f"{name}: missing description."
            )

        slug = str(
            voice.get("slug", "")
        ).strip()

        if not slug:

            slug = (
                f"{voice_number:02d}_"
                f"{slugify(name)}"
            )

        dialogs = voice.get("dialogs")

        if not isinstance(dialogs, list):
            raise ValueError(
                f"{name}: missing dialogs list."
            )

        if not dialogs:
            raise ValueError(
                f"{name}: dialogs list is empty."
            )

        clean_dialogs = []

        for dialog_number, dialog in enumerate(
            dialogs,
            start=1,
        ):

            if not isinstance(dialog, dict):
                raise ValueError(
                    f"{name}: dialog #{dialog_number} "
                    "is not an object."
                )

            dialog_name = str(
                dialog.get("name", "")
            ).strip()

            direction = str(
                dialog.get("direction", "")
            ).strip()

            text = str(
                dialog.get("text", "")
            ).strip()

            if not dialog_name:
                raise ValueError(
                    f"{name}: dialog #{dialog_number} "
                    "has no name."
                )

            if not direction:
                raise ValueError(
                    f"{name}/{dialog_name}: "
                    "missing direction."
                )

            if not text:
                raise ValueError(
                    f"{name}/{dialog_name}: "
                    "missing text."
                )

            clean_dialogs.append(
                {
                    "name": dialog_name,
                    "direction": direction,
                    "text": text,
                }
            )

        # Allow an explicit per-character seed in JSON.
        # Otherwise derive a deterministic one.
        seed = int(
            voice.get(
                "seed",
                BASE_SEED
                + ((voice_number - 1) * SEED_STEP),
            )
        )

        result.append(
            {
                "name": name,
                "slug": slug,
                "description": description,
                "seed": seed,
                "dialogs": clean_dialogs,
            }
        )

    return result


VOICE_DESIGNS = load_config(
    VOICE_FILE
)


# ============================================================
# Preserve exact input configuration
# ============================================================

shutil.copy2(
    VOICE_FILE,
    OUTPUT_DIR / "CONFIG_USED.json",
)


# ============================================================
# Catalog
# ============================================================

CATALOG = (
    OUTPUT_DIR
    / "voice_candidates.csv"
)


def rebuild_catalog():

    with CATALOG.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "voice",
            "voice_slug",
            "dialog",
            "seed",
            "duration_seconds",
            "wav",
            "direction",
            "text",
            "voice_description",
        ])

        for voice_number, voice in enumerate(
            VOICE_DESIGNS,
            start=1,
        ):

            voice_dir = (
                OUTPUT_DIR
                / voice["slug"]
            )

            for dialog_number, dialog in enumerate(
                voice["dialogs"],
                start=1,
            ):

                dialog_slug = slugify(
                    dialog["name"]
                )

                filename = (
                    f"{dialog_number:02d}_"
                    f"{dialog_slug}_"
                    f"seed_{voice['seed']}.wav"
                )

                wav_path = (
                    voice_dir
                    / filename
                )

                duration = ""

                if wav_path.exists():

                    try:

                        info = sf.info(
                            wav_path
                        )

                        duration = round(
                            info.frames
                            / info.samplerate,
                            2,
                        )

                    except Exception:
                        pass

                writer.writerow([
                    voice["name"],
                    voice["slug"],
                    dialog["name"],
                    voice["seed"],
                    duration,
                    str(
                        wav_path.relative_to(
                            ROOT
                        )
                    ),
                    dialog["direction"],
                    dialog["text"],
                    voice["description"],
                ])


rebuild_catalog()


# ============================================================
# Summary
# ============================================================

total_dialogs = sum(
    len(voice["dialogs"])
    for voice in VOICE_DESIGNS
)

print()
print("============================================================")
print("Qwen3-TTS 1.7B VoiceDesign")
print("Character Identity Auditions")
print("============================================================")
print()

print(f"Configuration: {VOICE_FILE}")
print(f"Output:        {OUTPUT_DIR}")
print(f"CPU threads:   {torch.get_num_threads()}")

print()
print(
    f"Characters:    {len(VOICE_DESIGNS)}"
)

print(
    f"Dialog tests:  {total_dialogs}"
)

print()
print(
    "Generation settings:"
)

print(
    f"  temperature = {TEMPERATURE}"
)

print(
    f"  top_p       = {TOP_P}"
)

print(
    f"  top_k       = {TOP_K}"
)

print()
print(
    "NOTE: all dialogs for a character use "
    "the same seed."
)
print()


# ============================================================
# Determine whether anything remains.
# ============================================================

remaining = 0

for voice in VOICE_DESIGNS:

    voice_dir = (
        OUTPUT_DIR
        / voice["slug"]
    )

    for dialog_number, dialog in enumerate(
        voice["dialogs"],
        start=1,
    ):

        filename = (
            f"{dialog_number:02d}_"
            f"{slugify(dialog['name'])}_"
            f"seed_{voice['seed']}.wav"
        )

        output_path = (
            voice_dir
            / filename
        )

        if not (
            output_path.exists()
            and output_path.stat().st_size
            > 10000
        ):
            remaining += 1


if remaining == 0:

    print(
        "Every requested sample already exists."
    )

    print(
        "Nothing to generate."
    )

    raise SystemExit(0)


print(
    f"Samples remaining: {remaining}"
)
print()


# ============================================================
# Load VoiceDesign model ONCE.
# ============================================================

print(
    "Loading Qwen3-TTS 1.7B VoiceDesign BF16..."
)

load_started = time.monotonic()

model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR),
    device_map="cpu",
    dtype=torch.bfloat16,
)

print(
    f"Model loaded in "
    f"{time.monotonic() - load_started:.1f} seconds."
)

print()


# ============================================================
# Generate
# ============================================================

overall_number = 0

try:

    for voice_number, voice in enumerate(
        VOICE_DESIGNS,
        start=1,
    ):

        voice_dir = (
            OUTPUT_DIR
            / voice["slug"]
        )

        voice_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ----------------------------------------------------
        # Save the stable character identity description.
        # ----------------------------------------------------

        (
            voice_dir
            / "VOICE_DESCRIPTION.txt"
        ).write_text(
            voice["description"]
            + "\n",
            encoding="utf-8",
        )


        print()
        print("############################################################")
        print(
            f"CHARACTER {voice_number}/"
            f"{len(VOICE_DESIGNS)}"
        )
        print(
            voice["name"]
        )
        print(
            f"Seed: {voice['seed']}"
        )
        print("############################################################")


        for dialog_number, dialog in enumerate(
            voice["dialogs"],
            start=1,
        ):

            overall_number += 1

            dialog_slug = slugify(
                dialog["name"]
            )

            filename = (
                f"{dialog_number:02d}_"
                f"{dialog_slug}_"
                f"seed_{voice['seed']}.wav"
            )

            output_path = (
                voice_dir
                / filename
            )


            # ------------------------------------------------
            # Character identity + current performance.
            #
            # The DIRECTION is NOT spoken.
            # ------------------------------------------------

            instruct = (
                voice["description"]
                + "\n\n"
                + "For this line: "
                + dialog["direction"]
            )


            # ------------------------------------------------
            # Save everything necessary to reproduce or later
            # clone this exact candidate.
            # ------------------------------------------------

            sidecar = (
                output_path
                .with_suffix(".json")
            )

            sidecar.write_text(
                json.dumps(
                    {
                        "voice_name": voice["name"],
                        "voice_slug": voice["slug"],
                        "seed": voice["seed"],
                        "dialog_name": dialog["name"],
                        "voice_description": (
                            voice["description"]
                        ),
                        "direction": (
                            dialog["direction"]
                        ),
                        "instruct": instruct,
                        "text": dialog["text"],
                        "generation": {
                            "temperature": TEMPERATURE,
                            "top_p": TOP_P,
                            "top_k": TOP_K,
                            "model": (
                                "Qwen3-TTS-12Hz-"
                                "1.7B-VoiceDesign"
                            ),
                            "dtype": "bfloat16",
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )


            # ------------------------------------------------
            # Resume support
            # ------------------------------------------------

            if (
                output_path.exists()
                and output_path.stat().st_size
                > 10000
            ):

                print()
                print(
                    f"SKIP: "
                    f"{voice['name']} / "
                    f"{dialog['name']}"
                )

                continue


            print()
            print("------------------------------------------------------------")
            print(
                f"Voice:     {voice['name']}"
            )
            print(
                f"Dialog:    {dialog['name']}"
            )
            print(
                f"Seed:      {voice['seed']}"
            )
            print(
                f"Output:    {output_path}"
            )
            print()
            print(
                f"Direction: {dialog['direction']}"
            )
            print()
            print(
                f"Text:      {dialog['text']}"
            )
            print("------------------------------------------------------------")


            # ------------------------------------------------
            # IMPORTANT:
            #
            # Reset to the SAME character seed before every
            # dialog for this character.
            #
            # Different instructions can still change the
            # speaker somewhat, but we are not adding a second
            # source of stochastic variation ourselves.
            # ------------------------------------------------

            seed = voice["seed"]

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)


            started = time.monotonic()

            try:

                with torch.inference_mode():

                    wavs, sample_rate = (
                        model.generate_voice_design(
                            text=dialog["text"],
                            language="English",
                            instruct=instruct,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            top_k=TOP_K,
                            max_new_tokens=(
                                MAX_NEW_TOKENS
                            ),
                        )
                    )


                wav = wavs[0]


                sf.write(
                    output_path,
                    wav,
                    sample_rate,
                )


                duration = (
                    len(wav)
                    / float(sample_rate)
                )

                elapsed = (
                    time.monotonic()
                    - started
                )


                print()
                print("COMPLETE")
                print(
                    f"Audio duration:  "
                    f"{duration:.2f} sec"
                )
                print(
                    f"Generation time: "
                    f"{elapsed:.1f} sec"
                )


            except Exception as exc:

                print()
                print(
                    f"FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )


            gc.collect()

            rebuild_catalog()


except KeyboardInterrupt:

    print()
    print()
    print(
        "Interrupted. Completed WAV files "
        "have been preserved."
    )

finally:

    rebuild_catalog()


print()
print("============================================================")
print("VOICE DESIGN RUN FINISHED")
print("============================================================")
print()

print(
    f"Output directory:\n"
    f"  {OUTPUT_DIR}"
)

print()

print(
    f"Catalog:\n"
    f"  {CATALOG}"
)

print()

