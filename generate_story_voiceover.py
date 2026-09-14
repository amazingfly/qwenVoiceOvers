#!/usr/bin/env python3
"""
Full-story line-by-line voiceover generator for Qwen3-TTS.

Supports:
  - Base model + voice cloning (ref_audio / ref_text)
  - Optional VoiceDesign fallback (description / instruct)

- Loads the model once and keeps it in memory for the entire story.
- Every line is generated separately.
- Resume support: skips lines that already have a valid WAV.
- Per-line config overrides are supported.
- Story-level progress: % done, sec/line, ETA.
- Per-line inference progress via HF StoppingCriteria (steps, %, sec/step, ETA).
- Checks requirements and runs setup_qwen_voiceover.sh if needed.
"""

import argparse
import csv
import gc
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SETUP_SCRIPT = ROOT / "setup_qwen_voiceover.sh"


# ---------------------------------------------------------------------------
# Requirement checks + auto-setup
# ---------------------------------------------------------------------------

def model_dir_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    if any(path.glob("*.safetensors")):
        return True
    if any(path.glob("*.bin")):
        return True
    if (path / "model.safetensors.index.json").is_file():
        return True
    if (path / "pytorch_model.bin.index.json").is_file():
        return True
    return False


def is_hf_repo_id(path_str: str) -> bool:
    return bool(re.match(r"^[\w.-]+/[\w.-]+$", path_str.strip()))


def ensure_qwen_import() -> None:
    try:
        from qwen_tts import Qwen3TTSModel  # noqa: F401
    except ImportError:
        print("qwen_tts is not importable. Running setup ...", flush=True)
        run_setup()
        try:
            from qwen_tts import Qwen3TTSModel  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "setup finished but qwen_tts still cannot be imported.\n"
                f"Try: {ROOT / '.venv' / 'bin' / 'python'} -m pip install -U qwen-tts\n"
                f"Original error: {exc}"
            ) from exc


def run_setup() -> None:
    if not SETUP_SCRIPT.is_file():
        raise SystemExit(
            f"Missing setup script: {SETUP_SCRIPT}\n"
            "Install deps manually or restore setup_qwen_voiceover.sh"
        )
    print()
    print("=" * 60)
    print(" Running setup_qwen_voiceover.sh")
    print("=" * 60)
    print(flush=True)
    result = subprocess.run(
        ["bash", str(SETUP_SCRIPT)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise SystemExit(f"Setup failed with exit code {result.returncode}")
    print(flush=True)


def resolve_model_path(raw: str) -> str:
    raw = raw.strip()
    if is_hf_repo_id(raw):
        return raw
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return str(path)


def ensure_model_available(raw_path: str, mode: str) -> str:
    raw_path = raw_path.strip()

    default_hf = (
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        if mode == "clone"
        else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    )
    default_local = ROOT / (
        "models/Qwen3-TTS-12Hz-1.7B-Base"
        if mode == "clone"
        else "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    )

    if is_hf_repo_id(raw_path):
        return raw_path

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()

    if model_dir_ready(path):
        return str(path)

    print(f"Model not ready at: {path}", flush=True)
    print("Running setup to install deps and download models ...", flush=True)
    run_setup()

    if model_dir_ready(path):
        return str(path)
    if model_dir_ready(default_local):
        print(f"Using downloaded model at: {default_local}", flush=True)
        return str(default_local)

    print(
        f"Local model still incomplete; falling back to Hugging Face id: {default_hf}",
        flush=True,
    )
    return default_hf


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

    return int(speaker.get("seed", 38117))


def resolve_path(root: Path, path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    return p


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def render_progress(done: int, total: int, avg_sec: float, bar_width: int = 28) -> str:
    if total <= 0:
        pct = 100.0
        filled = bar_width
    else:
        pct = 100.0 * done / total
        filled = int(bar_width * done / total)

    bar = "█" * filled + "░" * (bar_width - filled)
    remaining = max(total - done, 0)
    eta = avg_sec * remaining if avg_sec > 0 else 0.0

    return (
        f"[{bar}] {pct:5.1f}%  "
        f"{done}/{total}  "
        f"{avg_sec:6.1f}s/line  "
        f"ETA {format_duration(eta)}"
    )


def make_step_progress_criteria(max_new_tokens: int, report_every: int = 5):
    """
    Build HF StoppingCriteria that reports codec-step progress during generate().

    qwen_tts forwards **kwargs to Transformers generate(), so this is the
    supported way to observe per-step progress. Generation often ends early
    on EOS, so max_new_tokens is an upper bound (progress may jump to done).
    Returns (criteria_list_or_None, progress_state_dict).
    """
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None, {"supported": False, "steps": 0}

    state = {
        "supported": True,
        "steps": 0,
        "max_new_tokens": max_new_tokens,
        "t0": None,
    }

    class _StepProgress(StoppingCriteria):
        def __init__(self):
            self.step = 0
            self.t0 = time.monotonic()
            state["t0"] = self.t0

        def __call__(self, input_ids, scores, **kwargs):
            self.step += 1
            state["steps"] = self.step
            max_tok = max(max_new_tokens, 1)

            if self.step == 1 or self.step % report_every == 0 or self.step >= max_tok:
                elapsed = time.monotonic() - self.t0
                sec_per_step = elapsed / self.step if self.step else 0.0
                remaining = max(max_tok - self.step, 0)
                eta = sec_per_step * remaining
                pct = min(100.0 * self.step / max_tok, 100.0)
                bar_w = 24
                filled = int(bar_w * self.step / max_tok)
                filled = min(filled, bar_w)
                bar = "█" * filled + "░" * (bar_w - filled)
                # carriage-return update on one line
                msg = (
                    f"\r  infer [{bar}] {pct:5.1f}%  "
                    f"step {self.step}/{max_tok}  "
                    f"{sec_per_step:5.2f}s/step  "
                    f"ETA {format_duration(eta)}   "
                )
                sys.stdout.write(msg)
                sys.stdout.flush()

            return False  # never stop early; only report

    return StoppingCriteriaList([_StepProgress()]), state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate full-story voiceovers line-by-line with Qwen3-TTS (Base clone or VoiceDesign)."
    )
    parser.add_argument(
        "--config",
        default="story_script.json",
        help="Path to the story JSON config (default: story_script.json)",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Do not auto-run setup_qwen_voiceover.sh if requirements are missing",
    )
    parser.add_argument(
        "--step-report-every",
        type=int,
        default=5,
        help="Print inference step progress every N codec steps (default: 5)",
    )
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config).expanduser().resolve()

    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config not found: {CONFIG_PATH}")

    if not args.skip_setup:
        ensure_qwen_import()
    else:
        try:
            from qwen_tts import Qwen3TTSModel  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "qwen_tts not importable and --skip-setup was set.\n"
                f"{exc}"
            ) from exc

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    story = load_story(CONFIG_PATH)

    story_meta = story.get("story", {})
    output_rel = story_meta.get("output_dir", f"story-voiceovers/{CONFIG_PATH.stem}")
    OUTPUT_DIR = (ROOT / output_rel).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg = story.get("model", {})
    mode = model_cfg.get("mode", "clone").lower()

    raw_model_path = model_cfg.get(
        "path",
        "models/Qwen3-TTS-12Hz-1.7B-Base"
        if mode == "clone"
        else "models/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )

    if args.skip_setup:
        model_path = resolve_model_path(raw_model_path)
    else:
        model_path = ensure_model_available(raw_model_path, mode)

    device_map = model_cfg.get("device_map", "cpu")
    dtype_str = model_cfg.get("dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    threads = int(model_cfg.get("threads", os.environ.get("QWEN_THREADS", "6")))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    defaults = story.get("defaults", {})
    TEMPERATURE = float(defaults.get("temperature", 0.80))
    TOP_P = float(defaults.get("top_p", 0.90))
    TOP_K = int(defaults.get("top_k", 40))
    MAX_NEW_TOKENS = int(defaults.get("max_new_tokens", 1024))
    LANGUAGE = story_meta.get("language", "English")

    speakers = story["speakers"]
    lines = story["lines"]

    for key, spk in speakers.items():
        if mode == "clone":
            if not spk.get("ref_audio"):
                raise ValueError(
                    f"Speaker '{key}' is missing ref_audio (required for Base/clone mode)."
                )
            if not spk.get("ref_text"):
                raise ValueError(
                    f"Speaker '{key}' is missing ref_text (required for Base/clone mode)."
                )
            ref_path = resolve_path(ROOT, spk["ref_audio"])
            if not ref_path.exists():
                raise FileNotFoundError(
                    f"Speaker '{key}' ref_audio not found: {ref_path}"
                )
        else:
            if not spk.get("description"):
                raise ValueError(
                    f"Speaker '{key}' is missing description (required for VoiceDesign mode)."
                )

    shutil.copy2(CONFIG_PATH, OUTPUT_DIR / "CONFIG_USED.json")

    catalog_path = OUTPUT_DIR / "story_catalog.csv"

    def rebuild_catalog():
        with catalog_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "speaker", "seed", "duration_seconds",
                "wav", "direction", "text", "mode"
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
                    mode,
                ])

    rebuild_catalog()

    work = []
    for i, line in enumerate(lines):
        lid = str(line.get("id", f"{i+1:03d}"))
        speaker_key = line["speaker"]
        speaker = speakers[speaker_key]
        seed = get_line_seed(line, speaker, defaults, i)
        filename = f"{lid}_{slugify(speaker_key)}_seed_{seed}.wav"
        out = OUTPUT_DIR / filename
        if not (out.exists() and out.stat().st_size > 10000):
            work.append((i, line))

    total_work = len(work)

    print()
    print("=" * 60)
    print("Qwen3-TTS – Full Story Voiceover")
    print("=" * 60)
    print(f"Config:     {CONFIG_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Mode:       {mode}")
    print(f"Model:      {model_path}")
    print(f"Lines:      {len(lines)}")
    print(f"Remaining:  {total_work}")
    print(f"Threads:    {torch.get_num_threads()}")
    print()

    if total_work == 0:
        print("All lines already generated. Nothing to do.")
        return

    print("Loading model (this happens only once)...")
    t0 = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
    )
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")
    print()
    print(render_progress(0, total_work, 0.0))
    sys.stdout.flush()

    times = []
    done_count = 0
    step_progress_warned = False

    try:
        for work_index, (i, line) in enumerate(work):
            lid = str(line.get("id", f"{i+1:03d}"))
            speaker_key = line["speaker"]
            if speaker_key not in speakers:
                raise ValueError(f"Line {lid}: unknown speaker '{speaker_key}'")

            speaker = speakers[speaker_key]
            seed = get_line_seed(line, speaker, defaults, i)
            text = line["text"].strip()
            if not text:
                print(f"\nSKIP empty text on line {lid}")
                done_count += 1
                avg = (sum(times) / len(times)) if times else 0.0
                print(render_progress(done_count, total_work, avg))
                sys.stdout.flush()
                continue

            filename = f"{lid}_{slugify(speaker_key)}_seed_{seed}.wav"
            output_path = OUTPUT_DIR / filename

            temp = float(line.get("temperature", TEMPERATURE))
            top_p = float(line.get("top_p", TOP_P))
            top_k = int(line.get("top_k", TOP_K))
            max_tokens = int(line.get("max_new_tokens", MAX_NEW_TOKENS))

            sidecar = {
                "id": lid,
                "speaker_key": speaker_key,
                "speaker_name": speaker.get("name"),
                "seed": seed,
                "direction": line.get("direction", ""),
                "text": text,
                "mode": mode,
                "generation": {
                    "temperature": temp,
                    "top_p": top_p,
                    "top_k": top_k,
                    "max_new_tokens": max_tokens,
                    "dtype": dtype_str,
                },
            }

            if mode == "clone":
                ref_audio = str(resolve_path(ROOT, speaker["ref_audio"]))
                ref_text = speaker["ref_text"]
                sidecar["ref_audio"] = ref_audio
                sidecar["ref_text"] = ref_text
            else:
                description = line.get("description") or speaker["description"]
                direction = line.get("direction", "").strip()
                instruct = (
                    f"{description}\n\nFor this line: {direction}"
                    if direction else description
                )
                sidecar["instruct"] = instruct

            output_path.with_suffix(".json").write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            print()
            print("-" * 60)
            print(f"Line {lid}  |  {speaker.get('name', speaker_key)}  [{mode}]")
            print(f"Seed: {seed}")
            if mode == "clone":
                print(f"Ref:  {speaker['ref_audio']}")
            print(f"Direction: {line.get('direction', '')}")
            print(f"Text: {text}")
            print("-" * 60)
            sys.stdout.flush()

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            step_criteria, step_state = make_step_progress_criteria(
                max_tokens,
                report_every=max(1, args.step_report_every),
            )
            gen_kwargs = dict(
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_tokens,
            )
            if step_criteria is not None:
                gen_kwargs["stopping_criteria"] = step_criteria
            elif not step_progress_warned:
                print(
                    "(Per-step progress unavailable: transformers StoppingCriteria not found)",
                    flush=True,
                )
                step_progress_warned = True

            started = time.monotonic()
            try:
                with torch.inference_mode():
                    if mode == "clone":
                        wavs, sample_rate = model.generate_voice_clone(
                            text=text,
                            language=LANGUAGE,
                            ref_audio=ref_audio,
                            ref_text=ref_text,
                            **gen_kwargs,
                        )
                    else:
                        wavs, sample_rate = model.generate_voice_design(
                            text=text,
                            language=LANGUAGE,
                            instruct=instruct,
                            **gen_kwargs,
                        )

                # finish the infer progress line
                if step_state.get("steps", 0) > 0:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                wav = wavs[0]
                sf.write(output_path, wav, sample_rate)
                duration = len(wav) / float(sample_rate)
                elapsed = time.monotonic() - started
                times.append(elapsed)

                steps = step_state.get("steps", 0)
                if steps > 0:
                    print(
                        f"DONE  {duration:.2f}s audio  |  {elapsed:.1f}s wall  |  "
                        f"{steps} steps  |  {elapsed/steps:.2f}s/step"
                    )
                else:
                    print(f"DONE  {duration:.2f}s audio  |  {elapsed:.1f}s wall")
            except TypeError as exc:
                # Older path may reject stopping_criteria — retry once without it
                if step_criteria is not None and "stopping_criteria" in str(exc):
                    if not step_progress_warned:
                        print(
                            "\n(Model rejected stopping_criteria; "
                            "per-step progress disabled for this run)",
                            flush=True,
                        )
                        step_progress_warned = True
                    gen_kwargs.pop("stopping_criteria", None)
                    try:
                        with torch.inference_mode():
                            if mode == "clone":
                                wavs, sample_rate = model.generate_voice_clone(
                                    text=text,
                                    language=LANGUAGE,
                                    ref_audio=ref_audio,
                                    ref_text=ref_text,
                                    **gen_kwargs,
                                )
                            else:
                                wavs, sample_rate = model.generate_voice_design(
                                    text=text,
                                    language=LANGUAGE,
                                    instruct=instruct,
                                    **gen_kwargs,
                                )
                        wav = wavs[0]
                        sf.write(output_path, wav, sample_rate)
                        duration = len(wav) / float(sample_rate)
                        elapsed = time.monotonic() - started
                        times.append(elapsed)
                        print(f"DONE  {duration:.2f}s audio  |  {elapsed:.1f}s wall")
                    except Exception as exc2:
                        elapsed = time.monotonic() - started
                        times.append(elapsed)
                        print(f"\nFAILED: {type(exc2).__name__}: {exc2}")
                else:
                    elapsed = time.monotonic() - started
                    times.append(elapsed)
                    print(f"\nFAILED: {type(exc).__name__}: {exc}")
            except Exception as exc:
                elapsed = time.monotonic() - started
                times.append(elapsed)
                print(f"\nFAILED: {type(exc).__name__}: {exc}")

            done_count += 1
            avg = sum(times) / len(times)
            print(render_progress(done_count, total_work, avg))
            sys.stdout.flush()

            gc.collect()
            rebuild_catalog()

    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    if times:
        print(
            f"Finished {done_count}/{total_work}  |  "
            f"avg {sum(times)/len(times):.1f}s/line  |  "
            f"total gen {format_duration(sum(times))}"
        )
    print(f"Catalog: {catalog_path}")


if __name__ == "__main__":
    main()
