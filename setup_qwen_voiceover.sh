#!/usr/bin/env bash
# Install Python deps and download Qwen3-TTS 1.7B Base + VoiceDesign models.
#
# These models are public on the Hugging Face Hub. You do NOT need a Hugging Face
# account or token (HF_TOKEN). The strings Qwen/... are model *repo ids* (paths on
# the Hub), not login credentials.
#
# Optional: if HF_TOKEN is already exported, it will be used; otherwise downloads
# are anonymous.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

ENV_DIR="${ROOT}/.venv"
PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

# Public Hub repo ids (not API keys / login)
MODEL_BASE_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DESIGN_ID="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_BASE_DIR="${ROOT}/models/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DESIGN_DIR="${ROOT}/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

CACHE_DIR="${ROOT}/cache/huggingface"
TMP_DIR="${ROOT}/tmp"

mkdir -p \
  "${ROOT}/models" \
  "${CACHE_DIR}" \
  "${TMP_DIR}" \
  "${ROOT}/story-voiceovers" \
  "${ROOT}/voice-design-options"

export HF_HOME="${CACHE_DIR}"
export HF_HUB_CACHE="${CACHE_DIR}/hub"
export TMPDIR="${TMP_DIR}"
export TEMP="${TMP_DIR}"
export TMP="${TMP_DIR}"

# Do not require login. If the user has no token, force anonymous Hub access.
# (huggingface_hub also reads HF_TOKEN from the environment when present.)
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
fi

echo
echo "============================================================"
echo " Qwen Voiceover setup"
echo "============================================================"
echo "Project: ${ROOT}"
if [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Hub auth:  HF token detected (optional; not required for these models)"
else
  echo "Hub auth:  none (anonymous download — OK for public Qwen3-TTS models)"
fi
echo

# ------------------------------------------------------------------
# Virtualenv
# ------------------------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "Creating virtualenv at ${ENV_DIR} ..."
  python3 -m venv "${ENV_DIR}"
fi

PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

echo "Using: ${PYTHON}"
"${PYTHON}" -V
echo

# ------------------------------------------------------------------
# Python packages
# ------------------------------------------------------------------
echo "Installing / upgrading Python packages ..."
"${PIP}" install -U pip setuptools wheel

# transformers 4.57.x (pulled by qwen-tts) needs huggingface_hub < 1.0
"${PIP}" install "huggingface_hub>=0.34.0,<1.0"
"${PIP}" install -U qwen-tts soundfile numpy

if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch not found — installing CPU torch ..."
  "${PIP}" install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo
echo "Verifying qwen_tts import ..."
"${PYTHON}" -c "from qwen_tts import Qwen3TTSModel; print('qwen_tts OK')"

# ------------------------------------------------------------------
# Model download helper
# ------------------------------------------------------------------
model_looks_ready() {
  local dir="$1"
  [[ -f "${dir}/config.json" ]] || return 1
  if compgen -G "${dir}/*.safetensors" > /dev/null 2>&1; then
    return 0
  fi
  if compgen -G "${dir}/*.bin" > /dev/null 2>&1; then
    return 0
  fi
  if [[ -f "${dir}/model.safetensors.index.json" ]] || [[ -f "${dir}/pytorch_model.bin.index.json" ]]; then
    return 0
  fi
  return 1
}

download_model() {
  local repo_id="$1"
  local local_dir="$2"

  if model_looks_ready "${local_dir}"; then
    echo "Already present: ${local_dir}"
    return 0
  fi

  echo
  echo "Downloading public model (no HF login required)"
  echo "  repo id: ${repo_id}"
  echo "  local:   ${local_dir}"
  mkdir -p "${local_dir}"

  # token=None => anonymous. If HF_TOKEN is set in the environment,
  # huggingface_hub may still pick it up; we never require it.
  if ! "${PYTHON}" - "${repo_id}" "${local_dir}" <<'PY'
import os
import sys

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

repo_id, local_dir = sys.argv[1], sys.argv[2]

# Prefer explicit anonymous when no token is configured.
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,  # None is fine for public models
        resume_download=True,
    )
except TypeError:
    # Older huggingface_hub may not accept resume_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
    )
except HfHubHTTPError as exc:
    status = getattr(exc, "response", None)
    code = getattr(status, "status_code", None) if status is not None else None
    print(f"ERROR: Hub download failed for {repo_id}", file=sys.stderr)
    print(f"  {exc}", file=sys.stderr)
    if code in (401, 403):
        print(
            "  This usually means the repo is gated or private.\n"
            "  Public Qwen3-TTS models should not need a token.\n"
            "  If the Hub requires acceptance of terms, open the model page\n"
            "  in a browser while logged in, accept the license, then set\n"
            "  HF_TOKEN and re-run setup.",
            file=sys.stderr,
        )
    elif code == 429:
        print("  Rate limited — wait and retry.", file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(f"ERROR: download failed for {repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)

print(f"Finished: {repo_id}")
PY
  then
    echo "ERROR: download command failed for ${repo_id}" >&2
    exit 1
  fi

  if ! model_looks_ready "${local_dir}"; then
    echo "ERROR: download finished but model still looks incomplete: ${local_dir}" >&2
    echo "  Expected config.json and weight files (*.safetensors or *.bin)." >&2
    exit 1
  fi
}

# ------------------------------------------------------------------
# Models (public repo ids — not credentials)
# ------------------------------------------------------------------
download_model "${MODEL_BASE_ID}" "${MODEL_BASE_DIR}"
download_model "${MODEL_DESIGN_ID}" "${MODEL_DESIGN_DIR}"

echo
echo "============================================================"
echo " Setup complete"
echo "============================================================"
echo "Base:        ${MODEL_BASE_DIR}"
echo "VoiceDesign: ${MODEL_DESIGN_DIR}"
echo "Python:      ${PYTHON}"
echo
echo "Run story generation with:"
echo "  ${PYTHON} generate_story_voiceover.py --config story_script.json"
echo
