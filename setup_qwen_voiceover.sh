#!/usr/bin/env bash
# Install Python deps and download Qwen3-TTS 1.7B Base + VoiceDesign models.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

ENV_DIR="${ROOT}/.venv"
PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"

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

echo
echo "============================================================"
echo " Qwen Voiceover setup"
echo "============================================================"
echo "Project: ${ROOT}"
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

# torch: leave whatever qwen-tts / user environment already provides;
# only install if missing
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
  # weights: safetensors or bin or sharded index
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
  echo "Downloading ${repo_id}"
  echo "  -> ${local_dir}"
  mkdir -p "${local_dir}"

  "${PYTHON}" - "${repo_id}" "${local_dir}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print(f"Finished: {repo_id}")
PY

  if ! model_looks_ready "${local_dir}"; then
    echo "ERROR: download finished but model still looks incomplete: ${local_dir}" >&2
    exit 1
  fi
}

# ------------------------------------------------------------------
# Models
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
