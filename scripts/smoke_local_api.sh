#!/usr/bin/env bash
set -euo pipefail

HOST="${QWEN_MLX_HOST:-127.0.0.1}"
PORT="${1:-${QWEN_MLX_PORT:-8989}}"
MODEL="${QWEN_MLX_MODEL_ID:-qwen3-asr-mlx}"
ROLE="${QWEN_MLX_SMOKE_ROLE:-}"
if [[ -z "${ROLE}" ]]; then
  case "${PORT}" in
    8989) ROLE="release" ;;
    8990) ROLE="debug" ;;
    *) ROLE="local" ;;
  esac
fi
if [[ ! "${ROLE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "QWEN_MLX_SMOKE_ROLE must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/validation"
if [[ -f "${WORKSPACE_ROOT}/README_WORKSPACE.md" ]]; then
  ARTIFACT_ROOT="${WORKSPACE_ROOT}/artifacts/validation"
fi
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

AUDIO_FILE="${2:-$TMPDIR/smoke.wav}"
if [[ ! -f "$AUDIO_FILE" ]]; then
  if ! command -v say >/dev/null 2>&1 || ! command -v afconvert >/dev/null 2>&1; then
    echo "No input audio file provided. Auto-generating smoke audio requires macOS 'say' and 'afconvert'; pass an existing WAV file as the second argument." >&2
    exit 1
  fi
  say -o "$TMPDIR/smoke.aiff" "hello from qwen three asr smoke test"
  afconvert -f WAVE -d LEI16@16000 "$TMPDIR/smoke.aiff" "$AUDIO_FILE" >/dev/null 2>&1
fi

BASE_URL="http://${HOST}:${PORT}"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
OUT_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}-${ROLE}-${PORT}"
mkdir -p "$OUT_DIR"

echo "Checking healthz..."
curl --fail --silent --show-error "${BASE_URL}/healthz" | tee "${OUT_DIR}/healthz.json"

echo
echo "Checking models..."
curl --fail --silent --show-error "${BASE_URL}/v1/models" | tee "${OUT_DIR}/models.json"

echo
echo "Checking multipart transcription..."
curl --fail --silent --show-error \
  -F "file=@${AUDIO_FILE};type=audio/wav" \
  -F "model=${MODEL}" \
  "${BASE_URL}/v1/audio/transcriptions" | tee "${OUT_DIR}/transcription.json"

echo
echo "Checking chat completions audio path..."
AUDIO_BASE64="$(base64 < "$AUDIO_FILE" | tr -d '\n')"
cat > "${TMPDIR}/chat_request.json" <<EOF
{
  "model": "${MODEL}",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Transcribe the attached audio."
        },
        {
          "type": "input_audio",
          "input_audio": {
            "data": "${AUDIO_BASE64}",
            "format": "wav"
          }
        }
      ]
    }
  ]
}
EOF
curl --fail --silent --show-error \
  -H "Content-Type: application/json" \
  -d @"${TMPDIR}/chat_request.json" \
  "${BASE_URL}/v1/chat/completions" | tee "${OUT_DIR}/chat_completions.json"

echo
echo "Smoke checks completed. Evidence saved to ${OUT_DIR}"
