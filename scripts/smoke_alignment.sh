#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8989}"
AUDIO="${2:?Usage: smoke_alignment.sh PORT AUDIO TEXT_FILE LANGUAGE}"
TRANSCRIPT="${3:?Transcript file is required}"
LANGUAGE="${4:-en}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/validation"
if [[ -f "${WORKSPACE_ROOT}/README_WORKSPACE.md" ]]; then
  ARTIFACT_ROOT="${WORKSPACE_ROOT}/artifacts/validation"
fi
OUT_DIR="${ARTIFACT_ROOT}/$(date '+%Y%m%d-%H%M%S')-alignment-${PORT}"
mkdir -p "$OUT_DIR"
BASE_URL="http://${QWEN_MLX_HOST:-127.0.0.1}:${PORT}"
curl --silent --show-error "$BASE_URL/healthz" > "$OUT_DIR/healthz.json"
python3 - "$OUT_DIR/healthz.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p['components']['alignment']['ready']
assert p['components']['queue']['in_flight'] == 0
PY
for FORMAT in verbose_json srt vtt; do
  curl --fail-with-body --silent --show-error "$BASE_URL/v1/audio/alignments" \
    -F "file=@${AUDIO}" -F "text=<${TRANSCRIPT}" \
    --form-string "language=${LANGUAGE}" --form-string "response_format=${FORMAT}" \
    -o "$OUT_DIR/alignment.${FORMAT}"
done
python3 - "$OUT_DIR" "$TRANSCRIPT" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1]); text=Path(sys.argv[2]).read_bytes().decode('utf-8')
p=json.loads((out/'alignment.verbose_json').read_bytes())
assert p['task']=='align' and p['text']==text
assert ''.join(s['text'] for s in p['segments'])==text
assert p['timestamp_provenance']['words']=='forced_alignment'
assert p['timestamp_provenance']['timestamp_repair']=='none'
assert p['diagnostics']['structural_coverage_verified']
assert not p['diagnostics']['acoustic_accuracy_verified']
assert not p['diagnostics']['zero_duration_word_indices']
assert p['words'] and all(0 <= w['start'] < w['end'] <= p['duration'] for w in p['words'])
assert (out/'alignment.srt').read_bytes().startswith(b'1\n')
assert (out/'alignment.vtt').read_bytes().startswith(b'WEBVTT\n')
print(f"Alignment smoke passed: {len(p['words'])} words, {len(p['segments'])} segments")
PY
STATUS="$(curl --silent --show-error "$BASE_URL/audio/alignments" \
  -F "file=@${AUDIO}" -F "text=<${TRANSCRIPT}" --form-string "language=${LANGUAGE}" \
  -F 'response_format=json' -o "$OUT_DIR/invalid-format.json" -w '%{http_code}')"
[[ "$STATUS" == "400" ]]
echo "Evidence saved to $OUT_DIR"
