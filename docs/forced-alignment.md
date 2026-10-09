# Standalone forced alignment (v1.3.0)

Use an authoritative, verbatim transcript when its wording is already correct and only timing is needed. This operation runs Qwen3-ForcedAligner without speech recognition. It never corrects, retranscribes, translates, or automatically anchors the transcript.

## HTTP and local files

```bash
curl --fail-with-body http://127.0.0.1:8989/v1/audio/alignments \
  -F 'file=@recording.wav' \
  -F 'text=<transcript.txt' \
  -F 'language=en' \
  -F 'response_format=verbose_json'

uv run qwen3-asr-mlx-server align recording.wav \
  --text-file transcript.txt --language en \
  --response-format verbose_json --output alignment.json
```

`POST /audio/alignments` is an alias. HTTP accepts multipart `file`, `text`, `language`, optional `response_format`, and optional `regions` JSON. It does not accept server-local paths or URLs. `prompt` is not a transcript field. Only `verbose_json` (default), `srt`, and `vtt` are supported by this operation; transcription's `json` and `text` contracts remain unchanged.

CLI `--text` is a mutually exclusive alternative to `--text-file`. Text and region files must be UTF-8. Transcript bytes are decoded without stripping text or converting CRLF line endings. `--output/-o` writes atomically after complete success; stdout is the default (`-` is also accepted). Audio, transcript, and region files cannot be output targets, including symlinks and hard links. Exit codes are 2 for invalid input/resource limits, 1 for model/execution/invalid alignment output, and 130 for interruption.

The CLI requires only a usable aligner, selected by `QWEN_MLX_ALIGNER_PATH` or the usual model discovery. It does not load ASR, invoke setup, or download weights. HTTP serving retains combined startup and its legacy ASR readiness rule. `/healthz` adds `version` and `components.alignment`, whose readiness is separate from top-level ASR readiness. An alignment request can succeed even when ASR is missing and `/healthz` consequently returns 503.

Language is required; `auto` is rejected. Supported languages are validated against the loaded aligner. The standard model supports `zh`, `yue`, `en`, `de`, `es`, `fr`, `it`, `pt`, `ru`, `ko`, and `ja`; language names are also accepted. Japanese requires `nagisa`, Korean requires `soynlp`. No ASR language detection occurs.

## Operating limits and explicit regions

Each inference window must fit **all** of these limits:

| Resource | Limit |
| --- | --- |
| Audio per window | 30 seconds of mono 16 kHz PCM |
| Canonical alignment units per window | 256 (words or language-specific tokens/characters) |
| Encoded text/timestamp tokens per window | 2048, including the audio placeholder and timestamp markers |
| Full authoritative transcript | 100,000 Unicode code points |
| Explicit regions | 1000 |
| Regions JSON | 1 MiB |
| Decoded source duration | Six hours |

HTTP also enforces `QWEN_MLX_MAX_UPLOAD_MB` (default 100 MiB). Local-file CLI bypasses that upload cap, while retaining all alignment limits. Compressed audio/video uses ffmpeg and temporary PCM storage; memory used for an inference window does not grow with the source's duration. The six-hour decode cap also bounds temporary PCM size. `QWEN_MLX_ALIGNMENT_CHUNK_SECONDS` and ASR segmentation options do not enlarge these standalone limits.

The first release uses a conservative local operating boundary, validated with the installed `mlx-audio 0.4.4` and the standard bf16 aligner. Its 80 ms timestamp bins, 5000 classifier classes, and 8192-position text configuration do not prove accurate alignment over their entire representable range. Resource limits are not an acoustic accuracy guarantee, especially for very dense speech, mismatched scripts, silence, or repeated material.

For longer sources, supply known audio/text regions. Example full text: `Hello. World!` (13 Python Unicode code points):

```json
[
  {"start": 2.0, "end": 5.0, "text_start": 0, "text_end": 7},
  {"start": 35.0, "end": 38.0, "text_start": 7, "text_end": 13}
]
```

```bash
curl --fail-with-body http://127.0.0.1:8989/v1/audio/alignments \
  -F 'file=@long.wav' -F 'text=<transcript.txt' -F 'language=en' \
  -F 'regions=<regions.json'

uv run qwen3-asr-mlx-server align long.wav \
  --text-file transcript.txt --language en --regions-file regions.json \
  --response-format vtt --output subtitles.vtt
```

Each region has exactly four fields. `start` and `end` are finite seconds, nonnegative, ordered, and non-overlapping within the original decoded audio. Gaps and unused leading/trailing audio are allowed. `text_start` and `text_end` are half-open Python/Unicode **code-point** offsets, not UTF-8 byte offsets or JavaScript UTF-16 indexes. Text regions must start at 0, end at `len(text)`, and provide ordered contiguous coverage without omissions or duplicates. Every region needs alignable tokens and must fit the window limits. Splitting a token, or changing Japanese/Korean token ownership across a boundary, is rejected.

Audio endpoints round to the nearest 16 kHz sample (half-sample maximum rounding); source-end tolerance is half a sample. `regions` in verbose JSON reports the actual frame boundaries and offsets. Windows are extracted and aligned sequentially, then placed on the original source timeline. The complete plan is validated before inference. Oversized unanchored sources fail with an actionable region error; there is no proportional text slicing, hidden ASR fallback, or automatic subdivision of an oversized region.

## Text, timestamps, and diagnostics

Verbose JSON has `task: "align"`, `transcript_source: "supplied"`, and forced-alignment provenance. It contains no invented ASR probabilities, temperatures, detected language fields, or token-generation statistics.

- Top-level `text` is exactly the supplied Unicode text. Concatenating `segments[].text` reproduces it exactly, including punctuation, case, and whitespace.
- Words expose exact `text_start`/`text_end` source spans and original `word` text. `alignment_token` is the aligner's internal canonical unit; word spans omit surrounding punctuation and whitespace, which remain in segment slices.
- Segments group complete word spans at sentence boundaries and, for long sentences, at word boundaries near 12 seconds. This is formatting, not guessed acoustic segmentation. Segment `word_start`/`word_end` indexes partition the word list; all ranges are half-open.
- `start`/`end` are seconds on the original audio timeline. `duration` is the full decoded source duration, including audio outside explicit regions.
- Canonical token equality and exact source-span coverage are **structural checks**, not evidence that every supplied word was spoken. `diagnostics.acoustic_accuracy_verified` is always false. A mismatched transcript can still have structurally valid timings.

Standalone inference reads raw model timestamp slots before upstream's nearest-neighbor/interpolation repair. `timestamp_repair` is `none`. Negative, nonfinite, reversed intervals, backward starts/ends, and overlaps exceeding one 80 ms bin are rejected. A forward-moving overlap of at most one bin is preserved verbatim and listed in `overlapping_word_indices`; this accounts for the model's coarse resolution without inventing new times. Windows may overshoot their audio end by at most one bin plus one sample; only this boundary excess is shortened to the window end and recorded in `boundary_adjustments`. Larger excess is rejected.

Zero-duration words are retained only in verbose JSON, marked `timing_status: "zero_duration"`, and listed in `zero_duration_word_indices`. They are unresolved timings, not high-quality alignment. Warning diagnostics set provenance `alignment` to `applied_with_warnings`. SRT/VTT export rejects zero-duration words and any cue that collapses to zero duration at millisecond resolution. Literal `&`, `<`, and `>` use subtitle HTML character escapes to retain their displayed text. Subtitle exports also reject blank cue lines and literal `-->` in transcript content because they cannot preserve those characters as ordinary cue text; use verbose JSON for such transcripts.

HTTP errors: 400 for malformed input/limits, 413 for oversized uploads, 422 for unverifiable model timing/coverage or zero-duration subtitle export, 503 for an unavailable/failing aligner, and 429 with `Retry-After: 1` for admission saturation. Supplied transcript contents are not logged by the service. A queued cancellation leaves the inference queue; active native MLX work finishes before its files are removed and the cancellation is propagated. Between regions, cancellation prevents further model calls.

Run inference sequentially across separate service/CLI processes. The shared lock serializes ASR and alignment within one process, and both alignment routes share the existing admission and metrics boundary. Automatic anchoring of arbitrary long untimed transcripts remains a future extension.
