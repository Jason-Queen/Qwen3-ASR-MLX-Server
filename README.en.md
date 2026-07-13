# Qwen3-ASR-MLX-Server

English | [中文](./README.md)

A macOS-focused local FastAPI service based on `mlx-audio` and Qwen3-ASR (MLX), providing local Whisper / OpenAI-compatible transcription endpoints.

Compatible with transcription apps such as MacWhisper, filling in common response structures such as `verbose_json`, `segments`, `words`, word timestamps, and language fields, while handling MP3 / MP4 / M4A input transcoding.

Ready for use with Spokenly, OpenWhispr, and other OpenAI-compatible transcription clients. In local tests, Mandarin dictation is very fast, with quality close to Alibaba Cloud's official transcription service.

## Features

- Whisper-compatible transcription endpoints
  - `POST /v1/audio/transcriptions`
  - `POST /audio/transcriptions` (alias)
- Chat-compatible audio transcription endpoints
  - `POST /v1/chat/completions`
  - `POST /chat/completions` (alias)
- Health/models endpoints
  - `GET /healthz`
  - `GET /v1/models`
- Accepts multipart audio and JSON `audio_file` payloads (base64/buffer/list variants)
- Optional forced alignment (word timestamps) with Qwen3-ForcedAligner
- Terminal-based interactive setup flow

## Model Policy

This project only supports Qwen3-ASR MLX model naming (full precision + quantized variants), for example:

- `qwen3-asr-mlx`
- `qwen3-asr-1.7b-bf16`
- `qwen3-asr-1.7b-4bit`
- `mlx-community/Qwen3-ASR-1.7B-8bit`

If model files are missing, API requests fail with a clear setup hint. The service never auto-downloads models during requests.

## Usage

### 1) Clone the Repository

```bash
git clone https://github.com/Jason-Queen/Qwen3-ASR-MLX-Server.git
cd Qwen3-ASR-MLX-Server
```

### 2) Install

#### Default: uv

```bash
uv sync --python 3.11
```

If Python 3.11 is not installed yet, run:

```bash
uv python install 3.11
uv sync --python 3.11
```

#### Compatible: Conda

```bash
conda env create -f environment.yml
conda activate qwen3-asr-whisper
```

#### Compatible: venv / pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
```

### 3) Setup Models (required before transcription)

Interactive setup (choose Hugging Face or ModelScope):

```bash
uv run --python 3.11 qwen3_asr_mlx_server.py setup
```

Non-interactive setup example:

```bash
uv run --python 3.11 qwen3_asr_mlx_server.py setup --non-interactive --source huggingface
```

Choose your preferred model variant (interactive setup):

- During setup, enter your own `Repository id` (bf16 or quantized repo).
- You can also choose a custom local directory.
- The server loads the model files from the local directory you provide.

Common repository examples (Hugging Face):

- `mlx-community/Qwen3-ASR-1.7B-bf16`
- `mlx-community/Qwen3-ASR-1.7B-4bit`
- `mlx-community/Qwen3-ASR-1.7B-8bit`

Download the Aligner proactively (word timestamps):

- Run `uv run --python 3.11 qwen3_asr_mlx_server.py setup` and choose `y` for `Qwen3-ForcedAligner-0.6B-bf16`.
- You can run setup again later to download only the missing aligner.
- Use `--source huggingface` or `--source modelscope` to select download source.

### 4) Run Server

```bash
uv run --python 3.11 qwen3-asr-mlx-server
```

Default bind: `127.0.0.1:8989`

LAN mode (explicit opt-in):

```bash
QWEN_MLX_HOST=0.0.0.0 QWEN_MLX_PORT=8989 uv run --python 3.11 qwen3-asr-mlx-server
```

Security note:

- Default mode is localhost-only to avoid accidental exposure.
- Only enable `0.0.0.0` when you need LAN clients.
- If your machine has a public IP or port-forwarding enabled, do not expose this service directly to the Internet.

Load your own local model directories:

```bash
QWEN_MLX_MODEL_PATH=/path/to/your/qwen3-asr-model \
QWEN_MLX_ALIGNER_PATH=/path/to/your/qwen3-aligner-model \
uv run --python 3.11 qwen3-asr-mlx-server
```

If you use Conda or a `venv`, the existing startup command remains compatible:

```bash
python qwen3_asr_mlx_server.py
```

Notes:

- The request `model` field is used for compatibility and validation.
- The actual loaded model is determined by `QWEN_MLX_MODEL_PATH`.

### 5) Quick Test

```bash
curl -X POST 'http://127.0.0.1:8989/v1/audio/transcriptions' \
  -F "file=@/path/to/test.wav" \
  -F "model=qwen3-asr-1.7b-bf16" \
  -F "response_format=json"
```

### 6) Important Environment Variables

```bash
QWEN_MLX_MODEL_PATH=./Qwen3-ASR-1.7B-bf16
QWEN_MLX_ALIGNER_PATH=./Qwen3-ForcedAligner-0.6B-bf16
QWEN_MLX_MODEL_ID=qwen3-asr-mlx
QWEN_MLX_HOST=127.0.0.1
QWEN_MLX_PORT=8989
QWEN_MLX_MAX_UPLOAD_MB=100
QWEN_MLX_MAX_QUEUE_DEPTH=8
QWEN_MLX_ALIGNMENT_CHUNK_SECONDS=30
QWEN_MLX_ALIGNMENT_BOUNDARY_TOLERANCE_SECONDS=5
QWEN_MLX_AUTO_ALIGN_LANG_CODES=zh,en,ja,ko
QWEN_MLX_LOG_PROMPTS=0
```

Set `QWEN_MLX_LOG_PROMPTS=1` to explicitly enable request prompt preview logging; it is disabled by default. You can also enable it with a startup flag:

```bash
uv run --python 3.11 qwen3-asr-mlx-server --log-prompts
```

Chunk size tuning guidance:

- `30` (default, recommended): used as the target window for balanced timestamp quality and recognition stability; actual split points snap to nearby ASR segment ends within 5 seconds by default to avoid cutting through sentences.
- `15` (aggressive): potentially finer timestamps, but slower and more context-fragmented recognition.
- `60+`: faster, but usually coarser timestamps.
- `QWEN_MLX_ALIGNMENT_BOUNDARY_TOLERANCE_SECONDS=0` restores fixed-duration hard cuts.

## Notes

- `task=transcribe` is supported; `translate` is rejected.
- To avoid native MLX crashes under load, transcription and forced alignment inference stay serialized inside a single process. Admission allows one executing request plus `QWEN_MLX_MAX_QUEUE_DEPTH` in-flight waiters per process; excess requests receive `429` with `Retry-After: 1`.
- `/healthz` returns model, optional-aligner, and admission/queue readiness plus cumulative admission, inference-wait, decoded-audio memory, process peak-RSS, and cancellation metrics. It returns `503` when the required ASR model is not ready or admission is full; an unavailable optional aligner is reported without disabling text-only transcription.
- Plain `json` and `text` are text-only contracts. Explicit `alignment=on`, `word_timestamps=true`, or `timestamp_granularities` with either format returns `400` instead of performing alignment and hiding the result. Use `verbose_json` for JSON timestamps, or `srt`/`vtt` for subtitles. Chat Completions remains text-only and rejects timestamp output controls.
- Auto forced-alignment runs only for timestamp-capable outputs (`verbose_json`, `srt`, `vtt`).
- Auto forced-alignment for `zh/en/ja/ko` is runtime-gated by aligner/dependencies:
  - `zh/en`: aligner model must be ready
  - `ja`: aligner + `nagisa`
  - `ko`: aligner + `soynlp`
- In `verbose_json` responses:
  - `language` represents the dominant language, not the only language in the file
  - `primary_language` is an explicit alias of `language`
  - `segments[].language` represents per-segment language
  - `detected_languages[]` summarizes detected languages and durations from segments
  - `timestamp_provenance` identifies ASR, estimated, or forced-alignment segment timing; word timing source; and alignment status
- A request cancelled while waiting for the process-local inference lock leaves the wait queue. Native MLX work that has already started is allowed to finish before temporary audio is deleted; the request still completes as cancelled and is counted by phase.

## Warning

Review the source code carefully and validate it in your own environment before relying on it in production.

## Acknowledgements

Thanks to Alibaba, Apple's MLX project, MLX-Community, and the broader open-source ecosystem.

## Copyright and License

- Copyright (c) 2026 Qwen3-ASR-MLX-Server contributors
- Recommended and applied license for this repository: Apache-2.0 (see `LICENSE`)
- Third-party dependencies keep their own licenses.
- Model weights/tokenizers from Hugging Face or ModelScope keep their original upstream licenses and usage terms. You must comply with those terms separately.
