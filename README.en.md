# Qwen3-ASR-MLX-Server

English | [中文](./README.md)

A macOS-focused local FastAPI service for Qwen3-ASR (MLX), with Whisper-compatible endpoints.

Ready for use with Spokenly, MacWhisper, and other OpenAI-compatible transcription clients. In local tests, Mandarin dictation is ultra-fast, and quality is close to Alibaba Cloud official transcription service.

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
- Explicit setup flow (no request-time auto download)

## Model Policy

This project only supports Qwen3-ASR MLX model naming (full precision + quantized variants), for example:

- `qwen3-asr-mlx`
- `qwen3-asr-1.7b-bf16`
- `qwen3-asr-1.7b-4bit`
- `mlx-community/Qwen3-ASR-1.7B-8bit`

If model files are missing, API requests fail with a clear setup hint. The service never auto-downloads models during requests.

## Usage

### 1) Install

#### Conda

```bash
conda env create -f environment.yml
conda activate qwen3-asr-whisper
```

#### venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
```

### 2) Setup Models (required before transcription)

Interactive setup (choose Hugging Face or ModelScope):

```bash
python qwen3_asr_mlx_server.py setup
```

Non-interactive setup example:

```bash
python qwen3_asr_mlx_server.py setup --non-interactive --source huggingface
```

### 3) Run Server

```bash
python qwen3_asr_mlx_server.py
```

Default bind: `127.0.0.1:8989`

LAN mode (explicit opt-in):

```bash
QWEN_MLX_HOST=0.0.0.0 QWEN_MLX_PORT=8989 python qwen3_asr_mlx_server.py
```

Security note:

- Default mode is localhost-only to avoid accidental exposure.
- Only enable `0.0.0.0` when you need LAN clients.
- If your machine has a public IP or port-forwarding enabled, do not expose this service directly to the Internet.

### 4) Quick Test

```bash
curl -X POST 'http://127.0.0.1:8989/v1/audio/transcriptions' \
  -F "file=@/path/to/test.wav" \
  -F "model=qwen3-asr-1.7b-bf16" \
  -F "response_format=json"
```

### 5) Important Environment Variables

```bash
QWEN_MLX_MODEL_PATH=./Qwen3-ASR-1.7B-bf16
QWEN_MLX_ALIGNER_PATH=./Qwen3-ForcedAligner-0.6B-bf16
QWEN_MLX_MODEL_ID=qwen3-asr-mlx
QWEN_MLX_HOST=127.0.0.1
QWEN_MLX_PORT=8989
QWEN_MLX_MAX_UPLOAD_MB=100
QWEN_MLX_ALIGNMENT_CHUNK_SECONDS=30
QWEN_MLX_AUTO_ALIGN_LANG_CODES=zh,en,ja,ko
```

Chunk size tuning guidance:

- `30` (default, recommended): balanced timestamp quality and recognition stability.
- `15` (aggressive): potentially finer timestamps, but slower and more context-fragmented recognition.
- `60+`: faster, but usually coarser timestamps.

## Notes

- `task=transcribe` is supported; `translate` is rejected.
- Auto forced-alignment for `zh/en/ja/ko` is runtime-gated by aligner/dependencies:
  - `zh/en`: aligner model must be ready
  - `ja`: aligner + `nagisa`
  - `ko`: aligner + `soynlp`

## Warning

Review the source code carefully and validate it in your own environment before relying on it in production.

## Acknowledgements

Thanks to Alibaba, Apple's MLX project, MLX-Community, and the broader open-source ecosystem.

## Copyright and License

- Copyright (c) 2026 Qwen3-ASR-MLX-Server contributors
- Recommended and applied license for this repository: Apache-2.0 (see `LICENSE`)
- Third-party dependencies keep their own licenses.
- Model weights/tokenizers from Hugging Face or ModelScope keep their original upstream licenses and usage terms. You must comply with those terms separately.
