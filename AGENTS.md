# AGENTS.md

## Surface And Scope

This repository is a macOS-focused local transcription server built on Qwen3-ASR and MLX. Start a fresh clone with the primary setup path below. `README.md` and `README.en.md` contain the detailed Chinese and English user guides.

Default to `uv`, Python 3.11, and `127.0.0.1:8989`. Treat the repository as standalone: do not assume the maintainer's worktrees, service manager, reverse proxy, or model paths exist.

## Repo Map

- `README.md` and `README.en.md` are the user-facing setup and API guides.
- `qwen3_asr_mlx_server.py` is the CLI and package entry point.
- `whisper_mlx_server.py` implements setup, model resolution, inference, and the HTTP API.
- `pyproject.toml`, `uv.lock`, and `Makefile` define the environment and common commands.
- `tests/` covers the API and inference boundaries. `scripts/smoke_local_api.sh` exercises a running server with real audio.

## Workflow And Handoffs

Before downloading weights, confirm the source, model variant, destination, and available disk space. Reuse an existing ASR model through `QWEN_MLX_MODEL_PATH`. Word timestamps need the optional aligner; set `QWEN_MLX_ALIGNER_PATH` when it is outside the auto-detected locations. Interactive setup can skip the aligner, while `--non-interactive` downloads every missing configured model, including the aligner.

For MP3, MP4, M4A, or non-PCM WAV input, check that `ffmpeg` is on `PATH`. Keep the first server start in the foreground. Deployment is complete only after `/healthz` is ready, `/v1/models` responds, and one real audio file transcribes successfully.

On failure, record the command, relevant `QWEN_MLX_*` values, HTTP status and body, resolved model path, and server error before proposing a cause. Use the model and queue components in `/healthz` to separate readiness from capacity failures.

For code changes, update tests, exercise the affected live endpoint, and review the diff. Record the purpose, material risk or tradeoff, deviations, and validation evidence in the handoff.

## Commands And Verification

Run the primary setup from the repository root:

```bash
uv sync --python 3.11
uv run --python 3.11 qwen3_asr_mlx_server.py setup
uv run --python 3.11 qwen3-asr-mlx-server
```

Interactive setup supports Hugging Face and ModelScope. For unattended setup, append `--non-interactive --source huggingface` or `--source modelscope`. In an activated Conda environment or `venv`, use `python qwen3_asr_mlx_server.py setup` and then `python qwen3_asr_mlx_server.py`.

Check the running service:

```bash
curl -sS -i http://127.0.0.1:8989/healthz
curl -fsS http://127.0.0.1:8989/v1/models
curl -fsS http://127.0.0.1:8989/v1/audio/transcriptions \
  -F "file=@/path/to/test.wav" \
  -F "model=qwen3-asr-mlx" \
  -F "response_format=json"
```

Ready means `/healthz` returns HTTP 200 with `live: true`, `ready: true`, and a ready required model. HTTP 503 means the model is unavailable or admission capacity is full; use the response body to distinguish them.

Run `make test` for every code change. After starting the affected checkout, run `make smoke`; set `PORT=<port>` when the server does not use 8989.

## Constraints And Boundaries

- Serving never downloads models. A request's `model` value is validated as a supported alias; `QWEN_MLX_MODEL_PATH` selects the files that are loaded.
- The default bind is `127.0.0.1:8989`, and the server has no authentication. Bind to `0.0.0.0` only when the user requests LAN access, and never expose the service directly to the Internet.
- Only `task=transcribe` is supported. `translate` is rejected.
- Use `verbose_json`, `srt`, or `vtt` for timestamp-bearing output. Plain `json`, `text`, and Chat Completions reject timestamp controls.
- Inference is serialized per process. Excess in-flight work returns HTTP 429 with `Retry-After: 1`; validate MLX memory use and concurrency before adding workers or parallel processes.
- When public behavior or dependencies change, keep `README.md`, `README.en.md`, tests, package/API versions, `pyproject.toml`, `uv.lock`, and smoke behavior in sync.
