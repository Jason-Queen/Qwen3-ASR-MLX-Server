from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import contextvars
import importlib.util
import io
import inspect
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

try:
    import resource
except ImportError:  # pragma: no cover - resource is available on supported macOS/Linux hosts
    resource = None

try:
    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model
except Exception as exc:  # pragma: no cover
    generate_transcription = None
    load_model = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HF_ASR_REPO_ID = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_HF_ALIGNER_REPO_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"
SETUP_COMMAND_HINT = "python qwen3_asr_mlx_server.py setup"


def _workspace_root(base_dir: Path | None = None) -> Path | None:
    root = base_dir or BASE_DIR
    parent = root.parent
    if (parent / "README_WORKSPACE.md").exists():
        return parent
    if (parent / "qwen3-asr-whisper-api-debug").exists() and (
        parent / "qwen3-asr-whisper-api-github"
    ).exists():
        return parent
    return None


def _huggingface_hub_cache_dir() -> Path:
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)

    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "huggingface" / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def _huggingface_snapshot_dirs(repo_id: str) -> list[Path]:
    repo_cache_dir = _huggingface_hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    candidates: list[Path] = []

    ref_main = repo_cache_dir / "refs" / "main"
    if ref_main.exists():
        try:
            revision = ref_main.read_text(encoding="utf-8").strip()
        except OSError:
            revision = ""
        if revision:
            snapshot_dir = snapshots_dir / revision
            if snapshot_dir.exists():
                candidates.append(snapshot_dir)

    if snapshots_dir.exists():
        for path in sorted(
            (item for item in snapshots_dir.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            if path not in candidates:
                candidates.append(path)

    return candidates


def _candidate_asset_dirs(
    dirname: str,
    *,
    hf_repo: str | None = None,
    base_dir: Path | None = None,
) -> list[Path]:
    root = base_dir or BASE_DIR
    candidates: list[Path] = []
    if hf_repo:
        candidates.extend(_huggingface_snapshot_dirs(hf_repo))

    candidates.append(root / dirname)
    workspace_root = _workspace_root(root)
    if workspace_root is not None:
        candidates.append(workspace_root / "models" / dirname)
        candidates.append(workspace_root / dirname)
    return candidates


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _asset_dir_validation_error(path: Path) -> str | None:
    if not path.exists():
        return "path does not exist"
    if not path.is_dir():
        return "path is not a directory"

    missing_metadata = [
        name
        for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json")
        if not _is_nonempty_file(path / name)
    ]
    if missing_metadata:
        return f"missing required files: {', '.join(missing_metadata)}"

    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        if not _is_nonempty_file(index_path):
            return "model.safetensors.index.json is empty or unreadable"
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "model.safetensors.index.json is invalid"

        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return "model.safetensors.index.json has no weight_map entries"

        missing_shards = sorted(
            {
                shard_name
                for shard_name in (
                    str(candidate).strip() for candidate in weight_map.values()
                )
                if shard_name and not _is_nonempty_file(path / shard_name)
            }
        )
        if missing_shards:
            preview = ", ".join(missing_shards[:3])
            if len(missing_shards) > 3:
                preview = f"{preview}, ..."
            return f"missing model weight files: {preview}"
        return None

    if any(_is_nonempty_file(candidate) for candidate in path.glob("*.safetensors")):
        return None
    return "missing model weight files (*.safetensors)"


def _is_usable_asset_dir(path: Path) -> bool:
    return _asset_dir_validation_error(path) is None


def _default_asset_path(
    dirname: str,
    *,
    hf_repo: str | None = None,
    base_dir: Path | None = None,
) -> str:
    for path in _candidate_asset_dirs(dirname, hf_repo=hf_repo, base_dir=base_dir):
        if _is_usable_asset_dir(path):
            return str(path)

    workspace_root = _workspace_root(base_dir)
    if workspace_root is not None:
        return str(workspace_root / "models" / dirname)
    root = base_dir or BASE_DIR
    return str(root / dirname)


HF_ASR_REPO = os.getenv("QWEN_MLX_HF_ASR_REPO", DEFAULT_HF_ASR_REPO_ID)
MS_ASR_REPO = os.getenv("QWEN_MLX_MS_ASR_REPO", HF_ASR_REPO)
HF_ALIGNER_REPO = os.getenv("QWEN_MLX_HF_ALIGNER_REPO", DEFAULT_HF_ALIGNER_REPO_ID)
MS_ALIGNER_REPO = os.getenv("QWEN_MLX_MS_ALIGNER_REPO", HF_ALIGNER_REPO)
DEFAULT_MODEL_PATH = _default_asset_path("Qwen3-ASR-1.7B-bf16", hf_repo=HF_ASR_REPO)
MODEL_PATH = os.getenv("QWEN_MLX_MODEL_PATH", DEFAULT_MODEL_PATH)
MODEL_ID = os.getenv("QWEN_MLX_MODEL_ID", "qwen3-asr-mlx")
DEFAULT_ALIGNER_PATH = _default_asset_path(
    "Qwen3-ForcedAligner-0.6B-bf16",
    hf_repo=HF_ALIGNER_REPO,
)
ALIGNER_PATH = os.getenv("QWEN_MLX_ALIGNER_PATH", DEFAULT_ALIGNER_PATH)
ALIGNMENT_CHUNK_SECONDS = float(os.getenv("QWEN_MLX_ALIGNMENT_CHUNK_SECONDS", "30"))
ALIGNMENT_BOUNDARY_TOLERANCE_SECONDS = float(
    os.getenv("QWEN_MLX_ALIGNMENT_BOUNDARY_TOLERANCE_SECONDS", "5")
)
AUTO_ALIGN_LANG_CODES = {
    code.strip().lower()
    for code in os.getenv("QWEN_MLX_AUTO_ALIGN_LANG_CODES", "zh,en,ja,ko").split(",")
    if code.strip()
}
SEGMENT_REBUILD_FROM_WORDS = (
    os.getenv("QWEN_MLX_SEGMENT_REBUILD_FROM_WORDS", "1").strip().lower() not in {"0", "false", "no", "off"}
)
SEGMENT_MERGE_TARGET_SECONDS = float(os.getenv("QWEN_MLX_SEGMENT_MERGE_TARGET_SECONDS", "0"))
SEGMENT_HARD_MAX_SECONDS = float(os.getenv("QWEN_MLX_SEGMENT_HARD_MAX_SECONDS", "12"))
HOST = os.getenv("QWEN_MLX_HOST", "127.0.0.1")
PORT = int(os.getenv("QWEN_MLX_PORT", "8989"))
MAX_UPLOAD_MB = int(os.getenv("QWEN_MLX_MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_QUEUE_DEPTH = max(0, int(os.getenv("QWEN_MLX_MAX_QUEUE_DEPTH", "8")))
LOG_LEVEL = os.getenv("QWEN_MLX_LOG_LEVEL", "INFO").upper()
LOG_PROMPTS = os.getenv("QWEN_MLX_LOG_PROMPTS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
    "enable",
    "enabled",
}

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("qwen3_asr_api")

SUPPORTED_RESPONSE_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
AUTO_ALIGNMENT_RESPONSE_FORMATS = {"verbose_json", "srt", "vtt"}
SUPPORTED_TIMESTAMP_GRANULARITIES = {"word", "segment"}
MIME_TO_SUFFIX = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}

LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}
LANGUAGE_NAME_TO_CODE = {name.lower(): code for code, name in LANGUAGE_CODE_TO_NAME.items()}
DECODE_OPTION_TYPES: dict[str, str] = {
    "max_tokens": "int",
    "temperature": "float",
    "top_p": "float",
    "top_k": "int",
    "min_p": "float",
    "min_tokens_to_keep": "int",
    "repetition_penalty": "float",
    "repetition_context_size": "int",
    "prefill_step_size": "int",
    "chunk_duration": "float",
    "min_chunk_duration": "float",
}

EXPLICIT_MODEL_ALIASES = {
    MODEL_ID.lower(),
    "qwen3-asr",
    "qwen3-asr-mlx",
}
for alias in os.getenv("QWEN_MLX_EXTRA_MODEL_ALIASES", "").split(","):
    if alias.strip():
        EXPLICIT_MODEL_ALIASES.add(alias.strip().lower())

QWEN3_ASR_MLX_VARIANTS = {
    "bf16",
    "fp16",
    "4bit",
    "5bit",
    "6bit",
    "8bit",
}
for item in os.getenv("QWEN_MLX_ALLOWED_VARIANTS", "").split(","):
    if item.strip():
        QWEN3_ASR_MLX_VARIANTS.add(item.strip().lower())


@dataclass(frozen=True)
class SetupModelSpec:
    key: str
    display_name: str
    local_path: str
    hf_repo: str
    ms_repo: str
    required: bool = False


@dataclass
class TranscriptionResult:
    text: str
    language: str | None
    segments: list[dict[str, Any]]
    duration: float | None
    words: list[dict[str, Any]] = field(default_factory=list)
    segment_timestamp_source: str = "asr"
    word_timestamp_source: str | None = None
    alignment_status: str = "not_requested"


@dataclass
class AlignmentChunk:
    text: str
    canonical_tokens: list[str]
    words: list[dict[str, Any]]
    start: float
    end: float


@dataclass
class AlignmentOutput:
    words: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[AlignmentChunk] = field(default_factory=list)


@dataclass(frozen=True)
class TextPart:
    start: int
    end: int
    text: str


_CURRENT_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "qwen_mlx_request_id",
    default=None,
)
_CURRENT_CANCELLATION_EVENT: contextvars.ContextVar[Event | None] = contextvars.ContextVar(
    "qwen_mlx_cancellation_event",
    default=None,
)


class InferenceWaitCancelled(RuntimeError):
    """Raised inside a worker thread when its queued request was cancelled."""


def _process_peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    # Linux reports KiB; macOS reports bytes.
    if sys.platform.startswith("linux"):
        value *= 1024
    return max(0, value)


class InstrumentedInferenceLock:
    """Process-local inference lock with observable, cancellable waiters."""

    def __init__(self, poll_interval_seconds: float = 0.05) -> None:
        self._lock = Lock()
        self._state_lock = Lock()
        self._poll_interval_seconds = max(0.005, poll_interval_seconds)
        self._waiting = 0
        self._active = 0
        self._peak_waiting = 0
        self._acquisitions_total = 0
        self._waited_acquisitions_total = 0
        self._completed_total = 0
        self._cancelled_waiters_total = 0
        self._wait_seconds_total = 0.0
        self._wait_seconds_max = 0.0
        self._waiting_request_ids: set[str] = set()
        self._active_request_id: str | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        request_id = _CURRENT_REQUEST_ID.get()
        cancellation_event = _CURRENT_CANCELLATION_EVENT.get()
        started = time.perf_counter()

        if cancellation_event is not None and cancellation_event.is_set():
            raise InferenceWaitCancelled("request cancelled before inference admission")

        if not blocking:
            acquired = self._lock.acquire(blocking=False)
            if acquired:
                self._record_acquired(request_id, 0.0)
            return acquired

        # Do not count an uncontended acquisition as a queued waiter.
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            if cancellation_event is not None and cancellation_event.is_set():
                self._lock.release()
                raise InferenceWaitCancelled("request cancelled before inference started")
            self._record_acquired(request_id, 0.0)
            return True

        if timeout == 0:
            return False

        with self._state_lock:
            self._waiting += 1
            self._peak_waiting = max(self._peak_waiting, self._waiting)
            if request_id:
                self._waiting_request_ids.add(request_id)

        acquired = False
        deadline = None if timeout is None or timeout < 0 else started + timeout
        cancelled = False
        try:
            while True:
                if cancellation_event is not None and cancellation_event.is_set():
                    cancelled = True
                    raise InferenceWaitCancelled("request cancelled while waiting for inference")

                if cancellation_event is None and deadline is None:
                    acquired = self._lock.acquire()
                else:
                    interval = self._poll_interval_seconds
                    if deadline is not None:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            return False
                        interval = min(interval, remaining)
                    acquired = self._lock.acquire(timeout=interval)

                if acquired:
                    if cancellation_event is not None and cancellation_event.is_set():
                        self._lock.release()
                        acquired = False
                        cancelled = True
                        raise InferenceWaitCancelled(
                            "request cancelled while acquiring inference"
                        )
                    break
        finally:
            with self._state_lock:
                self._waiting = max(0, self._waiting - 1)
                if request_id:
                    self._waiting_request_ids.discard(request_id)
                if cancelled:
                    self._cancelled_waiters_total += 1

        if acquired and cancellation_event is not None and cancellation_event.is_set():
            self._lock.release()
            with self._state_lock:
                self._cancelled_waiters_total += 1
            raise InferenceWaitCancelled("request cancelled before inference started")

        if acquired:
            self._record_acquired(request_id, time.perf_counter() - started)
        return acquired

    def _record_acquired(self, request_id: str | None, wait_seconds: float) -> None:
        with self._state_lock:
            self._active = 1
            self._active_request_id = request_id
            self._acquisitions_total += 1
            if wait_seconds > 0:
                self._waited_acquisitions_total += 1
            self._wait_seconds_total += wait_seconds
            self._wait_seconds_max = max(self._wait_seconds_max, wait_seconds)

    def release(self) -> None:
        with self._state_lock:
            self._lock.release()
            self._active = 0
            self._active_request_id = None
            self._completed_total += 1

    def locked(self) -> bool:
        return self._lock.locked()

    def phase_for(self, request_id: str) -> str | None:
        with self._state_lock:
            if request_id == self._active_request_id:
                return "running"
            if request_id in self._waiting_request_ids:
                return "queued"
        return None

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            acquisitions = self._acquisitions_total
            waited_acquisitions = self._waited_acquisitions_total
            wait_total = self._wait_seconds_total
            return {
                "active": self._active,
                "waiting": self._waiting,
                "peak_waiting": self._peak_waiting,
                "acquisitions_total": acquisitions,
                "waited_acquisitions_total": waited_acquisitions,
                "completed_total": self._completed_total,
                "cancelled_waiters_total": self._cancelled_waiters_total,
                "wait_seconds_total": round(wait_total, 6),
                "wait_seconds_max": round(self._wait_seconds_max, 6),
                "wait_seconds_average": round(wait_total / waited_acquisitions, 6)
                if waited_acquisitions
                else 0.0,
            }

    def __enter__(self) -> InstrumentedInferenceLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> None:
        self.release()


class AdmissionLease:
    def __init__(self, metrics: RuntimeMetrics, request_id: str) -> None:
        self._metrics = metrics
        self.request_id = request_id
        self.audio_bytes = 0
        self._finished = False

    def set_audio_bytes(self, value: int) -> None:
        self._metrics.set_audio_bytes(self, value)

    def finish(self, outcome: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._metrics.finish(self, outcome)


class RuntimeMetrics:
    """Thread-safe per-process request, memory, and cancellation counters."""

    def __init__(self, max_queue_depth: int) -> None:
        self.max_queue_depth = max(0, max_queue_depth)
        self.capacity = self.max_queue_depth + 1
        self._lock = Lock()
        self._attempts_total = 0
        self._admitted_total = 0
        self._rejected_total = 0
        self._in_flight = 0
        self._peak_in_flight = 0
        self._completed_total = 0
        self._failed_total = 0
        self._cancelled_total = 0
        self._active_audio_bytes = 0
        self._peak_active_audio_bytes = 0
        self._decoded_audio_bytes_total = 0
        self._largest_request_audio_bytes = 0
        self._cancellation_phases = {
            "request": 0,
            "processing": 0,
            "queued": 0,
            "running": 0,
        }
        self._pending_cancellation_phases: dict[str, str] = {}
        self._client_disconnect_total = 0

    def try_admit(self, request_id: str) -> AdmissionLease | None:
        with self._lock:
            self._attempts_total += 1
            if self._in_flight >= self.capacity:
                self._rejected_total += 1
                return None
            self._admitted_total += 1
            self._in_flight += 1
            self._peak_in_flight = max(self._peak_in_flight, self._in_flight)
        return AdmissionLease(self, request_id)

    def set_audio_bytes(self, lease: AdmissionLease, value: int) -> None:
        normalized = max(0, int(value))
        with self._lock:
            delta = normalized - lease.audio_bytes
            lease.audio_bytes = normalized
            self._active_audio_bytes = max(0, self._active_audio_bytes + delta)
            if delta > 0:
                self._decoded_audio_bytes_total += delta
            self._peak_active_audio_bytes = max(
                self._peak_active_audio_bytes,
                self._active_audio_bytes,
            )
            self._largest_request_audio_bytes = max(
                self._largest_request_audio_bytes,
                normalized,
            )

    def note_cancellation_phase(self, request_id: str, phase: str) -> None:
        normalized = phase if phase in self._cancellation_phases else "processing"
        with self._lock:
            self._pending_cancellation_phases.setdefault(request_id, normalized)

    def note_client_disconnect(self) -> None:
        with self._lock:
            self._client_disconnect_total += 1

    def finish(self, lease: AdmissionLease, outcome: str) -> None:
        with self._lock:
            self._active_audio_bytes = max(0, self._active_audio_bytes - lease.audio_bytes)
            self._in_flight = max(0, self._in_flight - 1)
            if outcome == "completed":
                self._completed_total += 1
            elif outcome == "cancelled":
                self._cancelled_total += 1
                phase = self._pending_cancellation_phases.pop(lease.request_id, "request")
                self._cancellation_phases[phase] += 1
            else:
                self._failed_total += 1
                self._pending_cancellation_phases.pop(lease.request_id, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "admission": {
                    "capacity": self.capacity,
                    "max_queue_depth": self.max_queue_depth,
                    "in_flight": self._in_flight,
                    "peak_in_flight": self._peak_in_flight,
                    "attempts_total": self._attempts_total,
                    "admitted_total": self._admitted_total,
                    "rejected_total": self._rejected_total,
                    "completed_total": self._completed_total,
                    "failed_total": self._failed_total,
                },
                "memory": {
                    "active_audio_bytes": self._active_audio_bytes,
                    "peak_active_audio_bytes": self._peak_active_audio_bytes,
                    "decoded_audio_bytes_total": self._decoded_audio_bytes_total,
                    "largest_request_audio_bytes": self._largest_request_audio_bytes,
                    "process_peak_rss_bytes": _process_peak_rss_bytes(),
                },
                "cancellation": {
                    "cancelled_requests_total": self._cancelled_total,
                    "request_total": self._cancellation_phases["request"],
                    "processing_total": self._cancellation_phases["processing"],
                    "queued_total": self._cancellation_phases["queued"],
                    "running_total": self._cancellation_phases["running"],
                    "client_disconnect_total": self._client_disconnect_total,
                },
            }


_GLOBAL_INFERENCE_LOCK = InstrumentedInferenceLock()
runtime_metrics = RuntimeMetrics(MAX_QUEUE_DEPTH)


class MLXTranscriber:
    def __init__(self, model_path: str, infer_lock: Lock | None = None) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self._load_lock = Lock()
        # MLX inference must stay process-serial across ASR and alignment.
        self._infer_lock = infer_lock or _GLOBAL_INFERENCE_LOCK
        self._load_error: str | None = None

    def load(self, strict: bool = True) -> None:
        if self._model is not None:
            return

        if IMPORT_ERROR is not None:
            self._load_error = "mlx-audio import failed"
            if strict:
                raise RuntimeError(
                    "mlx-audio import failed. Install project dependencies first "
                    "(recommended: uv sync --python 3.11; alternatives: pip install -r "
                    "requirements.txt or conda env create -f environment.yml)."
                ) from IMPORT_ERROR
            return

        with self._load_lock:
            if self._model is None:
                model_dir = Path(self.model_path)
                validation_error = _asset_dir_validation_error(model_dir)
                if validation_error is not None:
                    self._load_error = (
                        f"invalid model directory '{self.model_path}': {validation_error}"
                    )
                    if strict:
                        raise FileNotFoundError(self._load_error)
                    return
                try:
                    self._model = load_model(str(model_dir))
                    self._load_error = None
                except Exception as exc:
                    self._load_error = str(exc)
                    if strict:
                        raise

    def is_ready(self) -> bool:
        self.load(strict=False)
        return self._model is not None

    def is_available(self) -> bool:
        return _is_usable_asset_dir(Path(self.model_path))

    def last_error(self) -> str | None:
        return self._load_error

    def readiness_snapshot(self) -> dict[str, Any]:
        validation_error = _asset_dir_validation_error(Path(self.model_path))
        ready = self._model is not None
        return {
            "ready": ready,
            "loaded": ready,
            "available": validation_error is None,
            "error": None if ready else self._load_error or validation_error or "not loaded",
        }

    def transcribe(
        self,
        audio_path: str,
        language: str | None,
        prompt: str | None,
        decode_options: dict[str, Any] | None = None,
    ) -> TranscriptionResult:
        self.load(strict=True)

        with self._infer_lock:
            # Use a basename; mlx-audio appends format extension (e.g. .txt).
            with tempfile.NamedTemporaryFile(suffix="", delete=False) as output_file:
                output_base = output_file.name

            try:
                kwargs: dict[str, Any] = {}
                signature = inspect.signature(generate_transcription)
                params = signature.parameters
                has_var_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
                )

                candidates: dict[str, Any] = {
                    "model": self._model,
                    "audio": audio_path,
                    "audio_path": audio_path,
                    "output_path": output_base,
                    "format": "txt",
                    "verbose": False,
                    "language": None if language in {None, "", "auto"} else language,
                    "prompt": prompt,
                    "context": prompt,
                    "system_prompt": prompt,
                }
                if decode_options:
                    candidates.update(decode_options)

                for key, value in candidates.items():
                    if value is None:
                        continue
                    if key in params or has_var_kwargs:
                        kwargs[key] = value

                logger.info(
                    "Calling generate_transcription with keys=%s audio=%s",
                    sorted(kwargs.keys()),
                    audio_path,
                )
                raw_result = generate_transcription(**kwargs)
                return _normalize_result(raw_result, f"{output_base}.txt", language)
            finally:
                for suffix in ("", ".txt", ".srt", ".vtt", ".json"):
                    try:
                        Path(f"{output_base}{suffix}").unlink(missing_ok=True)
                    except OSError:
                        pass


class MLXAligner:
    def __init__(self, model_path: str, infer_lock: Lock | None = None) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self._load_lock = Lock()
        self._processor_lock = Lock()
        # Share the same inference lock as the transcriber to avoid native MLX crashes.
        self._infer_lock = infer_lock or _GLOBAL_INFERENCE_LOCK
        self._load_error: str | None = None
        self._supported_languages_cache: set[str] | None = None

    def load(self, strict: bool = False) -> None:
        if self._model is not None:
            return

        if IMPORT_ERROR is not None:
            self._load_error = "mlx-audio import failed"
            if strict:
                raise RuntimeError(
                    "mlx-audio import failed. Install project dependencies first "
                    "(recommended: uv sync --python 3.11; alternatives: pip install -r "
                    "requirements.txt or conda env create -f environment.yml)."
                ) from IMPORT_ERROR
            return

        with self._load_lock:
            if self._model is not None:
                return
            model_dir = Path(self.model_path)
            validation_error = _asset_dir_validation_error(model_dir)
            if validation_error is not None:
                self._load_error = (
                    f"invalid aligner model directory '{self.model_path}': {validation_error}"
                )
                if strict:
                    raise FileNotFoundError(self._load_error)
                return
            try:
                self._model = load_model(str(model_dir))
                self._load_error = None
                self._supported_languages_cache = None
            except Exception as exc:
                self._load_error = str(exc)
                if strict:
                    raise

    def is_ready(self) -> bool:
        self.load(strict=False)
        return self._model is not None

    def last_error(self) -> str | None:
        return self._load_error

    def readiness_snapshot(self) -> dict[str, Any]:
        validation_error = _asset_dir_validation_error(Path(self.model_path))
        ready = self._model is not None
        return {
            "ready": ready,
            "loaded": ready,
            "available": validation_error is None,
            "error": None if ready else self._load_error or validation_error or "not loaded",
        }

    def supported_languages(self) -> set[str] | None:
        self.load(strict=False)
        if self._model is None:
            return None
        if self._supported_languages_cache is not None:
            return self._supported_languages_cache

        getter = getattr(self._model, "get_supported_languages", None)
        if not callable(getter):
            self._supported_languages_cache = set()
            return self._supported_languages_cache
        try:
            values = getter()
        except Exception:
            self._supported_languages_cache = set()
            return self._supported_languages_cache
        if not values:
            self._supported_languages_cache = set()
            return self._supported_languages_cache

        normalized = {str(item).strip().lower() for item in values if str(item).strip()}
        self._supported_languages_cache = normalized
        return self._supported_languages_cache

    def canonical_tokens(self, text: str, language: str) -> list[str]:
        if not text.strip():
            return []

        self.load(strict=False)
        if self._model is None:
            return []

        processor = getattr(self._model, "aligner_processor", None)
        encode_timestamp = getattr(processor, "encode_timestamp", None)
        if not callable(encode_timestamp):
            raise RuntimeError("forced aligner does not expose encode_timestamp")

        # The upstream processor lazily initializes language tokenizers. Serialize
        # access without counting CPU-only tokenization as native MLX inference.
        with self._processor_lock:
            word_list, _input_text = encode_timestamp(text, language)
        return [str(item) for item in word_list]

    def align(
        self,
        audio: str,
        text: str,
        language: str,
    ) -> list[dict[str, Any]]:
        if not text.strip():
            return []

        self.load(strict=False)
        if self._model is None:
            return []

        with self._infer_lock:
            with self._processor_lock:
                raw_result = self._model.generate(audio=audio, text=text, language=language)
        return _normalize_word_segments(raw_result)


SETUP_MODEL_SPECS = (
    SetupModelSpec(
        key="asr",
        display_name="Qwen3-ASR-1.7B-bf16 (MLX)",
        local_path=MODEL_PATH,
        hf_repo=HF_ASR_REPO,
        ms_repo=MS_ASR_REPO,
        required=True,
    ),
    SetupModelSpec(
        key="aligner",
        display_name="Qwen3-ForcedAligner-0.6B-bf16 (MLX)",
        local_path=ALIGNER_PATH,
        hf_repo=HF_ALIGNER_REPO,
        ms_repo=MS_ALIGNER_REPO,
        required=False,
    ),
)
SETUP_MODEL_SPEC_BY_KEY = {spec.key: spec for spec in SETUP_MODEL_SPECS}


def _is_interactive_terminal() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _missing_setup_models() -> list[SetupModelSpec]:
    missing: list[SetupModelSpec] = []
    for spec in SETUP_MODEL_SPECS:
        if not _is_usable_asset_dir(Path(spec.local_path)):
            missing.append(spec)
    return missing


def _prompt_text(prompt: str, default: str | None = None) -> str:
    if default is None:
        raw = input(f"{prompt}: ").strip()
        return raw
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def _prompt_yes_no(prompt: str, default_yes: bool = True) -> bool:
    hint = "Y/n" if default_yes else "y/N"
    raw = input(f"{prompt} ({hint}): ").strip().lower()
    if not raw:
        return default_yes
    return raw in {"y", "yes"}


def _manual_model_setup_help(env_var: str) -> str:
    return (
        f"Set {env_var} to an existing model directory, "
        f"or download it manually via: {SETUP_COMMAND_HINT}"
    )


def _prompt_download_source() -> str:
    while True:
        choice = input(
            "Select download source: [1] Hugging Face, [2] ModelScope (default: 1): "
        ).strip()
        if choice in {"", "1"}:
            return "huggingface"
        if choice == "2":
            return "modelscope"
        print("Invalid selection. Please enter 1 or 2.")


def _download_from_huggingface(repo_id: str, local_path: str) -> None:
    from huggingface_hub import snapshot_download

    target = Path(local_path)
    target.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {"repo_id": repo_id, "local_dir": str(target)}
    signature = inspect.signature(snapshot_download).parameters
    if "local_dir_use_symlinks" in signature:
        kwargs["local_dir_use_symlinks"] = False
    if "resume_download" in signature:
        kwargs["resume_download"] = True
    snapshot_download(**kwargs)


def _download_from_modelscope(repo_id: str, local_path: str) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "ModelScope downloader is unavailable. Install it first: pip install modelscope"
        ) from exc

    target = Path(local_path)
    target.mkdir(parents=True, exist_ok=True)

    signature = inspect.signature(ms_snapshot_download).parameters
    kwargs: dict[str, Any] = {}
    uses_local_dir = False
    if "local_dir" in signature:
        kwargs["local_dir"] = str(target)
        uses_local_dir = True
    elif "cache_dir" in signature:
        kwargs["cache_dir"] = str(target.parent)
    if "revision" in signature and os.getenv("QWEN_MLX_MS_REVISION"):
        kwargs["revision"] = os.getenv("QWEN_MLX_MS_REVISION")

    if "model_id" in signature:
        downloaded_path = ms_snapshot_download(model_id=repo_id, **kwargs)
    else:
        downloaded_path = ms_snapshot_download(repo_id, **kwargs)

    if uses_local_dir:
        return

    source = Path(str(downloaded_path))
    if source.exists() and source.resolve() != target.resolve():
        shutil.copytree(source, target, dirs_exist_ok=True)


def _download_model_snapshot(source: str, repo_id: str, local_path: str) -> None:
    logger.info("Downloading model source=%s repo_id=%s target=%s", source, repo_id, local_path)
    if source == "huggingface":
        _download_from_huggingface(repo_id, local_path)
        return
    if source == "modelscope":
        _download_from_modelscope(repo_id, local_path)
        return
    raise RuntimeError(f"Unsupported source: {source}")


def run_setup_wizard(
    *,
    source: str | None = None,
    non_interactive: bool = False,
) -> int:
    missing = _missing_setup_models()
    if not missing:
        print("All configured Qwen3 models already exist locally.")
        return 0

    chosen_source = source or _prompt_download_source()
    if chosen_source not in {"huggingface", "modelscope"}:
        print("Invalid source. Use 'huggingface' or 'modelscope'.")
        return 2

    if non_interactive:
        selected_specs = missing
    else:
        print("\nMissing models detected:")
        for spec in missing:
            required = "required" if spec.required else "optional"
            print(f"- {spec.display_name} ({required}) -> {spec.local_path}")
        print("")
        selected_specs = []
        for spec in missing:
            default_yes = spec.required
            if _prompt_yes_no(f"Download {spec.display_name} now?", default_yes=default_yes):
                selected_specs.append(spec)

    if not selected_specs:
        print("No models selected. Setup skipped.")
        return 1

    failed = False
    for spec in selected_specs:
        default_repo = spec.hf_repo if chosen_source == "huggingface" else spec.ms_repo
        repo_id = default_repo
        local_path = spec.local_path
        if not non_interactive:
            repo_id = _prompt_text(
                f"Repository id for {spec.display_name}",
                default=default_repo,
            ).strip()
            local_path = _prompt_text(
                f"Local directory for {spec.display_name}",
                default=spec.local_path,
            ).strip()
        if not repo_id:
            print(f"Skipped {spec.display_name}: empty repository id.")
            failed = True
            continue
        try:
            _download_model_snapshot(chosen_source, repo_id, local_path)
        except Exception as exc:
            failed = True
            print(f"Failed to download {spec.display_name}: {exc}")
        else:
            print(f"Downloaded {spec.display_name} -> {local_path}")

    still_missing = _missing_setup_models()
    if any(spec.required for spec in still_missing):
        print(
            "\nRequired ASR model is still missing. "
            "Server can start, but transcription requests will return an error until setup is completed."
        )
        return 1 if not failed else 2
    return 0 if not failed else 1


def maybe_prompt_initial_setup() -> None:
    validation_error = _asset_dir_validation_error(Path(MODEL_PATH))
    if validation_error is None:
        return

    logger.warning(
        "ASR model unavailable at %s: %s. %s",
        MODEL_PATH,
        validation_error,
        _manual_model_setup_help("QWEN_MLX_MODEL_PATH"),
    )
    if not _is_interactive_terminal():
        return
    print(
        "\nQwen3 ASR model is not available.\n"
        f"Resolved path: {MODEL_PATH}\n"
        f"Reason: {validation_error}\n"
        "The server checks the Hugging Face cache first, then repo/workspace-local model directories.\n"
        f"{_manual_model_setup_help('QWEN_MLX_MODEL_PATH')}\n"
        "Automatic download during normal serving is disabled.\n"
    )


def _normalize_requested_model_name(model_name: Any) -> str:
    if model_name is None:
        return MODEL_ID
    text = str(model_name).strip()
    return text or MODEL_ID


def _is_allowed_qwen3_asr_mlx_name(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    if not normalized:
        return False
    if normalized in EXPLICIT_MODEL_ALIASES:
        return True

    # Accept optional namespace prefixes like mlx-community/Qwen3-ASR-1.7B-8bit.
    slug = normalized.split("/")[-1]
    slug = slug.replace("_", "-")
    slug = re.sub(r"-+", "-", slug)
    if slug in {"qwen3-asr-1.7b", "qwen3-asr-1-7b"}:
        return True

    match = re.fullmatch(
        r"qwen3-asr-1(?:[.\-_]?7)b(?:-([a-z0-9]+))?",
        slug,
    )
    if not match:
        return False
    variant = match.group(1)
    if variant is None:
        return True
    return variant in QWEN3_ASR_MLX_VARIANTS


def _validate_supported_model_name(model_name: Any) -> str:
    raw_name = _normalize_requested_model_name(model_name)
    if _is_allowed_qwen3_asr_mlx_name(raw_name):
        return MODEL_ID

    supported_examples = [
        "qwen3-asr-mlx",
        "qwen3-asr-1.7b-bf16",
        "qwen3-asr-1.7b-4bit",
        "qwen3-asr-1.7b-8bit",
        "mlx-community/Qwen3-ASR-1.7B-bf16",
        "mlx-community/Qwen3-ASR-1.7B-8bit",
    ]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported model '{raw_name}'. "
            "This server only supports Qwen3-ASR MLX model names "
            "(full precision and quantized variants). "
            f"Examples: {', '.join(supported_examples)}"
        ),
    )


def _ensure_transcriber_ready_or_raise(requested_model: Any) -> str:
    canonical_model = _validate_supported_model_name(requested_model)
    if not transcriber.is_available():
        validation_error = _asset_dir_validation_error(Path(MODEL_PATH))
        detail = (
            f"Model files are missing or incomplete at '{MODEL_PATH}'. "
            "Automatic download is disabled. "
        )
        if validation_error:
            detail += f"reason={validation_error}. "
        detail += _manual_model_setup_help("QWEN_MLX_MODEL_PATH")
        raise HTTPException(
            status_code=400,
            detail=detail,
        )
    if not transcriber.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "ASR model exists but failed to initialize. "
                f"error={transcriber.last_error() or 'unknown'}"
            ),
        )
    return canonical_model


def _list_available_model_ids() -> list[str]:
    if not transcriber.is_available():
        return []

    return [
        MODEL_ID,
        "qwen3-asr-1.7b-bf16",
        "mlx-community/Qwen3-ASR-1.7B-bf16",
        "mlx-community/Qwen3-ASR-1.7B-4bit",
        "mlx-community/Qwen3-ASR-1.7B-5bit",
        "mlx-community/Qwen3-ASR-1.7B-6bit",
        "mlx-community/Qwen3-ASR-1.7B-8bit",
    ]


def _normalize_result(
    raw_result: Any,
    output_path: str,
    requested_language: str | None,
) -> TranscriptionResult:
    text: str | None = None
    raw_language: Any = None
    segments: list[dict[str, Any]] = []
    segment_timestamp_source = "none"

    if isinstance(raw_result, dict):
        text = _coalesce_text(raw_result)
        raw_language = raw_result.get("language")
        segments = _normalize_segments(raw_result.get("segments") or raw_result.get("chunks"))
        if segments:
            segment_timestamp_source = "asr"

    elif isinstance(raw_result, str):
        text = raw_result.strip()

    elif isinstance(raw_result, (list, tuple)):
        text_candidates = [item for item in raw_result if isinstance(item, str) and item.strip()]
        if text_candidates:
            text = " ".join(text_candidates).strip()

    else:
        text = _coalesce_text(raw_result)
        raw_language = getattr(raw_result, "language", None)
        segments = _normalize_segments(
            getattr(raw_result, "segments", None) or getattr(raw_result, "chunks", None)
        )
        if segments:
            segment_timestamp_source = "asr"

    if not text:
        try:
            text = Path(output_path).read_text(encoding="utf-8").strip()
        except OSError:
            text = None

    if not text and segments:
        text = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    if text is None:
        text = ""

    if not segments and text:
        approx_duration = max(1.0, len(text) / 12.0)
        segments = [{"id": 0, "start": 0.0, "end": approx_duration, "text": text}]
        segment_timestamp_source = "estimated"

    duration = max((seg.get("end", 0.0) for seg in segments), default=0.0)

    language = (
        _resolve_primary_language_from_segments(segments)
        or _select_primary_language_code(_extract_language_codes(raw_language))
        or _to_language_code(requested_language)
    )

    return TranscriptionResult(
        text=text,
        language=language,
        segments=segments,
        duration=duration,
        segment_timestamp_source=segment_timestamp_source,
    )


def _coalesce_text(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, dict):
        for key in ("text", "transcription", "result"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    for attr in ("text", "transcription", "result"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _prompt_log_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _preview_text(value: Any, max_chars: int = 120) -> str:
    text = _prompt_log_text(value)
    if not text:
        return "none"
    compact = " ".join(text.split())
    if not compact:
        return "none"
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."


def _log_prompt_preview(request_id: str, prompt: Any) -> None:
    if not LOG_PROMPTS:
        return
    prompt_text = _prompt_log_text(prompt)
    logger.info(
        "[%s] request_prompt prompt_len=%d prompt_preview=%s",
        request_id,
        len(prompt_text or ""),
        _preview_text(prompt_text),
    )


def _extract_language_values(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []

    if isinstance(value, dict):
        values: list[str] = []
        for key in ("language", "lang", "language_code", "detected_language"):
            if key in value:
                values.extend(_extract_language_values(value.get(key)))
        return values

    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_extract_language_values(item))
        return values

    for attr in ("language", "lang"):
        candidate = getattr(value, attr, None)
        if candidate is not None and candidate is not value:
            values = _extract_language_values(candidate)
            if values:
                return values

    return []


def _extract_language_codes(value: Any) -> list[str]:
    codes: list[str] = []
    for candidate in _extract_language_values(value):
        code = _to_language_code(candidate)
        if code:
            codes.append(code)
    return codes


def _select_primary_language_code(codes: list[str]) -> str | None:
    if not codes:
        return None

    counts: dict[str, int] = {}
    first_index: dict[str, int] = {}
    for index, code in enumerate(codes):
        counts[code] = counts.get(code, 0) + 1
        first_index.setdefault(code, index)

    return max(codes, key=lambda code: (counts[code], -first_index[code]))


def _normalize_segments(raw_segments: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_segments, (list, tuple)):
        return []

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_segments):
        if isinstance(item, dict):
            start = _to_float(item.get("start"), 0.0)
            end = _to_float(item.get("end"), start)
            if "timestamp" in item and isinstance(item["timestamp"], (list, tuple)):
                start = _to_float(item["timestamp"][0], start)
                end = _to_float(item["timestamp"][1], end)
            text = str(item.get("text", "")).strip()
            language = _select_primary_language_code(
                _extract_language_codes(item.get("language") or item.get("lang"))
            )
        else:
            start = _to_float(getattr(item, "start", None), 0.0)
            end = _to_float(getattr(item, "end", None), start)
            text = str(getattr(item, "text", "")).strip()
            language = _select_primary_language_code(
                _extract_language_codes(getattr(item, "language", None) or getattr(item, "lang", None))
            )

        if not text:
            continue

        segment = {
            "id": index,
            "start": start,
            "end": max(start, end),
            "text": text,
        }
        if language:
            segment["language"] = language
        normalized.append(segment)

    return normalized


def _resolve_primary_language_from_segments(segments: list[dict[str, Any]]) -> str | None:
    if not segments:
        return None

    stats: dict[str, dict[str, float | int]] = {}
    for index, segment in enumerate(segments):
        code = _to_language_code(str(segment.get("language", "")).strip())
        if not code:
            continue

        start = _to_float(segment.get("start"), 0.0)
        end = _to_float(segment.get("end"), start)
        weight = max(end - start, 0.0) or 1.0
        bucket = stats.setdefault(
            code,
            {"weight": 0.0, "count": 0, "first_index": index},
        )
        bucket["weight"] = float(bucket["weight"]) + weight
        bucket["count"] = int(bucket["count"]) + 1

    if not stats:
        return None

    return max(
        stats,
        key=lambda code: (
            float(stats[code]["weight"]),
            int(stats[code]["count"]),
            -int(stats[code]["first_index"]),
        ),
    )


def _segment_overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start = _to_float(left.get("start"), 0.0)
    left_end = _to_float(left.get("end"), left_start)
    right_start = _to_float(right.get("start"), 0.0)
    right_end = _to_float(right.get("end"), right_start)
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _restore_segment_languages_from_overlap(
    source_segments: list[dict[str, Any]],
    rebuilt_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not source_segments or not rebuilt_segments:
        return rebuilt_segments

    language_sources: list[dict[str, Any]] = []
    for index, segment in enumerate(source_segments):
        code = _to_language_code(str(segment.get("language", "")).strip())
        if not code:
            continue
        language_sources.append(
            {
                "language": code,
                "start": _to_float(segment.get("start"), 0.0),
                "end": _to_float(segment.get("end"), _to_float(segment.get("start"), 0.0)),
                "index": index,
            }
        )

    if not language_sources:
        return rebuilt_segments

    restored: list[dict[str, Any]] = []
    for segment in rebuilt_segments:
        item = dict(segment)
        existing_code = _to_language_code(str(item.get("language", "")).strip())
        if existing_code:
            item["language"] = existing_code
            restored.append(item)
            continue

        stats: dict[str, dict[str, float | int]] = {}
        for source in language_sources:
            overlap = _segment_overlap_seconds(item, source)
            if overlap <= 0:
                continue

            code = str(source["language"])
            bucket = stats.setdefault(
                code,
                {"weight": 0.0, "count": 0, "first_index": int(source["index"])},
            )
            bucket["weight"] = float(bucket["weight"]) + overlap
            bucket["count"] = int(bucket["count"]) + 1

        if stats:
            item["language"] = max(
                stats,
                key=lambda code: (
                    float(stats[code]["weight"]),
                    int(stats[code]["count"]),
                    -int(stats[code]["first_index"]),
                ),
            )
        restored.append(item)

    return restored


def _summarize_detected_languages(result: TranscriptionResult) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, float | int]] = {}
    for index, segment in enumerate(result.segments):
        code = _to_language_code(str(segment.get("language", "")).strip())
        if not code:
            continue

        start = _to_float(segment.get("start"), 0.0)
        end = _to_float(segment.get("end"), start)
        duration = max(end - start, 0.0)
        bucket = stats.setdefault(
            code,
            {"duration": 0.0, "count": 0, "first_index": index},
        )
        bucket["duration"] = float(bucket["duration"]) + duration
        bucket["count"] = int(bucket["count"]) + 1

    if not stats:
        fallback_code = _to_language_code(result.language)
        if not fallback_code:
            return []
        fallback_duration = max(_to_float(result.duration, 0.0), 0.0)
        return [{"code": fallback_code, "duration": round(fallback_duration, 3)}]

    ordered_codes = sorted(
        stats,
        key=lambda code: (
            -float(stats[code]["duration"]),
            -int(stats[code]["count"]),
            int(stats[code]["first_index"]),
        ),
    )
    return [
        {
            "code": code,
            "duration": round(float(stats[code]["duration"]), 3),
        }
        for code in ordered_codes
    ]


def _normalize_word_segments(raw: Any) -> list[dict[str, Any]]:
    raw_segments: Any = None
    if hasattr(raw, "segments"):
        raw_segments = raw.segments
    elif isinstance(raw, dict):
        raw_segments = raw.get("segments")

    if not isinstance(raw_segments, list):
        return []

    words: list[dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start = _to_float(item.get("start"), _to_float(item.get("start_time"), 0.0))
        end = _to_float(item.get("end"), _to_float(item.get("end_time"), start))
        words.append(
            {
                "word": text,
                "start": max(0.0, start),
                "end": max(max(0.0, start), end),
            }
        )
    return words


def _aligned_word_tokens(words: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("word", "")) for item in words]


def _token_mismatch_summary(expected: list[str], actual: list[str]) -> str:
    common_length = min(len(expected), len(actual))
    mismatch_index = next(
        (index for index in range(common_length) if expected[index] != actual[index]),
        common_length,
    )
    return (
        f"expected_tokens={len(expected)} actual_tokens={len(actual)} "
        f"first_mismatch={mismatch_index}"
    )


def _alignment_text_identity(text: str) -> str:
    return "".join(character for character in text if character == "'" or character.isalnum())


def _text_coverage_mismatch_summary(expected: str, actual: str) -> str:
    common_length = min(len(expected), len(actual))
    mismatch_index = next(
        (index for index in range(common_length) if expected[index] != actual[index]),
        common_length,
    )
    return (
        f"expected_characters={len(expected)} actual_characters={len(actual)} "
        f"first_mismatch={mismatch_index}"
    )


def _validate_aligned_word_identity(
    words: list[dict[str, Any]],
    canonical_tokens: list[str],
    *,
    context: str,
) -> None:
    actual_tokens = _aligned_word_tokens(words)
    if actual_tokens != canonical_tokens:
        raise RuntimeError(
            f"{context} token identity mismatch: "
            f"{_token_mismatch_summary(canonical_tokens, actual_tokens)}"
        )


def _validate_aligned_word_timeline(words: list[dict[str, Any]]) -> None:
    previous_start = 0.0
    for index, word in enumerate(words):
        start = _to_float(word.get("start"), 0.0)
        end = _to_float(word.get("end"), start)
        if end < start:
            raise RuntimeError(f"aligned word has end before start at index={index}")
        if index > 0 and start + 0.001 < previous_start:
            raise RuntimeError(f"aligned word timeline is not monotonic at index={index}")
        previous_start = start


def _normalize_timestamp_granularities(value: Any) -> list[str]:
    if value is None:
        return []

    items: list[str]
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = [str(part).strip() for part in value if str(part).strip()]
    else:
        items = [str(value).strip()]

    aliases = {"words": "word", "segments": "segment"}
    normalized: list[str] = []
    for item in items:
        candidate = aliases.get(item.lower(), item.lower())
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _timestamp_requests_alignment(granularities: list[str] | None) -> bool:
    if not granularities:
        return False
    normalized = {item.lower() for item in granularities}
    return bool(normalized & {"word", "words", "segment", "segments"})


def _validate_timestamp_response_contract(
    *,
    response_format: str,
    alignment_mode: str,
    timestamp_granularities: list[str] | None,
    chat_response: bool = False,
) -> None:
    granularities = timestamp_granularities or []
    unsupported = sorted(set(granularities) - SUPPORTED_TIMESTAMP_GRANULARITIES)
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported timestamp granularities: {', '.join(unsupported)}. "
                "Supported: segment, word."
            ),
        )

    timestamps_requested = bool(granularities) or alignment_mode == "on"
    if not timestamps_requested:
        return

    if chat_response:
        raise HTTPException(
            status_code=400,
            detail=(
                "Chat Completions returns transcript text only and cannot expose timestamps. "
                "Use /v1/audio/transcriptions with response_format=verbose_json."
            ),
        )

    if response_format in {"json", "text"}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"response_format={response_format} cannot expose timestamps. "
                "Use response_format=verbose_json for JSON timestamps, or srt/vtt for subtitles."
            ),
        )


def _parse_alignment_mode(value: Any) -> str:
    if value is None:
        return "auto"

    if isinstance(value, bool):
        return "on" if value else "off"

    raw = str(value).strip().lower()
    if raw in {"", "auto", "default"}:
        return "auto"
    if raw in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return "on"
    if raw in {"0", "false", "no", "n", "off", "disable", "disabled", "none"}:
        return "off"
    return "auto"


def _parse_optional_boolean(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
        return True
    if raw in {"0", "false", "no", "n", "off", "disable", "disabled"}:
        return False
    return None


def _to_language_code(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"auto", "unknown", "none"}:
        return None
    if lowered in LANGUAGE_NAME_TO_CODE:
        return LANGUAGE_NAME_TO_CODE[lowered]
    if lowered in LANGUAGE_CODE_TO_NAME:
        return lowered
    if "-" in lowered:
        prefix = lowered.split("-", 1)[0]
        if prefix in LANGUAGE_CODE_TO_NAME:
            return prefix
    if len(lowered) == 2 and lowered.isalpha():
        return lowered
    return None


def _to_aligner_language_name(value: str | None) -> str:
    code = _to_language_code(value)
    if code and code in LANGUAGE_CODE_TO_NAME:
        return LANGUAGE_CODE_TO_NAME[code]

    if isinstance(value, str) and value.strip():
        return value.strip().title()
    return "English"


def _detect_language_code_from_text(text: str) -> str | None:
    if not text:
        return None

    # Korean: Hangul syllables/jamo
    if re.search(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]", text):
        return "ko"
    # Japanese: Hiragana/Katakana
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", text):
        return "ja"
    # Chinese/Japanese shared ideographs fallback
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return None


def _resolve_detected_language_code(
    requested_language: str | None,
    result_language: str | None,
    text: str,
) -> str | None:
    return (
        _to_language_code(requested_language)
        or _to_language_code(result_language)
        or _detect_language_code_from_text(text)
    )


def _finalize_result_language(
    requested_language: str | None,
    result: TranscriptionResult,
) -> str | None:
    detected_language_code = _resolve_detected_language_code(
        requested_language=requested_language,
        result_language=result.language,
        text=result.text,
    )
    if detected_language_code:
        result.language = detected_language_code
    return detected_language_code


def _compute_alignment_enabled(
    alignment_mode: str,
    timestamps_requested: bool,
    response_format: str,
    detected_language_code: str | None,
    auto_align_lang_codes: set[str],
) -> bool:
    if alignment_mode == "off":
        return False
    if alignment_mode == "on" or timestamps_requested:
        return True
    if response_format not in AUTO_ALIGNMENT_RESPONSE_FORMATS:
        return False
    return (detected_language_code or "") in auto_align_lang_codes


def _alignment_dependency_missing(lang_code: str | None) -> str | None:
    code = (lang_code or "").lower()
    if code == "ja" and importlib.util.find_spec("nagisa") is None:
        return "missing dependency nagisa for Japanese alignment"
    if code == "ko" and importlib.util.find_spec("soynlp") is None:
        return "missing dependency soynlp for Korean alignment"
    return None


def _effective_auto_align_lang_codes(aligner: MLXAligner) -> set[str]:
    # Auto alignment must have a ready aligner model first.
    if not aligner.is_ready():
        return set()

    allowed = set(AUTO_ALIGN_LANG_CODES)
    if not allowed:
        return allowed

    for code in list(allowed):
        missing_dep = _alignment_dependency_missing(code)
        if missing_dep is not None:
            allowed.discard(code)

    supported_languages = aligner.supported_languages()
    if supported_languages:
        # Filter by aligner language support if metadata is available.
        for code in list(allowed):
            language_name = LANGUAGE_CODE_TO_NAME.get(code, "").lower()
            if language_name and language_name not in supported_languages:
                allowed.discard(code)

    return allowed


def _get_wav_duration_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate <= 0:
                return None
            return frame_count / float(frame_rate)
    except Exception:
        return None


def _should_rechunk_segments_for_alignment(
    segments: list[dict[str, Any]],
    duration_sec: float | None,
    chunk_seconds: float,
) -> bool:
    if not duration_sec or duration_sec <= chunk_seconds:
        return False

    expected_chunks = max(1, int(math.ceil(duration_sec / chunk_seconds)))
    return len(segments) < expected_chunks


def _segment_text_for_chunks(
    result: TranscriptionResult,
    chunk_ranges: list[tuple[str, float, float]],
) -> list[str]:
    # Assign each coarse ASR segment to exactly one alignment chunk to avoid
    # duplicated text across neighboring chunks (which can produce overlapping
    # word timestamps after chunk offset stitching).
    assigned: list[list[str]] = [[] for _ in chunk_ranges]
    for segment_index, segment in enumerate(result.segments):
        seg_text = str(segment.get("text", "")).strip()
        if not seg_text:
            continue
        seg_start = _to_float(segment.get("start"), 0.0)
        seg_end = _to_float(segment.get("end"), seg_start)
        if seg_end < seg_start:
            seg_end = seg_start

        best_idx = -1
        best_overlap = -1.0
        seg_mid = (seg_start + seg_end) / 2.0

        for idx, (_chunk_path, chunk_start, chunk_end) in enumerate(chunk_ranges):
            overlap = max(0.0, min(seg_end, chunk_end) - max(seg_start, chunk_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = idx
            elif overlap == best_overlap and best_idx >= 0:
                # Tie-breaker: prefer the chunk containing segment midpoint.
                in_best = chunk_ranges[best_idx][1] <= seg_mid < chunk_ranges[best_idx][2]
                in_curr = chunk_start <= seg_mid < chunk_end
                if in_curr and not in_best:
                    best_idx = idx

        if best_idx < 0:
            continue
        segment_duration = max(0.0, seg_end - seg_start)
        if segment_duration <= 0.0:
            raise RuntimeError(
                f"alignment chunk assignment requires positive ASR segment duration "
                f"at index={segment_index}"
            )
        if best_overlap + 0.001 < segment_duration:
            raise RuntimeError(
                f"ASR segment crosses alignment chunk boundary at index={segment_index} "
                f"segment_duration={segment_duration:.3f} overlap={best_overlap:.3f}"
            )
        assigned[best_idx].append(seg_text)

    # Do not estimate transcript slices for empty chunks. The caller verifies that
    # the exact canonical tokens from all assigned segment texts cover result.text.
    return [" ".join(items).strip() for items in assigned]


def _segment_end_boundaries_for_alignment(
    segments: list[dict[str, Any]],
    duration_sec: float,
) -> list[float]:
    boundaries: list[float] = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        end = _to_float(segment.get("end"), 0.0)
        if 0.0 < end < duration_sec:
            boundaries.append(end)

    deduped: list[float] = []
    for boundary in sorted(boundaries):
        if not deduped or abs(boundary - deduped[-1]) > 0.001:
            deduped.append(boundary)
    return deduped


def _alignment_chunk_boundaries_from_segments(
    segments: list[dict[str, Any]],
    duration_sec: float | None,
    max_chunk_seconds: float,
    boundary_tolerance_seconds: float,
) -> list[float]:
    if not duration_sec or duration_sec <= 0.0:
        return [0.0]
    if max_chunk_seconds <= 0.0 or duration_sec <= max_chunk_seconds:
        return [0.0, duration_sec]

    tolerance = max(0.0, boundary_tolerance_seconds)
    segment_boundaries = _segment_end_boundaries_for_alignment(segments, duration_sec)
    boundaries = [0.0]

    while duration_sec - boundaries[-1] > max_chunk_seconds:
        previous = boundaries[-1]
        target = previous + max_chunk_seconds
        selected = target

        if tolerance > 0.0 and segment_boundaries:
            lower = max(previous + 0.001, target - tolerance)
            upper = min(duration_sec - 0.001, target + tolerance)
            candidates = [
                boundary
                for boundary in segment_boundaries
                if lower <= boundary <= upper
            ]
            if candidates:
                selected = min(candidates, key=lambda boundary: (abs(boundary - target), boundary))

        selected = max(previous + 0.001, min(selected, duration_sec))
        if selected >= duration_sec:
            break
        boundaries.append(selected)

    if boundaries[-1] < duration_sec:
        boundaries.append(duration_sec)
    return boundaries


def _split_wav_into_temp_chunks(
    wav_path: str,
    max_chunk_seconds: float,
    split_boundaries: list[float] | None = None,
) -> list[tuple[str, float, float]]:
    if max_chunk_seconds <= 0:
        return [(wav_path, 0.0, _get_wav_duration_seconds(wav_path) or 0.0)]

    chunks: list[tuple[str, float, float]] = []
    with wave.open(wav_path, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            raise RuntimeError("Invalid WAV frame rate")

        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        comptype = wav_file.getcomptype()
        compname = wav_file.getcompname()
        total_frames = wav_file.getnframes()
        if total_frames <= 0:
            return chunks
        if split_boundaries:
            frame_offsets = [0]
            for boundary in split_boundaries[1:-1]:
                frame_offset = int(round(max(0.0, boundary) * frame_rate))
                frame_offset = max(1, min(frame_offset, total_frames - 1))
                if frame_offset > frame_offsets[-1]:
                    frame_offsets.append(frame_offset)
            if not frame_offsets or frame_offsets[-1] != total_frames:
                frame_offsets.append(total_frames)
        else:
            chunk_frames = max(1, int(frame_rate * max_chunk_seconds))
            frame_offsets = list(range(0, total_frames, chunk_frames))
            if not frame_offsets or frame_offsets[0] != 0:
                frame_offsets.insert(0, 0)
            if frame_offsets[-1] != total_frames:
                frame_offsets.append(total_frames)

        frame_offset = 0

        for next_frame_offset in frame_offsets[1:]:
            read_frames = max(0, next_frame_offset - frame_offset)
            if read_frames <= 0:
                continue
            frames = wav_file.readframes(read_frames)
            start_sec = frame_offset / float(frame_rate)
            end_sec = next_frame_offset / float(frame_rate)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                chunk_path = tmp_file.name
            with wave.open(chunk_path, "wb") as out_file:
                out_file.setnchannels(channels)
                out_file.setsampwidth(sample_width)
                out_file.setframerate(frame_rate)
                out_file.setcomptype(comptype, compname)
                out_file.writeframes(frames)
            chunks.append((chunk_path, start_sec, end_sec))
            frame_offset = next_frame_offset

    return chunks


def _apply_alignment(
    aligner: MLXAligner,
    audio_path: str,
    result: TranscriptionResult,
    aligner_language: str,
    duration_sec: float | None,
    max_chunk_seconds: float,
) -> AlignmentOutput:
    if not result.text.strip():
        return AlignmentOutput()

    if not duration_sec or duration_sec <= max_chunk_seconds:
        canonical_tokens = aligner.canonical_tokens(result.text, aligner_language)
        words = aligner.align(audio=audio_path, text=result.text, language=aligner_language)
        _validate_aligned_word_identity(
            words,
            canonical_tokens,
            context="forced alignment",
        )
        _validate_aligned_word_timeline(words)
        chunk_end = max(
            _to_float(duration_sec, 0.0),
            max((_to_float(word.get("end"), 0.0) for word in words), default=0.0),
        )
        return AlignmentOutput(
            words=words,
            chunks=[
                AlignmentChunk(
                    text=result.text,
                    canonical_tokens=canonical_tokens,
                    words=words,
                    start=0.0,
                    end=chunk_end,
                )
            ]
            if canonical_tokens
            else [],
        )

    split_boundaries = _alignment_chunk_boundaries_from_segments(
        result.segments,
        duration_sec,
        max_chunk_seconds,
        ALIGNMENT_BOUNDARY_TOLERANCE_SECONDS,
    )
    chunk_specs = _split_wav_into_temp_chunks(
        audio_path,
        max_chunk_seconds,
        split_boundaries=split_boundaries,
    )
    chunk_paths = [chunk_path for chunk_path, _, _ in chunk_specs]
    try:
        chunk_texts = _segment_text_for_chunks(result, chunk_specs)
        chunk_tokens = [
            aligner.canonical_tokens(chunk_text, aligner_language)
            if chunk_text.strip()
            else []
            for chunk_text in chunk_texts
        ]
        alignment_tokens = [
            token
            for tokens in chunk_tokens
            for token in tokens
        ]
        expected_text_identity = _alignment_text_identity(result.text)
        actual_text_identity = _alignment_text_identity(" ".join(chunk_texts))
        if actual_text_identity != expected_text_identity:
            raise RuntimeError(
                "alignment chunk transcript coverage mismatch: "
                f"{_text_coverage_mismatch_summary(expected_text_identity, actual_text_identity)}"
            )

        aligned_words: list[dict[str, Any]] = []
        aligned_chunks: list[AlignmentChunk] = []
        for (chunk_path, chunk_start, chunk_end), chunk_text, canonical_tokens in zip(
            chunk_specs,
            chunk_texts,
            chunk_tokens,
        ):
            if not chunk_text.strip():
                continue
            chunk_duration = max(0.0, float(chunk_end) - float(chunk_start))
            words = aligner.align(
                audio=chunk_path,
                text=chunk_text,
                language=aligner_language,
            )
            _validate_aligned_word_identity(
                words,
                canonical_tokens,
                context="forced alignment chunk",
            )
            offset_words: list[dict[str, Any]] = []
            for word in words:
                local_start = _to_float(word.get("start"), 0.0)
                local_end = _to_float(word.get("end"), local_start)
                if local_end < local_start:
                    local_end = local_start

                # Keep each aligned token inside the audio chunk boundary so
                # chunk stitching cannot drift later than the real timeline.
                if chunk_duration > 0.0:
                    local_start = max(0.0, min(local_start, chunk_duration))
                    local_end = max(local_start, min(local_end, chunk_duration))
                else:
                    local_start = 0.0
                    local_end = 0.0

                offset_words.append(
                    {
                        "word": str(word.get("word", "")),
                        "start": local_start + chunk_start,
                        "end": local_end + chunk_start,
                    }
                )
            aligned_words.extend(offset_words)
            aligned_chunks.append(
                AlignmentChunk(
                    text=chunk_text,
                    canonical_tokens=canonical_tokens,
                    words=offset_words,
                    start=float(chunk_start),
                    end=float(chunk_end),
                )
            )

        _validate_aligned_word_identity(
            aligned_words,
            alignment_tokens,
            context="stitched forced alignment",
        )
        _validate_aligned_word_timeline(aligned_words)
        return AlignmentOutput(words=aligned_words, chunks=aligned_chunks)
    finally:
        for chunk_path in chunk_paths:
            try:
                Path(chunk_path).unlink(missing_ok=True)
            except OSError:
                pass


def _text_parts_from_candidate_ends(text: str, candidate_ends: list[int]) -> list[TextPart]:
    content = text or ""
    if not content.strip():
        return []

    parts: list[TextPart] = []
    cursor = 0
    for end in sorted(set(candidate_ends)):
        if end <= cursor or end > len(content):
            continue
        raw_part = content[cursor:end]
        if raw_part.strip():
            parts.append(TextPart(start=cursor, end=end, text=raw_part.strip()))
            cursor = end

    if content[cursor:].strip():
        parts.append(TextPart(start=cursor, end=len(content), text=content[cursor:].strip()))
    elif parts and parts[-1].end < len(content):
        last = parts[-1]
        parts[-1] = TextPart(
            start=last.start,
            end=len(content),
            text=content[last.start:].strip(),
        )

    if not parts:
        return [TextPart(start=0, end=len(content), text=content.strip())]
    return parts


def _split_text_by_boundary_pattern(text: str, pattern: str) -> list[TextPart]:
    return _text_parts_from_candidate_ends(
        text,
        [match.end() for match in re.finditer(pattern, text or "")],
    )


def _sentence_candidate_ends(text: str) -> list[int]:
    ends = {
        match.end()
        for match in re.finditer(r"(?:[。！？!?；;]+|\.{2,})", text)
    }
    for match in re.finditer(r"\.", text):
        index = match.start()
        previous = text[index - 1] if index > 0 else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if previous == "." or following == ".":
            continue
        if previous.isalnum() and following.isalnum():
            continue

        prefix = text[: match.end()]
        suffix = text[match.end() :]
        if suffix.strip() and re.search(r"(?:\b[A-Za-z]\.){2,}$", prefix):
            continue
        ends.add(match.end())
    return sorted(ends)


def _split_text_into_sentences(text: str) -> list[TextPart]:
    # Candidate boundaries are verified against canonical tokenizer prefixes later.
    return _text_parts_from_candidate_ends(text, _sentence_candidate_ends(text))


def _safe_token_boundaries(
    *,
    text: str,
    words: list[dict[str, Any]],
    candidate_ends: list[int],
    tokenize: Callable[[str], list[str]],
) -> tuple[dict[int, int], str | None]:
    canonical_tokens = tokenize(text)
    actual_tokens = _aligned_word_tokens(words)
    if canonical_tokens != actual_tokens:
        return {}, _token_mismatch_summary(canonical_tokens, actual_tokens)

    safe_boundaries: dict[int, int] = {}
    for end in sorted(set(candidate_ends)):
        if end <= 0 or end >= len(text):
            continue
        prefix_tokens = tokenize(text[:end])
        token_count = len(prefix_tokens)
        if not 0 < token_count < len(canonical_tokens):
            continue
        if prefix_tokens == canonical_tokens[:token_count]:
            # Prefer the furthest character position for an equivalent token
            # boundary so punctuation and whitespace stay with the prior part.
            safe_boundaries[token_count] = max(
                end,
                safe_boundaries.get(token_count, 0),
            )

    return safe_boundaries, None


def _segments_from_token_boundaries(
    *,
    text: str,
    words: list[dict[str, Any]],
    token_boundaries: dict[int, int],
    selected_token_counts: list[int] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    selected = sorted(
        set(selected_token_counts if selected_token_counts is not None else token_boundaries)
    )
    selected = [count for count in selected if 0 < count < len(words)]

    segments: list[dict[str, Any]] = []
    word_cursor = 0
    char_cursor = 0
    for word_end in [*selected, len(words)]:
        char_end = token_boundaries.get(word_end, len(text))
        chunk_words = words[word_cursor:word_end]
        segment_text = text[char_cursor:char_end].strip()
        if not chunk_words or not segment_text:
            return [], (
                f"empty exact segment at token_start={word_cursor} "
                f"token_end={word_end}"
            )

        start = _to_float(chunk_words[0].get("start"), 0.0)
        end = _to_float(chunk_words[-1].get("end"), start)
        segments.append(
            {
                "id": len(segments),
                "start": max(0.0, start),
                "end": max(max(0.0, start), end),
                "text": segment_text,
                "_words": chunk_words,
            }
        )
        word_cursor = word_end
        char_cursor = char_end

    if word_cursor != len(words):
        return [], f"exact segment mapping consumed={word_cursor} total={len(words)}"
    return segments, None


def _map_text_parts_to_word_segments(
    *,
    text: str,
    words: list[dict[str, Any]],
    parts: list[TextPart],
    tokenize: Callable[[str], list[str]],
) -> tuple[list[dict[str, Any]], str | None]:
    safe_boundaries, error = _safe_token_boundaries(
        text=text,
        words=words,
        candidate_ends=[part.end for part in parts],
        tokenize=tokenize,
    )
    if error is not None:
        return [], error
    return _segments_from_token_boundaries(
        text=text,
        words=words,
        token_boundaries=safe_boundaries,
    )


def _build_sentence_segments_from_words(
    words: list[dict[str, Any]],
    text: str,
    lang_code: str | None,
    tokenize: Callable[[str], list[str]],
) -> tuple[list[dict[str, Any]], str | None]:
    if not words:
        return [], None

    sentences = _split_text_into_sentences(text)
    if not sentences:
        return [], "sentence splitter returned no text parts"

    return _map_text_parts_to_word_segments(
        text=text,
        words=words,
        parts=sentences,
        tokenize=tokenize,
    )


def _join_segment_texts(texts: list[str], lang_code: str | None) -> str:
    items = [item.strip() for item in texts if item and item.strip()]
    if not items:
        return ""
    if (lang_code or "").lower() in {"zh", "ja"}:
        return "".join(items)
    return " ".join(items)


def _merge_segments_by_target_duration(
    segments: list[dict[str, Any]],
    lang_code: str | None,
    target_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    if target_seconds <= 0:
        kept: list[dict[str, Any]] = []
        for idx, seg in enumerate(segments):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            item: dict[str, Any] = {
                "id": idx,
                "start": _to_float(seg.get("start"), 0.0),
                "end": _to_float(seg.get("end"), _to_float(seg.get("start"), 0.0)),
                "text": text,
            }
            seg_words = seg.get("_words")
            if isinstance(seg_words, list) and seg_words:
                item["_words"] = seg_words
            kept.append(item)
        return kept

    merged: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []

    def flush_group() -> None:
        nonlocal group
        if not group:
            return
        start = _to_float(group[0].get("start"), 0.0)
        end = _to_float(group[-1].get("end"), start)
        text = _join_segment_texts([str(item.get("text", "")) for item in group], lang_code)
        group_words: list[dict[str, Any]] = []
        for item in group:
            seg_words = item.get("_words")
            if isinstance(seg_words, list) and seg_words:
                group_words.extend(seg_words)
        if text:
            entry: dict[str, Any] = {
                "id": len(merged),
                "start": max(0.0, start),
                "end": max(max(0.0, start), end),
                "text": text,
            }
            if group_words:
                entry["_words"] = group_words
            merged.append(entry)
        group = []

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        group.append(
            {
                "start": _to_float(segment.get("start"), 0.0),
                "end": _to_float(segment.get("end"), _to_float(segment.get("start"), 0.0)),
                "text": text,
                "_words": segment.get("_words"),
            }
        )
        duration = _to_float(group[-1]["end"], 0.0) - _to_float(group[0]["start"], 0.0)
        if duration >= target_seconds:
            flush_group()

    flush_group()
    return merged


def _split_text_by_punctuation_pattern(text: str, pattern: str) -> list[TextPart]:
    return _split_text_by_boundary_pattern(text, f"(?:{pattern})")


def _build_subsegments_from_text_parts(
    segment: dict[str, Any],
    parts: list[TextPart],
    tokenize: Callable[[str], list[str]],
) -> tuple[list[dict[str, Any]], str | None]:
    if not parts:
        return [], "subsegment splitter returned no text parts"

    segment_words = segment.get("_words")
    if not isinstance(segment_words, list) or not segment_words:
        return [], "exact subsegment mapping requires aligned words"

    return _map_text_parts_to_word_segments(
        text=str(segment.get("text", "")),
        words=segment_words,
        parts=parts,
        tokenize=tokenize,
    )


def _broad_token_boundary_candidates(text: str, lang_code: str | None) -> list[int]:
    candidates = {match.end() for match in re.finditer(r"(?:\s+|[^\w\s]+)", text)}
    if (lang_code or "").lower() in {"zh", "ja"}:
        candidates.update(range(1, len(text)))
    candidates.add(len(text))
    return sorted(candidates)


def _select_hard_limit_token_boundaries(
    words: list[dict[str, Any]],
    available_token_counts: list[int],
    hard_max_seconds: float,
) -> list[int]:
    selected: list[int] = []
    cursor = 0
    available = sorted(count for count in set(available_token_counts) if 0 < count < len(words))
    while cursor < len(words):
        segment_start = _to_float(words[cursor].get("start"), 0.0)
        remaining_end = _to_float(words[-1].get("end"), segment_start)
        if remaining_end - segment_start <= hard_max_seconds:
            break

        after_cursor = [count for count in available if count > cursor]
        if not after_cursor:
            break
        within_limit = [
            count
            for count in after_cursor
            if _to_float(words[count - 1].get("end"), segment_start) - segment_start
            <= hard_max_seconds
        ]
        chosen = max(within_limit) if within_limit else min(after_cursor)
        selected.append(chosen)
        cursor = chosen

    return selected


def _build_hard_limit_subsegments(
    segment: dict[str, Any],
    lang_code: str | None,
    hard_max_seconds: float,
    tokenize: Callable[[str], list[str]],
) -> list[dict[str, Any]]:
    text = str(segment.get("text", ""))
    segment_words = segment.get("_words")
    if not text.strip() or not isinstance(segment_words, list) or not segment_words:
        return []

    safe_boundaries, error = _safe_token_boundaries(
        text=text,
        words=segment_words,
        candidate_ends=_broad_token_boundary_candidates(text, lang_code),
        tokenize=tokenize,
    )
    if error is not None:
        return []
    selected = _select_hard_limit_token_boundaries(
        segment_words,
        list(safe_boundaries),
        hard_max_seconds,
    )
    if not selected:
        return []
    subsegments, error = _segments_from_token_boundaries(
        text=text,
        words=segment_words,
        token_boundaries=safe_boundaries,
        selected_token_counts=selected,
    )
    return subsegments if error is None else []


def _split_long_segment_once(
    segment: dict[str, Any],
    lang_code: str | None,
    hard_max_seconds: float,
    tokenize: Callable[[str], list[str]],
) -> list[dict[str, Any]]:
    text = str(segment.get("text", "")).strip()
    start = _to_float(segment.get("start"), 0.0)
    end = _to_float(segment.get("end"), start)
    duration = max(0.0, end - start)
    if not text or duration <= hard_max_seconds:
        return [segment]

    # Priority 1: split by comma.
    comma_parts = _split_text_by_punctuation_pattern(text, r"[，,]")
    if len(comma_parts) > 1:
        subsegments, error = _build_subsegments_from_text_parts(
            segment,
            comma_parts,
            tokenize,
        )
        if error is None and len(subsegments) > 1:
            return subsegments

    # Priority 2: split by other punctuation.
    other_parts = _split_text_by_punctuation_pattern(text, r"[、；;:：。！？!?]+")
    if len(other_parts) > 1:
        subsegments, error = _build_subsegments_from_text_parts(
            segment,
            other_parts,
            tokenize,
        )
        if error is None and len(subsegments) > 1:
            return subsegments

    # Final fallback: split only at tokenizer-verified word boundaries. If no
    # exact boundary exists, preserve the longer correct segment.
    exact_subsegments = _build_hard_limit_subsegments(
        segment,
        lang_code,
        hard_max_seconds,
        tokenize,
    )
    return exact_subsegments if len(exact_subsegments) > 1 else [segment]


def _enforce_hard_duration_limit(
    segments: list[dict[str, Any]],
    lang_code: str | None,
    tokenize: Callable[[str], list[str]],
    hard_max_seconds: float = 6.0,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    if hard_max_seconds <= 0:
        hard_max_seconds = 6.0

    output: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = [dict(item) for item in segments]
    max_iterations = 2048
    iteration = 0

    while queue and iteration < max_iterations:
        iteration += 1
        current = queue.pop(0)
        start = _to_float(current.get("start"), 0.0)
        end = _to_float(current.get("end"), start)
        duration = max(0.0, end - start)
        text = str(current.get("text", "")).strip()
        if not text:
            continue

        if duration <= hard_max_seconds:
            item: dict[str, Any] = {"id": 0, "start": start, "end": end, "text": text}
            seg_words = current.get("_words")
            if isinstance(seg_words, list) and seg_words:
                item["_words"] = seg_words
            output.append(item)
            continue

        parts = _split_long_segment_once(
            current,
            lang_code,
            hard_max_seconds,
            tokenize,
        )
        if len(parts) <= 1:
            item: dict[str, Any] = {"id": 0, "start": start, "end": end, "text": text}
            seg_words = current.get("_words")
            if isinstance(seg_words, list) and seg_words:
                item["_words"] = seg_words
            output.append(item)
            continue
        queue = [dict(part) for part in parts if str(part.get("text", "")).strip()] + queue

    if queue:
        for current in queue:
            text = str(current.get("text", "")).strip()
            if not text:
                continue
            start = _to_float(current.get("start"), 0.0)
            end = _to_float(current.get("end"), start)
            item: dict[str, Any] = {"id": 0, "start": start, "end": max(start, end), "text": text}
            seg_words = current.get("_words")
            if isinstance(seg_words, list) and seg_words:
                item["_words"] = seg_words
            output.append(item)

    for idx, item in enumerate(output):
        item["id"] = idx
    return output


def _rebuild_segments_from_alignment(
    *,
    alignment: AlignmentOutput,
    aligner: MLXAligner,
    aligner_language: str,
    lang_code: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    if not alignment.words:
        return [], None

    def tokenize(text: str) -> list[str]:
        return aligner.canonical_tokens(text, aligner_language)

    rebuilt: list[dict[str, Any]] = []
    for chunk_index, chunk in enumerate(alignment.chunks):
        chunk_word_tokens = _aligned_word_tokens(chunk.words)
        if chunk_word_tokens != chunk.canonical_tokens:
            return [], (
                f"chunk={chunk_index} preserved token identity mismatch: "
                f"{_token_mismatch_summary(chunk.canonical_tokens, chunk_word_tokens)}"
            )
        chunk_segments, error = _build_sentence_segments_from_words(
            chunk.words,
            chunk.text,
            lang_code,
            tokenize,
        )
        if error is not None:
            return [], f"chunk={chunk_index} {error}"
        rebuilt.extend(chunk_segments)
    if not rebuilt:
        return [], "exact sentence mapping produced no segments"

    if SEGMENT_MERGE_TARGET_SECONDS > 0.0:
        rebuilt = _merge_segments_by_target_duration(
            rebuilt,
            lang_code,
            target_seconds=SEGMENT_MERGE_TARGET_SECONDS,
        )
    if SEGMENT_HARD_MAX_SECONDS > 0.0:
        rebuilt = _enforce_hard_duration_limit(
            rebuilt,
            lang_code,
            tokenize,
            hard_max_seconds=SEGMENT_HARD_MAX_SECONDS,
        )

    rebuilt_words: list[dict[str, Any]] = []
    for segment in rebuilt:
        segment_words = segment.get("_words")
        if not isinstance(segment_words, list) or not segment_words:
            return [], "rebuilt segment lost aligned word identity"
        rebuilt_words.extend(segment_words)
    if _aligned_word_tokens(rebuilt_words) != _aligned_word_tokens(alignment.words):
        return [], (
            "rebuilt segments do not partition aligned words: "
            f"{_token_mismatch_summary(_aligned_word_tokens(alignment.words), _aligned_word_tokens(rebuilt_words))}"
        )

    return _normalize_segments(rebuilt), None


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    minutes = (millis % 3_600_000) // 60_000
    secs = (millis % 60_000) // 1000
    ms = millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    return _format_srt_timestamp(seconds).replace(",", ".")


def _as_srt(result: TranscriptionResult) -> str:
    lines: list[str] = []
    for i, segment in enumerate(result.segments, start=1):
        lines.append(str(i))
        lines.append(
            f"{_format_srt_timestamp(segment['start'])} --> {_format_srt_timestamp(segment['end'])}"
        )
        lines.append(segment["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _as_vtt(result: TranscriptionResult) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for segment in result.segments:
        lines.append(
            f"{_format_vtt_timestamp(segment['start'])} --> {_format_vtt_timestamp(segment['end'])}"
        )
        lines.append(segment["text"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_transcription_response(result: TranscriptionResult, response_format: str) -> Response:
    if response_format == "text":
        return PlainTextResponse(content=result.text)

    if response_format == "verbose_json":
        primary_language = result.language or "unknown"
        detected_languages = _summarize_detected_languages(result)
        payload: dict[str, Any] = {
            "task": "transcribe",
            "language": primary_language,
            "primary_language": primary_language,
            "duration": result.duration,
            "text": result.text,
            "timestamp_provenance": {
                "segments": result.segment_timestamp_source,
                "words": result.word_timestamp_source,
                "alignment": result.alignment_status,
            },
            "segments": [
                {
                    "id": segment["id"],
                    "seek": 0,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    **(
                        {"language": segment["language"]}
                        if _to_language_code(str(segment.get("language", "")).strip())
                        else {}
                    ),
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                }
                for segment in result.segments
            ],
        }
        if detected_languages:
            payload["detected_languages"] = detected_languages
        if result.words:
            payload["words"] = [
                {
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                }
                for word in result.words
            ]
        return JSONResponse(content=payload)

    if response_format == "srt":
        return PlainTextResponse(content=_as_srt(result), media_type="text/plain; charset=utf-8")

    if response_format == "vtt":
        return PlainTextResponse(content=_as_vtt(result), media_type="text/vtt; charset=utf-8")

    return JSONResponse(content={"text": result.text})


def _normalize_task(task: Any) -> str | None:
    if task is None:
        return None
    value = str(task).strip().lower()
    if not value:
        return None
    return value


def _validate_task(task: Any) -> None:
    normalized = _normalize_task(task)
    if normalized in {None, "transcribe"}:
        return
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unsupported task '{normalized}'. "
            "This local server currently supports only task=transcribe."
        ),
    )


def _coerce_decode_option(value: Any, option_name: str, kind: str) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None

    try:
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "bool":
            if isinstance(value, bool):
                return value
            raw = str(value).strip().lower()
            if raw in {"1", "true", "yes", "y", "on"}:
                return True
            if raw in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"invalid boolean: {value}")
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decode option '{option_name}': {exc}",
        ) from exc

    return value


def _collect_decode_options(**kwargs: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for key, kind in DECODE_OPTION_TYPES.items():
        if key not in kwargs:
            continue
        value = _coerce_decode_option(kwargs[key], key, kind)
        if value is not None:
            options[key] = value
    return options


def _apply_json_decode_options(
    options: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(options)
    for key, kind in DECODE_OPTION_TYPES.items():
        if key not in payload:
            continue
        value = _coerce_decode_option(payload.get(key), key, kind)
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _normalize_chat_content_text(item: dict[str, Any]) -> str | None:
    item_type = str(item.get("type", "")).strip().lower()
    if item_type in {"text", "input_text"}:
        text = item.get("text")
        if text is None and item_type == "input_text":
            text = item.get("input_text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def _audio_spec_from_chat_item(item: dict[str, Any]) -> Any:
    item_type = str(item.get("type", "")).strip().lower()

    if item_type in {"input_audio", "audio"}:
        spec = item.get("input_audio") if item_type == "input_audio" else item.get("audio")
        if isinstance(spec, str):
            return spec
        if not isinstance(spec, dict):
            return None
        normalized = dict(spec)
        fmt = str(normalized.get("format", "wav")).lower().strip()
        if fmt and "/" not in fmt:
            normalized.setdefault("type", f"audio/{fmt}")
            normalized.setdefault("name", f"chat_audio.{fmt}")
        elif fmt:
            normalized.setdefault("type", fmt)
            normalized.setdefault("name", "chat_audio.wav")
        else:
            normalized.setdefault("name", "chat_audio.wav")
        if "base64" in normalized and "data" not in normalized:
            normalized["data"] = normalized["base64"]
        return normalized

    if item_type == "audio_url":
        audio_url = item.get("audio_url")
        if isinstance(audio_url, dict):
            return audio_url.get("url")
        return audio_url

    return None


def _extract_chat_audio(
    payload: dict[str, Any],
) -> tuple[bytes, str, str | None]:
    prompt_text = None

    # 1) direct payload form
    if "audio_file" in payload:
        audio_bytes, file_name = _parse_audio_file_payload(payload["audio_file"], payload)
        prompt_text = payload.get("prompt")
        if isinstance(prompt_text, str):
            prompt_text = prompt_text.strip() or None
        return audio_bytes, file_name, prompt_text

    for direct_key in ("input_audio", "audio", "audio_url"):
        if direct_key not in payload:
            continue
        if direct_key == "audio_url":
            spec = payload.get("audio_url")
            if isinstance(spec, dict):
                spec = spec.get("url")
        else:
            spec = payload.get(direct_key)
            if isinstance(spec, dict) and direct_key == "input_audio":
                wrapped = {"type": "input_audio", "input_audio": spec}
                spec = _audio_spec_from_chat_item(wrapped)
        if spec is None:
            continue
        audio_bytes, file_name = _parse_audio_file_payload(spec, payload)
        prompt_text = payload.get("prompt")
        if isinstance(prompt_text, str):
            prompt_text = prompt_text.strip() or None
        return audio_bytes, file_name, prompt_text

    # 2) OpenAI chat messages
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return b"", "audio.wav", None

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        prompt_parts: list[str] = []
        found_audio: tuple[bytes, str] | None = None

        if isinstance(content, str):
            text = content.strip()
            if text:
                prompt_parts.append(text)

        elif isinstance(content, dict):
            content = [content]

        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = _normalize_chat_content_text(item)
                if text:
                    prompt_parts.append(text)
                    continue

                audio_spec = _audio_spec_from_chat_item(item)
                if audio_spec is None:
                    continue
                try:
                    found_audio = _parse_audio_file_payload(audio_spec, payload)
                except HTTPException:
                    continue

        if found_audio:
            prompt = " ".join(prompt_parts).strip() or None
            return found_audio[0], found_audio[1], prompt

    return b"", "audio.wav", None


def _chat_completion_payload(
    *,
    model: str,
    text: str,
    request_id: str,
) -> dict[str, Any]:
    now = int(time.time())
    completion_id = f"chatcmpl-{request_id}"
    prompt_tokens = max(1, len(text) // 4)
    completion_tokens = max(1, len(text) // 4)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _decode_base64_audio(value: str, field_name: str) -> bytes:
    candidate = value.strip()
    if "," in candidate and ";base64" in candidate[:80]:
        candidate = candidate.split(",", 1)[1]

    candidate = "".join(candidate.split())
    if not candidate:
        return b""

    padding = (-len(candidate)) % 4
    if padding:
        candidate += "=" * padding

    try:
        return base64.b64decode(candidate, validate=True)
    except binascii.Error:
        try:
            return base64.urlsafe_b64decode(candidate)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid {field_name} base64: {exc}") from exc


def _decode_int_list_audio(values: list[Any], field_name: str) -> bytes:
    if not all(isinstance(v, int) and 0 <= v <= 255 for v in values):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: integer list must contain values in [0,255]",
        )
    return bytes(values)


def _decode_sample_list_audio(values: list[Any], field_name: str) -> bytes:
    if not values:
        return b""

    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}: sample list must contain numbers",
        )

    # Strategy:
    # 1) [-1, 1] => normalized float samples
    # 2) otherwise => treat as PCM magnitude and clip to int16 range
    max_abs = max(abs(float(v)) for v in values)
    normalized = max_abs <= 1.0

    pcm = bytearray()
    for value in values:
        if normalized:
            scaled = int(max(-1.0, min(1.0, float(value))) * 32767)
        else:
            scaled = int(round(float(value)))
            if scaled > 32767:
                scaled = 32767
            elif scaled < -32768:
                scaled = -32768
        pcm.extend(int(scaled).to_bytes(2, byteorder="little", signed=True))
    return bytes(pcm)


def _normalize_uploaded_filename(name: str | None, mime_type: str | None) -> str:
    default_name = "audio.wav"
    if not name:
        name = default_name

    suffix = Path(name).suffix.lower()
    if suffix:
        return name

    inferred_suffix = MIME_TO_SUFFIX.get((mime_type or "").lower(), ".wav")
    return f"{name}{inferred_suffix}"


def _detect_audio_container(audio_bytes: bytes) -> str | None:
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "wav"
    if audio_bytes[:4] == b"fLaC":
        return "flac"
    if audio_bytes[:4] == b"OggS":
        return "ogg"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    if len(audio_bytes) >= 12 and audio_bytes[4:8] == b"ftyp":
        return "mp4/m4a"
    if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
        return "webm/mkv"
    return None


def _is_standard_pcm16_mono_16k_wav(audio_bytes: bytes) -> bool:
    if _detect_audio_container(audio_bytes) != "wav":
        return False
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            return (
                wav_file.getnchannels() == 1
                and wav_file.getframerate() == 16000
                and wav_file.getsampwidth() == 2
                and wav_file.getcomptype() == "NONE"
            )
    except Exception:
        return False


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _wrap_raw_pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    frame_bytes = max(1, channels * sample_width)
    pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % frame_bytes)]

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(max(1, channels))
            wav_file.setsampwidth(sample_width if sample_width in (1, 2, 4) else 2)
            wav_file.setframerate(max(8000, sample_rate))
            wav_file.writeframes(pcm_bytes)
        return buffer.getvalue()


def _extract_audio_meta(payload: dict[str, Any], audio_spec: dict[str, Any]) -> tuple[int, int, int]:
    sample_rate = _as_int(
        audio_spec.get("sample_rate")
        or audio_spec.get("sampling_rate")
        or payload.get("sample_rate")
        or payload.get("sampling_rate")
        or payload.get("rate"),
        16000,
    )
    channels = _as_int(
        audio_spec.get("channels")
        or audio_spec.get("channel_count")
        or payload.get("channels")
        or payload.get("channel_count"),
        1,
    )
    sample_width = _as_int(
        audio_spec.get("sample_width")
        or payload.get("sample_width")
        or audio_spec.get("bytes_per_sample")
        or payload.get("bytes_per_sample")
        or (int(audio_spec.get("bits_per_sample")) // 8 if audio_spec.get("bits_per_sample") else 0)
        or (int(payload.get("bits_per_sample")) // 8 if payload.get("bits_per_sample") else 0),
        2,
    )
    if sample_width not in (1, 2, 4):
        sample_width = 2
    return sample_rate, max(1, channels), sample_width


def _parse_audio_file_payload(audio_spec: Any, payload: dict[str, Any]) -> tuple[bytes, str]:
    if isinstance(audio_spec, str):
        audio_bytes = _decode_base64_audio(audio_spec, "audio_file")
        file_name = "audio.wav"
        if _detect_audio_container(audio_bytes) is None and audio_bytes:
            sample_rate, channels, sample_width = _extract_audio_meta(payload, {})
            audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
        return audio_bytes, file_name

    if isinstance(audio_spec, dict):
        mime_type = audio_spec.get("type") or audio_spec.get("mime_type")
        file_name = _normalize_uploaded_filename(audio_spec.get("name"), mime_type)
        sample_rate, channels, sample_width = _extract_audio_meta(payload, audio_spec)

        if "data" in audio_spec:
            raw_data = audio_spec.get("data")
            if isinstance(raw_data, str):
                audio_bytes = _decode_base64_audio(raw_data, "audio_file.data")
                if _detect_audio_container(audio_bytes) is None and audio_bytes:
                    audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
                    file_name = "audio.wav"
                return audio_bytes, file_name
            if isinstance(raw_data, list):
                if all(isinstance(v, int) and 0 <= v <= 255 for v in raw_data):
                    audio_bytes = _decode_int_list_audio(raw_data, "audio_file.data")
                else:
                    # Sample arrays (float/int) from recorder buffers.
                    audio_bytes = _decode_sample_list_audio(raw_data, "audio_file.data")
                    audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, 2)
                    return audio_bytes, "audio.wav"

                if _detect_audio_container(audio_bytes) is None and audio_bytes:
                    audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
                    file_name = "audio.wav"
                return audio_bytes, file_name
            if isinstance(raw_data, dict) and raw_data.get("type") == "Buffer" and isinstance(raw_data.get("data"), list):
                audio_bytes = _decode_int_list_audio(raw_data["data"], "audio_file.data.data")
                if _detect_audio_container(audio_bytes) is None and audio_bytes:
                    audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
                    file_name = "audio.wav"
                return audio_bytes, file_name

        if "base64" in audio_spec and isinstance(audio_spec.get("base64"), str):
            audio_bytes = _decode_base64_audio(audio_spec["base64"], "audio_file.base64")
            if _detect_audio_container(audio_bytes) is None and audio_bytes:
                audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
                file_name = "audio.wav"
            return audio_bytes, file_name

        if "bytes" in audio_spec and isinstance(audio_spec.get("bytes"), list):
            if all(isinstance(v, int) and 0 <= v <= 255 for v in audio_spec["bytes"]):
                audio_bytes = _decode_int_list_audio(audio_spec["bytes"], "audio_file.bytes")
            else:
                audio_bytes = _decode_sample_list_audio(audio_spec["bytes"], "audio_file.bytes")
                audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, 2)
                return audio_bytes, "audio.wav"

            if _detect_audio_container(audio_bytes) is None and audio_bytes:
                audio_bytes = _wrap_raw_pcm_to_wav(audio_bytes, sample_rate, channels, sample_width)
                file_name = "audio.wav"
            return audio_bytes, file_name

        raise HTTPException(
            status_code=400,
            detail="Invalid audio_file object: expected data(str/list), base64(str), or bytes(list)",
        )

    raise HTTPException(status_code=400, detail="Invalid audio_file format in JSON body")


def _transcode_with_ffmpeg(input_path: str) -> tuple[str | None, str | None]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as converted:
        converted_path = converted.name

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        converted_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        Path(converted_path).unlink(missing_ok=True)
        return None, "ffmpeg transcode timed out after 20 seconds"

    if proc.returncode == 0 and Path(converted_path).exists() and Path(converted_path).stat().st_size > 44:
        return converted_path, None

    Path(converted_path).unlink(missing_ok=True)
    stderr = (proc.stderr or proc.stdout or "").strip()
    return None, stderr or "ffmpeg failed without stderr"


async def _run_blocking_safely(
    request_id: str,
    func: Any,
    *args: Any,
) -> Any:
    """Run native work without deleting its inputs while cancellation is unwinding."""

    cancellation_event = Event()
    request_token = _CURRENT_REQUEST_ID.set(request_id)
    cancellation_token = _CURRENT_CANCELLATION_EVENT.set(cancellation_event)
    worker = asyncio.create_task(asyncio.to_thread(func, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        phase = _GLOBAL_INFERENCE_LOCK.phase_for(request_id) or "processing"
        runtime_metrics.note_cancellation_phase(request_id, phase)
        cancellation_event.set()

        # A queued inference can now leave the instrumented lock. Native work that
        # has already started is not safely interruptible, so keep its files alive
        # until the worker exits and then propagate the request cancellation.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break

        if not worker.cancelled():
            try:
                worker.result()
            except InferenceWaitCancelled:
                pass
            except Exception as exc:
                logger.warning(
                    "[%s] blocking operation finished with error after cancellation: %s",
                    request_id,
                    exc,
                )
        raise
    finally:
        _CURRENT_CANCELLATION_EVENT.reset(cancellation_token)
        _CURRENT_REQUEST_ID.reset(request_token)


async def _run_transcription_pipeline(
    *,
    request_id: str,
    audio_bytes: bytes,
    file_name: str,
    language: str | None,
    prompt: str | None,
    alignment_mode: str,
    response_format: str,
    timestamp_granularities: list[str] | None,
    decode_options: dict[str, Any] | None = None,
) -> TranscriptionResult:
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file/audio_file is empty")

    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_UPLOAD_MB} MB",
        )

    decode_options = dict(decode_options or {})
    detected_container = _detect_audio_container(audio_bytes)
    logger.info(
        "[%s] Audio decoded bytes=%d file_name=%s container=%s language=%s",
        request_id,
        len(audio_bytes),
        file_name,
        detected_container or "unknown",
        language or "auto",
    )
    timestamps_requested = _timestamp_requests_alignment(timestamp_granularities)

    suffix = Path(file_name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
        audio_path = audio_file.name
        audio_file.write(audio_bytes)

    converted_audio_path: str | None = None
    ffmpeg_error: str | None = None
    ffmpeg_status = "not_run"
    try:
        if _is_standard_pcm16_mono_16k_wav(audio_bytes):
            infer_audio_path = audio_path
            ffmpeg_error = "none"
            ffmpeg_status = "skipped"
        else:
            converted_audio_path, ffmpeg_error = await _run_blocking_safely(
                request_id,
                _transcode_with_ffmpeg,
                audio_path,
            )
            infer_audio_path = converted_audio_path or audio_path
            ffmpeg_status = "ok" if converted_audio_path else "failed"

        logger.info(
            "[%s] ffmpeg_transcode=%s ffmpeg_error=%s infer_audio_path=%s",
            request_id,
            ffmpeg_status,
            ffmpeg_error or "none",
            infer_audio_path,
        )

        result = await _run_blocking_safely(
            request_id,
            transcriber.transcribe,
            infer_audio_path,
            language,
            prompt,
            decode_options,
        )

        wav_duration_sec = _get_wav_duration_seconds(infer_audio_path)
        audio_duration_sec = wav_duration_sec or result.duration
        detected_language_code = _finalize_result_language(language, result)
        auto_align_lang_codes = _effective_auto_align_lang_codes(aligner)
        result.alignment_status = "disabled" if alignment_mode == "off" else "not_requested"
        should_align = _compute_alignment_enabled(
            alignment_mode=alignment_mode,
            timestamps_requested=timestamps_requested,
            response_format=response_format,
            detected_language_code=detected_language_code,
            auto_align_lang_codes=auto_align_lang_codes,
        )
        if (
            not should_align
            and alignment_mode == "auto"
            and response_format in AUTO_ALIGNMENT_RESPONSE_FORMATS
        ):
            if aligner._model is None:
                result.alignment_status = "unavailable"
            elif detected_language_code is None:
                result.alignment_status = "unknown_language"
            else:
                result.alignment_status = "unsupported_language"
        logger.info(
            "[%s] alignment_mode=%s response_format=%s timestamps_requested=%s detected_lang=%s "
            "auto_align_langs=%s should_align=%s",
            request_id,
            alignment_mode,
            response_format,
            timestamps_requested,
            detected_language_code or "unknown",
            ",".join(sorted(auto_align_lang_codes)) or "none",
            should_align,
        )

        if should_align:
            result.alignment_status = "requested"
            if not aligner.is_ready():
                aligner_error = aligner.last_error() or "aligner not initialized"
                if alignment_mode == "on" or timestamps_requested:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Forced alignment requested but aligner model is unavailable. "
                            f"error={aligner_error}. "
                            f"{_manual_model_setup_help('QWEN_MLX_ALIGNER_PATH')}"
                        ),
                    )
                result.alignment_status = "unavailable"
                logger.warning("[%s] Alignment skipped: %s", request_id, aligner_error)
            else:
                if _should_rechunk_segments_for_alignment(
                    result.segments,
                    wav_duration_sec,
                    ALIGNMENT_CHUNK_SECONDS,
                ):
                    logger.info(
                        "[%s] Re-running transcription with chunk_duration=%ss for alignment",
                        request_id,
                        ALIGNMENT_CHUNK_SECONDS,
                    )
                    rechunk_options = dict(decode_options)
                    rechunk_options["chunk_duration"] = ALIGNMENT_CHUNK_SECONDS
                    result = await _run_blocking_safely(
                        request_id,
                        transcriber.transcribe,
                        infer_audio_path,
                        language,
                        prompt,
                        rechunk_options,
                    )
                    detected_language_code = _finalize_result_language(language, result)

                aligner_language = _to_aligner_language_name(
                    language or result.language or detected_language_code
                )
                aligner_language_code = _to_language_code(aligner_language)
                missing_dep = _alignment_dependency_missing(aligner_language_code)
                if missing_dep is not None:
                    if alignment_mode == "on" or timestamps_requested:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Forced alignment requested but {missing_dep}.",
                        )
                    result.alignment_status = "unavailable"
                    logger.warning("[%s] Alignment skipped: %s", request_id, missing_dep)
                    return result
                try:
                    alignment_output = await _run_blocking_safely(
                        request_id,
                        _apply_alignment,
                        aligner,
                        infer_audio_path,
                        result,
                        aligner_language,
                        wav_duration_sec,
                        ALIGNMENT_CHUNK_SECONDS,
                    )
                except Exception as align_exc:
                    if alignment_mode == "on" or timestamps_requested:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Forced alignment failed: {align_exc}",
                        ) from align_exc
                    result.alignment_status = "failed"
                    logger.warning(
                        "[%s] Forced alignment skipped after error: %s",
                        request_id,
                        align_exc,
                    )
                else:
                    aligned_words = alignment_output.words
                    result.words = aligned_words
                    result.word_timestamp_source = "forced_alignment" if aligned_words else None
                    result.alignment_status = "applied" if aligned_words else "empty"
                    if SEGMENT_REBUILD_FROM_WORDS and aligned_words:
                        source_segments = [dict(segment) for segment in result.segments]
                        rebuilt_segments, rebuild_error = _rebuild_segments_from_alignment(
                            alignment=alignment_output,
                            aligner=aligner,
                            aligner_language=aligner_language,
                            lang_code=detected_language_code,
                        )
                        if rebuilt_segments:
                            rebuilt_segments = _restore_segment_languages_from_overlap(
                                source_segments,
                                rebuilt_segments,
                            )
                            result.segments = rebuilt_segments
                            result.segment_timestamp_source = "forced_alignment"
                            logger.info(
                                "[%s] sentence_segments=on segments_rebuilt=%d",
                                request_id,
                                len(result.segments),
                            )
                        elif rebuild_error is not None:
                            logger.warning(
                                "[%s] sentence_segments=skipped reason=%s",
                                request_id,
                                rebuild_error,
                            )
                    logger.info(
                        "[%s] forced_alignment=%s words=%d segments=%d lang=%s duration=%.2fs",
                        request_id,
                        "on" if bool(aligned_words) else "empty",
                        len(aligned_words),
                        len(result.segments),
                        aligner_language,
                        float(audio_duration_sec or 0.0),
                    )
        return result

    except AttributeError as exc:
        logger.error(
            "[%s] AttributeError during transcription: %s\n%s",
            request_id,
            exc,
            traceback.format_exc(),
        )
        if "ndim" in str(exc):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid or unsupported audio payload. "
                    f"Please ensure Spokenly sends valid audio bytes/base64. "
                    f"decoded_size={len(audio_bytes)} bytes, "
                    f"detected_container={detected_container or 'unknown'}, "
                    f"file_name={file_name}, "
                    f"ffmpeg_transcode={ffmpeg_status}, "
                    f"request_id={request_id}"
                ),
            ) from exc
        raise HTTPException(
            status_code=500,
            detail=(
                f"Transcription failed: {exc}. "
                f"ffmpeg_transcode={ffmpeg_status}; "
                f"ffmpeg_error={ffmpeg_error or 'none'}; "
                f"request_id={request_id}"
            ),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "[%s] Exception during transcription: %s\n%s",
            request_id,
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Transcription failed: {exc}. "
                f"ffmpeg_transcode={ffmpeg_status}; "
                f"ffmpeg_error={ffmpeg_error or 'none'}; "
                f"request_id={request_id}"
            ),
        ) from exc
    finally:
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError:
            pass
        if converted_audio_path:
            try:
                Path(converted_audio_path).unlink(missing_ok=True)
            except OSError:
                pass


_ADMISSION_PATHS = {
    "/v1/audio/transcriptions",
    "/audio/transcriptions",
    "/v1/chat/completions",
    "/chat/completions",
}


class AdmissionMetricsMiddleware:
    """Bound transcription work before request bodies are materialized."""

    def __init__(self, app: Any, metrics: RuntimeMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in _ADMISSION_PATHS
        ):
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex[:8]
        lease = self.metrics.try_admit(request_id)
        if lease is None:
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Transcription admission capacity is full. "
                        "Retry after an in-flight request completes."
                    )
                },
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return

        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        state["admission_lease"] = lease
        status_code: int | None = None
        disconnect_observed = False

        async def receive_with_disconnect_metrics() -> dict[str, Any]:
            nonlocal disconnect_observed
            message = await receive()
            if message.get("type") == "http.disconnect" and not disconnect_observed:
                disconnect_observed = True
                self.metrics.note_client_disconnect()
            return message

        async def send_with_status_metrics(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        outcome = "failed"
        try:
            await self.app(
                scope,
                receive_with_disconnect_metrics,
                send_with_status_metrics,
            )
            outcome = "completed" if status_code is not None and status_code < 400 else "failed"
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            lease.finish(outcome)


transcriber = MLXTranscriber(model_path=MODEL_PATH)
aligner = MLXAligner(model_path=ALIGNER_PATH)
app = FastAPI(title="Qwen3-ASR-MLX-Server", version="1.1.1")

app.add_middleware(AdmissionMetricsMiddleware, metrics=runtime_metrics)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    transcriber.load(strict=False)
    if transcriber.is_ready():
        logger.info("ASR model enabled: %s", MODEL_PATH)
    else:
        logger.warning(
            "ASR model disabled: %s",
            transcriber.last_error() or f"model not found at {MODEL_PATH}",
        )
    aligner.load(strict=False)
    if aligner.is_ready():
        logger.info("Forced aligner enabled: %s", ALIGNER_PATH)
    else:
        logger.info(
            "Forced aligner disabled: %s",
            aligner.last_error() or f"model not found at {ALIGNER_PATH}",
        )


def _health_snapshot() -> tuple[dict[str, Any], int]:
    model_state = {
        "required": True,
        "model_id": MODEL_ID,
        **transcriber.readiness_snapshot(),
    }
    aligner_state = {
        "required": False,
        "configured_languages": sorted(AUTO_ALIGN_LANG_CODES),
        **aligner.readiness_snapshot(),
    }
    metrics = runtime_metrics.snapshot()
    queue_metrics = _GLOBAL_INFERENCE_LOCK.snapshot()
    metrics["queue"] = queue_metrics

    admission = metrics["admission"]
    queue_ready = admission["in_flight"] < admission["capacity"]
    queue_state = {
        "ready": queue_ready,
        "accepting": queue_ready,
        "capacity": admission["capacity"],
        "max_queue_depth": admission["max_queue_depth"],
        "in_flight": admission["in_flight"],
        "active_inference": queue_metrics["active"],
        "waiting_for_inference": queue_metrics["waiting"],
    }
    ready = bool(model_state["ready"] and queue_ready)
    payload = {
        "status": "ok" if ready else "unavailable",
        "live": True,
        "ready": ready,
        "components": {
            "model": model_state,
            "aligner": aligner_state,
            "queue": queue_state,
        },
        "metrics": metrics,
    }
    return payload, 200 if ready else 503


@app.get("/healthz")
async def healthz() -> Response:
    payload, status_code = _health_snapshot()
    return JSONResponse(content=payload, status_code=status_code)


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    model_ids = _list_available_model_ids()
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
            for model_id in model_ids
        ],
    }


@app.post("/v1/audio/transcriptions")
@app.post("/audio/transcriptions")
async def transcriptions(
    request: Request,
    file: UploadFile | None = File(default=None),
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    task: str | None = Form(default=None),
    alignment: str | None = Form(default=None),
    enable_alignment: str | None = Form(default=None),
    word_timestamps: str | bool | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    temperature: float | None = Form(default=None),
    top_p: float | None = Form(default=None),
    top_k: int | None = Form(default=None),
    min_p: float | None = Form(default=None),
    min_tokens_to_keep: int | None = Form(default=None),
    repetition_penalty: float | None = Form(default=None),
    repetition_context_size: int | None = Form(default=None),
    prefill_step_size: int | None = Form(default=None),
    chunk_duration: float | None = Form(default=None),
    min_chunk_duration: float | None = Form(default=None),
    timestamp_granularities: list[str] | None = Form(
        default=None,
        alias="timestamp_granularities[]",
    ),
) -> Response:
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8]
    _validate_task(task)
    requested_model_name = _normalize_requested_model_name(model)

    decode_options = _collect_decode_options(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        min_tokens_to_keep=min_tokens_to_keep,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        prefill_step_size=prefill_step_size,
        chunk_duration=chunk_duration,
        min_chunk_duration=min_chunk_duration,
    )

    timestamp_granularities = _normalize_timestamp_granularities(timestamp_granularities)
    alignment_mode = _parse_alignment_mode(
        alignment if alignment is not None else enable_alignment
    )
    if alignment is None and enable_alignment is None:
        word_ts_flag = _parse_optional_boolean(word_timestamps)
        if word_ts_flag is not None:
            alignment_mode = "on" if word_ts_flag else "off"

    if response_format not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported response_format '{response_format}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_RESPONSE_FORMATS))}"
            ),
        )

    audio_bytes = b""
    file_name = "audio.wav"
    content_type = request.headers.get("content-type", "")
    user_agent = request.headers.get("user-agent", "") or "unknown"
    client_host = request.client.host if request.client else "unknown"

    logger.info(
        "[%s] POST /v1/audio/transcriptions content_type=%s content_length=%s",
        request_id,
        content_type,
        request.headers.get("content-length", "unknown"),
    )
    logger.info(
        "[%s] request_meta client=%s user_agent=%s",
        request_id,
        client_host,
        user_agent,
    )

    if file is not None:
        _validate_timestamp_response_contract(
            response_format=response_format,
            alignment_mode=alignment_mode,
            timestamp_granularities=timestamp_granularities,
        )
        audio_bytes = await file.read()
        file_name = file.filename or file_name

    elif "application/json" in content_type:
        payload = await request.json()
        _validate_task(payload.get("task"))
        requested_model_name = _normalize_requested_model_name(
            payload.get("model", requested_model_name)
        )

        audio_spec = payload.get("audio_file")
        if audio_spec is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: file (multipart) or audio_file (json)",
            )

        decode_options = _apply_json_decode_options(decode_options, payload)

        language = language or payload.get("language")
        prompt = prompt or payload.get("prompt") or payload.get("context")

        json_granularities = payload.get("timestamp_granularities")
        if json_granularities is None:
            json_granularities = payload.get("timestamp_granularities[]")
        if json_granularities is not None:
            timestamp_granularities = _normalize_timestamp_granularities(json_granularities)

        if alignment is None and enable_alignment is None:
            if "alignment" in payload:
                alignment_mode = _parse_alignment_mode(payload.get("alignment"))
            elif "enable_alignment" in payload:
                alignment_mode = _parse_alignment_mode(payload.get("enable_alignment"))
            else:
                word_ts_flag = _parse_optional_boolean(payload.get("word_timestamps"))
                if word_ts_flag is not None:
                    alignment_mode = "on" if word_ts_flag else "off"

        if "response_format" in payload:
            response_format = str(payload["response_format"]).strip()
            if response_format not in SUPPORTED_RESPONSE_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported response_format '{response_format}'. "
                        f"Supported: {', '.join(sorted(SUPPORTED_RESPONSE_FORMATS))}"
                    ),
                )
        _validate_timestamp_response_contract(
            response_format=response_format,
            alignment_mode=alignment_mode,
            timestamp_granularities=timestamp_granularities,
        )
        audio_bytes, file_name = _parse_audio_file_payload(audio_spec, payload)
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing required form field: file",
        )

    admission_lease = getattr(request.state, "admission_lease", None)
    if isinstance(admission_lease, AdmissionLease):
        admission_lease.set_audio_bytes(len(audio_bytes))

    logger.info(
        "[%s] request_params model=%s language=%s response_format=%s alignment_mode=%s "
        "timestamp_granularities=%s file_name=%s decoded_bytes=%d",
        request_id,
        requested_model_name,
        language or "auto",
        response_format,
        alignment_mode,
        ",".join(timestamp_granularities) if timestamp_granularities else "none",
        file_name,
        len(audio_bytes),
    )
    _log_prompt_preview(request_id, prompt)

    _ensure_transcriber_ready_or_raise(requested_model_name)

    result = await _run_transcription_pipeline(
        request_id=request_id,
        audio_bytes=audio_bytes,
        file_name=file_name,
        language=language,
        prompt=prompt,
        alignment_mode=alignment_mode,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
        decode_options=decode_options,
    )
    return _build_transcription_response(result, response_format)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request) -> Response:
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        raise HTTPException(status_code=400, detail="chat/completions only supports JSON payload")

    payload = await request.json()
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:8]
    user_agent = request.headers.get("user-agent", "") or "unknown"
    client_host = request.client.host if request.client else "unknown"

    _validate_task(payload.get("task"))
    model = _normalize_requested_model_name(payload.get("model"))
    language = payload.get("language")
    prompt = payload.get("prompt") or payload.get("context")
    if isinstance(prompt, str):
        prompt = prompt.strip() or None

    decode_options = _apply_json_decode_options({}, payload)

    alignment_mode = _parse_alignment_mode(
        payload.get("alignment") if "alignment" in payload else payload.get("enable_alignment")
    )
    if "alignment" not in payload and "enable_alignment" not in payload:
        word_ts_flag = _parse_optional_boolean(payload.get("word_timestamps"))
        if word_ts_flag is not None:
            alignment_mode = "on" if word_ts_flag else "off"

    timestamp_granularities = _normalize_timestamp_granularities(
        payload.get("timestamp_granularities") or payload.get("timestamp_granularities[]")
    )
    _validate_timestamp_response_contract(
        response_format="json",
        alignment_mode=alignment_mode,
        timestamp_granularities=timestamp_granularities,
        chat_response=True,
    )
    canonical_model = _ensure_transcriber_ready_or_raise(model)

    audio_bytes, file_name, chat_prompt = _extract_chat_audio(payload)
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                "No audio found in chat payload. Provide audio_file or messages.content "
                "with type=input_audio/audio_url."
            ),
        )
    if not prompt:
        prompt = chat_prompt

    admission_lease = getattr(request.state, "admission_lease", None)
    if isinstance(admission_lease, AdmissionLease):
        admission_lease.set_audio_bytes(len(audio_bytes))

    logger.info(
        "[%s] POST /v1/chat/completions client=%s user_agent=%s model=%s language=%s "
        "alignment_mode=%s timestamp_granularities=%s file_name=%s decoded_bytes=%d",
        request_id,
        client_host,
        user_agent,
        model,
        language or "auto",
        alignment_mode,
        ",".join(timestamp_granularities) if timestamp_granularities else "none",
        file_name,
        len(audio_bytes),
    )
    _log_prompt_preview(request_id, prompt)

    result = await _run_transcription_pipeline(
        request_id=request_id,
        audio_bytes=audio_bytes,
        file_name=file_name,
        language=language,
        prompt=prompt,
        alignment_mode=alignment_mode,
        response_format="json",
        timestamp_granularities=timestamp_granularities,
        decode_options=decode_options,
    )

    return JSONResponse(
        content=_chat_completion_payload(
            model=canonical_model,
            text=result.text,
            request_id=request_id,
        )
    )


def main() -> None:
    import uvicorn

    global LOG_PROMPTS

    parser = argparse.ArgumentParser(description="Qwen3-ASR-MLX-Server")
    parser.add_argument(
        "command",
        nargs="?",
        default="serve",
        choices=["serve", "setup"],
        help="Run API server or interactive model setup",
    )
    parser.add_argument(
        "--source",
        choices=["huggingface", "modelscope"],
        default=None,
        help="Download source for setup command",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Setup mode: download all missing models with defaults",
    )
    parser.add_argument(
        "--log-prompts",
        action="store_true",
        help="Serve mode: log request prompt previews at INFO level (disabled by default)",
    )
    args = parser.parse_args()

    if args.command == "setup":
        raise SystemExit(run_setup_wizard(source=args.source, non_interactive=args.non_interactive))

    if args.log_prompts:
        LOG_PROMPTS = True

    if LOG_PROMPTS:
        logger.warning(
            "Prompt logging is enabled; request prompt previews will be written to INFO logs"
        )

    maybe_prompt_initial_setup()
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
