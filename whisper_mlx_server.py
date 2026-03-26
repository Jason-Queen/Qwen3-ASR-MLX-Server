from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
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
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response

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
SEGMENT_PAUSE_THRESHOLD_SECONDS = float(os.getenv("QWEN_MLX_SEGMENT_PAUSE_THRESHOLD_SECONDS", "0.75"))
SEGMENT_MAX_WORDS_PER_SEGMENT = int(os.getenv("QWEN_MLX_SEGMENT_MAX_WORDS_PER_SEGMENT", "28"))
HOST = os.getenv("QWEN_MLX_HOST", "127.0.0.1")
PORT = int(os.getenv("QWEN_MLX_PORT", "8989"))
MAX_UPLOAD_MB = int(os.getenv("QWEN_MLX_MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
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


class MLXTranscriber:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self._load_lock = Lock()
        self._infer_lock = Lock()
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
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._model: Any | None = None
        self._load_lock = Lock()
        self._infer_lock = Lock()
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

    if isinstance(raw_result, dict):
        text = _coalesce_text(raw_result)
        raw_language = raw_result.get("language")
        segments = _normalize_segments(raw_result.get("segments") or raw_result.get("chunks"))

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

    return [item.lower() for item in items if item]


def _timestamp_requests_word(granularities: list[str] | None) -> bool:
    if not granularities:
        return False
    normalized = {item.lower() for item in granularities}
    return "word" in normalized or "words" in normalized


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
    word_timestamps_requested: bool,
    detected_language_code: str | None,
    auto_align_lang_codes: set[str],
) -> bool:
    if alignment_mode == "off":
        return False
    if alignment_mode == "on" or word_timestamps_requested:
        return True
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


def _split_text_evenly(text: str, parts: int) -> list[str]:
    if parts <= 1:
        return [text]

    clean = text.strip()
    if not clean:
        return [""] * parts

    if " " in clean:
        tokens = clean.split()
        joiner = " "
    else:
        tokens = list(clean)
        joiner = ""

    total = len(tokens)
    if total == 0:
        return [""] * parts

    slices: list[str] = []
    for idx in range(parts):
        start = int(round(idx * total / parts))
        end = int(round((idx + 1) * total / parts))
        chunk_tokens = tokens[start:end]
        slices.append(joiner.join(chunk_tokens).strip())
    return slices


def _segment_text_for_chunks(
    result: TranscriptionResult,
    chunk_ranges: list[tuple[str, float, float]],
) -> list[str]:
    # Assign each coarse ASR segment to exactly one alignment chunk to avoid
    # duplicated text across neighboring chunks (which can produce overlapping
    # word timestamps after chunk offset stitching).
    assigned: list[list[str]] = [[] for _ in chunk_ranges]
    for segment in result.segments:
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
        assigned[best_idx].append(seg_text)

    chunk_texts: list[str] = [" ".join(items).strip() for items in assigned]

    non_empty = [text for text in chunk_texts if text]
    repeated_full_text = (
        len(non_empty) > 1
        and len(set(non_empty)) == 1
        and non_empty[0] == result.text.strip()
    )

    if len(non_empty) == len(chunk_texts) and not repeated_full_text:
        return chunk_texts

    fallback = _split_text_evenly(result.text, len(chunk_ranges))
    merged: list[str] = []
    for index, existing in enumerate(chunk_texts):
        merged.append(existing if existing and not repeated_full_text else fallback[index])
    return merged


def _split_wav_into_temp_chunks(
    wav_path: str,
    max_chunk_seconds: float,
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
        chunk_frames = max(1, int(frame_rate * max_chunk_seconds))
        total_frames = wav_file.getnframes()
        frame_offset = 0

        while frame_offset < total_frames:
            read_frames = min(chunk_frames, total_frames - frame_offset)
            frames = wav_file.readframes(read_frames)
            start_sec = frame_offset / float(frame_rate)
            end_sec = (frame_offset + read_frames) / float(frame_rate)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                chunk_path = tmp_file.name
            with wave.open(chunk_path, "wb") as out_file:
                out_file.setnchannels(channels)
                out_file.setsampwidth(sample_width)
                out_file.setframerate(frame_rate)
                out_file.setcomptype(comptype, compname)
                out_file.writeframes(frames)
            chunks.append((chunk_path, start_sec, end_sec))
            frame_offset += read_frames

    return chunks


def _apply_alignment(
    aligner: MLXAligner,
    audio_path: str,
    result: TranscriptionResult,
    aligner_language: str,
    duration_sec: float | None,
    max_chunk_seconds: float,
) -> list[dict[str, Any]]:
    if not result.text.strip():
        return []

    if not duration_sec or duration_sec <= max_chunk_seconds:
        return aligner.align(audio=audio_path, text=result.text, language=aligner_language)

    chunk_specs = _split_wav_into_temp_chunks(audio_path, max_chunk_seconds)
    chunk_paths = [chunk_path for chunk_path, _, _ in chunk_specs]
    try:
        chunk_texts = _segment_text_for_chunks(result, chunk_specs)
        aligned_words: list[dict[str, Any]] = []
        for (chunk_path, chunk_start, chunk_end), chunk_text in zip(chunk_specs, chunk_texts):
            if not chunk_text.strip():
                continue
            chunk_duration = max(0.0, float(chunk_end) - float(chunk_start))
            words = aligner.align(
                audio=chunk_path,
                text=chunk_text,
                language=aligner_language,
            )
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

                aligned_words.append(
                    {
                        "word": str(word.get("word", "")),
                        "start": local_start + chunk_start,
                        "end": local_end + chunk_start,
                    }
                )
        if duration_sec and duration_sec > 0.0:
            clamped: list[dict[str, Any]] = []
            for word in aligned_words:
                start = _to_float(word.get("start"), 0.0)
                end = _to_float(word.get("end"), start)
                start = max(0.0, min(start, duration_sec))
                end = max(start, min(end, duration_sec))
                item = {"word": str(word.get("word", "")), "start": start, "end": end}
                clamped.append(item)
            aligned_words = clamped
        aligned_words.sort(
            key=lambda item: (
                _to_float(item.get("start"), 0.0),
                _to_float(item.get("end"), 0.0),
            )
        )
        return aligned_words
    finally:
        for chunk_path in chunk_paths:
            try:
                Path(chunk_path).unlink(missing_ok=True)
            except OSError:
                pass


def _split_text_into_sentences(text: str) -> list[str]:
    content = (text or "").strip()
    if not content:
        return []

    # Keep sentence-ending punctuation with each sentence.
    sentence_break_pattern = r"([。！？!?；;]+|\.{2,}|\.)"
    parts = re.split(sentence_break_pattern, content)
    sentences: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        current += part
        if re.fullmatch(sentence_break_pattern, part):
            sentence = current.strip()
            if sentence:
                sentences.append(sentence)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _estimate_sentence_token_units(sentence: str, lang_code: str | None) -> int:
    text = sentence.strip()
    if not text:
        return 0

    code = (lang_code or "").lower()
    if code == "zh":
        tokens = re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", text)
        return len(tokens)
    if code == "ja":
        tokens = re.findall(r"[A-Za-z0-9']+|[\u3040-\u30ff\u31f0-\u31ff\u4e00-\u9fff]", text)
        return len(tokens)
    if code == "ko":
        tokens = re.findall(r"[A-Za-z0-9']+|[가-힣]+", text)
        return len(tokens)

    tokens = re.findall(r"[A-Za-z0-9']+|[\u4e00-\u9fff]", text)
    return len(tokens)


def _allocate_word_counts(total_words: int, weights: list[int]) -> list[int]:
    if total_words <= 0:
        return [0] * len(weights)
    if not weights:
        return []

    normalized = [max(0, int(w)) for w in weights]
    if not any(normalized):
        normalized = [1] * len(weights)

    total_weight = sum(normalized)
    raw = [(total_words * w) / total_weight for w in normalized]
    counts = [int(math.floor(x)) for x in raw]

    # If we have enough words, guarantee at least one word per sentence.
    if total_words >= len(counts):
        for i, count in enumerate(counts):
            if count == 0:
                counts[i] = 1

    diff = total_words - sum(counts)
    if diff > 0:
        fractions = [value - math.floor(value) for value in raw]
        order = sorted(range(len(counts)), key=lambda i: fractions[i], reverse=True)
        for i in order:
            if diff <= 0:
                break
            counts[i] += 1
            diff -= 1
    elif diff < 0:
        order = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
        for i in order:
            while diff < 0 and counts[i] > 1:
                counts[i] -= 1
                diff += 1
            if diff >= 0:
                break

    return counts


def _join_words_text(words: list[dict[str, Any]], lang_code: str | None) -> str:
    code = (lang_code or "").lower()
    items = [str(item.get("word", "")).strip() for item in words if str(item.get("word", "")).strip()]
    if code in {"zh", "ja"}:
        return "".join(items)
    return " ".join(items)


def _build_sentence_segments_from_words(
    words: list[dict[str, Any]],
    text: str,
    lang_code: str | None,
) -> list[dict[str, Any]]:
    if not words:
        return []

    sentences = _split_text_into_sentences(text)
    if not sentences:
        return []

    weights = [max(1, _estimate_sentence_token_units(sentence, lang_code)) for sentence in sentences]
    counts = _allocate_word_counts(len(words), weights)
    if not counts:
        return []

    segments: list[dict[str, Any]] = []
    cursor = 0
    for sentence, count in zip(sentences, counts):
        if cursor >= len(words):
            break
        if count <= 0:
            continue
        chunk_words = words[cursor : min(len(words), cursor + count)]
        if not chunk_words:
            continue
        start = _to_float(chunk_words[0].get("start"), 0.0)
        end = _to_float(chunk_words[-1].get("end"), start)
        seg_text = sentence.strip() or _join_words_text(chunk_words, lang_code)
        if not seg_text:
            seg_text = _join_words_text(chunk_words, lang_code)
        segments.append(
            {
                "id": len(segments),
                "start": max(0.0, start),
                "end": max(max(0.0, start), end),
                "text": seg_text,
                "_words": chunk_words,
            }
        )
        cursor += len(chunk_words)

    if cursor < len(words):
        remainder = words[cursor:]
        start = _to_float(remainder[0].get("start"), 0.0)
        end = _to_float(remainder[-1].get("end"), start)
        seg_text = _join_words_text(remainder, lang_code)
        if seg_text:
            segments.append(
                {
                    "id": len(segments),
                    "start": max(0.0, start),
                    "end": max(max(0.0, start), end),
                    "text": seg_text,
                    "_words": remainder,
                }
            )

    return [segment for segment in segments if segment.get("text")]


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


def _split_text_by_punctuation_pattern(text: str, pattern: str) -> list[str]:
    content = (text or "").strip()
    if not content:
        return []

    tokens = re.split(f"({pattern})", content)
    if len(tokens) <= 1:
        return [content]

    pieces: list[str] = []
    current = ""
    for token in tokens:
        if not token:
            continue
        current += token
        if re.fullmatch(pattern, token):
            item = current.strip()
            if item:
                pieces.append(item)
            current = ""
    if current.strip():
        pieces.append(current.strip())
    return pieces or [content]


def _build_subsegments_from_text_parts(
    segment: dict[str, Any],
    parts: list[str],
    lang_code: str | None,
) -> list[dict[str, Any]]:
    filtered_parts = [part.strip() for part in parts if part and part.strip()]
    if not filtered_parts:
        return []

    start = _to_float(segment.get("start"), 0.0)
    end = _to_float(segment.get("end"), start)
    duration = max(0.0, end - start)
    if duration <= 0:
        return [
            {"id": 0, "start": start, "end": start, "text": part}
            for part in filtered_parts
        ]

    weights = [max(1, _estimate_sentence_token_units(part, lang_code)) for part in filtered_parts]
    total_weight = float(sum(weights)) if weights else float(len(filtered_parts))
    segment_words = segment.get("_words")
    if isinstance(segment_words, list) and segment_words:
        counts = _allocate_word_counts(len(segment_words), weights)
        if counts:
            subsegments: list[dict[str, Any]] = []
            cursor_index = 0
            last_end = start
            for idx, (part, count) in enumerate(zip(filtered_parts, counts)):
                if idx == len(filtered_parts) - 1:
                    sub_words = segment_words[cursor_index:]
                else:
                    take = max(0, int(count))
                    sub_words = segment_words[cursor_index : min(len(segment_words), cursor_index + take)]
                if sub_words:
                    sub_start = _to_float(sub_words[0].get("start"), last_end)
                    sub_end = _to_float(sub_words[-1].get("end"), sub_start)
                    cursor_index += len(sub_words)
                else:
                    # Fallback when allocation yields an empty bucket.
                    if idx == len(filtered_parts) - 1:
                        sub_start = last_end
                        sub_end = end
                    else:
                        ratio = (
                            float(weights[idx]) / total_weight if total_weight > 0 else 1.0 / len(filtered_parts)
                        )
                        sub_start = last_end
                        sub_end = min(end, sub_start + (duration * ratio))
                if sub_end < sub_start:
                    sub_end = sub_start
                entry: dict[str, Any] = {
                    "id": idx,
                    "start": sub_start,
                    "end": sub_end,
                    "text": part,
                }
                if sub_words:
                    entry["_words"] = sub_words
                subsegments.append(entry)
                last_end = sub_end
            return subsegments

    subsegments: list[dict[str, Any]] = []
    cursor = start
    for idx, (part, weight) in enumerate(zip(filtered_parts, weights)):
        if idx == len(filtered_parts) - 1:
            sub_end = end
        else:
            ratio = float(weight) / total_weight if total_weight > 0 else 1.0 / len(filtered_parts)
            sub_end = min(end, cursor + (duration * ratio))
        if sub_end < cursor:
            sub_end = cursor
        subsegments.append(
            {
                "id": idx,
                "start": cursor,
                "end": sub_end,
                "text": part,
            }
        )
        cursor = sub_end

    return subsegments


def _split_long_segment_once(
    segment: dict[str, Any],
    lang_code: str | None,
    hard_max_seconds: float,
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
        return _build_subsegments_from_text_parts(segment, comma_parts, lang_code)

    # Priority 2: split by other punctuation.
    other_parts = _split_text_by_punctuation_pattern(text, r"[、；;:：。！？!?]+")
    if len(other_parts) > 1:
        return _build_subsegments_from_text_parts(segment, other_parts, lang_code)

    # Final fallback: even split by duration.
    chunks = max(2, int(math.ceil(duration / max(0.1, hard_max_seconds))))
    fallback_parts = _split_text_evenly(text, chunks)
    return _build_subsegments_from_text_parts(segment, fallback_parts, lang_code)


def _enforce_hard_duration_limit(
    segments: list[dict[str, Any]],
    lang_code: str | None,
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

        parts = _split_long_segment_once(current, lang_code, hard_max_seconds)
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
        item.pop("_words", None)
    return output


def _build_pause_segments_from_words(
    words: list[dict[str, Any]],
    lang_code: str | None,
    pause_threshold_sec: float = 0.75,
    max_words_per_segment: int = 28,
) -> list[dict[str, Any]]:
    if not words:
        return []

    segments: list[dict[str, Any]] = []
    chunk: list[dict[str, Any]] = []
    prev_end = None

    def flush_chunk() -> None:
        nonlocal chunk
        if not chunk:
            return
        start = _to_float(chunk[0].get("start"), 0.0)
        end = _to_float(chunk[-1].get("end"), start)
        seg_text = _join_words_text(chunk, lang_code)
        if seg_text:
            segments.append(
                {
                    "id": len(segments),
                    "start": max(0.0, start),
                    "end": max(max(0.0, start), end),
                    "text": seg_text,
                    "_words": list(chunk),
                }
            )
        chunk = []

    for word in words:
        start = _to_float(word.get("start"), 0.0)
        end = _to_float(word.get("end"), start)
        if chunk:
            gap = start - float(prev_end or start)
            if gap >= pause_threshold_sec or len(chunk) >= max_words_per_segment:
                flush_chunk()
        chunk.append({"word": word.get("word", ""), "start": start, "end": end})
        prev_end = end

    flush_chunk()
    return segments


def _rebuild_segments_from_words(
    *,
    words: list[dict[str, Any]],
    text: str,
    lang_code: str | None,
) -> list[dict[str, Any]]:
    if not words:
        return []

    rebuilt = _build_sentence_segments_from_words(words, text, lang_code)
    if not rebuilt:
        rebuilt = _build_pause_segments_from_words(
            words,
            lang_code,
            pause_threshold_sec=SEGMENT_PAUSE_THRESHOLD_SECONDS,
            max_words_per_segment=max(1, SEGMENT_MAX_WORDS_PER_SEGMENT),
        )
    if not rebuilt:
        return []

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
            hard_max_seconds=SEGMENT_HARD_MAX_SECONDS,
        )

    return _normalize_segments(rebuilt)


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


async def _run_transcription_pipeline(
    *,
    request_id: str,
    audio_bytes: bytes,
    file_name: str,
    language: str | None,
    prompt: str | None,
    alignment_mode: str,
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
    word_timestamps_requested = _timestamp_requests_word(timestamp_granularities)

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
            converted_audio_path, ffmpeg_error = await asyncio.to_thread(
                _transcode_with_ffmpeg, audio_path
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

        result = await asyncio.to_thread(
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
        should_align = _compute_alignment_enabled(
            alignment_mode=alignment_mode,
            word_timestamps_requested=word_timestamps_requested,
            detected_language_code=detected_language_code,
            auto_align_lang_codes=auto_align_lang_codes,
        )
        logger.info(
            "[%s] alignment_mode=%s word_ts_requested=%s detected_lang=%s auto_align_langs=%s should_align=%s",
            request_id,
            alignment_mode,
            word_timestamps_requested,
            detected_language_code or "unknown",
            ",".join(sorted(auto_align_lang_codes)) or "none",
            should_align,
        )

        if should_align:
            if not aligner.is_ready():
                aligner_error = aligner.last_error() or "aligner not initialized"
                if alignment_mode == "on" or word_timestamps_requested:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Forced alignment requested but aligner model is unavailable. "
                            f"error={aligner_error}. "
                            f"{_manual_model_setup_help('QWEN_MLX_ALIGNER_PATH')}"
                        ),
                    )
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
                    result = await asyncio.to_thread(
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
                    if alignment_mode == "on" or word_timestamps_requested:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Forced alignment requested but {missing_dep}.",
                        )
                    logger.warning("[%s] Alignment skipped: %s", request_id, missing_dep)
                    return result
                try:
                    aligned_words = await asyncio.to_thread(
                        _apply_alignment,
                        aligner,
                        infer_audio_path,
                        result,
                        aligner_language,
                        wav_duration_sec,
                        ALIGNMENT_CHUNK_SECONDS,
                    )
                except Exception as align_exc:
                    if alignment_mode == "on" or word_timestamps_requested:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Forced alignment failed: {align_exc}",
                        ) from align_exc
                    logger.warning(
                        "[%s] Forced alignment skipped after error: %s",
                        request_id,
                        align_exc,
                    )
                else:
                    result.words = aligned_words
                    if SEGMENT_REBUILD_FROM_WORDS and aligned_words:
                        source_segments = [dict(segment) for segment in result.segments]
                        rebuilt_segments = _rebuild_segments_from_words(
                            words=aligned_words,
                            text=result.text,
                            lang_code=detected_language_code,
                        )
                        if rebuilt_segments:
                            rebuilt_segments = _restore_segment_languages_from_overlap(
                                source_segments,
                                rebuilt_segments,
                            )
                            result.segments = rebuilt_segments
                            logger.info(
                                "[%s] sentence_segments=on segments_rebuilt=%d",
                                request_id,
                                len(result.segments),
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


transcriber = MLXTranscriber(model_path=MODEL_PATH)
aligner = MLXAligner(model_path=ALIGNER_PATH)
app = FastAPI(title="Qwen3-ASR-MLX-Server", version="1.0.1")

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


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


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
    request_id = uuid.uuid4().hex[:8]
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

        audio_bytes, file_name = _parse_audio_file_payload(audio_spec, payload)
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
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing required form field: file",
        )

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
    request_id = uuid.uuid4().hex[:8]
    user_agent = request.headers.get("user-agent", "") or "unknown"
    client_host = request.client.host if request.client else "unknown"

    _validate_task(payload.get("task"))
    model = _normalize_requested_model_name(payload.get("model"))
    canonical_model = _ensure_transcriber_ready_or_raise(model)
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
