from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import whisper_mlx_server as server


class ModelResolutionTests(unittest.TestCase):
    @staticmethod
    def _write_minimal_model_dir(
        path: Path,
        *,
        sharded: bool = False,
        missing_shard: bool = False,
    ) -> None:
        path.mkdir(parents=True)
        for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json"):
            (path / name).write_text("{}", encoding="utf-8")

        if sharded:
            shard_name = "model-00001-of-00001.safetensors"
            (path / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.layers.0": shard_name}}),
                encoding="utf-8",
            )
            if not missing_shard:
                (path / shard_name).write_bytes(b"weights")
            return

        (path / "model.safetensors").write_bytes(b"weights")

    def test_huggingface_snapshot_dirs_prefers_ref_main(self) -> None:
        with TemporaryDirectory() as tmp:
            hf_home = Path(tmp) / "hf-home"
            repo_cache_dir = (
                hf_home
                / "hub"
                / "models--mlx-community--Qwen3-ASR-1.7B-bf16"
            )
            older_snapshot = repo_cache_dir / "snapshots" / "older"
            main_snapshot = repo_cache_dir / "snapshots" / "main-sha"
            older_snapshot.mkdir(parents=True)
            main_snapshot.mkdir(parents=True)
            (repo_cache_dir / "refs").mkdir(parents=True)
            (repo_cache_dir / "refs" / "main").write_text("main-sha", encoding="utf-8")

            with mock.patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                resolved = server._huggingface_snapshot_dirs(
                    "mlx-community/Qwen3-ASR-1.7B-bf16"
                )

            self.assertGreaterEqual(len(resolved), 2)
            self.assertEqual(resolved[0], main_snapshot)
            self.assertIn(older_snapshot, resolved)

    def test_default_asset_path_prefers_huggingface_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspace"
            repo_dir = workspace_root / "qwen3-asr-whisper-api-debug"
            workspace_model_dir = workspace_root / "models" / "Qwen3-ASR-1.7B-bf16"
            cache_snapshot_dir = (
                workspace_root
                / "hf-home"
                / "hub"
                / "models--mlx-community--Qwen3-ASR-1.7B-bf16"
                / "snapshots"
                / "cache-sha"
            )
            repo_dir.mkdir(parents=True)
            self._write_minimal_model_dir(workspace_model_dir)
            self._write_minimal_model_dir(cache_snapshot_dir)
            (workspace_root / "README_WORKSPACE.md").write_text("workspace", encoding="utf-8")
            refs_dir = cache_snapshot_dir.parent.parent / "refs"
            refs_dir.mkdir(parents=True)
            (refs_dir / "main").write_text("cache-sha", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"HF_HOME": str(workspace_root / "hf-home")},
                clear=False,
            ):
                resolved = server._default_asset_path(
                    "Qwen3-ASR-1.7B-bf16",
                    hf_repo="mlx-community/Qwen3-ASR-1.7B-bf16",
                    base_dir=repo_dir,
                )

            self.assertEqual(resolved, str(cache_snapshot_dir))

    def test_default_asset_path_falls_back_to_workspace_models(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspace"
            repo_dir = workspace_root / "qwen3-asr-whisper-api-debug"
            workspace_model_dir = workspace_root / "models" / "Qwen3-ForcedAligner-0.6B-bf16"
            repo_dir.mkdir(parents=True)
            self._write_minimal_model_dir(workspace_model_dir)
            (workspace_root / "README_WORKSPACE.md").write_text("workspace", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"HF_HOME": str(workspace_root / "hf-home")},
                clear=False,
            ):
                resolved = server._default_asset_path(
                    "Qwen3-ForcedAligner-0.6B-bf16",
                    hf_repo="mlx-community/Qwen3-ForcedAligner-0.6B-bf16",
                    base_dir=repo_dir,
                )

            self.assertEqual(resolved, str(workspace_model_dir))

    def test_default_asset_path_skips_incomplete_huggingface_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_root = Path(tmp) / "workspace"
            repo_dir = workspace_root / "qwen3-asr-whisper-api-debug"
            workspace_model_dir = workspace_root / "models" / "Qwen3-ASR-1.7B-bf16"
            cache_snapshot_dir = (
                workspace_root
                / "hf-home"
                / "hub"
                / "models--mlx-community--Qwen3-ASR-1.7B-bf16"
                / "snapshots"
                / "cache-sha"
            )
            repo_dir.mkdir(parents=True)
            self._write_minimal_model_dir(workspace_model_dir)
            self._write_minimal_model_dir(
                cache_snapshot_dir,
                sharded=True,
                missing_shard=True,
            )
            (workspace_root / "README_WORKSPACE.md").write_text("workspace", encoding="utf-8")
            refs_dir = cache_snapshot_dir.parent.parent / "refs"
            refs_dir.mkdir(parents=True)
            (refs_dir / "main").write_text("cache-sha", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {"HF_HOME": str(workspace_root / "hf-home")},
                clear=False,
            ):
                resolved = server._default_asset_path(
                    "Qwen3-ASR-1.7B-bf16",
                    hf_repo="mlx-community/Qwen3-ASR-1.7B-bf16",
                    base_dir=repo_dir,
                )

            self.assertEqual(resolved, str(workspace_model_dir))

    def test_missing_setup_models_treats_incomplete_directory_as_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            incomplete_model_dir = Path(tmp) / "Qwen3-ASR-1.7B-bf16"
            self._write_minimal_model_dir(
                incomplete_model_dir,
                sharded=True,
                missing_shard=True,
            )
            spec = server.SetupModelSpec(
                key="asr",
                display_name="Qwen3-ASR-1.7B-bf16 (MLX)",
                local_path=str(incomplete_model_dir),
                hf_repo="mlx-community/Qwen3-ASR-1.7B-bf16",
                ms_repo="mlx-community/Qwen3-ASR-1.7B-bf16",
                required=True,
            )

            with mock.patch.object(server, "SETUP_MODEL_SPECS", (spec,)):
                missing = server._missing_setup_models()

            self.assertEqual(missing, [spec])


if __name__ == "__main__":
    unittest.main()
