from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import whisper_mlx_server as server


class ModelResolutionTests(unittest.TestCase):
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
            workspace_model_dir.mkdir(parents=True)
            cache_snapshot_dir.mkdir(parents=True)
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
            workspace_model_dir.mkdir(parents=True)
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


if __name__ == "__main__":
    unittest.main()
