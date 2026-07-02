from __future__ import annotations

import unittest
from unittest import mock

from fastapi.testclient import TestClient

import whisper_mlx_server as server


class MultipartTranscriptionEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def _post_with_form(self, data: dict[str, str]) -> tuple[int, dict[str, object]]:
        captured: dict[str, object] = {}

        async def fake_pipeline(**kwargs: object) -> server.TranscriptionResult:
            captured.update(kwargs)
            return server.TranscriptionResult(
                text="hello",
                language=None,
                segments=[],
                duration=None,
            )

        with (
            mock.patch.object(
                server,
                "_ensure_transcriber_ready_or_raise",
                return_value="qwen3-asr-mlx",
            ),
            mock.patch.object(
                server,
                "_run_transcription_pipeline",
                side_effect=fake_pipeline,
            ),
        ):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={"model": "qwen3-asr-mlx", **data},
                files={"file": ("audio.wav", b"fake audio", "audio/wav")},
            )

        return response.status_code, captured

    def test_multipart_word_timestamps_enables_alignment(self) -> None:
        status_code, captured = self._post_with_form({"word_timestamps": "true"})

        self.assertEqual(status_code, 200)
        self.assertEqual(captured["alignment_mode"], "on")

    def test_multipart_word_timestamps_can_disable_alignment(self) -> None:
        status_code, captured = self._post_with_form({"word_timestamps": "false"})

        self.assertEqual(status_code, 200)
        self.assertEqual(captured["alignment_mode"], "off")

    def test_multipart_alignment_overrides_word_timestamps(self) -> None:
        status_code, captured = self._post_with_form(
            {
                "alignment": "off",
                "word_timestamps": "true",
            }
        )

        self.assertEqual(status_code, 200)
        self.assertEqual(captured["alignment_mode"], "off")


if __name__ == "__main__":
    unittest.main()
