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

    def test_multipart_response_format_is_forwarded_to_pipeline(self) -> None:
        status_code, captured = self._post_with_form({"response_format": "verbose_json"})

        self.assertEqual(status_code, 200)
        self.assertEqual(captured["response_format"], "verbose_json")


class AlignmentDecisionTests(unittest.TestCase):
    def test_auto_alignment_is_disabled_for_plain_json(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            word_timestamps_requested=False,
            response_format="json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertFalse(enabled)

    def test_auto_alignment_is_disabled_for_plain_text(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            word_timestamps_requested=False,
            response_format="text",
            detected_language_code="en",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertFalse(enabled)

    def test_auto_alignment_is_enabled_for_verbose_json_supported_language(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            word_timestamps_requested=False,
            response_format="verbose_json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertTrue(enabled)

    def test_explicit_word_timestamps_override_plain_json(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            word_timestamps_requested=True,
            response_format="json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertTrue(enabled)

    def test_explicit_alignment_off_overrides_word_timestamps(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="off",
            word_timestamps_requested=True,
            response_format="verbose_json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertFalse(enabled)


if __name__ == "__main__":
    unittest.main()
