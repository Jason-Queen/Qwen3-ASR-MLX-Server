from __future__ import annotations

import base64
import os
import tempfile
import unittest
import wave
from unittest import mock

from fastapi.testclient import TestClient

import whisper_mlx_server as server


class MultipartTranscriptionEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def _post_with_form(self, data: dict[str, str]) -> tuple[object, dict[str, object]]:
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

        return response, captured

    def test_multipart_word_timestamps_enables_alignment(self) -> None:
        response, captured = self._post_with_form(
            {
                "response_format": "verbose_json",
                "word_timestamps": "true",
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["alignment_mode"], "on")

    def test_multipart_word_timestamps_can_disable_alignment(self) -> None:
        response, captured = self._post_with_form({"word_timestamps": "false"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["alignment_mode"], "off")

    def test_multipart_alignment_overrides_word_timestamps(self) -> None:
        response, captured = self._post_with_form(
            {
                "alignment": "off",
                "word_timestamps": "true",
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["alignment_mode"], "off")

    def test_multipart_response_format_is_forwarded_to_pipeline(self) -> None:
        response, captured = self._post_with_form({"response_format": "verbose_json"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["response_format"], "verbose_json")

    def test_plain_json_rejects_word_timestamps_instead_of_hiding_them(self) -> None:
        response, captured = self._post_with_form({"word_timestamps": "true"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("response_format=verbose_json", response.json()["detail"])
        self.assertEqual(captured, {})

    def test_plain_json_rejects_timestamp_granularities(self) -> None:
        response, captured = self._post_with_form(
            {"timestamp_granularities[]": "word"}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot expose timestamps", response.json()["detail"])
        self.assertEqual(captured, {})


class JsonTimestampEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)

    def _post_transcription(
        self,
        values: dict[str, object],
    ) -> tuple[object, dict[str, object]]:
        captured: dict[str, object] = {}

        async def fake_pipeline(**kwargs: object) -> server.TranscriptionResult:
            captured.update(kwargs)
            return server.TranscriptionResult(
                text="hello",
                language="en",
                segments=[{"id": 0, "start": 0.0, "end": 0.5, "text": "hello"}],
                duration=0.5,
                words=[{"word": "hello", "start": 0.0, "end": 0.5}],
                segment_timestamp_source="forced_alignment",
                word_timestamp_source="forced_alignment",
                alignment_status="applied",
            )

        payload = {
            "model": "qwen3-asr-mlx",
            "audio_file": base64.b64encode(b"fake audio").decode("ascii"),
            **values,
        }
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
            response = self.client.post("/v1/audio/transcriptions", json=payload)
        return response, captured

    def test_json_transport_rejects_hidden_timestamps_in_plain_json_response(self) -> None:
        with mock.patch.object(server, "_parse_audio_file_payload") as audio_parser:
            response, captured = self._post_transcription({"word_timestamps": True})

        self.assertEqual(response.status_code, 400)
        self.assertIn("response_format=verbose_json", response.json()["detail"])
        self.assertEqual(captured, {})
        audio_parser.assert_not_called()

    def test_json_transport_verbose_json_exposes_timestamp_provenance(self) -> None:
        response, captured = self._post_transcription(
            {
                "response_format": "verbose_json",
                "word_timestamps": True,
                "timestamp_granularities": ["word"],
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["alignment_mode"], "on")
        self.assertEqual(captured["response_format"], "verbose_json")
        self.assertEqual(
            response.json()["timestamp_provenance"],
            {
                "segments": "forced_alignment",
                "words": "forced_alignment",
                "alignment": "applied",
            },
        )

    def test_chat_timestamp_controls_fail_before_audio_inference(self) -> None:
        with (
            mock.patch.object(
                server,
                "_ensure_transcriber_ready_or_raise",
                return_value="qwen3-asr-mlx",
            ),
            mock.patch.object(server, "_run_transcription_pipeline") as pipeline,
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3-asr-mlx",
                    "audio_file": base64.b64encode(b"fake audio").decode("ascii"),
                    "word_timestamps": True,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "Chat Completions returns transcript text only",
            response.json()["detail"],
        )
        pipeline.assert_not_called()


class AlignmentDecisionTests(unittest.TestCase):
    def test_auto_alignment_is_disabled_for_plain_json(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            timestamps_requested=False,
            response_format="json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertFalse(enabled)

    def test_auto_alignment_is_disabled_for_plain_text(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            timestamps_requested=False,
            response_format="text",
            detected_language_code="en",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertFalse(enabled)

    def test_auto_alignment_is_enabled_for_verbose_json_supported_language(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            timestamps_requested=False,
            response_format="verbose_json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertTrue(enabled)

    def test_explicit_word_timestamps_override_plain_json(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            timestamps_requested=True,
            response_format="json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertTrue(enabled)

    def test_segment_timestamp_granularity_override_plain_json(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="auto",
            timestamps_requested=server._timestamp_requests_alignment(["segment"]),
            response_format="json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertTrue(enabled)

    def test_explicit_alignment_off_overrides_word_timestamps(self) -> None:
        enabled = server._compute_alignment_enabled(
            alignment_mode="off",
            timestamps_requested=True,
            response_format="verbose_json",
            detected_language_code="zh",
            auto_align_lang_codes={"zh", "en"},
        )

        self.assertFalse(enabled)


class TimestampResponseContractTests(unittest.TestCase):
    def test_verbose_json_accepts_word_timestamp_request(self) -> None:
        server._validate_timestamp_response_contract(
            response_format="verbose_json",
            alignment_mode="on",
            timestamp_granularities=["word"],
        )

    def test_plain_json_rejects_explicit_alignment(self) -> None:
        with self.assertRaisesRegex(
            server.HTTPException,
            "response_format=verbose_json",
        ):
            server._validate_timestamp_response_contract(
                response_format="json",
                alignment_mode="on",
                timestamp_granularities=[],
            )

    def test_chat_rejects_timestamp_output_controls(self) -> None:
        with self.assertRaisesRegex(
            server.HTTPException,
            "Chat Completions returns transcript text only",
        ):
            server._validate_timestamp_response_contract(
                response_format="json",
                alignment_mode="auto",
                timestamp_granularities=["segment"],
                chat_response=True,
            )

    def test_plural_timestamp_granularities_normalize_to_contract_values(self) -> None:
        self.assertEqual(
            server._normalize_timestamp_granularities(["words", "segments", "word"]),
            ["word", "segment"],
        )


class AlignmentChunkBoundaryTests(unittest.TestCase):
    def test_chunk_boundaries_snap_to_near_segment_end(self) -> None:
        boundaries = server._alignment_chunk_boundaries_from_segments(
            segments=[
                {"id": 0, "start": 0.0, "end": 28.6, "text": "first sentence"},
                {"id": 1, "start": 28.6, "end": 61.2, "text": "second sentence"},
            ],
            duration_sec=70.0,
            max_chunk_seconds=30.0,
            boundary_tolerance_seconds=5.0,
        )

        self.assertEqual(boundaries, [0.0, 28.6, 61.2, 70.0])

    def test_chunk_boundaries_fall_back_to_fixed_target_without_near_segment_end(self) -> None:
        boundaries = server._alignment_chunk_boundaries_from_segments(
            segments=[
                {"id": 0, "start": 0.0, "end": 12.0, "text": "too early"},
                {"id": 1, "start": 12.0, "end": 48.0, "text": "too late"},
            ],
            duration_sec=70.0,
            max_chunk_seconds=30.0,
            boundary_tolerance_seconds=5.0,
        )

        self.assertEqual(boundaries, [0.0, 30.0, 60.0, 70.0])

    def test_chunk_boundaries_keep_fixed_target_when_tolerance_is_zero(self) -> None:
        boundaries = server._alignment_chunk_boundaries_from_segments(
            segments=[{"id": 0, "start": 0.0, "end": 28.6, "text": "near target"}],
            duration_sec=65.0,
            max_chunk_seconds=30.0,
            boundary_tolerance_seconds=0.0,
        )

        self.assertEqual(boundaries, [0.0, 30.0, 60.0, 65.0])

    def test_split_wav_uses_segment_snapped_boundaries(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name

        chunk_paths: list[str] = []
        try:
            with wave.open(wav_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(10)
                wav_file.writeframes(b"\x00\x00" * 70)

            chunks = server._split_wav_into_temp_chunks(
                wav_path,
                max_chunk_seconds=3.0,
                split_boundaries=[0.0, 2.5, 5.0, 7.0],
            )
            chunk_paths = [path for path, _start, _end in chunks]
            ranges = [(round(start, 1), round(end, 1)) for _path, start, end in chunks]

            self.assertEqual(ranges, [(0.0, 2.5), (2.5, 5.0), (5.0, 7.0)])
        finally:
            for path in chunk_paths:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            try:
                os.unlink(wav_path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
