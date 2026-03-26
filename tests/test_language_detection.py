from __future__ import annotations

import json
import unittest

from whisper_mlx_server import TranscriptionResult
from whisper_mlx_server import _build_transcription_response
from whisper_mlx_server import _finalize_result_language
from whisper_mlx_server import _normalize_result
from whisper_mlx_server import _restore_segment_languages_from_overlap


class LanguageDetectionTests(unittest.TestCase):
    def test_normalize_result_accepts_top_level_language_list(self) -> None:
        result = _normalize_result(
            {
                "text": "Make America great again.",
                "language": ["English"],
            },
            output_path="/tmp/unused.txt",
            requested_language=None,
        )

        self.assertEqual(result.language, "en")

    def test_normalize_result_prefers_primary_segment_language(self) -> None:
        result = _normalize_result(
            {
                "text": "Hello again 你好",
                "segments": [
                    {"text": "Hello", "start": 0.0, "end": 2.0, "language": "English"},
                    {"text": "again", "start": 2.0, "end": 4.0, "language": "English"},
                    {"text": "你好", "start": 4.0, "end": 4.5, "language": "Chinese"},
                ],
            },
            output_path="/tmp/unused.txt",
            requested_language=None,
        )

        self.assertEqual(result.language, "en")
        self.assertEqual(result.segments[0]["language"], "en")
        self.assertEqual(result.segments[2]["language"], "zh")

    def test_normalize_result_uses_requested_language_code(self) -> None:
        result = _normalize_result(
            {"text": "让美国再次伟大。"},
            output_path="/tmp/unused.txt",
            requested_language="Chinese",
        )

        self.assertEqual(result.language, "zh")

    def test_finalize_result_language_uses_text_fallback(self) -> None:
        result = TranscriptionResult(
            text="让美国再次伟大。",
            language=None,
            segments=[{"id": 0, "start": 0.0, "end": 1.0, "text": "让美国再次伟大。"}],
            duration=1.0,
        )

        detected = _finalize_result_language(None, result)

        self.assertEqual(detected, "zh")
        self.assertEqual(result.language, "zh")

    def test_verbose_json_exposes_segment_languages_and_detected_languages(self) -> None:
        result = TranscriptionResult(
            text="你好 world 早晨",
            language="zh",
            segments=[
                {"id": 0, "start": 0.0, "end": 1.5, "text": "你好", "language": "zh"},
                {"id": 1, "start": 1.5, "end": 2.0, "text": "world", "language": "en"},
                {"id": 2, "start": 2.0, "end": 4.5, "text": "早晨", "language": "yue"},
                {"id": 3, "start": 4.5, "end": 6.0, "text": "你好", "language": "zh"},
            ],
            duration=6.0,
        )

        response = _build_transcription_response(result, "verbose_json")
        payload = json.loads(response.body)

        self.assertEqual(payload["language"], "zh")
        self.assertEqual(payload["segments"][0]["language"], "zh")
        self.assertEqual(payload["segments"][1]["language"], "en")
        self.assertEqual(payload["segments"][2]["language"], "yue")
        self.assertEqual(
            payload["detected_languages"],
            [
                {"code": "zh", "duration": 3.0},
                {"code": "yue", "duration": 2.5},
                {"code": "en", "duration": 0.5},
            ],
        )

    def test_verbose_json_detected_languages_falls_back_to_top_level_language(self) -> None:
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            segments=[{"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world"}],
            duration=2.0,
        )

        response = _build_transcription_response(result, "verbose_json")
        payload = json.loads(response.body)

        self.assertNotIn("language", payload["segments"][0])
        self.assertEqual(payload["detected_languages"], [{"code": "en", "duration": 2.0}])

    def test_restore_segment_languages_from_overlap_uses_dominant_source_language(self) -> None:
        source_segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello", "language": "en"},
            {"id": 1, "start": 2.0, "end": 5.0, "text": "你好世界", "language": "zh"},
            {"id": 2, "start": 5.0, "end": 8.0, "text": "早晨", "language": "yue"},
        ]
        rebuilt_segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "Hello"},
            {"id": 1, "start": 1.0, "end": 3.0, "text": "Hello 你好"},
            {"id": 2, "start": 3.0, "end": 6.0, "text": "世界 早晨"},
            {"id": 3, "start": 6.0, "end": 8.0, "text": "早晨"},
        ]

        restored = _restore_segment_languages_from_overlap(source_segments, rebuilt_segments)

        self.assertEqual([segment.get("language") for segment in restored], ["en", "en", "zh", "yue"])

    def test_restore_segment_languages_from_overlap_keeps_existing_language(self) -> None:
        source_segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello", "language": "en"},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "你好", "language": "zh"},
        ]
        rebuilt_segments = [
            {"id": 0, "start": 0.0, "end": 4.0, "text": "Hello 你好", "language": "yue"},
        ]

        restored = _restore_segment_languages_from_overlap(source_segments, rebuilt_segments)

        self.assertEqual(restored[0]["language"], "yue")


if __name__ == "__main__":
    unittest.main()
