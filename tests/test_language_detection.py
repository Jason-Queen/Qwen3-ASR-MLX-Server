from __future__ import annotations

import unittest

from whisper_mlx_server import TranscriptionResult
from whisper_mlx_server import _finalize_result_language
from whisper_mlx_server import _normalize_result


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


if __name__ == "__main__":
    unittest.main()
