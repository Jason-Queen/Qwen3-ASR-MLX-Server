from __future__ import annotations

import asyncio
import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import ForceAlignProcessor

import whisper_mlx_server as server


class UpstreamTokenizerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.processor = ForceAlignProcessor()

    def tokenizer(self, language: str):
        return lambda text: self.processor.encode_timestamp(text, language)[0]

    def timed_words(
        self,
        tokens: list[str],
        *,
        offset: float = 0.0,
        step: float = 1.0,
        width: float = 0.6,
    ) -> list[dict[str, object]]:
        return [
            {
                "word": token,
                "start": offset + index * step,
                "end": offset + index * step + width,
            }
            for index, token in enumerate(tokens)
        ]

    def assert_exact_partition(
        self,
        segments: list[dict[str, object]],
        expected_tokens: list[str],
    ) -> None:
        actual_tokens = [
            str(word["word"])
            for segment in segments
            for word in segment.get("_words", [])
        ]
        self.assertEqual(actual_tokens, expected_tokens)


class ExactSentenceMappingTests(UpstreamTokenizerTestCase):
    def test_hyphenated_word_uses_upstream_token_counts(self) -> None:
        text = "state-of-the-art systems. This is next."
        tokenize = self.tokenizer("English")
        tokens = tokenize(text)
        words = self.timed_words(tokens)

        segments, error = server._build_sentence_segments_from_words(
            words,
            text,
            "en",
            tokenize,
        )

        self.assertIsNone(error)
        self.assertEqual([segment["text"] for segment in segments], [
            "state-of-the-art systems.",
            "This is next.",
        ])
        self.assertEqual(
            [[word["word"] for word in segment["_words"]] for segment in segments],
            [["stateoftheart", "systems"], ["This", "is", "next"]],
        )
        self.assert_exact_partition(segments, tokens)

    def test_regression_sentence_ends_at_anything_not_previous_word(self) -> None:
        text = "You never know anything. This is next."
        tokenize = self.tokenizer("English")
        tokens = tokenize(text)
        words = self.timed_words(tokens, step=2.4, width=0.64)

        segments, error = server._build_sentence_segments_from_words(
            words,
            text,
            "en",
            tokenize,
        )

        self.assertIsNone(error)
        self.assertEqual([word["word"] for word in segments[0]["_words"]], [
            "You",
            "never",
            "know",
            "anything",
        ])
        self.assertEqual(segments[0]["end"], words[3]["end"])
        self.assert_exact_partition(segments, tokens)

    def test_acronym_decimal_and_unicode_do_not_create_false_boundaries(self) -> None:
        text = "The U.S. result was 3.5% with café déjà-vu. This works."
        tokenize = self.tokenizer("English")
        tokens = tokenize(text)
        words = self.timed_words(tokens)

        segments, error = server._build_sentence_segments_from_words(
            words,
            text,
            "en",
            tokenize,
        )

        self.assertIsNone(error)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["text"], "The U.S. result was 3.5% with café déjà-vu.")
        self.assert_exact_partition(segments, tokens)

    def test_chinese_decimal_is_not_treated_as_sentence_end(self) -> None:
        text = "AI模型达到3.5%。下一句。"
        tokenize = self.tokenizer("Chinese")
        tokens = tokenize(text)
        words = self.timed_words(tokens)

        segments, error = server._build_sentence_segments_from_words(
            words,
            text,
            "zh",
            tokenize,
        )

        self.assertIsNone(error)
        self.assertEqual(
            [segment["text"] for segment in segments],
            ["AI模型达到3.5%。", "下一句。"],
        )
        self.assert_exact_partition(segments, tokens)

    def test_japanese_and_korean_use_upstream_language_tokenizers(self) -> None:
        cases = [
            ("Japanese", "ja", "人工知能の未来です。次の文です。"),
            ("Korean", "ko", "인공지능의 미래입니다. 다음 문장입니다."),
        ]
        for language, code, text in cases:
            with self.subTest(language=language):
                tokenize = self.tokenizer(language)
                try:
                    tokens = tokenize(text)
                except ImportError as exc:
                    self.skipTest(str(exc))
                words = self.timed_words(tokens)
                segments, error = server._build_sentence_segments_from_words(
                    words,
                    text,
                    code,
                    tokenize,
                )
                self.assertIsNone(error)
                self.assertEqual(len(segments), 2)
                self.assert_exact_partition(segments, tokens)

    def test_token_identity_mismatch_fails_closed(self) -> None:
        text = "You never know anything. This is next."
        tokenize = self.tokenizer("English")
        tokens = tokenize(text)
        words = self.timed_words(tokens)
        del words[3]

        segments, error = server._build_sentence_segments_from_words(
            words,
            text,
            "en",
            tokenize,
        )

        self.assertEqual(segments, [])
        self.assertIsNotNone(error)
        self.assertIn("expected_tokens=7 actual_tokens=6", str(error))


class ExactHardLimitTests(UpstreamTokenizerTestCase):
    def test_hard_limit_splits_only_at_verified_word_boundaries(self) -> None:
        text = "alpha beta gamma delta epsilon zeta"
        tokenize = self.tokenizer("English")
        tokens = tokenize(text)
        words = self.timed_words(tokens, step=1.2, width=0.8)
        source = {
            "id": 0,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": text,
            "_words": words,
        }

        segments = server._enforce_hard_duration_limit(
            [source],
            "en",
            tokenize,
            hard_max_seconds=2.5,
        )

        self.assertEqual([segment["text"] for segment in segments], [
            "alpha beta",
            "gamma delta",
            "epsilon zeta",
        ])
        self.assertTrue(
            all(float(segment["end"]) - float(segment["start"]) <= 2.5 for segment in segments)
        )
        self.assert_exact_partition(segments, tokens)

    def test_unsplittable_long_token_is_preserved_without_estimated_time(self) -> None:
        text = "state-of-the-art"
        tokenize = self.tokenizer("English")
        words = [{"word": tokenize(text)[0], "start": 0.0, "end": 20.0}]
        source = {"id": 0, "start": 0.0, "end": 20.0, "text": text, "_words": words}

        segments = server._enforce_hard_duration_limit(
            [source],
            "en",
            tokenize,
            hard_max_seconds=6.0,
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 20.0)
        self.assertEqual(segments[0]["text"], text)


class FakeAligner:
    def __init__(self, processor: ForceAlignProcessor) -> None:
        self.processor = processor
        self.calls: list[tuple[str, str]] = []

    def canonical_tokens(self, text: str, language: str) -> list[str]:
        return self.processor.encode_timestamp(text, language)[0]

    def align(self, audio: str, text: str, language: str) -> list[dict[str, object]]:
        self.calls.append((audio, text))
        tokens = self.canonical_tokens(text, language)
        return [
            {"word": token, "start": index * 0.5, "end": index * 0.5 + 0.4}
            for index, token in enumerate(tokens)
        ]


class AlignmentChunkContractTests(UpstreamTokenizerTestCase):
    def make_wav(self, duration_seconds: float = 4.0) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(10)
            wav_file.writeframes(b"\x00\x00" * int(duration_seconds * 10))
        return wav_path

    def test_long_alignment_preserves_chunk_text_and_word_order(self) -> None:
        wav_path = self.make_wav()
        aligner = FakeAligner(self.processor)
        result = server.TranscriptionResult(
            text="alpha beta. gamma delta.",
            language="en",
            segments=[
                {"id": 0, "start": 0.0, "end": 2.0, "text": "alpha beta."},
                {"id": 1, "start": 2.0, "end": 4.0, "text": "gamma delta."},
            ],
            duration=4.0,
        )
        try:
            output = server._apply_alignment(
                aligner,
                wav_path,
                result,
                "English",
                4.0,
                2.0,
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        self.assertEqual(len(output.chunks), 2)
        self.assertEqual([chunk.text for chunk in output.chunks], [
            "alpha beta.",
            "gamma delta.",
        ])
        self.assertEqual(
            [word["word"] for word in output.words],
            ["alpha", "beta", "gamma", "delta"],
        )
        self.assertEqual(output.words[2]["start"], 2.0)

    def test_long_alignment_rejects_incomplete_chunk_transcript_coverage(self) -> None:
        wav_path = self.make_wav()
        aligner = FakeAligner(self.processor)
        result = server.TranscriptionResult(
            text="alpha beta. gamma delta.",
            language="en",
            segments=[
                {"id": 0, "start": 0.0, "end": 2.0, "text": "alpha beta."},
            ],
            duration=4.0,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "chunk transcript coverage mismatch"):
                server._apply_alignment(
                    aligner,
                    wav_path,
                    result,
                    "English",
                    4.0,
                    2.0,
                )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        self.assertEqual(aligner.calls, [])

    def test_long_alignment_rejects_segment_crossing_audio_chunk_boundary(self) -> None:
        wav_path = self.make_wav()
        aligner = FakeAligner(self.processor)
        result = server.TranscriptionResult(
            text="alpha beta gamma delta",
            language="en",
            segments=[
                {"id": 0, "start": 0.0, "end": 4.0, "text": "alpha beta gamma delta"},
            ],
            duration=4.0,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "crosses alignment chunk boundary"):
                server._apply_alignment(
                    aligner,
                    wav_path,
                    result,
                    "English",
                    4.0,
                    2.0,
                )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        self.assertEqual(aligner.calls, [])

    def test_segment_rebuild_consumes_each_alignment_chunk_exactly(self) -> None:
        aligner = FakeAligner(self.processor)
        first_text = "alpha beta."
        second_text = "gamma delta."
        first_tokens = aligner.canonical_tokens(first_text, "English")
        second_tokens = aligner.canonical_tokens(second_text, "English")
        first_words = self.timed_words(first_tokens, offset=0.0)
        second_words = self.timed_words(second_tokens, offset=3.0)
        alignment = server.AlignmentOutput(
            words=[*first_words, *second_words],
            chunks=[
                server.AlignmentChunk(first_text, first_tokens, first_words, 0.0, 2.0),
                server.AlignmentChunk(second_text, second_tokens, second_words, 3.0, 5.0),
            ],
        )

        with (
            mock.patch.object(server, "SEGMENT_MERGE_TARGET_SECONDS", 0.0),
            mock.patch.object(server, "SEGMENT_HARD_MAX_SECONDS", 0.0),
        ):
            segments, error = server._rebuild_segments_from_alignment(
                alignment=alignment,
                aligner=aligner,
                aligner_language="English",
                lang_code="en",
            )

        self.assertIsNone(error)
        self.assertEqual([segment["text"] for segment in segments], [first_text, second_text])
        self.assertEqual([segment["start"] for segment in segments], [0.0, 3.0])
        self.assertEqual([segment["end"] for segment in segments], [1.6, 4.6])


class AlignmentFallbackProvenanceTests(UpstreamTokenizerTestCase):
    def make_wav_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 1600)
        return buffer.getvalue()

    def test_segment_mapping_failure_keeps_asr_segments_and_aligned_words(self) -> None:
        source = server.TranscriptionResult(
            text="alpha beta.",
            language="en",
            segments=[{"id": 0, "start": 0.0, "end": 0.1, "text": "alpha beta."}],
            duration=0.1,
            segment_timestamp_source="asr",
        )
        words = self.timed_words(["alpha", "beta"], step=0.04, width=0.03)
        alignment = server.AlignmentOutput(
            words=words,
            chunks=[
                server.AlignmentChunk(
                    text=source.text,
                    canonical_tokens=["alpha", "beta"],
                    words=words,
                    start=0.0,
                    end=0.1,
                )
            ],
        )

        with (
            mock.patch.object(server.transcriber, "transcribe", return_value=source),
            mock.patch.object(server.aligner, "is_ready", return_value=True),
            mock.patch.object(server, "_effective_auto_align_lang_codes", return_value={"en"}),
            mock.patch.object(server, "_should_rechunk_segments_for_alignment", return_value=False),
            mock.patch.object(server, "_alignment_dependency_missing", return_value=None),
            mock.patch.object(server, "_apply_alignment", return_value=alignment),
            mock.patch.object(
                server,
                "_rebuild_segments_from_alignment",
                return_value=([], "chunk=0 expected_tokens=2 actual_tokens=1"),
            ),
        ):
            result = asyncio.run(
                server._run_transcription_pipeline(
                    request_id="mapping-fallback-test",
                    audio_bytes=self.make_wav_bytes(),
                    file_name="audio.wav",
                    language="en",
                    prompt=None,
                    alignment_mode="on",
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
            )

        self.assertEqual(result.segments, source.segments)
        self.assertEqual(result.segment_timestamp_source, "asr")
        self.assertEqual(result.words, words)
        self.assertEqual(result.word_timestamp_source, "forced_alignment")
        self.assertEqual(result.alignment_status, "applied")


if __name__ == "__main__":
    unittest.main()
