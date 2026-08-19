from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from fastapi import HTTPException

import whisper_mlx_server as server


def make_wav(path: Path, duration_seconds: float = 0.1) -> None:
    frame_count = int(16000 * duration_seconds)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * frame_count)


def make_result(duration: float = 0.1) -> server.TranscriptionResult:
    return server.TranscriptionResult(
        text="hello",
        language="en",
        segments=[{"id": 0, "start": 0.0, "end": duration, "text": "hello"}],
        duration=duration,
        words=[{"word": "hello", "start": 0.0, "end": duration}],
        segment_timestamp_source="forced_alignment",
        word_timestamp_source="forced_alignment",
        alignment_status="applied",
        processed_duration=duration,
    )


class CliParserTests(unittest.TestCase):
    def test_legacy_no_command_and_global_log_prompts_route_to_serve(self) -> None:
        self.assertEqual(server._normalized_cli_argv([]), ["serve"])
        self.assertEqual(
            server._normalized_cli_argv(["--log-prompts"]),
            ["serve", "--log-prompts"],
        )

    def test_transcribe_parser_exposes_timestamp_and_decode_controls(self) -> None:
        args = server._build_cli_parser().parse_args(
            [
                "transcribe",
                "audio.wav",
                "--response-format",
                "verbose_json",
                "--alignment",
                "on",
                "--word-timestamps",
                "--timestamp-granularity",
                "word",
                "--max-tokens",
                "12345",
            ]
        )

        self.assertEqual(args.command, "transcribe")
        self.assertEqual(args.alignment, "on")
        self.assertTrue(args.word_timestamps)
        self.assertEqual(args.timestamp_granularity, ["word"])
        self.assertEqual(args.max_tokens, 12345)

    def test_missing_local_file_returns_usage_error(self) -> None:
        args = server._build_cli_parser().parse_args(
            ["transcribe", "/definitely/missing/audio.wav"]
        )
        with mock.patch.object(server, "_ensure_transcriber_ready_or_raise") as ensure:
            self.assertEqual(server._run_local_transcribe_command(args), 2)
        ensure.assert_not_called()

    def test_directory_and_empty_file_return_usage_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / "empty.wav"
            empty_path.touch()
            for path in (Path(temp_dir), empty_path):
                with self.subTest(path=path):
                    args = server._build_cli_parser().parse_args(
                        ["transcribe", str(path)]
                    )
                    self.assertEqual(server._run_local_transcribe_command(args), 2)

    def test_output_must_not_overwrite_source_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            make_wav(audio_path)
            original_bytes = audio_path.read_bytes()
            args = server._build_cli_parser().parse_args(
                ["transcribe", str(audio_path), "--output", str(audio_path)]
            )

            self.assertEqual(server._run_local_transcribe_command(args), 2)
            self.assertEqual(audio_path.read_bytes(), original_bytes)


class CliOutputTests(unittest.TestCase):
    def test_all_supported_output_formats_serialize(self) -> None:
        result = make_result()
        self.assertEqual(
            json.loads(server._serialize_cli_output(result, "json")),
            {"text": "hello"},
        )
        self.assertEqual(server._serialize_cli_output(result, "text"), b"hello\n")
        self.assertTrue(server._serialize_cli_output(result, "srt").startswith(b"1\n"))
        self.assertTrue(server._serialize_cli_output(result, "vtt").startswith(b"WEBVTT\n"))

    def test_verbose_json_cli_and_http_share_the_same_payload(self) -> None:
        result = make_result()
        cli_payload = json.loads(
            server._serialize_cli_output(result, "verbose_json").decode("utf-8")
        )
        http_payload = json.loads(
            server._build_transcription_response(result, "verbose_json").body.decode("utf-8")
        )
        self.assertEqual(cli_payload, http_payload)

    def test_output_file_is_replaced_after_complete_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "result.json"
            target.parent.mkdir()
            target.write_text("old", encoding="utf-8")

            server._emit_cli_output(b'{"text":"new"}\n', str(target))

            self.assertEqual(target.read_text(encoding="utf-8"), '{"text":"new"}\n')
            self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_cli_maps_options_and_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            output_path = Path(temp_dir) / "result.json"
            make_wav(audio_path)
            args = server._build_cli_parser().parse_args(
                [
                    "transcribe",
                    str(audio_path),
                    "--response-format",
                    "verbose_json",
                    "--alignment",
                    "on",
                    "--word-timestamps",
                    "--timestamp-granularity",
                    "word",
                    "--output",
                    str(output_path),
                ]
            )

            captured: dict[str, object] = {}

            async def fake_path_pipeline(**kwargs: object) -> server.TranscriptionResult:
                captured.update(kwargs)
                return make_result()

            with (
                mock.patch.object(server, "_ensure_transcriber_ready_or_raise"),
                mock.patch.object(
                    server,
                    "_run_transcription_path_pipeline",
                    side_effect=fake_path_pipeline,
                ),
            ):
                self.assertEqual(server._run_local_transcribe_command(args), 0)

            self.assertEqual(captured["audio_path"], str(audio_path.resolve()))
            self.assertEqual(captured["alignment_mode"], "on")
            self.assertEqual(captured["timestamp_granularities"], ["word"])
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["timestamp_provenance"]["words"], "forced_alignment")

    def test_explicit_alignment_overrides_word_timestamp_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            make_wav(audio_path)
            args = server._build_cli_parser().parse_args(
                [
                    "transcribe",
                    str(audio_path),
                    "--response-format",
                    "verbose_json",
                    "--alignment",
                    "off",
                    "--word-timestamps",
                ]
            )
            captured: dict[str, object] = {}

            async def fake_path_pipeline(**kwargs: object) -> server.TranscriptionResult:
                captured.update(kwargs)
                return make_result()

            with (
                mock.patch.object(server, "_ensure_transcriber_ready_or_raise"),
                mock.patch.object(
                    server,
                    "_run_transcription_path_pipeline",
                    side_effect=fake_path_pipeline,
                ),
                mock.patch.object(server, "_emit_cli_output"),
            ):
                self.assertEqual(server._run_local_transcribe_command(args), 0)

            self.assertEqual(captured["alignment_mode"], "off")

    def test_plain_json_rejects_hidden_word_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            make_wav(audio_path)
            args = server._build_cli_parser().parse_args(
                ["transcribe", str(audio_path), "--word-timestamps"]
            )

            with (
                mock.patch.object(server, "_ensure_transcriber_ready_or_raise") as ensure,
                mock.patch.object(server, "_run_transcription_path_pipeline") as pipeline,
            ):
                self.assertEqual(server._run_local_transcribe_command(args), 1)

            ensure.assert_not_called()
            pipeline.assert_not_called()


class LongAudioBudgetTests(unittest.TestCase):
    def test_automatic_budget_scales_in_twenty_minute_windows(self) -> None:
        self.assertEqual(
            server._decode_options_with_auto_token_budget({}, 1200.0)["max_tokens"],
            8192,
        )
        self.assertEqual(
            server._decode_options_with_auto_token_budget({}, 1200.001)["max_tokens"],
            16384,
        )
        self.assertEqual(
            server._decode_options_with_auto_token_budget({}, 7692.5)["max_tokens"],
            57344,
        )

    def test_explicit_budget_is_never_replaced(self) -> None:
        self.assertEqual(
            server._decode_options_with_auto_token_budget(
                {"max_tokens": 1234},
                7692.5,
            )["max_tokens"],
            1234,
        )

    def test_normalization_preserves_empty_tail_chunk_duration(self) -> None:
        result = server._normalize_result(
            {
                "text": "hello",
                "language": "en",
                "segments": [
                    {"start": 0.0, "end": 1200.0, "text": "hello"},
                    {"start": 1200.0, "end": 2400.0, "text": ""},
                ],
            },
            "/missing/transcript.txt",
            None,
        )
        self.assertEqual(result.duration, 1200.0)
        self.assertEqual(result.processed_duration, 2400.0)

    def test_empty_raw_chunk_list_records_zero_processed_duration(self) -> None:
        result = server._normalize_result(
            {"text": "", "segments": [], "generation_tokens": 0},
            "/missing/transcript.txt",
            None,
        )
        self.assertEqual(result.processed_duration, 0.0)
        self.assertEqual(result.generation_tokens, 0)

    def test_incomplete_processed_timeline_fails_loudly(self) -> None:
        result = make_result(1200.0)
        result.processed_duration = 1200.0
        with self.assertRaisesRegex(RuntimeError, "did not cover the complete audio timeline"):
            server._validate_processed_audio_coverage(
                result,
                2400.0,
                stage="ASR transcription",
            )

    def test_complete_processed_timeline_reports_real_audio_duration(self) -> None:
        result = make_result(2400.1)
        server._validate_processed_audio_coverage(
            result,
            2400.0,
            stage="ASR transcription",
        )
        self.assertEqual(result.duration, 2400.0)

    def test_generation_budget_exhaustion_fails_even_with_full_timeline(self) -> None:
        result = make_result(1200.0)
        result.generation_tokens = 8192
        with self.assertRaisesRegex(RuntimeError, "exhausted max_tokens=8192"):
            server._validate_processed_audio_coverage(
                result,
                1200.0,
                stage="ASR transcription",
                max_tokens=8192,
            )


class PathPipelineTests(unittest.TestCase):
    def test_http_limit_does_not_call_path_pipeline(self) -> None:
        with (
            mock.patch.object(server, "MAX_UPLOAD_BYTES", 4),
            mock.patch.object(server, "_run_transcription_path_pipeline") as path_pipeline,
        ):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    server._run_transcription_pipeline(
                        request_id="http-limit-test",
                        audio_bytes=b"12345",
                        file_name="audio.wav",
                        language=None,
                        prompt=None,
                        alignment_mode="off",
                        response_format="json",
                        timestamp_granularities=None,
                    )
                )
        self.assertEqual(raised.exception.status_code, 413)
        path_pipeline.assert_not_called()

    def test_path_pipeline_ignores_http_limit_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            make_wav(audio_path)
            result = make_result()

            with (
                mock.patch.object(server, "MAX_UPLOAD_BYTES", 1),
                mock.patch.object(server.transcriber, "transcribe", return_value=result),
                mock.patch.object(server, "_effective_auto_align_lang_codes", return_value=set()),
            ):
                output = asyncio.run(
                    server._run_transcription_path_pipeline(
                        request_id="path-limit-test",
                        audio_path=str(audio_path),
                        file_name=audio_path.name,
                        language="en",
                        prompt=None,
                        alignment_mode="off",
                        response_format="json",
                        timestamp_granularities=None,
                    )
                )

            self.assertEqual(output.text, "hello")
            self.assertTrue(audio_path.exists())
            self.assertGreater(audio_path.stat().st_size, 1)

    def test_converted_temp_file_is_cleaned_without_deleting_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "audio.bin"
            converted_path = Path(temp_dir) / "converted.wav"
            source_path.write_bytes(b"not a wav")
            make_wav(converted_path)
            result = make_result()

            with (
                mock.patch.object(server, "_is_standard_pcm16_mono_16k_wav_path", return_value=False),
                mock.patch.object(
                    server,
                    "_transcode_with_ffmpeg",
                    return_value=(str(converted_path), None),
                ),
                mock.patch.object(server.transcriber, "transcribe", return_value=result),
                mock.patch.object(server, "_effective_auto_align_lang_codes", return_value=set()),
            ):
                asyncio.run(
                    server._run_transcription_path_pipeline(
                        request_id="path-cleanup-test",
                        audio_path=str(source_path),
                        file_name=source_path.name,
                        language="en",
                        prompt=None,
                        alignment_mode="off",
                        response_format="json",
                        timestamp_granularities=None,
                    )
                )

            self.assertTrue(source_path.exists())
            self.assertFalse(converted_path.exists())

    def test_ffmpeg_failure_is_not_silently_bypassed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "audio.bin"
            source_path.write_bytes(b"not audio")

            with (
                mock.patch.object(server, "_is_standard_pcm16_mono_16k_wav_path", return_value=False),
                mock.patch.object(
                    server,
                    "_transcode_with_ffmpeg",
                    return_value=(None, "decode failed"),
                ),
                mock.patch.object(server.transcriber, "transcribe") as transcribe,
            ):
                with self.assertRaises(HTTPException) as raised:
                    asyncio.run(
                        server._run_transcription_path_pipeline(
                            request_id="ffmpeg-failure-test",
                            audio_path=str(source_path),
                            file_name=source_path.name,
                            language=None,
                            prompt=None,
                            alignment_mode="off",
                            response_format="json",
                            timestamp_granularities=None,
                        )
                    )

            self.assertEqual(raised.exception.status_code, 500)
            self.assertIn("ffmpeg transcode failed", str(raised.exception.detail))
            transcribe.assert_not_called()
            self.assertTrue(source_path.exists())


if __name__ == "__main__":
    unittest.main()
