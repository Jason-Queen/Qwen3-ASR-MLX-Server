from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from fastapi import HTTPException
from fastapi.testclient import TestClient
from mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner import ForceAlignProcessor

import forced_alignment as forced
import whisper_mlx_server as server


def make_wav(path: Path, seconds: float = 3) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x00\x00" * round(seconds * 16000))


class FakeModel:
    def __init__(self) -> None:
        self.aligner_processor = ForceAlignProcessor()
        self.config = SimpleNamespace(timestamp_segment_time=80)
        self._tokenizer = SimpleNamespace(encode=lambda text, **kw: [0] * (text.count("<timestamp>") + len(text) // 10))
        self.paths: list[Path] = []
        self.calls: list[dict] = []
        self.timestamps = None
        self.on_generate = None

    def get_supported_languages(self):
        return ["English", "Chinese", "Japanese", "Korean", "Cantonese", "German"]

    def generate(self, *, audio, text, language):
        self.paths.append(Path(audio))
        with wave.open(audio) as wav:
            self.calls.append({"text": text, "frames": wav.getnframes()})
        if self.on_generate:
            self.on_generate(Path(audio))
        tokens, _ = self.aligner_processor.encode_timestamp(text, language)
        timestamps = np.arange(len(tokens) * 2) * 80 if self.timestamps is None else np.array(self.timestamps)
        items = self.aligner_processor.parse_timestamp(tokens, timestamps)
        return SimpleNamespace(segments=[{"text": w["text"], "start": w["start_time"] / 1000, "end": w["end_time"] / 1000} for w in items])


class ContractTests(unittest.TestCase):
    def request(self, text="hello", regions=None):
        return forced.parse_request(text, "en", "verbose_json", regions)

    def test_text_and_language_input_errors(self):
        for text in (None, "", "  \n", 3, "a\x00b", "a\ud800", "x" * 100001):
            with self.subTest(text_type=type(text)), self.assertRaises(forced.AlignmentInputError):
                self.request(text)
        for lang in (None, "", "auto"):
            with self.assertRaises(forced.AlignmentInputError):
                forced.parse_request("hello", lang, "verbose_json")
        for fmt in ("json", "text", "xml"):
            with self.assertRaises(forced.AlignmentInputError):
                forced.parse_request("hello", "en", fmt)

    def test_subtitle_text_validation_preserves_single_crlf(self):
        self.assertEqual(forced.parse_request("hello\r\n", "en", "vtt").text, "hello\r\n")
        for text in ("hello\n\nworld", "hello\r\n \r\nworld", "a --> b"):
            with self.assertRaises(forced.AlignmentInputError):
                forced.parse_request(text, "en", "srt")

    def test_region_schema_ranges_and_full_coverage(self):
        valid = {"start": 0, "end": 1, "text_start": 0, "text_end": 5}
        invalid = ['[{"start": 0, "start": 1, "end": 2, "text_start": 0, "text_end": 5}]', "no json", "null", "{}", [], [dict(valid, extra=1)], [dict(valid, start=-1)],
                   [dict(valid, start=float("nan"))], [dict(valid, end=float("inf"))],
                   [dict(valid, end=0)], [dict(valid, end=31)], [dict(valid, text_start=1)],
                   [dict(valid, text_end=4)], [dict(valid, text_end=True)], [dict(valid, start=True)],
                   [dict(valid, text_end=2.0)], [valid, valid]]
        for regions in invalid:
            with self.subTest(regions=regions), self.assertRaises(forced.AlignmentInputError):
                self.request(regions=regions)
        self.assertEqual(self.request(regions=json.dumps([valid])).regions[0].text_end, 5)

    def test_duration_and_sample_boundaries(self):
        self.assertEqual(len(forced.plan_windows(self.request(), 480000)), 1)
        for frames in (0, 480001, forced.MAX_SOURCE_SECONDS * 16000 + 1):
            with self.assertRaises(forced.AlignmentInputError):
                forced.plan_windows(self.request(), frames)
        region = [{"start": 1.00003, "end": 2.00002, "text_start": 0, "text_end": 5}]
        window = forced.plan_windows(self.request(regions=region), 32000)[0]
        self.assertEqual((window.start_frame, window.end_frame), (16000, 32000))
        region[0]["end"] = 2.001
        with self.assertRaises(forced.AlignmentInputError):
            forced.plan_windows(self.request(regions=region), 32000)

    def test_word_validation_rejects_invalid_numbers_and_unrepaired_order(self):
        invalid = [(-1, 1), (float("nan"), 1), (0, float("inf")), (2, 1), (0, 3.1),
                   ("0", 1), (True, 1)]
        for start, end in invalid:
            with self.subTest(start=start, end=end), self.assertRaises(forced.AlignmentOutputError):
                forced.validate_words([{"word": "hello", "start": start, "end": end}], ["hello"], 3)
        with self.assertRaises(forced.AlignmentOutputError):
            forced.validate_words([{"word": "a", "start": 0, "end": 1}, {"word": "b", "start": .8, "end": 2}], ["a", "b"], 3)
        for raw in ([], [{"word": "different", "start": 0, "end": 1}], [None]):
            with self.assertRaises(forced.AlignmentOutputError):
                forced.validate_words(raw, ["hello"], 3)

    def test_one_bin_overlap_is_preserved_but_backward_end_is_rejected(self):
        words, _ = forced.validate_words([
            {"word": "a", "start": .8, "end": 1.0},
            {"word": "b", "start": .92, "end": 1.2}], ["a", "b"], 3)
        self.assertEqual(words[1]["start"], .92)
        with self.assertRaises(forced.AlignmentOutputError):
            forced.validate_words([{"word": "a", "start": .8, "end": 1},
                                   {"word": "b", "start": .92, "end": .96}], ["a", "b"], 3)

    def test_zero_duration_and_resolution_overshoot_are_explicit(self):
        words, adjustments = forced.validate_words([{"word": "a", "start": 3.04, "end": 3.04}], ["a"], 3)
        self.assertEqual(words[0]["timing_status"], "zero_duration")
        self.assertEqual(len(adjustments), 2)
        self.assertEqual(words[0]["start"], 3)

    def test_exact_mapping_repeated_unicode_and_punctuation(self):
        processor = ForceAlignProcessor()
        cases = [("English", "  Go, go! State-of-the-art café 3.5%?\r\n"),
                 ("Chinese", "  今天，今天！AI模型3.5%。\n"),
                 ("Japanese", "人工知能の未来です。次の文です。"),
                 ("Korean", "인공지능의 미래입니다. 다음 문장입니다.")]
        for language, text in cases:
            with self.subTest(language=language):
                tokens, _ = processor.encode_timestamp(text, language)
                spans = forced.source_spans(text, tokens)
                self.assertEqual(len(spans), len(tokens))
                self.assertEqual("".join(server._alignment_text_identity(text[a:b]) for a, b in spans), "".join(tokens))
        with self.assertRaises(forced.AlignmentInputError):
            forced.source_spans("Hello", ["hello"])
        with self.assertRaises(forced.AlignmentInputError):
            forced.source_spans("...", [])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name) / "audio.wav"
        make_wav(self.audio)
        self.model = FakeModel()
        self.aligner = server.MLXAligner("unused")
        self.aligner._model = self.model
        for name, value in (("aligner", self.aligner),):
            patch = mock.patch.object(server, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        for name in ("load", "is_ready", "transcribe"):
            patch = mock.patch.object(server.transcriber, name, side_effect=AssertionError("ASR must not be called"))
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(server, "generate_transcription", side_effect=AssertionError("ASR generation prohibited"))
        patch.start()
        self.addCleanup(patch.stop)

    async def pipeline(self, text="Hello, hello!\r\n", language="en", regions=None):
        return await server._run_alignment_path_pipeline(
            request_id="alignment-test", audio_path=str(self.audio),
            alignment_request=forced.parse_request(text, language, "verbose_json", regions))

    def test_asr_unavailable_exact_source_spans_and_truthful_payload(self):
        text = "  Hello, hello!\r\nCafé. \t"
        result = asyncio.run(self.pipeline(text))
        payload = result.payload()
        self.assertEqual(payload["task"], "align")
        self.assertEqual(payload["text"], text)
        self.assertEqual("".join(s["text"] for s in payload["segments"]), text)
        self.assertFalse(payload["diagnostics"]["acoustic_accuracy_verified"])
        self.assertEqual(payload["timestamp_provenance"]["timestamp_repair"], "none")
        for word in payload["words"]:
            self.assertEqual(word["word"], text[word["text_start"]:word["text_end"]])
        for segment in payload["segments"]:
            self.assertEqual(segment["text"], text[segment["text_start"]:segment["text_end"]])
            self.assertNotIn("avg_logprob", segment)
        self.assertTrue(all(not p.exists() for p in self.model.paths))
        self.assertIsInstance(self.model.aligner_processor, ForceAlignProcessor)

    def test_combining_marks_belong_to_the_original_word_span(self):
        result = asyncio.run(self.pipeline("cafe\u0301 naïve."))
        self.assertEqual(result.words[0]["word"], "cafe\u0301")
        self.assertEqual(result.words[0]["alignment_token"], "cafe")
        self.assertEqual(result.words[0]["text_end"], 5)

    def test_sample_endpoint_does_not_gain_a_float_rounding_overshoot(self):
        make_wav(self.audio, 1.1)
        self.model.timestamps = [0, 80]
        regions = [{"start": 1.0000625, "end": 1.0400625, "text_start": 0, "text_end": 5}]
        result = asyncio.run(self.pipeline("hello", regions=regions))
        self.assertEqual(result.words[0]["end"], result.regions[0]["end"])
        self.assertEqual(result.diagnostics["boundary_adjustments"][0]["adjusted"], result.regions[0]["end"])

    def test_multiregion_global_timing_and_known_source_offsets(self):
        make_wav(self.audio, 35)
        text = "Hello.  Hello!\n"
        regions = [{"start": 2.00003, "end": 3.00002, "text_start": 0, "text_end": 8},
                   {"start": 32, "end": 34, "text_start": 8, "text_end": len(text)}]
        result = asyncio.run(self.pipeline(text, regions=regions))
        self.assertEqual([w["start"] for w in result.words], [2, 32])
        self.assertEqual([call["frames"] for call in self.model.calls], [16000, 32000])
        self.assertEqual("".join(s["text"] for s in result.segments), text)
        self.assertEqual(result.segments[1]["word_start"], 1)
        self.assertEqual(result.duration, 35)
        self.assertEqual(result.words[1]["text_start"], 8)

    def test_all_regions_validate_before_any_inference(self):
        cases = [("hello", [{"start": 0, "end": 1, "text_start": 0, "text_end": 2},
                            {"start": 1, "end": 2, "text_start": 2, "text_end": 5}]),
                 ("hello ...", [{"start": 0, "end": 1, "text_start": 0, "text_end": 6},
                                {"start": 1, "end": 2, "text_start": 6, "text_end": 9}])]
        for text, regions in cases:
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(self.pipeline(text, regions=regions))
            self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(self.model.calls, [])

    def test_upstream_model_wrapper_processor_owner(self):
        class Wrapper:
            def __init__(self, model):
                self._model = model
            def __getattr__(self, name):
                return getattr(self._model, name)
        self.aligner._model = Wrapper(self.model)
        result = asyncio.run(self.pipeline("hello"))
        self.assertEqual(result.words[0]["word"], "hello")
        self.assertIsInstance(self.model.aligner_processor, ForceAlignProcessor)
        self.assertNotIn("aligner_processor", vars(self.aligner._model))

    def test_unit_and_encoded_text_limits(self):
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(self.pipeline("a " * 257))
        self.assertEqual(ctx.exception.status_code, 400)
        self.model._tokenizer.encode = lambda *a, **kw: [0] * 2049
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(self.pipeline("Hello"))
        self.assertIn("2048", str(ctx.exception.detail))
        self.assertEqual(self.model.calls, [])

    def test_truncated_source_is_rejected_even_outside_requested_regions(self):
        self.audio.write_bytes(self.audio.read_bytes()[:-100])
        regions = [{"start": 0, "end": 1, "text_start": 0, "text_end": 5}]
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(self.pipeline("hello", regions=regions))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("truncated", ctx.exception.detail)
        self.assertEqual(self.model.calls, [])

    def test_overlimit_audio_rejected_before_loading_model(self):
        make_wav(self.audio, 30.01)
        with mock.patch.object(self.aligner, "prepare_standalone") as prepare:
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(self.pipeline())
        self.assertIn("--regions-file", str(ctx.exception.detail))
        prepare.assert_not_called()

    def test_unsupported_language_dependency_missing_aligner(self):
        for language in ("auto", "xx", "French"):
            with self.assertRaises((HTTPException, forced.AlignmentInputError)):
                asyncio.run(self.pipeline(language=language))
        with mock.patch.object(server, "_alignment_dependency_missing", return_value="missing dependency nagisa"):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(self.pipeline(language="ja"))
            self.assertIn("nagisa", ctx.exception.detail)
        self.aligner._model = None
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(self.pipeline())
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(self.model.calls, [])

    def test_no_upstream_timestamp_repair_or_legacy_clamping(self):
        self.model.timestamps = [160, 80, 240, 320]
        original = self.model.aligner_processor
        with mock.patch.object(original, "fix_timestamp", side_effect=AssertionError("No repair")):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(self.pipeline("hello hello"))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIs(self.model.aligner_processor, original)
        self.assertTrue(all(not p.exists() for p in self.model.paths))

    def test_zero_duration_json_diagnostic_and_subtitle_failure(self):
        self.model.timestamps = [0, 0]
        result = asyncio.run(self.pipeline("Hello"))
        self.assertEqual(result.diagnostics["zero_duration_word_indices"], [0])
        self.assertEqual(result.payload()["timestamp_provenance"]["alignment"], "applied_with_warnings")
        for fmt in ("srt", "vtt"):
            with self.assertRaises(forced.AlignmentOutputError):
                server._alignment_output(result, fmt)

    def test_subtitle_rejects_positive_interval_that_rounds_to_zero(self):
        make_wav(self.audio, 1 / 16000)
        self.model.timestamps = [0, 80]
        result = asyncio.run(self.pipeline("Hello"))
        self.assertGreater(result.words[0]["end"], result.words[0]["start"])
        for fmt in ("srt", "vtt"):
            with self.assertRaises(forced.AlignmentOutputError) as ctx:
                server._alignment_output(result, fmt)
            self.assertIn("millisecond", str(ctx.exception))

    def test_decoding_failure_and_converted_audio_cleanup(self):
        self.audio.write_bytes(b"compressed")
        converted = Path(self.temp.name) / "converted.wav"
        make_wav(converted)
        with mock.patch.object(server, "_transcode_with_ffmpeg", return_value=(str(converted), None)):
            result = asyncio.run(self.pipeline("Hello"))
        self.assertFalse(converted.exists())
        self.assertEqual(result.text, "Hello")
        with mock.patch.object(server, "_transcode_with_ffmpeg", return_value=(None, "decode failed")):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(self.pipeline())
        self.assertEqual(ctx.exception.status_code, 400)

    def test_running_cancellation_keeps_files_until_native_exit_and_stops_next_region(self):
        started, release = threading.Event(), threading.Event()
        observed = []
        def block(path):
            started.set()
            release.wait(3)
            observed.append(path.exists())
        self.model.on_generate = block
        regions = [{"start": 0, "end": 1, "text_start": 0, "text_end": 6},
                   {"start": 1, "end": 2, "text_start": 6, "text_end": 11}]
        async def scenario():
            task = asyncio.create_task(self.pipeline("Hello hello", regions=regions))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                task.cancel()
                await asyncio.sleep(.02)
                self.assertFalse(task.done())
                self.assertTrue(self.model.paths[0].exists())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        asyncio.run(scenario())
        self.assertEqual(observed, [True])
        self.assertEqual(len(self.model.calls), 1)
        self.assertFalse(self.model.paths[0].exists())

    def test_queued_cancellation_leaves_inference_queue_and_cleans_conversion(self):
        self.audio.write_bytes(b"compressed")
        converted = Path(self.temp.name) / "converted.wav"
        make_wav(converted)
        async def scenario():
            task = asyncio.create_task(self.pipeline())
            for _ in range(200):
                if server._GLOBAL_INFERENCE_LOCK.phase_for("alignment-test") == "queued":
                    break
                await asyncio.sleep(.005)
            self.assertEqual(server._GLOBAL_INFERENCE_LOCK.phase_for("alignment-test"), "queued")
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        server._GLOBAL_INFERENCE_LOCK.acquire()
        try:
            with mock.patch.object(server, "_transcode_with_ffmpeg", return_value=(str(converted), None)):
                asyncio.run(scenario())
        finally:
            server._GLOBAL_INFERENCE_LOCK.release()
        self.assertEqual(self.model.calls, [])
        self.assertFalse(converted.exists())

    def test_http_cli_parity_and_http_cap_does_not_apply_to_cli(self):
        client = TestClient(server.app)
        output = Path(self.temp.name) / "result.json"
        for fmt in ("verbose_json", "srt", "vtt"):
            data = {"text": "Hello!", "language": "en", "response_format": fmt}
            response = client.post("/v1/audio/alignments", data=data, files={"file": ("a.wav", self.audio.read_bytes())})
            self.assertEqual(response.status_code, 200, response.text)
            with mock.patch.object(server, "MAX_UPLOAD_BYTES", 1):
                code = server.main(["align", str(self.audio), "--text", "Hello!", "--language", "en", "--response-format", fmt, "-o", str(output)])
            self.assertEqual(code, 0)
            content = output.read_bytes()
            if fmt == "verbose_json":
                self.assertEqual(json.loads(content), response.json())
            else:
                self.assertEqual(content, response.content)
        with mock.patch.object(server, "MAX_UPLOAD_BYTES", 1):
            response = client.post("/audio/alignments", data=data, files={"file": ("a.wav", self.audio.read_bytes())})
        self.assertEqual(response.status_code, 413)

    def test_http_validation_empty_and_alias_route(self):
        client = TestClient(server.app)
        for data in ({}, {"text": "hello"}, {"text": "hello", "language": "en", "regions": "{}"},
                     {"text": "hello", "language": "en", "response_format": "json"}):
            response = client.post("/audio/alignments", data=data, files={"file": ("a.wav", self.audio.read_bytes())})
            self.assertEqual(response.status_code, 400)
        response = client.post("/audio/alignments", data={"text": "hello", "language": "en"}, files={"file": ("a.wav", b"")})
        self.assertEqual(response.status_code, 400)
        response = client.post("/audio/alignments", data={"text": "hello", "language": "en"}, files={"file": ("a.wav", self.audio.read_bytes())})
        self.assertEqual(response.status_code, 200)

    def test_subtitle_literal_markup_and_sentence_text_coverage(self):
        result = asyncio.run(self.pipeline("Rock & roll <tag>.\r\n"))
        content = server._alignment_output(result, "vtt")
        self.assertIn("Rock &amp; roll &lt;tag&gt;.\r\n", content)
        self.assertEqual(result.text, "Rock & roll <tag>.\r\n")

    def test_http_model_error_does_not_log_transcript_and_cleans_upload(self):
        captured = []
        original = server._alignment_path_sync
        def observe(path, request, timeout):
            captured.append(Path(path))
            return original(path, request, timeout)
        def fail(_):
            raise RuntimeError("private-supplied-transcript")
        self.model.on_generate = fail
        with mock.patch.object(server, "_alignment_path_sync", side_effect=observe), self.assertLogs(server.logger, level="ERROR") as logs:
            response = TestClient(server.app).post("/v1/audio/alignments", data={"text": "hello", "language": "en"}, files={"file": ("a.wav", self.audio.read_bytes())})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-supplied-transcript", str(logs.output) + response.text)
        self.assertTrue(all(not p.exists() for p in captured + self.model.paths))
        self.assertEqual(server.runtime_metrics.snapshot()["admission"]["in_flight"], 0)

    def test_cli_utf8_crlf_regions_and_all_source_aliases(self):
        transcript = Path(self.temp.name) / "text.txt"
        transcript.write_bytes(b"Hello!\r\n")
        regions = Path(self.temp.name) / "regions.json"
        regions.write_text(json.dumps([{"start": 0, "end": 1, "text_start": 0, "text_end": 8}]))
        output = Path(self.temp.name) / "result.json"
        argv = ["align", str(self.audio), "--text-file", str(transcript), "--language", "en", "--regions-file", str(regions)]
        self.assertEqual(server.main([*argv, "-o", str(output)]), 0)
        self.assertEqual(json.loads(output.read_bytes())["text"], "Hello!\r\n")
        for source in (self.audio, transcript, regions):
            original = source.read_bytes()
            for kind in ("direct", "symlink", "hardlink"):
                alias = Path(self.temp.name) / f"alias-{source.name}-{kind}"
                if kind == "symlink":
                    alias.symlink_to(source)
                elif kind == "hardlink":
                    os.link(source, alias)
                else:
                    alias = source
                self.assertEqual(server.main([*argv, "-o", str(alias)]), 2)
                self.assertEqual(source.read_bytes(), original)
        transcript.write_bytes(b"\xff")
        old_output = output.read_bytes()
        self.assertEqual(server.main([*argv, "-o", str(output)]), 2)
        self.assertEqual(output.read_bytes(), old_output)

    def test_cli_errors_interruption_and_atomic_output(self):
        output = Path(self.temp.name) / "result.json"
        output.write_text("keep")
        argv = ["align", str(self.audio), "--text", "hello", "--language", "en", "-o", str(output)]
        for error, expected in ((HTTPException(400, "input"), 2), (HTTPException(503, "model"), 1), (KeyboardInterrupt(), 130)):
            with mock.patch.object(server, "_run_alignment_path_pipeline", side_effect=error):
                self.assertEqual(server.main(argv), expected)
            self.assertEqual(output.read_text(), "keep")
        with mock.patch.object(server.os, "replace", side_effect=OSError("write failure")):
            self.assertEqual(server.main(argv), 1)
        self.assertEqual(output.read_text(), "keep")
        self.assertEqual(list(output.parent.glob(".result.json.*.tmp")), [])


class AdmissionTests(unittest.TestCase):
    def test_both_routes_reject_before_reading_body_with_retry_after(self):
        for path in ("/audio/alignments", "/v1/audio/alignments"):
            metrics = server.RuntimeMetrics(0)
            held = metrics.try_admit("held")
            endpoint = mock.AsyncMock()
            receive = mock.AsyncMock(side_effect=AssertionError("must not read rejected body"))
            sent = []
            async def send(message):
                sent.append(message)
            asyncio.run(server.AdmissionMetricsMiddleware(endpoint, metrics)(
                {"type": "http", "method": "POST", "path": path}, receive, send))
            endpoint.assert_not_called()
            self.assertEqual(sent[0]["status"], 429)
            self.assertIn((b"retry-after", b"1"), sent[0]["headers"])
            held.finish("completed")

    def test_alignment_readiness_is_additive_when_asr_missing(self):
        with mock.patch.object(server.transcriber, "readiness_snapshot", return_value={"ready": False}), mock.patch.object(server.aligner, "readiness_snapshot", return_value={"ready": True}):
            payload, status = server._health_snapshot()
        self.assertEqual(status, 503)
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["components"]["alignment"]["ready"])


if __name__ == "__main__":
    unittest.main()
