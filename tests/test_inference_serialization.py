from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

import whisper_mlx_server as server


class _FakeAlignerModel:
    def __init__(self, tracker: "_ConcurrencyTracker") -> None:
        self._tracker = tracker

    def generate(self, *, audio: str, text: str, language: str) -> list[dict[str, object]]:
        self._tracker.run("aligner")
        return [{"word": text, "start": 0.0, "end": 0.1}]


class _ConcurrencyTracker:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    def run(self, name: str) -> None:
        with self._guard:
            self.calls.append(f"{name}:start")
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.05)
        finally:
            with self._guard:
                self.calls.append(f"{name}:end")
                self.active -= 1


class InferenceSerializationTests(unittest.TestCase):
    def test_global_instances_share_same_inference_lock(self) -> None:
        self.assertIs(server.transcriber._infer_lock, server.aligner._infer_lock)

    def test_transcriber_and_aligner_do_not_overlap_when_sharing_lock(self) -> None:
        tracker = _ConcurrencyTracker()
        shared_lock = threading.Lock()
        transcriber = server.MLXTranscriber("/tmp/asr", infer_lock=shared_lock)
        aligner = server.MLXAligner("/tmp/aligner", infer_lock=shared_lock)
        transcriber._model = object()
        aligner._model = _FakeAlignerModel(tracker)

        def fake_generate_transcription(**_: object) -> dict[str, object]:
            tracker.run("transcriber")
            return {"text": "hello"}

        with mock.patch.object(server, "generate_transcription", side_effect=fake_generate_transcription):
            thread1 = threading.Thread(
                target=transcriber.transcribe,
                args=("/tmp/audio.wav", None, None),
            )
            thread2 = threading.Thread(
                target=aligner.align,
                args=("/tmp/audio.wav", "hello", "en"),
            )
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()

        self.assertEqual(tracker.max_active, 1)
        self.assertEqual(len(tracker.calls), 4)
        self.assertIn(tracker.calls[:2], (["transcriber:start", "transcriber:end"], ["aligner:start", "aligner:end"]))
        self.assertIn(tracker.calls[2:], (["transcriber:start", "transcriber:end"], ["aligner:start", "aligner:end"]))
        self.assertNotEqual(tracker.calls[0].split(":")[0], tracker.calls[2].split(":")[0])


if __name__ == "__main__":
    unittest.main()
