from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest import mock

import whisper_mlx_server as server


class RuntimeMetricsTests(unittest.TestCase):
    def test_admission_memory_and_cancellation_metrics_share_one_request_lifecycle(self) -> None:
        metrics = server.RuntimeMetrics(max_queue_depth=1)
        first = metrics.try_admit("first")
        second = metrics.try_admit("second")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNone(metrics.try_admit("rejected"))
        assert first is not None
        assert second is not None

        first.set_audio_bytes(10)
        second.set_audio_bytes(7)
        metrics.note_cancellation_phase("first", "queued")
        first.finish("cancelled")
        second.finish("completed")

        snapshot = metrics.snapshot()
        self.assertEqual(
            snapshot["admission"],
            {
                "capacity": 2,
                "max_queue_depth": 1,
                "in_flight": 0,
                "peak_in_flight": 2,
                "attempts_total": 3,
                "admitted_total": 2,
                "rejected_total": 1,
                "completed_total": 1,
                "failed_total": 0,
            },
        )
        self.assertEqual(snapshot["memory"]["active_audio_bytes"], 0)
        self.assertEqual(snapshot["memory"]["peak_active_audio_bytes"], 17)
        self.assertEqual(snapshot["memory"]["decoded_audio_bytes_total"], 17)
        self.assertEqual(snapshot["memory"]["largest_request_audio_bytes"], 10)
        self.assertEqual(snapshot["cancellation"]["cancelled_requests_total"], 1)
        self.assertEqual(snapshot["cancellation"]["queued_total"], 1)


class InstrumentedInferenceLockTests(unittest.TestCase):
    def test_successful_wait_records_queue_depth_and_wait_duration(self) -> None:
        inference_lock = server.InstrumentedInferenceLock(poll_interval_seconds=0.005)
        waiter_finished = threading.Event()

        inference_lock.acquire()

        def wait_for_inference() -> None:
            request_token = server._CURRENT_REQUEST_ID.set("timed-waiter")
            try:
                with inference_lock:
                    pass
            finally:
                server._CURRENT_REQUEST_ID.reset(request_token)
                waiter_finished.set()

        thread = threading.Thread(target=wait_for_inference)
        thread.start()
        deadline = time.monotonic() + 1.0
        while inference_lock.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.02)
        inference_lock.release()
        self.assertTrue(waiter_finished.wait(timeout=1.0))
        thread.join(timeout=1.0)

        snapshot = inference_lock.snapshot()
        self.assertEqual(snapshot["peak_waiting"], 1)
        self.assertEqual(snapshot["acquisitions_total"], 2)
        self.assertEqual(snapshot["waited_acquisitions_total"], 1)
        self.assertEqual(snapshot["completed_total"], 2)
        self.assertGreater(snapshot["wait_seconds_total"], 0.0)
        self.assertGreater(snapshot["wait_seconds_max"], 0.0)
        self.assertEqual(
            snapshot["wait_seconds_average"],
            snapshot["wait_seconds_total"],
        )

    def test_cancelled_waiter_leaves_process_local_inference_queue(self) -> None:
        inference_lock = server.InstrumentedInferenceLock(poll_interval_seconds=0.005)
        cancel_event = threading.Event()
        waiter_started = threading.Event()
        waiter_finished = threading.Event()
        errors: list[BaseException] = []

        inference_lock.acquire()

        def wait_for_inference() -> None:
            request_token = server._CURRENT_REQUEST_ID.set("queued-request")
            cancellation_token = server._CURRENT_CANCELLATION_EVENT.set(cancel_event)
            waiter_started.set()
            try:
                with inference_lock:
                    errors.append(AssertionError("cancelled waiter acquired inference lock"))
            except server.InferenceWaitCancelled:
                pass
            except BaseException as exc:  # pragma: no cover - reported by the assertion below
                errors.append(exc)
            finally:
                server._CURRENT_CANCELLATION_EVENT.reset(cancellation_token)
                server._CURRENT_REQUEST_ID.reset(request_token)
                waiter_finished.set()

        thread = threading.Thread(target=wait_for_inference)
        thread.start()
        self.assertTrue(waiter_started.wait(timeout=1.0))

        deadline = time.monotonic() + 1.0
        while inference_lock.snapshot()["waiting"] != 1 and time.monotonic() < deadline:
            time.sleep(0.005)

        self.assertEqual(inference_lock.phase_for("queued-request"), "queued")
        cancel_event.set()
        self.assertTrue(waiter_finished.wait(timeout=1.0))
        thread.join(timeout=1.0)

        snapshot = inference_lock.snapshot()
        self.assertEqual(errors, [])
        self.assertEqual(snapshot["waiting"], 0)
        self.assertEqual(snapshot["cancelled_waiters_total"], 1)
        self.assertEqual(snapshot["active"], 1)
        inference_lock.release()


class BlockingCancellationTests(unittest.TestCase):
    def test_cancelling_queued_native_work_preserves_cancelled_error(self) -> None:
        request_id = "queued-cancel-test"
        worker_executed = threading.Event()
        lease = server.runtime_metrics.try_admit(request_id)
        self.assertIsNotNone(lease)
        assert lease is not None
        server._GLOBAL_INFERENCE_LOCK.acquire()

        def queued_work() -> None:
            with server._GLOBAL_INFERENCE_LOCK:
                worker_executed.set()

        async def scenario() -> None:
            task = asyncio.create_task(
                server._run_blocking_safely(request_id, queued_work)
            )
            deadline = asyncio.get_running_loop().time() + 1.0
            while (
                server._GLOBAL_INFERENCE_LOCK.phase_for(request_id) != "queued"
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.005)
            self.assertEqual(
                server._GLOBAL_INFERENCE_LOCK.phase_for(request_id),
                "queued",
            )
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        before = server.runtime_metrics.snapshot()["cancellation"]
        try:
            asyncio.run(scenario())
        finally:
            if server._GLOBAL_INFERENCE_LOCK.locked():
                server._GLOBAL_INFERENCE_LOCK.release()
            lease.finish("cancelled")
        after = server.runtime_metrics.snapshot()["cancellation"]

        self.assertFalse(worker_executed.is_set())
        self.assertEqual(
            after["cancelled_requests_total"],
            before["cancelled_requests_total"] + 1,
        )
        self.assertEqual(after["queued_total"], before["queued_total"] + 1)

    def test_running_native_work_finishes_before_cancellation_unwinds(self) -> None:
        request_id = "blocking-cancel-test"
        worker_started = threading.Event()
        worker_release = threading.Event()
        lease = server.runtime_metrics.try_admit(request_id)
        self.assertIsNotNone(lease)
        assert lease is not None

        def blocking_work() -> None:
            worker_started.set()
            worker_release.wait(timeout=2.0)

        async def scenario() -> None:
            task = asyncio.create_task(
                server._run_blocking_safely(request_id, blocking_work)
            )
            started = await asyncio.to_thread(worker_started.wait, 1.0)
            self.assertTrue(started)
            task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            worker_release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task

        before = server.runtime_metrics.snapshot()["cancellation"]
        try:
            asyncio.run(scenario())
        finally:
            worker_release.set()
            lease.finish("cancelled")
        after = server.runtime_metrics.snapshot()["cancellation"]

        self.assertEqual(
            after["cancelled_requests_total"],
            before["cancelled_requests_total"] + 1,
        )
        self.assertEqual(
            after["processing_total"],
            before["processing_total"] + 1,
        )


class HealthReadinessTests(unittest.TestCase):
    @staticmethod
    def _runtime_snapshot(*, in_flight: int = 0, capacity: int = 9) -> dict[str, object]:
        return {
            "admission": {
                "capacity": capacity,
                "max_queue_depth": capacity - 1,
                "in_flight": in_flight,
                "peak_in_flight": in_flight,
                "attempts_total": 0,
                "admitted_total": 0,
                "rejected_total": 0,
                "completed_total": 0,
                "failed_total": 0,
            },
            "memory": {},
            "cancellation": {},
        }

    @staticmethod
    def _queue_snapshot() -> dict[str, object]:
        return {
            "active": 0,
            "waiting": 0,
            "peak_waiting": 0,
            "acquisitions_total": 0,
            "waited_acquisitions_total": 0,
            "completed_total": 0,
            "cancelled_waiters_total": 0,
            "wait_seconds_total": 0.0,
            "wait_seconds_max": 0.0,
            "wait_seconds_average": 0.0,
        }

    def test_optional_aligner_does_not_hide_base_transcription_readiness(self) -> None:
        with (
            mock.patch.object(
                server.transcriber,
                "readiness_snapshot",
                return_value={"ready": True, "loaded": True, "available": True, "error": None},
            ),
            mock.patch.object(
                server.aligner,
                "readiness_snapshot",
                return_value={
                    "ready": False,
                    "loaded": False,
                    "available": False,
                    "error": "aligner missing",
                },
            ),
            mock.patch.object(
                server.runtime_metrics,
                "snapshot",
                return_value=self._runtime_snapshot(),
            ),
            mock.patch.object(
                server._GLOBAL_INFERENCE_LOCK,
                "snapshot",
                return_value=self._queue_snapshot(),
            ),
        ):
            payload, status_code = server._health_snapshot()

        self.assertEqual(status_code, 200)
        self.assertTrue(payload["ready"])
        self.assertTrue(payload["components"]["model"]["ready"])
        self.assertFalse(payload["components"]["aligner"]["ready"])
        self.assertTrue(payload["components"]["queue"]["ready"])
        self.assertIn("queue", payload["metrics"])

    def test_required_model_or_full_admission_capacity_returns_not_ready(self) -> None:
        with (
            mock.patch.object(
                server.transcriber,
                "readiness_snapshot",
                return_value={
                    "ready": False,
                    "loaded": False,
                    "available": True,
                    "error": "not loaded",
                },
            ),
            mock.patch.object(
                server.aligner,
                "readiness_snapshot",
                return_value={"ready": True, "loaded": True, "available": True, "error": None},
            ),
            mock.patch.object(
                server.runtime_metrics,
                "snapshot",
                return_value=self._runtime_snapshot(in_flight=1, capacity=1),
            ),
            mock.patch.object(
                server._GLOBAL_INFERENCE_LOCK,
                "snapshot",
                return_value=self._queue_snapshot(),
            ),
        ):
            payload, status_code = server._health_snapshot()

        self.assertEqual(status_code, 503)
        self.assertFalse(payload["ready"])
        self.assertFalse(payload["components"]["model"]["ready"])
        self.assertFalse(payload["components"]["queue"]["ready"])


class AdmissionMiddlewareTests(unittest.TestCase):
    def test_full_capacity_is_rejected_before_endpoint_execution(self) -> None:
        metrics = server.RuntimeMetrics(max_queue_depth=0)
        held = metrics.try_admit("held")
        self.assertIsNotNone(held)
        assert held is not None
        endpoint_called = False
        sent: list[dict[str, object]] = []

        async def endpoint(scope: object, receive: object, send: object) -> None:
            nonlocal endpoint_called
            endpoint_called = True

        middleware = server.AdmissionMetricsMiddleware(endpoint, metrics)

        async def receive() -> dict[str, object]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, object]) -> None:
            sent.append(message)

        try:
            asyncio.run(
                middleware(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/audio/transcriptions",
                        "state": {},
                    },
                    receive,
                    send,
                )
            )
        finally:
            held.finish("completed")

        response_start = next(item for item in sent if item["type"] == "http.response.start")
        self.assertFalse(endpoint_called)
        self.assertEqual(response_start["status"], 429)
        self.assertEqual(metrics.snapshot()["admission"]["rejected_total"], 1)


if __name__ == "__main__":
    unittest.main()
