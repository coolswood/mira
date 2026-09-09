"""Tests for the live progress tracker (core/progress.py)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from mira.core.progress import (
    INDEXING,
    REVIEW,
    ProgressTracker,
    indexing_key,
    review_key,
)


class TestJobLifecycle:
    def test_begin_creates_running_job(self):
        tracker = ProgressTracker()
        tracker.begin("acme/api#1", REVIEW, "acme/api", pr_number=1, pr_title="T")

        job = tracker.get("acme/api#1")
        assert job is not None
        assert job.status == "running"
        assert job.kind == REVIEW
        assert job.pr_number == 1

    def test_finish_sets_terminal_state_and_duration(self):
        tracker = ProgressTracker()
        tracker.begin("acme/api#1", REVIEW, "acme/api", pr_number=1)

        job = tracker.get("acme/api#1")
        assert job is not None
        job.started_at = time.time() - 10
        tracker.finish("acme/api#1")

        job = tracker.get("acme/api#1")
        assert job is not None
        assert job.status == "completed"
        assert job.finished_at > 0
        assert job.elapsed_s() >= 9

    def test_fail_records_error(self):
        tracker = ProgressTracker()
        tracker.begin("acme/api#2", REVIEW, "acme/api", pr_number=2)
        tracker.fail("acme/api#2", "boom")

        job = tracker.get("acme/api#2")
        assert job is not None
        assert job.status == "failed"
        assert job.error == "boom"

    def test_active_and_all_queries(self):
        tracker = ProgressTracker()
        tracker.begin("a#1", REVIEW, "a", pr_number=1)
        tracker.begin("a#2", REVIEW, "a", pr_number=2)
        tracker.begin("index:a", INDEXING, "a")
        tracker.finish("a#1")

        assert [j.key for j in tracker.get_active()] == ["a#2", "index:a"]
        assert len(tracker.get_all()) == 3


class TestNoopKeys:
    """None keys and unknown keys must be silent no-ops — callers without a
    tracked job (CLI paths, tests) stay untouched."""

    def test_none_key_noops_everywhere(self):
        tracker = ProgressTracker()
        tracker.set_stage(None, "review")
        tracker.plan_chunks(None, 3)
        tracker.chunk_started(None, 1, 2)
        tracker.chunk_finished(None, 1)
        tracker.set_files(None, 5)
        tracker.call_started(None)
        tracker.call_item(None, "reasoning")
        tracker.call_usage(None, input_tokens=1)
        tracker.call_finished(None, ok=True, duration_s=1)
        tracker.finish(None)
        assert tracker.get_all() == []

    def test_unknown_key_noops(self):
        tracker = ProgressTracker()
        tracker.begin("known", REVIEW, "r")
        tracker.set_stage("unknown", "review")
        tracker.chunk_finished("unknown", 1)
        tracker.finish("unknown")

        job = tracker.get("known")
        assert job is not None
        assert job.status == "running"


class TestReviewWorkUnits:
    def test_chunk_progress_and_eta(self):
        tracker = ProgressTracker()
        key = review_key("acme", "api", 9)
        tracker.begin(key, REVIEW, "acme/api", pr_number=9)
        tracker.plan_chunks(key, 4)

        job = tracker.get(key)
        assert job is not None
        job.started_at = time.time() - 60

        tracker.chunk_started(key, 1, 10)
        tracker.chunk_finished(key, 1, comments=2)
        tracker.chunk_finished(key, 2)

        job = tracker.get(key)
        assert job is not None
        assert job.chunks_done == 2
        eta = job.eta_s()
        assert eta is not None and eta > 0
        # Two of four chunks took 60s -> ~60s remaining (capped at elapsed).
        assert eta <= 61

    def test_eta_is_none_without_chunks(self):
        tracker = ProgressTracker()
        tracker.begin("a#1", REVIEW, "a", pr_number=1)
        job = tracker.get("a#1")
        assert job is not None
        assert job.eta_s() is None

    def test_indexing_jobs_never_report_chunk_eta(self):
        tracker = ProgressTracker()
        tracker.begin(indexing_key("acme", "api"), INDEXING, "acme/api")
        key = indexing_key("acme", "api")
        tracker.plan_chunks(key, 3)
        tracker.chunk_finished(key, 1)
        job = tracker.get(key)
        assert job is not None
        assert job.eta_s() is None

    def test_stage_transitions_recorded_once(self):
        tracker = ProgressTracker()
        key = "a#1"
        tracker.begin(key, REVIEW, "a", pr_number=1)
        tracker.set_stage(key, "review")
        tracker.set_stage(key, "review")  # duplicate — ignored

        job = tracker.get(key)
        assert job is not None
        assert job.stage == "review"
        stage_events = [e for e in job.events if e.kind == "stage_changed"]
        assert len(stage_events) == 1


class TestCallAccounting:
    def test_call_lifecycle_and_per_stage_counts(self):
        tracker = ProgressTracker()
        key = "a#1"
        tracker.begin(key, REVIEW, "a", pr_number=1)
        tracker.set_stage(key, "review")

        tracker.call_started(key)
        tracker.call_started(key)
        tracker.call_item(key, "reasoning")
        tracker.call_item(key, "reasoning")
        tracker.call_item(key, "agent_message")
        tracker.call_usage(
            key,
            input_tokens=100,
            cached_input_tokens=20,
            output_tokens=10,
            reasoning_output_tokens=5,
        )
        tracker.call_finished(key, ok=True, duration_s=4.2)
        tracker.call_finished(key, ok=False, duration_s=1.0, error="timeout")

        job = tracker.get(key)
        assert job is not None
        assert job.calls_started == 2
        assert job.calls_in_flight == 0
        assert job.calls_failed == 1
        assert job.calls_by_stage == {"review": 2}
        assert job.items == {"reasoning": 2, "agent_message": 1}
        assert job.tokens_input == 100
        assert job.tokens_cached == 20
        assert job.tokens_output == 10
        assert job.tokens_reasoning == 5

    def test_snapshot_shape(self):
        tracker = ProgressTracker()
        key = "a#1"
        tracker.begin(key, REVIEW, "a", pr_number=1, pr_title="hi")
        tracker.plan_chunks(key, 2)
        tracker.call_started(key)
        data = tracker.get(key).as_dict()  # type: ignore[union-attr]

        assert data["key"] == key
        assert data["status"] == "running"
        assert data["chunks_total"] == 2
        assert data["calls_in_flight"] == 1
        assert data["events"][-1]["kind"] == "call_started"
        assert data["eta_s"] is None


class TestEventPublishing:
    async def test_snapshot_reaches_sse_bus(self, monkeypatch: pytest.MonkeyPatch):
        """_publish resolves the bus lazily, so patching the module attribute
        reroutes events into our test bus."""
        from mira.dashboard import events as events_module
        from mira.dashboard.events import EventBus

        bus = EventBus()
        monkeypatch.setattr(events_module, "bus", bus)

        queue = await bus.subscribe()
        tracker = ProgressTracker()
        tracker.begin("a#1", REVIEW, "a", pr_number=1)

        event = await asyncio.wait_for(queue.get(), timeout=2)
        assert event.type == "job_progress"
        assert event.data["job"]["key"] == "a#1"
        assert event.data["event"]["kind"] == "job_started"

    async def test_publish_failure_never_breaks_tracking(self, monkeypatch: pytest.MonkeyPatch):
        from mira.dashboard import events as events_module

        tracker = ProgressTracker()

        # A broken bus must not take tracking down — publish is best-effort.
        class BrokenBus:
            def emit(self, *args, **kwargs):
                raise RuntimeError("bus down")

        monkeypatch.setattr(events_module, "bus", BrokenBus())
        tracker.begin("a#2", REVIEW, "a", pr_number=2)
        assert tracker.get("a#2") is not None


class TestTtlEviction:
    def test_finished_jobs_evicted_after_ttl(self):
        tracker = ProgressTracker(ttl_seconds=0)
        tracker.begin("a#1", REVIEW, "a", pr_number=1)
        tracker.finish("a#1")
        # With ttl=0 any finished job is already evictable (the get() calls
        # below trigger eviction), so backdate the finish explicitly.
        job = tracker._jobs["a#1"]
        job.finished_at = time.time() - 10

        assert tracker.get("a#1") is None
        assert tracker.get_all() == []

    def test_running_jobs_never_evicted(self):
        tracker = ProgressTracker(ttl_seconds=0)
        tracker.begin("a#1", REVIEW, "a", pr_number=1)
        job = tracker.get("a#1")
        assert job is not None
        job.started_at = time.time() - 9999

        assert tracker.get("a#1") is not None


class TestLockedSnapshots:
    """snapshot_all/snapshot_active serialize under the lock. HTTP endpoints
    must render from these — live JobProgress objects are mutated by worker
    threads while a response is being built."""

    def test_snapshots_are_dicts_and_filtered(self):
        tracker = ProgressTracker()
        tracker.begin("a#1", REVIEW, "a", pr_number=1)
        tracker.begin("index:a", INDEXING, "a")
        tracker.finish("index:a")

        all_snaps = tracker.snapshot_all()
        active_snaps = tracker.snapshot_active()

        assert [j["key"] for j in all_snaps] == ["a#1", "index:a"]
        assert [j["key"] for j in active_snaps] == ["a#1"]
        assert all(isinstance(j, dict) for j in all_snaps + active_snaps)
        assert all_snaps[1]["status"] == "completed"
        assert isinstance(all_snaps[0]["events"], list)

    def test_snapshot_keys_match_as_dict(self):
        tracker = ProgressTracker()
        tracker.begin("a#1", REVIEW, "a", pr_number=1, pr_title="t")
        tracker.plan_chunks("a#1", 3)

        snap = tracker.snapshot_active()[0]
        job = tracker.get("a#1")
        assert job is not None
        assert set(snap) == set(job.as_dict())

    def test_snapshot_under_concurrent_event_writes(self):
        """Regression: serializing live objects outside the lock can raise
        'deque mutated during iteration' when workers append events."""

        tracker = ProgressTracker()
        tracker.begin("a#1", REVIEW, "a", pr_number=1)
        stop = threading.Event()

        def hammer():
            i = 0
            while not stop.is_set():
                tracker.add_event("a#1", "call_item", f"e{i}")
                i += 1

        worker = threading.Thread(target=hammer)
        worker.start()
        try:
            for _ in range(300):
                snaps = tracker.snapshot_active()
                assert len(snaps) == 1
                assert snaps[0]["key"] == "a#1"
        finally:
            stop.set()
            worker.join()
