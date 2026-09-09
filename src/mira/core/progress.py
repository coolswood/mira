"""Live progress tracking for long-running jobs (PR reviews, indexing).

A single in-memory tracker answers "is it working or wedged?" for every
long-running job: which stage it is in, how many chunks/files are done, how
many LLM calls are in flight and what they already cost in tokens. Providers
publish call-level events (``call_started`` / ``call_usage`` / ...), the
review engine publishes stage and chunk events, and the dashboard exposes the
snapshots over ``GET /api/progress`` plus the existing SSE stream, so the UI
can draw live progress bars without polling.

Thread-safe: LLM calls run inside asyncio tasks but trackers mutate from any
task (and from thread executors), so a plain lock guards all state.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Finished jobs stay queryable for this long, then are evicted — the durable
# record lives in the per-repo review events DB (same policy as review_status).
_TTL_SECONDS = 3600
_MAX_EVENTS = 30
_MAX_ITEMS = 20
# Indexing: emit an SSE-visible milestone every N processed files.
_FILES_EVENT_EVERY = 25

# Job kinds.
REVIEW = "review"
INDEXING = "indexing"

# Review stages (engine pipeline order; security/dependency pass in parallel
# with chunks, so their transitions appear in the event log but do not rewind
# the coarse stage).
STAGE_STARTING = "starting"
STAGE_WALKTHROUGH = "walkthrough"
STAGE_REVIEW = "review"
STAGE_CRITIQUE = "critique"
STAGE_SUMMARY = "summary"
STAGE_POSTING = "posting"
STAGE_FETCH = "fetching"  # indexing: repo tree/tarball
STAGE_FILES = "files"  # indexing: LLM file batches
STAGE_DIRECTORIES = "directories"  # indexing: directory summaries


def review_key(owner: str, repo: str, pr_number: int) -> str:
    return f"{owner}/{repo}#{pr_number}"


def indexing_key(owner: str, repo: str) -> str:
    return f"index:{owner}/{repo}"


@dataclass
class ProgressEvent:
    """One log-worthy thing that happened inside a job."""

    ts: float
    kind: str
    message: str
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "ts": self.ts,
            "kind": self.kind,
            "message": self.message,
            "data": dict(self.data),
        }


@dataclass
class JobProgress:
    """Mutable state of one long-running job."""

    key: str
    kind: str  # REVIEW | INDEXING
    repo: str  # "owner/repo"
    pr_number: int = 0
    pr_title: str = ""
    pr_url: str = ""
    status: str = "running"  # running | completed | failed | cancelled
    stage: str = STAGE_STARTING
    review_round: int = 1
    files_total: int = 0
    files_done: int = 0
    chunks_total: int = 0
    chunks_done: int = 0
    calls_started: int = 0
    calls_in_flight: int = 0
    calls_failed: int = 0
    calls_by_stage: dict = field(default_factory=dict)
    tokens_input: int = 0
    tokens_cached: int = 0
    tokens_output: int = 0
    tokens_reasoning: int = 0
    items: dict = field(default_factory=dict)  # codex item type -> count
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    error: str = ""
    events: deque = field(default_factory=lambda: deque(maxlen=_MAX_EVENTS))

    def elapsed_s(self, now: float | None = None) -> float:
        end = self.finished_at or (now or time.time())
        return max(0.0, end - self.started_at)

    def eta_s(self) -> float | None:
        """Estimated seconds remaining, based on finished chunks (reviews only)."""
        if self.kind != REVIEW or self.chunks_total <= 0 or self.chunks_done <= 0:
            return None
        if self.chunks_done >= self.chunks_total:
            return None
        elapsed = self.elapsed_s()
        if elapsed <= 0:
            return None
        # Chunks start after walkthrough/context setup, so the per-chunk rate
        # is derived from the review phase only; subtracting the whole elapsed
        # would overestimate remaining time on multi-chunk PRs. Clamp to the
        # elapsed time to stay conservative when setup dominated.
        rate = elapsed / self.chunks_done
        return max(0.0, rate * (self.chunks_total - self.chunks_done))

    def as_dict(self, now: float | None = None, include_events: bool = True) -> dict:
        eta = self.eta_s()
        data: dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "repo": self.repo,
            "pr_number": self.pr_number,
            "pr_title": self.pr_title,
            "pr_url": self.pr_url,
            "status": self.status,
            "stage": self.stage,
            "review_round": self.review_round,
            "files_total": self.files_total,
            "files_done": self.files_done,
            "chunks_total": self.chunks_total,
            "chunks_done": self.chunks_done,
            "calls_started": self.calls_started,
            "calls_in_flight": self.calls_in_flight,
            "calls_failed": self.calls_failed,
            "calls_by_stage": dict(self.calls_by_stage),
            "tokens_input": self.tokens_input,
            "tokens_cached": self.tokens_cached,
            "tokens_output": self.tokens_output,
            "tokens_reasoning": self.tokens_reasoning,
            "items": dict(self.items),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(self.elapsed_s(now), 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "error": self.error,
        }
        if include_events:
            data["events"] = [e.as_dict() for e in self.events]
        return data


class ProgressTracker:
    """Thread-safe registry of long-running job states (reviews + indexing)."""

    def __init__(self, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobProgress] = {}
        self._ttl = ttl_seconds

    # ── job lifecycle ──────────────────────────────────────────────

    def begin(
        self,
        key: str,
        kind: str,
        repo: str,
        pr_number: int = 0,
        pr_title: str = "",
        pr_url: str = "",
    ) -> None:
        with self._lock:
            self._evict_locked()
            self._jobs[key] = JobProgress(
                key=key,
                kind=kind,
                repo=repo,
                pr_number=pr_number,
                pr_title=pr_title,
                pr_url=pr_url,
            )
            self._record(key, "job_started", f"{kind} started for {repo}")

    def finish(self, key: str, status: str = "completed", error: str = "") -> None:
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                return
            job.status = status
            job.error = error
            job.finished_at = time.time()
            job.updated_at = job.finished_at
            message = f"{job.kind} {status} in {job.elapsed_s():.0f}s"
            if error:
                message += f": {error[:200]}"
            self._record(key, "job_finished", message, status=status)

    def fail(self, key: str, error: str) -> None:
        self.finish(key, status="failed", error=error)

    # ── stage / work units ─────────────────────────────────────────
    #
    # All key-taking methods accept ``None`` and no-op: callers without a
    # job (CLI review_diff, unit tests, other providers) stay untouched.

    def _job(self, key: str | None) -> JobProgress | None:
        """Locked job lookup that tolerates a None key."""
        if key is None:
            return None
        return self._jobs.get(key)

    def set_stage(self, key: str | None, stage: str, detail: str = "") -> None:
        with self._lock:
            job = self._job(key)
            if not job or job.stage == stage:
                return
            job.stage = stage
            message = f"stage: {stage}"
            if detail:
                message += f" — {detail}"
            self._record(key, "stage_changed", message, stage=stage)

    def set_round(self, key: str | None, review_round: int) -> None:
        with self._lock:
            job = self._job(key)
            if job:
                job.review_round = review_round

    def set_files(self, key: str | None, total: int, done: int = 0) -> None:
        with self._lock:
            job = self._job(key)
            if not job:
                return
            job.files_total = total
            job.files_done = done
            # Emit sparsely: every files milestone keeps the SSE feed alive
            # during long quiet stretches of LLM work without drowning the
            # event log (indexing batches complete every few seconds).
            if done and done % _FILES_EVENT_EVERY == 0:
                self._record(key, "files_progress", f"{done}/{total} files processed", done=done)

    def set_files_done(self, key: str | None, done: int) -> None:
        with self._lock:
            job = self._job(key)
            if job:
                job.files_done = done

    def plan_chunks(self, key: str | None, total: int) -> None:
        with self._lock:
            job = self._job(key)
            if not job:
                return
            job.chunks_total = total
            self._record(key, "chunks_planned", f"review planned in {total} chunk(s)")

    def chunk_started(self, key: str | None, index: int, files: int) -> None:
        with self._lock:
            job = self._job(key)
            if not job:
                return
            self._record(
                key,
                "chunk_started",
                f"chunk {index}/{job.chunks_total} started ({files} files)",
                chunk=index,
                files=files,
            )

    def chunk_finished(
        self, key: str | None, index: int, comments: int = 0, ok: bool = True
    ) -> None:
        with self._lock:
            job = self._job(key)
            if not job:
                return
            if ok:
                # Count completions, not the max finished index: chunks run
                # concurrently and finish out of order, so max() would jump
                # the counter to 100% while earlier chunks are still in flight.
                job.chunks_done += 1
            message = f"chunk {index}/{job.chunks_total} finished"
            if not ok:
                message += " (failed)"
            elif comments:
                message += f" — {comments} finding(s)"
            self._record(key, "chunk_finished", message, chunk=index, comments=comments)

    # ── LLM calls (published by providers) ─────────────────────────

    def call_started(self, key: str) -> None:
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                return
            job.calls_started += 1
            job.calls_in_flight += 1
            stage = job.stage
            job.calls_by_stage[stage] = job.calls_by_stage.get(stage, 0) + 1
            self._record(
                key,
                "call_started",
                f"LLM call #{job.calls_started} started ({stage}, {job.calls_in_flight} in flight)",
                stage=stage,
            )

    def call_item(self, key: str, item_type: str) -> None:
        """Record a streaming item observed inside a running call."""
        if not item_type:
            return
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                return
            job.items[item_type] = job.items.get(item_type, 0) + 1
            if sum(job.items.values()) % 10 == 1:  # avoid flooding the log
                self._record(key, "call_item", f"model activity: {item_type}", item=item_type)

    def call_usage(
        self,
        key: str,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_output_tokens: int = 0,
    ) -> None:
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                return
            job.tokens_input += max(0, input_tokens)
            job.tokens_cached += max(0, cached_input_tokens)
            job.tokens_output += max(0, output_tokens)
            job.tokens_reasoning += max(0, reasoning_output_tokens)

    def call_finished(self, key: str, ok: bool, duration_s: float, error: str = "") -> None:
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                return
            job.calls_in_flight = max(0, job.calls_in_flight - 1)
            if not ok:
                job.calls_failed += 1
            message = (
                f"LLM call finished in {duration_s:.0f}s"
                f" (in {job.tokens_input}, out {job.tokens_output})"
            )
            if not ok:
                message = f"LLM call failed after {duration_s:.0f}s: {error[:200]}"
            self._record(key, "call_finished", message, ok=ok)

    # ── free-form events ───────────────────────────────────────────

    def add_event(self, key: str, kind: str, message: str, **data: Any) -> None:
        with self._lock:
            self._record(key, kind, message, **data)

    # ── queries ────────────────────────────────────────────────────

    def get(self, key: str) -> JobProgress | None:
        with self._lock:
            self._evict_locked()
            return self._jobs.get(key)

    def get_active(self) -> list[JobProgress]:
        with self._lock:
            self._evict_locked()
            return [j for j in self._jobs.values() if j.status == "running"]

    def get_all(self) -> list[JobProgress]:
        with self._lock:
            self._evict_locked()
            return sorted(self._jobs.values(), key=lambda j: j.started_at)

    def snapshot_all(self) -> list[dict]:
        """Serialized snapshots taken under the lock.

        HTTP handlers must render from these, not from the live objects
        returned by ``get_all``/``get_active``: worker threads mutate jobs
        concurrently, and iterating a job's deque/dicts while a writer
        appends raises RuntimeError mid-request.
        """
        with self._lock:
            self._evict_locked()
            return [j.as_dict() for j in sorted(self._jobs.values(), key=lambda j: j.started_at)]

    def snapshot_active(self) -> list[dict]:
        with self._lock:
            self._evict_locked()
            return [j.as_dict() for j in self._jobs.values() if j.status == "running"]

    def __iter__(self) -> Iterator[JobProgress]:
        return iter(self.get_all())

    # ── internals ──────────────────────────────────────────────────

    def _record(self, key: str | None, kind: str, message: str, **data: Any) -> None:
        """Append an event and push the snapshot to SSE. Caller holds the lock."""
        job = self._job(key)
        if not job:
            return
        event = ProgressEvent(ts=time.time(), kind=kind, message=message, data=data)
        job.events.append(event)
        job.updated_at = event.ts
        _publish(job, event)

    def _evict_locked(self) -> None:
        now = time.time()
        dead = [
            key
            for key, job in self._jobs.items()
            if job.status != "running" and now - job.finished_at > self._ttl
        ]
        for key in dead:
            del self._jobs[key]


def _publish(job: JobProgress, event: ProgressEvent) -> None:
    """Best-effort push of the job snapshot to the dashboard SSE bus.

    Lazy import: mira.dashboard.events pulls no DB, but core must not depend
    on the dashboard package at module import time (serve/CLI entry points).
    """
    try:
        from mira.dashboard.events import bus

        bus.emit(
            "job_progress",
            {"job": job.as_dict(include_events=False), "event": event.as_dict()},
        )
    except Exception:  # pragma: no cover - outside serve mode
        pass


# Global singleton.
tracker = ProgressTracker()
