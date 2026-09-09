"""Failed review passes must be visible in the dashboard Activities feed.

A pipeline crash (LLM provider error, unexpected exception, …) used to leave
only a log traceback and an ephemeral in-memory progress failure — nothing the
user could see after the fact. These tests pin the contract: the failure is
persisted as a review event (status='failed') with a safe, secret-free error
summary, and flows through the /api/activity feed.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.dashboard import api
from mira.dashboard.db import AppDatabase
from mira.dashboard.routers import core
from mira.exceptions import LLMError
from mira.index.store import IndexStore
from mira.platforms.handlers import _safe_error_summary, run_pr_review


@pytest.fixture
def index_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point IndexStore at a scratch dir and force the SQLite backend."""
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return tmp_path


@pytest.fixture
def mock_app_db(monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    """A real AppDatabase (registered repo) wired into the dashboard API."""
    db = AppDatabase(url="", admin_password="admin")
    db.register_repo("acme", "web")
    monkeypatch.setattr(api, "_app_db", db)
    return db


def _failing_review_patches(exc: Exception):  # type: ignore[no-untyped-def]
    """Patch handlers so run_pr_review's engine raises *exc*."""
    engine = AsyncMock()
    engine.review_pr = AsyncMock(side_effect=exc)
    return (
        patch("mira.platforms.handlers.ReviewEngine", return_value=engine),
        patch("mira.platforms.handlers.create_llm", return_value=AsyncMock()),
        patch("mira.platforms.handlers.load_config", return_value=MagicMock()),
        patch("mira.dashboard.api._app_db", new_callable=MagicMock),
    )


async def _run_review() -> None:
    await run_pr_review(
        provider=AsyncMock(),
        owner="acme",
        repo="web",
        number=42,
        pr_url="https://github.com/acme/web/pull/42",
        is_private=False,
        bot_name="mira-bot",
        platform="github",
        pr_title="Fix auth redirect loop",
    )


async def test_failed_review_is_persisted_as_event(index_dir) -> None:
    """A crashing engine records a status='failed' review event (and re-raises)."""
    patches = _failing_review_patches(RuntimeError("kaboom"))
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        pytest.raises(RuntimeError, match="kaboom"),
    ):
        await _run_review()

    store = IndexStore.open("acme", "web")
    try:
        events = store.list_review_events()
        assert len(events) == 1
        ev = events[0]
        assert ev.status == "failed"
        assert ev.error == "RuntimeError: kaboom"
        assert ev.pr_number == 42
        assert ev.pr_title == "Fix auth redirect loop"
        assert ev.pr_url == "https://github.com/acme/web/pull/42"
        # Failure rows carry no findings — counters stay zeroed.
        assert ev.comments_posted == 0
        assert ev.blockers == 0
        assert ev.tokens_used == 0
    finally:
        store.close()


async def test_llm_error_uses_safe_message_not_internals(index_dir) -> None:
    """LLMError records the catalog's safe variant — never the raw internals."""
    err = LLMError("api_error", status=502, body="SUPER-SECRET payload trace")
    patches = _failing_review_patches(err)
    with patches[0], patches[1], patches[2], patches[3], pytest.raises(LLMError):
        await _run_review()

    store = IndexStore.open("acme", "web")
    try:
        (ev,) = store.list_review_events()
        assert ev.status == "failed"
        assert ev.error == "api_error: LLM API error 502"
        assert "SUPER-SECRET" not in ev.error
    finally:
        store.close()


async def test_failure_reachable_via_activity_feed(index_dir, mock_app_db) -> None:
    """The persisted failure flows through GET /api/activity with status/error."""
    patches = _failing_review_patches(RuntimeError("provider down"))
    with patches[0], patches[1], patches[2], patches[3], pytest.raises(RuntimeError):
        await _run_review()

    out = core.list_activity()
    failed = [e for e in out.events if e.status == "failed"]
    assert len(failed) == 1
    ev = failed[0]
    assert ev.owner == "acme"
    assert ev.repo == "web"
    assert ev.pr_number == 42
    assert "provider down" in ev.error
    assert ev.comments_posted == 0

    # The error text is searchable, so an admin can find the outage.
    by_error = core.list_activity(q="provider down")
    assert [e.id for e in by_error.events] == [ev.id]

    # Completed reviews still report the default status.
    store = IndexStore.open("acme", "web")
    try:
        store.record_review(
            pr_number=43,
            pr_title="Later PR",
            pr_url="https://github.com/acme/web/pull/43",
            comments_posted=1,
            blockers=0,
            warnings=1,
            created_at=999.0,
        )
    finally:
        store.close()
    statuses = {e.pr_number: e.status for e in core.list_activity().events}
    assert statuses == {42: "failed", 43: "completed"}


async def test_failure_stats_exclude_failed_passes(index_dir) -> None:
    """Failed passes don't pollute review counts or averages."""
    patches = _failing_review_patches(RuntimeError("nope"))
    with patches[0], patches[1], patches[2], patches[3], pytest.raises(RuntimeError):
        await _run_review()

    store = IndexStore.open("acme", "web")
    try:
        store.record_review(
            pr_number=7,
            pr_title="Good PR",
            pr_url="https://github.com/acme/web/pull/7",
            comments_posted=2,
            blockers=1,
            warnings=0,
            duration_ms=1500,
            created_at=50.0,
        )
        stats = store.get_review_stats()
        assert stats["total_reviews"] == 1  # the failed pass is not a review
        assert stats["total_comments"] == 2
        # The zero-duration failure pass must not dilute the average.
        assert stats["avg_duration_ms"] == 1500

        quality = store.get_review_quality_by_author("")
        assert quality["reviews"] == 1
        assert quality["blockers"] == 1
    finally:
        store.close()


def test_safe_error_summary_llm_error() -> None:
    err = LLMError("codex_timeout", seconds=900)
    assert _safe_error_summary(err) == "codex_timeout: Codex CLI timed out"


def test_safe_error_summary_generic_flattens_and_truncates() -> None:
    err = RuntimeError("line one\n  line two\t" + "x" * 500)
    summary = _safe_error_summary(err)
    assert summary.startswith("RuntimeError: line one line two")
    assert "\n" not in summary
    assert len(summary) <= 300


def test_migrated_rows_without_status_default_to_completed(tmp_path) -> None:
    """A pre-existing DB (before status/error columns) keeps working: old rows
    read as completed, and failure recording works on the migrated table."""
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE review_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, pr_number INTEGER NOT NULL DEFAULT 0, "
        "pr_title TEXT NOT NULL DEFAULT '', pr_url TEXT NOT NULL DEFAULT '', "
        "author TEXT NOT NULL DEFAULT '', comments_posted INTEGER NOT NULL DEFAULT 0, "
        "blockers INTEGER NOT NULL DEFAULT 0, warnings INTEGER NOT NULL DEFAULT 0, "
        "suggestions INTEGER NOT NULL DEFAULT 0, files_reviewed INTEGER NOT NULL DEFAULT 0, "
        "lines_changed INTEGER NOT NULL DEFAULT 0, tokens_used INTEGER NOT NULL DEFAULT 0, "
        "duration_ms INTEGER NOT NULL DEFAULT 0, categories TEXT NOT NULL DEFAULT '', "
        "author_avatar_url TEXT NOT NULL DEFAULT '', reviewed_paths TEXT NOT NULL DEFAULT '', "
        "created_at REAL NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO review_events (pr_number, pr_title, comments_posted, created_at) "
        "VALUES (1, 'Old PR', 3, 100.0)"
    )
    conn.commit()
    conn.close()

    store = IndexStore(db_path)
    try:
        old = store.list_review_events()[0]
        assert old.status == "completed"
        assert old.error == ""

        store.record_review_failure(
            pr_number=2,
            pr_title="New PR",
            pr_url="https://github.com/acme/web/pull/2",
            error="RuntimeError: modern problems",
        )
        events = {e.pr_number: e for e in store.list_review_events()}
        assert events[1].status == "completed"
        assert events[2].status == "failed"
        assert events[2].error == "RuntimeError: modern problems"
    finally:
        store.close()
