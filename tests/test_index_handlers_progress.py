"""Progress lifecycle of run_incremental_index: the job must end on every
exit path — finish on success, fail on exception — otherwise the dashboard
shows a stuck "running" job forever (running entries are never TTL-evicted)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mira.core.progress import ProgressTracker, indexing_key

import mira.core.progress as progress_module
import mira.platforms.index_handlers as ih


@pytest.fixture()
def isolated(monkeypatch: pytest.MonkeyPatch) -> ProgressTracker:
    """Stub out everything run_incremental_index touches before the indexing
    call itself; the tracker is swapped for a fresh instance."""
    tracker = ProgressTracker()
    monkeypatch.setattr(progress_module, "tracker", tracker)
    monkeypatch.setattr(ih, "_get_app_db", lambda: None)
    monkeypatch.setattr(ih, "load_config", lambda: SimpleNamespace(llm={}))
    monkeypatch.setattr(
        "mira.dashboard.models_config.llm_config_for", lambda purpose, base: None
    )
    monkeypatch.setattr(ih, "create_llm", lambda cfg: SimpleNamespace(progress_key=None))
    return tracker


def _fake_store(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStore:
        def all_paths(self):
            return ["a.py"]

        def close(self):
            pass

        @staticmethod
        def open(*args, **kwargs):
            return FakeStore()

    monkeypatch.setattr(ih, "IndexStore", FakeStore)


async def test_success_finishes_job(isolated: ProgressTracker, monkeypatch):
    _fake_store(monkeypatch)

    async def fake_index_diff(**kwargs):
        return 3

    monkeypatch.setattr(ih, "index_diff", fake_index_diff)

    await ih.run_incremental_index(
        owner="acme",
        repo="api",
        fetcher=None,
        changed_paths=["a.py"],
        removed_paths=[],
        default_branch="main",
    )

    job = isolated.get(indexing_key("acme", "api"))
    assert job is not None
    assert job.status == "completed"


async def test_indexing_error_fails_job_and_reraises(
    isolated: ProgressTracker, monkeypatch: pytest.MonkeyPatch
):
    _fake_store(monkeypatch)

    async def boom(**kwargs):
        raise RuntimeError("llm exploded")

    monkeypatch.setattr(ih, "index_diff", boom)

    with pytest.raises(RuntimeError, match="llm exploded"):
        await ih.run_incremental_index(
            owner="acme",
            repo="api",
            fetcher=None,
            changed_paths=["a.py"],
            removed_paths=[],
            default_branch="main",
        )

    job = isolated.get(indexing_key("acme", "api"))
    assert job is not None
    assert job.status == "failed"
    assert "llm exploded" in job.error


async def test_store_open_error_fails_job(isolated: ProgressTracker, monkeypatch):
    class BoomStore:
        @staticmethod
        def open(*args, **kwargs):
            raise RuntimeError("no store")

    monkeypatch.setattr(ih, "IndexStore", BoomStore)

    with pytest.raises(RuntimeError, match="no store"):
        await ih.run_incremental_index(
            owner="acme",
            repo="api",
            fetcher=None,
            changed_paths=["a.py"],
            removed_paths=[],
            default_branch="main",
        )

    job = isolated.get(indexing_key("acme", "api"))
    assert job is not None
    assert job.status == "failed"
