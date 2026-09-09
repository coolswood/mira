"""Parallel-model comparison: settings resolution + validation, model
attribution in the stores, the compare endpoint's round grouping and overlap
clustering, and the engine's shadow-mode gates (nothing posted to the
platform, everything recorded)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from mira.config import CompareModelConfig, LLMConfig, MiraConfig
from mira.core.engine import ReviewEngine
from mira.dashboard import api
from mira.dashboard.db import AppDatabase
from mira.dashboard.models_config import (
    MAX_COMPARE_MODELS,
    get_compare_models,
    llm_config_for_compare,
    validate_compare_entries,
)
from mira.index.store import IndexStore
from mira.llm.provider import LLMProvider
from mira.models import PRInfo


# ── Settings resolution + validation ─────────────────────────────────────────


class _FakeDB:
    """Settings-table stand-in: get_compare_models only reads one key."""

    def __init__(self, value: str | None) -> None:
        self.value = value

    def get_setting(self, key: str) -> str:
        assert key == "compare_models"
        return self.value or ""


def _patch_db(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    monkeypatch.setattr(api, "_app_db", _FakeDB(value))


class TestValidateCompareEntries:
    def test_happy_path_normalizes_effort(self):
        out = validate_compare_entries(
            [
                {"provider": "", "model": "gemini-3.8-flash-high", "reasoning_effort": "off"},
                {"provider": "codex-cli", "model": "gpt-5.1-codex", "reasoning_effort": "medium"},
            ],
            "antigravity-cli",
        )
        assert out[0] == {"provider": "", "model": "gemini-3.8-flash-high", "reasoning_effort": None}
        assert out[1]["reasoning_effort"] == "medium"

    def test_rejects_cap(self):
        entries = [{"provider": "", "model": f"m{i}"} for i in range(MAX_COMPARE_MODELS + 1)]
        with pytest.raises(ValueError, match="comparison model"):
            validate_compare_entries(entries, "antigravity-cli")

    def test_rejects_duplicate_model(self):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            validate_compare_entries(
                [{"provider": "", "model": "x"}, {"provider": "", "model": "x"}],
                "antigravity-cli",
            )

    def test_rejects_unknown_provider(self):
        with pytest.raises(ValueError, match="known provider"):
            validate_compare_entries([{"provider": "skynet", "model": "x"}], "antigravity-cli")

    def test_rejects_empty_model(self):
        with pytest.raises(ValueError):
            validate_compare_entries([{"provider": "", "model": "  "}], "antigravity-cli")

    def test_rejects_effort_the_backend_lacks(self):
        # antigravity-cli has no xhigh; the active backend applies when the
        # entry doesn't pin a provider.
        with pytest.raises(ValueError, match="not supported by antigravity-cli"):
            validate_compare_entries(
                [{"provider": "", "model": "x", "reasoning_effort": "xhigh"}],
                "antigravity-cli",
            )


class TestGetCompareModels:
    def test_blank_db_falls_back_to_config(self, monkeypatch: pytest.MonkeyPatch):
        _patch_db(monkeypatch, "")
        config = LLMConfig(compare_models=[CompareModelConfig(model="a"), CompareModelConfig(model="b")])
        assert [e.model for e in get_compare_models(config)] == ["a", "b"]

    def test_db_blob_shadows_config(self, monkeypatch: pytest.MonkeyPatch):
        _patch_db(monkeypatch, json.dumps([{"provider": "codex-cli", "model": "c", "reasoning_effort": "low"}]))
        config = LLMConfig(compare_models=[CompareModelConfig(model="a")])
        entries = get_compare_models(config)
        assert [e.model for e in entries] == ["c"]
        assert entries[0].provider == "codex-cli"

    def test_garbage_rows_dropped_and_capped(self, monkeypatch: pytest.MonkeyPatch):
        blob = json.dumps(
            [
                {"provider": "", "model": "ok"},
                {"provider": "bogus", "model": "nope"},
                {"model": ""},
            ]
            + [{"provider": "", "model": f"extra{i}"} for i in range(MAX_COMPARE_MODELS)]
        )
        _patch_db(monkeypatch, blob)
        entries = get_compare_models(LLMConfig())
        assert [e.model for e in entries] == ["ok", "extra0", "extra1"][:MAX_COMPARE_MODELS]


class TestLLMConfigForCompare:
    def test_same_provider_entry_keeps_transport(self):
        base = LLMConfig(provider="antigravity-cli", antigravity_home="/data/gemini")
        out = llm_config_for_compare(
            CompareModelConfig(provider="", model="gemini-3.8-flash-low", reasoning_effort=None),
            base,
        )
        assert out.provider == "antigravity-cli"
        assert out.model == "gemini-3.8-flash-low"
        assert out.antigravity_home == "/data/gemini"
        assert out.reasoning_effort is None

    def test_cross_provider_entry_swaps_provider_only(self):
        base = LLMConfig(provider="antigravity-cli", antigravity_home="/data/gemini", codex_home="/data/codex")
        out = llm_config_for_compare(
            CompareModelConfig(provider="codex-cli", model="gpt-5.1-codex", reasoning_effort="medium"),
            base,
        )
        assert out.provider == "codex-cli"
        assert out.model == "gpt-5.1-codex"
        assert out.reasoning_effort == "medium"
        # Homes travel with the base — both CLI backends stay usable.
        assert out.codex_home == "/data/codex"
        assert out.antigravity_home == "/data/gemini"


class TestSetModelsCompareRoundtrip:
    def test_save_and_reload(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """PUT with a compare list → GET resolves it back (full settings stack)."""
        from mira.dashboard.api import CompareModelEntry, ModelsUpdate
        from mira.dashboard.routers import admin

        monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
        db = AppDatabase(url="", admin_password="admin")
        monkeypatch.setattr(api, "_app_db", db)
        # No live catalog fetch in tests.
        monkeypatch.setattr("mira.dashboard.model_catalog.fetch_catalog", AsyncMock(return_value=None))

        request = MagicMock()
        request.state.user = SimpleNamespaceAdmin()
        body = ModelsUpdate(
            indexing_model="",
            review_model="",
            compare_models=[CompareModelEntry(provider="", model="gemini-3.8-flash-medium")],
        )
        admin.set_models(body, request)

        import asyncio

        got = asyncio.run(admin.get_models())
        assert [(e.provider, e.model) for e in got.compare_models] == [("", "gemini-3.8-flash-medium")]
        assert got.compare_source == "dashboard"
        assert got.compare_max == MAX_COMPARE_MODELS

    def test_rejects_bad_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from mira.dashboard.api import CompareModelEntry, ModelsUpdate
        from mira.dashboard.routers import admin

        monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
        db = AppDatabase(url="", admin_password="admin")
        monkeypatch.setattr(api, "_app_db", db)

        request = MagicMock()
        request.state.user = SimpleNamespaceAdmin()
        body = ModelsUpdate(
            indexing_model="",
            review_model="",
            compare_models=[
                CompareModelEntry(provider="", model="a"),
                CompareModelEntry(provider="", model="a"),
            ],
        )
        with pytest.raises(HTTPException) as exc:
            admin.set_models(body, request)
        assert exc.value.status_code == 400


def SimpleNamespaceAdmin():  # noqa: N802 — small local helper, not a class
    ns = MagicMock()
    ns.is_admin = True
    return ns


# ── Store attribution + migration ────────────────────────────────────────────


class TestReviewEventAttribution:
    def test_record_and_read_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
        store = IndexStore.open("acme", "web")
        try:
            store.record_review(
                pr_number=7,
                pr_title="T",
                pr_url="https://github.com/acme/web/pull/7",
                comments_posted=1,
                blockers=0,
                warnings=1,
                model="gemini-3.8-flash-low",
                kind="compare",
                head_sha="abc123",
            )
            events = store.list_review_events_for_pr(7)
            assert events[0].model == "gemini-3.8-flash-low"
            assert events[0].kind == "compare"
            assert events[0].head_sha == "abc123"
        finally:
            store.close()

    def test_failure_row_carries_attribution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
        store = IndexStore.open("acme", "web")
        try:
            store.record_review_failure(
                pr_number=7,
                pr_title="T",
                pr_url="https://github.com/acme/web/pull/7",
                error="boom",
                model="gpt-5.1-codex",
                kind="compare",
            )
            events = store.list_review_events_for_pr(7)
            assert events[0].status == "failed"
            assert events[0].model == "gpt-5.1-codex"
            assert events[0].kind == "compare"
        finally:
            store.close()

    def test_migration_adds_columns_to_legacy_db(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A DB created before the attribution columns must migrate on open."""
        monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
        # IndexStore layout for GitHub repos: {index_dir}/{owner}/{repo}.db
        db_path = tmp_path / "acme" / "web.db"
        db_path.parent.mkdir(parents=True)
        conn = sqlite3.connect(db_path)
        # The original review_events schema, before any post-schema columns
        # (author/avatar/reviewed_paths/status/error and the attribution trio).
        conn.execute(
            "CREATE TABLE review_events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, pr_number INTEGER, pr_title TEXT, "
            "pr_url TEXT, comments_posted INTEGER, blockers INTEGER, warnings INTEGER, "
            "suggestions INTEGER, files_reviewed INTEGER, lines_changed INTEGER, "
            "tokens_used INTEGER, duration_ms INTEGER, categories TEXT, created_at REAL)"
        )
        conn.execute(
            "INSERT INTO review_events (pr_number, pr_title, pr_url, comments_posted, created_at) "
            "VALUES (1, 'old', 'u', 0, 1.0)"
        )
        conn.commit()
        conn.close()

        store = IndexStore.open("acme", "web")
        try:
            events = store.list_review_events_for_pr(1)
            # Legacy rows read back with the migration defaults.
            assert events[0].kind == "review"
            assert events[0].model == ""
            assert events[0].head_sha == ""
        finally:
            store.close()


# ── Compare endpoint: round grouping + overlap ───────────────────────────────


@pytest.fixture
def compare_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    db = AppDatabase(url="", admin_password="admin")
    monkeypatch.setattr(api, "_app_db", db)
    return db


def _seed_compare_round(db: AppDatabase) -> None:
    db.register_repo("acme", "web")
    store = IndexStore.open("acme", "web")
    url = "https://github.com/acme/web/pull/5"
    main = store.record_review(
        pr_number=5, pr_title="Cmp", pr_url=url, comments_posted=2,
        blockers=1, warnings=1, model="main-model", kind="review", head_sha="sha-round-1",
        created_at=100.0,
    )
    store.add_review_comments(main.id, 5, url, [
        {"path": "src/a.py", "line": 1, "severity": "blocker", "category": "bug",
         "title": "Null deref in handler", "body": "crashes"},
        {"path": "src/b.py", "line": 9, "severity": "warning", "category": "bug",
         "title": "Unused variable x", "body": "dead"},
    ])
    shadow = store.record_review(
        pr_number=5, pr_title="Cmp", pr_url=url, comments_posted=2,
        blockers=1, warnings=1, model="shadow-model", kind="compare", head_sha="sha-round-1",
        created_at=110.0,
    )
    store.add_review_comments(shadow.id, 5, url, [
        # Same file+line+category → duplicate of the main finding.
        {"path": "src/a.py", "line": 1, "severity": "blocker", "category": "bug",
         "title": "Handler dereferences null", "body": "boom"},
        # Nothing like it in main → unique to the shadow model.
        {"path": "src/c.py", "line": 3, "severity": "warning", "category": "perf",
         "title": "N+1 query in loop", "body": "slow"},
    ])
    # An older round on a different SHA must stay its own group.
    store.record_review(
        pr_number=5, pr_title="Cmp", pr_url=url, comments_posted=0,
        blockers=0, warnings=0, model="main-model", kind="review", head_sha="sha-round-0",
        created_at=50.0,
    )
    store.close()


def test_compare_endpoint_groups_rounds_and_overlaps(compare_db: AppDatabase):
    _seed_compare_round(compare_db)
    detail = api.get_activity_compare("acme", "web", 5)

    assert len(detail.rounds) == 2
    latest = detail.rounds[0]
    assert latest.head_sha == "sha-round-1"
    kinds = {p.kind for p in latest.passes}
    assert kinds == {"review", "compare"}
    # One shadow pass → one overlap row: 1 shared, 1 main-only, 1 unique.
    assert len(latest.overlaps) == 1
    overlap = latest.overlaps[0]
    assert overlap.model == "shadow-model"
    assert overlap.shared == 1
    assert overlap.only_main == 1
    assert overlap.only_compare == 1
    # Older round grouped separately, no compare passes there.
    assert detail.rounds[1].head_sha == "sha-round-0"
    assert detail.rounds[1].overlaps == []


def test_compare_endpoint_404_for_unknown_pr(compare_db: AppDatabase):
    _seed_compare_round(compare_db)
    with pytest.raises(HTTPException) as exc:
        api.get_activity_compare("acme", "web", 999)
    assert exc.value.status_code == 404


# ── Engine shadow gates ──────────────────────────────────────────────────────

_SAMPLE_LLM_RESPONSE = json.dumps(
    {
        "summary": "ok",
        "comments": [
            {
                "path": "src/main.py",
                "line": 2,
                "severity": "warning",
                "category": "bug",
                "title": "Test finding",
                "body": "body",
                "confidence": 0.9,
            }
        ],
        "key_issues": [],
    }
)

_SAMPLE_DIFF = """diff --git a/src/main.py b/src/main.py
new file mode 100644
--- /dev/null
+++ b/src/main.py
@@ -0,0 +1,2 @@
+def hello():
+    return "world"
"""


class TestEngineShadowMode:
    @pytest.mark.asyncio
    async def test_shadow_posts_nothing_but_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))

        llm = MagicMock(spec=LLMProvider)
        llm.review = AsyncMock(return_value=_SAMPLE_LLM_RESPONSE)
        llm.complete = AsyncMock(return_value=_SAMPLE_LLM_RESPONSE)
        llm.walkthrough = AsyncMock(return_value=_SAMPLE_LLM_RESPONSE)
        llm.count_tokens = MagicMock(return_value=100)
        llm.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        provider = AsyncMock()
        provider.get_pr_info.return_value = PRInfo(
            title="T", description="D", base_branch="main", head_branch="f",
            url="https://github.com/acme/web/pull/9", number=9,
            owner="acme", repo="web", head_sha="sha99",
        )
        provider.get_pr_diff.return_value = _SAMPLE_DIFF
        provider.get_unresolved_bot_threads = AsyncMock(return_value=[])
        provider.get_all_bot_threads = AsyncMock(return_value=[])

        engine = ReviewEngine(
            config=MiraConfig(), llm=llm, provider=provider,
            bot_name="mira", shadow=True, model_label="shadow-x",
        )
        result = await engine.review_pr("https://github.com/acme/web/pull/9")

        # Nothing reaches the platform in shadow mode.
        provider.post_review.assert_not_called()
        provider.post_comment.assert_not_called()
        provider.find_bot_comment.assert_not_called()
        provider.update_comment.assert_not_called()
        provider.resolve_threads.assert_not_called()
        # No walkthrough generated (cost gate).
        llm.walkthrough.assert_not_called()
        assert result.walkthrough is None

        # The pass IS recorded with full attribution.
        store = IndexStore.open("acme", "web")
        try:
            events = store.list_review_events_for_pr(9)
            assert len(events) == 1
            assert events[0].kind == "compare"
            assert events[0].model == "shadow-x"
            assert events[0].head_sha == "sha99"
            assert events[0].status == "completed"
        finally:
            store.close()
