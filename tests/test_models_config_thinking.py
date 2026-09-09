"""Tests for review thinking-mode resolution and the models endpoint.

Covers:
- Per-purpose effort resolution (`get_*_thinking_mode`) precedence and
  off/empty normalization.
- `llm_config_for` setting `reasoning_effort` per purpose, from the DB.
- `set_models` validating the thinking modes and persisting them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from mira.config import LLMConfig
from mira.dashboard.api import ModelsUpdate
from mira.dashboard.db import AppDatabase
from mira.dashboard.models_config import (
    get_indexing_thinking_mode,
    get_review_thinking_mode,
    get_security_thinking_mode,
    llm_config_for,
)
from mira.dashboard.routers.admin import set_models


def _admin_req():
    from types import SimpleNamespace

    user = SimpleNamespace(is_admin=True)
    return SimpleNamespace(state=SimpleNamespace(user=user))


@pytest.fixture
def in_memory_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    """Fresh per-test SQLite DB swapped in for the module-level `_app_db`."""
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    db = AppDatabase(url="", admin_password="admin")
    monkeypatch.setattr("mira.dashboard.api._app_db", db)
    return db


class TestGetReviewThinkingMode:
    def test_db_value_wins(self):
        cfg = LLMConfig(review_reasoning_effort="low")
        assert get_review_thinking_mode(cfg, "high") == "high"

    def test_falls_back_to_config(self):
        cfg = LLMConfig(review_reasoning_effort="medium")
        assert get_review_thinking_mode(cfg, None) == "medium"

    def test_default_is_none(self):
        assert get_review_thinking_mode(LLMConfig(), None) is None

    @pytest.mark.parametrize("value", ["off", ""])
    def test_off_and_empty_normalize_to_none(self, value: str):
        assert get_review_thinking_mode(LLMConfig(), value) is None

    @pytest.mark.parametrize("db_value", ["off", "", None])
    def test_off_db_value_does_not_shadow_config(self, db_value: str | None):
        # Saving the models form always writes "off" by default; that must not
        # permanently disable a mira.yaml-level reasoning effort.
        cfg = LLMConfig(review_reasoning_effort="high")
        assert get_review_thinking_mode(cfg, db_value) == "high"


class TestLLMConfigFor:
    def test_review_picks_up_thinking_mode(self, in_memory_db: AppDatabase):
        in_memory_db.set_setting("review_thinking_mode", "high")
        resolved = llm_config_for("review", LLMConfig())
        assert resolved.reasoning_effort == "high"

    def test_indexing_never_sets_thinking_mode(self, in_memory_db: AppDatabase):
        in_memory_db.set_setting("review_thinking_mode", "high")
        resolved = llm_config_for("indexing", LLMConfig())
        assert resolved.reasoning_effort is None

    def test_review_default_off_is_none(self, in_memory_db: AppDatabase):
        resolved = llm_config_for("review", LLMConfig())
        assert resolved.reasoning_effort is None


class TestSetModelsThinkingValidation:
    def test_rejects_invalid_thinking_mode(self, in_memory_db: AppDatabase):
        body = ModelsUpdate(
            indexing_model="anthropic/claude-haiku-4-5",
            review_model="anthropic/claude-sonnet-4-6",
            review_thinking_mode="ultra",
        )
        with pytest.raises(HTTPException) as exc:
            set_models(body, _admin_req())
        assert exc.value.status_code == 400

    def test_persists_valid_thinking_mode(self, in_memory_db: AppDatabase):
        body = ModelsUpdate(
            indexing_model="anthropic/claude-haiku-4-5",
            review_model="anthropic/claude-sonnet-4-6",
            review_thinking_mode="medium",
        )
        assert set_models(body, _admin_req()) == {"ok": True}
        assert in_memory_db.get_setting("review_thinking_mode") == "medium"

    def test_persists_xhigh_thinking_mode(self, in_memory_db: AppDatabase):
        body = ModelsUpdate(
            indexing_model="anthropic/claude-haiku-4-5",
            review_model="anthropic/claude-sonnet-4-6",
            review_thinking_mode="xhigh",
        )
        assert set_models(body, _admin_req()) == {"ok": True}
        assert in_memory_db.get_setting("review_thinking_mode") == "xhigh"

    def test_off_clears_setting_so_config_can_win(self, in_memory_db: AppDatabase):
        # "off" must not be persisted as a literal — it'd shadow a mira.yaml
        # override. It's stored as "" (the column is NOT NULL) and reads as unset.
        body = ModelsUpdate(
            indexing_model="anthropic/claude-haiku-4-5",
            review_model="anthropic/claude-sonnet-4-6",
            review_thinking_mode="off",
        )
        assert set_models(body, _admin_req()) == {"ok": True}
        assert in_memory_db.get_setting("review_thinking_mode") == ""
        cfg = LLMConfig(review_reasoning_effort="high")
        assert (
            get_review_thinking_mode(cfg, in_memory_db.get_setting("review_thinking_mode"))
            == "high"
        )


class TestGetIndexingThinkingMode:
    def test_db_value_wins(self):
        cfg = LLMConfig(indexing_reasoning_effort="low")
        assert get_indexing_thinking_mode(cfg, "medium") == "medium"

    def test_falls_back_to_config(self):
        cfg = LLMConfig(indexing_reasoning_effort="low")
        assert get_indexing_thinking_mode(cfg, None) == "low"

    def test_default_is_none(self):
        # Indexing stays reasoning-free unless explicitly opted in.
        assert get_indexing_thinking_mode(LLMConfig(), None) is None

    @pytest.mark.parametrize("db_value", ["off", "", None])
    def test_off_db_value_does_not_shadow_config(self, db_value: str | None):
        cfg = LLMConfig(indexing_reasoning_effort="medium")
        assert get_indexing_thinking_mode(cfg, db_value) == "medium"


class TestGetSecurityThinkingMode:
    def test_own_db_value_wins(self):
        cfg = LLMConfig(review_reasoning_effort="low")
        assert get_security_thinking_mode(cfg, "high", "medium") == "high"

    def test_falls_back_to_review_db_setting(self):
        # Historical behavior: security followed the review mode. Instances
        # that never touched the new selector keep working exactly as before.
        assert get_security_thinking_mode(LLMConfig(), None, "xhigh") == "xhigh"

    def test_falls_back_to_security_config_then_review_config(self):
        assert get_security_thinking_mode(
            LLMConfig(security_reasoning_effort="medium"), None, None
        ) == "medium"
        assert get_security_thinking_mode(
            LLMConfig(review_reasoning_effort="high"), None, None
        ) == "high"

    @pytest.mark.parametrize(
        "own,review",
        [("off", "off"), ("", "off"), (None, None), ("off", "")],
    )
    def test_off_normalizes_to_none(self, own, review):
        assert get_security_thinking_mode(LLMConfig(), own, review) is None

    def test_explicit_off_beats_active_review_setting(self):
        # The dashboard's security selector must be able to turn reasoning
        # off while the review effort is active — a literal stored "off" is
        # an explicit disable, not an unset value.
        assert get_security_thinking_mode(LLMConfig(), "off", "high") is None

    def test_legacy_empty_still_inherits_review_setting(self):
        # Rows written by older builds ("off" cleared to "") keep the
        # historical inherit-from-review behavior.
        assert get_security_thinking_mode(LLMConfig(), "", "high") == "high"

    def test_review_db_beats_security_yaml(self):
        # Dashboard settings (DB) sit above mira.yaml across the board.
        cfg = LLMConfig(security_reasoning_effort="low")
        assert get_security_thinking_mode(cfg, None, "medium") == "medium"


class TestLLMConfigForPerPurposeEffort:
    def test_indexing_picks_up_its_own_mode(self, in_memory_db: AppDatabase):
        in_memory_db.set_setting("indexing_thinking_mode", "low")
        resolved = llm_config_for("indexing", LLMConfig())
        assert resolved.reasoning_effort == "low"

    def test_indexing_does_not_inherit_review_mode(self, in_memory_db: AppDatabase):
        in_memory_db.set_setting("review_thinking_mode", "high")
        resolved = llm_config_for("indexing", LLMConfig())
        assert resolved.reasoning_effort is None

    def test_security_uses_own_mode_before_review_mode(self, in_memory_db: AppDatabase):
        in_memory_db.set_setting("review_thinking_mode", "high")
        in_memory_db.set_setting("security_thinking_mode", "medium")
        resolved = llm_config_for("security", LLMConfig())
        assert resolved.reasoning_effort == "medium"

    def test_security_follows_review_mode_when_own_unset(self, in_memory_db: AppDatabase):
        in_memory_db.set_setting("review_thinking_mode", "high")
        resolved = llm_config_for("security", LLMConfig())
        assert resolved.reasoning_effort == "high"


class TestSetModelsPerPurposeEffort:
    def test_persists_all_three_efforts(self, in_memory_db: AppDatabase):
        body = ModelsUpdate(
            indexing_model="m1",
            review_model="m2",
            security_model="m3",
            indexing_thinking_mode="low",
            review_thinking_mode="high",
            security_thinking_mode="medium",
        )
        assert set_models(body, _admin_req()) == {"ok": True}
        assert in_memory_db.get_setting("indexing_thinking_mode") == "low"
        assert in_memory_db.get_setting("review_thinking_mode") == "high"
        assert in_memory_db.get_setting("security_thinking_mode") == "medium"

    def test_off_clears_review_and_indexing_but_persists_for_security(self, in_memory_db: AppDatabase):
        in_memory_db.set_setting("indexing_thinking_mode", "low")
        in_memory_db.set_setting("security_thinking_mode", "high")
        body = ModelsUpdate(
            indexing_model="m1",
            review_model="m2",
            indexing_thinking_mode="off",
            review_thinking_mode="off",
            security_thinking_mode="off",
        )
        assert set_models(body, _admin_req()) == {"ok": True}
        assert in_memory_db.get_setting("indexing_thinking_mode") == ""
        assert in_memory_db.get_setting("review_thinking_mode") == ""
        # Security keeps the literal: its resolution falls through to the
        # review *dashboard* setting, so "explicitly off" must persist to be
        # distinguishable from "unset" ("" rows inherit review).
        assert in_memory_db.get_setting("security_thinking_mode") == "off"

    def test_security_off_survives_round_trip_with_active_review(self, in_memory_db: AppDatabase):
        # The dashboard flow that used to lie: review effort on, security
        # explicitly off — after save+resolve, security must stay off.
        body = ModelsUpdate(
            indexing_model="m1",
            review_model="m2",
            review_thinking_mode="high",
            security_thinking_mode="off",
        )
        assert set_models(body, _admin_req()) == {"ok": True}
        resolved = llm_config_for("security", LLMConfig())
        assert resolved.reasoning_effort is None
        # ...while the review pass keeps its own effort.
        assert llm_config_for("review", LLMConfig()).reasoning_effort == "high"

    @pytest.mark.parametrize("field", ["indexing_thinking_mode", "security_thinking_mode"])
    def test_rejects_invalid_per_purpose_mode(self, in_memory_db: AppDatabase, field: str):
        body = ModelsUpdate(
            indexing_model="m1",
            review_model="m2",
            **{field: "ultra"},
        )
        with pytest.raises(HTTPException) as exc:
            set_models(body, _admin_req())
        assert exc.value.status_code == 400
