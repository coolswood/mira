"""Tests for the backend-aware dynamic model catalog."""

from __future__ import annotations

import asyncio

import pytest

from mira.config import LLMConfig
from mira.dashboard import model_catalog
from mira.dashboard.model_catalog import active_backend, build_options, fetch_catalog


@pytest.fixture(autouse=True)
def _clear_cache():
    model_catalog._cache.clear()
    yield
    model_catalog._cache.clear()


class TestActiveBackend:
    def test_default_is_openrouter(self):
        assert active_backend(LLMConfig()) == "openrouter"

    def test_bedrock_provider(self):
        assert active_backend(LLMConfig(provider="bedrock")) == "bedrock"

    def test_codex_cli_provider(self):
        assert active_backend(LLMConfig(provider="codex-cli")) == "codex-cli"

    def test_antigravity_cli_provider(self):
        assert active_backend(LLMConfig(provider="antigravity-cli")) == "antigravity-cli"

    def test_generic_endpoint(self):
        assert (
            active_backend(LLMConfig(base_url="http://localhost:11434/v1")) == "openai-compatible"
        )


class TestBuildOptions:
    def test_registry_filtered_by_backend(self):
        openrouter = [m["value"] for m in build_options("openrouter", None, "review")]
        bedrock = [m["value"] for m in build_options("bedrock", None, "review")]
        assert "anthropic/claude-sonnet-4-6" in openrouter
        assert "us.anthropic.claude-sonnet-4-6-v1:0" not in openrouter
        assert "us.anthropic.claude-sonnet-4-6-v1:0" in bedrock
        assert "anthropic/claude-sonnet-4-6" not in bedrock

    def test_codex_backend_only_offers_codex_models(self):
        values = [m["value"] for m in build_options("codex-cli", None, "review")]
        assert values == ["gpt-5.1-codex", "codex-default", "gpt-5.1-codex-mini"]

    def test_antigravity_backend_only_offers_antigravity_models(self):
        values = [m["value"] for m in build_options("antigravity-cli", None, "review")]
        # The curated fallback for when the live `agy models` fetch fails;
        # recommended (the CLI default) sorts first.
        assert values == [
            "antigravity-default",
            "gemini-3.1-pro-high",
            "gemini-3.8-flash-high",
            "gemini-3.8-flash-medium",
        ]

    def test_antigravity_live_catalog_merges_with_registry(self):
        dynamic = [
            {"value": "gemini-3.8-flash-low", "label": "Gemini 3.8 Flash (Low)"},
            {"value": "gpt-oss-120b-medium", "label": "GPT-OSS 120B (Medium)"},
        ]
        options = build_options("antigravity-cli", dynamic, "indexing")
        values = [m["value"] for m in options]
        # Registry entry wins over the live row (keeps the recommended badge);
        # live-only models are appended.
        assert "gemini-3.8-flash-low" in values
        assert "gpt-oss-120b-medium" in values
        assert values[0] == "antigravity-default"
        assert options[0]["recommended"] is True

    def test_openrouter_does_not_offer_codex_models(self):
        values = [m["value"] for m in build_options("openrouter", None, "review")]
        assert "codex-default" not in values
        assert "antigravity-default" not in values

    def test_dynamic_merged_and_deduped_against_registry(self):
        dynamic = [
            {"value": "anthropic/claude-sonnet-4.6", "label": "Anthropic: Claude Sonnet 4.6"},
            {"value": "mistralai/mistral-large-3", "label": "Mistral Large 3"},
        ]
        options = build_options("openrouter", dynamic, "review")
        values = [m["value"] for m in options]
        # Dot-form alias of a registry id is dropped; genuinely new model kept.
        assert "anthropic/claude-sonnet-4.6" not in values
        assert "anthropic/claude-sonnet-4-6" in values
        assert "mistralai/mistral-large-3" in values

    def test_generic_endpoint_uses_dynamic_only(self):
        dynamic = [{"value": "llama-3.3-70b", "label": "llama-3.3-70b"}]
        values = [m["value"] for m in build_options("openai-compatible", dynamic, "review")]
        assert values == ["llama-3.3-70b"]

    def test_generic_endpoint_falls_back_to_registry(self):
        values = [m["value"] for m in build_options("openai-compatible", None, "review")]
        assert "anthropic/claude-sonnet-4-6" in values

    def test_recommended_sort_first(self):
        options = build_options("openrouter", None, "indexing")
        assert options[0]["recommended"] is True


class TestFetchCatalog:
    @pytest.mark.asyncio
    async def test_failure_returns_none_and_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        calls = 0

        async def boom(config, tools_only):
            nonlocal calls
            calls += 1
            raise RuntimeError("no network")

        monkeypatch.setattr(model_catalog, "_fetch_openai_style", boom)
        assert await fetch_catalog(LLMConfig()) is None
        # A dead endpoint must not re-block every settings-page load.
        assert await fetch_catalog(LLMConfig()) is None
        assert calls == 1

    @pytest.mark.asyncio
    async def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        calls = 0

        async def fake(config, tools_only):
            nonlocal calls
            calls += 1
            return [{"value": "m", "label": "m"}]

        monkeypatch.setattr(model_catalog, "_fetch_openai_style", fake)
        assert await fetch_catalog(LLMConfig()) == [{"value": "m", "label": "m"}]
        assert await fetch_catalog(LLMConfig()) == [{"value": "m", "label": "m"}]
        assert calls == 1

    @pytest.mark.asyncio
    async def test_concurrent_cold_fetches_coalesce(self, monkeypatch: pytest.MonkeyPatch):
        calls = 0

        async def slow(config, tools_only):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return [{"value": "m", "label": "m"}]

        monkeypatch.setattr(model_catalog, "_fetch_openai_style", slow)
        results = await asyncio.gather(*(fetch_catalog(LLMConfig()) for _ in range(5)))
        assert all(r == [{"value": "m", "label": "m"}] for r in results)
        assert calls == 1

    def test_bedrock_cache_key_includes_profile(self):
        # Switching aws_profile must not serve the previous account's catalog.
        a = LLMConfig(provider="bedrock", aws_profile="account-a")
        b = LLMConfig(provider="bedrock", aws_profile="account-b")
        assert a.region == b.region
        # Keys derived the same way fetch_catalog does.
        key_a = f"bedrock:{a.region}:{a.aws_profile or ''}"
        key_b = f"bedrock:{b.region}:{b.aws_profile or ''}"
        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_antigravity_live_catalog_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        calls = 0

        def fake_list_models(config, timeout=30.0):
            nonlocal calls
            calls += 1
            return [{"value": "gemini-3.8-flash-low", "label": "Gemini 3.8 Flash (Low)"}]

        monkeypatch.setattr("mira.llm.antigravity_cli.list_models", fake_list_models)
        cfg = LLMConfig(provider="antigravity-cli")
        expected = [{"value": "gemini-3.8-flash-low", "label": "Gemini 3.8 Flash (Low)"}]
        assert await fetch_catalog(cfg) == expected
        assert await fetch_catalog(cfg) == expected
        assert calls == 1

    @pytest.mark.asyncio
    async def test_antigravity_fetch_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        from mira.exceptions import LLMError

        def boom(config, timeout=30.0):
            raise LLMError("antigravity_models_failed", detail="sign in")

        monkeypatch.setattr("mira.llm.antigravity_cli.list_models", boom)
        assert await fetch_catalog(LLMConfig(provider="antigravity-cli")) is None

    @pytest.mark.asyncio
    async def test_codex_has_no_live_catalog(self):
        # Codex has no model-list subcommand; the registry serves its dropdown.
        assert await fetch_catalog(LLMConfig(provider="codex-cli")) is None
