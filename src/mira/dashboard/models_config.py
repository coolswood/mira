"""Model resolution — reads from DB settings first, falls back to config.

Model lists, pricing, and capabilities all come from
``src/mira/llm/models.json`` via ``mira.llm.registry``. Add or remove a
model there; this file picks it up automatically.
"""

from __future__ import annotations

import logging

from mira.config import LLMConfig
from mira.llm import registry

logger = logging.getLogger(__name__)

MODEL_PRICING: dict[str, tuple[float, float]] = {
    model_id: registry.pricing(model_id) for model_id in registry.all_models()
}

# Thinking-mode options for the review model. "off" disables extended thinking
# (today's behavior); low/medium/high/xhigh map to the provider's unified
# ``reasoning.effort``; "max" is a top level remapped per provider (OpenRouter
# sends it as "xhigh"). Single source for the dashboard dropdown and validation.
THINKING_MODES: list[dict[str, str]] = [
    {"value": "off", "label": "Off"},
    {"value": "low", "label": "Low"},
    {"value": "medium", "label": "Medium"},
    {"value": "high", "label": "High"},
    {"value": "xhigh", "label": "XHigh"},
    # Top "max" level (sent as "xhigh" on OpenRouter, which rejects "max").
    # Sits above "xhigh". Not every provider supports it.
    {"value": "max", "label": "Max"},
]
THINKING_MODE_VALUES = {m["value"] for m in THINKING_MODES}

# Reasoning-effort levels each backend actually honors; the dashboard dropdown
# is filtered to this set and _BACKEND_EFFORT_HINTS explains the mapping.
# Levels outside a backend's set are either clamped by the provider
# (antigravity: xhigh/max → high) or unsupported (bedrock has no "xhigh";
# OpenRouter rejects "max" — the provider profile remaps it to "xhigh").
_BACKEND_EFFORT_LEVELS: dict[str, list[str]] = {
    "antigravity-cli": ["off", "low", "medium", "high"],
    "codex-cli": ["off", "low", "medium", "high"],
    "bedrock": ["off", "low", "medium", "high", "max"],
    "openrouter": ["off", "low", "medium", "high", "xhigh", "max"],
    "openai-compatible": ["off", "low", "medium", "high", "xhigh", "max"],
}

_BACKEND_EFFORT_HINTS: dict[str, str] = {
    "antigravity-cli": (
        "Sent to agy as --effort when the CLI picks the model (\"Inherit\"). "
        "Named models carry their effort in the id — gemini-*-high/-medium/-low, "
        "Claude (Thinking) — so choose the level by picking the model."
    ),
    "codex-cli": "Sent to Codex as model_reasoning_effort; xhigh/max are sent as high.",
    "bedrock": "Maps to Claude thinking budget tokens (low 2K, medium 8K, high 16K, max 32K).",
    "openrouter": "Sent as reasoning.effort; \"max\" is remapped to \"xhigh\" (OpenRouter rejects \"max\").",
    "openai-compatible": (
        "Sent as reasoning.effort; honored when the endpoint's model supports reasoning."
    ),
}


def thinking_modes_for_backend(backend: str) -> list[dict[str, str]]:
    """THINKING_MODES filtered to the levels the backend honors."""
    allowed = set(_BACKEND_EFFORT_LEVELS.get(backend, sorted(THINKING_MODE_VALUES)))
    return [m for m in THINKING_MODES if m["value"] in allowed]


def effort_hint(backend: str) -> str:
    """One-line explanation of how the effort level reaches the backend."""
    return _BACKEND_EFFORT_HINTS.get(backend, _BACKEND_EFFORT_HINTS["openai-compatible"])

# API-protocol options for the Models page. Single source for the dropdown
# and validation, mirroring THINKING_MODES.
API_STYLES: list[dict[str, str]] = [
    {"value": "chat", "label": "Chat Completions"},
    {"value": "responses", "label": "Responses API"},
]
API_STYLE_VALUES = {m["value"] for m in API_STYLES}


def resolve_api_style(config: LLMConfig, db_value: str | None = None) -> str:
    """Resolve the API protocol: DB → config.api_style → "chat"."""
    if db_value and db_value in API_STYLE_VALUES:
        return db_value
    return config.api_style if config.api_style in API_STYLE_VALUES else "chat"


def estimate_indexing_cost(file_count: int, model: str) -> dict:
    """Estimate cost of indexing N files with the given model.

    Based on actual indexer behavior:
    - Files batched 5-at-a-time
    - Each batch uses ~4K input tokens (prompt + 5 file contents ~500 lines avg)
    - Each batch outputs ~2K tokens (summaries + symbols JSON)
    - Plus a directory summarization pass at the end (~1 call per 10 files)
    """
    if file_count == 0:
        return {"estimated_usd": 0.0, "input_tokens": 0, "output_tokens": 0}

    input_price, output_price = MODEL_PRICING.get(model, (3.00, 15.00))

    # File summarization batches
    batches = (file_count + 4) // 5  # ceil div
    # Estimate: 800 tokens per file input, 400 tokens per file output
    input_tokens = file_count * 800 + batches * 500  # +prompt overhead per batch
    output_tokens = file_count * 400

    # Directory summarization pass
    dir_batches = max(1, file_count // 10)
    input_tokens += dir_batches * 1500
    output_tokens += dir_batches * 300

    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

    return {
        "estimated_usd": round(cost, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def get_indexing_model(config: LLMConfig, db_value: str | None = None) -> str:
    """Resolve the indexing model: DB → config.indexing_model → config.model."""
    if db_value:
        return db_value
    if config.indexing_model:
        return config.indexing_model
    return config.model


def get_review_model(config: LLMConfig, db_value: str | None = None) -> str:
    """Resolve the review model: DB → config.review_model → config.model."""
    if db_value:
        return db_value
    if config.review_model:
        return config.review_model
    return config.model


def get_security_model(
    config: LLMConfig,
    db_value: str | None = None,
    db_review_model: str | None = None,
) -> str:
    """Resolve the security-pass model: DB → config.security_model → review tier.

    The review-tier fallback includes the dashboard's review_model setting
    (``db_review_model``) — without it, an instance whose review model lives
    only in the DB silently falls all the way back to ``config.model``.

    Never falls back to ``indexing_model`` — the security sweep is the
    highest-stakes cheap pass, and silently downgrading it to the indexing
    tier trades security recall for indexing cost savings.
    """
    if db_value:
        return db_value
    if config.security_model:
        return config.security_model
    return get_review_model(config, db_review_model)


def get_review_thinking_mode(config: LLMConfig, db_value: str | None = None) -> str | None:
    """Resolve the review thinking mode: DB → config.review_reasoning_effort → None.

    A DB value of "off" or "" counts as unset and falls through to the
    mira.yaml-level setting — saving the models form always writes this key
    (default "off"), so a stored "off" must not permanently shadow a config
    override. "off" anywhere normalizes to None ("no reasoning").
    """
    resolved = db_value if (db_value and db_value != "off") else config.review_reasoning_effort
    if not resolved or resolved == "off":
        return None
    return resolved


def get_indexing_thinking_mode(config: LLMConfig, db_value: str | None = None) -> str | None:
    """Resolve the indexing thinking mode: DB → config.indexing_reasoning_effort → None.

    Indexing defaults to no reasoning (it's the highest-volume, lowest-stakes
    pass) — an effort only applies when set explicitly here or in mira.yaml.
    """
    resolved = db_value if (db_value and db_value != "off") else config.indexing_reasoning_effort
    if not resolved or resolved == "off":
        return None
    return resolved


def get_security_thinking_mode(
    config: LLMConfig,
    db_value: str | None = None,
    db_review_value: str | None = None,
) -> str | None:
    """Resolve the security-pass thinking mode.

    Chain: security's own DB setting → the review DB setting (the historical
    shared behavior — before per-task selectors existed, security simply
    followed the review mode) → config.security_reasoning_effort →
    config.review_reasoning_effort → None. "off" normalizes to None at every
    step, exactly like :func:`get_review_thinking_mode`.
    """
    resolved = db_value if (db_value and db_value != "off") else None
    if resolved is None:
        resolved = db_review_value if (db_review_value and db_review_value != "off") else None
    if resolved is None:
        resolved = config.security_reasoning_effort
    if resolved is None:
        resolved = config.review_reasoning_effort
    if not resolved or resolved == "off":
        return None
    return resolved


def llm_config_for(purpose: str, base: LLMConfig) -> LLMConfig:
    """Return an LLMConfig with the appropriate model set for the given purpose.

    Reads the DB setting first (via _app_db), falls back to config fields.
    Logs the effective model and where it came from, so a dashboard override
    shadowing mira.yaml is visible instead of silent (issue #124).
    """
    db_model: str | None = None
    db_thinking: str | None = None
    db_review_thinking: str | None = None
    db_review: str | None = None
    db_style: str | None = None
    try:
        from mira.dashboard.api import _app_db

        if _app_db is not None:
            if purpose == "indexing":
                db_model = _app_db.get_setting("indexing_model")
                db_thinking = _app_db.get_setting("indexing_thinking_mode")
            elif purpose == "review":
                db_model = _app_db.get_setting("review_model")
                db_thinking = _app_db.get_setting("review_thinking_mode")
            elif purpose == "security":
                db_model = _app_db.get_setting("security_model")
                db_thinking = _app_db.get_setting("security_thinking_mode")
                # The security effort falls back to the review setting when
                # its own is unset — the historical shared behavior.
                db_review_thinking = _app_db.get_setting("review_thinking_mode")
                db_review = _app_db.get_setting("review_model")
            db_style = _app_db.get_setting("api_style")
    except Exception:
        pass  # DB not available — resolve from config fields alone

    resolved_style = resolve_api_style(base, db_style)
    if purpose == "indexing":
        resolved = get_indexing_model(base, db_model)
        config_model = base.indexing_model
        thinking_mode = get_indexing_thinking_mode(base, db_thinking)
    elif purpose == "security":
        resolved = get_security_model(base, db_model, db_review)
        config_model = base.security_model or base.review_model
        thinking_mode = get_security_thinking_mode(base, db_thinking, db_review_thinking)
    elif purpose == "review":
        resolved = get_review_model(base, db_model)
        config_model = base.review_model
        thinking_mode = get_review_thinking_mode(base, db_thinking)
    else:
        return base.model_copy(update={"reasoning_effort": None, "api_style": resolved_style})

    source = "dashboard setting" if db_model else ("mira.yaml" if config_model else "default")
    logger.info("%s model: %s (source: %s)", purpose.capitalize(), resolved, source)
    return base.model_copy(
        update={"model": resolved, "reasoning_effort": thinking_mode, "api_style": resolved_style}
    )
