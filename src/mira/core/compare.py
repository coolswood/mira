"""Parallel-model comparison: shadow reviews that post nothing.

For every configured comparison model (dashboard ``compare_models`` list or
``llm.compare_models`` in mira.yaml) a full review pipeline runs alongside the
main one on the same PR. The shadow pass reads the same PR state (round
detection, incremental diff) so conditions match the main run, but writes
nothing to the platform: no inline comments, no walkthrough, no thread
resolution, no rule learning, no review-progress/SHA bookkeeping. Its only
output is a ``review_events`` row with ``kind='compare'`` plus the comments it
would have posted — the dashboard's compare view is built from those rows.

Failures are per-model and never propagate: a shadow crash must not take down
the real review running next to it.
"""

from __future__ import annotations

import asyncio
import logging

from mira.config import CompareModelConfig, MiraConfig
from mira.core.engine import ReviewEngine
from mira.dashboard.models_config import get_compare_models, llm_config_for_compare
from mira.llm import create_llm

logger = logging.getLogger(__name__)


def _progress_begin(owner: str, repo: str, number: int, label: str, pr_title: str, pr_url: str) -> str | None:
    """Register the shadow job in the live progress tracker (best-effort)."""
    try:
        from mira.core.progress import COMPARE, compare_key, tracker

        key = compare_key(owner, repo, number, label)
        tracker.begin(key, COMPARE, f"{owner}/{repo}", pr_number=number, pr_title=pr_title, pr_url=pr_url)
        return key
    except Exception as exc:
        logger.debug("Compare progress tracking unavailable: %s", exc)
        return None


def _progress_finish(key: str | None, error: str = "") -> None:
    if not key:
        return
    try:
        from mira.core.progress import tracker

        if error:
            tracker.fail(key, error[:500])
        else:
            tracker.finish(key)
    except Exception:
        pass


def _record_compare_failure(
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    pr_title: str,
    platform: str,
    entry: CompareModelConfig,
    exc: BaseException,
) -> None:
    from mira.platforms.handlers import _safe_error_summary

    try:
        from mira.index.store import IndexStore

        store = IndexStore.open(owner, repo, platform=platform)
        try:
            store.record_review_failure(
                pr_number=number,
                pr_title=pr_title,
                pr_url=pr_url,
                error=_safe_error_summary(exc),
                model=entry.model,
                kind="compare",
            )
        finally:
            store.close()
    except Exception as record_exc:
        logger.warning(
            "Failed to record compare-failure event for %s #%d (%s): %s",
            f"{owner}/{repo}",
            number,
            entry.model,
            record_exc,
        )


async def _run_one(
    entry: CompareModelConfig,
    config: MiraConfig,
    provider: object,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    pr_title: str,
    bot_name: str,
    platform: str,
) -> None:
    label = entry.model
    progress_key = _progress_begin(owner, repo, number, label, pr_title, pr_url)
    logger.info("Compare shadow review of %s with model %s", pr_url, label)
    try:
        from mira.dashboard.models_config import llm_config_for

        llm = create_llm(llm_config_for_compare(entry, config.llm))
        indexing_llm = create_llm(llm_config_for("indexing", config.llm))
        for llm_tier in (llm, indexing_llm):
            llm_tier.progress_key = progress_key  # type: ignore[attr-defined]
        engine = ReviewEngine(
            config=config,
            llm=llm,
            provider=provider,
            bot_name=bot_name,
            indexing_llm=indexing_llm,
            shadow=True,
            model_label=label,
        )
        engine._progress_suffix = f":cmp:{label}"  # type: ignore[attr-defined]
        await engine.review_pr(pr_url)
        _progress_finish(progress_key)
        logger.info("Compare shadow review of %s with %s complete", pr_url, label)
    except Exception as exc:
        _progress_finish(progress_key, str(exc))
        _record_compare_failure(owner, repo, number, pr_url, pr_title, platform, entry, exc)
        logger.warning("Compare shadow review of %s with %s failed: %s", pr_url, label, exc)


async def run_compare_reviews(
    config: MiraConfig,
    provider: object,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    pr_title: str = "",
    bot_name: str = "miracodeai",
    platform: str = "github",
) -> None:
    """Run one shadow review per configured comparison model, in parallel.

    Never raises: each shadow is independent and a failing model is recorded
    as its own failed pass. Returns immediately when no comparison models are
    configured.
    """
    entries = get_compare_models(config.llm)
    if not entries:
        return
    logger.info("Running %d compare shadow review(s) for %s", len(entries), pr_url)
    await asyncio.gather(
        *[
            _run_one(
                entry,
                config,
                provider,
                owner,
                repo,
                number,
                pr_url,
                pr_title,
                bot_name,
                platform,
            )
            for entry in entries
        ],
        return_exceptions=True,
    )
