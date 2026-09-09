"""The l10n review pass: deterministic checks + LLM translation QA.

Runs alongside the main review when the PR touches localization files
(ARB today). Deterministic checks (key parity, placeholders, ICU structure)
cost no LLM quota; the LLM pass does meaning/grammar/formality QA on the
*changed keys only*, batched, so a 20-locale PR costs a couple of calls.

Findings anchor to the changed key's line inside a diffed file — including
when the actual problem sits in a locale file the diff-size budget skipped
(the pass fetches full localization files from the PR head itself).
"""

import asyncio
import logging
from dataclasses import dataclass
from fnmatch import fnmatch

from mira.config import L10nConfig
from mira.l10n.arb import ArbFile, changed_keys, hunk_plus_lines, parse_arb
from mira.l10n.checks import (
    L10nFinding,
    check_against_template,
    check_family_parity,
    check_locale_tag,
    check_untranslated,
    check_value_structure,
)
from mira.llm.response_parser import (
    ResponseParseError,
    convert_to_review_comments,
    parse_llm_response,
)
from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL
from mira.models import PRInfo, ReviewComment, Severity

logger = logging.getLogger(__name__)

_FETCH_CONCURRENCY = 5


@dataclass
class _Row:
    """One changed key with all its translations worth LLM checking.

    A single row carries every target locale for the key, so one batched
    call reviews all languages of a changed string at once.
    """

    key: str
    anchor_path: str
    anchor_line: int
    source_locale: str
    source: str
    description: str
    targets: list[tuple[str, str]]  # (locale, translation)


def is_l10n_path(path: str, patterns: list[str]) -> bool:
    return any(fnmatch(path, pat) for pat in patterns)


def _severity(value: str) -> Severity:
    try:
        return Severity.from_str(value)
    except Exception:
        return Severity.SUGGESTION


def _findings_to_comments(findings: list[L10nFinding]) -> list[ReviewComment]:
    return [
        ReviewComment(
            path=f.path,
            line=f.line,
            end_line=None,
            severity=_severity(f.severity),
            category="l10n",
            title=f.title[:80],
            body=f.body,
            confidence=f.confidence,
            suggestion=f.suggestion,
            source_pass="l10n",
        )
        for f in findings
    ]


async def _fetch_all(
    provider: object,
    pr_info: PRInfo,
    ref: str,
    paths: list[str],
) -> dict[str, str]:
    sem = asyncio.Semaphore(_FETCH_CONCURRENCY)

    async def _one(path: str) -> tuple[str, str]:
        async with sem:
            try:
                content = await provider.get_file_content(pr_info, path, ref)  # type: ignore[attr-defined]
                return (path, content) if content else (path, "")
            except Exception as exc:
                logger.debug("l10n pass: fetch failed for %s: %s", path, exc)
                return (path, "")

    results = await asyncio.gather(*[_one(p) for p in paths])
    return {path: content for path, content in results if content}


def _parse_l10n_yaml(content: str) -> tuple[str, str]:
    """Tiny parser for Flutter ``l10n.yaml`` → (arb_dir, template_filename).

    Deliberately not a YAML dependency: the file is flat ``key: value`` lines.
    """
    arb_dir = ""
    template = ""
    for line in content.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "arb-dir":
            arb_dir = value.strip("/")
        elif key == "template-arb-file":
            template = value
    return arb_dir, template


def _parent_dir(path: str) -> str:
    """Directory part of a repo path; root-level files share ``""``.

    ``path.rsplit("/", 1)[0]`` alone returns the filename itself for
    slash-less paths, which makes every root-level localization file its
    own "directory" and silently breaks family grouping between sibling
    locales living in the repository root.
    """
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _discover_family(
    tree_paths: list[str],
    patterns: list[str],
    diffed_paths: set[str],
) -> tuple[list[str], str | None]:
    """All localization files in the repo plus a template-path guess.

    Template preference: diffed file with the most sibling locales; refined
    later by ``l10n.yaml`` when it is fetchable.
    """
    all_arb = [p for p in tree_paths if is_l10n_path(p, patterns)] if tree_paths else []
    if not all_arb:
        all_arb = sorted(diffed_paths)
    dirs = {_parent_dir(p) for p in diffed_paths}
    family = [p for p in all_arb if _parent_dir(p) in dirs]
    template = None
    if family:
        by_dir: dict[str, list[str]] = {}
        for p in family:
            by_dir.setdefault(_parent_dir(p), []).append(p)
        # The directory holding the most diffed files wins (arb-dir, not src modules).
        main_dir = max(by_dir, key=lambda d: sum(1 for p in by_dir[d] if p in diffed_paths))
        candidates = [p for p in by_dir[main_dir] if p in diffed_paths]
        template = candidates[0] if candidates else None
    return family, template


async def l10n_review_pass(
    llm: object,
    l10n_files: list,
    *,
    provider: object | None,
    pr_info: PRInfo | None,
    repo_tree: list[str] | None = None,
    pr_title: str = "",
    indexing_llm: object | None = None,
    config: L10nConfig | None = None,
) -> list[ReviewComment]:
    """Review changed localization keys. Never raises into the main review."""
    cfg = config or L10nConfig()
    if not l10n_files:
        return []

    diffed_paths = {f.path for f in l10n_files}
    keys_per_file = {
        f.path: changed_keys(hunk_plus_lines([h.content for h in f.hunks])) for f in l10n_files
    }
    if not any(keys_per_file.values()):
        return []

    # ── Fetch full localization files from the PR head ──────────────────
    # l10n.yaml is the authoritative template source, so it is fetched first
    # and its pick wins. The diff-shape heuristic (_discover_family) is only
    # a fallback for repos without the file: a wrong guess floods false
    # "missing from template" findings (cognitive_psy PR #684 — split-fragment
    # src/ru/*.arb sources out-populated the real arb-dir in the diff, and the
    # heuristic crowned an unrelated fragment as the template).
    yaml_candidate: str | None = None
    template_path: str | None = None
    contents: dict[str, str] = {}
    if provider is not None and pr_info is not None:
        family_paths, guessed = _discover_family(repo_tree or [], cfg.patterns, diffed_paths)
        ref = pr_info.head_branch
        l10n_yaml = await _fetch_all(provider, pr_info, ref, ["l10n.yaml"])
        raw = l10n_yaml.get("l10n.yaml", "")
        if raw:
            arb_dir, template_name = _parse_l10n_yaml(raw)
            yaml_candidate = f"{arb_dir}/{template_name}" if arb_dir else template_name
        template_path = yaml_candidate or guessed
        fetch = sorted(set(family_paths) | diffed_paths | ({yaml_candidate} if yaml_candidate else set()))
        contents = await _fetch_all(provider, pr_info, ref, fetch)

    parsed: dict[str, ArbFile] = {
        path: parse_arb(path, content) for path, content in contents.items()
    }
    fetched = set(contents)

    # The diff-line regex also matches string lines nested inside @key
    # metadata ("description": ..., placeholder type tables). Keep only keys
    # that actually exist at the top level of the parsed file.
    for path, keys in keys_per_file.items():
        arb = parsed.get(path)
        if arb is not None and keys:
            keys_per_file[path] = keys & set(arb.entries)

    template: ArbFile | None = None
    template_from_yaml = False
    if yaml_candidate:
        cand = parsed.get(yaml_candidate)
        if cand is not None and cand.parsed_ok:
            template = cand
            template_from_yaml = True
    if template is None and template_path and template_path != yaml_candidate:
        cand = parsed.get(template_path)
        if cand is not None and cand.parsed_ok:
            template = cand
    family: dict[str, ArbFile] = {}
    if template is not None:
        tdir = _parent_dir(template.path)
        family = {p: a for p, a in parsed.items() if a.parsed_ok and _parent_dir(p) == tdir}
        # Heuristic pick only: a diffed file with more entries than the guess
        # means the guess was wrong — trust the data instead. An explicit
        # l10n.yaml pick is authoritative and never overruled.
        if not template_from_yaml:
            biggest = max(family.values(), key=lambda a: len(a.entries), default=None)
            if biggest is not None and len(biggest.entries) > len(template.entries) * 2:
                template = biggest
                family = {
                    p: a
                    for p, a in parsed.items()
                    if a.parsed_ok and _parent_dir(p) == _parent_dir(template.path)
                }

    # ── Deterministic checks (no quota) ─────────────────────────────────
    findings: list[L10nFinding] = []
    for f in l10n_files:
        keys = keys_per_file[f.path]
        if not keys or f.path not in fetched:
            # Without the full file there is nothing honest to check — a
            # failed fetch must not look like "invalid JSON".
            continue
        arb = parsed[f.path]
        is_template = template is not None and arb.path == template.path
        findings.extend(check_value_structure(arb, keys, is_template=is_template))
        if template is not None and template.parsed_ok:
            # Locale-tag check applies only to the arb-dir family: module
            # source files (e.g. src/ru/home_bot.arb) carry no locale tag.
            if arb.path in family:
                findings.extend(check_locale_tag(arb, keys))
            findings.extend(check_against_template(arb, keys, template))
            if is_template:
                findings.extend(check_family_parity(template, keys, family))
                findings.extend(check_untranslated(template, keys, family))

    comments: list[ReviewComment] = _findings_to_comments(findings)

    # ── LLM translation QA on changed keys ─────────────────────────────
    if cfg.semantic_check and template is not None and template.parsed_ok:
        rows = _build_rows(l10n_files, keys_per_file, parsed, template, family, cfg)
        if rows:
            qa_llm = indexing_llm or llm
            try:
                comments.extend(
                    await _run_llm_qa(
                        qa_llm, llm, rows, template, parsed, l10n_files, cfg, pr_title
                    )
                )
            except Exception as exc:
                logger.warning("l10n LLM QA failed (%s); keeping deterministic findings", exc)

    if comments:
        logger.info("l10n pass produced %d candidate comment(s)", len(comments))
    return comments


def _tone_samples(arb: ArbFile, exclude: set[str], limit: int = 3) -> list[str]:
    """Short stable strings of a locale, as a formality/tone reference."""
    candidates = [e for k, e in arb.entries.items() if k not in exclude and e.value]
    candidates.sort(key=lambda e: (len(e.value), e.key))
    return [e.value for e in candidates[:limit]]


def _build_rows(
    l10n_files: list,
    keys_per_file: dict[str, set[str]],
    parsed: dict[str, ArbFile],
    template: ArbFile,
    family: dict[str, ArbFile],
    cfg: L10nConfig,
) -> list[_Row]:
    """One row per (changed key, target locale worth checking).

    A key changed in the template checks *every* family locale that has a
    translation; a key changed in one locale file checks that file's own
    translation against the template.
    """
    rows: list[_Row] = []
    seen: set[str] = set()
    # Template file first: its changed keys build full-family rows (every
    # locale as a target); keys changed only in a locale file then fall
    # through to single-target rows instead of stealing the key.
    template_first = [f for f in l10n_files if template and f.path == template.path] + [
        f for f in l10n_files if not template or f.path != template.path
    ]
    for f in template_first:
        arb = parsed.get(f.path)
        if arb is None or not arb.parsed_ok:
            continue
        for key in sorted(keys_per_file.get(f.path, set())):
            if key in seen:
                continue
            entry = arb.entries.get(key)
            if entry is None:
                continue
            src_entry = template.entries.get(key)
            if arb.path == template.path:
                if src_entry is None:
                    continue
                targets = [
                    (locale_arb.locale, value)
                    for path, locale_arb in sorted(family.items())
                    if path != template.path
                    for value in [locale_arb.value(key)]
                    if value is not None
                ]
                anchor_path, anchor_line = template.path, src_entry.line
            else:
                source = src_entry.value if src_entry else ""
                if not source or source == entry.value:
                    continue
                targets = [(arb.locale, entry.value)]
                anchor_path, anchor_line = arb.path, entry.line
            if not targets:
                continue  # nothing translated yet: deterministic findings cover it
            seen.add(key)
            rows.append(
                _Row(
                    key=key,
                    anchor_path=anchor_path,
                    anchor_line=anchor_line,
                    source_locale=template.locale,
                    source=src_entry.value if src_entry else entry.value,
                    description=src_entry.description if src_entry else "",
                    targets=targets,
                )
            )
    if len(rows) > cfg.max_llm_calls * cfg.max_keys_per_call:
        logger.info(
            "l10n QA: %d row(s) exceed the %d-call budget; checking the first %d",
            len(rows),
            cfg.max_llm_calls,
            cfg.max_llm_calls * cfg.max_keys_per_call,
        )
        rows = rows[: cfg.max_llm_calls * cfg.max_keys_per_call]
    return rows


async def _run_llm_qa(
    qa_llm: object,
    fallback_llm: object,
    rows: list[_Row],
    template: ArbFile,
    parsed: dict[str, ArbFile],
    l10n_files: list,
    cfg: L10nConfig,
    pr_title: str = "",
) -> list[ReviewComment]:
    from mira.llm.prompts.l10n import build_l10n_review_prompt

    anchors = {(r.anchor_path, r.anchor_line) for r in rows}
    changed_keys_all = {r.key for r in rows}
    tone: dict[str, list[str]] = {}
    for arb in parsed.values():
        if arb.parsed_ok and arb.locale and arb.locale not in tone:
            tone[arb.locale] = _tone_samples(arb, changed_keys_all)

    out: list[ReviewComment] = []
    for start in range(0, len(rows), cfg.max_keys_per_call):
        batch = rows[start : start + cfg.max_keys_per_call]
        messages = build_l10n_review_prompt(
            rows=batch,
            template_locale=template.locale,
            tone_samples=tone,
            pr_title=pr_title,
        )
        try:
            raw = await qa_llm.complete_with_tools(  # type: ignore[attr-defined]
                messages=messages,
                tools=[SUBMIT_REVIEW_TOOL],
                temperature=0.0,
            )
        except Exception as exc:
            if qa_llm is fallback_llm:
                logger.warning("l10n QA call failed: %s", exc)
                break
            logger.warning("l10n QA on indexing tier failed (%s); retrying on review LLM", exc)
            try:
                raw = await fallback_llm.complete_with_tools(  # type: ignore[attr-defined]
                    messages=messages,
                    tools=[SUBMIT_REVIEW_TOOL],
                    temperature=0.0,
                )
            except Exception as exc2:
                logger.warning("l10n QA call failed: %s", exc2)
                break
        try:
            parsed_response = parse_llm_response(raw)
            converted = convert_to_review_comments(parsed_response, diff_files=l10n_files)
        except ResponseParseError as exc:
            logger.warning("l10n QA parse error: %s", exc)
            continue
        except Exception as exc:
            logger.warning("l10n QA conversion failed: %s", exc)
            continue
        for c in converted:
            if (c.path, c.line) in anchors:
                c.category = "l10n"
                c.source_pass = "l10n"
                out.append(c)
    return out
