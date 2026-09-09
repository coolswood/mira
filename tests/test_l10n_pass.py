"""l10n pass — ARB parsing, deterministic checks, and the QA orchestration.

All LLM/provider interactions are stubbed; no real codex calls.
"""

from __future__ import annotations

import json

from mira.config import L10nConfig
from mira.l10n.arb import (
    balanced_braces,
    changed_keys,
    extract_placeholders,
    hunk_plus_lines,
    icu_verbs,
    locale_from_filename,
    parse_arb,
)
from mira.l10n.checks import (
    check_against_template,
    check_family_parity,
    check_locale_tag,
    check_untranslated,
    check_value_structure,
)
from mira.l10n.pipeline import _build_rows, is_l10n_path, l10n_review_pass
from mira.models import FileChangeType, FileDiff, HunkInfo, PRInfo

RU = """{
  "@@locale": "ru",
  "app_title": "Наше приложение",
  "greeting": "Привет, {name}!",
  "@greeting": {"description": "Приветствие на главной", "placeholders": {"name": {"type": "String"}}},
  "visits_one": "Это твой {days} визит",
  "@visits_one": {"description": "Дни подряд", "placeholders": {"days": {"type": "String"}}},
  "brand_new": "Добро пожаловать в наше прекрасное приложение"
}
"""

DE = """{
  "@@locale": "de",
  "app_title": "Unsere App",
  "greeting": "Hallo {nims}!",
  "brand_new": "Добро пожаловать в наше прекрасное приложение"
}
"""

IT = """{
  "@@locale": "it",
  "app_title": "La nostra app",
  "greeting": "Ciao {name"
}
"""

L10N_YAML = "arb-dir: lib/l10n\ntemplate-arb-file: app_ru.arb\n"


def _fdiff(path: str, plus_lines: list[str], target_start: int = 2) -> FileDiff:
    content = "\n".join("+" + line for line in plus_lines)
    hunk = HunkInfo(
        source_start=target_start - 1,
        source_length=1,
        target_start=target_start,
        target_length=len(plus_lines),
        content=content,
    )
    return FileDiff(path=path, change_type=FileChangeType.MODIFIED, hunks=[hunk])


class FakeProvider:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = files
        self.fetched: list[str] = []

    async def get_file_content(self, pr_info: PRInfo, path: str, ref: str) -> str:
        self.fetched.append(path)
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(path)


class FakeLLM:
    def __init__(self, response: str = '{"comments": []}') -> None:
        self.response = response
        self.calls = 0

    async def complete_with_tools(self, messages, tools, temperature=0.0):  # noqa: ANN001
        self.calls += 1
        return self.response


def _pr_info() -> PRInfo:
    return PRInfo(
        title="l10n update",
        description="",
        base_branch="main",
        head_branch="feature/l10n",
        url="https://example.test/o/r/pull/1",
        number=1,
        owner="o",
        repo="r",
    )


# ── arb.py ───────────────────────────────────────────────────────────


def test_parse_arb_entries_lines_and_metadata() -> None:
    arb = parse_arb("lib/l10n/app_ru.arb", RU)
    assert arb.locale == "ru"
    assert set(arb.entries) == {"app_title", "greeting", "visits_one", "brand_new"}
    greeting = arb.entries["greeting"]
    assert greeting.line == 4
    assert greeting.description == "Приветствие на главной"
    assert greeting.placeholders == {"name": "String"}


def test_locale_from_filename() -> None:
    assert locale_from_filename("lib/l10n/app_pt_BR.arb") == "pt_BR"
    assert locale_from_filename("messages.arb") == "messages"


def test_changed_keys_from_hunks() -> None:
    lines = hunk_plus_lines(
        ['+  "greeting": "Привет, {name}!",\n+  "@greeting": {"description": "x"},\n-  "old": "y",']
    )
    assert changed_keys(lines) == {"greeting"}


def test_extract_placeholders_with_plural_nesting() -> None:
    value = "{count, plural, other {# дней для {name}}} и {extra}"
    assert extract_placeholders(value) == {"count", "extra"}
    assert icu_verbs(value) == {"plural"}


def test_quoted_braces_are_not_placeholders() -> None:
    value = "It''s a '{literal}' {real}"
    assert extract_placeholders(value) == {"real"}


def test_balanced_braces_detects_unclosed() -> None:
    assert balanced_braces("Ciao {name}!")
    assert not balanced_braces("Ciao {name")
    assert not balanced_braces("Ciao {name}} extra {")


# ── checks.py ────────────────────────────────────────────────────────


def _fixtures() -> tuple:  # type: ignore[no-untyped-def]
    ru = parse_arb("lib/l10n/app_ru.arb", RU)
    de = parse_arb("lib/l10n/app_de.arb", DE)
    it = parse_arb("lib/l10n/app_it.arb", IT)
    family = {a.path: a for a in (ru, de, it)}
    changed = {"greeting", "visits_one", "brand_new"}
    return ru, de, it, family, changed


def test_family_parity_aggregates_missing_and_placeholders() -> None:
    ru, _de, _it, family, changed = _fixtures()
    findings = check_family_parity(ru, changed, family)
    by_title = {f.title: f for f in findings}
    visits = [f for t, f in by_title.items() if "visits_one" in t]
    assert visits and "app_de.arb" in visits[0].body and "app_it.arb" in visits[0].body
    greeting = [f for t, f in by_title.items() if "greeting" in t]
    assert greeting and "{nims}" in greeting[0].body


def test_untranslated_flags_identical_long_value_only() -> None:
    ru, _de, _it, family, changed = _fixtures()
    findings = check_untranslated(ru, changed, family)
    assert len(findings) == 1
    assert findings[0].severity == "suggestion"
    assert "brand_new" in findings[0].title
    assert "app_de.arb" in findings[0].body


def test_value_structure_flags_unbalanced_braces() -> None:
    ru, _de, it, _family, changed = _fixtures()
    findings = check_value_structure(it, {"greeting"}, is_template=False)
    assert findings and findings[0].severity == "blocker"


def test_value_structure_flags_undeclared_placeholder_in_template() -> None:
    # gen-l10n requires every {arg} in the value to be declared in @key;
    # here the metadata declares only "x" while the value uses {name}.
    ru_with_bad_meta = RU.replace(
        '"placeholders": {"name": {"type": "String"}}},',
        '"placeholders": {"x": {"type": "String"}}},',
        1,  # only the @greeting line, not @visits_one
    )
    assert ru_with_bad_meta != RU  # the fixture replace must actually match
    ru = parse_arb("lib/l10n/app_ru.arb", ru_with_bad_meta)
    findings = check_value_structure(ru, {"greeting"}, is_template=True)
    assert findings and "не объявлен" in findings[0].title


def test_value_structure_flags_invalid_json() -> None:
    arb = parse_arb("lib/l10n/app_ru.arb", RU[:-2])  # cut the closing brace
    assert not arb.parsed_ok
    findings = check_value_structure(arb, {"greeting"}, is_template=True)
    assert findings and findings[0].severity == "blocker"


def test_locale_tag_mismatch() -> None:
    arb = parse_arb("lib/l10n/app_it.arb", IT.replace('"@@locale": "it"', '"@@locale": "de"'))
    findings = check_locale_tag(arb, {"greeting"})
    assert findings and "app_it.arb" in findings[0].path


def test_check_against_template_flags_unknown_key() -> None:
    ru = parse_arb("lib/l10n/app_ru.arb", RU)
    de_typo = parse_arb(
        "lib/l10n/app_de.arb",
        DE.replace('"\n}', '",\n  "unknown_key": "Was ist das?"\n}'),
    )
    assert "unknown_key" in de_typo.entries  # fixture sanity: still valid JSON
    findings = check_against_template(de_typo, {"greeting", "unknown_key"}, ru)
    assert len(findings) == 1
    assert "unknown_key" in findings[0].title


# ── pass.py ──────────────────────────────────────────────────────────

FILES = {
    "l10n.yaml": L10N_YAML,
    "lib/l10n/app_ru.arb": RU,
    "lib/l10n/app_de.arb": DE,
    "lib/l10n/app_it.arb": IT,
    "lib/main.dart": "void main() {}",
}
TREE = [
    "lib/l10n/app_ru.arb",
    "lib/l10n/app_de.arb",
    "lib/l10n/app_it.arb",
    "lib/l10n/src/ru/home_bot.arb",
    "lib/main.dart",
]


def _llm_response() -> str:
    ru = parse_arb("lib/l10n/app_ru.arb", RU)
    line = ru.entries["greeting"].line
    return json.dumps(
        {
            "comments": [
                {
                    "path": "lib/l10n/app_ru.arb",
                    "line": line,
                    "severity": "suggestion",
                    "category": "l10n",
                    "title": "Регистр: greeting",
                    "body": "В немецкой локали «Hallo» — верно, но убедитесь в единообразии.",
                    "confidence": 0.7,
                    "suggestion": "Hallo {name}!",
                },
                {
                    "path": "lib/l10n/app_fr.arb",
                    "line": 1,
                    "severity": "warning",
                    "category": "l10n",
                    "title": "Галлюцинация",
                    "body": "Файла нет ни в диффе, ни в анкорах.",
                    "confidence": 0.9,
                },
            ]
        }
    )


async def test_pass_end_to_end_with_mocks() -> None:
    provider = FakeProvider(FILES)
    llm = FakeLLM(_llm_response())
    diffed = [
        _fdiff(
            "lib/l10n/app_ru.arb",
            [
                '"greeting": "Привет, {name}!",',
                '"visits_one": "Это твой {days} визит",',
                '"brand_new": "Добро пожаловать в наше прекрасное приложение"',
            ],
            target_start=4,
        ),
        _fdiff("lib/l10n/app_de.arb", ['"greeting": "Hallo {nims}!",'], target_start=4),
        _fdiff("lib/l10n/app_it.arb", ['"greeting": "Ciao {name"'], target_start=4),
    ]
    comments = await l10n_review_pass(
        llm,
        diffed,
        provider=provider,
        pr_info=_pr_info(),
        repo_tree=TREE,
        pr_title="l10n update",
        config=L10nConfig(),
    )

    assert llm.calls == 1  # one small batch, no quota abuse
    categories = {c.source_pass for c in comments}
    assert categories == {"l10n"}
    # Deterministic: broken braces in it.arb, placeholder mismatch + missing
    # visits_one in de/it, untranslated brand_new in de.
    assert any("Несбалансированные" in c.title for c in comments)
    assert any("Перевод неполный" in c.title and "visits_one" in c.title for c in comments)
    assert any("Возможно, непереведённая" in c.title for c in comments)
    # LLM comment survived only with a valid anchor; the hallucinated one is gone.
    assert any("Регистр" in c.title for c in comments)
    assert not any("Галлюцинация" in c.title for c in comments)


async def test_pass_without_provider_returns_no_findings() -> None:
    llm = FakeLLM()
    diffed = [_fdiff("lib/l10n/app_ru.arb", ['"greeting": "Привет, {name}!",'])]
    comments = await l10n_review_pass(
        llm,
        diffed,
        provider=None,
        pr_info=None,
        repo_tree=TREE,
        config=L10nConfig(),
    )
    assert comments == []
    assert llm.calls == 0


async def test_pass_respects_llm_call_budget() -> None:
    provider = FakeProvider(FILES)
    llm = FakeLLM(_llm_response())
    diffed = [_fdiff("lib/l10n/app_ru.arb", ['"greeting": "Привет, {name}!",'])]
    cfg = L10nConfig(max_llm_calls=0)
    await l10n_review_pass(
        llm,
        diffed,
        provider=provider,
        pr_info=_pr_info(),
        repo_tree=TREE,
        config=cfg,
    )
    assert llm.calls == 0


def test_build_rows_dedupes_and_covers_family() -> None:
    ru = parse_arb("lib/l10n/app_ru.arb", RU)
    de = parse_arb("lib/l10n/app_de.arb", DE)
    it = parse_arb("lib/l10n/app_it.arb", IT)
    family = {a.path: a for a in (ru, de, it)}
    parsed = {a.path: a for a in (ru, de, it)}
    cfg = L10nConfig()
    rows = _build_rows(
        [
            _fdiff("lib/l10n/app_ru.arb", ['"greeting": "x"']),
            _fdiff("lib/l10n/app_de.arb", ['"greeting": "y"']),
        ],
        {
            "lib/l10n/app_ru.arb": {"greeting"},
            "lib/l10n/app_de.arb": {"greeting"},
        },
        parsed,
        ru,
        family,
        cfg,
    )
    # One row per key, carrying every translated locale as a target.
    assert len(rows) == 1
    assert rows[0].key == "greeting"
    assert dict(rows[0].targets) == {"de": "Hallo {nims}!", "it": "Ciao {name"}


def test_is_l10n_path() -> None:
    assert is_l10n_path("lib/l10n/app_ru.arb", ["*.arb"])
    assert not is_l10n_path("lib/main.dart", ["*.arb"])
