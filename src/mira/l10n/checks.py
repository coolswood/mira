"""Deterministic l10n checks — no LLM, no quota.

Every check anchors its findings at the line of a *changed* key inside a
*diffed* file, so the comment posts as a normal inline review comment even
when the broken locale file itself was skipped by the diff-size budget.
Cross-locale problems (missing translations, dropped placeholders) are
aggregated into one comment per key instead of one per locale file.
"""

from dataclasses import dataclass

from mira.l10n.arb import (
    ArbFile,
    balanced_braces,
    extract_placeholders,
)

# Values at or below this length are often brand names ("OK", "Получить")
# that legitimately stay identical across locales.
_UNTRANSLATED_MIN_LEN = 16


@dataclass
class L10nFinding:
    """A candidate inline comment produced by an l10n check."""

    path: str
    line: int
    severity: str  # blocker | warning | suggestion | nitpick
    title: str
    body: str
    confidence: float
    suggestion: str | None = None


def _locales_missing_key(key: str, template: ArbFile, family: dict[str, ArbFile]) -> list[str]:
    """Paths of family locale files that lack ``key`` (template excluded)."""
    missing = []
    for path, arb in family.items():
        if path == template.path:
            continue
        if key not in arb.entries:
            missing.append(path)
    return sorted(missing)


def check_family_parity(
    template: ArbFile,
    changed_keys: set[str],
    family: dict[str, ArbFile],
) -> list[L10nFinding]:
    """Changed template keys that never reached some locales, plus stale
    placeholder sets in existing translations.

    One aggregated finding per key so a PR adding 20 keys to 20 locales
    cannot flood the review with 400 comments.
    """
    findings: list[L10nFinding] = []
    for key in sorted(changed_keys):
        entry = template.entries.get(key)
        if entry is None:
            continue
        missing = _locales_missing_key(key, template, family)
        broken: list[str] = []
        expected = set(entry.placeholders) or extract_placeholders(entry.value)
        if expected:
            for path, arb in family.items():
                if path == template.path:
                    continue
                other = arb.entries.get(key)
                if other is None:
                    continue
                actual = extract_placeholders(other.value)
                if actual != expected:
                    dropped = sorted(expected - actual)
                    extra = sorted(actual - expected)
                    detail = []
                    if dropped:
                        detail.append("нет " + ", ".join(f"{{{p}}}" for p in dropped))
                    if extra:
                        detail.append("лишние " + ", ".join(f"{{{p}}}" for p in extra))
                    broken.append("{} ({})".format(path, "; ".join(detail)))
        if not missing and not broken:
            continue
        parts = []
        if missing:
            parts.append(f"Ключ не переведён в {len(missing)} локал(ях): {', '.join(missing)}.")
        if broken:
            parts.append("Плейсхолдеры не совпадают с исходником: {}.".format("; ".join(broken)))
        findings.append(
            L10nFinding(
                path=template.path,
                line=entry.line,
                severity="warning",
                title="Перевод неполный: " + key,
                body=" ".join(parts) + f" Шаблон: «{_short(entry.value, 80)}»",
                confidence=0.95,
            )
        )
    return findings


def check_untranslated(
    template: ArbFile,
    changed_keys: set[str],
    family: dict[str, ArbFile],
) -> list[L10nFinding]:
    """Changed keys whose translation is byte-identical to the source text.

    Heuristic, so severity stays at ``suggestion`` and short strings
    (brand names, single words) are exempt.
    """
    findings: list[L10nFinding] = []
    for key in sorted(changed_keys):
        entry = template.entries.get(key)
        if entry is None or len(entry.value) < _UNTRANSLATED_MIN_LEN or " " not in entry.value:
            continue
        same = sorted(
            path
            for path, arb in family.items()
            if path != template.path
            and arb.value(key) is not None
            and arb.value(key) == entry.value
        )
        if not same:
            continue
        findings.append(
            L10nFinding(
                path=template.path,
                line=entry.line,
                severity="suggestion",
                title="Возможно, непереведённая строка: " + key,
                body="Значение совпадает с исходным ({}) в: {}. Если это не осознанный "
                "выбор (бренд, латиница) — нужен перевод.".format(template.locale, ", ".join(same)),
                confidence=0.7,
            )
        )
    return findings


def check_value_structure(
    arb: ArbFile,
    changed_keys: set[str],
    *,
    is_template: bool,
) -> list[L10nFinding]:
    """Per-file structure checks on changed values: broken JSON, unbalanced
    braces, undeclared placeholders, broken ICU verbs.

    These break ``flutter gen-l10n`` outright, hence the severities.
    """
    findings: list[L10nFinding] = []
    anchor = min(
        (arb.entries[k].line for k in changed_keys if k in arb.entries),
        default=1,
    )
    if not arb.parsed_ok:
        findings.append(
            L10nFinding(
                path=arb.path,
                line=anchor,
                severity="blocker",
                title="Файл локализации не парсится",
                body=f"{arb.path} — невалидный JSON. gen-l10n упадёт на сборке.",
                confidence=0.99,
            )
        )
        return findings

    for key in sorted(changed_keys):
        entry = arb.entries.get(key)
        if entry is None:
            continue
        if entry.value and not balanced_braces(entry.value):
            findings.append(
                L10nFinding(
                    path=arb.path,
                    line=entry.line,
                    severity="blocker",
                    title="Несбалансированные фигурные скобки: " + key,
                    body=f"В «{_short(entry.value, 60)}» не закрыт `{{...}}` (или лишний `}}`). ICU-парсер и "
                    "gen-l10n упадут.",
                    confidence=0.97,
                )
            )
            continue
        if is_template and entry.placeholders:
            declared = set(entry.placeholders)
            used = extract_placeholders(entry.value)
            undeclared = sorted(used - declared)
            if undeclared:
                findings.append(
                    L10nFinding(
                        path=arb.path,
                        line=entry.line,
                        severity="warning",
                        title="Плейсхолдер не объявлен в метаданных: " + key,
                        body="Используются {}, но в @{}/placeholders объявлены только {}. "
                        "gen-l10n упадёт.".format(
                            ", ".join(f"{{{p}}}" for p in undeclared),
                            key,
                            ", ".join(f"{{{p}}}" for p in sorted(declared)) or "∅",
                        ),
                        confidence=0.9,
                    )
                )
    return findings


def check_locale_tag(arb: ArbFile, changed_keys: set[str]) -> list[L10nFinding]:
    """``@@locale`` inside the file must match the filename suffix.

    ``app_pt_BR.arb`` → expected tag ``pt_BR`` (everything after the stem)."""
    stem = arb.path.rsplit("/", 1)[-1][: -len(".arb")]
    expected = stem.split("_", 1)[1] if "_" in stem else stem
    if arb.parsed_ok and arb.locale and arb.locale != expected:
        anchor = min(
            (arb.entries[k].line for k in changed_keys if k in arb.entries),
            default=1,
        )
        return [
            L10nFinding(
                path=arb.path,
                line=anchor,
                severity="warning",
                title="@@locale не совпадает с именем файла",
                body=f"Файл называет локаль «{expected}», а @@locale внутри — «{arb.locale}». "
                "gen-l10n сопоставляет их по имени файла; расхождение даёт неверную "
                "локаль в рантайме.",
                confidence=0.9,
            )
        ]
    return []


def check_against_template(
    arb: ArbFile,
    changed_keys: set[str],
    template: ArbFile | None,
) -> list[L10nFinding]:
    """Keys edited in a locale file that don't exist in the template at all.

    Almost always a typo in the key name or a string added to one locale
    only — both silently dead in the built app.
    """
    if template is None or arb.path == template.path:
        return []
    # Structural mismatch guard: a real locale file shares nearly all keys
    # with its template. If virtually nothing overlaps, this "template" does
    # not govern the file (split-fragment source trees, wrong heuristic
    # guess) and the check would only flood false positives — skip it.
    if len(arb.entries) >= 8:
        shared = sum(1 for k in arb.entries if k in template.entries)
        if shared < len(arb.entries) * 0.2:
            return []
    findings: list[L10nFinding] = []
    for key in sorted(changed_keys):
        if key in template.entries:
            continue
        entry = arb.entries.get(key)
        if entry is None:
            continue
        findings.append(
            L10nFinding(
                path=arb.path,
                line=entry.line,
                severity="warning",
                title="Ключа нет в шаблоне: " + key,
                body=f"Ключ «{key}» есть в {arb.path} (locale {arb.locale}), но отсутствует в шаблоне {template.path}. "
                "Либо опечатка в имени ключа, либо строку не завели в источнике — "
                "в приложении она мёртвая.",
                confidence=0.9,
            )
        )
    return findings


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
