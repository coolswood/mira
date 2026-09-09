"""ARB file parsing for the l10n pass.

ARB files are JSON with a flat structure: ``"key": "value"`` entries,
``"@key": {...}`` metadata objects, and a special ``"@@locale"`` tag.
Values may contain ICU message syntax (``{name}``, ``{name, plural, ...}``)
and HTML-ish markup.

The parser is deliberately line-oriented for line numbers (needed to anchor
inline review comments) and delegates value/metadata decoding to ``json`` so
escaped strings come back correct.
"""

import json
import re
from dataclasses import dataclass, field

# A top-level "key": or "@key": or "@@locale": at the start of a line.
_KEY_LINE_RE = re.compile(r'^\s*"((?:[^"\\]|\\.)+)"\s*:')
# A "key": at the start of a `+`/`-` diff line inside a hunk.
_DIFF_KEY_LINE_RE = re.compile(r'^\s*[+-]\s*"((?:[^"\\]|\\.)+)"\s*:')

_LOCALE_TAG = "@@locale"
_META_PREFIX = "@"


@dataclass
class ArbEntry:
    """One localization key with its value and optional metadata."""

    key: str
    value: str
    line: int  # 1-based line of the "key": in the file
    description: str = ""
    placeholders: dict[str, str] = field(default_factory=dict)  # name -> ICU type


@dataclass
class ArbFile:
    """A parsed ARB file."""

    path: str
    locale: str  # from @@locale, or derived from the filename
    entries: dict[str, ArbEntry] = field(default_factory=dict)
    parsed_ok: bool = True

    def value(self, key: str) -> str | None:
        entry = self.entries.get(key)
        return entry.value if entry else None


def locale_from_filename(path: str) -> str:
    """``lib/l10n/app_pt_BR.arb`` → ``pt_BR``; ``messages.arb`` → ``messages``.

    Convention: ``<stem>_<locale>.arb`` — everything after the first
    underscore is the locale tag.
    """
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".arb"):
        name = name[: -len(".arb")]
    return name.split("_", 1)[1] if "_" in name else name


def parse_arb(path: str, content: str) -> ArbFile:
    """Parse ARB content into an :class:`ArbFile`.

    Never raises: an unparseable file comes back with ``parsed_ok=False`` and
    whatever the line scan could see, so the caller can flag the broken JSON.
    """
    arb = ArbFile(path=path, locale=locale_from_filename(path))
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            arb.parsed_ok = False
            data = {}
    except (json.JSONDecodeError, ValueError):
        arb.parsed_ok = False
        data = {}

    if isinstance(data.get(_LOCALE_TAG), str):
        arb.locale = data[_LOCALE_TAG]

    # Line scan for anchoring; keys appear before their @-metadata block.
    seen: dict[str, int] = {}
    for lineno, line in enumerate(content.splitlines(), start=1):
        match = _KEY_LINE_RE.match(line)
        if match:
            seen.setdefault(match.group(1), lineno)

    for raw_key, lineno in seen.items():
        if raw_key == _LOCALE_TAG:
            continue
        if raw_key.startswith(_META_PREFIX):
            continue  # metadata is folded into its key below
        if raw_key not in data:
            # The line scan has no depth information: a "description": or
            # "days": line nested inside @key metadata matches just like a
            # top-level entry. Only keys present at the JSON top level count.
            continue
        raw_value = data[raw_key]
        if not isinstance(raw_value, str):
            # Non-string entries (rare) still get anchored but no ICU checks.
            raw_value = str(raw_value) if raw_value is not None else ""
        meta = data.get(_META_PREFIX + raw_key)
        description = ""
        placeholders: dict[str, str] = {}
        if isinstance(meta, dict):
            desc = meta.get("description")
            if isinstance(desc, str):
                description = desc
            raw_placeholders = meta.get("placeholders")
            if isinstance(raw_placeholders, dict):
                for name, spec in raw_placeholders.items():
                    ptype = ""
                    if isinstance(spec, dict):
                        ptype = str(spec.get("type", ""))
                    placeholders[str(name)] = ptype
        arb.entries[raw_key] = ArbEntry(
            key=raw_key,
            value=raw_value,
            line=lineno,
            description=description,
            placeholders=placeholders,
        )
    return arb


def changed_keys(file_diff_hunk_lines: list[str]) -> set[str]:
    """Localization keys touched by ``+`` diff lines of one file.

    Only ``+`` lines count: a key whose line was edited shows up as a
    ``+`` replacement, and purely deleted keys are intentionally ignored
    (stale-translation cleanup is not this pass's job).
    """
    keys: set[str] = set()
    for line in file_diff_hunk_lines:
        if not line.startswith("+") or line.startswith("+++"):
            continue
        match = _DIFF_KEY_LINE_RE.match(line)
        if match and not match.group(1).startswith(_META_PREFIX):
            keys.add(match.group(1))
    return keys


def hunk_plus_lines(hunk_contents: list[str]) -> list[str]:
    """Flat ``+``-prefixed lines across a file's hunks (as stored in HunkInfo)."""
    lines: list[str] = []
    for content in hunk_contents:
        lines.extend(content.splitlines())
    return lines


_ICU_VERBS = ("plural", "select", "selectordinal")
_ARG_HEADER_RE = re.compile(
    r"^\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*(" + "|".join(_ICU_VERBS) + r")\b)?"
)


def _skip_quoted(value: str, start: int) -> int:
    """Index past ICU quoting at ``value[start] == "'"`` ('…' or literal '')."""
    if value.startswith("''", start):
        return start + 2
    j = value.find("'", start + 1)
    return j + 1 if j != -1 else len(value)


def _iter_arguments(value: str) -> list[str]:
    """Top-level ``{...}`` argument texts in an ICU message, quotes respected."""
    args: list[str] = []
    depth = 0
    start = 0
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "'":
            i = _skip_quoted(value, i)
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                args.append(value[start : i + 1])
            elif depth < 0:
                break  # unbalanced; balanced_braces() reports this
        i += 1
    return args


def balanced_braces(value: str) -> bool:
    """True when every ``{`` in the ICU value has a matching ``}``."""
    depth = 0
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "'":
            i = _skip_quoted(value, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0


def extract_placeholders(value: str) -> set[str]:
    """Top-level ``{name}`` argument names in an ICU message value.

    Names inside plural/select sub-messages are intentionally excluded:
    placeholder parity across locales is about the top-level arguments.
    """
    names: set[str] = set()
    for arg in _iter_arguments(value):
        match = _ARG_HEADER_RE.match(arg)
        if match:
            names.add(match.group(1))
    return names


def icu_verbs(value: str) -> set[str]:
    """ICU construct verbs used at the top level (``plural``, ``select``, ...)."""
    verbs: set[str] = set()
    for arg in _iter_arguments(value):
        match = _ARG_HEADER_RE.match(arg)
        if match and match.group(2):
            verbs.add(match.group(2))
    return verbs
