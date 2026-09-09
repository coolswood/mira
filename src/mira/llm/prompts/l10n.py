"""Prompt builder for the l10n translation-QA pass.

System message carries the reviewer instructions (rendered from
``l10n_review.jinja2``); the user message is the batch table itself — one
line per (changed key × target locale) with an exact ``file:line`` anchor
the model must echo back, plus per-locale tone samples.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from mira.l10n.pipeline import _Row

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _shorten(text: str, limit: int = 240) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_l10n_review_prompt(
    rows: list[_Row],
    template_locale: str,
    tone_samples: dict[str, list[str]],
    pr_title: str = "",
) -> list[dict[str, str]]:
    """Messages for one batch of l10n QA rows."""
    env = _get_template_env()
    template = env.get_template("l10n_review.jinja2")
    system_content = template.render(
        pr_title=pr_title,
        template_locale=template_locale,
        tone_samples=tone_samples,
    )

    lines: list[str] = []
    for i, row in enumerate(rows, start=1):
        desc = f" | desc: {_shorten(row.description, 160)}" if row.description else ""
        lines.append(
            f"[{i}] key={row.key} | anchor={row.anchor_path}:{row.anchor_line} | "
            f"src({row.source_locale}): {_shorten(row.source, 200)}{desc}"
        )
        for locale, value in row.targets:
            lines.append(f"      {locale}: {_shorten(value, 200)}")
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(lines) if lines else "(no rows)"},
    ]
