"""Localization (l10n) consistency checking.

Deterministic checks plus an LLM semantic pass over changed keys in
localization files (ARB / Flutter gen-l10n today, extensible via config
patterns). Findings anchor to the changed key's line so they post as
normal inline review comments and flow through the standard filter /
critique pipeline.
"""
