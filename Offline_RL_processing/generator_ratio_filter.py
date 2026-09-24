"""Shared ratio filtering helpers for generated evaluation scripts."""

from __future__ import annotations

import re
from collections.abc import Sequence

_RATIO_TOKEN = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$")
_RATIO_PREFIX = re.compile(
    r"^(?:retained_ratios|retain_ratios|learn_ratios|ratios)[_=\s-]*",
    re.IGNORECASE,
)


def normalize_ratio_filter(ratio: str | Sequence[object] | None) -> str | None:
    """Normalize --ratio input to the underscore-joined directory token.

    Both --ratio 0.9_1.0_1.0 and --ratio 0.9 1.0 1.0 are accepted.
    """
    if ratio is None:
        return None
    if isinstance(ratio, (list, tuple)):
        raw = "_".join(str(value) for value in ratio)
    else:
        raw = str(ratio)
    raw = _RATIO_PREFIX.sub("", raw.strip())
    tokens = [token for token in re.split(r"[_\s,]+", raw) if token]
    if not tokens:
        raise ValueError("--ratio must contain at least one numeric ratio value.")
    normalized = []
    for token in tokens:
        token = token.strip().lower().replace("p", ".")
        if not _RATIO_TOKEN.fullmatch(token):
            raise ValueError(
                "--ratio values must be numeric, for example 0.9_1.0_1.0."
            )
        normalized.append(token)
    return "_".join(normalized)


def path_matches_ratio(path_str: str, ratio: str | Sequence[object] | None) -> bool:
    """Return whether a path contains the requested ratio directory token."""
    ratio_tag = normalize_ratio_filter(ratio)
    if ratio_tag is None:
        return True
    normalized_path = str(path_str).replace("\\", "/")
    return re.search(
        rf"(?:^|[/_]){re.escape(ratio_tag)}(?:$|[/_])",
        normalized_path,
        flags=re.IGNORECASE,
    ) is not None


def ratio_filename_suffix(ratio: str | Sequence[object] | None) -> str:
    """Return the output filename marker for an enabled ratio filter."""
    ratio_tag = normalize_ratio_filter(ratio)
    return f"_ratio_{ratio_tag}" if ratio_tag else ""
