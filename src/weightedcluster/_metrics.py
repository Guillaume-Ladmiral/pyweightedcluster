# pyweightedcluster metrics utilities
from typing import Literal  # noqa: I001

MetricName = Literal[
    "HAM",
    "HAMMING",
    "hamming",
    "OM",
    "OPTIMAL",
    "OPTIMAL_MATCHING",
    "om",
    "LCS",
    "lcs",
    "LONGEST_COMMON_SUBSEQUENCE",
    "DHD",
    "dhd",
    "DYNAMIC_HAMMING",
    "OMSPELL",
    "omspell",
]


def normalize_metric(metric: str) -> tuple[str, str]:
    """Normalize user metric name to canonical short token and pythonic alias.

    Returns (canonical, alias), e.g., ("HAM", "hamming").

    Migration note (Phase 1 / A4, DUPLICATES.md): this is a deliberate KEEP — a
    tiny, self-contained metric-name canonicalizer for this project's public
    ``seqdist_args`` surface, not a distance computation. It is the front door
    that ``_seqdist`` uses before delegating the actual distance to traminex, so
    it is "folded into A1" rather than replaced. traminex's own method-name
    validation is private and uses a different vocabulary, so reusing it here
    would couple to an internal API for no real gain (charter section 4 allows a
    small self-contained helper as a recorded decision).
    """
    m = str(metric)
    mu = m.upper()
    if mu in ("HAM", "HAMMING"):
        return "HAM", "hamming"
    if mu in ("OM", "OPTIMAL", "OPTIMAL_MATCHING"):
        return "OM", "om"
    if mu in ("LCS", "LONGEST_COMMON_SUBSEQUENCE"):
        return "LCS", "lcs"
    if mu in ("DHD", "DYNAMIC_HAMMING"):
        return "DHD", "dhd"
    if mu in ("OMSPELL",):
        return "OMSPELL", "omspell"
    raise ValueError(f"Unsupported metric: {metric}")
