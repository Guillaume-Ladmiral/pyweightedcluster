"""Strict handling of current-governance annotations on historical documents."""

from __future__ import annotations

GOVERNANCE_NOTICE_MARKER = "> **Governance notice (2026-08-03):**"

APPROVED_GOVERNANCE_NOTICES = frozenset(
    {
        """> **Governance notice (2026-08-03):** This protocol remains historical study
> evidence. Its blanket blocking and immutability rules are superseded by
> [`SCIENTIFIC_GOVERNANCE.md`](SCIENTIFIC_GOVERNANCE.md); the current B1
> decision is [`BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`](BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md).""",
        """> **Governance notice (2026-08-03):** This review preserves its original
> evidence and decision as history. Current interpretation is governed by
> [`SCIENTIFIC_GOVERNANCE.md`](SCIENTIFIC_GOVERNANCE.md) and
> [`BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`](BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md).""",
        """> **Governance notice (2026-08-03):** This report preserves its original
> evidence and decision as history. Current interpretation is governed by
> [`SCIENTIFIC_GOVERNANCE.md`](SCIENTIFIC_GOVERNANCE.md) and
> [`BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`](BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md).""",
        """> **Governance notice (2026-08-03):** This review remains historical evidence.
> Its recursive authorization boundary is superseded by
> [`SCIENTIFIC_GOVERNANCE.md`](SCIENTIFIC_GOVERNANCE.md); the current B1
> decision is [`BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`](BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md).""",
        """> **Governance notice (2026-08-03):** This decision remains historical
> evidence. Current interpretation is governed by
> [`SCIENTIFIC_GOVERNANCE.md`](SCIENTIFIC_GOVERNANCE.md) and
> [`BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`](BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md).""",
        """> **Governance notice (2026-08-03):** This document is historical execution
> authority. Its blanket blocking and immutability rules are superseded by
> [`SCIENTIFIC_GOVERNANCE.md`](SCIENTIFIC_GOVERNANCE.md); the current B1
> decision is [`BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md`](BAYESIAN_RARCAT_B1_GOVERNANCE_CORRECTION.md).""",
    }
)


def historical_document_payload_bytes(document: bytes) -> bytes:
    """Return hashable historical bytes after one strictly approved annotation."""
    text = document.decode("utf-8")
    lines = text.splitlines(keepends=True)
    marker_indexes = [
        index for index, line in enumerate(lines) if line.startswith(GOVERNANCE_NOTICE_MARKER)
    ]
    if not marker_indexes:
        return document
    if marker_indexes != [2] or len(lines) < 4 or lines[1].strip():
        raise ValueError("governance notice must appear once immediately after the title")

    end = marker_indexes[0]
    notice_lines: list[str] = []
    while end < len(lines) and lines[end].startswith(">"):
        notice_lines.append(lines[end].rstrip("\r\n"))
        end += 1
    notice = "\n".join(notice_lines)
    if notice not in APPROVED_GOVERNANCE_NOTICES:
        raise ValueError("unapproved or modified governance notice")
    if end >= len(lines) or lines[end].strip():
        raise ValueError("governance notice must be followed by one blank line")

    return "".join(lines[: marker_indexes[0]] + lines[end + 1 :]).encode("utf-8")
