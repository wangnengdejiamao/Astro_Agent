"""Output guards.

The most important guard is for ``paper_refine``: Claude Code is allowed to
rewrite phrasing and tighten logic, but it must NOT introduce new scientific
claims that have no backing in the existing draft. We implement a conservative
heuristic: any sentence containing a quantitative claim (number + unit, or
"sigma" / "p<", "detection of", etc.) that does not appear in the original
draft is flagged.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence


class PaperRefinementGuardError(ValueError):
    """Raised when refined draft contains unsupported new science claims."""


# Conservative patterns that suggest a NEW scientific claim.
_CLAIM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:sigma|σ)\b", re.IGNORECASE),
    re.compile(r"\bp\s*<\s*0?\.\d+", re.IGNORECASE),
    re.compile(r"\b(?:detection|discovery|confirmation)\s+of\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:Msun|M_\\?odot|Mjup|au|pc|kpc|Mpc|Gyr|Myr|km/s)\b"),
    re.compile(r"\b(?:redshift|z)\s*=\s*\d+(?:\.\d+)?", re.IGNORECASE),
)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def extract_changed_files(stdout: str) -> List[str]:
    """Best-effort extraction of file paths Claude Code reports as changed."""
    paths: list[str] = []
    for m in re.finditer(r"(?:changed|modified|wrote|created):\s*([^\s]+)", stdout, re.IGNORECASE):
        paths.append(m.group(1))
    return list(dict.fromkeys(paths))


def find_new_claims(original: str, refined: str) -> List[str]:
    """Return sentences in ``refined`` that contain a claim pattern but do not
    appear (normalized) anywhere in ``original``."""
    orig_norm = _normalize(original)
    flagged: list[str] = []
    for sent in _split_sentences(refined):
        if not any(p.search(sent) for p in _CLAIM_PATTERNS):
            continue
        if _normalize(sent) in orig_norm:
            continue
        # Allow if every claim token in the sentence appears in original.
        tokens = []
        for p in _CLAIM_PATTERNS:
            tokens.extend(m.group(0).lower() for m in p.finditer(sent))
        if tokens and all(tok in orig_norm for tok in tokens):
            continue
        flagged.append(sent)
    return flagged


def guard_paper_refinement(original: str, refined: str) -> None:
    """Raise PaperRefinementGuardError if refined draft adds new claims."""
    new_claims = find_new_claims(original, refined)
    if new_claims:
        raise PaperRefinementGuardError(
            "Refined draft introduces unsupported claims:\n- "
            + "\n- ".join(new_claims[:10])
        )


__all__ = [
    "guard_paper_refinement",
    "PaperRefinementGuardError",
    "find_new_claims",
    "extract_changed_files",
]
