"""Multi-specialist drafter (LangGraph multi-agent pattern, 2026).

Each specialist's system prompt is now assembled from the layered
white-dwarf domain template in `analysis_agent.prompts.wd_domain` so
that the L0..L4 (citation discipline, role, domain priors, output
contract, failure modes) layers are shared across the writer team and
can be hot-swapped via the Prompt Lab overrides.

  PhysicistAgent      -> Methods, Results
  WriterAgent         -> Abstract, Introduction, Discussion, Conclusions
  CriticAgent         -> reads the assembled paper, produces structured
                         JSON critique (complementary signal to paper_qc).
"""

from __future__ import annotations

from typing import Dict

from .prompts import wd_domain


# Lazily assembled at import time; the override layer means the strings
# can still change between requests (a server reload is not required to
# pick up an override). To force a re-read on every call, the
# `system_prompt_for` helper re-invokes the builder each time.

def _physicist_system() -> str:
    return (
        wd_domain.system_for_role("physicist")
        + "\n"
        + "## L5. Task framing\n"
        + "You will be asked to draft Methods or Results. Honour the "
        + "section task prompt verbatim. Return ONLY LaTeX for the "
        + "requested section (no preamble).\n"
    )


def _writer_system() -> str:
    return (
        wd_domain.system_for_role("writer")
        + "\n"
        + "## L5. Task framing\n"
        + "You will be asked to draft Abstract, Introduction, Discussion, "
        + "or Conclusions. Honour the section task prompt verbatim. "
        + "Return ONLY LaTeX for the requested section (no preamble).\n"
    )


def _critic_system() -> str:
    return (
        wd_domain.system_for_role("critic")
        + "\n"
        + wd_domain.CRITIC_TASK
    )


SECTION_TO_SPECIALIST: Dict[str, str] = {
    "Abstract": "writer",
    "Introduction": "writer",
    "Data": "physicist",
    "Methods": "physicist",
    "Results": "physicist",
    "Discussion": "writer",
    "Conclusions": "writer",
}


def specialist_for(section: str) -> str:
    return SECTION_TO_SPECIALIST.get(section, "writer")


def system_prompt_for(specialist: str) -> str:
    if specialist == "physicist":
        return _physicist_system()
    if specialist == "critic":
        return _critic_system()
    return _writer_system()


# Backwards compatibility for any caller that still imports these names.
PHYSICIST_SYSTEM = _physicist_system()
WRITER_SYSTEM = _writer_system()
CRITIC_SYSTEM = _critic_system()


__all__ = [
    "PHYSICIST_SYSTEM",
    "WRITER_SYSTEM",
    "CRITIC_SYSTEM",
    "SECTION_TO_SPECIALIST",
    "specialist_for",
    "system_prompt_for",
]
