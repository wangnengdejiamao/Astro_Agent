"""Codex-derived operating rules adapted for the astronomy agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import tools


CODEX_STYLE_RULES: List[Dict[str, str]] = [
    {
        "name": "bounded_context",
        "rule": "Every model-facing fragment must be bounded, summarized, and preferably stored as an artifact path.",
    },
    {
        "name": "structured_tools",
        "rule": "Tool outputs are JSON/CSV/LaTeX artifacts with stable paths; prose summaries are secondary.",
    },
    {
        "name": "progressive_skills",
        "rule": "Keep core skill instructions short and load references only when needed.",
    },
    {
        "name": "review_first",
        "rule": "Scientific and code reviews list concrete risks before summaries or praise.",
    },
    {
        "name": "integration_tests",
        "rule": "Agent-logic changes require integration/smoke tests that exercise the whole workflow.",
    },
    {
        "name": "minimal_edits",
        "rule": "Prefer small, local changes that match existing project patterns and do not rewrite unrelated files.",
    },
    {
        "name": "no_secret_persistence",
        "rule": "API keys and tokens must come from environment variables and must not be written to artifacts.",
    },
    {
        "name": "human_approval_for_risk",
        "rule": "Network-heavy, destructive, or scientifically ambiguous actions must produce an approval or review gate.",
    },
]


TOOL_CALLING_TIPS: List[Dict[str, str]] = [
    {
        "tool": "astro_toolbox",
        "tip": "Run as deterministic data/model modules; capture module_status.csv and do not infer missing outputs.",
    },
    {
        "tool": "rag_pipeline",
        "tip": "Use method_only searches for modeling decisions and cite retrieved local bibcodes in the manuscript.",
    },
    {
        "tool": "prompt2graph KG",
        "tip": "Use KG hits for method transfer suggestions, not as direct evidence for target-specific parameters.",
    },
    {
        "tool": "LLM provider",
        "tip": "Use only for outline, prose drafting, and critique; all data checks remain deterministic.",
    },
    {
        "tool": "QA gate",
        "tip": "If any three-iteration rule fails, write the abnormal report before any final paper claim.",
    },
]


def guidance_payload() -> Dict[str, Any]:
    return {
        "source": "codex-main.zip distilled local operating practices",
        "rules": CODEX_STYLE_RULES,
        "tool_calling_tips": TOOL_CALLING_TIPS,
    }


def write_guidance(root: Path) -> str:
    return tools.json_dump(root / "codex_style_guidance.json", guidance_payload())
