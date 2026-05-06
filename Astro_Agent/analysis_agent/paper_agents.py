"""Manifest for the five astronomy PaperOrchestra sub-agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PaperAgentSpec:
    name: str
    role: str
    inputs: List[str]
    outputs: List[str]
    hard_rules: List[str]


def five_agent_manifest() -> Dict[str, object]:
    agents = [
        PaperAgentSpec(
            name="Outline Agent",
            role="Build strict manuscript, figure, literature, and section plan from local run artifacts.",
            inputs=["idea.md", "experimental_log.md", "template.tex", "conference_guidelines.md"],
            outputs=["outline.json"],
            hard_rules=[
                "No invented science claims.",
                "Every planned section must map to local artifacts or RAG/KG evidence.",
            ],
        ),
        PaperAgentSpec(
            name="Plotting Agent",
            role="Prepare figure plan, captions, and reusable diagram/plot artifacts.",
            inputs=["outline.json", "astrotool output files", "module_status.csv"],
            outputs=["figures/captions.json", "figures/*.dot or figures/*.png"],
            hard_rules=[
                "Prefer existing astrotool figures when present.",
                "If a figure cannot be rendered, emit a placeholder plan instead of fabricating data.",
            ],
        ),
        PaperAgentSpec(
            name="Literature Review Agent",
            role="Convert local RAG hits into refs.bib and Introduction/Related Work evidence.",
            inputs=["rag_results.json", "kg_results.json", "references.bib when available"],
            outputs=["refs.bib", "citation_pool.json", "drafts/intro_relwork.tex"],
            hard_rules=[
                "Use verified local bibcodes where available.",
                "Do not guess authors or citations.",
            ],
        ),
        PaperAgentSpec(
            name="Section Writing Agent",
            role="Draft a complete ApJ-compatible manuscript from outline, evidence, figures, and QA state.",
            inputs=["outline.json", "experimental_log.md", "intro_relwork.tex", "refs.bib", "captions.json"],
            outputs=["drafts/paper.tex"],
            hard_rules=[
                "One coherent manuscript pass, not disconnected section fragments.",
                "Preserve QA caveats and units for all values.",
            ],
        ),
        PaperAgentSpec(
            name="Content Refinement Agent",
            role="Run bounded reviewer/reviser iterations and accept or revert by explicit halt rules.",
            inputs=["drafts/paper.tex", "peer review questions", "QA gate"],
            outputs=["refinement/worklog.json", "final/paper.tex"],
            hard_rules=[
                "Maximum three iterations.",
                "Revert if rigor or factual grounding worsens.",
                "Never remove human-review caveats unless the QA gate clears.",
            ],
        ),
    ]
    return {
        "framework": "Astronomy-adapted PaperOrchestra",
        "agents": [asdict(agent) for agent in agents],
        "parallelizable": ["Plotting Agent", "Literature Review Agent"],
    }
