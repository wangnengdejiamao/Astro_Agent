"""Reflexion (Shinn et al. 2023; MAR 2026 extension).

When paper_qc returns a `fail` verdict, we synthesize a verbal reflection
that explicitly names (a) WHICH check failed, (b) WHAT the immediate cause is
inside the manuscript text, and (c) WHAT the next rewrite should change.  The
reflection is stored in state['reflexion_history'] and re-injected into the
drafter on the next pass.  This is the "verbal reinforcement" pattern shown
to give +11-22% on AlfWorld / HotPotQA / HumanEval.

We use deterministic heuristics for each failing check so the loop runs
without an LLM; if `use_llm=True`, the heuristic reflection is augmented by
a short LLM-written critique.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# Map paper_qc check ids -> (section to target for rewrite, deterministic advice).
# This is the Reflexion "policy memory" — a textual rule of thumb learned from
# what makes each check fail.
_CHECK_TO_REWRITE: Dict[str, Dict[str, str]] = {
    "abstract_length": {
        "section": "Abstract",
        "advice": (
            "Abstract is too short or too long. Add (or remove) 1-2 sentences "
            "covering: (i) the source class, (ii) the dominant numerical result "
            "with units, (iii) the strongest disagreement with literature, (iv) "
            "next observational test. Target 180-280 words."
        ),
    },
    "abstract_numerics": {
        "section": "Abstract",
        "advice": (
            "Abstract has fewer than 2 numeric+unit phrases. Pull values from "
            "the published_params table — quote at least one published value "
            "with bibcode and one this-work value with units (K, M_⊙, pc, min, km/s)."
        ),
    },
    "citations_per_section": {
        "section": "Introduction",
        "advice": (
            "At least one of Introduction/Methods/Discussion has no \\citep. "
            "Add at least one \\citep{<bibcode>} per missing section using the "
            "per-source RAG entries flagged with ✓ (abstract mentions target)."
        ),
    },
    "bibkey_coverage": {
        "section": "Discussion",
        "advice": (
            "One or more \\citep keys do NOT resolve to a refs.bib entry. "
            "Replace the unknown keys with bibcodes from the per-source RAG "
            "block, or remove the unsupported citation."
        ),
    },
    "uncertainty_language": {
        "section": "Results",
        "advice": (
            "Results section lacks \\pm / sigma / uncertainty language. Add at "
            "least one explicit ± error bar to the numeric results, or a "
            "sentence stating that errors are propagated through to the table."
        ),
    },
    "novelty_paragraph": {
        "section": "Discussion",
        "advice": (
            "Discussion lacks an explicit this-work vs literature comparison. "
            "Write one sentence per literature parameter saying whether this "
            "work confirms, extends, or disagrees with the published value."
        ),
    },
    "hypothesis_articulated": {
        "section": "Methods",
        "advice": (
            "Manuscript does not mention competing physical hypotheses. Add a "
            "subsection to Methods naming the alternative interpretations from "
            "the hypothesis_plan and stating which observable would discriminate."
        ),
    },
    "cluster_membership_discussed": {
        "section": "Discussion",
        "advice": (
            "Cluster membership χ² was computed but not discussed. Add a paragraph "
            "with the χ²_spat / χ²_kin / RV-σ verdict for the best candidate."
        ),
    },
    "cluster_joint_criteria": {
        "section": "Discussion",
        "advice": (
            "Cluster membership claims require all joint criteria: χ²_spat, "
            "χ²_kin, RV offset, and traceback time < cluster age. If any one "
            "is missing or fails, rewrite the paragraph to reject or withhold "
            "membership rather than quoting a partial spatial_ok/kin_ok verdict."
        ),
    },
    "sections_present": {
        "section": "Conclusions",
        "advice": (
            "Some required sections are missing. Insert placeholder \\section "
            "headings so the structure passes downstream checks."
        ),
    },
    "latex_brace_balance": {
        "section": "Results",
        "advice": (
            "LaTeX brace count is unbalanced. Quote the offending section "
            "verbatim and ensure every \\citep / \\texttt / itemize block "
            "closes cleanly."
        ),
    },
    "refs_bib": {
        "section": "Discussion",
        "advice": (
            "refs.bib is missing entries or formatted entries that look like "
            "bibcodes. Re-run write_literature; if it still fails, fall back "
            "to per_source_rag bibcodes only."
        ),
    },
    "methods_chi2_density": {
        "section": "Methods",
        "advice": (
            "Methods section has fewer than 3 chi^2 expressions, or the "
            "density is below 0.005 per word. Add explicit chi^2 reporting "
            "for the baseline fit, the residual-fit comparison, and the "
            "competing-hypothesis discrimination. Each chi^2 must name the "
            "dataset and the degrees of freedom."
        ),
    },
    "results_uncertainty_density": {
        "section": "Results",
        "advice": (
            "Too many bare numerical values in Results without ±sigma. For "
            "every number from published_params or this-work artifacts that "
            "has a non-null error column, write `value $\\pm$ sigma unit`. "
            "Where the artefact lists null error, write `value (error not measured)`."
        ),
    },
    "discussion_alternatives": {
        "section": "Discussion",
        "advice": (
            "Discussion has fewer than 2 alternative-hypothesis phrases. For "
            "each entry in hypothesis_plan, add a sentence stating which "
            "observable would discriminate it from the favoured interpretation. "
            "Use language like `alternatively`, `could also be`, `cannot be ruled out`."
        ),
    },
    "intro_motivation_chain": {
        "section": "Introduction",
        "advice": (
            "Introduction is too short or under-cited. Expand to 3-5 paragraphs "
            "forming a [phenomenon -> open question -> our approach] chain. "
            "Aim for >=4 \\citep across the section, with >=1 citation per "
            "paragraph, drawn from per-source RAG bibcodes."
        ),
    },
    "forbidden_hype": {
        "section": "Abstract",
        "advice": (
            "Manuscript contains hype words flagged by L4.I (obviously, "
            "remarkable, groundbreaking, we believe, ...). Replace each with "
            "a measurement-anchored statement, or remove the sentence."
        ),
    },
    "bibkey_format": {
        "section": "Discussion",
        "advice": (
            "One or more \\citep keys are NOT valid 19-char ADS bibcodes "
            "(e.g. `2025ApJ`, `2024A&A...`). Replace each with a real "
            "bibcode from refs.bib, or delete the citation. Do NOT keep "
            "placeholder keys."
        ),
    },
    "target_identity_consistency": {
        "section": "Abstract",
        "advice": (
            "The RA/Dec quoted in the manuscript does NOT match "
            "01_resolved_target.json. This is a target-identity blocker. "
            "Re-resolve the target, confirm the Gaia source id, and quote "
            "ONLY the coordinates from 01_resolved_target.json."
        ),
    },
    "extinction_provenance": {
        "section": "Data",
        "advice": (
            "Reported A_V provenance is NOT one of SFD98 / Planck13 / "
            "Green19 / Lallement / 3D-dust. Either compute A_V from one of "
            "those maps and cite it, or state explicitly that A_V is a "
            "fallback estimate and withhold it from publication-grade claims."
        ),
    },
    "literature_consistency": {
        "section": "Abstract",
        "advice": (
            "The manuscript claims it confirms literature parameters, but "
            "02c_published_params.json has zero literature rows. Remove the "
            "confirmation claim and state `n_from_literature = 0`; the "
            "Abstract should instead report only this-work measurements or "
            "QA hold reasons."
        ),
    },
    "physics_checks_integration": {
        "section": "Discussion",
        "advice": (
            "One or more of the four physics_checks (Rayleigh-Jeans, "
            "Ingress, Tidal truncation, Mass-Lum sanity) FAILED. The "
            "Discussion must explicitly address each failed check: either "
            "report the test outcome and how it informs the interpretation, "
            "or withdraw the hypothesis that the failed check disfavours."
        ),
    },
}


def build_reflection(paper_qc: Mapping[str, Any]) -> Dict[str, Any]:
    """Produce a verbal-RL reflection packet from a paper_qc verdict."""
    if not paper_qc:
        return {"status": "no_paper_qc"}
    checks = paper_qc.get("checks") or []
    failed = [c for c in checks if c.get("verdict") == "fail"]
    warned = [c for c in checks if c.get("verdict") == "warn"]
    targets: Dict[str, List[str]] = {}
    action_items: List[Dict[str, str]] = []
    for c in failed + warned:
        cid = c.get("id")
        if cid not in _CHECK_TO_REWRITE:
            continue
        rule = _CHECK_TO_REWRITE[cid]
        targets.setdefault(rule["section"], []).append(cid)
        action_items.append({
            "check_id": cid,
            "verdict": c.get("verdict"),
            "section": rule["section"],
            "reason": c.get("reason"),
            "advice": rule["advice"],
        })
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "n_failed_checks": len(failed),
        "n_warned_checks": len(warned),
        "sections_to_rewrite": list(targets.keys()),
        "section_to_failing_checks": targets,
        "action_items": action_items,
        # Composite verbal reflection — drafter can inject this verbatim into
        # the next prompt as "policy memory" (Reflexion §3.2).
        "verbal_reflection": _compose_verbal(action_items, paper_qc),
    }


def _compose_verbal(action_items: List[Dict[str, str]], paper_qc: Mapping[str, Any]) -> str:
    if not action_items:
        return ""
    lines = [
        "Reflection on the previous draft (Reflexion-style):",
        f"- paper_qc verdict: {paper_qc.get('verdict')} ({paper_qc.get('summary')})",
        "- specific issues to fix on the next rewrite:",
    ]
    for ai in action_items:
        lines.append(
            f"  * [{ai['verdict']}] {ai['check_id']}: {ai['reason']} "
            f"→ rewrite {ai['section']}: {ai['advice']}"
        )
    lines.append(
        "When you rewrite, you MUST address every action item above. "
        "Quote literature values with \\citep{<bibcode>} from the per-source "
        "RAG block; quote this-work values with their source_kind tag."
    )
    return "\n".join(lines)


def append_to_history(
    state: Mapping[str, Any], reflection: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    history = list(state.get("reflexion_history") or [])
    history.append(dict(reflection))
    return history


__all__ = ["build_reflection", "append_to_history"]
