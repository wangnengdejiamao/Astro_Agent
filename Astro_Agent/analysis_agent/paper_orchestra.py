"""Astronomy-specific PaperOrchestra adapter.

This module maps the original PaperOrchestra five-agent paper-writing pattern
onto the local astronomy workflow: astro_toolbox outputs, SQLite RAG, and the
white-dwarf knowledge graph.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import codex_style, paper_agents, tools
from .llm_client import LLMClient, load_model_config


ANTI_LEAKAGE = """Use only the supplied run artifacts, local RAG/KG evidence, and explicitly cited literature.
Do not import author names, institutions, result values, citations, or claims from memory.
Do not invent numerical physical parameters. If a quantity is absent or uncertified, write that it is unavailable pending human review.
"""


def _workspace(root: Path) -> Path:
    return tools.ensure_dir(root / "paper_orchestra")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _sha256(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _bib_key(row: Dict[str, Any], index: int) -> str:
    bibcode = str(row.get("bibcode") or "")
    key = re.sub(r"[^A-Za-z0-9]+", "", bibcode)
    return key or f"localref{index}"


def collect_rag_rows(rag_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for item in rag_results:
        for row in item.get("rows", []):
            bibcode = row.get("bibcode")
            if not bibcode or bibcode in seen:
                continue
            seen.add(bibcode)
            rows.append(row)
    return rows


def write_inputs(workspace: Path, state: Dict[str, Any]) -> Dict[str, str]:
    inputs = tools.ensure_dir(workspace / "inputs")
    tools.ensure_dir(inputs / "figures")
    data_fetch = state.get("data_fetch", {})
    output = data_fetch.get("existing_outputs", {})
    rag_rows = collect_rag_rows(state.get("rag_results", []))

    idea = f"""## Problem Statement
Analyze the astronomical source `{state.get('target')}` using a reproducible multi-wavelength workflow. The source coordinates are RA={state.get('ra_deg')} deg and Dec={state.get('dec_deg')} deg in ICRS.

## Core Hypothesis
The source can be scientifically interpreted only after catalog cross-matching, photometric/spectroscopic data retrieval, RAG/KG method selection, three modeling iterations, and a strict QA gate.

## Proposed Methodology (Detailed Technical Approach)
- Query SIMBAD and survey data through `astro_toolbox`.
- Use local SQLite RAG to retrieve white-dwarf and compact-binary methods.
- Use the local knowledge graph to transfer methods such as SED fitting, Bayesian inference, radial-velocity modeling, 6D phase-space analysis, and orbit traceback.
- Let the Structure Planner choose spectroscopy+SED or photometric HRD+SED fallback routes before fitting.
- Let the Method Scout investigate reusable or newer methods from local RAG/KG and optional LLM review.
- Enforce three modeling iterations: baseline, residual/physics review, and errors/systematics.
- Run Model Supervisor checks after fitting; unresolved repair actions keep final parameters blocked.
- Trigger human review before ApJ drafting when data are missing, modules fail, rare spectral behavior appears, or models fail to converge.

## Expected Contribution
An auditable ApJ-style analysis package that separates certified physical parameters from pending or anomalous results.

## Agent Operating Style
This workflow uses Codex-style bounded context, structured artifacts, deterministic tool calls, integration smoke tests, and review-first reporting.
"""

    experimental_log = {
        "target": state.get("target"),
        "coordinates": {"ra_deg": state.get("ra_deg"), "dec_deg": state.get("dec_deg"), "frame": "ICRS"},
        "qa": state.get("qa", {}),
        "analysis_plan": state.get("analysis_plan", {}),
        "method_scout": state.get("method_scout", {}),
        "model_supervision": state.get("model_supervision", {}),
        "iterations": state.get("iterations", []),
        "astrotool_summary": output,
        "simbad_crossmatch": data_fetch.get("simbad_crossmatch", {}),
        "kg_results": state.get("kg_results", [])[:12],
        "rag_bibcodes": [row.get("bibcode") for row in rag_rows[:30]],
        "codex_style": codex_style.guidance_payload(),
        "paper_agents": paper_agents.five_agent_manifest(),
    }
    log_text = "# Experimental Log\n\n"
    log_text += "## 1. Experimental Setup\n"
    log_text += "- Workflow: Astro Chief Investigator + astronomy PaperOrchestra adapter.\n"
    log_text += "- Data sources: SIMBAD, astro_toolbox survey modules, local RAG database, local knowledge graph.\n"
    log_text += "- QA policy: no final numerical claims unless the QA gate is clear.\n\n"
    log_text += "## 2. Raw Numeric Data\n"
    log_text += "```json\n" + json.dumps(experimental_log, ensure_ascii=False, indent=2) + "\n```\n\n"
    log_text += "## 3. Qualitative Observations\n"
    for iteration in state.get("iterations", []):
        log_text += f"- Iteration {iteration.get('iteration')} ({iteration.get('name')}) status: {iteration.get('status')}.\n"
    for reason in state.get("qa", {}).get("reasons", []):
        log_text += f"- QA hold reason: {reason}.\n"

    template = r"""\documentclass[twocolumn]{aastex631}
\shorttitle{Automated Astronomical Source Analysis}
\shortauthors{Chief Investigator Agent}
\begin{document}
\title{}
\author{Chief Investigator Agent}
\affiliation{Automated Astronomy Workflow Laboratory}
\begin{abstract}
\end{abstract}
\section{Introduction}
\section{Data}
\section{Methods}
\section{Results}
\section{Discussion}
\section{Conclusions}
\acknowledgments
\end{document}
"""

    guidelines = """# ApJ/AAS Guidelines for This Agent

- Use `aastex631` compatible LaTeX.
- Use ICRS coordinates in decimal degrees and state units for every numerical quantity.
- Preserve uncertainty language. Do not report a final physical parameter unless the QA gate is clear.
- Required sections: Abstract, Introduction, Data, Methods, Results, Discussion, Conclusions.
- Include a reproducibility paragraph naming local RAG, KG, and astro_toolbox artifacts.
- Cite local RAG papers where available.
"""

    paths = {
        "idea": tools.write_text(inputs / "idea.md", idea),
        "experimental_log": tools.write_text(inputs / "experimental_log.md", log_text),
        "template": tools.write_text(inputs / "template.tex", template),
        "guidelines": tools.write_text(inputs / "conference_guidelines.md", guidelines),
    }
    return paths


def build_outline(workspace: Path, state: Dict[str, Any], use_llm: bool, provider: Optional[str] = None) -> Dict[str, Any]:
    outline = {
        "plotting_plan": [
            {
                "figure_id": "fig_workflow_overview",
                "plot_type": "diagram",
                "data_source": "both",
                "aspect_ratio": "16:9",
                "purpose": "Show data fetching, RAG/KG retrieval, three modeling iterations, QA gate, and ApJ writing.",
            },
            {
                "figure_id": "fig_available_outputs",
                "plot_type": "plot",
                "data_source": "experimental_log.md",
                "aspect_ratio": "4:3",
                "purpose": "Summarize astro_toolbox module status and available artifacts.",
            },
        ],
        "intro_related_work_plan": {
            "introduction_strategy": "Use local RAG hits to frame white-dwarf/compact-binary multi-wavelength analysis.",
            "related_work_strategy": [
                "SED fitting with Gaia parallax priors",
                "spectroscopic classification and residual analysis",
                "radial-velocity, period, and 6D kinematic methods",
                "systematic uncertainty propagation",
            ],
        },
        "section_plan": [
            {"section_title": "Introduction", "content_bullets": ["Define target and scientific motivation from local evidence."]},
            {"section_title": "Data", "content_bullets": ["Report SIMBAD and astro_toolbox module status."]},
            {"section_title": "Methods", "content_bullets": ["Describe RAG/KG method transfer and the three-iteration rule."]},
            {"section_title": "Results", "content_bullets": ["Report only QA-certified results; otherwise state hold reasons."]},
            {"section_title": "Discussion", "content_bullets": ["Discuss model limitations and human-review triggers."]},
            {"section_title": "Conclusions", "content_bullets": ["Summarize certified findings and pending work."]},
        ],
    }
    if use_llm:
        outline = _llm_json(
            system=ANTI_LEAKAGE + "\nYou are the Outline Agent adapted for ApJ astronomy papers. Return strict JSON only.",
            user=_read(workspace / "inputs" / "idea.md") + "\n\n" + _read(workspace / "inputs" / "experimental_log.md"),
            fallback=outline,
            provider=provider,
        )
    tools.json_dump(workspace / "outline.json", outline)
    return outline


def write_literature(workspace: Path, state: Dict[str, Any]) -> Dict[str, str]:
    rag_rows = collect_rag_rows(state.get("rag_results", []))
    bib_entries = []
    intro_lines = [
        r"\section{Introduction}",
        "Multi-wavelength characterization of compact stellar remnants requires strict control of catalog identity, units, model priors, and systematic uncertainties.",
    ]
    for idx, row in enumerate(rag_rows[:30], start=1):
        key = _bib_key(row, idx)
        title = str(row.get("title") or "Untitled").replace("{", "").replace("}", "")
        year = row.get("year") or ""
        journal = row.get("journal") or ""
        bib_entries.append(
            f"@article{{{key},\n"
            f"  title = {{{title}}},\n"
            f"  year = {{{year}}},\n"
            f"  journal = {{{journal}}},\n"
            f"  note = {{{row.get('bibcode', '')}}}\n"
            f"}}\n"
        )
    if rag_rows:
        keys = ", ".join(r"\citep{" + _bib_key(row, idx) + "}" for idx, row in enumerate(rag_rows[:6], start=1))
        intro_lines.append(f"The local RAG database retrieved method examples relevant to this workflow, including {keys}.")
    intro_lines.append(r"\section{Related Work}")
    intro_lines.append("Relevant method families include SED fitting, spectroscopic classification, radial-velocity analysis, light-curve period searches, and systematic error propagation.")

    intro_path = tools.write_text(workspace / "drafts" / "intro_relwork.tex", "\n\n".join(intro_lines) + "\n")
    refs_path = tools.write_text(workspace / "refs.bib", "\n".join(bib_entries))
    citation_pool = {"papers": rag_rows[:30], "source": "local SQLite RAG"}
    pool_path = tools.json_dump(workspace / "citation_pool.json", citation_pool)
    return {"intro_relwork": intro_path, "refs": refs_path, "citation_pool": pool_path}


def write_figures(workspace: Path, state: Dict[str, Any]) -> Dict[str, str]:
    figures = tools.ensure_dir(workspace / "figures")
    captions = {
        "fig_workflow_overview": "Auditable workflow from source identity and survey retrieval through RAG/KG method selection, three modeling iterations, QA gate, and ApJ writing.",
        "fig_available_outputs": "Summary of local astro_toolbox module status and available artifacts for the target run.",
    }
    diagram = """digraph workflow {
  rankdir=LR;
  source -> data_fetcher -> rag_kg -> baseline -> residuals -> systematics -> qa_gate;
  qa_gate -> paper_orchestra [label="clear"];
  qa_gate -> human_review [label="hold"];
}
"""
    tools.write_text(figures / "fig_workflow_overview.dot", diagram)
    tools.json_dump(figures / "captions.json", captions)
    return {"captions": str(figures / "captions.json"), "workflow_dot": str(figures / "fig_workflow_overview.dot")}


# --------------------------------------------------------------------------- #
# Evidence packing helpers — used by section-wise writers and reviewer
# --------------------------------------------------------------------------- #


SECTION_EVIDENCE_QUERIES: Dict[str, List[str]] = {
    "Introduction": [
        "white dwarf compact binary multi-wavelength characterization motivation",
        "magnetic white dwarf cataclysmic variable population context",
    ],
    "Data": [
        "SDSS DESI LAMOST KOA spectroscopy reduction quality flags",
        "ZTF WISE Gaia photometry epoch survey cadence",
    ],
    "Methods": [
        "white dwarf SED fitting Bayesian inference Gaia parallax prior",
        "radial velocity orbit traceback 6D phase space binary",
        "cooling age Koester atmosphere grid systematic",
    ],
    "Results": [
        "fitted parameters uncertainty intervals systematic error budget",
    ],
    "Discussion": [
        "alternative interpretations subdwarf binary contamination accretion",
    ],
    "Conclusions": [
        "auditable reproducibility provenance summary",
    ],
}


def pack_section_evidence(state: Dict[str, Any], section: str, k: int = 4) -> str:
    """Build a compact evidence block for a single section's LLM call."""
    rag_rows = collect_rag_rows(state.get("rag_results", []))
    kg_rows = state.get("kg_results", []) or []
    queries = SECTION_EVIDENCE_QUERIES.get(section, [])
    rag_pick: List[Dict[str, Any]] = []
    if queries:
        for q in queries:
            try:
                rag_pick.extend(tools.search_rag(q, method_only=False, limit=k))
            except Exception:
                pass
    seen = set()
    rag_dedup: List[Dict[str, Any]] = []
    for row in (rag_pick or rag_rows)[: 3 * k]:
        bib = row.get("bibcode")
        if bib in seen:
            continue
        seen.add(bib)
        rag_dedup.append(row)
        if len(rag_dedup) >= k:
            break
    rag_block = tools.format_rag_bullets(rag_dedup, max_items=k) or "(no RAG hits)"
    kg_block = tools.format_kg_bullets(kg_rows, max_items=k) or "(no KG hits)"
    return (
        f"### Evidence for `{section}`\n\n"
        f"#### Local RAG bibcodes (cite as \\citep{{<bibkey>}} where appropriate)\n{rag_block}\n\n"
        f"#### KG method-transfer triples\n{kg_block}\n"
    )


SECTION_PROMPTS: Dict[str, str] = {
    "Abstract": "Write a single-paragraph ApJ abstract (150–250 words). State the target, what was retrieved, the QA status, and what is intentionally withheld. Do NOT invent numbers.",
    "Introduction": "Write the Introduction (3–5 paragraphs). Frame the science motivation using ONLY the supplied RAG bibcodes; cite using \\citep{<bibkey>} from refs.bib.",
    "Data": "Write the Data section. Enumerate which astro_toolbox modules ran and their statuses from experimental_log. Describe which surveys provided what.",
    "Methods": "Write the Methods section. Map the Structure Planner route, Method Scout evidence, three-iteration rule, and Model Supervisor checks. If spectra are absent, describe HRD+SED photometric fallback and its limits. Do NOT report numerical results here.",
    "Results": "Write the Results section. If QA gate is not 'clear', explicitly state that final numerical parameters are withheld and list hold reasons plus unresolved Model Supervisor actions. Otherwise, only quote numbers that exist in experimental_log.",
    "Discussion": "Write the Discussion. Discuss alternative interpretations, model limitations, and human-review triggers. Use RAG bibcodes for comparison.",
    "Conclusions": "Write the Conclusions (1–2 paragraphs). Summarize certified findings vs pending work, and reference the provenance manifest.",
}


def write_section(
    workspace: Path,
    state: Dict[str, Any],
    section: str,
    bibkeys: List[str],
    use_llm: bool,
    provider: Optional[str] = None,
) -> str:
    """Generate a single section as standalone LaTeX (no preamble)."""
    fallback_lines: Dict[str, str] = {
        "Abstract": rf"\begin{{abstract}}We present an automated, auditable analysis of {state.get('target')} at ICRS $\alpha={state.get('ra_deg')}$ deg, $\delta={state.get('dec_deg')}$ deg. Numerical results are withheld pending QA clearance.\end{{abstract}}",
        "Introduction": r"\section{Introduction}Multi-wavelength characterization of compact stellar remnants requires strict control of catalog identity, units, model priors, and systematic uncertainties.",
        "Data": r"\section{Data}The Data Fetcher recorded SIMBAD cross-matching and astro_toolbox module status; module statuses are machine-readable in the run directory.",
        "Methods": r"\section{Methods}The analysis follows the three-iteration rule (baseline, residuals, systematics). Method selection is guided by the local KG and RAG.",
        "Results": rf"\section{{Results}}QA gate status: \texttt{{{state.get('qa', {}).get('apj_gate', 'unknown')}}}. Final physical parameters are withheld unless QA is clear.",
        "Discussion": r"\section{Discussion}The workflow is intentionally conservative: failed convergence, rare spectral behavior, or missing uncertainty propagation triggers human review.",
        "Conclusions": r"\section{Conclusions}This package provides a reproducible trail from target identity through manuscript generation. Final interpretation depends on QA clearance.",
    }
    fallback = fallback_lines.get(section, f"\\section{{{section}}}")
    if not use_llm:
        return fallback

    evidence = pack_section_evidence(state, section)
    bib_hint = (
        "Available bib keys: " + ", ".join(bibkeys[:30])
        if bibkeys
        else "No bib keys are available; do NOT invent any."
    )
    system = (
        ANTI_LEAKAGE
        + "\nYou are the Section Writing Agent for an ApJ paper using aastex631."
        + " Return ONLY LaTeX for THIS section (no preamble, no \\begin{document}, no \\end{document})."
    )
    instr = SECTION_PROMPTS.get(section, f"Write the {section} section.")
    user = (
        f"Target: {state.get('target')}\n"
        f"Coordinates: RA={state.get('ra_deg')}, Dec={state.get('dec_deg')}\n"
        f"QA gate: {json.dumps(state.get('qa', {}))[:1500]}\n\n"
        f"{instr}\n\n{bib_hint}\n\n{evidence}"
    )
    return _llm_text(system=system, user=user, fallback=fallback, provider=provider)


def assemble_paper(workspace: Path, sections: Dict[str, str], state: Dict[str, Any]) -> str:
    target = state.get("target")
    head = (
        r"\documentclass[twocolumn]{aastex631}" + "\n"
        rf"\shorttitle{{Automated Analysis of {target}}}" + "\n"
        r"\shortauthors{Chief Investigator Agent}" + "\n"
        r"\begin{document}" + "\n"
        rf"\title{{An Auditable Multi-Wavelength Analysis of {target}}}" + "\n"
        r"\author{Chief Investigator Agent}" + "\n"
        r"\affiliation{Automated Astronomy Workflow Laboratory}" + "\n\n"
    )
    body = "\n\n".join(
        sections.get(name, "")
        for name in ["Abstract", "Introduction", "Data", "Methods", "Results", "Discussion", "Conclusions"]
    )
    tail = "\n\n" + r"\acknowledgments" + "\n" + "Generated from local astrotool, RAG, and KG artifacts.\n" + r"\bibliography{refs}" + "\n" + r"\end{document}" + "\n"
    return head + body + tail


def llm_review(state: Dict[str, Any], paper_tex: str, provider: Optional[str] = None) -> Dict[str, Any]:
    """Have the LLM act as a sharp ApJ reviewer; returns JSON scores + questions."""
    fallback = {
        "score": {"rigor": 3, "grounding": 3, "clarity": 3, "figures": 2, "overall": 60},
        "questions": reviewer_questions(state),
        "actions": ["Tighten language; ensure QA caveats are preserved."],
        "decision": "minor_revise",
    }
    system = (
        ANTI_LEAKAGE
        + "\nYou are a sharp ApJ peer reviewer. Return STRICT JSON with keys "
        + "score{rigor,grounding,clarity,figures,overall (0-100)}, questions[], actions[], decision in {accept,minor_revise,major_revise,reject}."
    )
    user = "Manuscript:\n\n" + paper_tex[:18000]
    return _llm_json(system=system, user=user, fallback=fallback, provider=provider)


# --------------------------------------------------------------------------- #
# Legacy single-shot writer (kept for backward compatibility)
# --------------------------------------------------------------------------- #


def write_section_draft(workspace: Path, state: Dict[str, Any], use_llm: bool, provider: Optional[str] = None) -> str:
    qa = state.get("qa", {})
    rag_rows = collect_rag_rows(state.get("rag_results", []))
    cite = ""
    if rag_rows:
        cite = r"\citep{" + ",".join(_bib_key(row, idx) for idx, row in enumerate(rag_rows[:5], start=1)) + "}"
    body = rf"""\documentclass[twocolumn]{{aastex631}}
\shorttitle{{Automated Analysis of {state.get('target')}}}
\shortauthors{{Chief Investigator Agent}}
\begin{{document}}
\title{{An Auditable Multi-Wavelength Analysis of {state.get('target')}}}
\author{{Chief Investigator Agent}}
\affiliation{{Automated Astronomy Workflow Laboratory}}

\begin{{abstract}}
We present an automated, auditable analysis workflow for {state.get('target')} at ICRS coordinates
$\alpha={state.get('ra_deg')}$ deg and $\delta={state.get('dec_deg')}$ deg. The workflow couples
survey data retrieval, local white-dwarf literature retrieval, knowledge-graph method transfer,
three mandatory modeling iterations, and a strict QA gate before ApJ-style manuscript drafting.
\end{{abstract}}

\section{{Introduction}}
{_read(workspace / "drafts" / "intro_relwork.tex")}

\section{{Data}}
The Data Fetcher stage records SIMBAD cross-matching and astrotool outputs under the run directory.
All coordinates are handled in ICRS decimal degrees. Module statuses are machine-readable in the
workflow artifacts.

\section{{Methods}}
The analysis follows the adapted PaperOrchestra plan: outline construction, figure and literature
assembly, single-pass section writing, and bounded refinement. The scientific modeling stage is
constrained by the three-iteration rule: baseline fitting, residual and physical-plausibility review,
and statistical plus systematic uncertainty review. Local RAG/KG evidence guides method selection {cite}.

\section{{Results}}
QA gate status: \texttt{{{qa.get('apj_gate', 'unknown')}}}. Final numerical physical parameters are not
reported unless the QA gate is clear. Current hold reasons are: {", ".join(qa.get('reasons', [])) or "none"}.

\section{{Discussion}}
The workflow is intentionally conservative. Strong emission lines, failed convergence, implausible
cooling ages, missing uncertainty propagation, or unresolved module failures trigger human review.

\section{{Conclusions}}
This package provides a reproducible trail from target identity through data acquisition, method
retrieval, modeling checks, and manuscript generation. The final scientific interpretation remains
conditional on the QA gate and human review where triggered.

\acknowledgments
This manuscript was generated from local astrotool, RAG, and knowledge-graph artifacts.
\bibliography{{../refs}}
\end{{document}}
"""
    if use_llm:
        body = _llm_text(
            system=ANTI_LEAKAGE + "\nYou are the Section Writing Agent for ApJ astronomy. Return complete aastex631 LaTeX only.",
            user=_read(workspace / "inputs" / "idea.md") + "\n\n" + _read(workspace / "inputs" / "experimental_log.md") + "\n\nDraft seed:\n" + body,
            fallback=body,
            provider=provider,
        )
    return tools.write_text(workspace / "drafts" / "paper.tex", body)


def refine(
    workspace: Path,
    state: Dict[str, Any],
    use_llm: bool,
    provider: Optional[str] = None,
    target_score: int = 80,
    max_iters: int = 3,
) -> Dict[str, Any]:
    """Real review-driven refinement: LLM reviewer scores; below target → rewrite weakest section."""
    best = _read(workspace / "drafts" / "paper.tex")
    worklog: Dict[str, Any] = {"iterations": [], "best_iter": 0, "target_score": target_score}

    rag_rows = collect_rag_rows(state.get("rag_results", []))
    bibkeys = [_bib_key(row, idx) for idx, row in enumerate(rag_rows[:30], start=1)]

    last_review: Dict[str, Any] = {}
    for index in range(1, max_iters + 1):
        review = llm_review(state, best, provider=provider) if use_llm else {
            "score": {"overall": 75, "rigor": 3, "grounding": 3, "clarity": 3, "figures": 2},
            "questions": reviewer_questions(state),
            "actions": [],
            "decision": "minor_revise",
        }
        last_review = review
        overall = int(review.get("score", {}).get("overall", 0) or 0)
        accepted = overall >= target_score or review.get("decision") == "accept"

        # If not accepted and use_llm, ask the LLM to rewrite the weakest section.
        rewrite_target: Optional[str] = None
        if use_llm and not accepted:
            scores = review.get("score", {}) or {}
            axis = min(
                ("grounding", "rigor", "clarity", "figures"),
                key=lambda k: scores.get(k, 5),
            )
            axis_to_section = {
                "grounding": "Methods",
                "rigor": "Results",
                "clarity": "Discussion",
                "figures": "Data",
            }
            rewrite_target = axis_to_section.get(axis, "Discussion")
            new_section = write_section(workspace, state, rewrite_target, bibkeys, use_llm=True, provider=provider)
            best = _replace_section(best, rewrite_target, new_section)

        iter_dir = tools.ensure_dir(workspace / "refinement" / f"iter{index}")
        tools.write_text(iter_dir / "paper.tex", best)
        tools.json_dump(iter_dir / "review.json", review)
        worklog["iterations"].append(
            {
                "iter": index,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "review": review,
                "rewrote_section": rewrite_target,
                "decision": "accept" if accepted else "revise",
            }
        )
        worklog["best_iter"] = index
        if accepted:
            worklog["halted_because"] = f"score_{overall}_>=_{target_score}"
            break
    else:
        worklog["halted_because"] = "iteration_cap_reached"

    final_dir = tools.ensure_dir(workspace / "final")
    final_tex = tools.write_text(final_dir / "paper.tex", best)
    worklog_path = tools.json_dump(workspace / "refinement" / "worklog.json", worklog)
    return {
        "final_tex": final_tex,
        "worklog": worklog_path,
        "worklog_data": worklog,
        "final_review": last_review,
    }


def _replace_section(tex: str, section_name: str, new_body: str) -> str:
    """Replace `\\section{Name} ... up to next \\section or \\acknowledgments` with new_body."""
    pattern = re.compile(
        r"\\section\{" + re.escape(section_name) + r"\}.*?(?=\\section\{|\\acknowledgments|\\end\{document\})",
        re.DOTALL,
    )
    body = new_body if new_body.strip().startswith("\\section") else f"\\section{{{section_name}}}\n{new_body}\n\n"
    if not body.endswith("\n\n"):
        body = body + "\n\n"
    # Use a replacement function so LaTeX backslashes from LLM output are
    # inserted literally instead of being parsed as regex replacement escapes.
    new_tex, n = pattern.subn(lambda _match: body, tex, count=1)
    return new_tex if n else tex + "\n\n" + body


def reviewer_questions(state: Dict[str, Any]) -> List[str]:
    return [
        "Do the selected atmosphere models cover the target's Teff/logg/composition regime, or is the fit pinned to a grid boundary?",
        "Are light-curve periods robust against survey cadence aliases and independently supported by another band?",
        "Are Gaia parallax uncertainty, extinction, flux calibration, and model-grid systematics included in the confidence intervals?",
        "If emission lines, X-ray detections, or IR excess are present, why is a single-WD interpretation still preferred?",
    ]


def provenance(workspace: Path) -> str:
    files = [
        workspace / "inputs" / "idea.md",
        workspace / "inputs" / "experimental_log.md",
        workspace / "inputs" / "template.tex",
        workspace / "inputs" / "conference_guidelines.md",
        workspace / "outline.json",
        workspace / "refs.bib",
        workspace / "final" / "paper.tex",
    ]
    data = {"created_at": datetime.utcnow().isoformat(timespec="seconds"), "files": {}}
    for path in files:
        if path.exists():
            data["files"][str(path.relative_to(workspace))] = _sha256(path)
    return tools.json_dump(workspace / "provenance.json", data)


def run_astro_paper_orchestra(
    root: Path,
    state: Dict[str, Any],
    use_llm: bool = False,
    provider: Optional[str] = None,
    sectionwise: bool = True,
    target_score: int = 80,
    max_refine_iters: int = 3,
) -> Dict[str, Any]:
    """Run the full astronomy PaperOrchestra adapter.

    sectionwise=True (default) uses the new per-section LLM writer + LLM-reviewer
    refinement loop. sectionwise=False falls back to the legacy single-shot writer.
    """
    workspace = _workspace(root)
    tools.ensure_dir(workspace / "drafts")
    tools.ensure_dir(workspace / "refinement")
    guidance = codex_style.write_guidance(workspace)
    agent_manifest = tools.json_dump(workspace / "agents_manifest.json", paper_agents.five_agent_manifest())
    inputs = write_inputs(workspace, state)
    outline = build_outline(workspace, state, use_llm=use_llm, provider=provider)
    literature = write_literature(workspace, state)
    figures = write_figures(workspace, state)

    if sectionwise:
        rag_rows = collect_rag_rows(state.get("rag_results", []))
        bibkeys = [_bib_key(row, idx) for idx, row in enumerate(rag_rows[:30], start=1)]
        sections: Dict[str, str] = {}
        for name in ["Abstract", "Introduction", "Data", "Methods", "Results", "Discussion", "Conclusions"]:
            sections[name] = write_section(workspace, state, name, bibkeys, use_llm=use_llm, provider=provider)
            tools.write_text(workspace / "drafts" / f"section_{name.lower()}.tex", sections[name])
        body = assemble_paper(workspace, sections, state)
        draft = tools.write_text(workspace / "drafts" / "paper.tex", body)
    else:
        draft = write_section_draft(workspace, state, use_llm=use_llm, provider=provider)

    refinement = refine(workspace, state, use_llm=use_llm, provider=provider, target_score=target_score, max_iters=max_refine_iters)
    prov = provenance(workspace)
    return {
        "status": "written",
        "workspace": str(workspace),
        "inputs": inputs,
        "codex_style_guidance": guidance,
        "agents_manifest": agent_manifest,
        "outline": str(workspace / "outline.json"),
        "literature": literature,
        "figures": figures,
        "draft": draft,
        "final_tex": refinement["final_tex"],
        "worklog": refinement["worklog"],
        "final_review": refinement.get("final_review", {}),
        "provenance": prov,
        "used_llm": use_llm,
        "provider": provider,
        "sectionwise": sectionwise,
    }


def _llm_client(provider: Optional[str]) -> LLMClient:
    return LLMClient(load_model_config(provider)) if provider else LLMClient()


def _llm_json(system: str, user: str, fallback: Dict[str, Any], provider: Optional[str] = None) -> Dict[str, Any]:
    try:
        text = _llm_client(provider).complete(system=system, user=user, temperature=0.1, max_output_tokens=5000)
        return json.loads(_strip_fence(text))
    except Exception as exc:
        fallback = dict(fallback)
        fallback["llm_fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return fallback


def _llm_text(system: str, user: str, fallback: str, provider: Optional[str] = None) -> str:
    try:
        text = _llm_client(provider).complete(system=system, user=user, temperature=0.2, max_output_tokens=9000)
        return _strip_fence(text) or fallback
    except Exception:
        return fallback


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text
