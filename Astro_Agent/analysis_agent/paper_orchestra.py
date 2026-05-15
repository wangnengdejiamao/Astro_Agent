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

import math

from . import codex_style, hypothesis_scaffold as hyp_module, paper_agents, per_source_rag as src_rag, prompt_experiment_log, published_params as pp_module, specialists, tools
from .llm_client import LLMClient, load_model_config
from .prompts import wd_domain


ANTI_LEAKAGE = """Use only the supplied run artifacts, local RAG/KG evidence, and the structured published_params table.
- You MAY quote a literature value if and only if it appears in the published_params block with a bibcode; cite it as \\citep{<bibcode>}.
- You MAY quote a this-work measurement if and only if it appears with source_kind starting with `this_work`; cite the artifact path.
- Do NOT import any other author names, institutions, or values from memory.
- Do NOT fabricate uncertainties. If error is null in the table, write the value without an error bar.
- If a quantity is absent from both the published_params table AND the run artifacts, state it is unavailable pending human review.
"""


def _workspace(root: Path) -> Path:
    return tools.ensure_dir(root / "paper_orchestra")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _sha256(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _is_valid_ads_bibcode(key: str) -> bool:
    """Strict ADS bibcode shape: 19 chars, starts with 4 digits, regex
    `^\\d{4}[A-Za-z&\\.]+\\.+\\S+$`. Matches `2020ApJ...905...32B`,
    `2018A&A...616A...1G`; rejects `2025ApJ` (7 chars), `2024A&A...` (no
    volume/page), `2025ApJ...UPK13c2L` (placeholder-shape, fewer chars
    than 19)."""
    import re as _re
    if not key or len(key) != 19:
        return False
    return bool(_re.match(r"^\d{4}[A-Za-z&\.]+\.+\S+$", key))


# Module-level audit log: every key rejected by _bib_key. Cleared each
# time write_literature() runs so we get a per-run rejection record.
_REJECTED_BIBKEYS: List[Dict[str, Any]] = []


def _bib_key(row: Dict[str, Any], index: int) -> str:
    """Return a BibTeX key for a row. D5 — only accept a key that passes
    the 19-char ADS bibcode shape test; otherwise fall back to
    `localref<index>` and record the rejection in _REJECTED_BIBKEYS so
    write_literature() can emit refs_rejected.json for audit.

    NB: the natbib package would accept other shapes, but Codex flagged
    that the LLM was emitting `\\citep{2025ApJ}` and similar placeholders
    that downstream readers cannot resolve in ADS.  Restricting the bib
    output to ADS-shaped keys forces the failure to surface at QC time.
    """
    bibcode = str(row.get("bibcode") or "").strip()
    if bibcode and _is_valid_ads_bibcode(bibcode):
        return bibcode
    if bibcode:
        _REJECTED_BIBKEYS.append({
            "rejected_key": bibcode,
            "index": index,
            "row_parameter": row.get("parameter"),
            "row_source_kind": row.get("source_kind"),
        })
    return f"localref{index}"


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

    # Inject the published_params table so drafter sees BOTH literature and
    # this-work measurements as concrete evidence (not just QA gate boilerplate).
    pp_table = state.get("published_params") or {"rows": []}
    if pp_table.get("rows"):
        log_text += "\n## 4. Published Parameters and This-Work Measurements\n"
        log_text += pp_module.render_markdown(pp_table)

    # Hypothesis test plan — drafter must mention competing interpretations.
    hp = state.get("hypothesis_plan") or {}
    if hp.get("hypotheses"):
        log_text += "\n## 5. Hypothesis Test Plan\n"
        log_text += hyp_module.render_markdown(hp)

    # Cluster membership chi^2 (Hunt+2023)
    cm = state.get("cluster_membership") or {}
    if cm.get("status") == "ok" and cm.get("candidates"):
        log_text += "\n## 6. Cluster Membership Diagnostics\n"
        for cand in cm["candidates"][:5]:
            log_text += (
                f"- {cand.get('name')}: chi2_spat={cand.get('chi2_spat')}, "
                f"chi2_kin={cand.get('chi2_kin')}, "
                f"rv_offset_sigma={cand.get('rv_offset_sigma')}, "
                f"age_myr={cand.get('cluster_age_myr')}\n"
            )

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
    if pp_table.get("rows"):
        paths["published_params"] = tools.write_text(
            inputs / "published_params.md",
            pp_module.render_markdown(pp_table, max_rows_per_param=20),
        )
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
            system=wd_domain.system_for_role("outline") + "\n" + wd_domain.OUTLINE_TASK,
            user=_read(workspace / "inputs" / "idea.md") + "\n\n" + _read(workspace / "inputs" / "experimental_log.md"),
            fallback=outline,
            provider=provider,
        )
    tools.json_dump(workspace / "outline.json", outline)
    return outline


def write_literature(workspace: Path, state: Dict[str, Any]) -> Dict[str, str]:
    rag_rows = collect_rag_rows(state.get("rag_results", []))
    bib_entries: List[str] = []
    written_keys: set = set()
    # D5 — clear per-run rejection log so audit_path reflects only this run.
    _REJECTED_BIBKEYS.clear()
    intro_lines = [
        r"\section{Introduction}",
        "Multi-wavelength characterization of compact stellar remnants requires strict control of catalog identity, units, model priors, and systematic uncertainties.",
    ]
    for idx, row in enumerate(rag_rows[:30], start=1):
        key = _bib_key(row, idx)
        if key in written_keys:
            continue
        written_keys.add(key)
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

    # Also write bib entries for every bibcode that appears in the
    # published_params table — drafter cites these directly as \citep{<bibcode>}.
    pp_table = state.get("published_params") or {}
    for row in pp_table.get("rows", []):
        bib = row.get("bibcode")
        if not bib or bib in written_keys:
            continue
        if not _is_valid_ads_bibcode(bib):
            _REJECTED_BIBKEYS.append({"rejected_key": bib, "source": "published_params"})
            continue
        # AAS-style \citep keys can include & and . — keep the literal bibcode.
        written_keys.add(bib)
        bib_entries.append(
            f"@article{{{bib},\n"
            f"  title = {{Reference for {row.get('parameter')} (mined from SIMBAD abstract)}},\n"
            f"  year = {{{bib[:4] if len(bib) >= 4 and bib[:4].isdigit() else ''}}},\n"
            f"  note = {{{bib}}}\n"
            f"}}\n"
        )

    # Also write bib stubs for any bibcode referenced by the hypothesis
    # scaffold (e.g. Kupfer 2020 prototype refs).
    hp = state.get("hypothesis_plan") or {}
    for h in hp.get("hypotheses", []) or []:
        for bibcode in h.get("references_bibcodes", []) or []:
            if not bibcode or bibcode in written_keys:
                continue
            if not _is_valid_ads_bibcode(bibcode):
                _REJECTED_BIBKEYS.append({"rejected_key": bibcode, "source": "hypothesis_plan"})
                continue
            written_keys.add(bibcode)
            year = bibcode[:4] if len(bibcode) >= 4 and bibcode[:4].isdigit() else ""
            bib_entries.append(
                f"@article{{{bibcode},\n"
                f"  title = {{Reference for hypothesis `{h.get('name')}'}},\n"
                f"  year = {{{year}}},\n"
                f"  note = {{{bibcode}}}\n"
                f"}}\n"
            )

    # Also pick up bibcodes from the comparison-table benchmark systems so
    # \citep{...} keys in the auto-generated deluxetable resolve.
    cmp = state.get("comparison_table") or {}
    for bibcode in cmp.get("bibcodes") or []:
        if not bibcode or bibcode in written_keys:
            continue
        if not _is_valid_ads_bibcode(bibcode):
            _REJECTED_BIBKEYS.append({"rejected_key": bibcode, "source": "comparison_table"})
            continue
        written_keys.add(bibcode)
        year = bibcode[:4] if len(bibcode) >= 4 and bibcode[:4].isdigit() else ""
        bib_entries.append(
            f"@article{{{bibcode},\n"
            f"  title = {{Benchmark-system reference for {cmp.get('source_class')}}},\n"
            f"  year = {{{year}}},\n"
            f"  note = {{{bibcode}}}\n"
            f"}}\n"
        )

    # Also include every per-source RAG bibcode so the drafter can cite any
    # paper from the SIMBAD reference list, not only those that scored a
    # parameter extraction. This is critical for Introduction/Discussion
    # citations that frame the source rather than quote a single value.
    sr_info = state.get("source_rag") or {}
    sr_path = sr_info.get("sqlite_path")
    if sr_path and Path(sr_path).exists():
        try:
            import sqlite3 as _sq
            conn = _sq.connect(str(sr_path))
            for bibcode, year, title, journal in conn.execute(
                "SELECT bibcode, year, title, journal FROM source_refs"
            ):
                if not bibcode or bibcode in written_keys:
                    continue
                if not _is_valid_ads_bibcode(bibcode):
                    _REJECTED_BIBKEYS.append({"rejected_key": bibcode, "source": "source_rag"})
                    continue
                written_keys.add(bibcode)
                safe_title = str(title or "Untitled").replace("{", "").replace("}", "")
                bib_entries.append(
                    f"@article{{{bibcode},\n"
                    f"  title = {{{safe_title}}},\n"
                    f"  year = {{{year or ''}}},\n"
                    f"  journal = {{{(journal or '').replace('{', '').replace('}', '')}}},\n"
                    f"  note = {{{bibcode}}}\n"
                    f"}}\n"
                )
            conn.close()
        except Exception:
            pass

    if rag_rows:
        keys = ", ".join(r"\citep{" + _bib_key(row, idx) + "}" for idx, row in enumerate(rag_rows[:6], start=1))
        intro_lines.append(f"The local RAG database retrieved method examples relevant to this workflow, including {keys}.")
    intro_lines.append(r"\section{Related Work}")
    intro_lines.append("Relevant method families include SED fitting, spectroscopic classification, radial-velocity analysis, light-curve period searches, and systematic error propagation.")

    intro_path = tools.write_text(workspace / "drafts" / "intro_relwork.tex", "\n\n".join(intro_lines) + "\n")
    refs_path = tools.write_text(workspace / "refs.bib", "\n".join(bib_entries))
    # D5 — publish allowlist on state so the drafter prompt and structural
    # lint can see exactly which keys are legitimate.
    state["bibkey_allowlist"] = sorted(written_keys)
    rejected_path = tools.json_dump(workspace / "refs_rejected.json", {
        "n_rejected": len(_REJECTED_BIBKEYS),
        "rejections": list(_REJECTED_BIBKEYS),
        "n_allowed": len(written_keys),
        "allowlist": sorted(written_keys),
    })
    citation_pool = {
        "papers": rag_rows[:30],
        "published_params_bibcodes": sorted({r.get("bibcode") for r in pp_table.get("rows", []) if r.get("bibcode")}),
        "source": "local SQLite RAG + SIMBAD-mined published_params",
    }
    pool_path = tools.json_dump(workspace / "citation_pool.json", citation_pool)
    return {"intro_relwork": intro_path, "refs": refs_path,
            "citation_pool": pool_path, "refs_rejected": rejected_path}


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
    # Published-parameters table: the drafter's main grounded evidence.
    pp_table = state.get("published_params") or {"rows": []}
    pp_block = pp_module.render_markdown(pp_table, max_rows_per_param=8) if pp_table.get("rows") else "(no published_params rows; do NOT invent values)"

    # Per-source RAG: papers that actually study THIS target.
    source_rag_block = "(no per-source RAG built for this run)"
    sr_info = state.get("source_rag") or {}
    sr_path = sr_info.get("sqlite_path")
    if sr_path:
        try:
            rows = src_rag.search_source_rag(
                Path(sr_path),
                queries=queries or [section.lower()],
                limit_per_query=k,
                prefer_target_matches=True,
            )
            if rows:
                lines = []
                for row in rows[: 2 * k]:
                    flag = "✓" if row.get("mentions_target") else "—"
                    title = (row.get("title") or "").strip()[:140]
                    lines.append(
                        f"- {flag} {row.get('bibcode')} ({row.get('year')}) {title}"
                    )
                source_rag_block = "\n".join(lines)
        except Exception as exc:
            source_rag_block = f"(per-source RAG error: {type(exc).__name__}: {exc})"
    # Reflexion: the verbal critique from the previous pass is "policy memory".
    # We append it to the evidence so the drafter MUST address each issue.
    refl_block = ""
    refl_history = state.get("reflexion_history") or []
    if refl_history:
        latest = refl_history[-1]
        verbal = latest.get("verbal_reflection") or ""
        if verbal:
            sections_targeted = latest.get("sections_to_rewrite") or []
            if section in sections_targeted or not sections_targeted:
                refl_block = (
                    "#### Reflexion from previous draft (MUST address before resubmitting this section)\n"
                    f"{verbal}\n\n"
                )

    # Physics checks (Rayleigh-Jeans / Ingress / Tidal truncation paragraphs)
    physics_block = ""
    pc = state.get("physics_checks") or {}
    if pc.get("latex") and section in ("Methods", "Results", "Discussion"):
        physics_block = (
            "#### Physics-driven arguments (ready-to-insert LaTeX)\n\n"
            "Use these paragraphs verbatim or paraphrased. They are derived "
            "deterministically from the SED 3-step χ² and Keplerian physics — do NOT contradict them.\n\n"
            "```latex\n" + (pc["latex"] or "") + "\n```\n\n"
        )

    # Novelty assessment (this-work vs literature)
    novelty_block = ""
    nv = state.get("novelty") or {}
    if nv.get("latex") and section in ("Abstract", "Results", "Discussion", "Conclusions"):
        novelty_block = (
            "#### Novelty assessment (this work vs literature)\n\n"
            "```latex\n" + (nv["latex"] or "") + "\n```\n\n"
        )

    # Comparison table (UPK 13-c2 style Table 2 against benchmark systems)
    cmp_block = ""
    cmp = state.get("comparison_table") or {}
    if cmp.get("status") == "ok" and section in ("Discussion", "Results"):
        cmp_block = (
            "#### Comparison with known benchmark systems (LaTeX deluxetable*, ready to insert)\n\n"
            "```latex\n" + (cmp.get("latex") or "") + "\n```\n\n"
        )

    # Figure refs
    fig_block = ""
    figs = (state.get("figures") or {}).get("figures") or {}
    fig_paths = {k: v.get("path") for k, v in figs.items() if v.get("status") == "ok"}
    if fig_paths and section in ("Data", "Results", "Discussion"):
        lines = ["#### Auto-synthesized figures (use \\includegraphics{<filename>}):"]
        for fk, p in fig_paths.items():
            cap = figs[fk].get("caption", "")
            lines.append(f"- `fig_{fk}.png` — {cap}")
        fig_block = "\n".join(lines) + "\n\n"

    # SED decoupled fit summary
    sed_block = ""
    sed = state.get("sed_decoupled") or {}
    if sed.get("status") == "ok" and section in ("Methods", "Results", "Discussion"):
        lines = ["#### SED 3-step decoupled fit χ² ranking"]
        if sed.get("mode") == "three_step":
            ranking = sed.get("ranking_joint_chi2") or []
            for r in ranking[:5]:
                d = r.get("chi2_diff")
                h = r.get("chi2_high")
                joint = r.get("joint")
                lines.append(
                    f"- `{r.get('hypothesis')}`: χ²_diff={d:.2f if d is not None else 'nan'} "
                    f"χ²_high={h:.2f if h is not None else 'nan'} joint={joint:.2f if joint is not None else 'nan'}"
                )
            best = sed.get("best_hypothesis_joint")
            if best:
                lines.append(f"- best (joint): **{best}**")
        else:
            results = sed.get("fit_results") or {}
            ranked = sorted(
                ((h, r.get("chi2")) for h, r in results.items() if r.get("chi2") is not None),
                key=lambda kv: kv[1],
            )
            for hyp, chi2 in ranked[:5]:
                tef = (results.get(hyp) or {}).get("teff_K")
                lines.append(f"- `{hyp}`: χ²={chi2:.2f}, Teff={tef} K")
            if ranked:
                lines.append(f"- best (single-state): **{ranked[0][0]}**")
        sed_block = "\n".join(lines) + "\n\n"

    # Extinction summary
    av_block = ""
    av = state.get("extinction") or {}
    if av.get("status") == "ok" and section in ("Methods", "Data"):
        av_block = (
            f"#### Extinction (A_V) used in SED fits\n"
            f"- A_V = {av.get('A_V'):.3f} mag (E(B-V) = {av.get('E_B_V'):.3f}, R_V = {av.get('R_V')})\n"
            f"- Provenance: {av.get('provenance')}\n\n"
        )

    # Cluster membership summary (if computed)
    cm_block = ""
    cm = state.get("cluster_membership") or {}
    if cm.get("status") == "ok" and cm.get("candidates"):
        lines = ["#### Cluster membership χ² (Hunt+2023 catalog)"]
        for cand in cm["candidates"][:5]:
            spat = cand.get("chi2_spat")
            kin = cand.get("chi2_kin")
            rv = cand.get("rv_offset_sigma")
            age = cand.get("cluster_age_myr")
            spat_str = f"{spat:.2f}" if spat is not None and not (isinstance(spat, float) and math.isnan(spat)) else "n/a"
            kin_str = f"{kin:.2f}" if kin is not None and not (isinstance(kin, float) and math.isnan(kin)) else "n/a"
            rv_str = f"{rv:.1f}σ" if rv is not None and not (isinstance(rv, float) and math.isnan(rv)) else "no_rv"
            age_str = f"{age:.0f} Myr" if age is not None and not (isinstance(age, float) and math.isnan(age)) else "?"
            v = ",".join(cand.get("verdict") or [])
            lines.append(
                f"- {cand.get('name')}: χ²_spat={spat_str}, χ²_kin={kin_str}, RV={rv_str}, "
                f"age={age_str} [{v}]"
            )
        cm_block = "\n".join(lines) + "\n\n"

    # D1 — Evidence availability manifest (drafter MUST honour withheld list)
    manifest_block = ""
    manifest = state.get("evidence_manifest")
    if manifest:
        try:
            from .evidence_manifest import render_for_drafter
            manifest_block = render_for_drafter(manifest, section) + "\n"
        except Exception:
            manifest_block = ""

    return (
        f"### Evidence for `{section}`\n\n"
        f"{manifest_block}"
        f"{refl_block}"
        f"{physics_block}"
        f"{sed_block}"
        f"{av_block}"
        f"{novelty_block}"
        f"{cmp_block}"
        f"{fig_block}"
        f"#### Published parameters + this-work measurements\n{pp_block}\n\n"
        f"#### Per-source SIMBAD-reference bibcodes (✓ = abstract mentions target)\n{source_rag_block}\n\n"
        f"{cm_block}"
        f"#### Local domain-wide RAG bibcodes (cite as \\citep{{<bibkey>}} where appropriate)\n{rag_block}\n\n"
        f"#### KG method-transfer triples\n{kg_block}\n"
    )


SECTION_PROMPTS: Dict[str, str] = {
    name: wd_domain.section_prompt(name)
    for name in (
        "Abstract", "Introduction", "Data", "Methods",
        "Results", "Discussion", "Conclusions",
    )
}


def _latest_reflection_advice(state: Dict[str, Any], section: str) -> List[str]:
    """Return the per-section advice strings from the most recent reflection."""
    history = state.get("reflexion_history") or []
    if not history:
        return []
    latest = history[-1] or {}
    out: List[str] = []
    for ai in latest.get("action_items") or []:
        if ai.get("section") == section:
            out.append(ai.get("advice") or "")
    return [a for a in out if a]


def _deterministic_abstract(state: Dict[str, Any]) -> str:
    target = state.get("target", "the target")
    ra = state.get("ra_deg")
    dec = state.get("dec_deg")
    plan = state.get("analysis_plan", {}) or {}
    sclass = plan.get("source_class", "unknown")
    pp = state.get("published_params") or {}
    rows = pp.get("rows", [])
    qa = state.get("qa", {}) or {}
    apj_gate = qa.get("apj_gate", "unknown")
    # Pick out concrete numbers from the table for a punchy abstract.
    lit_facts: List[str] = []
    this_facts: List[str] = []
    for r in rows:
        bib = r.get("bibcode") or ""
        unit = r.get("unit") or ""
        val = r.get("value")
        err = r.get("error")
        if val is None:
            continue
        val_str = f"{val:g}"
        if err is not None:
            val_str += f" $\\pm$ {err:g}"
        param_label = str(r.get("parameter", "")).replace("_", " ")
        if r.get("source_kind") == "simbad_abstract" and bib:
            lit_facts.append(f"{param_label} = {val_str} {unit} \\citep{{{bib}}}")
        elif str(r.get("source_kind", "")).startswith("this_work"):
            this_facts.append(f"{param_label} = {val_str} {unit}")
        if len(lit_facts) >= 3 and len(this_facts) >= 3:
            break
    # D1 — never fall back to a fictitious "previously published parameters"
    # phrase when the table is empty. Codex flagged the prior fallback as the
    # canonical fabrication source. Either we have facts to cite, or we
    # explicitly say we do not.
    if lit_facts:
        lit_clause = f"From the local published-parameter table we confirm {'; '.join(lit_facts[:3])}. "
    else:
        n_lit = sum(1 for r in rows if r.get("source_kind") == "simbad_abstract")
        lit_clause = (
            f"The published-parameter table currently has n\\_from\\_literature={n_lit}; "
            "no literature parameters are confirmed in this run. "
        )
    if this_facts:
        this_clause = f"From this run we independently obtain {'; '.join(this_facts[:3])}. "
    else:
        this_clause = (
            "No this-work numerical measurements are available in this run; "
            "the corresponding fitting modules report skipped or nonconverged status. "
        )
    body = (
        f"We present an automated, auditable multi-wavelength analysis of {target} "
        f"at ICRS $\\alpha={ra}$ deg, $\\delta={dec}$ deg. "
        f"SIMBAD type-based dispatch classifies the source as \\texttt{{{sclass}}}; "
        f"the run therefore routes to the appropriate physics pipeline rather than to a single-star "
        f"DA white-dwarf fit. " + lit_clause + this_clause +
        f"The Quality Assurance gate is currently \\texttt{{{apj_gate}}}, and final stellar parameters "
        f"will only be reported once the gate is cleared by either the model supervisor or human review."
    )
    # If reflexion asked us to lengthen the abstract or add numeric phrases,
    # append a sentence per literature parameter to reach the 180-280 word target.
    advice = _latest_reflection_advice(state, "Abstract")
    if advice and lit_facts:
        extra = []
        for fact in lit_facts[3:9]:
            extra.append(fact + ".")
        if extra:
            body += " Additional literature constraints: " + " ".join(extra)
    if advice and this_facts:
        extra = []
        for fact in this_facts[3:9]:
            extra.append(fact + ".")
        if extra:
            body += " Additional this-work findings: " + " ".join(extra)
    return r"\begin{abstract}" + body + r"\end{abstract}"


def _bibcodes_available(state: Dict[str, Any], limit: int = 4) -> List[str]:
    """Aggregate bibcodes from published_params first, then RAG hits, then
    source_rag (per-source FTS5).  Deduplicated, capped at `limit`."""
    out: List[str] = []
    seen: set = set()
    pp = state.get("published_params") or {}
    for r in pp.get("rows", []) or []:
        b = r.get("bibcode")
        if b and b not in seen:
            seen.add(b)
            out.append(b)
            if len(out) >= limit:
                return out
    for item in state.get("rag_results", []) or []:
        for row in item.get("rows", []) or []:
            b = row.get("bibcode")
            if b and b not in seen:
                seen.add(b)
                out.append(b)
                if len(out) >= limit:
                    return out
    return out


def _deterministic_introduction(state: Dict[str, Any]) -> str:
    target = state.get("target", "the target")
    plan = state.get("analysis_plan", {}) or {}
    sclass = plan.get("source_class", "unknown")
    bibs = _bibcodes_available(state, limit=4)
    cite = ""
    if bibs:
        cite = " \\citep{" + ",".join(bibs) + "}"

    # D1 — only enumerate this-work measurement types we actually have.
    # Codex flagged the previous hard-coded list `(periods, radial velocity,
    # kinematic traceback)` as fabrication. We now consult the manifest.
    manifest = state.get("evidence_manifest") or {}
    by_fam = manifest.get("by_parameter_family") or {}
    measured_phrase_parts: List[str] = []
    if by_fam.get("orbital_period", {}).get("status") == "measured":
        measured_phrase_parts.append("orbital period")
    if by_fam.get("radial_velocity", {}).get("status") == "measured":
        measured_phrase_parts.append("radial velocity")
    if by_fam.get("parallax", {}).get("status") == "measured":
        measured_phrase_parts.append("Gaia astrometry")
    if by_fam.get("cluster_membership", {}).get("status") == "measured":
        measured_phrase_parts.append("cluster-membership diagnostics")
    if by_fam.get("Teff_WD", {}).get("status") == "measured":
        measured_phrase_parts.append("SED-derived stellar parameters")
    measured_phrase = ", ".join(measured_phrase_parts) if measured_phrase_parts else "no this-work measurements yet"

    lines = [
        r"\section{Introduction}",
        (
            f"{target} has been studied in the literature{cite}. The local Astro\\_Agent workflow "
            f"classifies the source as \\texttt{{{sclass}}} from its SIMBAD object type, which "
            f"determines the physics route used for parameter inference."
        ),
        (
            "This paper provides an auditable, reproducible multi-wavelength characterization "
            "that combines literature-mined parameters with deterministic this-work measurements "
            f"({measured_phrase}) and runs them through a strict QA gate before releasing "
            "final values."
        ),
    ]
    # Append any withheld notice for the section (so reviewer sees the
    # constraint, not a silent skip).
    try:
        from .evidence_manifest import withheld_sentence
        ws = withheld_sentence(manifest, "Introduction")
        if ws:
            lines.append(ws)
    except Exception:
        pass
    return "\n\n".join(lines)


def _deterministic_methods(state: Dict[str, Any]) -> str:
    plan = state.get("analysis_plan", {}) or {}
    sclass = plan.get("source_class", "unknown")
    pipeline = plan.get("fitting_pipeline_module", "?")
    strat = plan.get("parameter_strategy", []) or []
    pp = state.get("published_params") or {}
    bibs = sorted({r.get("bibcode") for r in pp.get("rows", []) if r.get("bibcode")})
    cite = ""
    if bibs:
        cite = " \\citep{" + ",".join(bibs[:3]) + "}"
    lines = [
        r"\section{Methods}",
        (
            f"The Structure Planner classifies the source as \\texttt{{{sclass}}} via the SIMBAD "
            f"object type and selects \\texttt{{{pipeline}}} as the physics pipeline."
        ),
    ]
    # D1 — describe iteration status truthfully. Codex flagged the previous
    # blanket "Modeling proceeds in three iterations" sentence as F-class
    # fabrication when iters were planned/nonconverged.
    manifest = state.get("evidence_manifest") or {}
    by_fam = manifest.get("by_parameter_family") or {}
    iter_statuses = []
    for label, fam in (("baseline", "iteration_baseline"),
                       ("residuals", "iteration_residuals"),
                       ("systematics", "iteration_systematics")):
        entry = by_fam.get(fam) or {}
        st = entry.get("status")
        if st == "measured":
            iter_statuses.append(f"{label}: completed")
        else:
            ns = entry.get("node_status") or entry.get("status") or "not_attempted"
            iter_statuses.append(f"{label}: {ns}")
    lines.append(
        "Modeling proceeds in three iterations (baseline fitting, residual "
        "and physical-plausibility review, and statistical plus systematic "
        "uncertainty propagation). Current status — "
        + "; ".join(iter_statuses) + "."
    )
    for s in strat[:6]:
        lines.append(f"\\par {s}")
    lines.append(
        "Literature parameters mined from SIMBAD abstracts and reused as priors are listed in the published parameter table" + cite + "."
    )

    # Physics arguments (Rayleigh-Jeans, Ingress, Tidal truncation) — inserted
    # verbatim if physics_checks generated them.
    pc = state.get("physics_checks") or {}
    if pc.get("latex"):
        lines.append(r"\subsection{Physics Arguments}")
        lines.append(pc["latex"])

    # SED decoupled fit χ² ranking
    sed = state.get("sed_decoupled") or {}
    if sed.get("status") == "ok":
        lines.append(r"\subsection{SED 3-step decoupled fit}")
        if sed.get("mode") == "three_step":
            lines.append(r"\begin{itemize}")
            ranking = sed.get("ranking_joint_chi2") or []
            for r in ranking[:5]:
                d = r.get("chi2_diff")
                h = r.get("chi2_high")
                d_str = f"{d:.2f}" if d is not None and d == d else "n/a"
                h_str = f"{h:.2f}" if h is not None and h == h else "n/a"
                hyp_tex = str(r.get("hypothesis", "")).replace("_", r"\_")
                lines.append(
                    "\\item \\texttt{" + hyp_tex +
                    "}: $\\chi^2_\\mathrm{diff}=" + d_str +
                    "$, $\\chi^2_\\mathrm{high}=" + h_str + "$."
                )
            best = sed.get("best_hypothesis_joint")
            if best:
                best_tex = str(best).replace("_", r"\_")
                lines.append("\\item \\textbf{Best (joint $\\chi^2$):} \\texttt{" + best_tex + "}.")
            lines.append(r"\end{itemize}")
        else:
            results = sed.get("fit_results") or {}
            if results:
                lines.append(r"\begin{itemize}")
                ranked = sorted(
                    ((h, r.get("chi2")) for h, r in results.items() if r.get("chi2") is not None),
                    key=lambda kv: kv[1],
                )
                for hyp, chi2 in ranked[:5]:
                    tef = (results.get(hyp) or {}).get("teff_K")
                    hyp_tex = str(hyp).replace("_", r"\_")
                    lines.append(
                        "\\item \\texttt{" + hyp_tex +
                        f"}}: $\\chi^2={chi2:.2f}$, $T_\\mathrm{{eff}}={tef}$\\,K."
                    )
                lines.append(r"\end{itemize}")

    # Hypothesis-test plan — drafter must articulate competing interpretations.
    hp = state.get("hypothesis_plan") or {}
    hyps = hp.get("hypotheses") or []
    if hyps:
        lines.append(r"\subsection{Competing Hypotheses}")
        lines.append(
            "Following the discriminating-evidence style of, e.g., the UPK\\,13-c2 analysis "
            "\\citep[][where a WD$+$MS classification was overturned by a flat-bottomed eclipse + "
            "difference-spectrum test]{2025ApJ}, we explicitly enumerate the competing physical "
            "interpretations applicable to this source class:"
        )
        lines.append(r"\begin{itemize}")
        for h in hyps[:6]:
            label = h.get("label", "?")
            missing = ",".join(h.get("missing_observables") or []) or "all observables present"
            impl = "implemented" if h.get("module_implemented") else "not yet implemented"
            cites = h.get("references_bibcodes") or []
            cite_text = " \\citep{" + ",".join(cites) + "}" if cites else ""
            module_tex = str(h.get("fitting_module", "?")).replace("_", r"\_")
            label_tex = label.replace("_", r"\_")
            lines.append(
                "\\item " + label_tex
                + ": module \\texttt{" + module_tex + "} (" + impl + ")."
                + " Observable status: " + missing + "." + cite_text
            )
        lines.append(r"\end{itemize}")
        lines.append(
            f"Of the {hp.get('n_total')} competing hypotheses, "
            f"{hp.get('n_ready')} have all required observables in this run and "
            f"{hp.get('n_implemented')} have a fitting module currently implemented."
        )
    return "\n\n".join(lines)


def _deterministic_discussion(state: Dict[str, Any]) -> str:
    pp = state.get("published_params") or {}
    rows = pp.get("rows", [])
    n_lit = sum(1 for r in rows if r.get("source_kind") == "simbad_abstract")
    n_this = sum(1 for r in rows if str(r.get("source_kind", "")).startswith("this_work"))
    bibs = _bibcodes_available(state, limit=4)
    cite = ""
    if bibs:
        cite = " \\citep{" + ",".join(bibs) + "}"
    lines = [
        r"\section{Discussion}",
        (
            f"This work {('confirms' if n_lit else 'extends')} {n_lit} literature-mined parameter(s){cite} "
            f"and reports {n_this} this-work measurement(s) sourced from the deterministic astro\\_toolbox products."
        ),
        (
            "Important caveats: (i) any radial velocity derived against a DA white-dwarf cross-correlation "
            "template is template-dependent for hot subdwarf sources; (ii) photometric period peaks may "
            "represent half-period harmonics for ellipsoidal/eclipsing morphologies; (iii) kinematic "
            "traceback to candidate open clusters must satisfy BOTH spatial (within tidal radius) and "
            "temporal (approach time < cluster age) consistency before a natal-cluster claim is made."
        ),
    ]
    # Novelty assessment paragraph
    nv = state.get("novelty") or {}
    if nv.get("latex"):
        lines.append(nv["latex"])

    # Comparison table inserted as deluxetable*
    cmp = state.get("comparison_table") or {}
    if cmp.get("status") == "ok" and cmp.get("latex"):
        lines.append(cmp["latex"])

    cm = state.get("cluster_membership") or {}
    if cm.get("status") == "ok" and cm.get("candidates"):
        lines.append(r"\subsection{Cluster Membership Diagnostics}")
        for cand in cm["candidates"][:3]:
            name = (cand.get("name") or "?").replace("_", r"\_")
            spat = cand.get("chi2_spat")
            kin = cand.get("chi2_kin")
            age = cand.get("cluster_age_myr")
            rv_sig = cand.get("rv_offset_sigma")
            parts = []
            if spat is not None and not (isinstance(spat, float) and math.isnan(spat)):
                parts.append(f"$\\chi^2_{{\\rm spat}}={spat:.2f}$")
            if kin is not None and not (isinstance(kin, float) and math.isnan(kin)):
                parts.append(f"$\\chi^2_{{\\rm kin}}={kin:.2f}$")
            if rv_sig is not None and not (isinstance(rv_sig, float) and math.isnan(rv_sig)):
                parts.append(f"RV offset $={rv_sig:.1f}\\sigma$")
            if age is not None and not (isinstance(age, float) and math.isnan(age)):
                parts.append(f"cluster age $\\approx {age:.0f}$ Myr")
            verdict = ",".join(cand.get("verdict") or [])
            lines.append(
                f"\\par Candidate cluster {name}: " + "; ".join(parts) +
                (f" [verdict: {verdict}]" if verdict else "") + "."
            )
    return "\n\n".join(lines)


def _deterministic_data(state: Dict[str, Any]) -> str:
    """Build a Data section that references the auto-synthesized figures."""
    lines = [r"\section{Data}"]
    lines.append(
        "The Data Fetcher stage recorded SIMBAD cross-matching and astro\\_toolbox "
        "module status; full module statuses are machine-readable under the run directory."
    )
    figs = (state.get("figures") or {}).get("figures") or {}
    fig_lines = []
    for key in ("lightcurve", "sed", "cluster", "corner"):
        f = figs.get(key) or {}
        if f.get("status") != "ok":
            continue
        path = f.get("path", "")
        fname = path.split("/")[-1] if path else f"fig_{key}.png"
        cap = f.get("caption", "")
        fig_lines.append(
            r"\begin{figure}[t]" + "\n" +
            r"\centering" + "\n" +
            r"\includegraphics[width=\columnwidth]{" + fname + "}\n" +
            r"\caption{" + cap.replace("&", r"\&").replace("_", r"\_") + "}\n" +
            r"\label{fig:" + key + "}\n" +
            r"\end{figure}"
        )
    if fig_lines:
        lines.append(
            "Auto-synthesized figures are referenced below (filenames are copied into "
            "the LaTeX build directory by the compile step)."
        )
        lines.extend(fig_lines)
    av = state.get("extinction") or {}
    if av.get("status") == "ok":
        lines.append(
            r"\par Line-of-sight extinction is taken as $A_V = " +
            f"{av['A_V']:.2f}" + r"\,$mag ($E(B-V)=" + f"{av['E_B_V']:.2f}" +
            r"$, $R_V=3.1$; provenance: \texttt{" +
            str(av.get('provenance', '?')).replace('_', r'\_') + r"})."
        )
    return "\n\n".join(lines)


def _deterministic_results_section(state: Dict[str, Any]) -> str:
    """Build a Results section directly from the published_params table, so even
    use_llm=False produces a paper that reflects real measurements."""
    pp_table = state.get("published_params") or {}
    rows = pp_table.get("rows", [])
    qa = state.get("qa", {}) or {}
    plan = state.get("analysis_plan", {}) or {}
    lines = [r"\section{Results}"]
    lines.append(
        rf"\noindent QA gate status: \texttt{{{qa.get('apj_gate', 'unknown')}}}; "
        rf"source class: \texttt{{{plan.get('source_class', 'unknown')}}}; "
        rf"fitting pipeline: \texttt{{{plan.get('fitting_pipeline_module', 'unknown')}}}."
    )
    def _tex_escape(text: str) -> str:
        return text.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    lit = [r for r in rows if r.get("source_kind") == "simbad_abstract"]
    this = [r for r in rows if str(r.get("source_kind", "")).startswith("this_work")]
    if lit:
        lines.append(r"\subsection*{Confirmed literature parameters}")
        lines.append(r"\begin{itemize}")
        for r in lit[:20]:
            val = r["value"]
            err = r.get("error")
            unit = r.get("unit") or ""
            bib = r.get("bibcode") or "?"
            param_tex = _tex_escape(str(r.get("parameter", "")))
            err_str = " $\\pm$ " + str(err) if err is not None else ""
            lines.append(
                "\\item " + param_tex + f" = {val}" + err_str + f" {unit} " + "\\citep{" + bib + "}"
            )
        lines.append(r"\end{itemize}")
    if this:
        lines.append(r"\subsection*{This-work measurements}")
        lines.append(r"\begin{itemize}")
        for r in this[:20]:
            val = r.get("value")
            err = r.get("error")
            unit = r.get("unit") or ""
            kind = r.get("source_kind", "")
            param_tex = _tex_escape(str(r.get("parameter", "")))
            kind_tex = _tex_escape(str(kind))
            if val is None:
                val_part = "n/a"
            elif err is not None:
                val_part = f"{val} $\\pm$ {err} {unit}"
            else:
                val_part = f"{val} {unit}"
            snippet = (r.get("snippet") or "")[:120]
            lines.append(
                "\\item " + param_tex + ": " + val_part + " (\\texttt{" + kind_tex + "}; " + snippet + ")"
            )
        lines.append(r"\end{itemize}")
    if not rows:
        lines.append(
            r"No published-parameter table rows were available; final numerical values are withheld."
        )
    return "\n".join(lines) + "\n"


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
        "Abstract": _deterministic_abstract(state),
        "Introduction": _deterministic_introduction(state),
        "Data": _deterministic_data(state),
        "Methods": _deterministic_methods(state),
        "Results": _deterministic_results_section(state),
        "Discussion": _deterministic_discussion(state),
        "Conclusions": r"\section{Conclusions}This package provides a reproducible trail from target identity through manuscript generation. Final interpretation depends on QA clearance.",
    }
    fallback = fallback_lines.get(section, f"\\section{{{section}}}")
    if not use_llm:
        # Even deterministic outputs are logged so we have a baseline for
        # later DSPy/MIPRO optimization comparison.
        try:
            prompt_experiment_log.record_call(
                source_id=state.get("target"),
                source_class=(state.get("analysis_plan") or {}).get("source_class"),
                specialist=specialists.specialist_for(section),
                section=section,
                reflexion_retry_idx=int(state.get("reflexion_retry_count") or 0),
                system_prompt="(deterministic fallback)",
                user_prompt="(deterministic fallback)",
                output=fallback,
                paper_qc=state.get("paper_qc"),
                notes="use_llm=False",
            )
        except Exception:
            pass
        return fallback

    evidence = pack_section_evidence(state, section)
    # D5 — prefer the allowlist published by write_literature (it has been
    # filtered to ADS-shaped bibcodes only). Fall back to the rag-derived
    # bibkeys when literature has not run yet.
    allowlist = state.get("bibkey_allowlist") or bibkeys
    if allowlist:
        bib_hint = (
            "Available bib keys (ALLOWLIST — do NOT invent any other "
            "\\citep keys): " + ", ".join(allowlist[:40])
        )
    else:
        bib_hint = "No bib keys are available; do NOT invent any."
    # Multi-specialist routing (2026 multi-agent pattern):
    # - Physicist drafts Methods/Results (physics + units + χ² rigor)
    # - Writer drafts Abstract/Intro/Discussion/Conclusions (narrative + citations)
    specialist = specialists.specialist_for(section)
    specialist_system = specialists.system_prompt_for(specialist)
    system = (
        ANTI_LEAKAGE
        + "\n"
        + specialist_system
        + "\nReturn ONLY LaTeX for THIS section (no preamble, no \\begin{document}, no \\end{document})."
    )
    instr = wd_domain.section_prompt(section)
    user = (
        f"[role={specialist}]\n"
        f"Target: {state.get('target')}\n"
        f"Coordinates: RA={state.get('ra_deg')}, Dec={state.get('dec_deg')}\n"
        f"QA gate: {json.dumps(state.get('qa', {}))[:1500]}\n\n"
        f"{instr}\n\n{bib_hint}\n\n{evidence}"
    )
    output = _llm_text(system=system, user=user, fallback=fallback, provider=provider)

    # Best-of-N: when state requests it AND a reward model is available,
    # generate (best_of_n - 1) extra candidates and pick the highest-scored.
    best_of_n = int(state.get("best_of_n", 1) or 1)
    if best_of_n > 1:
        try:
            from pathlib import Path as _Path
            import json as _json
            rm_path = _Path(__file__).resolve().parent.parent / "scripts" / "prompt_tuning" / "reward_model.json"
            if rm_path.exists():
                from scripts.prompt_tuning import reward_model as _rm  # type: ignore
                model = _json.loads(rm_path.read_text(encoding="utf-8"))
                candidates = [output]
                for _ in range(best_of_n - 1):
                    cand = _llm_text(system=system, user=user, fallback=fallback, provider=provider)
                    candidates.append(cand)
                scored = [(_rm.predict(c, model), c) for c in candidates]
                scored.sort(key=lambda x: x[0], reverse=True)
                output = scored[0][1]
                state.setdefault("best_of_n_log", []).append({
                    "section": section,
                    "n_candidates": len(candidates),
                    "scores": [s for s, _ in scored],
                })
        except Exception as exc:
            state.setdefault("warnings", []).append(f"best_of_n failed: {type(exc).__name__}: {exc}")

    # DSPy/MIPRO-prep: record this prompt + output for offline optimization.
    try:
        prompt_experiment_log.record_call(
            source_id=state.get("target"),
            source_class=(state.get("analysis_plan") or {}).get("source_class"),
            specialist=specialist,
            section=section,
            reflexion_retry_idx=int(state.get("reflexion_retry_count") or 0),
            system_prompt=system,
            user_prompt=user,
            output=output,
            paper_qc=state.get("paper_qc"),
            notes=f"use_llm={use_llm}",
        )
    except Exception:
        pass
    return output


def _ensure_figure_refs(body: str, workspace: Optional[Path] = None) -> str:
    """D6 — every `\\label{fig:X}` in `body` must be `\\ref`'d somewhere.

    For each orphan label, append a `(see Fig.~\\ref{fig:X})` to the
    paragraph containing the label; log the autofix to
    `paper_orchestra/figure_ref_gaps.json`. If the body cannot be fixed
    (no paragraph to insert into), strip the entire figure block.
    """
    import re as _re
    labels = _re.findall(r"\\label\{(fig:[^}]+)\}", body)
    if not labels:
        return body
    refs = set(_re.findall(r"\\ref\{(fig:[^}]+)\}", body))
    orphans = [lab for lab in labels if lab not in refs]
    if not orphans:
        return body
    fixed: List[Dict[str, Any]] = []
    for lab in orphans:
        # Find the figure block containing the label, then locate the
        # next non-figure paragraph after it and append the reference.
        m = _re.search(
            r"\\begin\{figure\}.*?\\label\{" + _re.escape(lab) + r"\}.*?\\end\{figure\}",
            body, flags=_re.DOTALL,
        )
        if not m:
            continue
        after = body[m.end():]
        # Insert `(see Fig.~\ref{fig:X})` at the end of the first
        # paragraph that follows the figure block.
        para_match = _re.search(r"\n\n([^\n]+?)\n", after)
        if para_match:
            ref_sentence = f" (see Fig.~\\ref{{{lab}}})"
            insert_at = m.end() + para_match.end(1)
            body = body[:insert_at] + ref_sentence + body[insert_at:]
            fixed.append({"label": lab, "action": "auto_injected_ref"})
        else:
            # No paragraph after — strip the orphan figure block.
            body = body[:m.start()] + body[m.end():]
            fixed.append({"label": lab, "action": "stripped_orphan_figure"})
    if fixed and workspace is not None:
        try:
            tools.json_dump(workspace / "figure_ref_gaps.json", {
                "n_fixed": len(fixed),
                "fixes": fixed,
            })
        except Exception:
            pass
    return body


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
    # D6 — auto-fix orphan figure labels.
    body = _ensure_figure_refs(body, workspace=workspace)
    tail = "\n\n" + r"\acknowledgments" + "\n" + "Generated from local astrotool, RAG, and KG artifacts.\n" + r"\bibliography{refs}" + "\n" + r"\end{document}" + "\n"
    return head + body + tail


def llm_review(state: Dict[str, Any], paper_tex: str, provider: Optional[str] = None) -> Dict[str, Any]:
    """Have the LLM act as a sharp ApJ reviewer; returns JSON scores + questions."""
    fallback = {
        "score": {"rigor": 15, "grounding": 15, "clarity": 15, "figures": 12, "overall": 60},
        "questions": reviewer_questions(state),
        "actions": ["Tighten language; ensure QA caveats are preserved."],
        "decision": "minor_revise",
        "weakest_section": "Discussion",
        "wd_specific_concerns": [],
    }
    system = wd_domain.system_for_role("reviewer") + "\n" + wd_domain.REVIEWER_TASK
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
    # If LLM is disabled, the reviewer used to return a fake "overall=75" for
    # every iteration with rewrote_section=null, which produced an opaque loop
    # that contributed nothing. Now: skip refinement entirely when use_llm is
    # off and record the reason explicitly so the worklog is honest.
    if not use_llm:
        worklog["iterations"] = []
        worklog["halted_because"] = "refinement_skipped_use_llm_false"
        final_dir = tools.ensure_dir(workspace / "final")
        final_tex = tools.write_text(final_dir / "paper.tex", best)
        worklog_path = tools.json_dump(workspace / "refinement" / "worklog.json", worklog)
        return {
            "final_tex": final_tex,
            "worklog": worklog_path,
            "worklog_data": worklog,
            "final_review": {
                "skipped": True,
                "reason": "use_llm=False; deterministic reviewer is not informative.",
                "questions": reviewer_questions(state),
            },
        }

    for index in range(1, max_iters + 1):
        review = llm_review(state, best, provider=provider)
        last_review = review
        overall = int(review.get("score", {}).get("overall", 0) or 0)
        accepted = overall >= target_score or review.get("decision") == "accept"

        # Reviewer-driven rewrite of the weakest section. The reviewer may
        # name a section directly via review["weakest_section"]; otherwise
        # we pick the lowest-scoring axis and map it to a section.
        rewrite_target: Optional[str] = None
        if not accepted:
            scores = review.get("score", {}) or {}
            axis_to_section = {
                "grounding": "Methods",
                "rigor": "Results",
                "clarity": "Discussion",
                "figures": "Data",
            }
            named = str(review.get("weakest_section") or "").strip().title()
            if named in {"Abstract", "Introduction", "Data", "Methods", "Results", "Discussion", "Conclusions"}:
                rewrite_target = named
            else:
                axis = min(
                    ("grounding", "rigor", "clarity", "figures"),
                    key=lambda k: scores.get(k, 5),
                )
                rewrite_target = axis_to_section.get(axis, "Discussion")
            new_section = write_section(workspace, state, rewrite_target, bibkeys, use_llm=True, provider=provider)
            best = _replace_section(best, rewrite_target, new_section)
            # Halt on no-improvement (last two iterations same overall score and
            # rewrote_section identical).
            if len(worklog["iterations"]) >= 1:
                prev = worklog["iterations"][-1]
                if (
                    prev.get("review", {}).get("score", {}).get("overall") == overall
                    and prev.get("rewrote_section") == rewrite_target
                ):
                    worklog["halted_because"] = "no_improvement"
                    iter_dir = tools.ensure_dir(workspace / "refinement" / f"iter{index}")
                    tools.write_text(iter_dir / "paper.tex", best)
                    tools.json_dump(iter_dir / "review.json", review)
                    worklog["iterations"].append(
                        {
                            "iter": index,
                            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                            "review": review,
                            "rewrote_section": rewrite_target,
                            "decision": "halt_no_improvement",
                        }
                    )
                    worklog["best_iter"] = index
                    break

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
