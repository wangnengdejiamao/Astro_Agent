"""Deterministic Toolbox/KG audit used by Codex reviewer integration.

This module mirrors Persona E in ``codex_reviewer_alignment.md``.  It is
intentionally conservative and cheap: it only reads local run artifacts and
checks whether manuscript claims are backed by executed workflow nodes,
class-aware KG/RAG retrieval, method-scout/toolbox status, per-source RAG, and
citation provenance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from .prompts.retrieval import RERANK_KEYS

_CITE_COMMAND = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")

_NODE_ARTIFACTS = (
    "02b_analysis_plan.json",
    "02e_cluster_membership.json",
    "02h_sed_decoupled.json",
    "02i_physics_checks.json",
    "02j_light_curve_geometry.json",
    "02k_eclipse_mcmc.json",
    "02l_ads_live.json",
    "02m_novelty.json",
    "02n_comparison_table.json",
    "05_iteration_1_baseline.json",
    "06_iteration_2_residuals.json",
    "07_iteration_3_systematics.json",
    "07b_model_supervision.json",
    "08_qa_gate.json",
)

_BAD_STATUSES = {"skipped", "dry_run", "dry-run", "nonconverged", "error", "planned", "no_inputs"}

_CLAIM_PATTERNS = {
    "02h_sed_decoupled.json": ("sed", "decoupled", "composite", "rayleigh-jeans", "rayleigh jeans"),
    "02j_light_curve_geometry.json": ("period", "light curve", "light-curve", "ingress", "egress", "half-period"),
    "02k_eclipse_mcmc.json": ("mcmc", "eclipse", "inclination", "posterior"),
    "05_iteration_1_baseline.json": ("baseline fit", "baseline fitting", "modeling proceeds"),
    "06_iteration_2_residuals.json": ("residual", "plausibility", "physics review"),
    "07_iteration_3_systematics.json": ("systematic", "error budget", "uncertainty propagation"),
    "07b_model_supervision.json": ("model supervisor", "supervision"),
}

_MODULE_PATTERNS = {
    "cluster_membership": ("02e_cluster_membership.json", ("cluster", "membership", "traceback")),
    "wd_fitting": ("05_iteration_1_baseline.json", ("wd_fitting", "white-dwarf fit", "wd fit")),
    "sed": ("02h_sed_decoupled.json", ("astro_toolbox.sed", "sed fit", "sed fitting")),
    "disk_eclipse_mcmc": ("02k_eclipse_mcmc.json", ("disk_eclipse", "eclipse mcmc")),
    "period_analysis": ("02j_light_curve_geometry.json", ("period_analysis", "period search")),
}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _read_json(path: Path) -> Any:
    text = _read_text(path)
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_cite_keys(tex: str) -> List[str]:
    keys: List[str] = []
    for chunk in _CITE_COMMAND.findall(tex or ""):
        for key in chunk.split(","):
            key = key.strip()
            if key:
                keys.append(key)
    return keys


def _status_of(obj: Any) -> str:
    if isinstance(obj, Mapping):
        return str(obj.get("status") or obj.get("apj_gate") or "unknown")
    return "missing"


def _text_has_any(tex: str, needles: Sequence[str]) -> Optional[str]:
    lower = tex.lower()
    for needle in needles:
        if needle.lower() in lower:
            return needle
    return None


def _flatten_rows(obj: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(obj, list):
        for item in obj:
            yield from _flatten_rows(item)
    elif isinstance(obj, Mapping):
        if "_rerank_score" in obj or "subject" in obj or "object" in obj:
            yield obj
        for key in ("rows", "kg_hits", "rag_hits", "hits"):
            value = obj.get(key)
            if value is not None:
                yield from _flatten_rows(value)


def _source_class(run_dir: Path, artifacts: Mapping[str, Any]) -> str:
    for key in ("analysis_plan", "hypothesis_plan", "qa_gate"):
        obj = artifacts.get(key)
        if isinstance(obj, Mapping) and obj.get("source_class"):
            return str(obj.get("source_class"))
    ap = _read_json(run_dir / "02b_analysis_plan.json")
    if isinstance(ap, Mapping) and ap.get("source_class"):
        return str(ap.get("source_class"))
    return "unknown"


def _expected_keys(source_class: str) -> List[str]:
    base = list(RERANK_KEYS.get(source_class, []))
    if source_class and source_class != "white_dwarf_binary":
        base += list(RERANK_KEYS.get("white_dwarf_binary", []))
    return list(dict.fromkeys(base))


def _project_root_from_run(run_dir: Path) -> Path:
    for parent in [run_dir] + list(run_dir.parents):
        if (parent / "astro_toolbox").exists() and (parent / "analysis_agent").exists():
            return parent
    return Path(__file__).resolve().parents[1]


def _citation_sources(artifacts: Mapping[str, Any]) -> Dict[str, Set[str]]:
    sources: Dict[str, Set[str]] = {}

    pub = artifacts.get("published_params")
    if isinstance(pub, Mapping):
        for row in pub.get("rows") or []:
            if isinstance(row, Mapping) and row.get("bibcode"):
                sources.setdefault(str(row["bibcode"]), set()).add("published_params")

    source_rag = artifacts.get("source_rag")
    if isinstance(source_rag, Mapping):
        for key in ("source_refs", "refs", "rows"):
            for row in source_rag.get(key) or []:
                if isinstance(row, Mapping):
                    bib = row.get("bibcode") or row.get("key")
                    if bib:
                        sources.setdefault(str(bib), set()).add(f"source_rag.{key}")

    comp = artifacts.get("comparison_table")
    if isinstance(comp, Mapping):
        for bib in comp.get("bibcodes") or []:
            sources.setdefault(str(bib), set()).add("comparison_table.bibcodes")

    hyp = artifacts.get("hypothesis_plan")
    if isinstance(hyp, Mapping):
        for h in hyp.get("hypotheses") or []:
            if not isinstance(h, Mapping):
                continue
            for bib in h.get("references_bibcodes") or []:
                sources.setdefault(str(bib), set()).add("hypothesis_plan.references_bibcodes")

    return sources


def run_toolbox_kg_audit(run_dir: Path | str, manuscript: Optional[str] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    tex = manuscript if manuscript is not None else _read_text(run_dir / "paper_orchestra" / "final" / "paper.tex")

    artifacts = {
        "analysis_plan": _read_json(run_dir / "02b_analysis_plan.json"),
        "published_params": _read_json(run_dir / "02c_published_params.json"),
        "source_rag": _read_json(run_dir / "02d_source_rag.json"),
        "cluster_membership": _read_json(run_dir / "02e_cluster_membership.json"),
        "hypothesis_plan": _read_json(run_dir / "02f_hypothesis_plan.json"),
        "comparison_table": _read_json(run_dir / "02n_comparison_table.json"),
        "qa_gate": _read_json(run_dir / "08_qa_gate.json"),
    }
    source_class = _source_class(run_dir, artifacts)
    expected_keys = _expected_keys(source_class)
    project_root = _project_root_from_run(run_dir)

    node_status_inconsistencies: List[Dict[str, Any]] = []
    statuses: Dict[str, str] = {}
    for rel in _NODE_ARTIFACTS:
        obj = _read_json(run_dir / rel)
        status = _status_of(obj) if obj is not None else "missing"
        statuses[rel] = status
        if status.lower() in _BAD_STATUSES:
            hit = _text_has_any(tex, _CLAIM_PATTERNS.get(rel, ()))
            if hit:
                node_status_inconsistencies.append({
                    "artifact": rel,
                    "status": status,
                    "manuscript_quote": hit,
                    "comment": f"manuscript invokes `{hit}` but {rel} status is {status}",
                })
    for module, (rel, needles) in _MODULE_PATTERNS.items():
        hit = _text_has_any(tex, needles)
        status = statuses.get(rel) or _status_of(_read_json(run_dir / rel))
        if hit and status.lower() in _BAD_STATUSES:
            node_status_inconsistencies.append({
                "artifact": rel,
                "status": status,
                "manuscript_quote": hit,
                "comment": f"module-like claim `{module}` appears in manuscript but supporting node is {status}",
            })

    kg_retrieval_issues: List[Dict[str, Any]] = []
    kg_results = _read_json(run_dir / "04_kg_results.json")
    for idx, row in enumerate(_flatten_rows(kg_results)):
        score = row.get("_rerank_score")
        why = str(row.get("_rerank_why") or "")
        try:
            score_f = float(score)
        except Exception:
            score_f = 0.0
        matched_expected = any(k.lower() in why.lower() for k in expected_keys)
        if score_f <= 0:
            kg_retrieval_issues.append({
                "row_index": idx,
                "issue": "rerank_score<=0 — no class keyword matched",
                "row_subject": row.get("subject") or row.get("title"),
                "row_object": row.get("object") or row.get("abstract"),
            })
        elif expected_keys and not matched_expected:
            kg_retrieval_issues.append({
                "row_index": idx,
                "issue": "_rerank_why does not name an expected RERANK_KEYS term",
                "row_subject": row.get("subject") or row.get("title"),
                "row_object": row.get("object") or row.get("abstract"),
            })

    method_scout_issues: List[Dict[str, Any]] = []
    scout = _read_json(run_dir / "04c_method_scout.json")
    if isinstance(scout, Mapping):
        scout_keys = scout.get("rerank_keys") or []
        if expected_keys and not set(expected_keys).issubset(set(scout_keys)):
            method_scout_issues.append({
                "field": "rerank_keys",
                "issue": f"method_scout rerank_keys do not include all expected keys for source_class={source_class}",
            })
        spec = scout.get("algorithm_spec") or {}
        if isinstance(spec, Mapping):
            module = str(spec.get("target_module") or "")
            module_path = project_root / "astro_toolbox" / f"{module}.py" if module else None
            planned = module in {"rv_orbit_method", "literature_method", "spectral_fitting"}
            if not module:
                method_scout_issues.append({"field": "algorithm_spec.target_module", "issue": "missing target_module"})
            elif module_path and not module_path.exists() and not planned:
                method_scout_issues.append({
                    "field": "algorithm_spec.target_module",
                    "issue": f"astro_toolbox/{module}.py does not exist and is not marked as planned",
                })
        gap = scout.get("toolbox_gap") or _read_json(run_dir / "04e_toolbox_gap.json")
        if isinstance(gap, Mapping):
            gap_status = str(gap.get("status") or "")
            target_module = str(gap.get("target_module") or "")
            if gap_status == "ready_for_tool_write" and target_module.lower() in tex.lower() and "ran" in tex.lower():
                method_scout_issues.append({
                    "field": "toolbox_gap.status",
                    "issue": f"toolbox_gap is ready_for_tool_write for {target_module}, but manuscript appears to claim execution",
                })
    else:
        method_scout_issues.append({"field": "04c_method_scout.json", "issue": "missing or unparsable"})

    per_source_rag_issues: List[Dict[str, Any]] = []
    sr = artifacts.get("source_rag")
    if isinstance(sr, Mapping):
        n_refs = int(sr.get("n_refs") or 0)
        n_target = int(sr.get("n_refs_mentioning_target") or 0)
        intro = re.search(r"\\section\{\s*Introduction\s*\}(.*?)(?=\\section\{|\\end\{document\})", tex, flags=re.DOTALL)
        intro_body = intro.group(1) if intro else ""
        if _extract_cite_keys(intro_body) and n_refs == 0:
            per_source_rag_issues.append({
                "finding": "n_refs=0 but Introduction uses citations for source framing",
                "manuscript_quote": intro_body.strip()[:220],
            })
        if re.search(r"studied|literature|published|reference", intro_body, flags=re.IGNORECASE) and n_target == 0:
            per_source_rag_issues.append({
                "finding": "n_refs_mentioning_target=0 but Introduction implies target-specific literature",
                "manuscript_quote": intro_body.strip()[:220],
            })
    else:
        per_source_rag_issues.append({"finding": "02d_source_rag.json missing", "manuscript_quote": ""})

    citation_sources = _citation_sources(artifacts)
    citation_provenance: List[Dict[str, Any]] = []
    for key in _extract_cite_keys(tex):
        found = sorted(citation_sources.get(key) or [])
        citation_provenance.append({
            "key": key,
            "found_in": found or None,
            "verdict": "ok" if found else "fabricated",
        })

    total_nodes = len(_NODE_ARTIFACTS)
    bad_nodes = sum(1 for status in statuses.values() if status.lower() in _BAD_STATUSES or status == "missing")
    toolbox_score = max(0.0, 1.0 - (bad_nodes + len(node_status_inconsistencies) + len(method_scout_issues)) / max(total_nodes + 4, 1))
    kg_rows = list(_flatten_rows(kg_results))
    kg_score = 1.0 if not kg_rows and source_class == "unknown" else max(0.0, 1.0 - len(kg_retrieval_issues) / max(len(kg_rows), 1))
    if per_source_rag_issues:
        kg_score = max(0.0, kg_score - 0.25)
    fabricated = sum(1 for c in citation_provenance if c["verdict"] == "fabricated")
    if fabricated:
        kg_score = max(0.0, kg_score - min(0.5, 0.1 * fabricated))

    return {
        "node_status_inconsistencies": node_status_inconsistencies,
        "kg_retrieval_issues": kg_retrieval_issues,
        "method_scout_issues": method_scout_issues,
        "per_source_rag_issues": per_source_rag_issues,
        "citation_provenance": citation_provenance,
        "toolbox_coverage_score_0_to_1": round(toolbox_score, 2),
        "kg_alignment_score_0_to_1": round(kg_score, 2),
        "comment": (
            f"source_class={source_class}; {bad_nodes}/{total_nodes} audited nodes are missing/skipped/planned/nonconverged/error; "
            f"{fabricated} cited keys lack artifact provenance."
        ),
    }


__all__ = ["run_toolbox_kg_audit"]
