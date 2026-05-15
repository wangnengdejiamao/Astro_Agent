"""Per-section evidence-availability manifest (D1).

Read every numbered artifact's `status` plus the `published_params.rows`
and emit one structured dict that the drafter consults BEFORE writing.
Each parameter family is tagged with one of:

    measured                  — value+error in published_params, source
                                artifact is `ok`
    withheld_node_unavailable — node returned skipped / nonconverged /
                                error / planned / dry_run / no_inputs
    withheld_qa_hold          — measurement exists but 08_qa_gate is hold
    withheld_unsupported_prov — provenance does not meet domain contract
                                (e.g. extinction = fallback_latitude_scaling)
    not_attempted             — no artifact for this module in this run

Manifest layout (also persisted as `02o_evidence_manifest.json`):

    {
      "build_timestamp": ...,
      "run_dir": ...,
      "qa_gate": "clear" | "hold_for_human" | ...,
      "by_parameter_family": {
         "parallax":            {"status": "measured", "value": 1.013, ...},
         "proper_motion":       {"status": "measured", ...},
         "orbital_period":      {"status": "withheld_node_unavailable",
                                 "reason": "02j_light_curve_geometry status=skipped"},
         "radial_velocity":     {"status": "not_attempted", ...},
         "cluster_membership":  {"status": "withheld_node_unavailable",
                                 "reason": "cluster_membership status=insufficient"},
         "extinction":          {"status": "withheld_unsupported_prov",
                                 "reason": "A_V provenance=fallback_latitude_scaling"},
         "iteration_baseline":  {"status": "withheld_node_unavailable", ...},
         ...
      },
      "by_section": {
         "Abstract":     ["parallax", "proper_motion"],   // measured fams
         "Methods":      [],                              // none yet
         ...
      },
      "withheld_sentences": {
         "Results":     "We withhold the orbital period and radial velocity ...",
         "Discussion":  "Cluster membership remains insufficient ..."
      }
    }

The drafter renders the manifest into every section's evidence block via
`paper_orchestra.pack_section_evidence`. The deterministic section
writers also import this module to refuse fabricated claims when
use_llm=False.

Single source of truth for which artifact represents which node: shared
with `toolbox_kg_audit` via `_NODE_ARTIFACTS`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

# --------------------------------------------------------------------------- #
# Single source of truth — moved from toolbox_kg_audit.py to avoid drift.    #
# --------------------------------------------------------------------------- #

NODE_ARTIFACTS: Tuple[str, ...] = (
    "02b_analysis_plan.json",
    "02e_cluster_membership.json",
    "02g_extinction.json",
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

BAD_STATUSES = {"skipped", "dry_run", "dry-run", "nonconverged",
                "error", "planned", "no_inputs", "unavailable",
                "insufficient", "needs_human"}

# Per-section visibility — which parameter families a given section is
# allowed to discuss. The drafter prompt also gets this so it knows what
# NOT to discuss.
SECTION_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "Abstract":     ("parallax", "proper_motion", "extinction", "source_class",
                     "orbital_period", "radial_velocity",
                     "Teff_WD", "M_WD", "cluster_membership"),
    "Introduction": ("source_class", "cluster_membership"),
    "Data":         ("parallax", "proper_motion", "extinction",
                     "photometry", "spectroscopy"),
    "Methods":      ("source_class", "fitting_pipeline",
                     "iteration_baseline", "iteration_residuals",
                     "iteration_systematics", "sed_decoupled",
                     "eclipse_mcmc"),
    "Results":      ("parallax", "proper_motion", "Teff_WD", "M_WD",
                     "orbital_period", "radial_velocity",
                     "cluster_membership", "extinction"),
    "Discussion":   ("cluster_membership", "source_class", "novelty",
                     "comparison_table"),
    "Conclusions":  ("source_class", "cluster_membership"),
}


# Map parameter families to their source artifacts. A family is `measured`
# only if (a) the relevant artifact's status is `ok` AND (b) at least one
# published_params row matches a `pp_match` key.
FAMILY_SOURCES: Dict[str, Dict[str, Any]] = {
    "parallax":                {"pp_match": ("parallax_mas",), "node": None},
    "proper_motion":           {"pp_match": ("pmRA_mas_per_yr", "pmDE_mas_per_yr"), "node": None},
    "extinction":              {"pp_match": (), "node": "02g_extinction.json"},
    "orbital_period":          {"pp_match": ("P_orb_min", "P_orb_day", "P_orb_h"), "node": "02j_light_curve_geometry.json"},
    "radial_velocity":         {"pp_match": ("radial_velocity_km_s", "K_km_s"), "node": None},
    "Teff_WD":                 {"pp_match": ("Teff_WD_K", "T_eff_WD"), "node": "02h_sed_decoupled.json"},
    "M_WD":                    {"pp_match": ("M_WD_Msun", "M_WD"), "node": "02h_sed_decoupled.json"},
    "cluster_membership":      {"pp_match": (), "node": "02e_cluster_membership.json"},
    "sed_decoupled":           {"pp_match": (), "node": "02h_sed_decoupled.json"},
    "eclipse_mcmc":            {"pp_match": (), "node": "02k_eclipse_mcmc.json"},
    "iteration_baseline":      {"pp_match": (), "node": "05_iteration_1_baseline.json"},
    "iteration_residuals":     {"pp_match": (), "node": "06_iteration_2_residuals.json"},
    "iteration_systematics":   {"pp_match": (), "node": "07_iteration_3_systematics.json"},
    "novelty":                 {"pp_match": (), "node": "02m_novelty.json"},
    "comparison_table":        {"pp_match": (), "node": "02n_comparison_table.json"},
    "source_class":            {"pp_match": (), "node": "02b_analysis_plan.json"},
    "fitting_pipeline":        {"pp_match": (), "node": "02b_analysis_plan.json"},
    "photometry":              {"pp_match": (), "node": "02_data_fetch.json"},
    "spectroscopy":            {"pp_match": (), "node": "02_data_fetch.json"},
}


def _read_json(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _node_status(run_dir: Path, artifact: Optional[str]) -> str:
    if not artifact:
        return "not_required"
    obj = _read_json(run_dir / artifact)
    if obj is None:
        return "not_attempted"
    return str(obj.get("status") or obj.get("apj_gate") or "unknown")


def _pp_match(rows: List[Mapping[str, Any]], names: Tuple[str, ...]) -> Optional[Mapping[str, Any]]:
    if not names:
        return None
    for r in rows:
        if str(r.get("parameter")) in names and r.get("value") is not None:
            return r
    return None


def _qa_hold_reason(qa: Mapping[str, Any]) -> Optional[str]:
    gate = str(qa.get("apj_gate") or "")
    if gate in ("hold", "hold_for_human"):
        reasons = qa.get("reasons") or []
        return f"qa_gate={gate}: {reasons[0] if reasons else 'no reason'}"
    return None


def build_manifest(run_dir: Path, state: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Read every status field and produce the manifest dict. Pure I/O —
    safe to call multiple times. `state` is optional; when supplied we
    prefer values from in-memory state over disk."""
    pp = (state or {}).get("published_params") if state else _read_json(run_dir / "02c_published_params.json")
    pp_rows = (pp or {}).get("rows") or []
    qa = (state or {}).get("qa") if state else _read_json(run_dir / "08_qa_gate.json")
    qa = qa or {}

    by_family: Dict[str, Dict[str, Any]] = {}
    hold = _qa_hold_reason(qa)
    for family, spec in FAMILY_SOURCES.items():
        node_status = _node_status(run_dir, spec.get("node"))
        match = _pp_match(pp_rows, tuple(spec.get("pp_match") or ()))

        entry: Dict[str, Any] = {"node_artifact": spec.get("node")}

        # Decision tree (re-ordered after first review):
        # 1. If we have a real measurement, surface it — even under QA hold —
        #    but tag with `qa_hold_caveat` so the drafter knows to add a
        #    "pending QA clearance" qualifier.
        if match is not None:
            entry.update(
                status="measured",
                value=match.get("value"),
                error=match.get("error"),
                unit=match.get("unit"),
                bibcode=match.get("bibcode"),
                source_kind=match.get("source_kind"),
            )
            if hold:
                entry["qa_hold_caveat"] = hold
        # 2. Node is named, status is bad → withhold (node-level failure
        #    dominates over QA hold because the data is genuinely absent).
        elif spec.get("node") and node_status in BAD_STATUSES:
            entry.update(
                status="withheld_node_unavailable",
                reason=f"{spec['node']} status={node_status}",
                node_status=node_status,
            )
        # 3. Node converged (`ok`) but no extractable scalar → qualitative.
        elif spec.get("node") and node_status == "ok":
            entry.update(status="measured",
                         qualitative=True,
                         reason=f"{spec['node']} status=ok (qualitative)")
            if hold:
                entry["qa_hold_caveat"] = hold
        # 4. QA hold + no node info + no row → still withhold on hold.
        elif hold:
            entry.update(status="withheld_qa_hold", reason=hold)
        # 5. No artifact present at all.
        elif spec.get("node") and node_status == "not_attempted":
            entry.update(status="not_attempted",
                         reason=f"{spec['node']} missing or unparseable")
        else:
            entry.update(status="not_attempted",
                         reason="no node and no published_params row")

        # Provenance-quality overrides: extinction is `measured` only if
        # its provenance is in the accepted set.
        if family == "extinction" and entry.get("status") == "measured":
            ext = _read_json(run_dir / "02g_extinction.json") or {}
            prov = str(ext.get("provenance") or "").lower()
            from .prompts.wd_domain import ACCEPTED_EXTINCTION_PROVENANCES
            if not any(p in prov for p in ACCEPTED_EXTINCTION_PROVENANCES):
                entry.update(status="withheld_unsupported_prov",
                             reason=f"A_V provenance `{prov}` not in accepted list",
                             provenance=prov)

        by_family[family] = entry

    # Per-section visibility: only families that are measured AND in the
    # section's allowlist are "available". The drafter sees both lists.
    by_section: Dict[str, Dict[str, List[str]]] = {}
    for section, fams in SECTION_FAMILIES.items():
        measured: List[str] = []
        withheld: List[str] = []
        for fam in fams:
            st = (by_family.get(fam) or {}).get("status")
            if st == "measured":
                measured.append(fam)
            elif st and st.startswith("withheld") or st == "not_attempted":
                withheld.append(fam)
        by_section[section] = {"measured": measured, "withheld": withheld}

    # Withheld-sentence templates so the drafter can copy them verbatim.
    withheld_sentences: Dict[str, str] = {}
    for section, info in by_section.items():
        if not info["withheld"]:
            continue
        clauses = []
        for fam in info["withheld"]:
            entry = by_family.get(fam) or {}
            reason = entry.get("reason") or entry.get("status")
            clauses.append(f"the {fam.replace('_', ' ')} ({reason})")
        if clauses:
            withheld_sentences[section] = (
                "We withhold reporting "
                + "; ".join(clauses)
                + " pending the corresponding artifact becoming available."
            )

    return {
        "build_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_dir": str(run_dir),
        "qa_gate": qa.get("apj_gate"),
        "by_parameter_family": by_family,
        "by_section": by_section,
        "withheld_sentences": withheld_sentences,
        "section_families_table": SECTION_FAMILIES,
    }


def render_for_drafter(manifest: Mapping[str, Any], section: str) -> str:
    """Compact text block to inject at the top of a section's evidence."""
    by_section = (manifest.get("by_section") or {}).get(section) or {}
    measured = by_section.get("measured") or []
    withheld = by_section.get("withheld") or []
    by_fam = manifest.get("by_parameter_family") or {}
    lines = [
        f"#### Evidence availability for `{section}`",
        "STRICT RULE: you may quote a parameter family ONLY if it appears in",
        "the `measured` list below. For every family in `withheld`/`not_attempted`",
        "you MUST emit a `withheld pending` sentence rather than fabricate a",
        "value. Copy from `withheld_sentences` verbatim if helpful.",
        "",
        f"- measured: {measured or 'none'}",
        f"- withheld:  {withheld or 'none'}",
    ]
    if withheld:
        lines.append("")
        lines.append("Per-family withhold reasons:")
        for fam in withheld:
            entry = by_fam.get(fam) or {}
            lines.append(f"  * {fam}: {entry.get('reason') or entry.get('status')}")
    if (manifest.get("withheld_sentences") or {}).get(section):
        lines.append("")
        lines.append("Suggested withholding sentence:")
        lines.append(f"  > {manifest['withheld_sentences'][section]}")
    return "\n".join(lines) + "\n"


def is_family_measured(manifest: Mapping[str, Any], family: str) -> bool:
    fam = (manifest.get("by_parameter_family") or {}).get(family) or {}
    return fam.get("status") == "measured"


def withheld_sentence(manifest: Mapping[str, Any], section: str) -> str:
    return (manifest.get("withheld_sentences") or {}).get(section, "")


__all__ = [
    "NODE_ARTIFACTS", "BAD_STATUSES", "SECTION_FAMILIES", "FAMILY_SOURCES",
    "build_manifest", "render_for_drafter",
    "is_family_measured", "withheld_sentence",
]
