"""QA Gate v2 — classified reasons.

Replaces the old flat ``reasons: [...]`` shape with four buckets so that
``route_after_qa`` can distinguish:
- infrastructure problems (dry-run, missing module_status, network) → toolbox_evolution
- science problems (non-stellar contaminant, unconverged fit on real data) → abnormal_report
- writing problems (missing refs, unsupported claim) → paper_repair
- pure dry-run → dry_run_summary

The legacy ``reasons`` key is kept (= union of all buckets) so existing
artifact consumers and tests stay green.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


# ---------------------------------------------------------------------------
# Reason classification
# ---------------------------------------------------------------------------

_INFRA_NEEDLES = (
    "dry-run",
    "dry_run",
    "module_status",
    "wd fitting output is unavailable",  # baseline produced no output
    "survey data have not been fetched",
    "no module_status.csv",
    "missing adapter",
    "network",
)

_WRITING_NEEDLES = (
    "missing refs",
    "refs.bib",
    "unsupported claim",
    "paper draft",
    "outline",
)


def _classify(reason: str) -> str:
    low = reason.lower()
    if any(n in low for n in _INFRA_NEEDLES):
        return "infrastructure"
    if any(n in low for n in _WRITING_NEEDLES):
        return "writing"
    return "science"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def classify_qa(
    legacy_qa: Mapping[str, Any],
    *,
    supervision: Optional[Mapping[str, Any]] = None,
    simbad_prior: Optional[Mapping[str, Any]] = None,
    dry_run: bool = False,
    modeling_skipped_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the v2 QA gate dict from legacy inputs."""
    infra: list[str] = []
    science: list[str] = []
    writing: list[str] = []
    blocking: list[str] = []
    warnings: list[str] = list(legacy_qa.get("warnings") or [])

    legacy_reasons = list(legacy_qa.get("reasons") or [])

    for r in legacy_reasons:
        bucket = _classify(r)
        # if modeling was skipped, do NOT call its derived reasons "science"
        if modeling_skipped_reason and "did not converge" in r.lower():
            infra.append(f"[{modeling_skipped_reason}] {r}")
            continue
        if modeling_skipped_reason and "unavailable" in r.lower():
            infra.append(f"[{modeling_skipped_reason}] {r}")
            continue
        if bucket == "infrastructure":
            infra.append(r)
        elif bucket == "writing":
            writing.append(r)
        else:
            science.append(r)

    if supervision and supervision.get("status") == "needs_repair":
        for trig in supervision.get("human_review_triggers", []) or []:
            science.append(trig)

    if dry_run:
        infra.append("workflow is in dry-run mode; survey data have not been fetched")

    if simbad_prior:
        sev = simbad_prior.get("severity")
        for flag in simbad_prior.get("science_flags", []) or []:
            if sev == "blocking":
                blocking.append(flag)
                science.append(flag)
            elif sev == "warning":
                warnings.append(flag)

    # de-dupe but preserve order
    def _uniq(xs: List[str]) -> List[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    infra = _uniq(infra)
    science = _uniq(science)
    writing = _uniq(writing)
    blocking = _uniq(blocking)
    warnings = _uniq(warnings)

    # ------------------------------------------------------------------
    # Gate decision + routing
    # ------------------------------------------------------------------
    if dry_run:
        apj_gate = "hold_for_human"
        recommended_branch = "dry_run_summary"
    elif blocking or science:
        apj_gate = "hold_for_human"
        recommended_branch = "abnormal_report"
    elif infra and not science:
        apj_gate = "hold_for_human"
        recommended_branch = "toolbox_evolution"
    elif writing and not (science or infra):
        apj_gate = "hold_for_human"
        recommended_branch = "paper_repair"
    else:
        apj_gate = "pass"
        recommended_branch = "paper"

    can_paper = apj_gate == "pass"
    can_outline = not (blocking or science)  # outlines may proceed unless science is bad
    can_abnormal = bool(blocking or science)

    union_reasons = infra + science + writing
    if not union_reasons and warnings:
        # preserve legacy behaviour: warnings alone keep the gate as legacy did
        pass

    return {
        "apj_gate": apj_gate,
        "human_review_required": apj_gate != "pass",
        "infrastructure_reasons": infra,
        "science_reasons": science,
        "writing_reasons": writing,
        "blocking_reasons": blocking,
        "warnings": warnings,
        "recommended_branch": recommended_branch,
        "can_generate_paper": can_paper,
        "can_generate_outline": can_outline,
        "can_generate_abnormal_report": can_abnormal,
        # backward-compat: legacy `reasons` is the union (existing readers keep working)
        "reasons": union_reasons,
        "checked_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }


def route_after_qa_v2(state: Mapping[str, Any]) -> str:
    qa = state.get("qa") or {}
    branch = qa.get("recommended_branch")
    if branch == "paper":
        return "paper"
    if branch == "dry_run_summary":
        return "dry_run_summary"
    if branch == "toolbox_evolution":
        return "toolbox_evolution_only"
    if branch == "paper_repair":
        return "paper_repair"
    return "abnormal"


def write_qa_artifact(state: Dict[str, Any], qa: Dict[str, Any]) -> Optional[str]:
    output_root = state.get("output_root")
    if not output_root:
        return None
    try:
        d = Path(output_root)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "08_qa_gate.json"
        path.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
        state.setdefault("artifacts", []).append(str(path))
        return str(path)
    except Exception:
        return None


__all__ = ["classify_qa", "route_after_qa_v2", "write_qa_artifact"]
