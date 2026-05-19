"""Structural lint (D7).

Static, post-drafter, pre-paper_qc pass that catches three Codex blocker
categories at the manuscript level rather than at QC time:

  1. every numeric+unit "we measure X = V unit" in Results that names a
     parameter must have a matching `published_params.rows` row with
     status `measured` in `evidence_manifest`. Otherwise flag as
     `unsupported_result_claim`.
  2. every `\\citep{<key>}` key must appear in `state.bibkey_allowlist`
     (D5 publishes this). Otherwise `unsupported_citation`.
  3. every `\\label{fig:X}` must be `\\ref{fig:X}`'d somewhere. (D6
     already auto-fixes this at assemble time; we still report any
     residual orphan as `orphan_figure`.)

Output is written to `<run>/09a_structural_lint.json`. Every violation
is also appended to `state.reflexion_history`-style `action_items`
(check_id=`structural_lint_<rule>`) so the existing reflexion loop
handles the rewrite without a new branch in the graph. The node never
raises and never blocks the workflow.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_CITE_KEY_RE = re.compile(r"\\cite[pt]?\*?\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")
_LABEL_RE = re.compile(r"\\label\{(fig:[^}]+)\}")
_REF_RE = re.compile(r"\\ref\{(fig:[^}]+)\}")

# Patterns of "we measure / we report / we obtain X = value unit". The
# parameter slug is the lowercased token immediately preceding `=`.
_MEASUREMENT_RE = re.compile(
    r"(?:we\s+(?:measure|report|obtain|find|derive|confirm)|this\s+work\s+(?:yields|gives))"
    r"[^.]{1,40}?"
    r"([A-Za-z_][A-Za-z0-9_]*?)\s*=\s*\$?\s*\-?\d+(?:\.\d+)?",
    flags=re.IGNORECASE,
)

# Map manuscript-parameter slugs to evidence_manifest family ids.
_SLUG_TO_FAMILY = {
    "parallax": "parallax",
    "plx": "parallax",
    "pmra": "proper_motion",
    "pmde": "proper_motion",
    "p_orb": "orbital_period",
    "porb": "orbital_period",
    "period": "orbital_period",
    "rv": "radial_velocity",
    "k_rv": "radial_velocity",
    "teff": "Teff_WD",
    "teff_wd": "Teff_WD",
    "m_wd": "M_WD",
    "mass": "M_WD",
    "a_v": "extinction",
    "av": "extinction",
}


def _extract_cite_keys(tex: str) -> List[str]:
    out: List[str] = []
    for chunk in _CITE_KEY_RE.findall(tex or ""):
        for key in chunk.split(","):
            key = key.strip()
            if key:
                out.append(key)
    return out


def _section_body(tex: str, name: str) -> str:
    pat = (
        rf"\\section\{{\s*{re.escape(name)}\s*\}}(.*?)"
        rf"(?=\\section\{{|\\acknowledgments|\\end\{{document\}})"
    )
    m = re.search(pat, tex, flags=re.DOTALL)
    return m.group(1) if m else ""


def lint_manuscript(
    tex: str,
    *,
    bibkey_allowlist: Optional[List[str]] = None,
    evidence_manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return {violations: [...], n_violations: int, ...}."""
    violations: List[Dict[str, Any]] = []
    allowlist = set(bibkey_allowlist or [])
    by_fam = (evidence_manifest or {}).get("by_parameter_family") or {}

    # Rule 1 — unsupported result claims.
    results_body = _section_body(tex, "Results")
    for match in _MEASUREMENT_RE.finditer(results_body):
        slug = match.group(1).lower().strip("_")
        family = _SLUG_TO_FAMILY.get(slug) or _SLUG_TO_FAMILY.get(slug.replace("$", ""))
        if family is None:
            # Unknown parameter slug — don't fail; just record as info.
            violations.append({
                "rule": "unknown_parameter_slug",
                "severity": "info",
                "snippet": match.group(0)[:120],
                "comment": f"could not map slug `{slug}` to a manifest family",
            })
            continue
        entry = by_fam.get(family) or {}
        if entry.get("status") != "measured":
            violations.append({
                "rule": "unsupported_result_claim",
                "severity": "blocker",
                "section": "Results",
                "snippet": match.group(0)[:160],
                "parameter_slug": slug,
                "family": family,
                "family_status": entry.get("status"),
                "comment": (
                    f"manuscript reports a value for `{slug}` but evidence_manifest "
                    f"marks family `{family}` as `{entry.get('status')}`: "
                    f"{entry.get('reason')}"
                ),
            })

    # Rule 2 — unsupported citations.
    if allowlist:
        for key in set(_extract_cite_keys(tex)):
            if key not in allowlist:
                violations.append({
                    "rule": "unsupported_citation",
                    "severity": "blocker",
                    "key": key,
                    "comment": f"\\citep{{{key}}} is not in bibkey_allowlist",
                })

    # Rule 3 — orphan figure labels.
    labels = set(_LABEL_RE.findall(tex))
    refs = set(_REF_RE.findall(tex))
    for lab in (labels - refs):
        violations.append({
            "rule": "orphan_figure",
            "severity": "major",
            "label": lab,
            "comment": f"\\label{{{lab}}} has no matching \\ref",
        })

    n_blocker = sum(1 for v in violations if v.get("severity") == "blocker")
    n_major = sum(1 for v in violations if v.get("severity") == "major")
    n_info = sum(1 for v in violations if v.get("severity") == "info")
    return {
        "verdict": "fail" if n_blocker else ("warn" if n_major else "pass"),
        "n_blocker": n_blocker,
        "n_major": n_major,
        "n_info": n_info,
        "n_violations": len(violations),
        "violations": violations,
    }


def run_structural_lint(
    *,
    manuscript_path: Path,
    bibkey_allowlist: Optional[List[str]] = None,
    evidence_manifest: Optional[Mapping[str, Any]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read the manuscript, run the lint, optionally persist."""
    tex = ""
    if manuscript_path and manuscript_path.exists():
        try:
            tex = manuscript_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            tex = ""
    if not tex:
        result = {"verdict": "warn", "reason": "no manuscript",
                  "violations": [], "n_violations": 0,
                  "n_blocker": 0, "n_major": 0, "n_info": 0}
    else:
        result = lint_manuscript(
            tex,
            bibkey_allowlist=bibkey_allowlist,
            evidence_manifest=evidence_manifest,
        )
    result["manuscript_path"] = str(manuscript_path) if manuscript_path else None
    if output_path is not None:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except Exception:
            pass
    return result


def to_reflexion_action_items(lint: Mapping[str, Any]) -> List[Dict[str, str]]:
    """Translate violations into reflexion action_items the drafter
    consumes via `_CHECK_TO_REWRITE`-style routing."""
    items: List[Dict[str, str]] = []
    for v in lint.get("violations") or []:
        rule = v.get("rule")
        severity = v.get("severity") or "minor"
        if rule == "unsupported_result_claim":
            items.append({
                "check_id": f"structural_lint_{rule}",
                "verdict": "fail",
                "section": v.get("section") or "Results",
                "reason": v.get("comment") or rule,
                "advice": (
                    "Remove the unsupported result. Replace it with the "
                    f"withheld-pending sentence for family `{v.get('family')}` "
                    "or with a literature value cited from refs.bib."
                ),
            })
        elif rule == "unsupported_citation":
            items.append({
                "check_id": f"structural_lint_{rule}",
                "verdict": "fail",
                "section": "Discussion",
                "reason": v.get("comment") or rule,
                "advice": (
                    f"\\citep{{{v.get('key')}}} is NOT in the bibkey allowlist. "
                    "Replace it with a key from the allowlist or delete the "
                    "citation entirely."
                ),
            })
        elif rule == "orphan_figure":
            items.append({
                "check_id": f"structural_lint_{rule}",
                "verdict": "warn",
                "section": "Data",
                "reason": v.get("comment") or rule,
                "advice": (
                    f"Reference \\label{{{v.get('label')}}} in the body or "
                    "remove the orphan figure block."
                ),
            })
    return items


__all__ = ["lint_manuscript", "run_structural_lint",
           "to_reflexion_action_items"]
