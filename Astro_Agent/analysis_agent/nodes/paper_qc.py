"""Paper QC checklist (Plan A6).

Run a deterministic set of checks on the drafted manuscript.  Each check
returns a `pass`/`fail`/`warn` verdict with a short reason.  The verdict
is recorded in `state['paper_qc']` and surfaced in the frontend workflow
trace next to the drafter step.

These checks are intentionally cheap (regex + file existence + LaTeX
balance) so that they can run after every drafter pass and gate the
peer_reviewer step.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


_REQUIRED_SECTIONS = (
    "Abstract",
    "Introduction",
    "Data",
    "Methods",
    "Results",
    "Discussion",
    "Conclusions",
)

_NUMBER_WITH_UNIT = re.compile(
    r"(?<![A-Za-z_\\])"
    r"-?\d+(?:\.\d+)?"
    r"\s*"
    r"(K|M_?\\?odot|M_?⊙|M_?sun|pc|kpc|min|minute|minutes|"
    r"km/s|km\\,s|km\\,s\\$\\^\\{-1\\}\\$|h|hr|hours|days?|Myr|Gyr|"
    r"mas/yr|mas\\,yr|mas|deg|°|R_?sun|R_?⊙|L_?sun|L_?⊙)"
)

_COORD_TOL_DEG = 0.01


def _read_text(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _count(pattern: str, text: str, flags: int = re.IGNORECASE) -> int:
    return len(re.findall(pattern, text, flags=flags))


_CITE_COMMAND = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")


def _extract_cite_keys(tex: str) -> List[str]:
    """Extract citation keys, including forms with optional cite arguments."""
    keys: List[str] = []
    for chunk in _CITE_COMMAND.findall(tex or ""):
        for key in chunk.split(","):
            key = key.strip()
            if key:
                keys.append(key)
    return keys


def _is_ads_bibcode(key: str) -> bool:
    if not key or len(key) != 19:
        return False
    if not re.match(r"^\d{4}[A-Za-z0-9&.]{15}$", key):
        return False
    upper = key.upper()
    if any(token in upper for token in ("XXX", "YYY", "TBD", "UPK13C2")):
        return False
    return bool(re.search(r"\d", key[4:]))


def _check_section_presence(tex: str) -> Dict[str, Any]:
    missing: List[str] = []
    for name in _REQUIRED_SECTIONS:
        if name.lower() == "abstract":
            ok = r"\begin{abstract}" in tex
        else:
            ok = bool(re.search(rf"\\section\{{\s*{re.escape(name)}\s*\}}", tex))
        if not ok:
            missing.append(name)
    return {
        "id": "sections_present",
        "verdict": "pass" if not missing else "fail",
        "reason": "All required sections present" if not missing
            else f"Missing sections: {', '.join(missing)}",
    }


def _check_abstract_length(tex: str) -> Dict[str, Any]:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
    if not m:
        return {"id": "abstract_length", "verdict": "fail", "reason": "no abstract block"}
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", m.group(1))
    words = [w for w in re.findall(r"\b[A-Za-z][A-Za-z\-']+", body) if len(w) > 1]
    n = len(words)
    if 120 <= n <= 350:
        return {"id": "abstract_length", "verdict": "pass", "reason": f"{n} words"}
    if 80 <= n < 120 or 350 < n <= 450:
        return {"id": "abstract_length", "verdict": "warn", "reason": f"{n} words (target 120-350)"}
    return {"id": "abstract_length", "verdict": "fail", "reason": f"{n} words (target 120-350)"}


def _check_abstract_numerics(tex: str) -> Dict[str, Any]:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
    if not m:
        return {"id": "abstract_numerics", "verdict": "fail", "reason": "no abstract block"}
    body = m.group(1)
    hits = _NUMBER_WITH_UNIT.findall(body)
    n = len(hits)
    if n >= 2:
        return {"id": "abstract_numerics", "verdict": "pass", "reason": f"{n} numeric+unit phrases"}
    if n == 1:
        return {"id": "abstract_numerics", "verdict": "warn", "reason": "only 1 numeric+unit phrase"}
    return {
        "id": "abstract_numerics",
        "verdict": "fail",
        "reason": "no numeric+unit phrase in abstract (target: orbit period, Teff, mass, etc.)",
    }


def _check_citations_per_section(tex: str) -> Dict[str, Any]:
    weak: List[str] = []
    for name in ("Introduction", "Methods", "Discussion"):
        section_pat = (
            rf"\\section\{{\s*{re.escape(name)}\s*\}}(.*?)"
            rf"(?=\\section\{{|\\acknowledgments|\\end\{{document\}})"
        )
        m = re.search(section_pat, tex, flags=re.DOTALL)
        body = m.group(1) if m else ""
        if not _extract_cite_keys(body):
            weak.append(name)
    if not weak:
        return {"id": "citations_per_section", "verdict": "pass", "reason": "all key sections cite"}
    return {
        "id": "citations_per_section",
        "verdict": "warn" if len(weak) == 1 else "fail",
        "reason": f"Sections without \\cite*: {', '.join(weak)}",
    }


def _check_refs_bib(workspace_root: Optional[str]) -> Dict[str, Any]:
    if not workspace_root:
        return {"id": "refs_bib", "verdict": "warn", "reason": "no workspace path provided"}
    p = Path(workspace_root) / "refs.bib"
    if not p.exists():
        return {"id": "refs_bib", "verdict": "fail", "reason": "refs.bib missing"}
    text = p.read_text(encoding="utf-8", errors="replace")
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", text)
    if not keys:
        return {"id": "refs_bib", "verdict": "fail", "reason": "refs.bib has no entries"}
    malformed = [k for k in keys if not _is_ads_bibcode(k)]
    if malformed:
        return {
            "id": "refs_bib",
            "verdict": "fail",
            "reason": f"{len(malformed)}/{len(keys)} refs.bib keys are not valid 19-char ADS bibcodes (e.g. {malformed[:3]})",
        }
    return {"id": "refs_bib", "verdict": "pass", "reason": f"{len(keys)} entries, 100% valid ADS bibcodes"}


def _check_brace_balance(tex: str) -> Dict[str, Any]:
    # Count un-escaped { and }
    opens = len(re.findall(r"(?<!\\)\{", tex))
    closes = len(re.findall(r"(?<!\\)\}", tex))
    if opens == closes:
        return {"id": "latex_brace_balance", "verdict": "pass", "reason": f"{opens} pairs"}
    return {
        "id": "latex_brace_balance",
        "verdict": "fail",
        "reason": f"unbalanced: {{ count={opens}, }} count={closes}",
    }


def _check_undefined_bibkey(tex: str, workspace_root: Optional[str]) -> Dict[str, Any]:
    keys = _extract_cite_keys(tex)
    if not keys:
        return {"id": "bibkey_coverage", "verdict": "warn", "reason": "no \\cite* keys used"}
    bib_keys: set = set()
    if workspace_root:
        p = Path(workspace_root) / "refs.bib"
        if p.exists():
            bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", p.read_text(encoding="utf-8", errors="replace")))
    missing = [k for k in keys if k not in bib_keys]
    if not missing:
        return {"id": "bibkey_coverage", "verdict": "pass", "reason": f"all {len(keys)} cite keys defined"}
    return {
        "id": "bibkey_coverage",
        "verdict": "fail",
        "reason": f"{len(missing)} cite keys not in refs.bib (e.g. {missing[:3]})",
    }


def _check_uncertainty_language(tex: str) -> Dict[str, Any]:
    m = re.search(
        r"\\section\{\s*Results\s*\}(.*?)(?=\\section\{|\\acknowledgments|\\end\{document\})",
        tex,
        flags=re.DOTALL,
    )
    if not m:
        return {"id": "uncertainty_language", "verdict": "warn", "reason": "no Results section to inspect"}
    body = m.group(1)
    if re.search(r"\\pm|±|uncertainty|systematic|sigma|\$\\sigma\$", body, flags=re.IGNORECASE):
        return {"id": "uncertainty_language", "verdict": "pass", "reason": "Results mentions uncertainty"}
    return {
        "id": "uncertainty_language",
        "verdict": "warn",
        "reason": "Results section has no \\pm / uncertainty / sigma language",
    }


def _check_hypothesis_articulated(tex: str, hypothesis_plan: Mapping[str, Any]) -> Dict[str, Any]:
    """A paper that does not consider at least one alternative interpretation
    is suspect; the UPK13-c2 paper's strength is its explicit WD+MS vs MS+MS
    hypothesis comparison.  Pass if the manuscript names at least two competing
    physical interpretations, or if the hypothesis plan has only one."""
    plan_hyps = (hypothesis_plan or {}).get("hypotheses") or []
    if len(plan_hyps) <= 1:
        return {
            "id": "hypothesis_articulated",
            "verdict": "warn",
            "reason": f"only {len(plan_hyps)} hypothesis defined for this source class",
        }
    # Count how many hypothesis labels appear in the manuscript
    labels = [h.get("label", "") for h in plan_hyps]
    found = sum(1 for label in labels if label and label.lower().split()[0] in tex.lower())
    if found >= 2:
        return {"id": "hypothesis_articulated", "verdict": "pass", "reason": f"{found} of {len(labels)} hypotheses appear in text"}
    if found == 1:
        return {"id": "hypothesis_articulated", "verdict": "warn", "reason": "only 1 hypothesis name found in text"}
    return {
        "id": "hypothesis_articulated",
        "verdict": "fail",
        "reason": "no hypothesis label from the plan found in the manuscript",
    }


def _check_cluster_membership_discussed(tex: str, cluster_membership: Mapping[str, Any]) -> Dict[str, Any]:
    cm = cluster_membership or {}
    if not cm or cm.get("status") != "ok" or not cm.get("candidates"):
        return {"id": "cluster_membership_discussed", "verdict": "warn", "reason": "no cluster_membership artifact"}
    # If we computed a chi^2_spat AND chi^2_kin, the manuscript should reference at least one of them.
    body = tex.lower()
    if "chi^2" in body or "chi^{2}" in body or "chi2" in body or r"\chi^{2}" in body or "membership" in body or "cluster" in body:
        return {"id": "cluster_membership_discussed", "verdict": "pass", "reason": "membership language present"}
    return {
        "id": "cluster_membership_discussed",
        "verdict": "warn",
        "reason": "cluster_membership computed but not referenced in manuscript",
    }


def _check_cluster_joint_criteria(tex: str, cluster_membership: Mapping[str, Any]) -> Dict[str, Any]:
    """Cluster membership must use spatial, kinematic, and traceback tests.

    Prior 9 in wd_domain says any missing/failing component rejects membership.
    The older QC only checked that the word "cluster" appeared, which allowed
    spatial_ok/kin_ok summaries without traceback-time support.
    """
    cm = cluster_membership or {}
    candidates = cm.get("candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return {"id": "cluster_joint_criteria", "verdict": "warn", "reason": "no cluster candidates to evaluate"}
    body = tex.lower()
    problems: List[str] = []
    for cand in candidates:
        if not isinstance(cand, Mapping):
            continue
        name = cand.get("name") or "<unnamed>"
        missing: List[str] = []
        if cand.get("chi2_spat") is None:
            missing.append("chi2_spat")
        if cand.get("chi2_kin") is None:
            missing.append("chi2_kin")
        if cand.get("traceback_time_myr") is None and cand.get("traceback_time") is None:
            missing.append("traceback_time")
        rv = cand.get("rv_offset_sigma")
        if rv is None or str(rv).lower() == "nan":
            missing.append("rv_offset_sigma")
        verdict_text = " ".join(str(v) for v in cand.get("verdict") or []).lower()
        if any(part in verdict_text for part in ("reject", "fail")):
            continue
        if missing:
            problems.append(f"{name} missing {', '.join(missing)}")
    if not problems:
        return {"id": "cluster_joint_criteria", "verdict": "pass",
                "reason": "cluster candidates include spatial/kinematic/traceback/RV criteria or explicit rejection"}
    rejects_in_text = bool(re.search(
        r"membership\s+(?:is\s+)?reject(?:ed)?"
        r"|reject(?:ed)?\s+(?:cluster\s+)?membership"
        r"|cannot\s+claim\s+(?:cluster\s+)?membership"
        r"|not\s+a\s+(?:cluster\s+)?member"
        r"|withhold(?:ing)?\s+(?:the\s+)?(?:cluster\s+)?membership"
        r"|membership\s+pending",
        body,
        flags=re.DOTALL,
    ))
    if rejects_in_text:
        return {"id": "cluster_joint_criteria", "verdict": "warn",
                "reason": f"artifact incomplete ({problems[:2]}), but manuscript uses rejection/withholding language"}
    return {"id": "cluster_joint_criteria", "verdict": "fail",
            "reason": "cluster membership lacks required joint criteria: " + "; ".join(problems[:3])}


def _check_novelty_paragraph(tex: str, published_params: Mapping[str, Any]) -> Dict[str, Any]:
    rows = published_params.get("rows", []) if isinstance(published_params, Mapping) else []
    n_this = sum(1 for r in rows if str(r.get("source_kind", "")).startswith("this_work"))
    n_lit = sum(1 for r in rows if r.get("source_kind") == "simbad_abstract")
    if n_this == 0 and n_lit == 0:
        return {"id": "novelty_paragraph", "verdict": "warn", "reason": "no params table to evaluate"}
    body = tex.lower()
    has_compare = bool(re.search(r"this work|in this paper|we (find|report|measure|confirm)", body))
    if has_compare and n_this >= 1 and n_lit >= 1:
        return {
            "id": "novelty_paragraph",
            "verdict": "pass",
            "reason": f"this_work={n_this}, lit={n_lit}, comparison language detected",
        }
    return {
        "id": "novelty_paragraph",
        "verdict": "warn",
        "reason": "missing explicit this-work vs literature comparison",
    }


# --------------------------------------------------------------------------- #
# Section-aware checks (added when SECTION_PROMPTS gained ApJ-style rules).    #
# Each new check inspects ONE section body and quantifies a single property:   #
# chi^2 density, uncertainty density, alternative-hypothesis language, intro   #
# motivation chain, and forbidden hype words.                                  #
# --------------------------------------------------------------------------- #

_FORBIDDEN_HYPE = (
    "obviously", "remarkable", "remarkably", "groundbreaking",
    "we believe", "novel result", "clearly demonstrates",
    "unprecedented", "incredibly", "amazing", "astonishing",
    "without doubt",
)


def _section_body(tex: str, name: str) -> str:
    """Return the LaTeX body of a single section, empty string if absent."""
    if name.lower() == "abstract":
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=re.DOTALL)
        return m.group(1) if m else ""
    pat = (
        rf"\\section\{{\s*{re.escape(name)}\s*\}}(.*?)"
        rf"(?=\\section\{{|\\acknowledgments|\\end\{{document\}})"
    )
    m = re.search(pat, tex, flags=re.DOTALL)
    return m.group(1) if m else ""


def _word_count(body: str) -> int:
    stripped = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", body)
    return len(re.findall(r"\b[A-Za-z][A-Za-z\-']+", stripped))


def _check_methods_chi2_density(tex: str) -> Dict[str, Any]:
    body = _section_body(tex, "Methods")
    if not body:
        return {"id": "methods_chi2_density", "verdict": "warn", "reason": "no Methods section"}
    n_chi = len(re.findall(r"\\chi\^?\{?2\}?|\$\\chi\^2\$|chi\^2|chi\s*squared", body, flags=re.IGNORECASE))
    words = max(_word_count(body), 1)
    density = n_chi / words
    if n_chi >= 3 and density >= 0.005:
        return {"id": "methods_chi2_density", "verdict": "pass",
                "reason": f"{n_chi} chi^2 in {words} words (density={density:.4f})"}
    if n_chi >= 1:
        return {"id": "methods_chi2_density", "verdict": "warn",
                "reason": f"only {n_chi} chi^2 in {words} words; aim >=3 with density>=0.005"}
    return {"id": "methods_chi2_density", "verdict": "fail",
            "reason": "no chi^2 expression in Methods"}


def _check_results_uncertainty_density(tex: str) -> Dict[str, Any]:
    body = _section_body(tex, "Results")
    if not body:
        return {"id": "results_uncertainty_density", "verdict": "warn", "reason": "no Results section"}
    n_pm = len(re.findall(r"\\pm|\$\\pm\$|±", body))
    n_num = len(re.findall(r"(?<![A-Za-z_\\])\d+(?:\.\d+)?", body))
    if n_num == 0:
        return {"id": "results_uncertainty_density", "verdict": "warn",
                "reason": "Results has no numerical values"}
    ratio = n_pm / n_num
    if ratio >= 0.5 and n_pm >= 3:
        return {"id": "results_uncertainty_density", "verdict": "pass",
                "reason": f"{n_pm} \\pm vs {n_num} numbers (ratio={ratio:.2f})"}
    if ratio >= 0.25 or n_pm >= 1:
        return {"id": "results_uncertainty_density", "verdict": "warn",
                "reason": f"{n_pm} \\pm vs {n_num} numbers; aim >=0.5 ratio"}
    return {"id": "results_uncertainty_density", "verdict": "fail",
            "reason": f"Results has {n_num} bare numbers and no \\pm"}


def _check_discussion_alternatives(tex: str) -> Dict[str, Any]:
    body = _section_body(tex, "Discussion")
    if not body:
        return {"id": "discussion_alternatives", "verdict": "warn", "reason": "no Discussion section"}
    pattern = r"alternativ(e|ely)|could also be|might also be|rule[- ]?out|cannot be ruled out|disfavour|disfavor"
    n = len(re.findall(pattern, body, flags=re.IGNORECASE))
    if n >= 2:
        return {"id": "discussion_alternatives", "verdict": "pass",
                "reason": f"{n} alternative-hypothesis phrases"}
    if n == 1:
        return {"id": "discussion_alternatives", "verdict": "warn",
                "reason": "only 1 alternative phrase; aim >=2"}
    return {"id": "discussion_alternatives", "verdict": "fail",
            "reason": "no alternative-hypothesis language in Discussion"}


def _check_intro_motivation_chain(tex: str) -> Dict[str, Any]:
    body = _section_body(tex, "Introduction")
    if not body:
        return {"id": "intro_motivation_chain", "verdict": "warn", "reason": "no Introduction section"}
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if _word_count(p) > 15]
    if len(paragraphs) < 3:
        return {"id": "intro_motivation_chain", "verdict": "warn",
                "reason": f"only {len(paragraphs)} substantive paragraphs; aim >=3"}
    n_cites = len(_extract_cite_keys(body))
    if n_cites < 4:
        return {"id": "intro_motivation_chain", "verdict": "warn",
                "reason": f"only {n_cites} citations; aim >=4"}
    return {"id": "intro_motivation_chain", "verdict": "pass",
            "reason": f"{len(paragraphs)} paragraphs, {n_cites} citations"}


def _check_forbidden_hype(tex: str) -> Dict[str, Any]:
    lower = tex.lower()
    found: List[str] = [w for w in _FORBIDDEN_HYPE if w in lower]
    if not found:
        return {"id": "forbidden_hype", "verdict": "pass", "reason": "no hype words"}
    if len(found) <= 1:
        return {"id": "forbidden_hype", "verdict": "warn",
                "reason": f"1 hype word: {found[0]}"}
    return {"id": "forbidden_hype", "verdict": "fail",
            "reason": f"{len(found)} hype words: {found[:4]}"}


# --------------------------------------------------------------------------- #
# Codex-derived checks (added after Codex reviewer pass flagged these         #
# as paper_qc blind spots).                                                   #
# --------------------------------------------------------------------------- #

def _check_bibkey_format(tex: str, workspace_root: Optional[str]) -> Dict[str, Any]:
    """Every \\citep key must be a real 19-char ADS bibcode shape, not a
    placeholder like `2025ApJ` or `2025ApJ...UPK13c2L`."""
    cites = _extract_cite_keys(tex)
    if not cites:
        return {"id": "bibkey_format", "verdict": "warn", "reason": "no \\cite* keys to validate"}
    bad: List[str] = []
    for k in cites:
        if not _is_ads_bibcode(k):
            bad.append(k)
    if not bad:
        return {"id": "bibkey_format", "verdict": "pass",
                "reason": f"all {len(cites)} cite keys are 19-char ADS bibcodes"}
    return {"id": "bibkey_format", "verdict": "fail",
            "reason": f"{len(bad)} non-ADS-shape keys (e.g. {bad[:3]})"}


def _check_target_identity_consistency(
    tex: str,
    resolved_target: Optional[Mapping[str, Any]],
    gold: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The (RA, Dec) quoted in the Abstract must match 01_resolved_target.json
    within 0.01 deg. Mismatch usually means the run pointed at the wrong
    object — the most catastrophic failure mode."""
    if not resolved_target or resolved_target.get("status") != "ok":
        if gold and (gold.get("ra_deg") is not None or gold.get("dec_deg") is not None):
            return {"id": "target_identity_consistency", "verdict": "fail",
                    "reason": "gold target coordinates exist but no resolved_target to compare against"}
        return {"id": "target_identity_consistency", "verdict": "warn",
                "reason": "no resolved_target to compare against"}
    ra_resolved = resolved_target.get("ra_deg")
    dec_resolved = resolved_target.get("dec_deg")
    if ra_resolved is None or dec_resolved is None:
        return {"id": "target_identity_consistency", "verdict": "warn",
                "reason": "resolved_target has no coordinates"}
    # Find every RA / Dec mention in the manuscript. Capture decimal degrees.
    found_ra: List[float] = []
    found_dec: List[float] = []
    # Patterns: $\alpha=281.0806$, RA=281.0806, $\delta=-17.89$, Dec=-17.89.
    for m in re.finditer(r"(?:\\alpha|RA|R\.A\.)\s*=?\s*\$?\s*(-?\d+(?:\.\d+)?)", tex):
        try:
            found_ra.append(float(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"(?:\\delta|Dec|DEC)\s*=?\s*\$?\s*(-?\d+(?:\.\d+)?)", tex):
        try:
            found_dec.append(float(m.group(1)))
        except Exception:
            pass
    if not found_ra or not found_dec:
        return {"id": "target_identity_consistency", "verdict": "warn",
                "reason": "no RA/Dec mention in manuscript"}
    mismatches = []
    for ra in found_ra:
        if abs(ra - ra_resolved) > _COORD_TOL_DEG:
            mismatches.append(f"RA={ra} vs resolved={ra_resolved}")
            break
    for dec in found_dec:
        if abs(dec - dec_resolved) > _COORD_TOL_DEG:
            mismatches.append(f"Dec={dec} vs resolved={dec_resolved}")
            break
    if gold:
        gold_ra = gold.get("ra_deg")
        gold_dec = gold.get("dec_deg")
        if gold_ra is not None:
            try:
                if abs(float(ra_resolved) - float(gold_ra)) > _COORD_TOL_DEG:
                    mismatches.append(f"resolved RA={ra_resolved} vs gold={gold_ra}")
            except Exception:
                pass
        if gold_dec is not None:
            try:
                if abs(float(dec_resolved) - float(gold_dec)) > _COORD_TOL_DEG:
                    mismatches.append(f"resolved Dec={dec_resolved} vs gold={gold_dec}")
            except Exception:
                pass
        aliases = [str(x).replace(" ", "").lower() for x in [gold.get("target")] + list(gold.get("alias") or []) if x]
        resolved_name = str(resolved_target.get("target") or "").replace(" ", "").lower()
        if aliases and resolved_name and not any(a in resolved_name or resolved_name in a for a in aliases):
            mismatches.append(f"resolved target `{resolved_target.get('target')}` not in gold aliases")
    if not mismatches:
        return {"id": "target_identity_consistency", "verdict": "pass",
                "reason": f"manuscript, resolved target, and gold coordinates agree within {_COORD_TOL_DEG} deg" if gold else
                f"manuscript RA/Dec matches resolved target within {_COORD_TOL_DEG} deg"}
    return {"id": "target_identity_consistency", "verdict": "fail",
            "reason": "; ".join(mismatches)}


_ACCEPTED_EXTINCTION_PROVENANCES = None  # populated lazily from wd_domain


def _accepted_ext_provenances() -> tuple:
    global _ACCEPTED_EXTINCTION_PROVENANCES
    if _ACCEPTED_EXTINCTION_PROVENANCES is None:
        try:
            from ..prompts.wd_domain import ACCEPTED_EXTINCTION_PROVENANCES as _A
            _ACCEPTED_EXTINCTION_PROVENANCES = tuple(_A)
        except Exception:
            _ACCEPTED_EXTINCTION_PROVENANCES = (
                "sfd", "planck", "green", "bayestar", "lallement", "3d_dust", "3d-dust",
            )
    return _ACCEPTED_EXTINCTION_PROVENANCES


def _check_extinction_provenance(
    extinction: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """A_V must come from SFD98 / Planck13 / Green19 / Lallement / 3D-dust.
    `fallback_latitude_scaling` is publication-disqualifying."""
    if not extinction or extinction.get("status") != "ok":
        return {"id": "extinction_provenance", "verdict": "warn",
                "reason": "no extinction artifact present"}
    prov = str(extinction.get("provenance") or "").lower()
    if not prov:
        return {"id": "extinction_provenance", "verdict": "fail",
                "reason": "extinction has no provenance field"}
    for accepted in _accepted_ext_provenances():
        if accepted in prov:
            return {"id": "extinction_provenance", "verdict": "pass",
                    "reason": f"A_V provenance accepted: {prov}"}
    return {"id": "extinction_provenance", "verdict": "fail",
            "reason": f"A_V provenance `{prov}` is not SFD98/Planck13/Green19/Lallement/3D-dust"}


def _check_literature_consistency(
    tex: str, published_params: Mapping[str, Any]
) -> Dict[str, Any]:
    """If the manuscript claims it "confirms literature parameters" but the
    published_params table has zero literature rows, the claim is fabricated."""
    rows = published_params.get("rows", []) if isinstance(published_params, Mapping) else []
    n_lit = sum(1 for r in rows if r.get("source_kind") == "simbad_abstract")
    pat = (
        r"confirm[s]?\s+(?:previously\s+)?published"
        r"|confirm[s]?\s+literature"
        r"|reproduce[s]?\s+(?:the\s+)?published"
        r"|previously\s+published\s+parameter"
        r"|literature\s+parameters\s+mined"
    )
    has_claim = bool(re.search(pat, tex, flags=re.IGNORECASE))
    if not has_claim:
        return {"id": "literature_consistency", "verdict": "pass",
                "reason": "no literature-confirmation claim"}
    if n_lit >= 1:
        return {"id": "literature_consistency", "verdict": "pass",
                "reason": f"claim is grounded by {n_lit} literature rows"}
    return {"id": "literature_consistency", "verdict": "fail",
            "reason": "manuscript claims literature confirmation but published_params has 0 literature rows"}


def _check_physics_checks_integration(
    physics_checks: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Surface the physics_checks summary inside paper_qc so failures there
    cannot be silently ignored by drafter / reviewer."""
    if not physics_checks:
        return {"id": "physics_checks_integration", "verdict": "warn",
                "reason": "no physics_checks artifact"}
    status = str(physics_checks.get("status") or "").lower()
    if status in {"fail", "failed", "error", "nonconverged", "no_inputs"}:
        return {"id": "physics_checks_integration", "verdict": "fail",
                "reason": f"physics_checks artifact status is {status}"}
    # physics_checks emits different shapes across versions. Accept either:
    #   (a) per-check ids at top level / under "checks": rayleigh_jeans, ...
    #   (b) a `sections` array of {name, verdict, ...}
    known = ("rayleigh_jeans", "ingress_time", "tidal_truncation", "mass_lum_sanity")
    fails: List[str] = []
    seen = 0
    for cid in known:
        info = physics_checks.get(cid)
        if not isinstance(info, Mapping):
            sub = physics_checks.get("checks")
            if isinstance(sub, Mapping):
                info = sub.get(cid)
        if not isinstance(info, Mapping):
            continue
        seen += 1
        verdict = info.get("verdict") or ("pass" if info.get("status") == "ok" else "warn")
        if verdict == "fail":
            fails.append(cid)
    if seen == 0:
        # Fallback: scan a `sections` array (newer physics_checks schema).
        sections = physics_checks.get("sections") or []
        if isinstance(sections, list):
            for s in sections:
                if not isinstance(s, Mapping):
                    continue
                seen += 1
                if s.get("verdict") == "fail":
                    fails.append(s.get("name") or s.get("id") or "<anonymous>")
    if seen == 0:
        return {"id": "physics_checks_integration", "verdict": "warn",
                "reason": "physics_checks present but no recognised check ids or sections"}
    if fails:
        return {"id": "physics_checks_integration", "verdict": "fail",
                "reason": f"{len(fails)}/{seen} physics checks failed: {fails}"}
    return {"id": "physics_checks_integration", "verdict": "pass",
            "reason": f"all {seen} physics checks pass"}


def _load_json_maybe(path: Optional[str]) -> Optional[Mapping[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    return obj if isinstance(obj, Mapping) else None


def _pick_gold_from_name(run_name: str, target_name: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    gold_dir = Path(__file__).resolve().parents[2] / "scripts" / "ablation" / "golds"
    if not gold_dir.exists():
        return None
    haystacks = [run_name or "", target_name or ""]
    for p in sorted(gold_dir.glob("*_gold.json")):
        gold = _load_json_maybe(str(p))
        if not gold:
            continue
        names = [gold.get("target", "")] + list(gold.get("alias") or [])
        names.append(p.stem.replace("_gold", ""))
        for needle in names:
            needle_norm = str(needle).replace(" ", "").lower()
            if needle_norm and any(needle_norm in h.replace(" ", "").lower() for h in haystacks):
                return gold
    return None


def run_paper_qc(
    *,
    final_tex_path: Optional[str],
    workspace_root: Optional[str],
    published_params_table: Optional[Mapping[str, Any]],
    hypothesis_plan: Optional[Mapping[str, Any]] = None,
    cluster_membership: Optional[Mapping[str, Any]] = None,
    resolved_target: Optional[Mapping[str, Any]] = None,
    gold: Optional[Mapping[str, Any]] = None,
    gold_path: Optional[str] = None,
    extinction: Optional[Mapping[str, Any]] = None,
    physics_checks: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all checks and return the result dict."""
    tex = _read_text(final_tex_path)
    checks: List[Dict[str, Any]] = []
    if not tex:
        return {
            "verdict": "fail",
            "summary": "no manuscript to QC",
            "checks": [],
            "n_pass": 0,
            "n_warn": 0,
            "n_fail": 1,
        }
    checks.append(_check_section_presence(tex))
    checks.append(_check_abstract_length(tex))
    checks.append(_check_abstract_numerics(tex))
    checks.append(_check_citations_per_section(tex))
    checks.append(_check_refs_bib(workspace_root))
    checks.append(_check_brace_balance(tex))
    checks.append(_check_undefined_bibkey(tex, workspace_root))
    checks.append(_check_uncertainty_language(tex))
    checks.append(_check_novelty_paragraph(tex, published_params_table or {"rows": []}))
    checks.append(_check_hypothesis_articulated(tex, hypothesis_plan or {}))
    checks.append(_check_cluster_membership_discussed(tex, cluster_membership or {}))
    checks.append(_check_cluster_joint_criteria(tex, cluster_membership or {}))
    checks.append(_check_methods_chi2_density(tex))
    checks.append(_check_results_uncertainty_density(tex))
    checks.append(_check_discussion_alternatives(tex))
    checks.append(_check_intro_motivation_chain(tex))
    checks.append(_check_forbidden_hype(tex))
    # Codex-derived blind-spot fixes:
    if gold is None:
        gold = _load_json_maybe(gold_path)
    if gold is None and resolved_target:
        gold = _pick_gold_from_name(
            str(resolved_target.get("target") or final_tex_path or ""),
            str(resolved_target.get("target") or ""),
        )
    checks.append(_check_bibkey_format(tex, workspace_root))
    checks.append(_check_target_identity_consistency(tex, resolved_target, gold))
    checks.append(_check_extinction_provenance(extinction))
    checks.append(_check_literature_consistency(tex, published_params_table or {"rows": []}))
    checks.append(_check_physics_checks_integration(physics_checks))

    n_pass = sum(1 for c in checks if c["verdict"] == "pass")
    n_warn = sum(1 for c in checks if c["verdict"] == "warn")
    n_fail = sum(1 for c in checks if c["verdict"] == "fail")
    if n_fail == 0 and n_warn <= 2:
        verdict = "pass"
    elif n_fail == 0:
        verdict = "warn"
    else:
        verdict = "fail"
    return {
        "verdict": verdict,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "checks": checks,
        "summary": f"{n_pass} pass / {n_warn} warn / {n_fail} fail",
        "manuscript_path": final_tex_path,
    }


__all__ = ["run_paper_qc"]
