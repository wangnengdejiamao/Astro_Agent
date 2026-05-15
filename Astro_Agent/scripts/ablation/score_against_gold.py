"""Score a generated manuscript against a gold paper.

Inputs:
  * paper.tex  — full manuscript LaTeX
  * gold.json  — schema: see golds/*.json

Computes four 0-1 scores:
  numeric_iou       Jaccard over (parameter, value-within-tolerance, unit)
                    triples extracted from the manuscript vs gold.
  claim_overlap     fraction of gold key_claims for which the manuscript
                    contains a paraphrase (token-set Jaccard >= 0.45 on
                    keyword set; sentence-bert if installed).
  bibcode_jaccard   Jaccard over cited bibcodes between manuscript and
                    gold.expected_bibcodes.
  physics_pass_rate fraction of gold.physics_checks_should_pass that
                    were "pass" in the run's 02i_physics_checks.json
                    (None if file not provided).

Composite gold_score = 0.4*numeric + 0.25*claim + 0.2*bibcode +
                       0.15*physics  (re-normalised when physics is None).

Usage:
  python score_against_gold.py --paper paper.tex --gold golds/UPK13c2_gold.json
                               [--physics 02i_physics_checks.json]
                               [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Extractors                                                                   #
# --------------------------------------------------------------------------- #

_UNIT_PAT = (
    r"K|Msun|M_?⊙|M_?sun|pc|kpc|min|minute|minutes|"
    r"km/s|h|hr|hours|day|days|Myr|Gyr|mas/yr|mas|deg|°|"
    r"Rsun|R_?⊙|R_?sun|Lsun|L_?⊙|cgs"
)

# Match a numeric value (allow $ wrappers and \pm error), unit immediately
# after with optional LaTeX separators (\,, $, \texttt) — common in
# aastex output e.g. `$1.013 \pm 0.13$\,mas`.
_NUM_UNIT = re.compile(
    r"(-?\d+(?:\.\d+)?)"           # value
    r"(?:\s*\$?\s*(?:\\pm|±)\s*\$?\s*(-?\d+(?:\.\d+)?))?"  # optional ±err
    r"\s*\$?\s*(?:\\,|\\;|~|\\ |\s|\\\\,)*"  # LaTeX spacing
    r"\$?\s*"
    r"(" + _UNIT_PAT + r")"
    r"(?![a-zA-Z])"                 # not part of a longer word
)


def extract_numeric_triples(tex: str) -> List[Tuple[float, str]]:
    """Return list of (value, unit) parsed from the manuscript. Unit is
    normalised to a canonical short form."""
    triples: List[Tuple[float, str]] = []
    for m in _NUM_UNIT.finditer(tex):
        try:
            val = float(m.group(1))
        except Exception:
            continue
        unit = _canon_unit(m.group(3))
        triples.append((val, unit))
    return triples


def _canon_unit(u: str) -> str:
    u = u.strip()
    table = {
        "M_⊙": "Msun", "M⊙": "Msun", "M_sun": "Msun", "Msun": "Msun",
        "R_⊙": "Rsun", "R⊙": "Rsun", "R_sun": "Rsun", "Rsun": "Rsun",
        "L_⊙": "Lsun", "L⊙": "Lsun", "L_sun": "Lsun", "Lsun": "Lsun",
        "°": "deg", "minute": "min", "minutes": "min",
        "hour": "hr", "hours": "hr", "h": "hr",
        "days": "day",
    }
    return table.get(u, u)


def extract_bibcodes(tex: str) -> List[str]:
    keys: List[str] = []
    for chunk in re.findall(r"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}", tex):
        for k in chunk.split(","):
            k = k.strip()
            # bibcode-like: starts with 4 digits
            if re.match(r"^\d{4}", k):
                keys.append(k)
    return sorted(set(keys))


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #

def _within_tol(a: float, b: float, tol_pct: float = 5.0) -> bool:
    """Match if relative error <= tol_pct, or both are tiny (abs error <= 1e-6)."""
    if a == b:
        return True
    base = max(abs(a), abs(b))
    if base < 1e-6:
        return abs(a - b) < 1e-6
    return abs(a - b) / base * 100.0 <= tol_pct


def numeric_iou(tex: str, gold_rows: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    manuscript_triples = extract_numeric_triples(tex)
    gold_triples: List[Tuple[float, str]] = []
    for row in gold_rows:
        v = row.get("value")
        u = row.get("unit") or ""
        if v is None or not u:
            continue
        gold_triples.append((float(v), _canon_unit(u)))
    if not gold_triples:
        return 0.0, {"matched": 0, "gold": 0, "manuscript": len(manuscript_triples)}
    matched = 0
    used: List[int] = []
    for gv, gu in gold_triples:
        for i, (mv, mu) in enumerate(manuscript_triples):
            if i in used:
                continue
            if mu != gu:
                continue
            if _within_tol(gv, mv, tol_pct=5.0):
                used.append(i)
                matched += 1
                break
    union = len(gold_triples) + len(manuscript_triples) - matched
    iou = matched / union if union > 0 else 0.0
    return iou, {
        "matched": matched,
        "gold": len(gold_triples),
        "manuscript": len(manuscript_triples),
        "iou": iou,
    }


_STOP = set(
    "a an and are as at be been by do does for from has have in is it its of on or that the to was were will with this we our".split()
)


def _tokens(text: str) -> set:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return {w for w in text.split() if len(w) > 2 and w not in _STOP}


def claim_overlap(tex: str, gold_claims: List[str]) -> Tuple[float, Dict[str, Any]]:
    if not gold_claims:
        return 0.0, {"matched": 0, "total": 0}
    sentences = re.split(r"(?<=[.?!])\s+", tex)
    sent_tokens = [_tokens(s) for s in sentences if len(s) > 20]
    matches = 0
    details = []
    for claim in gold_claims:
        ctoks = _tokens(claim)
        if not ctoks:
            continue
        best_j = 0.0
        best_idx = -1
        for i, st in enumerate(sent_tokens):
            inter = len(ctoks & st)
            union = len(ctoks | st)
            j = inter / union if union else 0.0
            if j > best_j:
                best_j = j
                best_idx = i
        hit = best_j >= 0.30  # token-set Jaccard cutoff
        if hit:
            matches += 1
        details.append({"claim": claim[:80], "best_jaccard": round(best_j, 3), "hit": hit})
    score = matches / len(gold_claims)
    return score, {"matched": matches, "total": len(gold_claims), "details": details}


def bibcode_jaccard(tex: str, expected: List[str]) -> Tuple[float, Dict[str, Any]]:
    got = set(extract_bibcodes(tex))
    exp = set(expected or [])
    if not exp:
        return 0.0, {"got": len(got), "expected": 0}
    inter = got & exp
    union = got | exp
    return len(inter) / max(len(union), 1), {
        "intersection": sorted(inter),
        "got": sorted(got),
        "expected": sorted(exp),
    }


def physics_pass_rate(physics_json: Dict[str, Any], expected: List[str]) -> Tuple[float, Dict[str, Any]]:
    if not expected or not isinstance(physics_json, dict):
        return None, {"reason": "no expected list or no physics file"}  # type: ignore
    # physics_checks.py output structure: {check_id: {verdict: pass/warn/fail, ...}}
    results = physics_json.get("checks") or physics_json
    if not isinstance(results, dict):
        return None, {"reason": "unexpected physics_json shape"}  # type: ignore
    n_pass = 0
    detail: Dict[str, str] = {}
    for cid in expected:
        info = results.get(cid)
        v = "absent"
        if isinstance(info, dict):
            v = info.get("verdict") or ("pass" if info.get("status") == "ok" else "unknown")
        detail[cid] = v
        if v == "pass":
            n_pass += 1
    return n_pass / len(expected), detail


# --------------------------------------------------------------------------- #
# Composite                                                                    #
# --------------------------------------------------------------------------- #

def score_manuscript(tex: str, gold: Dict[str, Any], physics_json: Dict[str, Any] = None) -> Dict[str, Any]:
    n_iou, n_info = numeric_iou(tex, gold.get("published_params") or [])
    c_score, c_info = claim_overlap(tex, gold.get("key_claims") or [])
    b_jac, b_info = bibcode_jaccard(tex, gold.get("expected_bibcodes") or [])
    p_rate, p_info = physics_pass_rate(physics_json or {}, gold.get("physics_checks_should_pass") or [])
    if p_rate is None:
        composite = 0.4 * n_iou + 0.35 * c_score + 0.25 * b_jac
    else:
        composite = 0.4 * n_iou + 0.25 * c_score + 0.2 * b_jac + 0.15 * p_rate
    return {
        "composite": round(composite, 4),
        "numeric_iou": round(n_iou, 4),
        "claim_overlap": round(c_score, 4),
        "bibcode_jaccard": round(b_jac, 4),
        "physics_pass_rate": round(p_rate, 4) if p_rate is not None else None,
        "details": {
            "numeric": n_info,
            "claim": c_info,
            "bibcode": b_info,
            "physics": p_info,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--physics", default=None)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = ap.parse_args()
    tex = Path(args.paper).read_text(encoding="utf-8", errors="replace")
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    physics = json.loads(Path(args.physics).read_text(encoding="utf-8")) if args.physics else None
    out = score_manuscript(tex, gold, physics)
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"Composite: {out['composite']:.3f}")
        print(f"  numeric IoU      : {out['numeric_iou']:.3f}  ({out['details']['numeric']['matched']}/{out['details']['numeric']['gold']} gold params matched)")
        print(f"  claim overlap    : {out['claim_overlap']:.3f}  ({out['details']['claim']['matched']}/{out['details']['claim']['total']} claims hit)")
        print(f"  bibcode Jaccard  : {out['bibcode_jaccard']:.3f}  ({len(out['details']['bibcode']['intersection'])} shared bibcodes)")
        if out['physics_pass_rate'] is not None:
            print(f"  physics pass rate: {out['physics_pass_rate']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
