"""Novelty detector — structured this-work vs literature differential.

For every parameter we have measured in this run that also has at least
one literature value in `published_params`, compute:

  * absolute and signed Δ = (this_work − literature)
  * σ-equivalent if both have errors (otherwise label as `no_error_bars`)
  * verdict ∈ {confirm, tension, extend, new}
        confirm  — |Δ/σ| < 1
        tension  — |Δ/σ| > 3
        extend   — has this-work but no literature value (genuinely new)
        new      — neither side has the quantity but this run reports it

Returns a structured dict + LaTeX paragraph drafter can insert into Discussion.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Tuple


def _values_by_param(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group rows by parameter, then by 'this_work' vs 'literature'."""
    by_param: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: {"this_work": [], "literature": []}
    )
    for r in rows or []:
        param = r.get("parameter")
        if not param:
            continue
        kind = "literature" if r.get("source_kind") == "simbad_abstract" else "this_work"
        by_param[param][kind].append(r)
    return by_param


def _compare(this_val: float, this_err: float, lit_val: float, lit_err: float) -> Dict[str, Any]:
    delta = this_val - lit_val
    sigma_combined = math.sqrt((this_err or 0) ** 2 + (lit_err or 0) ** 2)
    if sigma_combined > 0:
        ratio = abs(delta) / sigma_combined
    else:
        ratio = None
    verdict = "no_error_bars"
    if ratio is not None:
        if ratio < 1:
            verdict = "confirm"
        elif ratio < 3:
            verdict = "consistent"
        else:
            verdict = "tension"
    return {
        "delta": delta,
        "sigma_combined": sigma_combined if sigma_combined > 0 else None,
        "delta_over_sigma": ratio,
        "verdict": verdict,
    }


def compute_novelty(published_params: Mapping[str, Any]) -> Dict[str, Any]:
    rows = list(published_params.get("rows") or [])
    by_param = _values_by_param(rows)
    items: List[Dict[str, Any]] = []
    for param, halves in by_param.items():
        tw = halves["this_work"]
        lit = halves["literature"]
        if tw and lit:
            # take median this-work value
            tw_sorted = sorted([t for t in tw if t.get("value") is not None],
                               key=lambda r: r["value"])
            lit_sorted = sorted([l for l in lit if l.get("value") is not None],
                                key=lambda r: r["value"])
            if not tw_sorted or not lit_sorted:
                continue
            tw_med = tw_sorted[len(tw_sorted) // 2]
            lit_med = lit_sorted[len(lit_sorted) // 2]
            cmp_ = _compare(
                float(tw_med["value"] or 0),
                float(tw_med.get("error") or 0),
                float(lit_med["value"] or 0),
                float(lit_med.get("error") or 0),
            )
            items.append({
                "parameter": param,
                "kind": "comparison",
                "this_work_value": tw_med.get("value"),
                "this_work_error": tw_med.get("error"),
                "this_work_unit": tw_med.get("unit"),
                "lit_value": lit_med.get("value"),
                "lit_error": lit_med.get("error"),
                "lit_bibcode": lit_med.get("bibcode"),
                **cmp_,
            })
        elif tw and not lit:
            for t in tw:
                if t.get("value") is None:
                    continue
                items.append({
                    "parameter": param,
                    "kind": "this_work_only",
                    "this_work_value": t.get("value"),
                    "this_work_error": t.get("error"),
                    "this_work_unit": t.get("unit"),
                    "verdict": "extend",
                })
                break
        elif lit and not tw:
            # Pure literature value (not measured here) — recorded for completeness
            items.append({
                "parameter": param,
                "kind": "literature_only",
                "lit_value": lit[0].get("value"),
                "lit_error": lit[0].get("error"),
                "lit_bibcode": lit[0].get("bibcode"),
                "verdict": "no_this_work_measurement",
            })
    # Bucket counts
    counts: Dict[str, int] = defaultdict(int)
    for it in items:
        counts[it.get("verdict") or "?"] += 1
    return {
        "items": items,
        "verdict_counts": dict(counts),
        "n_items": len(items),
    }


def render_latex(novelty: Mapping[str, Any], max_rows: int = 12) -> str:
    items = novelty.get("items") or []
    if not items:
        return ""
    lines = [
        r"\paragraph{Novelty assessment (this work vs.\ literature).}",
        (
            "Of the " + str(len(items)) + " parameters with at least one entry in "
            r"the published-parameter table, we find:"
        ),
        r"\begin{itemize}",
    ]
    # Summary counts
    for verdict, n in sorted((novelty.get("verdict_counts") or {}).items(),
                              key=lambda kv: -kv[1]):
        lines.append(f"  \\item {n} {verdict.replace('_', ' ')}.")
    lines.append(r"\end{itemize}")
    # Per-item detail
    lines.append(r"Per-parameter breakdown:")
    lines.append(r"\begin{itemize}")
    for it in items[:max_rows]:
        param = str(it.get("parameter", "?")).replace("_", r"\_")
        unit = it.get("this_work_unit") or ""
        if it.get("kind") == "comparison":
            tv = it.get("this_work_value"); te = it.get("this_work_error")
            lv = it.get("lit_value"); le = it.get("lit_error")
            bib = it.get("lit_bibcode")
            verdict = it.get("verdict")
            dos = it.get("delta_over_sigma")
            te_str = f" $\\pm$ {te:g}" if te is not None else ""
            le_str = f" $\\pm$ {le:g}" if le is not None else ""
            dos_str = f" ($\\Delta/\\sigma = {dos:.1f}$)" if dos is not None else ""
            lines.append(
                f"  \\item {param}: this work = {tv}{te_str} {unit}; "
                f"literature = {lv}{le_str} {unit} \\citep{{{bib}}} "
                f"$\\rightarrow$ \\textbf{{{verdict}}}{dos_str}."
            )
        elif it.get("kind") == "this_work_only":
            tv = it.get("this_work_value"); te = it.get("this_work_error")
            te_str = f" $\\pm$ {te:g}" if te is not None else ""
            lines.append(
                f"  \\item {param}: this work = {tv}{te_str} {unit} "
                r"$\rightarrow$ \textbf{extend} (no published value)."
            )
        elif it.get("kind") == "literature_only":
            lv = it.get("lit_value")
            bib = it.get("lit_bibcode")
            lines.append(
                f"  \\item {param}: literature = {lv} \\citep{{{bib}}}; "
                r"\textbf{not measured here}."
            )
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


__all__ = ["compute_novelty", "render_latex"]
