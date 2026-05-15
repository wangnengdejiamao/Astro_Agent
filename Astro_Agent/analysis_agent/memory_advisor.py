"""Memory advisor — CORAL-style persistent shared memory query.

For each new run, query the shared learning ledger (built by kg_writeback)
and produce an advisory packet:

  - which fitting methods have historically succeeded on this source class,
  - which hypotheses have already been validated,
  - which cluster candidates the agent has seen on similar sources.

The advisor's output is appended to the structure_planner's evidence so the
planner picks methods grounded in run-history, not only in static rules.
This converts the agent from "stateless per-run" to "stateful across runs",
which is the CORAL pattern reported in mid-2026.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _ledger_path(default_root: Path) -> Path:
    return default_root / "_learning_ledger.sqlite"


def query_advice(
    *,
    source_class: str,
    source_id: Optional[str] = None,
    ledger_path: Optional[Path] = None,
    output_root_for_default: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return advice for the structure_planner / drafter.

    Output keys:
      - method_success_rate: list of {method, n_total, n_success, success_rate}
      - prior_hypotheses_verified: list of (source_class, hypothesis_name, n_runs_module_implemented_and_ready)
      - prior_cluster_hits: list of clusters that have produced plausible
            (chi2_spat<9 AND chi2_kin<12) matches for THIS source class
      - prior_params_for_same_source: list of params already seen for this source_id
    """
    if ledger_path is None:
        if output_root_for_default is not None:
            ledger_path = _ledger_path(output_root_for_default.parent)
        else:
            return {"status": "no_ledger_path"}
    if not Path(ledger_path).exists():
        return {"status": "no_ledger_yet", "ledger_path": str(ledger_path)}

    conn = sqlite3.connect(str(ledger_path))
    conn.row_factory = sqlite3.Row
    out: Dict[str, Any] = {"status": "ok", "ledger_path": str(ledger_path)}
    try:
        # 1) Method success rate scoped to this source class
        method_rows: List[Dict[str, Any]] = []
        for row in conn.execute(
            "SELECT method_name, "
            "  COUNT(*) AS n_total, "
            "  SUM(CASE WHEN status LIKE '%qa=clear_for_draft%' THEN 1 ELSE 0 END) AS n_success, "
            "  SUM(CASE WHEN status LIKE '%mismatch=True%' THEN 1 ELSE 0 END) AS n_mismatch "
            "FROM method_runs WHERE source_class = ? "
            "GROUP BY method_name ORDER BY n_success DESC, n_total DESC",
            (source_class,),
        ):
            n_total = int(row["n_total"] or 0)
            n_success = int(row["n_success"] or 0)
            rate = (n_success / n_total) if n_total else 0.0
            method_rows.append({
                "method_name": row["method_name"],
                "n_total": n_total,
                "n_success": n_success,
                "n_mismatch": int(row["n_mismatch"] or 0),
                "success_rate": rate,
            })
        out["method_success_rate"] = method_rows
        if method_rows:
            out["recommended_method"] = max(
                method_rows,
                key=lambda r: (r["success_rate"], r["n_success"], -r["n_mismatch"]),
            )["method_name"]
        else:
            out["recommended_method"] = None

        # 2) Hypotheses that were marked ready_to_run AND module_implemented in
        # past runs of THIS source class — those are the "battle-tested" ones.
        hyp_rows: List[Dict[str, Any]] = []
        for row in conn.execute(
            "SELECT hypothesis_name, "
            "  COUNT(*) AS n_total, "
            "  SUM(ready_to_run) AS n_ready, "
            "  SUM(module_implemented) AS n_impl "
            "FROM hypothesis_results WHERE source_class = ? "
            "GROUP BY hypothesis_name "
            "ORDER BY n_impl DESC, n_ready DESC",
            (source_class,),
        ):
            hyp_rows.append({
                "hypothesis_name": row["hypothesis_name"],
                "n_total": int(row["n_total"] or 0),
                "n_ready": int(row["n_ready"] or 0),
                "n_impl": int(row["n_impl"] or 0),
            })
        out["prior_hypotheses_verified"] = hyp_rows

        # 3) Cluster candidates that have been plausible hosts before for
        # sources in this class — useful prior for new candidate matches.
        cluster_rows: List[Dict[str, Any]] = []
        for row in conn.execute(
            "SELECT cm.cluster_name, COUNT(*) AS n_hits "
            "FROM cluster_membership_log cm "
            "JOIN method_runs mr ON cm.source_id = mr.source_id "
            "WHERE mr.source_class = ? "
            "  AND cm.chi2_spat IS NOT NULL AND cm.chi2_spat < 9.0 "
            "  AND cm.chi2_kin  IS NOT NULL AND cm.chi2_kin  < 12.0 "
            "GROUP BY cm.cluster_name ORDER BY n_hits DESC",
            (source_class,),
        ):
            cluster_rows.append({
                "cluster_name": row["cluster_name"],
                "n_plausible_hits_in_class": int(row["n_hits"] or 0),
            })
        out["prior_cluster_hits"] = cluster_rows[:10]

        # 4) Previously recorded parameters for the SAME source_id (replay)
        prior_rows: List[Dict[str, Any]] = []
        if source_id:
            for row in conn.execute(
                "SELECT parameter, value, error, unit, bibcode, source_kind, timestamp "
                "FROM param_extractions WHERE source_id = ? "
                "ORDER BY timestamp DESC LIMIT 50",
                (source_id,),
            ):
                prior_rows.append(dict(row))
        out["prior_params_for_same_source"] = prior_rows
        out["n_prior_runs_for_same_source"] = len({r["timestamp"] for r in prior_rows})
    finally:
        conn.close()
    return out


def render_markdown(advice: Mapping[str, Any]) -> str:
    if advice.get("status") != "ok":
        return f"_(no advisor data: {advice.get('status')})_"
    lines = [
        "## Memory Advisor (cross-run learning ledger)",
        "",
        f"- ledger: `{advice.get('ledger_path')}`",
    ]
    msr = advice.get("method_success_rate") or []
    if msr:
        lines.append("- method success rate for this source class:")
        for r in msr:
            lines.append(
                f"  - `{r['method_name']}`: {r['n_success']}/{r['n_total']} "
                f"({r['success_rate']*100:.0f}% success, {r['n_mismatch']} model_mismatch)"
            )
        if advice.get("recommended_method"):
            lines.append(f"- recommended method: `{advice.get('recommended_method')}`")
    hyps = advice.get("prior_hypotheses_verified") or []
    if hyps:
        lines.append("- prior hypotheses seen for this source class:")
        for h in hyps[:6]:
            lines.append(
                f"  - `{h['hypothesis_name']}`: implemented in {h['n_impl']} of {h['n_total']} runs"
            )
    cm = advice.get("prior_cluster_hits") or []
    if cm:
        lines.append("- cluster candidates with plausible hits in this class:")
        for c in cm[:5]:
            lines.append(f"  - `{c['cluster_name']}` ({c['n_plausible_hits_in_class']} prior hits)")
    prior_params = advice.get("prior_params_for_same_source") or []
    if prior_params:
        lines.append(f"- {len(prior_params)} prior param rows for this source_id "
                     f"in {advice.get('n_prior_runs_for_same_source')} previous run(s)")
    return "\n".join(lines) + "\n"


__all__ = ["query_advice", "render_markdown"]
