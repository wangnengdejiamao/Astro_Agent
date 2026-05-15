"""KG writeback.

At the end of every analysis run, append a small set of triples to a local
SQLite "learning ledger" so subsequent runs of the same source class can
look up:
  - which methods have been applied to which source class,
  - whether they converged or failed,
  - which physical parameters this run recorded for a given source.

This is intentionally a tiny side-ledger, not a write-back into the heavy
`white_dwarf_kg` knowledge graph (whose rebuild is expensive).  Plan B4
proper will later promote the ledger into Neo4j triples.

Tables:
  method_runs(method_name, source_id, source_class, status, timestamp, run_dir)
  param_extractions(source_id, parameter, value, error, unit, bibcode, source_kind, timestamp)
  hypothesis_results(source_id, source_class, hypothesis_name, ready_to_run, module_implemented, timestamp)
  cluster_membership(source_id, cluster_name, chi2_spat, chi2_kin, rv_offset_sigma, age_myr, timestamp)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS method_runs (
    method_name   TEXT,
    source_id     TEXT,
    source_class  TEXT,
    status        TEXT,
    timestamp     TEXT,
    run_dir       TEXT
);
CREATE TABLE IF NOT EXISTS param_extractions (
    source_id     TEXT,
    parameter     TEXT,
    value         REAL,
    error         REAL,
    unit          TEXT,
    bibcode       TEXT,
    source_kind   TEXT,
    timestamp     TEXT
);
CREATE TABLE IF NOT EXISTS hypothesis_results (
    source_id            TEXT,
    source_class         TEXT,
    hypothesis_name      TEXT,
    ready_to_run         INTEGER,
    module_implemented   INTEGER,
    timestamp            TEXT
);
CREATE TABLE IF NOT EXISTS cluster_membership_log (
    source_id        TEXT,
    cluster_name     TEXT,
    chi2_spat        REAL,
    chi2_kin         REAL,
    rv_offset_sigma  REAL,
    age_myr          REAL,
    timestamp        TEXT
);
CREATE INDEX IF NOT EXISTS idx_method_runs_source ON method_runs(source_id);
CREATE INDEX IF NOT EXISTS idx_param_source ON param_extractions(source_id);
"""


def _default_ledger_path() -> Path:
    pkg = Path(__file__).resolve().parent.parent
    return pkg / "output" / "analysis_agent" / "_learning_ledger.sqlite"


def write_run(
    *,
    source_id: str,
    state: Mapping[str, Any],
    run_dir: str,
    ledger_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write a run summary to the learning ledger. Returns a small report."""
    ledger_path = ledger_path or _default_ledger_path()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    plan = state.get("analysis_plan") or {}
    source_class = plan.get("source_class") or "unknown"
    pipeline_module = plan.get("fitting_pipeline_module") or "unknown"

    iter_status: List[Dict[str, Any]] = state.get("iterations") or []
    qa = state.get("qa") or {}
    qa_gate = qa.get("apj_gate") or "unknown"

    n_method = 0
    n_param = 0
    n_hyp = 0
    n_cluster = 0

    conn = sqlite3.connect(str(ledger_path))
    try:
        conn.executescript(_SCHEMA)
        # 1) method run: one row per (pipeline_module, source) summarizing QA verdict
        conn.execute(
            "INSERT INTO method_runs VALUES (?,?,?,?,?,?)",
            (
                pipeline_module,
                source_id,
                source_class,
                f"qa={qa_gate}; iterations={len(iter_status)}; mismatch={qa.get('model_mismatch')}",
                timestamp,
                run_dir,
            ),
        )
        n_method = 1

        # 2) every published_params row
        pp = state.get("published_params") or {}
        for row in pp.get("rows", []) or []:
            try:
                conn.execute(
                    "INSERT INTO param_extractions VALUES (?,?,?,?,?,?,?,?)",
                    (
                        source_id,
                        str(row.get("parameter")),
                        float(row["value"]) if row.get("value") is not None else None,
                        float(row["error"]) if row.get("error") is not None else None,
                        str(row.get("unit") or ""),
                        str(row.get("bibcode") or ""),
                        str(row.get("source_kind") or ""),
                        timestamp,
                    ),
                )
                n_param += 1
            except (TypeError, ValueError):
                continue

        # 3) hypothesis plan rows
        hp = state.get("hypothesis_plan") or {}
        for h in hp.get("hypotheses", []) or []:
            conn.execute(
                "INSERT INTO hypothesis_results VALUES (?,?,?,?,?,?)",
                (
                    source_id,
                    source_class,
                    str(h.get("name")),
                    1 if h.get("ready_to_run") else 0,
                    1 if h.get("module_implemented") else 0,
                    timestamp,
                ),
            )
            n_hyp += 1

        # 4) cluster membership candidates
        cm = state.get("cluster_membership") or {}
        for cand in cm.get("candidates", []) or []:
            def _f(x):
                try:
                    return float(x) if x is not None else None
                except (TypeError, ValueError):
                    return None
            conn.execute(
                "INSERT INTO cluster_membership_log VALUES (?,?,?,?,?,?,?)",
                (
                    source_id,
                    str(cand.get("name") or ""),
                    _f(cand.get("chi2_spat")),
                    _f(cand.get("chi2_kin")),
                    _f(cand.get("rv_offset_sigma")),
                    _f(cand.get("cluster_age_myr")),
                    timestamp,
                ),
            )
            n_cluster += 1

        conn.commit()
    finally:
        conn.close()

    return {
        "status": "ok",
        "ledger_path": str(ledger_path),
        "timestamp": timestamp,
        "source_id": source_id,
        "source_class": source_class,
        "n_method_run_rows": n_method,
        "n_param_rows": n_param,
        "n_hypothesis_rows": n_hyp,
        "n_cluster_rows": n_cluster,
    }


def query_method_success_rate(
    *,
    source_class: Optional[str] = None,
    ledger_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return success-rate by method, optionally restricted to a source class.

    A "successful" row is one whose status contains 'qa=clear_for_draft'.
    """
    ledger_path = ledger_path or _default_ledger_path()
    if not Path(ledger_path).exists():
        return []
    conn = sqlite3.connect(str(ledger_path))
    conn.row_factory = sqlite3.Row
    try:
        if source_class:
            rows = conn.execute(
                "SELECT method_name, "
                "COUNT(*) AS n_total, "
                "SUM(CASE WHEN status LIKE '%qa=clear_for_draft%' THEN 1 ELSE 0 END) AS n_success "
                "FROM method_runs WHERE source_class = ? GROUP BY method_name",
                (source_class,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT method_name, source_class, "
                "COUNT(*) AS n_total, "
                "SUM(CASE WHEN status LIKE '%qa=clear_for_draft%' THEN 1 ELSE 0 END) AS n_success "
                "FROM method_runs GROUP BY method_name, source_class",
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


__all__ = ["write_run", "query_method_success_rate"]
