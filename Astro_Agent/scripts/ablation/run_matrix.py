"""Run the ablation matrix.

For each (config, target) cell we:
  1. start from the target's existing astrotool_run (so we DON'T re-fetch
     surveys — that part of the pipeline is not the variable under
     study);
  2. invoke run_workflow with the cell's flags;
  3. read 09_paper_qc.json, 02i_physics_checks.json, and final/paper.tex;
  4. compute gold-score via score_against_gold.score_manuscript;
  5. append one row to ablation_results.csv.

Flags-to-state mapping (the LangGraph state already supports most knobs;
the rest are surfaced as state booleans that workflow.py reads):

  prompt_variant            -> sets env PROMPT_VARIANT; the prompts
                               module reads it to choose overrides.
  specialists_split         -> when False, route every section to
                               'writer' specialist regardless of section.
  kg_enabled                -> when False, kg_navigator returns an empty
                               kg_results list.
  per_source_rag_enabled    -> when False, source_research_pipeline is
                               skipped.
  physics_checks_enabled    -> when False, physics_checks_node returns
                               a noop result.
  reflexion_max_iters       -> integer; existing knob.
  best_of_n                 -> integer; only honoured when the reward
                               model is loaded (PR5).

Usage:
  python run_matrix.py --rows baseline,loo_kg_off --targets UPK13c2
  python run_matrix.py --all       # the full 14 x 2 grid (slow)
  python run_matrix.py --dry-run   # echo what would run, no calls

Outputs: scripts/ablation/ablation_results.csv (appends; safe to re-run).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

PKG = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PKG.parent))   # add repo root for `Astro_Agent.*` imports
sys.path.insert(0, str(PKG))           # add Astro_Agent dir for `analysis_agent.*`

from analysis_agent.workflow import run_workflow  # noqa: E402
from analysis_agent.nodes import paper_qc as paper_qc_node  # noqa: E402
import score_against_gold  # type: ignore  # noqa: E402


RESULTS_CSV = Path(__file__).resolve().parent / "ablation_results.csv"


CSV_COLUMNS = [
    "timestamp",
    "config_id",
    "target_id",
    "output_root",
    "qc_pass", "qc_warn", "qc_fail", "qc_verdict",
    "physics_pass_rate",
    "numeric_iou", "claim_overlap", "bibcode_jaccard",
    "composite",
    "reviewer_overall",
    "wall_seconds",
    "error",
]


def load_matrix(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_state_kwargs(target_cfg: Dict[str, Any], flags: Dict[str, Any],
                       defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Translate flags + target into kwargs accepted by run_workflow."""
    out_root_template = defaults.get("output_root_template", "output/analysis_agent/ablation_{config_id}_{target_short}")
    output_root = out_root_template.format(
        config_id=flags["_config_id"], target_short=target_cfg["short"]
    )
    return {
        "target": target_cfg["target"],
        "ra_deg": target_cfg["ra_deg"],
        "dec_deg": target_cfg["dec_deg"],
        "output_root": output_root,
        "gold_path": target_cfg.get("gold"),
        "astrotool_run": target_cfg.get("astrotool_run"),
        "use_llm": defaults.get("use_llm", True),
        "llm_provider": defaults.get("provider", "claude"),
        # Ablation flags carried as state extras (read by workflow.py):
        "ablation_flags": {k: v for k, v in flags.items() if not k.startswith("_")},
        "reflexion_max_iters": int(flags.get("reflexion_max_iters", 1)),
        "best_of_n": int(flags.get("best_of_n", 1)),
        "kg_navigator_enabled": bool(flags.get("kg_enabled", True)),
        "source_research_package": bool(flags.get("per_source_rag_enabled", True)),
        "physics_checks_enabled": bool(flags.get("physics_checks_enabled", True)),
        "specialists_split_enabled": bool(flags.get("specialists_split", True)),
    }


def score_run(output_root: Path, gold_path: Path) -> Dict[str, Any]:
    """After run_workflow finishes, compute all metrics."""
    paper_tex = output_root / "paper_orchestra" / "final" / "paper.tex"
    qc_json = output_root / "09_paper_qc.json"
    physics_json = output_root / "02i_physics_checks.json"
    reviewer_json = output_root / "09c_reviewer.json"

    metrics: Dict[str, Any] = {
        "qc_pass": 0, "qc_warn": 0, "qc_fail": 0, "qc_verdict": "missing",
        "physics_pass_rate": None,
        "numeric_iou": 0.0, "claim_overlap": 0.0, "bibcode_jaccard": 0.0,
        "composite": 0.0,
        "reviewer_overall": None,
    }
    if qc_json.exists():
        qc = json.loads(qc_json.read_text(encoding="utf-8"))
        metrics["qc_pass"] = int(qc.get("n_pass") or 0)
        metrics["qc_warn"] = int(qc.get("n_warn") or 0)
        metrics["qc_fail"] = int(qc.get("n_fail") or 0)
        metrics["qc_verdict"] = qc.get("verdict") or "missing"
    if not paper_tex.exists():
        return metrics
    tex = paper_tex.read_text(encoding="utf-8", errors="replace")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    physics = json.loads(physics_json.read_text(encoding="utf-8")) if physics_json.exists() else None
    scored = score_against_gold.score_manuscript(tex, gold, physics)
    metrics.update({
        "numeric_iou": scored["numeric_iou"],
        "claim_overlap": scored["claim_overlap"],
        "bibcode_jaccard": scored["bibcode_jaccard"],
        "physics_pass_rate": scored["physics_pass_rate"],
        "composite": scored["composite"],
    })
    if reviewer_json.exists():
        try:
            rv = json.loads(reviewer_json.read_text(encoding="utf-8"))
            metrics["reviewer_overall"] = (rv.get("score") or {}).get("overall")
        except Exception:
            pass
    return metrics


def ensure_csv_header() -> None:
    if RESULTS_CSV.exists() and RESULTS_CSV.stat().st_size > 0:
        return
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def append_row(row: Dict[str, Any]) -> None:
    ensure_csv_header()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row.get(k, "") for k in CSV_COLUMNS])


def run_one(config: Dict[str, Any], target_cfg: Dict[str, Any],
            defaults: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    flags = dict(config["flags"])
    flags["_config_id"] = config["id"]
    kwargs = build_state_kwargs(target_cfg, flags, defaults)
    output_root = Path(kwargs["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["PROMPT_VARIANT"] = flags.get("prompt_variant", "wd_domain_v1")
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_id": config["id"],
        "target_id": target_cfg["id"],
        "output_root": str(output_root),
        "error": "",
    }
    print(f"\n=== {config['id']} x {target_cfg['id']} -> {output_root} ===")
    if dry_run:
        print("  (dry-run; flags:", json.dumps(flags, ensure_ascii=False), ")")
        row.update({"qc_pass": 0, "qc_warn": 0, "qc_fail": 0, "qc_verdict": "dry-run",
                    "composite": 0.0, "wall_seconds": 0})
        append_row(row)
        return row
    t0 = time.time()
    try:
        run_workflow(**kwargs)
    except TypeError:
        # run_workflow signature varies; fall back to passing the AnalysisState
        # constructor's accepted args only (target/ra/dec/output_root/use_llm).
        try:
            run_workflow(
                target=kwargs["target"],
                ra_deg=kwargs["ra_deg"],
                dec_deg=kwargs["dec_deg"],
                output_root=kwargs["output_root"],
                use_llm=kwargs["use_llm"],
                llm_provider=kwargs.get("llm_provider"),
                astrotool_run=kwargs.get("astrotool_run"),
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["wall_seconds"] = int(time.time() - t0)
    metrics = score_run(output_root, Path(target_cfg["gold"]))
    row.update(metrics)
    append_row(row)
    print(f"  qc {metrics['qc_pass']}/{metrics['qc_warn']}/{metrics['qc_fail']}  "
          f"composite={metrics['composite']:.3f}  wall={row['wall_seconds']}s "
          f"err={row['error'] or 'none'}")
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(Path(__file__).resolve().parent / "ablation_matrix.yaml"))
    ap.add_argument("--rows", default="", help="comma-sep config ids; empty = all")
    ap.add_argument("--targets", default="", help="comma-sep target ids; empty = all")
    ap.add_argument("--all", action="store_true", help="run the full grid")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matrix = load_matrix(Path(args.matrix))
    defaults = matrix.get("defaults") or {}
    configs = matrix["configs"]
    targets = matrix["targets"]
    if args.rows:
        wanted = set(args.rows.split(","))
        configs = [c for c in configs if c["id"] in wanted]
    if args.targets:
        twanted = set(args.targets.split(","))
        targets = [t for t in targets if t["id"] in twanted]
    if not args.all and not args.rows and not args.targets:
        print("Refusing to run all 28 cells without --all; pass --rows / --targets / --all.", file=sys.stderr)
        return 1
    print(f"Plan: {len(configs)} configs x {len(targets)} targets = {len(configs)*len(targets)} cells")
    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        for tgt in targets:
            rows.append(run_one(cfg, tgt, defaults, args.dry_run))
    print(f"\nDone. Wrote {len(rows)} rows to {RESULTS_CSV}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
