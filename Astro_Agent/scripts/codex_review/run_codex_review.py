"""Run Codex as a five-persona reviewer over an existing run.

What this does:
  1. Loads the manuscript + all key artifact JSONs for a given run.
  2. Loads `analysis_agent/prompts/codex_reviewer_alignment.md` as system prompt.
  3. Calls Codex (via the user's local CLI — set via $CODEX_BIN, default `codex`)
     with the system prompt and a structured user message containing every
     artifact path + the gold paper.
  4. Parses Codex's JSON reply, validates it against the schema, and writes:
       - <run>/09c_reviewer.json       (the full reviewer dump)
       - appends to <run>/09b_reflexion.json `action_items`
         so the next agent re-run picks the advice up automatically.
  5. Prints a short summary table to stdout.

Why subprocess to the Codex CLI instead of an HTTP call: the user's
relay already authorises the CLI, and the five-persona prompt is large;
keeping it on disk and piping is more reproducible than embedding in
shell args.

Usage:
  python scripts/codex_review/run_codex_review.py --run UPK13c2_stage34
  python scripts/codex_review/run_codex_review.py --run ZTFJ2130_v21_stage34 \
      --gold scripts/ablation/golds/ZTFJ2130_gold.json
  python scripts/codex_review/run_codex_review.py --run X --dry-run  # echo prompt only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PKG = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PKG / "output" / "analysis_agent"
SYSTEM_PROMPT_PATH = PKG / "analysis_agent" / "prompts" / "codex_reviewer_alignment.md"
GOLD_DIR = PKG / "scripts" / "ablation" / "golds"
RERANK_KEYS_PATH = PKG / "analysis_agent" / "prompts" / "retrieval.py"

CODEX_BIN = os.getenv("CODEX_BIN", "codex")

if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

try:
    from analysis_agent.toolbox_kg_audit import run_toolbox_kg_audit
except Exception:  # pragma: no cover - runner should still work without package imports
    run_toolbox_kg_audit = None  # type: ignore


# Artifacts the five personas need to read.
ARTIFACT_KEYS = [
    ("manuscript",       "paper_orchestra/final/paper.tex"),
    ("manuscript_draft", "paper_orchestra/drafts/paper.tex"),  # fallback if final missing
    ("refs_bib",         "paper_orchestra/refs.bib"),
    ("data_fetch",       "02_data_fetch.json"),
    ("paper_qc",         "09_paper_qc.json"),
    ("physics_checks",   "02i_physics_checks.json"),
    ("published_params", "02c_published_params.json"),
    ("source_rag",       "02d_source_rag.json"),
    ("cluster_membership","02e_cluster_membership.json"),
    ("hypothesis_plan",  "02f_hypothesis_plan.json"),
    ("novelty",          "02m_novelty.json"),
    ("comparison_table", "02n_comparison_table.json"),
    ("qa_gate",          "08_qa_gate.json"),
    ("reflexion",        "09b_reflexion.json"),
    ("resolved_target",  "01_resolved_target.json"),
    ("analysis_plan",    "02b_analysis_plan.json"),
    # Persona E (Toolbox & KG auditor) inputs:
    ("rag_results",      "03_rag_results.json"),
    ("kg_results",       "04_kg_results.json"),
    ("kg_graph_report",  "04b_kg_graph_report.json"),
    ("method_scout",     "04c_method_scout.json"),
    ("toolbox_gap",      "04e_toolbox_gap.json"),
    ("iteration_baseline","05_iteration_1_baseline.json"),
    ("iteration_residuals","06_iteration_2_residuals.json"),
    ("iteration_systematics","07_iteration_3_systematics.json"),
    ("model_supervision","07b_model_supervision.json"),
    ("sed_decoupled",    "02h_sed_decoupled.json"),
    ("light_curve_geom", "02j_light_curve_geometry.json"),
    ("eclipse_mcmc",     "02k_eclipse_mcmc.json"),
    ("extinction",       "02g_extinction.json"),
    ("figures",          "11_figures.json"),
    ("latex_compile",    "12_latex_compile.json"),
]


def _read_text(p: Path) -> Optional[str]:
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_json(p: Path) -> Any:
    text = _read_text(p)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return {"_raw_text_truncated": text[:1500], "_parse_error": True}


def pick_gold(run_name: str, target_name: Optional[str], explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    # Heuristic: pick the gold whose `alias` or `target` substring appears in the run/target name.
    candidates = sorted(GOLD_DIR.glob("*_gold.json"))
    for g in candidates:
        try:
            obj = json.loads(g.read_text(encoding="utf-8"))
        except Exception:
            continue
        names = [obj.get("target", "")] + list(obj.get("alias") or [])
        for n in names:
            if n and (n.replace(" ", "").lower() in run_name.lower().replace(" ", "")
                      or (target_name and n.replace(" ", "").lower() in target_name.lower().replace(" ", ""))):
                return g
        # plain stem match (UPK13c2_gold -> UPK13c2)
        if g.stem.replace("_gold", "").lower() in run_name.lower():
            return g
    return candidates[0] if candidates else None


def collect_payload(run_name: str, gold_path: Optional[Path]) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_name
    if not run_dir.exists():
        raise SystemExit(f"run not found: {run_dir}")
    payload: Dict[str, Any] = {"run": run_name, "run_dir": str(run_dir)}
    # manuscript: prefer final/, fallback draft/
    final_tex = run_dir / "paper_orchestra" / "final" / "paper.tex"
    draft_tex = run_dir / "paper_orchestra" / "drafts" / "paper.tex"
    tex = _read_text(final_tex) or _read_text(draft_tex)
    payload["manuscript_path"] = str(final_tex if final_tex.exists() else draft_tex)
    payload["manuscript"] = tex or "(no paper.tex found)"
    payload["refs_bib"] = _read_text(run_dir / "paper_orchestra" / "refs.bib") or ""
    payload["artifacts"] = {}
    for key, rel in ARTIFACT_KEYS[3:]:  # skip the three we just handled
        p = run_dir / rel
        payload["artifacts"][key] = _read_json(p) if p.suffix == ".json" else _read_text(p)
    target = (payload["artifacts"].get("resolved_target") or {}).get("target") or run_name
    payload["target"] = target
    payload["gold_path"] = str(gold_path) if gold_path else None
    payload["gold"] = json.loads(gold_path.read_text(encoding="utf-8")) if gold_path else None
    payload["retrieval_prompt_path"] = str(RERANK_KEYS_PATH)
    payload["retrieval_prompt"] = _read_text(RERANK_KEYS_PATH) or ""
    return payload


def build_user_message(payload: Dict[str, Any]) -> str:
    """User-role message. Keeps the manuscript inline but truncates very long
    artifacts to keep the prompt under ~30k tokens."""
    def _trim(obj: Any, limit: int = 4000) -> Any:
        if obj is None:
            return None
        if isinstance(obj, str):
            return obj[:limit] + (f"\n...[+{len(obj)-limit} bytes truncated]" if len(obj) > limit else "")
        try:
            text = json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            text = str(obj)
        if len(text) > limit:
            text = text[:limit] + f"\n...[+{len(text)-limit} bytes truncated]"
        return text

    chunks = [
        f"# Run under review: `{payload['run']}`",
        f"Target: {payload['target']}",
        f"Run directory: {payload['run_dir']}",
        "",
        "## Manuscript (paper.tex)",
        "```latex",
        _trim(payload['manuscript'], 18000),
        "```",
        "",
        "## refs.bib (first 4 KB)",
        "```bibtex",
        _trim(payload['refs_bib'], 4000),
        "```",
    ]
    for key in ("resolved_target", "data_fetch", "paper_qc", "physics_checks", "published_params", "source_rag",
                "cluster_membership", "hypothesis_plan", "novelty",
                "comparison_table", "qa_gate", "reflexion", "analysis_plan",
                "rag_results", "kg_results", "kg_graph_report", "method_scout",
                "toolbox_gap", "iteration_baseline", "iteration_residuals",
                "iteration_systematics", "model_supervision",
                "sed_decoupled", "light_curve_geom", "eclipse_mcmc", "extinction",
                "figures", "latex_compile"):
        obj = payload["artifacts"].get(key)
        if obj is None:
            chunks.append(f"\n## {key}\n(missing)")
            continue
        chunks.append(f"\n## {key}\n```json\n{_trim(obj, 3500)}\n```")
    chunks.append(f"\n## retrieval.py RERANK_KEYS ({payload.get('retrieval_prompt_path')})")
    chunks.append("```python")
    chunks.append(_trim(payload.get("retrieval_prompt") or "", 6000))
    chunks.append("```")
    if payload.get("gold"):
        chunks.append(f"\n## Gold paper for novelty comparison ({payload['gold_path']})")
        chunks.append("```json")
        chunks.append(_trim(payload["gold"], 6000))
        chunks.append("```")
    chunks.append(
        "\n---\nFollow the system prompt EXACTLY. Return a single JSON document with "
        "schema_version `codex_reviewer_v2` and all five persona sections. No prose "
        "outside the JSON."
    )
    return "\n".join(chunks)


def call_codex(system_prompt: str, user_message: str, model: Optional[str]) -> str:
    """Pipe the user message into the Codex CLI with the system prompt as a flag.
    Falls back to writing both to disk and printing the command if the CLI is
    not on PATH (so the user can run it manually)."""
    import shutil
    cli = shutil.which(CODEX_BIN)
    if not cli:
        raise RuntimeError(
            f"`{CODEX_BIN}` not on PATH. Set $CODEX_BIN or run the printed command manually."
        )
    cmd: List[str] = [cli, "exec", "--system", system_prompt]
    if model:
        cmd += ["--model", model]
    cmd += ["--input", "-"]   # stdin
    proc = subprocess.run(cmd, input=user_message, text=True,
                          capture_output=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"codex exited {proc.returncode}: {proc.stderr[:500]}")
    return proc.stdout.strip()


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        # remove leading ```json (or ```), then trailing ```
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl != -1 else text
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def persist(run_name: str, reviewer_json: Dict[str, Any]) -> Dict[str, str]:
    run_dir = RUNS_DIR / run_name
    reviewer_path = run_dir / "09c_reviewer.json"
    reviewer_path.write_text(json.dumps(reviewer_json, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    # Append to reflexion
    reflexion_path = run_dir / "09b_reflexion.json"
    try:
        reflex = json.loads(reflexion_path.read_text(encoding="utf-8")) if reflexion_path.exists() else {}
    except Exception:
        reflex = {}
    reflex.setdefault("action_items", [])
    for adv in reviewer_json.get("next_actions_for_reflexion", []) or []:
        reflex["action_items"].append({
            "check_id": "codex_reviewer",
            "verdict": reviewer_json.get("referee", {}).get("decision", "minor_revise"),
            "section": adv.get("section"),
            "reason": "codex_reviewer_alignment",
            "advice": adv.get("advice"),
        })
    tb = reviewer_json.get("toolbox_kg_audit") or {}
    tb_items: List[Dict[str, str]] = []
    for issue in (tb.get("node_status_inconsistencies") or [])[:2]:
        tb_items.append({
            "section": "Methods",
            "advice": (
                f"Align manuscript claims with workflow artifact {issue.get('artifact')} "
                f"status={issue.get('status')}; remove or withhold any result whose node did not execute cleanly."
            ),
        })
    for issue in (tb.get("per_source_rag_issues") or [])[:1]:
        tb_items.append({
            "section": "Introduction",
            "advice": f"Repair per-source literature grounding: {issue.get('finding')}.",
        })
    fabricated = [c for c in (tb.get("citation_provenance") or []) if c.get("verdict") == "fabricated"]
    if fabricated:
        tb_items.append({
            "section": "Discussion",
            "advice": (
                "Replace fabricated/unprovenanced citations with bibcodes supplied by published_params, "
                "source_rag, comparison_table, or hypothesis_plan; first bad keys: "
                + ", ".join(str(c.get("key")) for c in fabricated[:3])
            ),
        })
    for adv in tb_items:
        reflex["action_items"].append({
            "check_id": "toolbox_kg_audit",
            "verdict": reviewer_json.get("referee", {}).get("decision", "major_revise"),
            "section": adv.get("section"),
            "reason": "toolbox_kg_audit",
            "advice": adv.get("advice"),
        })
    reflex["codex_reviewer_overall"] = (
        reviewer_json.get("referee", {}).get("score", {}).get("overall")
    )
    reflex["codex_reviewer_decision"] = reviewer_json.get("referee", {}).get("decision")
    if tb:
        reflex["codex_toolbox_coverage_score"] = tb.get("toolbox_coverage_score_0_to_1")
        reflex["codex_kg_alignment_score"] = tb.get("kg_alignment_score_0_to_1")
    reflex["codex_reviewer_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    reflexion_path.write_text(json.dumps(reflex, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"reviewer": str(reviewer_path), "reflexion": str(reflexion_path)}


def summarise(reviewer_json: Dict[str, Any]) -> None:
    ref = reviewer_json.get("referee", {})
    s = ref.get("score", {})
    audit = reviewer_json.get("audit", {})
    tb = reviewer_json.get("toolbox_kg_audit", {}) or {}
    print("\n" + "=" * 70)
    print(f"Codex reviewer summary — decision: {ref.get('decision')}")
    print(f"  rigor={s.get('rigor')}  grounding={s.get('grounding')}  "
          f"clarity={s.get('clarity')}  figures={s.get('figures')}  "
          f"OVERALL={s.get('overall')}")
    print(f"  weakest_section: {ref.get('weakest_section')}")
    print(f"  ungrounded values flagged: {len(audit.get('ungrounded_values') or [])}")
    print(f"  unresolved cites: {len(audit.get('unresolved_cites') or [])}")
    dom = reviewer_json.get("astronomer", {}).get("domain_checks") or []
    failed = [d for d in dom if d.get("verdict") == "fail"]
    print(f"  domain checks failed: {len(failed)} / {len(dom)}")
    if tb:
        print(f"  toolbox coverage: {tb.get('toolbox_coverage_score_0_to_1')}  "
              f"KG alignment: {tb.get('kg_alignment_score_0_to_1')}")
        print(f"  node-status inconsistencies: {len(tb.get('node_status_inconsistencies') or [])}")
        print(f"  KG retrieval issues: {len(tb.get('kg_retrieval_issues') or [])}")
        print(f"  fabricated citations: {sum(1 for c in (tb.get('citation_provenance') or []) if c.get('verdict')=='fabricated')}")
    prof = reviewer_json.get("professor", {})
    print(f"  ApJ acceptance probability: {prof.get('apj_acceptance_probability_pct')}%")
    print("=" * 70)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--model", default=None, help="codex model id; default = whatever CLI is configured with")
    ap.add_argument("--dry-run", action="store_true",
                    help="don't call codex; just print the assembled prompts")
    args = ap.parse_args()

    sys_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    payload = collect_payload(args.run, pick_gold(args.run, None, args.gold))
    user_msg = build_user_message(payload)

    if args.dry_run:
        print("=" * 30, "SYSTEM PROMPT", "=" * 30)
        print(sys_prompt[:1500] + "\n... (truncated)")
        print("=" * 30, "USER MESSAGE", "=" * 30)
        print(user_msg[:2500] + "\n... (truncated)")
        print(f"\nWould call: {CODEX_BIN} exec --system <prompt> --input - (stdin = user msg, "
              f"~{len(user_msg)} chars)")
        return 0

    print(f"[codex-review] calling {CODEX_BIN} (manuscript={len(payload['manuscript'])} chars, "
          f"user message={len(user_msg)} chars)…")
    try:
        raw = call_codex(sys_prompt, user_msg, args.model)
    except Exception as exc:
        print(f"codex call failed: {exc}", file=sys.stderr)
        return 2
    try:
        obj = json.loads(_strip_json_fence(raw))
    except Exception as exc:
        # Save the raw text for debugging and exit with 3.
        debug = RUNS_DIR / args.run / "09c_reviewer_raw.txt"
        debug.write_text(raw, encoding="utf-8")
        print(f"codex returned non-JSON; raw saved to {debug}: {exc}", file=sys.stderr)
        return 3
    if obj.get("schema_version") not in ("codex_reviewer_v1", "codex_reviewer_v2"):
        print(f"WARN: schema_version mismatch ({obj.get('schema_version')!r})", file=sys.stderr)
    if run_toolbox_kg_audit is not None and not obj.get("toolbox_kg_audit"):
        obj["toolbox_kg_audit"] = run_toolbox_kg_audit(
            RUNS_DIR / args.run,
            manuscript=payload.get("manuscript"),
        )
        if obj.get("schema_version") == "codex_reviewer_v1":
            obj["schema_version"] = "codex_reviewer_v2"
    paths = persist(args.run, obj)
    print(f"[codex-review] wrote: {paths['reviewer']}")
    print(f"[codex-review] appended action items into: {paths['reflexion']}")
    summarise(obj)
    return 0


if __name__ == "__main__":
    sys.exit(main())
