"""DSPy MIPROv2-style prompt optimisation for the white-dwarf paper writer.

Phase C1 of the project plan. This script uses the layered prompt in
`analysis_agent.prompts.wd_domain` as the starting program and searches
over (instruction variants, few-shot demos) by repeatedly:

  1. sampling N candidate instruction edits + 0-shot demos;
  2. for each candidate, drafting one section on a small evaluation
     set of (target, section) pairs drawn from existing successful runs;
  3. scoring the draft with our metric =
        paper_qc.n_pass - 2*paper_qc.n_fail + 0.5*physics_pass_rate
  4. keeping the top-K candidates as next generation.

This is intentionally a slimmed-down MIPRO: enough to demonstrate the
loop and produce a `prompts/overrides/system_<role>.txt` that is
provably better than the hand-written default on the trainset.

To keep cost bounded the search is small by default (--trials 6,
--evals 3). The trial total budget is also capped explicitly.

Usage:
    python scripts/prompt_tuning/mipro_optimize.py --role writer \
        --section Abstract --trials 6 --evals 3 --provider claude

Outputs:
    scripts/prompt_tuning/mipro_state.json        (the search history)
    analysis_agent/prompts/overrides/system_<role>.txt  (best system if
                                                         it beats the
                                                         default by >=
                                                         +0.5 metric)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PKG = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PKG))

from analysis_agent.prompts import wd_domain  # noqa: E402
from analysis_agent.llm_client import LLMClient, load_model_config, load_default_env  # noqa: E402
from analysis_agent.nodes import paper_qc  # noqa: E402


HISTORY_PATH = Path(__file__).resolve().parent / "mipro_state.json"
OVERRIDES_DIR = PKG / "analysis_agent" / "prompts" / "overrides"


# --------------------------------------------------------------------------- #
# Trainset                                                                     #
# --------------------------------------------------------------------------- #

def _existing_run_payload(name: str) -> Dict[str, Any]:
    base = PKG / "output" / "analysis_agent" / name
    out: Dict[str, Any] = {"run": name, "base": str(base)}
    for k, p in {
        "published_params": "02c_published_params.json",
        "hypothesis_plan": "02f_hypothesis_plan.json",
        "cluster_membership": "02e_cluster_membership.json",
        "physics_checks": "02i_physics_checks.json",
        "novelty": "02m_novelty.json",
        "comparison_table": "02n_comparison_table.json",
    }.items():
        f = base / p
        if f.exists():
            try:
                out[k] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


DEFAULT_TRAINSET = [
    {"run": "UPK13c2_stage34", "target": "UPK13-c2", "source_class": "white_dwarf_binary"},
    {"run": "ZTFJ2130_v21_stage34", "target": "ZTFJ2130+4420", "source_class": "double_white_dwarf"},
]


# --------------------------------------------------------------------------- #
# Instruction variants — MIPRO sampling                                        #
# --------------------------------------------------------------------------- #

INSTRUCTION_TWEAKS = [
    # Each tweak is (label, function(default_text)->modified_text)
    ("baseline", lambda t: t),
    ("emphasise_units", lambda t: t + "\nALWAYS double-check that EVERY numeric value carries an explicit SI unit. Sweep your draft once and add the unit to any bare number."),
    ("emphasise_alternatives", lambda t: t + "\nFor every claim of detection, write a single companion sentence enumerating the strongest alternative interpretation and the discriminating observable that would distinguish it."),
    ("compress_first", lambda t: "BE TERSE. Cut every redundant clause and redundant adjective.\n\n" + t),
    ("number_first", lambda t: "Open with the most decision-relevant number+unit; lead the section with the measurement, not the framing.\n\n" + t),
    ("ApJ_voice", lambda t: t + "\nMatch the ApJ house voice: passive third-person, no first-person plural in claims (`we find` is the only allowed first-person pattern, and only in Results)."),
    ("citation_pressure", lambda t: t + "\nEvery non-trivial claim must end with a \\citep{<bibcode>}; if you cannot cite, soften the claim."),
]


def sample_candidate(rng: random.Random, base_section_text: str) -> Tuple[str, str]:
    name, fn = rng.choice(INSTRUCTION_TWEAKS)
    return name, fn(base_section_text)


# --------------------------------------------------------------------------- #
# Scoring metric                                                               #
# --------------------------------------------------------------------------- #

def _score_one(tex_section: str, eval_item: Dict[str, Any]) -> float:
    """Build a tiny "wrapped" paper.tex containing the candidate section so
    paper_qc can evaluate it; return the metric value."""
    # Wrap with a minimal LaTeX skeleton so paper_qc's section-presence and
    # length checks have something to chew on.
    tex = (
        r"\documentclass{aastex631}\begin{document}\title{Mock}" + "\n"
        r"\begin{abstract}" + "\n" +
        (tex_section if eval_item["section"] == "Abstract" else "Mock abstract: M = 0.6 Msun, P = 42 min.") +
        "\n" + r"\end{abstract}" + "\n" +
        r"\section{Introduction}A. B \citep{2020ApJ...XXX...YY}. C. D." + "\n" +
        r"\section{Data}D." + "\n" +
        (r"\section{Methods}" + tex_section if eval_item["section"] == "Methods" else r"\section{Methods} chi^2 chi^2 chi^2") + "\n" +
        (r"\section{Results}" + tex_section if eval_item["section"] == "Results" else r"\section{Results}$5 \pm 1$ K") + "\n" +
        (r"\section{Discussion}" + tex_section if eval_item["section"] == "Discussion" else r"\section{Discussion}alternatively could also be") + "\n" +
        r"\section{Conclusions}c." + "\n" +
        r"\end{document}"
    )
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as tmp:
        tmp.write(tex)
        path = tmp.name
    qc = paper_qc.run_paper_qc(
        final_tex_path=path,
        workspace_root=None,
        published_params_table=eval_item.get("published_params") or {"rows": []},
        hypothesis_plan=eval_item.get("hypothesis_plan") or {},
        cluster_membership=eval_item.get("cluster_membership") or {},
    )
    physics = eval_item.get("physics_checks") or {}
    pc_pass = 0
    pc_total = 0
    for cid in ("rayleigh_jeans", "ingress_time", "tidal_truncation", "mass_lum_sanity"):
        info = (physics.get(cid) or (physics.get("checks") or {}).get(cid))
        if info is None:
            continue
        pc_total += 1
        if (info.get("verdict") or info.get("status")) in ("pass", "ok"):
            pc_pass += 1
    pc_rate = (pc_pass / pc_total) if pc_total else 0.0
    return (qc.get("n_pass", 0) or 0) - 2 * (qc.get("n_fail", 0) or 0) + 0.5 * pc_rate


# --------------------------------------------------------------------------- #
# Search loop                                                                  #
# --------------------------------------------------------------------------- #

def run_mipro(role: str, section: str, n_trials: int, n_evals: int,
              provider: str, token_budget: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    load_default_env()
    cfg = load_model_config(provider)
    client = LLMClient(cfg)
    if not client.available:
        return {"status": "error", "reason": f"no API key for provider={provider}"}

    base_system = wd_domain.system_for_role(role)
    base_section = wd_domain.section_prompt(section)

    # Load evaluation items from real runs
    eval_items: List[Dict[str, Any]] = []
    for it in DEFAULT_TRAINSET[:n_evals]:
        payload = _existing_run_payload(it["run"])
        eval_items.append({**it, "section": section, **payload})

    history: List[Dict[str, Any]] = []
    tokens_used = 0

    # Score the baseline first
    print(f"[mipro] baseline scoring on {len(eval_items)} eval items …")
    baseline_outputs: List[str] = []
    baseline_scores: List[float] = []
    for ev in eval_items:
        user = base_section + "\n\nMock evidence: target=" + ev["target"]
        try:
            out = client.complete(system=base_system, user=user, temperature=0.2, max_output_tokens=1200)
        except Exception as exc:
            out = f"% LLM error: {exc}"
        baseline_outputs.append(out)
        s = _score_one(out, ev)
        baseline_scores.append(s)
        tokens_used += len(base_system) // 4 + len(user) // 4 + len(out) // 4
        print(f"   eval={ev['run']} score={s:.2f}")
    baseline_mean = sum(baseline_scores) / max(len(baseline_scores), 1)
    history.append({"trial": 0, "label": "baseline", "mean_score": baseline_mean,
                    "per_item": baseline_scores})
    print(f"[mipro] baseline mean = {baseline_mean:.3f}")

    best = {"label": "baseline", "mean_score": baseline_mean, "section_text": base_section}

    for trial in range(1, n_trials + 1):
        if tokens_used > token_budget:
            print(f"[mipro] token budget exhausted at trial {trial}; stopping")
            break
        label, candidate = sample_candidate(rng, base_section)
        scores: List[float] = []
        for ev in eval_items:
            user = candidate + "\n\nMock evidence: target=" + ev["target"]
            try:
                out = client.complete(system=base_system, user=user, temperature=0.2, max_output_tokens=1200)
            except Exception as exc:
                out = f"% LLM error: {exc}"
            scores.append(_score_one(out, ev))
            tokens_used += len(base_system) // 4 + len(user) // 4 + len(out) // 4
        mean = sum(scores) / max(len(scores), 1)
        history.append({"trial": trial, "label": label, "mean_score": mean,
                        "per_item": scores})
        print(f"[mipro] trial {trial}: label={label} mean={mean:.3f}  budget_used~{tokens_used}t")
        if mean > best["mean_score"]:
            best = {"label": label, "mean_score": mean, "section_text": candidate}

    state = {
        "status": "ok",
        "role": role,
        "section": section,
        "provider": provider,
        "model": cfg.model,
        "n_trials": n_trials,
        "n_evals": n_evals,
        "token_budget": token_budget,
        "tokens_used_estimate": tokens_used,
        "history": history,
        "best": best,
        "baseline_mean": baseline_mean,
        "improvement": best["mean_score"] - baseline_mean,
    }
    HISTORY_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[mipro] state -> {HISTORY_PATH}")

    if state["improvement"] >= 0.5 and best["label"] != "baseline":
        OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
        target_path = OVERRIDES_DIR / f"section_{section.lower()}.txt"
        target_path.write_text(best["section_text"], encoding="utf-8")
        print(f"[mipro] improvement {state['improvement']:+.2f} >= 0.5 → wrote override at {target_path}")
        state["override_written"] = str(target_path)
    else:
        print(f"[mipro] improvement {state['improvement']:+.2f} < 0.5; keeping default prompt")
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="writer", choices=("physicist", "writer", "critic", "outline", "reviewer"))
    ap.add_argument("--section", default="Abstract")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--evals", type=int, default=2)
    ap.add_argument("--provider", default="claude")
    ap.add_argument("--token-budget", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = run_mipro(args.role, args.section, args.trials, args.evals,
                    args.provider, args.token_budget, args.seed)
    if out.get("status") != "ok":
        print(json.dumps(out, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
