"""Command line entry point for the Chief Investigator analysis agent."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .workflow import run_workflow


def default_output_root(target: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in target).strip("_")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return str(Path("Astro_Agent") / "output" / "analysis_agent" / f"{safe or 'target'}_{stamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the astronomy Chief Investigator multi-agent workflow."
    )
    parser.add_argument("target", help="Target name or label")
    parser.add_argument("--ra", type=float, help="ICRS right ascension in decimal degrees")
    parser.add_argument("--dec", type=float, help="ICRS declination in decimal degrees")
    parser.add_argument("--output-root", help="Run output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Plan the data-fetch step without downloading survey data (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run astro_toolbox data fetching and modeling modules",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute even if dry-run is set; intended for scripted reruns",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use configured fox/deepseek API for PaperOrchestra writing calls",
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="Provider for writing calls, e.g. deepseek, fox, gemini. Defaults to environment config.",
    )
    parser.add_argument(
        "--astrotool-run",
        help="Use an existing astro_toolbox output directory instead of running/downloading it again.",
    )
    parser.add_argument(
        "--skip-simbad",
        action="store_true",
        help="Skip online SIMBAD cross-match. Useful for local workflow tests when network is slow.",
    )
    parser.add_argument(
        "--kg-report",
        action="store_true",
        help="Generate a large KG graph report inside this run directory.",
    )
    parser.add_argument(
        "--kg-report-llm",
        action="store_true",
        help="Use an LLM to interpret the KG report. Use with --kg-report.",
    )
    parser.add_argument(
        "--kg-report-provider",
        default="deepseek",
        help="Provider for KG report interpretation; default: deepseek.",
    )
    parser.add_argument(
        "--draft-on-hold",
        action="store_true",
        help="Still run PaperOrchestra when QA is on hold. The draft keeps QA warnings and is not final science.",
    )
    parser.add_argument(
        "--method-scout-llm",
        action="store_true",
        help="Use an LLM to investigate new/reusable analysis methods before fitting.",
    )
    parser.add_argument(
        "--method-scout-provider",
        default=None,
        help="Provider for method scouting, e.g. kimi or deepseek. Defaults to --llm-provider.",
    )
    parser.add_argument(
        "--source-research-package",
        action="store_true",
        help="Build the strict per-source evidence package: SIMBAD refs/PDFs, exact RAG hits, KG relations, HST/SED/line QA.",
    )
    parser.add_argument(
        "--download-simbad-pdfs",
        action="store_true",
        help="When --source-research-package is enabled, download all resolvable SIMBAD-linked PDFs.",
    )
    parser.add_argument(
        "--enable-claude-code",
        action="store_true",
        help="Delegate Supervisor repair tasks to Claude Code. Requires ASTRO_AGENT_CLAUDE_BIN or claude on PATH.",
    )
    parser.add_argument(
        "--claude-timeout",
        type=int,
        default=300,
        help="Seconds to wait for a Claude Code delegation before recording failure and continuing; default: 300.",
    )
    parser.add_argument(
        "--claude-permission-mode",
        default="plan",
        help="Permission mode passed to Claude Code; default: plan.",
    )
    parser.add_argument(
        "--max-supervision-rounds",
        type=int,
        default=2,
        help="Number of model-supervision rounds before QA; default: 2.",
    )
    parser.add_argument("--json", action="store_true", help="Print final state as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root or default_output_root(args.target)
    dry_run = not bool(args.execute)
    state = run_workflow(
        target=args.target,
        ra_deg=args.ra,
        dec_deg=args.dec,
        output_root=output_root,
        dry_run=dry_run,
        force=args.force,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        astrotool_run=args.astrotool_run,
        kg_report=args.kg_report,
        kg_report_llm=args.kg_report_llm,
        kg_report_provider=args.kg_report_provider,
        skip_simbad=args.skip_simbad,
        draft_on_hold=args.draft_on_hold,
        method_scout_llm=args.method_scout_llm,
        method_scout_provider=args.method_scout_provider,
        source_research_package=args.source_research_package,
        download_simbad_pdfs=args.download_simbad_pdfs,
        enable_claude_code=args.enable_claude_code,
        claude_timeout=args.claude_timeout,
        claude_permission_mode=args.claude_permission_mode,
        max_supervision_rounds=args.max_supervision_rounds,
    )
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    qa = state.get("qa", {})
    print(f"Output root: {Path(output_root).resolve()}")
    print(f"QA gate: {qa.get('apj_gate', 'unknown')}")
    if state.get("paper_orchestra"):
        print(f"PaperOrchestra workspace: {state['paper_orchestra'].get('workspace')}")
    if state.get("kg_graph_report", {}).get("status") == "written":
        print(f"KG graph report: {state['kg_graph_report'].get('output_root')}")
    if state.get("analysis_plan"):
        print(f"Analysis route: {state['analysis_plan'].get('route')}")
    if state.get("source_research", {}).get("status") == "written":
        print(f"Source research package: {state['source_research'].get('output_root')}")
        print(f"SIMBAD refs/PDFs: {state['source_research'].get('simbad_n_refs')} / {state['source_research'].get('simbad_pdf_count')}")
    if state.get("model_supervision"):
        print(f"Model supervision: {state['model_supervision'].get('status')}")
    if qa.get("reasons"):
        print("Human-review reasons:")
        for reason in qa["reasons"]:
            print(f"- {reason}")
    print("Artifacts:")
    for artifact in state.get("artifacts", []):
        print(f"- {artifact}")


if __name__ == "__main__":
    main()
