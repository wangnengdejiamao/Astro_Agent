"""LangGraph orchestration for the astronomy Chief Investigator agent."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from .state import AnalysisState
from . import codex_style, codex_tool, paper_agents, paper_orchestra, tools
from .graph_visualization_agent import run_graph_agent
from .llm_client import LLMClient, load_model_config


def _append_artifact(state: AnalysisState, path: str) -> None:
    state.setdefault("artifacts", [])
    if path not in state["artifacts"]:
        state["artifacts"].append(path)


def _run_dir(state: AnalysisState) -> Path:
    root = Path(state["output_root"]).resolve()
    return tools.ensure_dir(root)


def resolve_node(state: AnalysisState) -> AnalysisState:
    resolved = tools.resolve_target(
        state["target"],
        state.get("ra_deg"),
        state.get("dec_deg"),
    )
    state["resolved"] = resolved
    if resolved.get("status") == "ok":
        state["ra_deg"] = resolved["ra_deg"]
        state["dec_deg"] = resolved["dec_deg"]
    else:
        state.setdefault("errors", []).extend(resolved.get("errors", []))
    _append_artifact(state, tools.json_dump(_run_dir(state) / "01_resolved_target.json", resolved))
    return state


def data_fetcher_node(state: AnalysisState) -> AnalysisState:
    resolved = state.get("resolved", {})
    if resolved.get("status") != "ok":
        state["data_fetch"] = {"status": "skipped", "reason": "target not resolved"}
        return state

    ra = float(state["ra_deg"])
    dec = float(state["dec_deg"])
    if state.get("skip_simbad", False):
        simbad = {"status": "skipped", "reason": "skip_simbad enabled"}
    else:
        simbad = tools.query_simbad_crossmatch(ra, dec)
    existing_run = state.get("astrotool_run")
    if existing_run:
        output_root = Path(existing_run).resolve()
        astrotool = {
            "status": "existing",
            "output_root": str(output_root),
            "note": "using caller-supplied astro_toolbox output directory",
            "existing_outputs": tools.summarize_output_root(output_root),
        }
    else:
        output_root = _run_dir(state) / "astrotool_run"
        astrotool = tools.run_astrotool(
            target=state["target"],
            ra_deg=ra,
            dec_deg=dec,
            output_root=output_root,
            dry_run=bool(state.get("dry_run", True)),
            force=bool(state.get("force", False)),
        )
    pack = {
        "status": astrotool.get("status"),
        "simbad_crossmatch": simbad,
        "astrotool": astrotool,
        "existing_outputs": astrotool.get("existing_outputs", {}),
    }
    state["data_fetch"] = pack
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02_data_fetch.json", pack))
    return state


def structure_planner_node(state: AnalysisState) -> AnalysisState:
    """Choose the science route from available data before model fitting."""
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    run_summary = output.get("run_summary", {}) or {}
    module_names = {str(row.get("module", "")).lower() for row in output.get("module_rows", [])}
    files = {str(name).lower() for name in output.get("sample_files", [])}

    has_spectrum = bool(run_summary.get("spectra_available")) or any("spectrum" in name for name in module_names)
    has_hst = bool(run_summary.get("hst_spectrum_available")) or any("hst" in name for name in module_names | files)
    has_sed = bool(run_summary.get("sed_available")) or any(name == "sed" for name in module_names) or "sed.png" in files
    has_hrd = any("hr" in name for name in module_names) or "hr_diagram.png" in files
    has_rv = bool(run_summary.get("rv_report_available")) or any("rv" in name for name in module_names)
    has_period = bool(run_summary.get("period_products_available")) or any("period" in name or "fold" in name for name in files)

    if has_spectrum:
        route = "spectroscopy_plus_sed"
        parameter_strategy = [
            "Use optical/UV spectra for classification, line masks, Teff/logg/composition constraints, and emission-line rejection.",
            "Use SED and HR diagram as consistency checks against spectroscopic parameters.",
        ]
    elif has_sed and has_hrd:
        route = "photometric_hrd_sed_fallback"
        parameter_strategy = [
            "No spectrum is available; infer provisional Teff/radius/luminosity from SED plus Gaia/HR-diagram position.",
            "Do not claim spectral type, line detections, radial velocities, atmosphere composition, or logg precision from photometry alone.",
            "Use wider priors and require human review before final WD mass/cooling-age claims.",
        ]
    elif has_sed:
        route = "sed_only_fallback"
        parameter_strategy = [
            "Use SED-only fitting for rough temperature/radius constraints if distance/extinction priors exist.",
            "Mark logg/mass/cooling age as weakly constrained and block final ApJ parameter tables.",
        ]
    else:
        route = "insufficient_data"
        parameter_strategy = [
            "Fetch or repair photometry/spectroscopy before physical-model fitting.",
        ]

    plan = {
        "status": "written",
        "route": route,
        "available_evidence": {
            "spectrum": has_spectrum,
            "hst_spectrum": has_hst,
            "sed": has_sed,
            "hr_diagram": has_hrd,
            "rv": has_rv,
            "period_products": has_period,
        },
        "parameter_strategy": parameter_strategy,
        "fitting_priority": [
            "1. identity/crossmatch and unit verification",
            "2. spectral classification and emission-line checks when spectra exist",
            "3. SED fit with residual diagnostics",
            "4. HRD/Gaia consistency and distance/extinction systematics",
            "5. RV/period/binary checks when time-domain or spectra exist",
            "6. final parameters only after Supervisor and QA clearance",
        ],
    }
    state["analysis_plan"] = plan
    _append_artifact(state, tools.json_dump(_run_dir(state) / "02b_analysis_plan.json", plan))
    return state


def rag_navigator_node(state: AnalysisState) -> AnalysisState:
    queries = [
        "white dwarf SED fitting Gaia parallax effective temperature",
        "white dwarf radial velocity orbit traceback six dimensional phase space",
        "white dwarf Bayesian inference systematics uncertainty error propagation",
        "emission line white dwarf binary accretion spectrum residual fitting",
    ]
    results: List[Dict[str, Any]] = []
    for query in queries:
        rows = tools.search_rag(query, method_only=True, limit=5)
        results.append({"query": query, "rows": rows})
    state["rag_results"] = results
    _append_artifact(state, tools.json_dump(_run_dir(state) / "03_rag_results.json", results))
    return state


def kg_navigator_node(state: AnalysisState) -> AnalysisState:
    queries = [
        "SED fitting Bayesian inference parallax cooling age",
        "6D phase space orbit traceback radial velocity binary white dwarf",
        "residual systematics uncertainty model prior",
    ]
    rows = tools.search_kg(queries, limit=20)
    state["kg_results"] = rows
    _append_artifact(state, tools.json_dump(_run_dir(state) / "04_kg_results.json", rows))
    return state


def _strip_json_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def method_scout_node(state: AnalysisState) -> AnalysisState:
    """Investigate reusable/newer methods before deciding what Coder/Claude should change."""
    plan = state.get("analysis_plan", {})
    route = plan.get("route", "unknown")
    base_queries = [
        f"{route} white dwarf parameter inference method",
        "Gaia HR diagram SED fitting white dwarf no spectroscopy",
        "white dwarf SED residual systematic error Bayesian photometric fit",
        "spectral line fitting emission line white dwarf binary classification",
    ]
    if route in {"photometric_hrd_sed_fallback", "sed_only_fallback"}:
        base_queries.extend(
            [
                "photometric white dwarf effective temperature radius Gaia parallax extinction prior",
                "white dwarf mass cooling age constraints without spectroscopy limitations",
            ]
        )
    rag_hits = []
    for query in base_queries[:6]:
        rag_hits.append({"query": query, "rows": tools.search_rag(query, method_only=False, limit=4)})
    kg_hits = tools.search_kg(base_queries, limit=20)

    recommendations = [
        "Before trusting a model result, compare it against at least one independent observable: spectra if available, otherwise HR-diagram position and SED residuals.",
        "If spectra are absent, only publish photometric parameters as provisional and keep atmosphere composition/logg/mass/cooling age under stronger caveats.",
        "Any fit pinned to a model-grid boundary must be treated as nonconverged and handed back to Coder/Claude for prior/model repair.",
    ]
    scout: Dict[str, Any] = {
        "status": "written",
        "route": route,
        "queries": base_queries,
        "rag_hits": rag_hits,
        "kg_hits": kg_hits,
        "recommendations": recommendations,
        "llm_used": False,
    }

    if state.get("method_scout_llm", False):
        provider = state.get("method_scout_provider") or state.get("llm_provider") or "kimi"
        try:
            cfg = load_model_config(provider)
            client = LLMClient(cfg)
            compact = {
                "analysis_plan": plan,
                "rag_hits": [
                    {"query": item["query"], "rows": item["rows"][:2]}
                    for item in rag_hits
                ],
                "kg_hits": kg_hits[:8],
            }
            system = (
                "You are an astrophysics method-scout agent. Return strict JSON. "
                "You may propose new analysis methods, but every proposal must say whether it is supported by supplied RAG/KG evidence or is a hypothesis requiring human approval."
            )
            user = (
                "为这个未知源分析流程调查可迁移/更新的方法，特别关注无光谱时如何用 HR 图位置和 SED 拟合参数。"
                "输出 JSON keys: recommended_methods[], coder_tasks[], qa_risks[], human_review_triggers[].\n\n"
                + json.dumps(compact, ensure_ascii=False, indent=2)[:18000]
            )
            text = client.complete(system=system, user=user, temperature=0.1, max_output_tokens=5000)
            scout["llm_provider"] = cfg.provider
            scout["llm_model"] = cfg.model
            scout["llm_used"] = True
            scout["llm_recommendations"] = json.loads(_strip_json_fence(text))
        except Exception as exc:
            scout["llm_error"] = f"{type(exc).__name__}: {exc}"

    state["method_scout"] = scout
    _append_artifact(state, tools.json_dump(_run_dir(state) / "04c_method_scout.json", scout))
    return state


def kg_graph_report_node(state: AnalysisState) -> AnalysisState:
    if not state.get("kg_report", False):
        state["kg_graph_report"] = {"status": "skipped"}
        return state

    out_dir = _run_dir(state) / "kg_graph_report"
    try:
        result = run_graph_agent(
            output_root=out_dir,
            use_llm=bool(state.get("kg_report_llm", False)),
            provider=state.get("kg_report_provider") or state.get("llm_provider") or "deepseek",
        )
        result["status"] = "written"
        state["kg_graph_report"] = result
        for path in result.get("artifacts", {}).values():
            if isinstance(path, str) and path:
                _append_artifact(state, path)
    except Exception as exc:
        state["kg_graph_report"] = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "output_root": str(out_dir),
        }
        state.setdefault("warnings", []).append(f"KG graph report failed: {type(exc).__name__}: {exc}")

    _append_artifact(state, tools.json_dump(_run_dir(state) / "04b_kg_graph_report.json", state["kg_graph_report"]))
    return state


def source_research_package_node(state: AnalysisState) -> AnalysisState:
    """Build the strict per-source SIMBAD/RAG/KG/data evidence package."""
    if not state.get("source_research_package", False):
        state["source_research"] = {"status": "skipped", "reason": "source_research_package is false"}
        return state
    if state.get("ra_deg") is None or state.get("dec_deg") is None:
        result = {"status": "skipped", "reason": "coordinates unavailable"}
        state["source_research"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "04d_source_research_package.json", result))
        return state

    astrotool_root = state.get("data_fetch", {}).get("astrotool", {}).get("output_root") or state.get("astrotool_run")
    if not astrotool_root:
        result = {"status": "skipped", "reason": "astrotool output root unavailable"}
        state["source_research"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "04d_source_research_package.json", result))
        return state

    out_root = _run_dir(state) / "source_research_package"
    cmd = [
        sys.executable,
        "-m",
        "Astro_Agent.analysis_agent.source_research_pipeline",
        "--target",
        state["target"],
        "--ra",
        str(state["ra_deg"]),
        "--dec",
        str(state["dec_deg"]),
        "--output-root",
        str(out_root),
        "--astrotool-run",
        str(astrotool_root),
    ]
    if state.get("download_simbad_pdfs", False):
        cmd.append("--download-simbad-pdfs")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            text=True,
            capture_output=True,
            timeout=7200,
        )
        result: Dict[str, Any] = {
            "status": "written" if proc.returncode == 0 else "error",
            "output_root": str(out_root),
            "command": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        for name in (
            "source_research_package.json",
            "source_analysis_products.json",
            "source_research_report.md",
            "simbad_all_references.json",
            "simbad_pdf_downloads.json",
            "rag_exact_simbad_papers.json",
            "kg_source_relations.json",
        ):
            path = out_root / name
            if path.exists():
                result[name.replace(".", "_")] = str(path)
                _append_artifact(state, str(path))
        simbad_path = out_root / "simbad_all_references.json"
        if simbad_path.exists():
            simbad = json.loads(simbad_path.read_text(encoding="utf-8"))
            result["simbad_n_refs"] = simbad.get("n_refs")
        downloads_path = out_root / "simbad_pdf_downloads.json"
        if downloads_path.exists():
            downloads = json.loads(downloads_path.read_text(encoding="utf-8"))
            result["simbad_pdf_count"] = downloads.get("n_available_pdf")
        analysis_path = out_root / "source_analysis_products.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            result["sed_status"] = analysis.get("sed", {}).get("status")
            result["robust_emission_counts"] = {
                survey: payload.get("robust_emission_count")
                for survey, payload in analysis.get("line_fits", {}).items()
            }
        if proc.returncode != 0:
            state.setdefault("warnings", []).append("Source research package failed; see 04d_source_research_package.json")
    except Exception as exc:
        result = {
            "status": "error",
            "output_root": str(out_root),
            "error": f"{type(exc).__name__}: {exc}",
        }
        state.setdefault("warnings", []).append(f"Source research package failed: {type(exc).__name__}: {exc}")

    state["source_research"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "04d_source_research_package.json", result))
    return state


def baseline_iteration_node(state: AnalysisState) -> AnalysisState:
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    checks = tools.extract_physics_checks(output)
    using_existing = state.get("data_fetch", {}).get("status") == "existing"
    status = "completed" if using_existing else ("planned" if state.get("dry_run", True) else "completed")
    if not output.get("run_summary") and state.get("dry_run", True) and not using_existing:
        warnings = ["Baseline fit not executed in dry-run mode"]
    else:
        warnings = checks["warnings"]
    iteration = {
        "iteration": 1,
        "name": "Baseline",
        "status": status,
        "objective": "Run standard astro_toolbox spectra, photometry, SED, HRD, WD fitting, period and RV modules.",
        "inputs": {
            "astrotool_output_root": state.get("data_fetch", {}).get("astrotool", {}).get("output_root"),
            "rag_queries": [item["query"] for item in state.get("rag_results", [])],
        },
        "warnings": warnings,
        "checks": checks,
    }
    state.setdefault("iterations", []).append(iteration)
    _append_artifact(state, tools.json_dump(_run_dir(state) / "05_iteration_1_baseline.json", iteration))
    return state


def residual_physics_iteration_node(state: AnalysisState) -> AnalysisState:
    warnings: List[str] = []
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    run_summary = output.get("run_summary", {})
    if not run_summary.get("wd_fitting_available"):
        warnings.append("Cannot inspect WD residuals because WD fitting output is unavailable")
    if not output.get("status_file"):
        warnings.append("No module_status.csv was found for residual inspection")

    iteration = {
        "iteration": 2,
        "name": "Residuals & Physics",
        "status": "nonconverged" if warnings else "completed",
        "objective": "Inspect residuals and reject physically impossible parameter regions.",
        "physics_guards": [
            "cooling_age_gyr must not exceed 13.8 without explicit explanation",
            "logg should remain inside the atmosphere grid and a plausible WD range",
            "dust/IR excess models require temperatures and luminosities consistent with the SED",
            "period-folded light curves must be checked for aliases and survey cadence artifacts",
            "emission lines trigger binary/accretion alternatives before single-DA interpretation",
        ],
        "method_transfer_from_kg": state.get("kg_results", [])[:8],
        "warnings": warnings,
    }
    state.setdefault("iterations", []).append(iteration)
    _append_artifact(state, tools.json_dump(_run_dir(state) / "06_iteration_2_residuals.json", iteration))
    return state


def errors_systematics_iteration_node(state: AnalysisState) -> AnalysisState:
    warnings: List[str] = []
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    counts = output.get("module_status_counts", {})
    if counts.get("error", 0):
        warnings.append(f"{counts['error']} modules errored; confidence intervals must include data-missing systematics")
    if state.get("dry_run", True) and state.get("data_fetch", {}).get("status") != "existing":
        warnings.append("No numerical confidence interval can be certified in dry-run mode")

    iteration = {
        "iteration": 3,
        "name": "Errors & Systematics",
        "status": "nonconverged" if warnings else "completed",
        "objective": "Quantify statistical and systematic uncertainties before final parameters are accepted.",
        "required_error_budget": [
            "photometric zero points and extinction correction",
            "Gaia parallax uncertainty and possible zero-point offset",
            "spectral flux calibration and line-mask choices",
            "model-grid spacing and atmosphere-composition prior",
            "period aliasing and RV template mismatch where applicable",
        ],
        "warnings": warnings,
    }
    state.setdefault("iterations", []).append(iteration)
    _append_artifact(state, tools.json_dump(_run_dir(state) / "07_iteration_3_systematics.json", iteration))
    return state


def model_supervisor_node(state: AnalysisState) -> AnalysisState:
    """Continuously supervise model results and produce concrete repair tasks."""
    output = state.get("data_fetch", {}).get("existing_outputs", {})
    run_summary = output.get("run_summary", {}) or {}
    plan = state.get("analysis_plan", {})
    source_research = state.get("source_research", {})
    route = plan.get("route", "unknown")
    method_scout = state.get("method_scout", {})
    max_rounds = max(1, int(state.get("max_supervision_rounds", 2) or 2))

    rounds: List[Dict[str, Any]] = []
    open_actions: List[Dict[str, Any]] = []
    human_triggers: List[str] = []
    for round_index in range(1, max_rounds + 1):
        checks: List[str] = []
        actions: List[Dict[str, Any]] = []

        if route in {"photometric_hrd_sed_fallback", "sed_only_fallback"}:
            checks.append("spectra_absent_or_not_required_for_route")
            actions.append(
                {
                    "owner": "Coder/Claude",
                    "priority": "high",
                    "task": "Implement or run photometric fallback fitting: Gaia/HRD + SED + parallax/extinction priors for provisional Teff/radius/luminosity.",
                    "acceptance": "Output JSON with units, chi2/residuals, posterior intervals, and a caveat that logg/mass/cooling age are weak without spectra.",
                }
            )
        elif run_summary.get("spectra_available"):
            checks.append("spectra_available_for_classification_and_line_fitting")
            actions.append(
                {
                    "owner": "Coder/Claude",
                    "priority": "medium",
                    "task": "Verify spectral-line fitting products and reject boundary/false emission-line detections.",
                    "acceptance": "Line table includes wavelength, flux/EW, uncertainty, S/N, continuum window, and rejection reason for non-robust candidates.",
                }
            )

        if run_summary.get("sed_available") and not run_summary.get("wd_fitting_available"):
            actions.append(
                {
                    "owner": "Coder/Claude",
                    "priority": "high",
                    "task": "Repair or add WD/SED fitting result export for the existing SED products.",
                    "acceptance": "Create wd_fitting.json or wd_fit_results.csv with astropy-compatible units, priors, goodness-of-fit, residual summary, and model-grid boundary flags.",
                }
            )
            human_triggers.append("SED exists but WD fitting output is unavailable; final parameters remain blocked.")
        if source_research.get("sed_status") in {"poor_fit", "no_fit"}:
            actions.append(
                {
                    "owner": "Coder/Claude",
                    "priority": "high",
                    "task": "Improve the SED model using the source research package: keep SDSS/Gaia optical bands as the primary DA fit and treat WISE/SPHEREx as context unless provenance and model coverage are validated.",
                    "acceptance": "SED JSON reports primary_fit_bands, context_only_bands, residuals, chi2, and explicitly blocks final Teff/logg/mass/cooling-age claims when chi2 remains high.",
                }
            )
            human_triggers.append("Primary SED fit is poor or absent in the source research package.")
        robust_counts = source_research.get("robust_emission_counts") or {}
        if robust_counts:
            checks.append(f"source_research_robust_emission_counts={robust_counts}")

        if run_summary.get("hst_spectrum_available"):
            checks.append("hst_uv_context_available")
            actions.append(
                {
                    "owner": "Reviewer",
                    "priority": "medium",
                    "task": "Audit whether HST UV data are used as UV context or direct optical-classification evidence.",
                    "acceptance": "Paper text separates UV coverage/SNR limitations from optical spectral classification.",
                }
            )

        for recommendation in method_scout.get("recommendations", [])[:3]:
            checks.append(f"method_scout: {recommendation}")

        round_status = "needs_repair" if actions else "clear"
        rounds.append(
            {
                "round": round_index,
                "status": round_status,
                "checks": checks,
                "actions": actions,
            }
        )
        open_actions.extend(actions)
        if not actions:
            break

    deduped_actions: List[Dict[str, Any]] = []
    seen_action_keys = set()
    for action in open_actions:
        key = (action.get("owner"), action.get("priority"), action.get("task"))
        if key in seen_action_keys:
            continue
        seen_action_keys.add(key)
        deduped_actions.append(action)

    supervision = {
        "status": "needs_repair" if deduped_actions else "clear",
        "route": route,
        "rounds": rounds,
        "open_actions": deduped_actions,
        "human_review_triggers": sorted(set(human_triggers)),
        "policy": "Model outputs are not certified until Supervisor actions are resolved and QA gate clears.",
    }
    state["model_supervision"] = supervision
    _append_artifact(state, tools.json_dump(_run_dir(state) / "07b_model_supervision.json", supervision))
    return state


def claude_code_delegate_node(state: AnalysisState) -> AnalysisState:
    """Delegate concrete code/repair tasks to Claude Code when explicitly enabled."""
    supervision = state.get("model_supervision", {})
    actions = [
        action
        for action in supervision.get("open_actions", [])
        if str(action.get("owner", "")).lower().startswith("coder")
    ]
    if not state.get("enable_claude_code", False):
        result = {"status": "skipped", "reason": "enable_claude_code is false", "queued_actions": actions}
        state["claude_code"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "07c_claude_code.json", result))
        return state
    if not actions:
        result = {"status": "skipped", "reason": "no coder actions from model supervisor"}
        state["claude_code"] = result
        _append_artifact(state, tools.json_dump(_run_dir(state) / "07c_claude_code.json", result))
        return state

    prompt = (
        "You are Claude Code acting as the Coder node inside an astronomy research agent. "
        "Do not invent science results. Inspect the repository and propose/implement only bounded code changes needed for these tasks. "
        "If you edit files, list changed paths and required tests. Preserve existing user data.\n\n"
        f"Target: {state.get('target')}\n"
        f"Astrotool output: {state.get('data_fetch', {}).get('astrotool', {}).get('output_root')}\n"
        f"Analysis plan: {json.dumps(state.get('analysis_plan', {}), ensure_ascii=False)}\n"
        f"Actions: {json.dumps(actions, ensure_ascii=False, indent=2)}\n"
    )
    result = codex_tool.parse_claude_json(
        codex_tool.claude_code_exec(
            prompt,
            cwd=Path.cwd(),
            timeout=int(state.get("claude_timeout", 900) or 900),
            permission_mode=str(state.get("claude_permission_mode", "plan")),
        )
    )
    result["queued_actions"] = actions
    state["claude_code"] = result
    _append_artifact(state, tools.json_dump(_run_dir(state) / "07c_claude_code.json", result))
    return state


def qa_gate_node(state: AnalysisState) -> AnalysisState:
    qa = tools.detect_anomalies(state.get("data_fetch", {}), state.get("iterations", []))
    supervision = state.get("model_supervision", {})
    if supervision.get("status") == "needs_repair":
        qa.setdefault("reasons", [])
        qa["reasons"].extend(supervision.get("human_review_triggers", []))
        qa["reasons"].append("Model Supervisor has unresolved repair actions.")
        qa["human_review_required"] = True
    qa["apj_gate"] = "hold_for_human" if qa["human_review_required"] else "clear_for_draft"
    qa["checked_at_utc"] = datetime.utcnow().isoformat(timespec="seconds")
    state["qa"] = qa
    state["next_step"] = "abnormal" if qa["human_review_required"] else "paper"
    _append_artifact(state, tools.json_dump(_run_dir(state) / "08_qa_gate.json", qa))
    return state


def route_after_qa(state: AnalysisState) -> str:
    if state.get("next_step") == "abnormal" and state.get("draft_on_hold", False):
        return "paper"
    return state.get("next_step", "abnormal")


def abnormal_report_node(state: AnalysisState) -> AnalysisState:
    run_dir = _run_dir(state)
    qa = state.get("qa", {})
    text = [
        "# 异常分析报告",
        "",
        f"- Target: {state.get('target')}",
        f"- RA/Dec deg: {state.get('ra_deg')}, {state.get('dec_deg')}",
        f"- Gate: {qa.get('apj_gate')}",
        "",
        "## 暂停原因",
    ]
    for reason in qa.get("reasons", []):
        text.append(f"- {reason}")
    text.extend(
        [
            "",
            "## 已完成的三次迭代",
        ]
    )
    for iteration in state.get("iterations", []):
        text.append(
            f"- Iteration {iteration.get('iteration')}: {iteration.get('name')} "
            f"status={iteration.get('status')}"
        )
    text.extend(
        [
            "",
            "## 人类审核建议",
            "- 确认坐标、单位和交叉匹配对象是否正确。",
            "- 检查失败的巡天模块是否由网络、权限、无覆盖或真实非探测造成。",
            "- 若存在强发射线、非收敛拟合或异常冷却年龄，不要写入最终论文参数表。",
        ]
    )
    path = tools.write_text(run_dir / "abnormal_analysis_report.md", "\n".join(text) + "\n")
    guidance = codex_style.write_guidance(run_dir)
    manifest = tools.json_dump(run_dir / "agents_manifest.json", paper_agents.five_agent_manifest())
    state["abnormal_report"] = {"path": path, "status": "written"}
    _append_artifact(state, path)
    _append_artifact(state, guidance)
    _append_artifact(state, manifest)
    return state


def drafter_node(state: AnalysisState) -> AnalysisState:
    run_dir = _run_dir(state)
    orchestra = paper_orchestra.run_astro_paper_orchestra(
        run_dir,
        state,
        use_llm=bool(state.get("use_llm", False)),
        provider=state.get("llm_provider"),
        sectionwise=True,
        target_score=int(state.get("paper_target_score", 80)),
        max_refine_iters=int(state.get("paper_max_iters", 3)),
    )
    state["paper_orchestra"] = orchestra
    rag_rows = []
    for item in state.get("rag_results", []):
        rag_rows.extend(item.get("rows", []))
    context_path = tools.write_text(
        run_dir / "paper" / "method_context.md",
        "# Local RAG Evidence\n\n"
        + tools.format_rag_bullets(rag_rows)
        + "\n\n# KG Method Transfer\n\n"
        + tools.format_kg_bullets(state.get("kg_results", []))
        + "\n",
    )
    state["paper"] = {
        "status": "drafted",
        "tex": orchestra.get("final_tex"),
        "context": context_path,
        "paper_orchestra_workspace": orchestra.get("workspace"),
        "paper_orchestra_final_tex": orchestra.get("final_tex"),
        "final_review": orchestra.get("final_review", {}),
    }
    _append_artifact(state, context_path)
    for key in ("codex_style_guidance", "agents_manifest", "outline", "draft", "final_tex", "worklog", "provenance"):
        value = orchestra.get(key)
        if isinstance(value, str):
            _append_artifact(state, value)
    return state


def peer_reviewer_node(state: AnalysisState) -> AnalysisState:
    questions = [
        "所选大气模型网格是否覆盖目标的 Teff/logg/成分空间，网格边界附近的解是否被错误当成收敛解？",
        "光变曲线周期折叠是否被 ZTF/WISE/TESS 的采样窗口函数或日别名污染？是否有独立波段验证？",
        "Gaia 视差、消光、光谱通量定标和测光零点误差是否进入最终置信区间，而不是只报告统计拟合误差？",
        "若存在强发射线或 X-ray/IR excess，单白矮星模型是否仍是物理上最优解释？",
    ]
    text = "# Peer Reviewer Report\n\n" + "\n".join(f"- {q}" for q in questions) + "\n"
    path = tools.write_text(_run_dir(state) / "paper" / "peer_review.md", text)
    state["peer_review"] = {"status": "written", "questions": questions, "path": path}
    _append_artifact(state, path)
    return state


def toolbox_evolution_node(state: AnalysisState) -> AnalysisState:
    data = state.get("data_fetch", {})
    output = data.get("existing_outputs", {})
    missing = []
    for row in output.get("module_rows", []):
        if row.get("status") == "error":
            missing.append(
                {
                    "module": row.get("module"),
                    "note": row.get("note"),
                    "action": "Coder should reproduce the failure, add/patch the data adapter, add a smoke test, then update astro_toolbox/README.md.",
                }
            )
    if data.get("status") == "planned":
        missing.append(
            {
                "module": "all survey adapters",
                "note": "dry-run mode",
                "action": "Run with --execute before deciding whether new data-fetch scripts are required.",
            }
        )
    doc = {
        "status": "written",
        "policy": "When a missing capability is confirmed, the Coder must add the script, validate it on the target, and update astro_toolbox documentation in the same run.",
        "candidate_updates": missing,
    }
    path = tools.json_dump(_run_dir(state) / "toolbox_evolution_plan.json", doc)
    state["toolbox_evolution"] = {**doc, "path": path}
    _append_artifact(state, path)
    return state


def build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("resolve", resolve_node)
    graph.add_node("data_fetcher", data_fetcher_node)
    graph.add_node("structure_planner", structure_planner_node)
    graph.add_node("rag_navigator", rag_navigator_node)
    graph.add_node("kg_navigator", kg_navigator_node)
    graph.add_node("method_scout", method_scout_node)
    graph.add_node("kg_graph_report", kg_graph_report_node)
    graph.add_node("source_research_package", source_research_package_node)
    graph.add_node("iteration_1_baseline", baseline_iteration_node)
    graph.add_node("iteration_2_residuals", residual_physics_iteration_node)
    graph.add_node("iteration_3_systematics", errors_systematics_iteration_node)
    graph.add_node("model_supervisor", model_supervisor_node)
    graph.add_node("claude_code_delegate", claude_code_delegate_node)
    graph.add_node("qa_gate", qa_gate_node)
    graph.add_node("abnormal_report", abnormal_report_node)
    graph.add_node("drafter", drafter_node)
    graph.add_node("peer_reviewer", peer_reviewer_node)
    graph.add_node("toolbox_evolution", toolbox_evolution_node)

    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "data_fetcher")
    graph.add_edge("data_fetcher", "structure_planner")
    graph.add_edge("structure_planner", "rag_navigator")
    graph.add_edge("rag_navigator", "kg_navigator")
    graph.add_edge("kg_navigator", "method_scout")
    graph.add_edge("method_scout", "kg_graph_report")
    graph.add_edge("kg_graph_report", "source_research_package")
    graph.add_edge("source_research_package", "iteration_1_baseline")
    graph.add_edge("iteration_1_baseline", "iteration_2_residuals")
    graph.add_edge("iteration_2_residuals", "iteration_3_systematics")
    graph.add_edge("iteration_3_systematics", "model_supervisor")
    graph.add_edge("model_supervisor", "claude_code_delegate")
    graph.add_edge("claude_code_delegate", "qa_gate")
    graph.add_conditional_edges(
        "qa_gate",
        route_after_qa,
        {"abnormal": "abnormal_report", "paper": "drafter"},
    )
    graph.add_edge("abnormal_report", "toolbox_evolution")
    graph.add_edge("drafter", "peer_reviewer")
    graph.add_edge("peer_reviewer", "toolbox_evolution")
    graph.add_edge("toolbox_evolution", END)
    return graph.compile()


def run_workflow(
    target: str,
    output_root: str,
    ra_deg: float = None,
    dec_deg: float = None,
    dry_run: bool = True,
    force: bool = False,
    use_llm: bool = False,
    llm_provider: str = None,
    astrotool_run: str = None,
    kg_report: bool = False,
    kg_report_llm: bool = False,
    kg_report_provider: str = "deepseek",
    skip_simbad: bool = False,
    draft_on_hold: bool = False,
    method_scout_llm: bool = False,
    method_scout_provider: str = None,
    source_research_package: bool = False,
    download_simbad_pdfs: bool = False,
    enable_claude_code: bool = False,
    claude_timeout: int = 300,
    max_supervision_rounds: int = 2,
) -> AnalysisState:
    app = build_graph()
    initial: AnalysisState = {
        "target": target,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "output_root": output_root,
        "dry_run": dry_run,
        "force": force,
        "use_llm": use_llm,
        "llm_provider": llm_provider,
        "astrotool_run": astrotool_run,
        "kg_report": kg_report,
        "kg_report_llm": kg_report_llm,
        "kg_report_provider": kg_report_provider,
        "skip_simbad": skip_simbad,
        "draft_on_hold": draft_on_hold,
        "method_scout_llm": method_scout_llm,
        "method_scout_provider": method_scout_provider,
        "source_research_package": source_research_package,
        "download_simbad_pdfs": download_simbad_pdfs,
        "enable_claude_code": enable_claude_code,
        "claude_timeout": claude_timeout,
        "max_supervision_rounds": max_supervision_rounds,
        "artifacts": [],
        "warnings": [],
        "errors": [],
        "iterations": [],
    }
    return app.invoke(initial)
