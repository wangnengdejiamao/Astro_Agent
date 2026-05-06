"""Shared state schema for the astronomy analysis agent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AnalysisState(TypedDict, total=False):
    """Mutable workflow state passed through LangGraph nodes."""

    target: str
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    output_root: str
    dry_run: bool
    force: bool
    use_llm: bool
    llm_provider: str
    astrotool_run: Optional[str]
    kg_report: bool
    kg_report_llm: bool
    kg_report_provider: str
    skip_simbad: bool
    draft_on_hold: bool
    method_scout_llm: bool
    method_scout_provider: str
    source_research_package: bool
    download_simbad_pdfs: bool
    enable_claude_code: bool
    claude_timeout: int
    max_supervision_rounds: int

    resolved: Dict[str, Any]
    data_fetch: Dict[str, Any]
    analysis_plan: Dict[str, Any]
    rag_results: List[Dict[str, Any]]
    kg_results: List[Dict[str, Any]]
    kg_graph_report: Dict[str, Any]
    source_research: Dict[str, Any]
    method_scout: Dict[str, Any]
    iterations: List[Dict[str, Any]]
    model_supervision: Dict[str, Any]
    claude_code: Dict[str, Any]
    qa: Dict[str, Any]
    paper: Dict[str, Any]
    paper_orchestra: Dict[str, Any]
    peer_review: Dict[str, Any]
    toolbox_evolution: Dict[str, Any]
    abnormal_report: Dict[str, Any]

    artifacts: List[str]
    warnings: List[str]
    errors: List[str]
    next_step: str

    # Claude Code toolbox layer (additive, optional).
    claude_code_tasks: List[Dict[str, Any]]
    claude_code_results: List[Dict[str, Any]]
    claude_code_pending_review: List[Dict[str, Any]]
    toolbox_gap_report: Dict[str, Any]
    code_quality_report: Dict[str, Any]
    paper_refinement_report: Dict[str, Any]
    needs_human_review: bool
