"""LangGraph nodes for the Claude Code toolbox layer.

These are deliberately *additive* — they only read existing AnalysisState keys
(errors, paper_draft, toolbox_gap, latex_compile_error, etc.) and only write to
the new keys (claude_code_tasks, claude_code_results, *_report). They never
mutate analysis_plan, qa, paper, kg_results or other primary-pipeline state.
"""

from .claude_code_router import claude_code_router_node, decide_claude_code_tasks
from .claude_code_node import claude_code_execute_node
from .claude_code_diagnostic import claude_code_diagnostic_node
from .patch_review_node import patch_review_node
from .simbad_prior import simbad_prior_node, build_simbad_prior
from .qa_gate import classify_qa, route_after_qa_v2, write_qa_artifact

__all__ = [
    "claude_code_router_node",
    "decide_claude_code_tasks",
    "claude_code_execute_node",
    "claude_code_diagnostic_node",
    "patch_review_node",
    "simbad_prior_node",
    "build_simbad_prior",
    "classify_qa",
    "route_after_qa_v2",
    "write_qa_artifact",
]
