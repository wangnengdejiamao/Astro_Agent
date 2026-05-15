"""Literature-to-toolbox learning helpers.

This module keeps the "learn a method from papers" loop explicit:

1. compact RAG/KG evidence into a bounded packet,
2. turn the evidence into an executable algorithm specification,
3. build a toolbox gap that Claude Code can implement,
4. register a dynamic skill only after validation is explicitly reported.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


PACKAGE_DIR = Path(__file__).resolve().parent
ASTRO_AGENT_DIR = PACKAGE_DIR.parent


def _safe_slug(value: Any, fallback: str = "literature_method") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"m_{text}"
    return text[:60]


def _shorten(value: Any, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _iter_rag_rows(rag_hits: Iterable[Mapping[str, Any]]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for hit in rag_hits or []:
        query = str(hit.get("query", ""))
        for row in hit.get("rows", []) or []:
            if isinstance(row, Mapping):
                yield query, row


def collect_method_evidence(
    *,
    state: Mapping[str, Any],
    scout: Optional[Mapping[str, Any]] = None,
    max_rag: int = 10,
    max_kg: int = 10,
) -> Dict[str, Any]:
    """Collect bounded RAG/KG evidence used to justify a new method."""
    scout = scout or {}
    rag_hits = scout.get("rag_hits") or state.get("rag_results") or []
    kg_hits = scout.get("kg_hits") or state.get("kg_results") or []

    rag_refs: List[Dict[str, Any]] = []
    seen_bibcodes = set()
    for query, row in _iter_rag_rows(rag_hits):
        bibcode = str(row.get("bibcode") or row.get("id") or "")
        dedupe_key = (bibcode, row.get("chunk_id"))
        if dedupe_key in seen_bibcodes:
            continue
        seen_bibcodes.add(dedupe_key)
        rag_refs.append(
            {
                "query": query,
                "chunk_id": row.get("chunk_id"),
                "bibcode": bibcode,
                "title": row.get("title"),
                "year": row.get("year"),
                "journal": row.get("journal"),
                "section": row.get("section"),
                "methods": _json_loads_maybe(row.get("methods_json")),
                "instruments": _json_loads_maybe(row.get("instruments_json")),
                "snippet": _shorten(row.get("snippet"), 900),
            }
        )
        if len(rag_refs) >= max_rag:
            break

    kg_refs: List[Dict[str, Any]] = []
    for row in kg_hits or []:
        if not isinstance(row, Mapping):
            continue
        kg_refs.append(
            {
                "score": row.get("score"),
                "subject": row.get("subject"),
                "subject_type": row.get("subject_type"),
                "relation": row.get("relation"),
                "object": row.get("object"),
                "object_type": row.get("object_type"),
                "title": row.get("title"),
                "chunk_id": row.get("chunk_id"),
                "source": _shorten(row.get("source"), 700),
                "evidence": _shorten(row.get("evidence"), 500),
            }
        )
        if len(kg_refs) >= max_kg:
            break

    return {
        "target": state.get("target"),
        "route": (state.get("analysis_plan") or {}).get("route"),
        "rag_refs": rag_refs,
        "kg_refs": kg_refs,
        "reference_bibcodes": sorted({str(r.get("bibcode")) for r in rag_refs if r.get("bibcode")}),
        "collected_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def build_algorithm_extraction_prompt(evidence_pack: Mapping[str, Any]) -> tuple[str, str]:
    """Prompt used by method_scout_node when LLM extraction is enabled."""
    system = (
        "You are an astrophysics method-learning agent. Return strict JSON only. "
        "Extract executable analysis methods from the supplied RAG/KG evidence. "
        "Do not invent equations, thresholds, inputs, outputs, or validation criteria not supported by the evidence; "
        "mark unknown details as requiring_human_review."
    )
    user = (
        "From the evidence below, produce JSON with keys: "
        "algorithm_specs (list), recommended_methods (list), coder_tasks (list), "
        "qa_risks (list), human_review_triggers (list). Each algorithm_spec must include: "
        "method_name, target_module, objective, inputs, outputs, algorithm_steps, "
        "constraints, validation_plan, reference_bibcodes, confidence, unsupported_details.\n\n"
        + json.dumps(evidence_pack, ensure_ascii=False, indent=2)[:20000]
    )
    return system, user


def _first_llm_algorithm_spec(scout: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    parsed = scout.get("llm_recommendations")
    if not isinstance(parsed, Mapping):
        return None
    specs = parsed.get("algorithm_specs") or []
    if isinstance(specs, list) and specs:
        first = specs[0]
        if isinstance(first, Mapping):
            return dict(first)
    methods = parsed.get("recommended_methods") or []
    if isinstance(methods, list) and methods:
        first = methods[0]
        if isinstance(first, Mapping):
            return dict(first)
        return {"method_name": str(first), "algorithm_steps": [str(first)]}
    return None


def infer_target_module(route: str, evidence_text: str) -> str:
    text = f"{route} {evidence_text}".lower()
    if any(key in text for key in ("lsst", "ztf", "tess", "period", "light curve", "light-curve", "cadence", "fold")):
        return "period_analysis"
    if any(key in text for key in ("radial velocity", "rv", "orbit", "phase space", "traceback")):
        return "rv_orbit_method"
    if any(key in text for key in ("x-ray", "xray", "efeds", "erosita")):
        return "xray"
    if any(key in text for key in ("sed", "photometric", "parallax", "bayesian", "cooling age", "teff", "logg")):
        return "wd_fitting"
    if any(key in text for key in ("spectrum", "spectral", "line", "balmer", "emission")):
        return "spectral_fitting"
    return "literature_method"


def build_algorithm_spec(state: Mapping[str, Any], scout: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a normalized algorithm spec from LLM output or evidence fallback."""
    evidence_pack = scout.get("method_evidence") or collect_method_evidence(state=state, scout=scout)
    llm_spec = _first_llm_algorithm_spec(scout)
    route = str((state.get("analysis_plan") or {}).get("route", "unknown"))
    evidence_text = json.dumps(evidence_pack, ensure_ascii=False)[:12000]

    if llm_spec:
        method_name = str(llm_spec.get("method_name") or llm_spec.get("name") or "literature-derived method")
        target_module = _safe_slug(llm_spec.get("target_module") or infer_target_module(route, evidence_text))
        return {
            "status": "llm_extracted",
            "method_name": method_name,
            "target_module": target_module,
            "objective": llm_spec.get("objective") or f"Implement {method_name} for the current analysis route.",
            "inputs": llm_spec.get("inputs") or [],
            "outputs": llm_spec.get("outputs") or [],
            "algorithm_steps": llm_spec.get("algorithm_steps") or llm_spec.get("algorithm") or [],
            "constraints": llm_spec.get("constraints") or [],
            "validation_plan": llm_spec.get("validation_plan") or [],
            "reference_bibcodes": llm_spec.get("reference_bibcodes") or evidence_pack.get("reference_bibcodes", []),
            "confidence": llm_spec.get("confidence", "medium"),
            "unsupported_details": llm_spec.get("unsupported_details") or [],
            "evidence": evidence_pack,
        }

    target_module = infer_target_module(route, evidence_text)
    return {
        "status": "evidence_summary_only",
        "method_name": f"{route} literature method candidate",
        "target_module": target_module,
        "objective": "Candidate method was found in RAG/KG evidence, but no LLM-extracted executable spec is available.",
        "inputs": ["target identifier", "RA/Dec", "available local astro_toolbox products"],
        "outputs": ["structured result JSON", "validation report", "provenance-linked artifacts"],
        "algorithm_steps": [
            "Review the cited RAG/KG evidence before implementation.",
            "Extract equations, priors, model grids, quality cuts, and output schema with human review.",
            "Implement only after the executable specification is complete.",
        ],
        "constraints": [
            "Do not create a tool from evidence_summary_only without human or LLM extraction review.",
            "All numerical outputs must carry units and provenance.",
        ],
        "validation_plan": [
            "Create a smoke test using a known local target or fixture.",
            "Compare outputs against at least one independent observable or published sanity range.",
        ],
        "reference_bibcodes": evidence_pack.get("reference_bibcodes", []),
        "confidence": "low",
        "unsupported_details": ["algorithm details require extraction before code generation"],
        "evidence": evidence_pack,
    }


def build_toolbox_gap(state: Mapping[str, Any], algorithm_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an algorithm spec into a Claude Code TOOL_WRITE specification."""
    target_module = _safe_slug(algorithm_spec.get("target_module"))
    executable = algorithm_spec.get("status") == "llm_extracted"
    similar_modules = [
        "astro_toolbox/wd_fitting.py",
        "astro_toolbox/sed.py",
        "astro_toolbox/period_analysis.py",
        "astro_toolbox/rv_fitting.py",
        "astro_toolbox/orbit_traceback.py",
    ]
    return {
        "status": "ready_for_tool_write" if executable else "draft_needs_algorithm_extraction",
        "target_module": target_module,
        "method_name": algorithm_spec.get("method_name"),
        "capability": algorithm_spec.get("objective"),
        "algorithm_spec": dict(algorithm_spec),
        "reference_bibcodes": algorithm_spec.get("reference_bibcodes", []),
        "interface_contract": {
            "module_path": f"astro_toolbox/{target_module}.py",
            "primary_function": f"analyze_{target_module}",
            "signature": "def analyze_<module>(ra: float, dec: float, output_dir: str | Path, **kwargs) -> dict",
            "return_contract": "Return a JSON-serializable dict with status, products, warnings, provenance, validation.",
            "artifact_contract": "Write machine-readable JSON/CSV artifacts under output_dir and never silently overwrite unrelated files.",
            "batch_contract": "If the method is target-level, add an opt-in call in run_single_target_all_tools.py and include module_status.csv reporting.",
        },
        "validation_plan": algorithm_spec.get("validation_plan", []),
        "acceptance_criteria": [
            "Implementation is backed by the supplied reference_bibcodes and records them in output provenance.",
            "Smoke test or fixture validates success and one controlled failure path.",
            "Outputs include units, assumptions, boundary flags, and validation status.",
            "The module can be imported from astro_toolbox without breaking existing imports.",
            "No final scientific claim is emitted when required inputs are missing or validation fails.",
        ],
        "similar_modules": similar_modules,
        "skill_registration": {
            "claude_agent": f".claude/agents/astro-{target_module}-analyzer.md",
            "codex_skill": f"skills/astro-{target_module}/SKILL.md",
            "registry": "skills/registry.json",
            "gate": "Register only after validation_passed=true or tests_passed=true is reported by Claude Code.",
        },
        "human_gate": "Required before execution" if not executable else "Required before accepting scientific use",
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def render_toolbox_gap_prompt(gap: Mapping[str, Any]) -> str:
    return json.dumps(gap, ensure_ascii=False, indent=2)[:24000]


def _parsed_claude_payload(result: Mapping[str, Any]) -> Mapping[str, Any]:
    def _unwrap(obj: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = obj.get("result")
        if isinstance(nested, str):
            try:
                parsed_nested = json.loads(nested)
                if isinstance(parsed_nested, Mapping):
                    return parsed_nested
            except Exception:
                pass
        return obj

    parsed = result.get("parsed")
    if isinstance(parsed, Mapping):
        return _unwrap(parsed)
    raw_stdout = result.get("raw_stdout") or result.get("stdout") or ""
    if isinstance(raw_stdout, str):
        try:
            obj = json.loads(raw_stdout)
            if isinstance(obj, Mapping):
                return _unwrap(obj)
        except Exception:
            pass
    return {}


def validation_passed(result: Mapping[str, Any]) -> bool:
    payload = _parsed_claude_payload(result)
    for key in ("validation_passed", "tests_passed", "smoke_tests_passed"):
        if payload.get(key) is True:
            return True
    validation = payload.get("validation")
    if isinstance(validation, Mapping):
        return any(validation.get(key) is True for key in ("passed", "tests_passed", "smoke_tests_passed"))
    return False


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def register_dynamic_skill_if_valid(
    *,
    gap: Optional[Mapping[str, Any]],
    claude_result: Optional[Mapping[str, Any]],
    project_root: Path = ASTRO_AGENT_DIR,
) -> Dict[str, Any]:
    """Register a project-local Claude/Codex skill after explicit validation."""
    if not gap:
        return {"status": "skipped", "reason": "no toolbox_gap"}
    if not claude_result:
        return {"status": "skipped", "reason": "no claude_result"}
    if not validation_passed(claude_result):
        return {
            "status": "skipped",
            "reason": "validation_not_confirmed",
            "gate": "Claude Code must report validation_passed=true or tests_passed=true.",
        }

    target_module = _safe_slug(gap.get("target_module"))
    method_name = str(gap.get("method_name") or target_module)
    payload = _parsed_claude_payload(claude_result)
    changed_files = payload.get("changed_files") or claude_result.get("changed_files") or []
    tests = payload.get("tests") or claude_result.get("tests") or []

    agent_path = project_root / ".claude" / "agents" / f"astro-{target_module}-analyzer.md"
    skill_dir = project_root / "skills" / f"astro-{target_module}"
    skill_path = skill_dir / "SKILL.md"
    registry_path = project_root / "skills" / "registry.json"

    agent_path.parent.mkdir(parents=True, exist_ok=True)
    skill_dir.mkdir(parents=True, exist_ok=True)

    agent_path.write_text(
        "\n".join(
            [
                f"# astro-{target_module}-analyzer",
                "",
                f"Use this agent when an analysis requires the validated `{method_name}` capability.",
                "",
                "## Evidence",
                f"- Reference bibcodes: {', '.join(str(x) for x in gap.get('reference_bibcodes', [])) or 'not recorded'}",
                "",
                "## Tool Contract",
                f"- Module: `{(gap.get('interface_contract') or {}).get('module_path', f'astro_toolbox/{target_module}.py')}`",
                f"- Primary function: `{(gap.get('interface_contract') or {}).get('primary_function', f'analyze_{target_module}')}`",
                "- Require provenance, units, warnings, and validation status in every result.",
                "",
                "## Validation",
                f"- Changed files: {changed_files}",
                f"- Tests: {tests}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    skill_path.write_text(
        "\n".join(
            [
                f"# astro-{target_module}",
                "",
                f"Use this skill for the validated `{method_name}` analysis method added to `astro_toolbox`.",
                "",
                "## When To Use",
                f"- The current target needs `{gap.get('capability')}`.",
                "- Required inputs are available and match the module contract.",
                "",
                "## Workflow",
                f"1. Import the toolbox module described in `{(gap.get('interface_contract') or {}).get('module_path', '')}`.",
                "2. Run the primary analysis function with RA, Dec, output_dir, and method-specific kwargs.",
                "3. Check the returned validation status before using outputs in science claims.",
                "4. Cite the recorded reference bibcodes and local output artifacts.",
                "",
                "## Registration Metadata",
                f"- Registered at: {datetime.utcnow().isoformat(timespec='seconds')}Z",
                f"- Reference bibcodes: {', '.join(str(x) for x in gap.get('reference_bibcodes', [])) or 'not recorded'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    registry: Dict[str, Any] = {}
    if registry_path.exists():
        try:
            loaded = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                registry = dict(loaded)
        except Exception:
            registry = {}
    registry.setdefault("skills", {})
    registry["skills"][target_module] = {
        "method_name": method_name,
        "agent_path": str(agent_path),
        "skill_path": str(skill_path),
        "reference_bibcodes": gap.get("reference_bibcodes", []),
        "changed_files": changed_files,
        "tests": tests,
        "registered_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _write_json(registry_path, registry)

    return {
        "status": "registered",
        "target_module": target_module,
        "agent_path": str(agent_path),
        "skill_path": str(skill_path),
        "registry_path": str(registry_path),
    }
