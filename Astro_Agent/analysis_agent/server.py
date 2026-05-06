"""FastAPI server for Astro_Agent.

Endpoints:
- GET  /                  → 单页前端
- GET  /api/health        → 服务存活 + provider 状态
- POST /api/agent/run     → 跑完整 LangGraph workflow
- POST /api/rag/search    → 直接查 RAG SQLite
- POST /api/kg/search     → 直接查 KG (sqlite + json fallback)
- POST /api/toolbox/run   → 调用单个 astro_toolbox 模块函数
- POST /api/codex/exec    → 子进程调用 Codex CLI
- POST /api/claude/exec   → 子进程调用 Claude Code CLI
- GET  /api/runs          → 列出 output/analysis_agent 历史
- GET  /api/runs/{name}/{file} → 读取某次 run 的产物 JSON

启动:
    cd /mnt/c/Users/Administrator/Desktop/rag/Astro_Agent
    python -m uvicorn analysis_agent.server:app --host 0.0.0.0 --port 8765 --reload
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).resolve().parent
ASTRO_AGENT_DIR = PACKAGE_DIR.parent
sys.path.insert(0, str(ASTRO_AGENT_DIR))

from analysis_agent import codex_tool, tools  # noqa: E402
from analysis_agent.llm_client import LLMClient, load_default_env, load_model_config  # noqa: E402
from analysis_agent.workflow import run_workflow  # noqa: E402

load_default_env()

app = FastAPI(title="Astro_Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = ASTRO_AGENT_DIR / "web"
RUNS_DIR = ASTRO_AGENT_DIR / "output" / "analysis_agent"


@app.get("/")
def index():
    f = WEB_DIR / "index.html"
    if not f.exists():
        raise HTTPException(404, "frontend not built")
    return FileResponse(str(f))


@app.get("/api/health")
def health():
    cfg = load_model_config()
    client = LLMClient(cfg)
    return {
        "status": "ok",
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "llm_available": client.available,
        "rag_db_exists": tools.RAG_DB.exists(),
        "kg_index_exists": tools.KG_INDEX.exists(),
        "codex_bin": os.getenv("ASTRO_AGENT_CODEX_BIN", ""),
        "claude_bin": os.getenv("ASTRO_AGENT_CLAUDE_BIN", ""),
        "now": datetime.utcnow().isoformat() + "Z",
    }


# ---------- Agent ----------
class AgentRunReq(BaseModel):
    target: str = Field(..., description="Target name or label, e.g. 'Gaia DR3 ...'")
    ra: Optional[float] = None
    dec: Optional[float] = None
    execute: bool = False
    use_llm: bool = False
    llm_provider: Optional[str] = Field(None, description="deepseek, fox, gemini, kimi")
    astrotool_run: Optional[str] = Field(None, description="Existing astro_toolbox output directory")
    skip_simbad: bool = False
    kg_report: bool = False
    kg_report_llm: bool = False
    kg_report_provider: str = "deepseek"
    draft_on_hold: bool = False
    method_scout_llm: bool = False
    method_scout_provider: Optional[str] = None
    enable_claude_code: bool = False
    max_supervision_rounds: int = 2
    output_root: Optional[str] = None


@app.post("/api/agent/run")
def agent_run(req: AgentRunReq):
    safe = "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in req.target).strip("_")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = req.output_root or str(RUNS_DIR / f"{safe or 'target'}_{stamp}")
    state = run_workflow(
        target=req.target,
        ra_deg=req.ra,
        dec_deg=req.dec,
        output_root=out_root,
        dry_run=not req.execute,
        force=req.execute,
        use_llm=req.use_llm,
        llm_provider=req.llm_provider,
        astrotool_run=req.astrotool_run,
        skip_simbad=req.skip_simbad,
        kg_report=req.kg_report,
        kg_report_llm=req.kg_report_llm,
        kg_report_provider=req.kg_report_provider,
        draft_on_hold=req.draft_on_hold,
        method_scout_llm=req.method_scout_llm,
        method_scout_provider=req.method_scout_provider,
        enable_claude_code=req.enable_claude_code,
        max_supervision_rounds=req.max_supervision_rounds,
    )
    return {
        "output_root": out_root,
        "qa": state.get("qa", {}),
        "resolved": state.get("resolved", {}),
        "artifacts": state.get("artifacts", []),
        "analysis_plan": state.get("analysis_plan", {}),
        "method_scout": state.get("method_scout", {}),
        "model_supervision": state.get("model_supervision", {}),
        "claude_code": state.get("claude_code", {}),
        "kg_graph_report": state.get("kg_graph_report", {}),
        "abnormal_report": state.get("abnormal_report", {}),
    }


# ---------- RAG ----------
class RagReq(BaseModel):
    query: str
    method_only: bool = False
    limit: int = 5


@app.post("/api/rag/search")
def rag_search(req: RagReq):
    rows = tools.search_rag(req.query, method_only=req.method_only, limit=req.limit)
    return {"query": req.query, "n": len(rows), "rows": rows}


# ---------- KG ----------
class KgReq(BaseModel):
    queries: List[str]
    limit: int = 12


@app.post("/api/kg/search")
def kg_search(req: KgReq):
    rows = tools.search_kg(req.queries, limit=req.limit)
    return {"n": len(rows), "rows": rows}


# ---------- astro_toolbox ----------
class ToolboxReq(BaseModel):
    module: str = Field(..., description="e.g. 'sdss', 'ztf', 'sed'")
    function: str = Field(..., description="e.g. 'query_spectrum', 'query_lightcurve'")
    ra: float
    dec: float
    radius_arcsec: Optional[float] = None
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict)


@app.post("/api/toolbox/run")
def toolbox_run(req: ToolboxReq):
    sys.path.insert(0, str(ASTRO_AGENT_DIR))
    try:
        mod = __import__(f"astro_toolbox.{req.module}", fromlist=[req.function])
    except Exception as exc:
        raise HTTPException(400, f"import astro_toolbox.{req.module} failed: {exc}")
    fn = getattr(mod, req.function, None)
    if fn is None:
        raise HTTPException(400, f"{req.module}.{req.function} not found")
    kwargs = dict(req.extra_kwargs or {})
    if req.radius_arcsec is not None:
        kwargs.setdefault("radius_arcsec", req.radius_arcsec)
    try:
        result = fn(req.ra, req.dec, **kwargs)
    except TypeError:
        result = fn(req.ra, req.dec)
    return JSONResponse(content=_to_jsonable(result))


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion for numpy / astropy types."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    try:
        import json as _j

        _j.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ---------- Paper Orchestra (writing only) ----------
class PaperReq(BaseModel):
    run_name: str = Field(..., description="A folder under output/analysis_agent/, e.g. ZTFJ152934_full_agent_run")
    use_llm: bool = True
    llm_provider: Optional[str] = "deepseek"
    sectionwise: bool = True
    target_score: int = 80
    max_iters: int = 3


@app.post("/api/paper/draft")
def paper_draft(req: PaperReq):
    """Re-run only the PaperOrchestra writing on an existing agent run dir."""
    run_dir = (RUNS_DIR / req.run_name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    if not run_dir.exists():
        raise HTTPException(404, f"run not found: {req.run_name}")

    from analysis_agent import paper_orchestra

    state: Dict[str, Any] = {}
    for fname, key in [
        ("01_resolved_target.json", "resolved"),
        ("02_data_fetch.json", "data_fetch"),
        ("03_rag_results.json", "rag_results"),
        ("04_kg_results.json", "kg_results"),
        ("08_qa_gate.json", "qa"),
    ]:
        p = run_dir / fname
        if p.exists():
            state[key] = tools.read_json(p)
    iterations = []
    for fname in ("05_iteration_1_baseline.json", "06_iteration_2_residuals.json", "07_iteration_3_systematics.json"):
        p = run_dir / fname
        if p.exists():
            iterations.append(tools.read_json(p))
    state["iterations"] = iterations

    resolved = state.get("resolved") or {}
    state["target"] = resolved.get("target") or req.run_name
    state["ra_deg"] = resolved.get("ra_deg")
    state["dec_deg"] = resolved.get("dec_deg")

    result = paper_orchestra.run_astro_paper_orchestra(
        run_dir,
        state,
        use_llm=req.use_llm,
        provider=req.llm_provider,
        sectionwise=req.sectionwise,
        target_score=req.target_score,
        max_refine_iters=req.max_iters,
    )
    return result


@app.get("/api/paper/sections/{run_name}")
def paper_sections(run_name: str):
    """Return the per-section LaTeX files for inspection in the UI."""
    run_dir = (RUNS_DIR / run_name).resolve()
    drafts = run_dir / "paper_orchestra" / "drafts"
    if not drafts.exists():
        raise HTTPException(404, "no drafts; run /api/paper/draft first")
    out: Dict[str, str] = {}
    for f in sorted(drafts.iterdir()):
        if f.is_file() and f.suffix == ".tex":
            out[f.name] = f.read_text(encoding="utf-8", errors="replace")
    final_p = run_dir / "paper_orchestra" / "final" / "paper.tex"
    if final_p.exists():
        out["__final__paper.tex"] = final_p.read_text(encoding="utf-8", errors="replace")
    review_p = run_dir / "paper_orchestra" / "refinement" / "worklog.json"
    if review_p.exists():
        out["__review__worklog.json"] = review_p.read_text(encoding="utf-8")
    return out


# ---------- GUI: AstroQueryAll (gui.py 的多模块单目标接口) ----------
GUI_MODULE_LIST = [
    ("SDSS_spectrum", "Spectra", "SDSS DR18 optical spectrum"),
    ("GALAH", "Spectra", "GALAH DR4 info"),
    ("LAMOST", "Spectra", "LAMOST DR8 optical spectrum"),
    ("DESI", "Spectra", "DESI DR1 B/R/Z spectrum"),
    ("KOA_spectrum", "Spectra", "KOA/Keck local LRIS spectrum"),
    ("SPHEREx", "Spectra", "SPHEREx low-resolution spectrum"),
    ("ZTF_lightcurve", "LightCurve", "ZTF DR23 g/r/i"),
    ("WISE_lightcurve", "LightCurve", "NEOWISE W1/W2"),
    ("Gaia_lightcurve", "LightCurve", "Gaia DR3 epoch phot"),
    ("TESS", "LightCurve", "TESS SPOC"),
    ("Kepler/K2", "LightCurve", "Kepler or K2"),
    ("HST_spectrum", "Spectra", "HST COS/STIS spectrum"),
    ("HST_lightcurve", "LightCurve", "HST multi-epoch phot"),
    ("JWST_spectrum", "Spectra", "JWST NIRSpec/MIRI spectrum"),
    ("JWST_lightcurve", "LightCurve", "JWST multi-epoch phot"),
    ("SDSS_photometry", "Photometry", "SDSS ugriz"),
    ("GALEX", "Photometry", "GALEX FUV/NUV"),
    ("2MASS", "Photometry", "2MASS JHKs"),
    ("WISE_photometry", "Photometry", "AllWISE W1-W4"),
    ("X-ray", "X-ray", "ROSAT/XMM/Chandra/eROSITA"),
    ("HEASARC_Xray", "X-ray", "HEASARC Browse (Swift/NuSTAR/...)"),
    ("SED", "Analysis", "Multi-band SED + BB fit"),
    ("HR_diagram", "Analysis", "Gaia HR diagram + region/age"),
    ("Binary_SED", "Analysis", "WD+M-dwarf binary SED fit"),
    ("SIMBAD_refs", "Analysis", "SIMBAD literature refs"),
]


@app.get("/api/gui/modules")
def gui_modules():
    return {"modules": [{"name": n, "group": g, "desc": d} for n, g, d in GUI_MODULE_LIST]}


class GuiRunReq(BaseModel):
    ra: float
    dec: float
    enabled_modules: List[str] = Field(default_factory=list, description="If empty, runs all modules")


@app.post("/api/gui/run")
def gui_run(req: GuiRunReq):
    """Run AstroQueryAll on a single target with selected modules — same as gui.py 'Run' button."""
    scripts_dir = ASTRO_AGENT_DIR / "scripts"
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(ASTRO_AGENT_DIR))
    try:
        from test_toolbox import AstroQueryAll  # type: ignore
    except Exception as exc:
        raise HTTPException(500, f"AstroQueryAll import failed: {exc}")

    output_dir = ASTRO_AGENT_DIR / "output" / "astro_output" / f"RA{req.ra:.4f}_DEC{req.dec:.4f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    enabled = set(req.enabled_modules) if req.enabled_modules else None

    statuses: List[Dict[str, Any]] = []

    def cb(name: str, status: str, result: Any, elapsed: float) -> None:
        statuses.append({"module": name, "status": str(status), "elapsed_sec": float(elapsed)})

    q = AstroQueryAll(req.ra, req.dec, output_dir=str(output_dir), enabled_modules=enabled)
    q.status_callback = cb
    try:
        q.query_all()
        q.save_and_plot_all()
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}", "statuses": statuses, "output_dir": str(output_dir)}

    plots = sorted(str(p.relative_to(ASTRO_AGENT_DIR)) for p in output_dir.rglob("*.png"))
    csvs = sorted(str(p.relative_to(ASTRO_AGENT_DIR)) for p in output_dir.rglob("*.csv"))
    return {
        "status": "ok",
        "output_dir": str(output_dir),
        "statuses": statuses,
        "plots": plots,
        "csvs": csvs,
    }


@app.get("/api/files/{path:path}")
def get_file(path: str):
    """Serve a file from inside Astro_Agent/ (read-only, sandbox-checked)."""
    target = (ASTRO_AGENT_DIR / path).resolve()
    if not str(target).startswith(str(ASTRO_AGENT_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(str(target))


# ---------- Codex / Claude Code ----------
class ExecReq(BaseModel):
    prompt: str
    cwd: Optional[str] = None
    timeout: int = 600


@app.post("/api/codex/exec")
def codex_exec_route(req: ExecReq):
    cwd = Path(req.cwd) if req.cwd else None
    return codex_tool.codex_exec(req.prompt, cwd=cwd, timeout=req.timeout)


@app.post("/api/claude/exec")
def claude_exec_route(req: ExecReq):
    cwd = Path(req.cwd) if req.cwd else None
    return codex_tool.parse_claude_json(codex_tool.claude_code_exec(req.prompt, cwd=cwd, timeout=req.timeout))


# ---------- Claude Code Toolbox (additive layer) ----------
class ClaudeCodeTaskReq(BaseModel):
    type: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[str] = None
    dry_run: bool = True


@app.post("/api/claude_code/task")
def claude_code_task(req: ClaudeCodeTaskReq):
    """Build (and optionally execute) a single ClaudeCodeTask."""
    from claude_code_toolbox import build_task
    from claude_code_toolbox.client import make_default_client
    from claude_code_toolbox.schemas import ClaudeCodeTaskType

    try:
        ttype = ClaudeCodeTaskType(req.type)
    except ValueError:
        raise HTTPException(400, f"unknown task type: {req.type}")

    task = build_task(ttype, req.inputs)
    if req.dry_run:
        return {"task": task.model_dump(mode="json"), "dry_run": True}

    client = make_default_client(run_id=req.run_id)
    result = client.execute(task)
    return {"task": task.model_dump(mode="json"), "result": result.model_dump(mode="json")}


@app.get("/api/claude_code/runs/{run_id}")
def claude_code_runs(run_id: str):
    """List all Claude Code task outputs for a given run."""
    base = (ASTRO_AGENT_DIR / "runs" / run_id / "claude_code").resolve()
    runs_root = (ASTRO_AGENT_DIR / "runs").resolve()
    if not str(base).startswith(str(runs_root)):
        raise HTTPException(403, "path traversal blocked")
    if not base.exists():
        return {"run_id": run_id, "tasks": []}
    tasks = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            tasks.append({
                "task_id": d.name,
                "files": sorted(f.name for f in d.iterdir() if f.is_file()),
                "raw_output_path": str(d / "raw.json"),
            })
    return {"run_id": run_id, "tasks": tasks}


# ---------- Runs / artifacts ----------
@app.get("/api/runs")
def list_runs():
    if not RUNS_DIR.exists():
        return {"runs": []}
    runs = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if p.is_dir():
            runs.append({"name": p.name, "files": sorted(f.name for f in p.iterdir() if f.is_file())})
    return {"runs": runs[:50]}


@app.get("/api/runs/{name}/{filename}")
def get_run_artifact(name: str, filename: str):
    target = (RUNS_DIR / name / filename).resolve()
    if not str(target).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    if not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(target))
