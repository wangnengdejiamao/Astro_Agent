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
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

PACKAGE_DIR = Path(__file__).resolve().parent
ASTRO_AGENT_DIR = PACKAGE_DIR.parent
sys.path.insert(0, str(ASTRO_AGENT_DIR))

from analysis_agent import codex_tool, tools  # noqa: E402
from analysis_agent.llm_client import LLMClient, load_default_env, load_model_config  # noqa: E402
from analysis_agent.workflow import run_workflow  # noqa: E402
from analysis_agent.workflow_trace import build_trace, build_trace_from_run, workflow_blueprint  # noqa: E402

load_default_env()

app = FastAPI(title="Astro_Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = ASTRO_AGENT_DIR / "web"
KG_WORKSPACE = Path(os.getenv("ASTRO_AGENT_KG_WORKSPACE", str(ASTRO_AGENT_DIR.parent / ".local_kg")))
FRONTEND_DIR = KG_WORKSPACE / "frontend"
RUNS_DIR = ASTRO_AGENT_DIR / "output" / "analysis_agent"

# AI CLI tools configuration
KIMI_BIN = os.getenv("KIMI_BIN", "kimi")
CLAUDE_CODE_BIN = os.getenv("CLAUDE_CODE_BIN", "claude")
CODEX_BIN = os.getenv("CODEX_BIN", "codex")


@app.get("/")
def index():
    f = WEB_DIR / "index.html"
    if not f.exists():
        raise HTTPException(404, "frontend not built")
    return FileResponse(str(f))


@app.get("/favicon.ico")
def favicon():
    # Stop browsers spamming 404 in the log; we don't ship a real icon.
    return Response(status_code=204)


@app.get("/kg")
def kg_frontend():
    """Knowledge graph visualization page."""
    f = FRONTEND_DIR / "index.html"
    if not f.exists():
        raise HTTPException(404, "KG frontend not found")
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
        "codex_bin": CODEX_BIN if shutil.which(CODEX_BIN) else "",
        "claude_bin": CLAUDE_CODE_BIN if shutil.which(CLAUDE_CODE_BIN) else "",
        "kimi_bin": KIMI_BIN if shutil.which(KIMI_BIN) else "",
        "now": datetime.utcnow().isoformat() + "Z",
    }


# ---------- Agent ----------
class AgentRunReq(BaseModel):
    # Accept both `ra`/`dec` (legacy) and `ra_deg`/`dec_deg` (the names actually
    # used throughout the rest of the codebase) so that the frontend, CLI, and
    # docs don't have to remember which is the "blessed" form.
    target: str = Field(..., description="Target name or label, e.g. 'Gaia DR3 ...'")
    ra: Optional[float] = None
    dec: Optional[float] = None
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
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
    source_research_package: bool = False
    download_simbad_pdfs: bool = False
    enable_claude_code: bool = False
    claude_permission_mode: str = "plan"
    max_supervision_rounds: int = 2
    output_root: Optional[str] = None
    target_cluster: Optional[str] = Field(None, description="Cluster name from Hunt+2023 for membership chi^2")
    max_reflexion_retries: int = Field(2, description="How many drafter rewrites Reflexion can trigger")


@app.post("/api/agent/run")
def agent_run(req: AgentRunReq):
    safe = "".join(ch if ch.isalnum() or ch in "._+-" else "_" for ch in req.target).strip("_")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_root = req.output_root or str(RUNS_DIR / f"{safe or 'target'}_{stamp}")
    # Resolve coordinate aliases (ra_deg/dec_deg take precedence if set)
    ra_value = req.ra_deg if req.ra_deg is not None else req.ra
    dec_value = req.dec_deg if req.dec_deg is not None else req.dec
    state = run_workflow(
        target=req.target,
        ra_deg=ra_value,
        dec_deg=dec_value,
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
        source_research_package=req.source_research_package,
        download_simbad_pdfs=req.download_simbad_pdfs,
        enable_claude_code=req.enable_claude_code,
        claude_permission_mode=req.claude_permission_mode,
        max_supervision_rounds=req.max_supervision_rounds,
        target_cluster=req.target_cluster,
        max_reflexion_retries=req.max_reflexion_retries,
    )
    return {
        "output_root": out_root,
        "workflow_trace": build_trace(state, out_root),
        "qa": state.get("qa", {}),
        "resolved": state.get("resolved", {}),
        "artifacts": state.get("artifacts", []),
        "analysis_plan": state.get("analysis_plan", {}),
        "method_scout": state.get("method_scout", {}),
        "toolbox_gap": state.get("toolbox_gap", {}),
        "model_supervision": state.get("model_supervision", {}),
        "claude_code": state.get("claude_code", {}),
        "dynamic_skill_registration": state.get("dynamic_skill_registration", {}),
        "kg_graph_report": state.get("kg_graph_report", {}),
        "abnormal_report": state.get("abnormal_report", {}),
    }


@app.get("/api/agent/blueprint")
def agent_blueprint():
    return {"steps": workflow_blueprint()}


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


# ---------- KG Graph for Frontend ----------
KG_OUTPUT_DIR = KG_WORKSPACE / "output"
_KG_JSON_CACHE: Dict[str, Dict[str, Any]] = {}


def _safe_path_part(value: str, field_name: str) -> str:
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_name} 不能为空")
    if Path(value).is_absolute() or "/" in value or "\\" in value or value in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"{field_name} 不能包含路径分隔符: {value}")
    return value


def _read_kg_json(path: Path) -> Any:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    try:
        mtime = path.stat().st_mtime
        cache_key = str(path)
        cached = _KG_JSON_CACHE.get(cache_key)
        if cached and cached.get("mtime") == mtime:
            return cached.get("data")
        data = json.loads(path.read_text(encoding="utf-8"))
        _KG_JSON_CACHE[cache_key] = {"mtime": mtime, "data": data}
        return data
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"JSON 解析失败: {path}: {exc}")


def _resolve_kg_run_dir(dataset: str, timestamp: str = "") -> Path:
    dataset = _safe_path_part(dataset, "dataset")
    output_base = KG_OUTPUT_DIR / dataset
    if not output_base.is_dir():
        raise HTTPException(status_code=404, detail=f"数据集目录不存在: {output_base}")

    if timestamp:
        timestamp = _safe_path_part(timestamp, "timestamp")
        run_dir = output_base / timestamp
        if not run_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"运行目录不存在: {run_dir}")
        return run_dir

    preferred = output_base / "production_full"
    if preferred.is_dir():
        return preferred

    candidates = [
        path for path in output_base.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"未找到可用图谱运行目录: {output_base}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_kg_graph_file(dataset: str, timestamp: str = "", filename: str = "") -> Path:
    filename = filename or "multi_stage_deduplicated.json"
    filename = _safe_path_part(filename, "filename")
    run_dir = _resolve_kg_run_dir(dataset, timestamp)
    graph_path = run_dir / filename
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail=f"图谱文件不存在: {graph_path}")
    return graph_path

@app.get("/api/kg-summary/{dataset}")
def kg_summary(dataset: str, timestamp: str = "", filename: str = ""):
    """Return KG summary for frontend visualization."""
    try:
        graph_path = _resolve_kg_graph_file(dataset, timestamp, filename)
        run_dir = graph_path.parent
        summary = _read_kg_json(run_dir / "summary.json")
        if not isinstance(summary, dict):
            raise HTTPException(status_code=500, detail=f"summary.json 格式错误: {run_dir / 'summary.json'}")
        result = dict(summary)
        result["dataset_name"] = dataset
        result["timestamp"] = run_dir.name
        result["graph_file"] = graph_path.name
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取图谱摘要失败: {exc}")


@app.get("/api/source-profiles/{dataset}")
def source_profiles(dataset: str, timestamp: str = "", filename: str = "", feature: str = "", source: str = "", limit: int = 40):
    """Return source profiles for frontend visualization."""
    try:
        graph_path = _resolve_kg_graph_file(dataset, timestamp, filename)
        run_dir = graph_path.parent
        profiles_data = _read_kg_json(run_dir / "source_profiles.json")
        if not isinstance(profiles_data, dict):
            raise HTTPException(status_code=500, detail=f"source_profiles.json 格式错误: {run_dir / 'source_profiles.json'}")

        source_query = (source or "").strip().lower()
        safe_limit = max(1, min(int(limit or 40), 500))
        profiles: List[Dict[str, Any]] = []

        for source_name, profile in profiles_data.items():
            if not isinstance(profile, dict):
                continue
            if source_query and source_query not in str(source_name).lower():
                continue
            features = profile.get("features") if isinstance(profile.get("features"), dict) else {}
            if feature and feature not in features:
                continue
            feature_score = int(features.get(feature, 0)) if feature else sum(int(value or 0) for value in features.values())
            item = dict(profile)
            item["source"] = source_name
            item["feature_score"] = feature_score
            profiles.append(item)

        profiles.sort(key=lambda item: (item.get("feature_score", 0), str(item.get("source", ""))), reverse=True)
        return {
            "dataset_name": dataset,
            "timestamp": run_dir.name,
            "graph_file": graph_path.name,
            "count": len(profiles),
            "profiles": profiles[:safe_limit],
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"读取源画像失败: {exc}")


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


# ---------- AI CLI Tools ----------
class AiExecReq(BaseModel):
    tool: str = Field(..., description="kimi, claude, or codex")
    prompt: str = Field(..., description="Prompt to send to the AI")
    work_dir: Optional[str] = None
    timeout: int = 300


@app.post("/api/ai/exec")
def ai_exec(req: AiExecReq):
    """Execute AI CLI tools (kimi, claude, codex)."""
    tool_map = {
        "kimi": KIMI_BIN,
        "claude": CLAUDE_CODE_BIN,
        "codex": CODEX_BIN,
    }
    
    binary = tool_map.get(req.tool)
    if not binary:
        raise HTTPException(400, f"Unknown tool: {req.tool}")
    
    if not shutil.which(binary):
        raise HTTPException(400, f"{req.tool} not installed or not in PATH")
    
    work_dir = req.work_dir or str(ASTRO_AGENT_DIR)
    
    import subprocess
    try:
        if req.tool == "kimi":
            # Kimi CLI uses ACP (Agent Communication Protocol) 
            # The API key is for kimi-coding provider, not direct API
            # Use kimi CLI in acp mode or fall back to DeepSeek
            
            # Option 1: Try to use kimi CLI if available
            kimi_bin = shutil.which("kimi")
            if kimi_bin:
                # Try kimi chat command
                result = subprocess.run(
                    [kimi_bin, "chat", req.prompt],
                    capture_output=True,
                    text=True,
                    timeout=req.timeout,
                    cwd=work_dir
                )
                if result.returncode == 0:
                    return {
                        "tool": req.tool,
                        "returncode": 0,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "model": "kimi-cli",
                    }
            
            # Option 2: Fall back to DeepSeek API (same key works)
            import json as json_mod
            api_key = os.getenv("DEEPSEEK_API_KEY", "")
            base_url = "https://api.deepseek.com/v1"
            model = "deepseek-v4-pro"
            
            if not api_key:
                raise HTTPException(400, "No AI API key configured for Kimi fallback")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": req.prompt}],
                "temperature": 0.2,
                "max_tokens": 4096
            }
            
            import requests
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=req.timeout
            )
            
            if response.status_code != 200:
                raise HTTPException(500, f"Kimi fallback API error: {response.status_code} {response.text}")
            
            result_data = response.json()
            content = result_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return {
                "tool": req.tool,
                "returncode": 0,
                "stdout": content,
                "stderr": "(via DeepSeek fallback)",
                "model": model,
            }
        elif req.tool in ["claude", "codex"]:
            # Claude/Codex style CLI tools
            result = subprocess.run(
                [binary, "--cwd", work_dir, req.prompt],
                capture_output=True,
                text=True,
                timeout=req.timeout,
                cwd=work_dir
            )
            
            return {
                "tool": req.tool,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        else:
            raise HTTPException(400, f"Unknown tool execution path: {req.tool}")
        
    except subprocess.TimeoutExpired:
        raise HTTPException(504, f"{req.tool} execution timed out after {req.timeout}s")
    except Exception as exc:
        raise HTTPException(500, f"{req.tool} execution failed: {exc}")


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


@app.get("/api/runs/{name}/trace")
def get_run_trace(name: str):
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(404, "run not found")
    return {"run": name, "workflow_trace": build_trace_from_run(run_dir)}


# NOTE: the generic /api/runs/{name}/{filename} catch-all MUST come AFTER the
# specific routes below (/paper, /results, /figure/, /pdf, /rewrite), otherwise
# FastAPI matches it first and treats "paper" / "results" / etc. as filenames.


# ---------------------------------------------------------------------------
# Paper viewer endpoints (Round 4 frontend overhaul)
# ---------------------------------------------------------------------------

import re as _re


def _split_paper_sections(tex: str) -> List[Dict[str, Any]]:
    """Split a paper.tex into ordered sections.  Treats \\begin{abstract}…\\end
    as the Abstract section; \\section{X} starts a new section; \\acknowledgments
    ends the body."""
    sections: List[Dict[str, Any]] = []
    # Abstract block
    abs_m = _re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, flags=_re.DOTALL)
    if abs_m:
        sections.append({"name": "Abstract", "tex": abs_m.group(1).strip()})
    body_start = abs_m.end() if abs_m else 0
    body = tex[body_start:]
    body = _re.split(r"\\acknowledgments|\\bibliography", body, maxsplit=1)[0]
    # \section{X} ... \section{Y} ...
    sec_iter = list(_re.finditer(r"\\section\*?\{([^}]+)\}", body))
    for i, m in enumerate(sec_iter):
        name = m.group(1).strip()
        start = m.end()
        end = sec_iter[i + 1].start() if i + 1 < len(sec_iter) else len(body)
        sections.append({"name": name, "tex": body[start:end].strip()})
    return sections


def _list_run_figures(run_dir: Path) -> List[Dict[str, Any]]:
    """List PNG figures in paper_orchestra/figures/."""
    fig_dir = run_dir / "paper_orchestra" / "figures"
    if not fig_dir.exists():
        return []
    captions_path = fig_dir / "captions.json"
    captions = {}
    if captions_path.exists():
        try:
            captions = json.loads(captions_path.read_text())
        except Exception:
            captions = {}
    out = []
    for p in sorted(fig_dir.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
            continue
        key = p.stem.replace("fig_", "")
        out.append({
            "filename": p.name,
            "key": key,
            "caption": captions.get(key) or captions.get(p.stem) or "",
            "size_bytes": p.stat().st_size,
        })
    return out


def _load_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return _sanitize_json(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


import math as _math


def _sanitize_json(obj):
    """Recursively replace NaN/Inf with None so FastAPI can serialise.
    Pydantic/FastAPI's default encoder rejects non-finite floats."""
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    return obj


@app.get("/api/runs/{name}/paper")
def get_run_paper(name: str):
    """Return the drafted paper.tex split into sections + key auxiliary data
    for the paper viewer panel."""
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())) or not run_dir.exists():
        raise HTTPException(404, "run not found")
    tex_path = run_dir / "paper_orchestra" / "final" / "paper.tex"
    if not tex_path.exists():
        tex_path = run_dir / "paper_orchestra" / "drafts" / "paper.tex"
    if not tex_path.exists():
        raise HTTPException(404, "paper.tex not found in this run")
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    sections = _split_paper_sections(tex)
    # Title + authors from the preamble
    title_m = _re.search(r"\\title\{([^}]+)\}", tex)
    title = title_m.group(1) if title_m else name
    return {
        "run": name,
        "title": title,
        "tex_path": str(tex_path),
        "n_lines": tex.count("\n") + 1,
        "n_chars": len(tex),
        "sections": sections,
        "figures": _list_run_figures(run_dir),
        "paper_qc": _load_json_if_exists(run_dir / "09_paper_qc.json"),
        "novelty": _load_json_if_exists(run_dir / "02m_novelty.json"),
        "comparison_table": _load_json_if_exists(run_dir / "02n_comparison_table.json"),
        "physics_checks": _load_json_if_exists(run_dir / "02i_physics_checks.json"),
        "latex_compile": _load_json_if_exists(run_dir / "12_latex_compile.json"),
    }


@app.get("/api/runs/{name}/results")
def get_run_results(name: str):
    """Human-readable structured summary of the run.  Pulls highlights from
    the per-stage JSON artifacts and arranges them for the paper-viewer
    right column."""
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())) or not run_dir.exists():
        raise HTTPException(404, "run not found")

    def _load(p):
        return _load_json_if_exists(run_dir / p)

    resolved = _load("01_resolved_target.json") or {}
    plan = _load("02b_analysis_plan.json") or {}
    pp = _load("02c_published_params.json") or {}
    cluster = _load("02e_cluster_membership.json") or {}
    extinction = _load("02g_extinction.json") or {}
    sed = _load("02h_sed_decoupled.json") or {}
    geom = _load("02j_light_curve_geometry.json") or {}
    mcmc = _load("02k_eclipse_mcmc.json") or {}
    physics = _load("02i_physics_checks.json") or {}
    novelty = _load("02m_novelty.json") or {}
    qa = _load("08_qa_gate.json") or {}
    paper_qc = _load("09_paper_qc.json") or {}
    latex_compile_ = _load("12_latex_compile.json") or {}
    reflex = _load("09b_reflexion.json") or {}

    # Build a compact "headline numbers" table
    headlines: List[Dict[str, Any]] = []
    for r in (pp.get("rows") or []):
        param = r.get("parameter")
        val = r.get("value")
        if val is None:
            continue
        unit = r.get("unit") or ""
        err = r.get("error")
        kind = r.get("source_kind") or ""
        headlines.append({
            "parameter": param,
            "value": val,
            "error": err,
            "unit": unit,
            "kind": "literature" if kind == "simbad_abstract" else "this_work",
            "bibcode": r.get("bibcode"),
        })

    return {
        "run": name,
        "target": resolved.get("target"),
        "coords_deg": [resolved.get("ra_deg"), resolved.get("dec_deg")],
        "source_class": plan.get("source_class"),
        "fitting_pipeline": plan.get("fitting_pipeline_module"),
        "qa_gate": qa.get("apj_gate"),
        "model_mismatch": qa.get("model_mismatch"),
        "extinction_A_V": extinction.get("A_V"),
        "extinction_provenance": extinction.get("provenance"),
        "cluster_membership_best": (cluster.get("best_match") or {}).get("name")
            if cluster.get("status") == "ok" and cluster.get("candidates") else None,
        "cluster_candidates": cluster.get("candidates") or [],
        "sed_best_hypothesis": sed.get("best_hypothesis") or sed.get("best_hypothesis_joint"),
        "orbit": (geom.get("orbit") or {}),
        "eclipse_morphology": geom.get("morphology"),
        "eclipse_tau_over_P": geom.get("tau_over_P"),
        "ingress_days": geom.get("t_ingress_days"),
        "mcmc": {
            "status": mcmc.get("status"),
            "backend": mcmc.get("backend"),
            "e_pct": mcmc.get("e_pct"),
            "alpha_deg_pct": mcmc.get("alpha_deg_pct"),
            "omega_deg_pct": mcmc.get("omega_deg_pct"),
        },
        "physics_arguments_n": len((physics.get("sections") or [])),
        "novelty_summary": {
            "n_items": novelty.get("n_items"),
            "verdict_counts": novelty.get("verdict_counts"),
        },
        "paper_qc": {
            "verdict": paper_qc.get("verdict"),
            "summary": paper_qc.get("summary"),
            "n_pass": paper_qc.get("n_pass"),
            "n_warn": paper_qc.get("n_warn"),
            "n_fail": paper_qc.get("n_fail"),
            "checks": paper_qc.get("checks") or [],
        },
        "latex_compile": {
            "status": latex_compile_.get("status"),
            "pdf_path": latex_compile_.get("pdf_path"),
        },
        "reflexion_last_decision": reflex.get("routing_decision") or reflex.get("status"),
        "headline_numbers": headlines,
    }


@app.get("/api/runs/{name}/figure/{filename}")
def get_run_figure(name: str, filename: str):
    """Serve a PNG/PDF figure file from paper_orchestra/figures/."""
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    candidate = (run_dir / "paper_orchestra" / "figures" / filename).resolve()
    if not str(candidate).startswith(str(run_dir)) or not candidate.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(str(candidate))


@app.get("/api/runs/{name}/pdf")
def get_run_pdf(name: str):
    """Serve the compiled PDF if latex_compile succeeded."""
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    pdf_path = run_dir / "paper_orchestra" / "build" / "paper.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "compiled PDF not found (run latex_compile first)")
    return FileResponse(str(pdf_path), media_type="application/pdf")


@app.get("/api/runs/{name}/tex")
def get_run_tex(name: str):
    """Serve the final paper.tex (text/plain so browser previews it)."""
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    tex_path = run_dir / "paper_orchestra" / "final" / "paper.tex"
    if not tex_path.exists():
        tex_path = run_dir / "paper_orchestra" / "drafts" / "paper.tex"
    if not tex_path.exists():
        raise HTTPException(404, "paper.tex not found")
    return FileResponse(str(tex_path), media_type="text/plain; charset=utf-8")


class RewriteSectionReq(BaseModel):
    section: str  # one of: Abstract, Introduction, Data, Methods, Results, Discussion, Conclusions
    use_llm: bool = True
    llm_provider: Optional[str] = None
    additional_instructions: Optional[str] = None


@app.post("/api/runs/{name}/rewrite")
def rewrite_section(name: str, req: RewriteSectionReq):
    """Force-rewrite a single section of an existing run.  Synthesises a
    targeted Reflexion-style instruction that the section drafter will pick
    up on the next pass, then re-runs only the drafter + paper_qc + reflexion
    chain.  This is *not* a full workflow re-run."""
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())) or not run_dir.exists():
        raise HTTPException(404, "run not found")
    # Compose the targeted reflection in the same format as analysis_agent.reflexion
    advice = req.additional_instructions or (
        f"Rewrite the {req.section} section. "
        "Use the published_params table, hypothesis_plan, comparison_table, "
        "and any physics_checks LaTeX as ground truth. Keep length appropriate "
        "for an ApJ Letter (Abstract 180-280 words; Methods/Results 500-1200 words)."
    )
    reflection_path = run_dir / "09b_reflexion.json"
    refl = {
        "status": "ok",
        "user_triggered_rewrite": True,
        "sections_to_rewrite": [req.section],
        "section_to_failing_checks": {req.section: ["user_request"]},
        "action_items": [{
            "check_id": "user_request",
            "verdict": "user",
            "section": req.section,
            "reason": "user requested targeted rewrite via UI",
            "advice": advice,
        }],
        "verbal_reflection": (
            f"User-requested rewrite of the {req.section} section.\n"
            f"Specific advice: {advice}"
        ),
        "routing_decision": "rewrite",
    }
    reflection_path.write_text(json.dumps(refl, ensure_ascii=False, indent=2), encoding="utf-8")
    # We can't easily re-trigger only the drafter inside the LangGraph; the
    # cheapest path is to ask the user to re-run with the existing astrotool
    # and same output_root.  Return a structured response telling the UI
    # exactly how to do it.
    return {
        "status": "reflection_recorded",
        "section": req.section,
        "reflection_path": str(reflection_path),
        "note": (
            "Re-run the workflow with the same `output_root` to apply the "
            "rewrite (the drafter will pick up the new reflection from disk). "
            "POST /api/agent/run with the same output_root + use_llm=true."
        ),
        "suggested_run_payload": {
            "output_root": str(run_dir),
            "use_llm": req.use_llm,
            "llm_provider": req.llm_provider,
            "max_reflexion_retries": 3,
            "draft_on_hold": True,
        },
    }


# --------------------------------------------------------------------------- #
# Ablation dashboard endpoints (PR4 / D2)                                      #
# --------------------------------------------------------------------------- #

ABLATION_DIR = ASTRO_AGENT_DIR / "scripts" / "ablation"
ABLATION_CSV = ABLATION_DIR / "ablation_results.csv"
ABLATION_MATRIX_YAML = ABLATION_DIR / "ablation_matrix.yaml"


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    import csv as _csv
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            out.append(row)
    return out


@app.get("/api/ablation/matrix")
def ablation_matrix():
    """Return the planned (config_id, target_id) cells from the YAML matrix
    so the frontend can render an empty heatmap before any cells are run."""
    if not ABLATION_MATRIX_YAML.exists():
        raise HTTPException(404, "ablation_matrix.yaml missing")
    try:
        import yaml as _yaml
    except Exception:
        raise HTTPException(500, "PyYAML not installed")
    m = _yaml.safe_load(ABLATION_MATRIX_YAML.read_text(encoding="utf-8"))
    return {
        "configs": [{"id": c["id"], "description": c.get("description", ""),
                     "flags": c.get("flags", {})} for c in m.get("configs", [])],
        "targets": [{"id": t["id"], "target": t.get("target"), "short": t.get("short")}
                    for t in m.get("targets", [])],
        "defaults": m.get("defaults", {}),
    }


@app.get("/api/ablation/results")
def ablation_results():
    return {"rows": _read_csv_rows(ABLATION_CSV), "csv_path": str(ABLATION_CSV)}


@app.get("/api/ablation/leaderboard")
def ablation_leaderboard():
    rows = _read_csv_rows(ABLATION_CSV)
    def _f(r, k):
        try:
            return float(r.get(k, 0) or 0)
        except Exception:
            return 0.0
    rows = [r for r in rows if r.get("qc_verdict") not in ("", "dry-run", "missing")]
    rows.sort(key=lambda r: _f(r, "composite"), reverse=True)
    return {"top": rows[:10]}


class AblationRunReq(BaseModel):
    config_id: str
    target_id: str
    dry_run: bool = False


@app.post("/api/ablation/run")
def ablation_run(req: AblationRunReq):
    """Launch one ablation cell as a subprocess. Returns immediately with
    the spawned PID; results are appended to ablation_results.csv when the
    workflow completes."""
    if not (ABLATION_DIR / "run_matrix.py").exists():
        raise HTTPException(500, "run_matrix.py missing")
    import subprocess as _sp
    cmd = [
        sys.executable, str(ABLATION_DIR / "run_matrix.py"),
        "--rows", req.config_id, "--targets", req.target_id,
    ]
    if req.dry_run:
        cmd.append("--dry-run")
    proc = _sp.Popen(cmd, cwd=str(ASTRO_AGENT_DIR),
                     stdout=_sp.PIPE, stderr=_sp.PIPE)
    return {"pid": proc.pid, "cmd": " ".join(cmd), "config_id": req.config_id,
            "target_id": req.target_id, "dry_run": req.dry_run}


@app.get("/api/ablation/diff")
def ablation_diff(a: str, b: str):
    """Return a side-by-side comparison of two runs' paper.tex bodies."""
    def _read_tex(name: str) -> str:
        run_dir = (RUNS_DIR / name).resolve()
        if not str(run_dir).startswith(str(RUNS_DIR.resolve())):
            raise HTTPException(403, "path traversal")
        for p in (run_dir / "paper_orchestra" / "final" / "paper.tex",
                  run_dir / "paper_orchestra" / "drafts" / "paper.tex"):
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
        return ""
    import difflib as _diff
    a_tex = _read_tex(a).splitlines()
    b_tex = _read_tex(b).splitlines()
    delta = list(_diff.unified_diff(a_tex, b_tex, fromfile=a, tofile=b, n=2, lineterm=""))
    return {"a": a, "b": b, "n_lines_a": len(a_tex), "n_lines_b": len(b_tex),
            "unified_diff": "\n".join(delta[:4000])}


# --------------------------------------------------------------------------- #
# Prompt Lab endpoints (PR4 / D3)                                              #
# --------------------------------------------------------------------------- #

PROMPTS_DIR = PACKAGE_DIR / "prompts"
OVERRIDES_DIR = PROMPTS_DIR / "overrides"
PROMPT_EXP_DB = ASTRO_AGENT_DIR / "output" / "analysis_agent" / "_prompt_experiments.sqlite"


@app.get("/api/prompts/list")
def prompts_list():
    """Return the live system prompt per role + every section prompt,
    plus whether an override file shadows each one."""
    from analysis_agent.prompts import wd_domain as wd
    items: List[Dict[str, Any]] = []
    for role in ("physicist", "writer", "critic", "outline", "reviewer", "retrieval"):
        name = f"system_{role}"
        override = (OVERRIDES_DIR / f"{name}.txt").exists()
        items.append({
            "name": name, "kind": "system_role", "role": role,
            "has_override": override,
            "content": wd.system_for_role(role),
        })
    for sec in ("Abstract", "Introduction", "Data", "Methods",
                "Results", "Discussion", "Conclusions"):
        name = f"section_{sec.lower()}"
        override = (OVERRIDES_DIR / f"{name}.txt").exists()
        items.append({
            "name": name, "kind": "section", "section": sec,
            "has_override": override,
            "content": wd.section_prompt(sec),
        })
    return {"items": items, "overrides_dir": str(OVERRIDES_DIR)}


class PromptSaveReq(BaseModel):
    name: str = Field(..., description="e.g. system_physicist or section_methods")
    content: str = Field(..., description="Plain text override; empty string deletes the override")


@app.post("/api/prompts/save")
def prompts_save(req: PromptSaveReq):
    if "/" in req.name or "\\" in req.name or req.name.startswith("."):
        raise HTTPException(400, "invalid override name")
    OVERRIDES_DIR.mkdir(parents=True, exist_ok=True)
    p = OVERRIDES_DIR / f"{req.name}.txt"
    if not req.content.strip():
        if p.exists():
            p.unlink()
        return {"status": "deleted", "name": req.name}
    p.write_text(req.content, encoding="utf-8")
    return {"status": "written", "name": req.name, "path": str(p), "bytes": p.stat().st_size}


@app.get("/api/prompts/experiments")
def prompts_experiments():
    """Aggregate _prompt_experiments.sqlite by (specialist, section)."""
    if not PROMPT_EXP_DB.exists():
        return {"status": "empty", "by_specialist_section": [], "total": 0}
    import sqlite3 as _sq
    conn = _sq.connect(str(PROMPT_EXP_DB))
    conn.row_factory = _sq.Row
    try:
        rows = conn.execute(
            "SELECT specialist, section, COUNT(*) AS n, "
            "AVG(paper_qc_pass) AS avg_pass, AVG(paper_qc_fail) AS avg_fail, "
            "AVG(output_chars) AS avg_chars "
            "FROM prompt_runs GROUP BY specialist, section ORDER BY n DESC"
        ).fetchall()
        recent = conn.execute(
            "SELECT timestamp, specialist, section, paper_qc_pass, paper_qc_fail "
            "FROM prompt_runs ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) AS n FROM prompt_runs").fetchone()
    finally:
        conn.close()
    return {
        "status": "ok",
        "by_specialist_section": [dict(r) for r in rows],
        "recent": [dict(r) for r in recent],
        "total": int(total["n"] if total else 0),
    }


# --------------------------------------------------------------------------- #
# Reviewer pass endpoint (PR4 / D4)                                            #
# --------------------------------------------------------------------------- #

class ReviewerPassReq(BaseModel):
    provider: Optional[str] = None


@app.post("/api/runs/{name}/reviewer-pass")
def reviewer_pass(name: str, req: ReviewerPassReq):
    run_dir = (RUNS_DIR / name).resolve()
    if not str(run_dir).startswith(str(RUNS_DIR.resolve())) or not run_dir.exists():
        raise HTTPException(404, "run not found")
    tex_path = run_dir / "paper_orchestra" / "final" / "paper.tex"
    if not tex_path.exists():
        tex_path = run_dir / "paper_orchestra" / "drafts" / "paper.tex"
    if not tex_path.exists():
        raise HTTPException(404, "paper.tex not found")
    tex = tex_path.read_text(encoding="utf-8", errors="replace")
    # Build a minimal state for llm_review (it only reads anti-leakage block)
    from analysis_agent.paper_orchestra import llm_review
    state: Dict[str, Any] = {}
    try:
        result = llm_review(state, tex, provider=req.provider)
    except Exception as exc:
        raise HTTPException(500, f"reviewer call failed: {type(exc).__name__}: {exc}")
    out_path = run_dir / "09c_reviewer.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"result": result, "saved_to": str(out_path)}


# Catch-all artifact route — kept LAST so the specific /paper /results /figure
# /pdf /rewrite routes above match first.
@app.get("/api/runs/{name}/{filename}")
def get_run_artifact(name: str, filename: str):
    target = (RUNS_DIR / name / filename).resolve()
    if not str(target).startswith(str(RUNS_DIR.resolve())):
        raise HTTPException(403, "path traversal blocked")
    if not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(target))
