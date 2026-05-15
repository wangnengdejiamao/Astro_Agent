# 读取 graph_path 中的图谱数据并显示在页面上  filename需要遵守命名规则
# "graph_by_file_s_{schema_name}_p_{prompt_name}.json"
# "graph_by_content_{time_now}_session_{session_id}.json"
import json
import os
import sys
import glob
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import networkx as nx

# 添加当前目录到路径，确保可以导入graph_tools
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dotenv import load_dotenv

ENV_PATH = Path(current_dir) / ".env"
load_dotenv(ENV_PATH)


def _normalize_env_path(name: str) -> None:
    value = os.getenv(name)
    if not value:
        return
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(current_dir) / path
    os.environ[name] = str(path.resolve())


for _env_path_name in (
    "PROJECT_DIR",
    "PROJECT_ROOT",
    "OUTPUT_DIR",
    "GRAPHRAG_DIR",
    "GRAPH_INDEX_CACHE_DIR",
    "GRAPH_RETRIEVER_CACHE_DIR",
):
    _normalize_env_path(_env_path_name)

from graph_tools.graph_util import GraphAnalyzer
from graph_tools.retriever import GraphRetriever

# 图谱数据目录
OUTPUT_DIR = os.getenv("OUTPUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# 项目根目录（与 vis_graph_v1 一致，供 staged examples 等路径解析）
PROJECT_DIR = os.getenv("PROJECT_DIR") or os.path.dirname(os.path.abspath(__file__))


app = FastAPI(title="图谱可视化系统", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 缓存GraphAnalyzer实例
_analyzer_cache: Dict[str, GraphAnalyzer] = {}
_json_file_cache: Dict[str, Dict[str, Any]] = {}

# GraphRetriever实例用于获取chunk内容
graph_retriever = GraphRetriever()


def _safe_path_part(value: str, field_name: str) -> str:
    """Validate a single path segment received from the browser."""
    if not value:
        raise HTTPException(status_code=400, detail=f"{field_name} 不能为空")
    if os.path.isabs(value) or "/" in value or "\\" in value or value in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"{field_name} 不能包含路径分隔符: {value}")
    return value


def _read_json_file(path: str) -> Any:
    """Read a JSON file with a small mtime cache for large source profile files."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"文件不存在: {path}")
    try:
        mtime = os.path.getmtime(path)
        cached = _json_file_cache.get(path)
        if cached and cached.get("mtime") == mtime:
            return cached.get("data")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _json_file_cache[path] = {"mtime": mtime, "data": data}
        return data
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 解析失败: {path}: {str(e)}")


def _resolve_pipeline_run_dir(dataset_name: str, timestamp: Optional[str] = None) -> str:
    """Resolve output/{dataset}/{run} for pipeline-style KG files."""
    dataset_name = _safe_path_part(dataset_name, "dataset_name")
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    if not os.path.isdir(dataset_dir):
        raise HTTPException(status_code=404, detail=f"数据集目录不存在: {dataset_dir}")

    if timestamp:
        timestamp = _safe_path_part(timestamp, "timestamp")
        run_dir = os.path.join(dataset_dir, timestamp)
        if not os.path.isdir(run_dir):
            raise HTTPException(status_code=404, detail=f"运行目录不存在: {run_dir}")
        return run_dir

    preferred = os.path.join(dataset_dir, "production_full")
    if os.path.isdir(preferred):
        return preferred

    candidates = [
        os.path.join(dataset_dir, item)
        for item in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, item))
        and os.path.exists(os.path.join(dataset_dir, item, "summary.json"))
    ]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"未找到可用的图谱运行目录: {dataset_dir}")
    return max(candidates, key=os.path.getmtime)


def _resolve_pipeline_file(dataset_name: str, timestamp: Optional[str], filename: str) -> str:
    filename = _safe_path_part(filename, "filename")
    run_dir = _resolve_pipeline_run_dir(dataset_name, timestamp)
    file_path = os.path.join(run_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"图谱文件不存在: {file_path}")
    return file_path


def get_chunk_path(dataset_name: str) -> str:
    """根据 dataset_name 获取 chunk 文件路径"""
    chunk_path = os.path.join(OUTPUT_DIR, dataset_name, "chunks.txt")
    return chunk_path


def find_graph_file(dataset_name: str, graph_type: Optional[str] = None, 
                    schema_name: Optional[str] = None, prompt_name: Optional[str] = None,
                    session_id: Optional[str] = None) -> Optional[str]:
    """
    根据dataset_name查找对应的图谱JSON文件
    
    Args:
        dataset_name: 数据集名称（如 "paper_mini"）
        graph_type: 图谱类型，"file" 或 "content"，如果为None则优先查找file类型
        schema_name: schema名称（用于file类型）
        prompt_name: prompt名称（用于file类型）
        session_id: session ID（用于content类型）
    
    Returns:
        找到的文件路径，如果未找到则返回None
    """
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        return None
    
    # 查找所有JSON文件
    json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
    
    if graph_type == "file" or (graph_type is None and schema_name and prompt_name):
        # 查找 graph_by_file_s_{schema_name}_p_{prompt_name}.json
        pattern = f"graph_by_file_s_{schema_name}_p_{prompt_name}.json"
        for file_path in json_files:
            if os.path.basename(file_path) == pattern:
                return file_path
    elif graph_type == "content" or (graph_type is None and session_id):
        # 查找 graph_by_content_{time_now}_session_{session_id}.json
        pattern = f"graph_by_content_.*_session_{session_id}.json"
        for file_path in json_files:
            if re.match(pattern, os.path.basename(file_path)):
                return file_path
    
    # 如果没有指定类型，优先返回file类型，否则返回第一个找到的文件
    if graph_type is None:
        # 优先查找file类型
        for file_path in json_files:
            if "graph_by_file" in os.path.basename(file_path):
                return file_path
        # 如果没有file类型，返回第一个content类型
        for file_path in json_files:
            if "graph_by_content" in os.path.basename(file_path):
                return file_path
        # 如果都没有，返回第一个JSON文件
        if json_files:
            return json_files[0]
    
    return None


def get_analyzer(dataset_name: str, graph_path: Optional[str] = None) -> GraphAnalyzer:
    """
    获取或创建GraphAnalyzer实例（带缓存）
    """
    cache_key = f"{dataset_name}:{graph_path}"
    if cache_key not in _analyzer_cache:
        if graph_path is None:
            graph_path = find_graph_file(dataset_name)
        if graph_path is None or not os.path.exists(graph_path):
            raise HTTPException(
                status_code=404,
                detail=f"未找到数据集 {dataset_name} 的图谱文件"
            )
        _analyzer_cache[cache_key] = GraphAnalyzer(graph_path)
    return _analyzer_cache[cache_key]


@app.get("/api/graph/{dataset_name}")     # 请你修改前端，输入的session_id 是当前session_id，如果选择 graph_type = "file"，则直接展示图谱文件名称，如果选择 content，则传入当前session_id用于筛选，只有匹配上的session能被看到
async def get_full_graph(
    dataset_name: str,
    graph_type: Optional[str] = None,
    schema_name: Optional[str] = None,
    prompt_name: Optional[str] = None,
    session_id: Optional[str] = None,
    threshold_high: int = 2000,
    threshold_medium: int = 500
):
    """
    获取完整图的可视化数据
    
    如果选择 graph_type = "file"，则需要提供 schema_name 和 prompt_name 参数，直接展示对应的图谱文件
    如果选择 graph_type = "content"，则需要提供 session_id 参数，用于筛选对应的图谱文件
    
    Args:
        dataset_name: 数据集名称
        graph_type: 图谱类型，"file" 或 "content"
        schema_name: schema名称（用于file类型，必需）
        prompt_name: prompt名称（用于file类型，必需）
        session_id: session ID（用于content类型，必需）
        threshold_high: 高阈值
        threshold_medium: 中阈值
    
    Returns:
        图谱的可视化数据（ECharts格式）
    """
    try:
        # 参数验证
        if graph_type == "file":
            if not schema_name or not prompt_name:
                raise HTTPException(
                    status_code=400,
                    detail="当 graph_type='file' 时，必须提供 schema_name 和 prompt_name 参数"
                )
        elif graph_type == "content":
            if not session_id:
                raise HTTPException(
                    status_code=400,
                    detail="当 graph_type='content' 时，必须提供 session_id 参数"
                )
        
        # 查找图谱文件
        graph_path = find_graph_file(dataset_name, graph_type, schema_name, prompt_name, session_id)
        if graph_path is None:
            # 生成更详细的错误信息
            if graph_type == "file":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (schema: {schema_name}, prompt: {prompt_name})"
            elif graph_type == "content":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (session_id: {session_id})"
            else:
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件"
            raise HTTPException(status_code=404, detail=error_msg)
        
        # 获取分析器并转换数据
        analyzer = get_analyzer(dataset_name, graph_path)
        graph_data = analyzer.convert_to_echarts_format(
            max_nodes=1000,
            threshold_high=threshold_high,
            threshold_medium=threshold_medium
        )
        
        # 添加文件信息到返回数据中
        graph_data["graph_file"] = os.path.basename(graph_path)
        graph_data["graph_path"] = graph_path
        
        return graph_data
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取完整图失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/graph/{dataset_name}/download")
async def download_graph_file(
    dataset_name: str,
    graph_type: Optional[str] = None,
    schema_name: Optional[str] = None,
    prompt_name: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    下载原始图谱 JSON 文件
    
    Args:
        dataset_name: 数据集名称
        graph_type: 图谱类型，"file" 或 "content"
        schema_name: schema名称（用于file类型）
        prompt_name: prompt名称（用于file类型）
        session_id: session ID（用于content类型）
    
    Returns:
        原始图谱 JSON 文件
    """
    try:
        # 参数验证
        if graph_type == "file":
            if not schema_name or not prompt_name:
                raise HTTPException(
                    status_code=400,
                    detail="当 graph_type='file' 时，必须提供 schema_name 和 prompt_name 参数"
                )
        elif graph_type == "content":
            if not session_id:
                raise HTTPException(
                    status_code=400,
                    detail="当 graph_type='content' 时，必须提供 session_id 参数"
                )
        
        # 查找图谱文件
        graph_path = find_graph_file(dataset_name, graph_type, schema_name, prompt_name, session_id)
        if graph_path is None:
            if graph_type == "file":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (schema: {schema_name}, prompt: {prompt_name})"
            elif graph_type == "content":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (session_id: {session_id})"
            else:
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件"
            raise HTTPException(status_code=404, detail=error_msg)
        
        # 检查文件是否存在
        if not os.path.exists(graph_path):
            raise HTTPException(status_code=404, detail=f"图谱文件不存在: {graph_path}")
        
        # 返回文件
        return FileResponse(
            path=graph_path,
            filename=os.path.basename(graph_path),
            media_type='application/json'
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"下载图谱文件失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SubgraphRequest(BaseModel):
    dataset_name: str
    node_id: Optional[str] = None
    node_name: Optional[str] = None
    depth: int = 1
    threshold_high: Optional[int] = 2000
    threshold_medium: Optional[int] = 500
    graph_type: Optional[str] = None
    schema_name: Optional[str] = None
    prompt_name: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/api/subgraph")
async def get_subgraph(request: SubgraphRequest):
    """
    获取子图的可视化数据
    """
    try:
        graph_path = find_graph_file(
            request.dataset_name, 
            request.graph_type, 
            request.schema_name, 
            request.prompt_name, 
            request.session_id
        )
        if graph_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"未找到数据集 {request.dataset_name} 的图谱文件"
            )
        
        analyzer = get_analyzer(request.dataset_name, graph_path)
        
        if request.node_id:
            subgraph = analyzer.find_subgraph_by_node_id(request.node_id, depth=request.depth)
        elif request.node_name:
            subgraph = analyzer.find_subgraph_by_node_name(request.node_name, depth=request.depth)
        else:
            raise HTTPException(status_code=400, detail="必须提供node_id或node_name")
        
        if subgraph.number_of_nodes() == 0:
            error_msg = f"未找到子图。节点ID: {request.node_id}, 节点名称: {request.node_name}"
            raise HTTPException(status_code=404, detail=error_msg)
        
        threshold_high = request.threshold_high if request.threshold_high is not None else 2000
        threshold_medium = request.threshold_medium if request.threshold_medium is not None else 500
        
        graph_data = analyzer.convert_to_echarts_format(
            subgraph,
            max_nodes=1000,
            threshold_high=threshold_high,
            threshold_medium=threshold_medium,
            force_subgraph_mode=True
        )
        return graph_data
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取子图失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/node/{dataset_name}/{node_id}")
async def get_node_info(
    dataset_name: str,
    node_id: str,
    graph_type: Optional[str] = None,
    schema_name: Optional[str] = None,
    prompt_name: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    获取节点详细信息
    """
    try:
        graph_path = find_graph_file(
            dataset_name,
            graph_type,
            schema_name,
            prompt_name,
            session_id
        )
        if graph_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"未找到数据集 {dataset_name} 的图谱文件"
            )
        
        analyzer = get_analyzer(dataset_name, graph_path)
        
        if not analyzer.graph or node_id not in analyzer.graph:
            raise HTTPException(status_code=404, detail="节点未找到")
        
        node_data = analyzer.graph.nodes[node_id]
        props = node_data.get("properties", {}).copy()
        
        # 处理名称字段（可能是列表）
        name = props.get("name", node_id)
        if isinstance(name, list):
            name = "\n".join(str(n) for n in name) if name else str(node_id)
        elif not isinstance(name, str):
            name = str(name)
        props["name"] = name
        
        # 获取邻居节点
        neighbors = list(analyzer.graph.neighbors(node_id))
        if analyzer.graph.is_directed():
            predecessors = list(analyzer.graph.predecessors(node_id))
            successors = list(analyzer.graph.successors(node_id))
        else:
            predecessors = []
            successors = neighbors

        def _get_node_name(nid: str) -> str:
            n_data = analyzer.graph.nodes.get(nid, {})
            n_props = n_data.get("properties", {}) if isinstance(n_data, dict) else {}
            n_name = n_props.get("name", nid)
            if isinstance(n_name, list):
                return "\n".join(str(n) for n in n_name) if n_name else str(nid)
            if not isinstance(n_name, str):
                return str(n_name)
            return n_name

        def _to_float(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None

        incident_edges: List[Dict[str, Any]] = []
        seen_edge_keys: set[str] = set()
        max_incident_edges = 30

        def _maybe_add_edge(u: str, v: str, data: Dict[str, Any], key: Optional[str] = None) -> None:
            relation = data.get("relation", "related_to")
            evidence = data.get("evidence", "") or ""
            source_text = data.get("source", "") or ""

            edge_uid = f"{u}->{v}:{key}:{relation}"
            if edge_uid in seen_edge_keys:
                return
            seen_edge_keys.add(edge_uid)

            sort_score = (
                _to_float(data.get("triple_support_score"))
                or _to_float(data.get("node_accuracy_score"))
                or _to_float(data.get("start_accuracy_score"))
                or _to_float(data.get("end_accuracy_score"))
                or -1.0
            )

            edge_item: Dict[str, Any] = {
                "source": u,
                "target": v,
                "name": relation,
                "source_name": _get_node_name(u),
                "target_name": _get_node_name(v),
                "evidence": evidence,
                "source_text": source_text,
                "chunk_id": data.get("chunk_id"),
                "_sort_score": sort_score,
            }

            # 常用推理/打分字段：前端做细粒度展示用
            for k in [
                "node_accuracy_score",
                "node_reason",
                "triple_support_score",
                "triple_reason",
                "start_accuracy_score",
                "end_accuracy_score",
                "start_reason",
                "end_reason",
                "accuracy_score",
                "usefulness_score",
                "accuracy_reasoning",
                "usefulness_reasoning",
            ]:
                if k in data:
                    edge_item[k] = data.get(k)

            incident_edges.append(edge_item)

        is_multi = bool(getattr(analyzer.graph, "is_multigraph", lambda: False)())
        if is_multi:
            for u, v, k, data in analyzer.graph.in_edges(node_id, keys=True, data=True):
                _maybe_add_edge(u, v, data, key=str(k))
            for u, v, k, data in analyzer.graph.out_edges(node_id, keys=True, data=True):
                _maybe_add_edge(u, v, data, key=str(k))
        else:
            for u, v, data in analyzer.graph.in_edges(node_id, data=True):
                _maybe_add_edge(u, v, data, key=None)
            for u, v, data in analyzer.graph.out_edges(node_id, data=True):
                _maybe_add_edge(u, v, data, key=None)

        incident_edges.sort(key=lambda e: e.get("_sort_score", -1.0), reverse=True)
        incident_edges = incident_edges[:max_incident_edges]
        for e in incident_edges:
            e.pop("_sort_score", None)

        return {
            "id": node_id,
            "label": node_data.get("label", "entity"),
            "properties": props,
            "degree": analyzer.graph.degree(node_id),
            "neighbors_count": len(neighbors),
            "predecessors_count": len(predecessors),
            "successors_count": len(successors),
            "edges": incident_edges,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取节点信息失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class ChunkRequest(BaseModel):
    dataset_name: str
    chunk_id: str


@app.post("/api/chunk")
async def get_chunk_text(request: ChunkRequest):
    """根据chunk id获取原文"""
    try:
        chunk_path = get_chunk_path(request.dataset_name)
        if not os.path.exists(chunk_path):
            raise HTTPException(
                status_code=404,
                detail=f"未找到chunk文件: {chunk_path}"
            )
        chunk_data = graph_retriever.retrieve_chunk_and_title_by_id(
            request.chunk_id, chunk_path
        )
        if not chunk_data or (not chunk_data.get("title") and not chunk_data.get("text")):
            raise HTTPException(
                status_code=404,
                detail=f"未找到 chunk: {request.chunk_id}",
            )
        return {
            "chunk_id": request.chunk_id,
            "title": chunk_data.get("title", ""),
            "text": chunk_data.get("text", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取chunk失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/datasets")
async def list_datasets():
    """
    列出所有可用的数据集
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            return {"datasets": []}
        
        datasets = []
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                # 查找该数据集下的所有图谱文件
                json_files = glob.glob(os.path.join(item_path, "*.json"))
                graph_files = []
                for json_file in json_files:
                    filename = os.path.basename(json_file)
                    if "graph_by_file" in filename:
                        # 解析 graph_by_file_s_{schema_name}_p_{prompt_name}.json
                        match = re.match(r"graph_by_file_s_(.+?)_p_(.+?)\.json", filename)
                        if match:
                            graph_files.append({
                                "type": "file",
                                "filename": filename,
                                "schema_name": match.group(1),
                                "prompt_name": match.group(2),
                                "path": json_file
                            })
                    elif "graph_by_content" in filename:
                        # 解析 graph_by_content_{time_now}_session_{session_id}.json
                        match = re.match(r"graph_by_content_(.+?)_session_(.+?)\.json", filename)
                        if match:
                            graph_files.append({
                                "type": "content",
                                "filename": filename,
                                "time": match.group(1),
                                "session_id": match.group(2),
                                "path": json_file
                            })
                
                if graph_files:
                    datasets.append({
                        "name": item,
                        "graph_files": graph_files
                    })
        
        return {"datasets": datasets}
    except Exception as e:
        import traceback
        error_detail = f"列出数据集失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/kg-summary/{dataset_name}")
async def get_kg_summary(
    dataset_name: str,
    timestamp: Optional[str] = None,
    filename: str = "multi_stage_deduplicated.json",
):
    """Return the source-feature KG summary used by the white dwarf frontend."""
    try:
        graph_path = _resolve_pipeline_file(dataset_name, timestamp, filename)
        run_dir = os.path.dirname(graph_path)
        summary_path = os.path.join(run_dir, "summary.json")
        summary = _read_json_file(summary_path)
        if not isinstance(summary, dict):
            raise HTTPException(status_code=500, detail=f"summary.json 格式错误: {summary_path}")
        result = dict(summary)
        result["dataset_name"] = dataset_name
        result["timestamp"] = os.path.basename(run_dir)
        result["graph_file"] = os.path.basename(graph_path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"读取图谱摘要失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/source-profiles/{dataset_name}")
async def get_source_profiles(
    dataset_name: str,
    timestamp: Optional[str] = None,
    filename: str = "multi_stage_deduplicated.json",
    feature: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 40,
):
    """Return filtered source profiles for the canvas/profile frontend."""
    try:
        graph_path = _resolve_pipeline_file(dataset_name, timestamp, filename)
        run_dir = os.path.dirname(graph_path)
        profiles_path = os.path.join(run_dir, "source_profiles.json")
        profiles_data = _read_json_file(profiles_path)
        if not isinstance(profiles_data, dict):
            raise HTTPException(status_code=500, detail=f"source_profiles.json 格式错误: {profiles_path}")

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

            feature_score = int(features.get(feature, 0)) if feature else sum(int(v or 0) for v in features.values())
            item = dict(profile)
            item["source"] = source_name
            item["feature_score"] = feature_score
            profiles.append(item)

        profiles.sort(key=lambda item: (item.get("feature_score", 0), str(item.get("source", ""))), reverse=True)

        return {
            "dataset_name": dataset_name,
            "timestamp": os.path.basename(run_dir),
            "graph_file": os.path.basename(graph_path),
            "count": len(profiles),
            "profiles": profiles[:safe_limit],
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"读取源画像失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


# 静态文件服务（前端页面）
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    @app.get("/")
    async def read_root():
        """返回前端首页"""
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            return {"message": "前端页面未找到，请确保 frontend/index.html 存在"}
    
    @app.get("/index.html")
    async def read_index():
        """返回图谱可视化页面"""
        return FileResponse(os.path.join(frontend_dir, "index.html"))
    
    @app.get("/manage.html")
    async def read_manage():
        """返回图谱管理页面（包含Schema/Prompt管理、Corpus管理、生成图谱）"""
        file_path = os.path.join(frontend_dir, "manage.html")
        if os.path.exists(file_path):
            return FileResponse(file_path)
        else:
            return {"message": "页面未找到"}
    
    # 静态资源（CSS、JS等）
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


# 增删改查schema和prompt，其中默认 electrolytes 和 electrolytes 为默认schema和prompt 不可以被删除和修改
from prompt_saving import (
    list_schemas, list_prompts, get_schema, get_prompt,
    create_schema, create_prompt, update_schema, update_prompt,
    delete_schema, delete_prompt, schema_exists, prompt_exists,
    DEFAULT_SCHEMA_NAME, DEFAULT_PROMPT_NAME
)

# 新增 输入 dataset_name, schema_name, prompt_name, schema_content, prompt_content,等混合模式   返回 graph_path， time_now 为当前时间，session_id由前端输入
from tune_prompt import tune_prompt_mixed
from datetime import datetime

# 导入 corpus 管理模块
from corpus_saving import (
    list_corpus_items, check_corpus, save_corpus, delete_corpus
)


# ==================== Schema 和 Prompt 的增删改查 API ====================

@app.get("/api/schemas")
async def list_schemas_api():
    """列出所有可用的schema名称"""
    try:
        schemas = list_schemas()
        return {"schemas": schemas}
    except Exception as e:
        import traceback
        error_detail = f"列出schemas失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/prompts")
async def list_prompts_api():
    """列出所有可用的prompt名称"""
    try:
        prompts = list_prompts()
        return {"prompts": prompts}
    except Exception as e:
        import traceback
        error_detail = f"列出prompts失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/schema/{schema_name}")
async def get_schema_api(schema_name: str):
    """获取指定schema的内容"""
    try:
        schema = get_schema(schema_name)
        if schema is None:
            raise HTTPException(status_code=404, detail=f"Schema '{schema_name}' 不存在")
        return {"schema_name": schema_name, "schema_content": schema}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/prompt/{prompt_name}")
async def get_prompt_api(prompt_name: str):
    """获取指定prompt的内容"""
    try:
        prompt = get_prompt(prompt_name)
        if prompt is None:
            raise HTTPException(status_code=404, detail=f"Prompt '{prompt_name}' 不存在")
        return {"prompt_name": prompt_name, "prompt_content": prompt}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取prompt失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class CreateSchemaRequest(BaseModel):
    schema_name: str
    schema_content: dict
    force_overwrite: bool = False


@app.post("/api/schema/create")
async def create_schema_api(request: CreateSchemaRequest):
    """创建新的schema"""
    try:
        create_schema(request.schema_name, request.schema_content, request.force_overwrite)
        return {"message": f"Schema '{request.schema_name}' 创建成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"创建schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class CreatePromptRequest(BaseModel):
    prompt_name: str
    prompt_content: str
    force_overwrite: bool = False


@app.post("/api/prompt/create")
async def create_prompt_api(request: CreatePromptRequest):
    """创建新的prompt"""
    try:
        create_prompt(request.prompt_name, request.prompt_content, request.force_overwrite)
        return {"message": f"Prompt '{request.prompt_name}' 创建成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"创建prompt失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class UpdateSchemaRequest(BaseModel):
    schema_name: str
    schema_content: dict


@app.post("/api/schema/update")
async def update_schema_api(request: UpdateSchemaRequest):
    """更新已存在的schema"""
    try:
        update_schema(request.schema_name, request.schema_content)
        return {"message": f"Schema '{request.schema_name}' 更新成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"更新schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class UpdatePromptRequest(BaseModel):
    prompt_name: str
    prompt_content: str


@app.post("/api/prompt/update")
async def update_prompt_api(request: UpdatePromptRequest):
    """更新已存在的prompt"""
    try:
        update_prompt(request.prompt_name, request.prompt_content)
        return {"message": f"Prompt '{request.prompt_name}' 更新成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"更新prompt失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class DeleteSchemaRequest(BaseModel):
    schema_name: str


@app.post("/api/schema/delete")
async def delete_schema_api(request: DeleteSchemaRequest):
    """删除指定的schema"""
    try:
        delete_schema(request.schema_name)
        return {"message": f"Schema '{request.schema_name}' 删除成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"删除schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class DeletePromptRequest(BaseModel):
    prompt_name: str


@app.post("/api/prompt/delete")
async def delete_prompt_api(request: DeletePromptRequest):
    """删除指定的prompt"""
    try:
        delete_prompt(request.prompt_name)
        return {"message": f"Prompt '{request.prompt_name}' 删除成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"删除prompt失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


# ==================== Corpus 的增删查 API ====================

@app.get("/api/corpus")
async def list_corpus_api():
    """列出所有可用的corpus名称"""
    try:
        corpus_names = list_corpus_items()["corpus_names"]
        return {"corpus_names": corpus_names}
    except Exception as e:
        import traceback
        error_detail = f"列出corpus失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/corpus/{corpus_name}")
async def get_corpus_api(corpus_name: str):
    """获取指定corpus的内容"""
    try:
        corpus_content = check_corpus(corpus_name)
        return {
            "corpus_name": corpus_name,
            "corpus_content": corpus_content
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"获取corpus失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class CreateCorpusRequest(BaseModel):
    corpus_name: str
    corpus_content: str
    force_new: bool = False


@app.post("/api/corpus/create")
async def create_corpus_api(request: CreateCorpusRequest):
    """创建新的corpus"""
    try:
        save_corpus(request.corpus_name, request.corpus_content, request.force_new)
        return {"message": f"Corpus '{request.corpus_name}' 创建成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"创建corpus失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class UpdateCorpusRequest(BaseModel):
    corpus_name: str
    corpus_content: str


@app.post("/api/corpus/update")
async def update_corpus_api(request: UpdateCorpusRequest):
    """更新已存在的corpus"""
    try:
        save_corpus(request.corpus_name, request.corpus_content, force_new=True)
        return {"message": f"Corpus '{request.corpus_name}' 更新成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"更新corpus失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class DeleteCorpusRequest(BaseModel):
    corpus_name: str


@app.post("/api/corpus/delete")
async def delete_corpus_api(request: DeleteCorpusRequest):
    """删除指定的corpus"""
    try:
        delete_corpus(request.corpus_name)
        return {"message": f"Corpus '{request.corpus_name}' 删除成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"删除corpus失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


# ==================== 混合模式生成图谱 API ====================

class TunePromptMixedRequest(BaseModel):
    dataset_name: str
    schema_name: Optional[str] = None
    prompt_name: Optional[str] = None
    schema_content: Optional[dict] = None
    prompt_content: Optional[str] = None
    session_id: str


@app.post("/api/tune_prompt_mixed")
async def tune_prompt_mixed_api(request: TunePromptMixedRequest):
    """
    混合模式：支持通过文件或内容字符串使用 schema 和 prompt
    返回生成的图谱路径
    """
    try:
        # 验证 dataset_name 是否在 corpus 列表中
        available_corpus_names = list_corpus_items()["corpus_names"]
        if request.dataset_name not in available_corpus_names:
            raise HTTPException(
                status_code=400,
                detail=f"数据集名称 '{request.dataset_name}' 不存在。可用的数据集: {', '.join(available_corpus_names)}"
            )
        
        # 生成当前时间戳
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 调用混合模式函数
        graph_path = tune_prompt_mixed(
            dataset_name=request.dataset_name,
            schema_name=request.schema_name,
            prompt_name=request.prompt_name,
            schema_content=request.schema_content,
            prompt_content=request.prompt_content,
            time_now=time_now,
            session_id=request.session_id
        )
        
        return {
            "message": "图谱生成成功",
            "graph_path": graph_path,
            "time_now": time_now,
            "session_id": request.session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"生成图谱失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)



@app.get("/api/stage-examples")
async def get_stage_examples_api():
    """
    返回 Stage1/Stage2 下拉框的可选风格（即 prompts/staged_customized/stage*_example.json 的 key）。
    前端用于动态填充 stage1_style / stage2_style 选项。
    """
    try:
        staged_dir = os.path.join(PROJECT_DIR, "prompts", "staged_customized")

        stage1_path = os.path.join(staged_dir, "stage1_example.json")
        stage2_path = os.path.join(staged_dir, "stage2_example.json")

        if not os.path.exists(stage1_path) or not os.path.exists(stage2_path):
            missing = []
            if not os.path.exists(stage1_path):
                missing.append(stage1_path)
            if not os.path.exists(stage2_path):
                missing.append(stage2_path)
            raise HTTPException(status_code=404, detail=f"Stage examples 文件不存在: {', '.join(missing)}")

        with open(stage1_path, "r", encoding="utf-8") as f1:
            stage1_data = json.load(f1)
        with open(stage2_path, "r", encoding="utf-8") as f2:
            stage2_data = json.load(f2)

        if not isinstance(stage1_data, dict) or not isinstance(stage2_data, dict):
            raise HTTPException(status_code=500, detail="Stage examples JSON 格式应为顶层对象（key 为风格名称）。")

        return {
            "参数1": list(stage1_data.keys()),
            "参数2": list(stage2_data.keys()),
        }
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Stage examples JSON 解析失败: {str(e)}")
    except Exception as e:
        import traceback
        error_detail = f"获取 stage examples 失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6777)
