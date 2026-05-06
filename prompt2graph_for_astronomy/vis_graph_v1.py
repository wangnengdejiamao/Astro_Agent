# 加载.env文件中的环境变量
import os
from dotenv import load_dotenv

# 加载.env文件
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"已加载.env文件: {env_path}")
else:
    print(f".env文件不存在: {env_path}")

# 设置Matplotlib缓存目录，解决多进程环境中的问题
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

# 读取 graph_path 中的图谱数据并显示在页面上  filename需要遵守命名规则
# "graph_by_file_s_{schema_name}_p_{prompt_name}.json"
# "graph_by_content_{time_now}_session_{session_id}.json"
import json
import sys
import glob
import re
import logging
import asyncio
import threading
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

from graph_tools.graph_util import GraphAnalyzer
from graph_tools.retriever import GraphRetriever
from graph_tools.graph_util import load_graph_from_json

from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# 图谱数据目录
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# 项目根目录
PROJECT_DIR = os.getenv("PROJECT_DIR") or os.path.dirname(os.path.abspath(__file__))

# Logger
logger = logging.getLogger(__name__)

# Pipeline 配置文件路径
PIPELINE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "example_pipeline.yml")
SIMPLE_PIPELINE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "simple_pipeline.yml")
BALANCED_PIPELINE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "balanced_pipeline.yml")


app = FastAPI(title="图谱可视化系统", version="1.0.0")

# 最大并发生成图谱数量（从环境变量读取，默认为2）
MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "2"))

# 全局变量：跟踪正在运行的生成任务
# 格式: {session_id: {'start_time': datetime, 'cancelled': bool}}
active_generations = {}

# 全局变量：存储已完成/失败的生成任务结果
# 格式: {session_id: {'status': 'completed'|'failed', 'graph_files': [...], 'time_now': str, 'output_dir': str, 'error': str}}
generation_results = {}

# 全局信号量：控制同时运行的图谱生成任务数量
generation_semaphore = threading.Semaphore(MAX_CONCURRENT_GENERATIONS)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 延迟初始化GraphRetriever
def get_graph_retriever() -> GraphRetriever:
    """
    获取GraphRetriever实例（延迟初始化）
    """
    try:
        logger.info("创建GraphRetriever实例")
        retriever = GraphRetriever()
        logger.info("GraphRetriever实例创建成功")
        return retriever
    except Exception as e:
        logger.error(f"创建GraphRetriever实例失败: {str(e)}")
        raise


def get_unique_output_dir(base_dir: str) -> str:
    """
    生成唯一的输出目录路径，如果目录已存在则自动添加后缀
    例如: base_dir 已存在，则返回 base_dir_1，如果还存在则返回 base_dir_2，以此类推
    """
    if not os.path.exists(base_dir):
        return base_dir
    
    base_name = base_dir
    counter = 1
    while True:
        new_dir = f"{base_name}_{counter}"
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1


def get_chunk_path(dataset_name: str, timestamp: Optional[str] = None) -> Optional[str]:
    """
    根据 dataset_name 和 timestamp 获取 chunk 文件路径
    优先查找 output/{dataset_name}/chunks.txt
    如果指定了 timestamp，查找 output/{dataset_name}/{timestamp}/chunks_doc_*.txt
    如果未指定 timestamp，查找最新时间戳子目录的 chunks_doc_*.txt
    """
    chunk_path = os.path.join(OUTPUT_DIR, dataset_name, "chunks.txt")
    if os.path.exists(chunk_path):
        return chunk_path
    
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        return None
    
    if timestamp:
        target_dir = os.path.join(dataset_dir, timestamp)
        if os.path.isdir(target_dir):
            chunk_files = glob.glob(os.path.join(target_dir, "chunks_doc_*.txt"))
            if chunk_files:
                return chunk_files[0]
    else:
        subdirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
        if subdirs:
            subdirs.sort(key=lambda x: os.path.basename(x), reverse=True)
            latest_subdir = subdirs[0]
            chunk_files = glob.glob(os.path.join(latest_subdir, "chunks_doc_*.txt"))
            if chunk_files:
                return chunk_files[0]
    
    return None


def find_graph_file(dataset_name: str, graph_type: Optional[str] = None,
                    schema_name: Optional[str] = None, prompt_name: Optional[str] = None,
                    session_id: Optional[str] = None, timestamp: Optional[str] = None,
                    filename: Optional[str] = None) -> Optional[str]:
    """
    根据dataset_name查找对应的图谱JSON文件

    Args:
        dataset_name: 数据集名称（如 "paper_mini"）
        graph_type: 图谱类型，"file"、"content" 或 "pipeline"
                    - "file": 旧格式 graph_by_file_*.json
                    - "content": 旧格式 graph_by_content_*.json
                    - "pipeline": run_end2end_pipeline 生成的子目录中的文件
                    - None: 自动查找，优先 pipeline > file > content
        schema_name: schema名称（用于file类型）
        prompt_name: prompt名称（用于file类型）
        session_id: session ID（用于content类型）
        timestamp: 时间戳目录名称（用于pipeline类型），如果指定则只在对应子目录中查找

    Returns:
        找到的文件路径，如果未找到则返回None
    """
    logger.info(f"find_graph_file called: dataset_name={dataset_name}, graph_type={graph_type}, timestamp={timestamp}, filename={filename}, OUTPUT_DIR={OUTPUT_DIR}")

    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    logger.info(f"查找目录: {dataset_dir}, exists={os.path.exists(dataset_dir)}")

    if not os.path.exists(dataset_dir):
        return None

    # 精确匹配：当同时提供 timestamp 和 filename 时，直接返回指定文件
    if timestamp and filename:
        safe_filename = os.path.basename(filename)
        exact_path = os.path.join(dataset_dir, timestamp, safe_filename)
        if os.path.isfile(exact_path):
            logger.info(f"[精确匹配] timestamp={timestamp} filename={safe_filename} -> {exact_path}")
            return exact_path
        logger.warning(f"[精确匹配失败] 文件不存在: {exact_path}")
    # pipeline 无 timestamp 仅有 filename（根目录下）
    if filename and not timestamp:
        safe_filename = os.path.basename(filename)
        exact_path = os.path.join(dataset_dir, safe_filename)
        if os.path.isfile(exact_path):
            logger.info(f"[精确匹配-根目录] filename={safe_filename} -> {exact_path}")
            return exact_path
    
    json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
    logger.info(f"找到的直接 JSON 文件: {json_files}")
    
    subdirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    logger.info(f"找到的子目录: {subdirs}")

    pipeline_files = []
    if timestamp:
        target_dir = os.path.join(dataset_dir, timestamp)
        if os.path.isdir(target_dir):
            pipeline_files = glob.glob(os.path.join(target_dir, "*.json"))
            logger.info(f"[timestamp={timestamp}] 在子目录 {target_dir} 中查找: {pipeline_files}")
        else:
            logger.warning(f"[timestamp={timestamp}] 子目录不存在: {target_dir}")
    elif subdirs:
        subdirs.sort(key=lambda x: os.path.basename(x), reverse=True)
        # 遍历子目录（最新优先），找到第一个包含图谱文件的目录
        for candidate_subdir in subdirs:
            candidate_files = glob.glob(os.path.join(candidate_subdir, "*.json"))
            has_graph = any(
                os.path.basename(f).startswith(("graph_", "multi_stage"))
                for f in candidate_files
            )
            if has_graph:
                pipeline_files = candidate_files
                logger.info(f"在子目录 {candidate_subdir} 中找到图谱文件: {pipeline_files}")
                break
        else:
            # 所有子目录都没有图谱文件，用最新目录的文件列表（会在后续被过滤）
            pipeline_files = glob.glob(os.path.join(subdirs[0], "*.json"))
            logger.warning(f"所有子目录均无图谱文件，最新子目录文件: {pipeline_files}")

    if graph_type == "pipeline" or graph_type is None:
        if pipeline_files:
            # 优先返回 multi_stage_deduplicated.json
            for f in pipeline_files:
                if os.path.basename(f) == "multi_stage_deduplicated.json":
                    logger.info(f"找到 Pipeline multi_stage_deduplicated.json: {f}")
                    return f
            # 如果没有 multi_stage_deduplicated.json，则返回第一个包含 deduplicated 的文件
            for f in pipeline_files:
                if "deduplicated" in os.path.basename(f):
                    logger.info(f"找到 Pipeline deduplicated 文件: {f}")
                    return f
            # 查找 graph_doc_*.json（pipeline 生成的图谱文件）
            for f in pipeline_files:
                if "graph_doc_" in os.path.basename(f) and f.endswith(".json"):
                    logger.info(f"找到 Pipeline graph_doc 文件: {f}")
                    return f
            # 过滤掉非图谱文件（如 corpus_doc, chunks, cid_query_report 等），避免误返回
            graph_only = [f for f in pipeline_files if os.path.basename(f).startswith(("graph_", "multi_stage"))]
            if graph_only:
                logger.info(f"找到 Pipeline 图谱文件(filtered): {graph_only[0]}")
                return graph_only[0]
            logger.warning(f"子目录中无有效图谱文件，跳过: {[os.path.basename(f) for f in pipeline_files]}")
    
    if graph_type == "file" or (graph_type is None and schema_name and prompt_name):
        if pipeline_files:
            # 优先返回 multi_stage_deduplicated.json
            for f in pipeline_files:
                if os.path.basename(f) == "multi_stage_deduplicated.json":
                    logger.info(f"[file类型] 找到 Pipeline multi_stage_deduplicated.json: {f}")
                    return f
            # 如果没有 multi_stage_deduplicated.json，则返回第一个包含 deduplicated 的文件
            for f in pipeline_files:
                if "deduplicated" in os.path.basename(f):
                    logger.info(f"[file类型] 找到 Pipeline deduplicated 文件: {f}")
                    return f
            # 查找 graph_doc_*.json（pipeline 生成的图谱文件）
            for f in pipeline_files:
                if "graph_doc_" in os.path.basename(f) and f.endswith(".json"):
                    logger.info(f"[file类型] 找到 Pipeline graph_doc 文件: {f}")
                    return f
            graph_only = [f for f in pipeline_files if os.path.basename(f).startswith(("graph_", "multi_stage"))]
            if graph_only:
                logger.info(f"[file类型] 找到 Pipeline 图谱文件(filtered): {graph_only[0]}")
                return graph_only[0]
            logger.warning(f"[file类型] 子目录中无有效图谱文件，跳过: {[os.path.basename(f) for f in pipeline_files]}")

        pattern = f"graph_by_file_s_{schema_name}_p_{prompt_name}.json"
        for file_path in json_files:
            if os.path.basename(file_path) == pattern:
                return file_path
    
    if graph_type == "content" or (graph_type is None and session_id):
        pattern = f"graph_by_content_.*_session_{session_id}.json"
        for file_path in json_files:
            if re.match(pattern, os.path.basename(file_path)):
                return file_path
    
    if graph_type is None:
        for file_path in json_files:
            if "graph_by_file" in os.path.basename(file_path):
                return file_path
        for file_path in json_files:
            if "graph_by_content" in os.path.basename(file_path):
                return file_path
        # 只返回图谱文件，不返回 corpus/chunks 等非图谱文件
        graph_only = [f for f in json_files if os.path.basename(f).startswith(("graph_", "multi_stage"))]
        if graph_only:
            return graph_only[0]
    
    return None


def find_meta_graph_file(dataset_name: str, graph_path: Optional[str] = None) -> Optional[str]:
    """
    根据dataset_name查找对应的元图谱JSON文件
    
    Args:
        dataset_name: 数据集名称（如 "paper_mini_test"）
        graph_path: 基础图谱文件路径，用于推断元图谱文件名称
    
    Returns:
        找到的元图谱文件路径，如果未找到则返回None
    """
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    if not os.path.exists(dataset_dir):
        return None
    
    # Pipeline 模式：优先查找子目录中的最新元图谱文件
    subdirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
    if subdirs:
        subdirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_subdir = subdirs[0]
        meta_file = os.path.join(latest_subdir, f"{os.path.basename(latest_subdir)}_meta.json")
        if os.path.exists(meta_file):
            return meta_file
        # 查找任何 _meta.json 文件
        for f in glob.glob(os.path.join(latest_subdir, "*_meta.json")):
            return f
    
    json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
    
    if graph_path:
        base_name = os.path.splitext(os.path.basename(graph_path))[0]
        meta_pattern = f"{base_name}_meta.json"
        for file_path in json_files:
            if os.path.basename(file_path) == meta_pattern:
                return file_path
    
    for file_path in json_files:
        if "_meta.json" in os.path.basename(file_path):
            return file_path
    
    return None


def get_analyzer(dataset_name: str, graph_path: Optional[str] = None) -> GraphAnalyzer:
    """
    获取GraphAnalyzer实例
    """
    logger.info(f"get_analyzer called: dataset_name={dataset_name}, graph_path={graph_path}")
    
    if graph_path is None:
        graph_path = find_graph_file(dataset_name)
        logger.info(f"find_graph_file 返回: {graph_path}")
    if graph_path is None or not os.path.exists(graph_path):
        logger.error(f"图谱文件不存在: graph_path={graph_path}")
        raise HTTPException(
            status_code=404,
            detail=f"未找到数据集 {dataset_name} 的图谱文件"
        )
    logger.info(f"创建 GraphAnalyzer: {graph_path}")
    return GraphAnalyzer(graph_path)


@app.get("/api/graph/{dataset_name}")     # 请你修改前端，输入的session_id 是当前session_id，如果选择 graph_type = "file"，则直接展示图谱文件名称，如果选择 content，则传入当前session_id用于筛选，只有匹配上的session能被看到
async def get_full_graph(
    dataset_name: str,
    graph_type: Optional[str] = None,
    schema_name: Optional[str] = None,
    prompt_name: Optional[str] = None,
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    filename: Optional[str] = None,
    threshold_high: int = 2000,
    threshold_medium: int = 500,
    score_threshold: Optional[float] = None,
    score_type: Optional[str] = None
):
    """
    获取完整图的可视化数据

    如果选择 graph_type = "file"，则需要提供 schema_name 和 prompt_name 参数，直接展示对应的图谱文件
    如果选择 graph_type = "content"，则需要提供 session_id 参数，用于筛选对应的图谱文件
    如果选择 graph_type = "pipeline"，可以提供 timestamp 参数来指定具体的时间戳目录

    Args:
        dataset_name: 数据集名称
        graph_type: 图谱类型，"file"、"content" 或 "pipeline"
        schema_name: schema名称（用于file类型，必需）
        prompt_name: prompt名称（用于file类型，必需）
        session_id: session ID（用于content类型，必需）
        timestamp: 时间戳目录名称（用于pipeline类型，指定具体的时间戳目录）
        threshold_high: 高阈值
        threshold_medium: 中阈值
        score_threshold: 评分阈值，低于此值的边会标记为dimmed（可选）
        score_type: 评分字段名，支持 "accuracy_score"、"triple_support_score"、"usefulness_score"、
                   以及组合字段 "min_accuracy_usefulness"、"min_accuracy_triple"、"min_usefulness_triple"、"min_all"

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
        elif graph_type == "pipeline":
            pass

        # 查找图谱文件
        graph_path = find_graph_file(dataset_name, graph_type, schema_name, prompt_name, session_id, timestamp, filename)
        if graph_path is None:
            # 生成更详细的错误信息
            if graph_type == "file":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (schema: {schema_name}, prompt: {prompt_name})"
            elif graph_type == "content":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (session_id: {session_id})"
            elif graph_type == "pipeline":
                error_msg = f"未找到数据集 {dataset_name} 的 pipeline 图谱文件 (timestamp: {timestamp}, filename: {filename})"
            else:
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件"
            raise HTTPException(status_code=404, detail=error_msg)
        
        # 获取分析器并转换数据
        analyzer = get_analyzer(dataset_name, graph_path)
        graph_data = analyzer.convert_to_echarts_format(
            max_nodes=1000,
            threshold_high=threshold_high,
            threshold_medium=threshold_medium,
            score_threshold=score_threshold,
            score_type=score_type
        )
        
        # 添加文件信息到返回数据中
        graph_data["graph_file"] = os.path.basename(graph_path)
        graph_data["graph_path"] = graph_path
        
        # 查找并添加元图谱数据
        meta_graph_path = find_meta_graph_file(dataset_name, graph_path)
        if meta_graph_path and os.path.exists(meta_graph_path):
            try:
                with open(meta_graph_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                meta_graph_json = meta_data.get("meta_graph", meta_data)
                meta_nodes = meta_graph_json.get("nodes", [])
                meta_edges = meta_graph_json.get("edges", [])
                
                # 转换元图谱节点为 ECharts 格式
                meta_graph_nodes = []
                meta_graph_categories = set()
                for node in meta_nodes:
                    meta_graph_categories.add(node.get("schema_type_subj", "meta"))
                    meta_graph_categories.add(node.get("schema_type_obj", "meta"))
                    meta_graph_nodes.append({
                        "id": node.get("id", ""),
                        "name": node.get("subject", ""),
                        "category": node.get("schema_type_subj", "meta"),
                        "properties": {
                            "subject": node.get("subject", ""),
                            "relation": node.get("relation", ""),
                            "object": node.get("object", ""),
                            "schema_type_subj": node.get("schema_type_subj", ""),
                            "schema_type_obj": node.get("schema_type_obj", ""),
                            "source": node.get("source", ""),
                            "evidence": node.get("evidence", ""),
                            "chunk_ids": node.get("chunk_ids", [])
                        }
                    })
                
                # 转换元图谱边为 ECharts 格式：根据 meta_edges 用节点 id 连线，source/target 必须是 nodes 中的 id
                meta_graph_links = []
                node_ids = {n.get("id") for n in meta_nodes}
                for edge in meta_edges:
                    src = edge.get("source_triple_id", "")
                    tgt = edge.get("target_triple_id", "")
                    if src in node_ids and tgt in node_ids:
                        meta_graph_links.append({
                            "source": src,
                            "target": tgt,
                            "name": edge.get("relation", ""),
                            "value": 1,
                            "evidence": edge.get("evidence", ""),
                        })
                
                # 构建分类列表
                meta_categories = [{"name": cat} for cat in meta_graph_categories]
                
                graph_data["meta_graph"] = {
                    "nodes": meta_graph_nodes,
                    "links": meta_graph_links,
                    "categories": meta_categories,
                    "meta_graph_file": os.path.basename(meta_graph_path)
                }
            except Exception as e:
                logger.warning(f"加载元图谱失败: {e}")
        
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
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    filename: Optional[str] = None
):
    """
    下载原始图谱 JSON 文件

    Args:
        dataset_name: 数据集名称
        graph_type: 图谱类型，"file"、"content" 或 "pipeline"
        schema_name: schema名称（用于file类型）
        prompt_name: prompt名称（用于file类型）
        session_id: session ID（用于content类型）
        timestamp: 时间戳目录名称（用于pipeline类型）

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
        elif graph_type == "pipeline":
            pass

        # 查找图谱文件
        graph_path = find_graph_file(dataset_name, graph_type, schema_name, prompt_name, session_id, timestamp, filename)
        if graph_path is None:
            if graph_type == "file":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (schema: {schema_name}, prompt: {prompt_name})"
            elif graph_type == "content":
                error_msg = f"未找到数据集 {dataset_name} 的图谱文件 (session_id: {session_id})"
            elif graph_type == "pipeline":
                error_msg = f"未找到数据集 {dataset_name} 的 pipeline 图谱文件"
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
    timestamp: Optional[str] = None
    filename: Optional[str] = None


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
            request.session_id,
            request.timestamp,
            request.filename
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
    session_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    filename: Optional[str] = None,
    fuzzy: Optional[str] = None,
):
    """
    获取节点详细信息

    Args:
        fuzzy: 当设置为 "true" 时启用模糊匹配，支持包含匹配和编辑距离匹配
    """
    try:
        graph_path = find_graph_file(
            dataset_name,
            graph_type,
            schema_name,
            prompt_name,
            session_id,
            timestamp,
            filename
        )
        if graph_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"未找到数据集 {dataset_name} 的图谱文件"
            )

        analyzer = get_analyzer(dataset_name, graph_path)

        use_fuzzy = fuzzy and fuzzy.lower() == "true"

        if use_fuzzy:
            matched_nodes = analyzer.fuzzy_search_nodes(node_id)
            if not matched_nodes:
                raise HTTPException(
                    status_code=404,
                    detail=f"模糊搜索未找到包含 '{node_id}' 的节点"
                )

            subgraph = analyzer._expand_subgraph(matched_nodes, depth=2)

            subgraph_data = _build_subgraph_response(analyzer, subgraph, matched_nodes)
            return {
                "fuzzy_matched": True,
                "matched_count": len(matched_nodes),
                "matched_nodes": list(matched_nodes),
                "subgraph": subgraph_data,
            }
        else:
            if not analyzer.graph or node_id not in analyzer.graph:
                raise HTTPException(status_code=404, detail="节点未找到")

            node_data = analyzer.graph.nodes[node_id]
            props = node_data.get("properties", {}).copy()

            name = props.get("name", node_id)
            if isinstance(name, list):
                name = "\n".join(str(n) for n in name) if name else str(node_id)
            elif not isinstance(name, str):
                name = str(name)
            props["name"] = name

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
                "fuzzy_matched": False,
            }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取节点信息失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


def _build_subgraph_response(analyzer: Any, subgraph: Any, matched_nodes: set) -> Dict[str, Any]:
    """
    构建模糊匹配返回的子图响应
    """
    nodes_list = []
    edges_list = []
    matched_node_ids = set(matched_nodes)

    def _get_node_name(nid: str) -> str:
        n_data = analyzer.graph.nodes.get(nid, {})
        n_props = n_data.get("properties", {}) if isinstance(n_data, dict) else {}
        n_name = n_props.get("name", nid)
        if isinstance(n_name, list):
            return "\n".join(str(n) for n in n_name) if n_name else str(nid)
        if not isinstance(n_name, str):
            return str(n_name)
        return n_name

    for node_id in subgraph.nodes():
        node_data = subgraph.nodes[node_id]
        props = node_data.get("properties", {}).copy()
        name = props.get("name", node_id)
        if isinstance(name, list):
            name = "\n".join(str(n) for n in name) if name else str(node_id)
        elif not isinstance(name, str):
            name = str(name)
        props["name"] = name

        nodes_list.append({
            "id": node_id,
            "label": node_data.get("label", "entity"),
            "properties": props,
            "degree": subgraph.degree(node_id),
            "is_matched": node_id in matched_node_ids,
        })

    seen_edges = set()
    is_multi = bool(getattr(subgraph, "is_multigraph", lambda: False)())
    if is_multi:
        for u, v, k, data in subgraph.edges(keys=True, data=True):
            edge_key = f"{u}->{v}:{k}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges_list.append({
                    "source": u,
                    "target": v,
                    "name": data.get("relation", "related_to"),
                    "source_name": _get_node_name(u),
                    "target_name": _get_node_name(v),
                    "evidence": data.get("evidence", ""),
                })
    else:
        for u, v, data in subgraph.edges(data=True):
            edge_key = f"{u}->{v}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges_list.append({
                    "source": u,
                    "target": v,
                    "name": data.get("relation", "related_to"),
                    "source_name": _get_node_name(u),
                    "target_name": _get_node_name(v),
                    "evidence": data.get("evidence", ""),
                })

    return {
        "nodes": nodes_list,
        "edges": edges_list,
        "stats": {
            "total_nodes": subgraph.number_of_nodes(),
            "total_edges": len(edges_list),
            "matched_nodes_count": len(matched_node_ids),
        }
    }


class ChunkRequest(BaseModel):
    dataset_name: str
    chunk_id: str
    timestamp: Optional[str] = None


@app.post("/api/chunk")
async def get_chunk_text(request: ChunkRequest):
    """根据chunk id获取原文"""
    try:
        chunk_path = get_chunk_path(request.dataset_name, request.timestamp)
        if not chunk_path or not os.path.exists(chunk_path):
            raise HTTPException(
                status_code=404,
                detail=f"未找到chunk文件: {chunk_path or request.dataset_name}"
            )
        # 使用延迟初始化的GraphRetriever
        graph_retriever = get_graph_retriever()
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


@app.get("/api/evaluation/{dataset_name}")
async def get_evaluation_data(
    dataset_name: str,
    timestamp: Optional[str] = None
):
    """
    获取评估数据（元图谱评估、社区评估、社区报告）
    仅完整模式的数据集才会有评估文件
    """
    try:
        dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
        if not os.path.exists(dataset_dir):
            return {
                "has_evaluation": False,
                "meta_evaluation": None,
                "community_evaluation": None,
                "community_report": None,
                "timestamp": None
            }

        # 确定目标目录
        target_dir = None
        resolved_timestamp = timestamp
        if timestamp:
            candidate = os.path.join(dataset_dir, timestamp)
            if os.path.isdir(candidate):
                target_dir = candidate
        else:
            subdirs = [d for d in glob.glob(os.path.join(dataset_dir, "*")) if os.path.isdir(d)]
            if subdirs:
                subdirs.sort(key=lambda x: os.path.basename(x), reverse=True)
                target_dir = subdirs[0]
                resolved_timestamp = os.path.basename(target_dir)

        if not target_dir:
            return {
                "has_evaluation": False,
                "meta_evaluation": None,
                "community_evaluation": None,
                "community_report": None,
                "timestamp": resolved_timestamp
            }

        # 读取评估文件
        meta_eval = None
        meta_eval_path = os.path.join(target_dir, "multi_stage_deduplicated_meta_evaluation.json")
        if os.path.exists(meta_eval_path):
            with open(meta_eval_path, 'r', encoding='utf-8') as f:
                meta_eval = json.load(f)

        community_eval = None
        community_eval_path = os.path.join(target_dir, "multi_stage_deduplicated_community_report_evaluation.json")
        if os.path.exists(community_eval_path):
            with open(community_eval_path, 'r', encoding='utf-8') as f:
                community_eval = json.load(f)

        community_report = None
        community_report_path = os.path.join(target_dir, "multi_stage_deduplicated_community_report.txt")
        if os.path.exists(community_report_path):
            with open(community_report_path, 'r', encoding='utf-8') as f:
                community_report = f.read()

        has_evaluation = any([meta_eval, community_eval, community_report])

        return {
            "has_evaluation": has_evaluation,
            "meta_evaluation": meta_eval,
            "community_evaluation": community_eval,
            "community_report": community_report,
            "timestamp": resolved_timestamp
        }
    except Exception as e:
        import traceback
        error_detail = f"获取评估数据失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


def _profile_paths_for_graph(dataset_name: str, timestamp: Optional[str] = None, filename: Optional[str] = None) -> Dict[str, str]:
    graph_path = find_graph_file(
        dataset_name,
        graph_type="pipeline",
        timestamp=timestamp,
        filename=filename,
    )
    if not graph_path:
        raise HTTPException(status_code=404, detail="Graph file not found")
    run_dir = os.path.dirname(graph_path)
    return {
        "graph_path": graph_path,
        "summary_path": os.path.join(run_dir, "summary.json"),
        "profiles_path": os.path.join(run_dir, "source_profiles.json"),
    }


_JSON_FILE_CACHE = {}


def _load_json_cached(path: str) -> Any:
    """Load a JSON file once and refresh it only when the file changes."""
    mtime = os.path.getmtime(path)
    cached = _JSON_FILE_CACHE.get(path)
    if cached and cached.get("mtime") == mtime:
        return cached["data"]
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    _JSON_FILE_CACHE[path] = {"mtime": mtime, "data": data}
    return data


def _counter_total(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    total = 0
    for item in value.values():
        try:
            total += int(item)
        except Exception:
            pass
    return total


@app.get("/api/source-profiles/{dataset_name}")
async def get_source_profiles(
    dataset_name: str,
    timestamp: Optional[str] = None,
    filename: Optional[str] = None,
    feature: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 120,
):
    """Return source-centric feature profiles for the astronomy KG frontend."""
    paths = _profile_paths_for_graph(dataset_name, timestamp=timestamp, filename=filename)
    if not os.path.exists(paths["profiles_path"]):
        raise HTTPException(status_code=404, detail="source_profiles.json not found for this graph run")

    profiles = _load_json_cached(paths["profiles_path"])

    summary = {}
    if os.path.exists(paths["summary_path"]):
        summary = dict(_load_json_cached(paths["summary_path"]))

    rows = []
    source_query = (source or "").strip().lower()
    feature_query = (feature or "").strip()
    for name, profile in profiles.items():
        features = profile.get("features") or {}
        if feature_query and feature_query not in features:
            continue
        if source_query and source_query not in name.lower():
            continue
        rows.append(
            {
                "source": name,
                "feature_score": _counter_total(features),
                "features": features,
                "methods": profile.get("methods") or {},
                "parameters": profile.get("parameters") or {},
                "instruments": profile.get("instruments") or {},
                "papers": profile.get("papers") or {},
                "evidence": (profile.get("evidence") or [])[:5],
            }
        )

    rows.sort(key=lambda item: item["feature_score"], reverse=True)
    limit = max(1, min(int(limit or 120), 500))
    return {
        "summary": summary,
        "profiles": rows[:limit],
        "count": len(rows),
        "graph_file": os.path.basename(paths["graph_path"]),
        "profiles_file": os.path.basename(paths["profiles_path"]),
    }


@app.get("/api/kg-summary/{dataset_name}")
async def get_kg_summary(
    dataset_name: str,
    timestamp: Optional[str] = None,
    filename: Optional[str] = None,
):
    """Return the compact KG build summary."""
    paths = _profile_paths_for_graph(dataset_name, timestamp=timestamp, filename=filename)
    if not os.path.exists(paths["summary_path"]):
        raise HTTPException(status_code=404, detail="summary.json not found for this graph run")
    summary = dict(_load_json_cached(paths["summary_path"]))
    summary["graph_file"] = os.path.basename(paths["graph_path"])
    return summary


@app.get("/api/datasets")
async def list_datasets():
    """
    列出所有可用的数据集
    支持三种格式：
    1. graph_by_file_*.json（旧格式）
    2. graph_by_content_*.json（旧格式）
    3. 子目录中的 *_deduplicated.json（Pipeline 格式）
    """
    try:
        if not os.path.exists(OUTPUT_DIR):
            return {"datasets": []}
        
        datasets = []
        for item in os.listdir(OUTPUT_DIR):
            item_path = os.path.join(OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                graph_files = []
                
                # 1. 查找根目录下的旧格式 JSON 文件
                json_files = glob.glob(os.path.join(item_path, "*.json"))
                for json_file in json_files:
                    filename = os.path.basename(json_file)
                    if "graph_by_file" in filename:
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
                        match = re.match(r"graph_by_content_(.+?)_session_(.+?)\.json", filename)
                        if match:
                            graph_files.append({
                                "type": "content",
                                "filename": filename,
                                "time": match.group(1),
                                "session_id": match.group(2),
                                "path": json_file
                            })
                
                # 2. 查找子目录中的 Pipeline 格式图谱文件
                # 每个子目录里的所有图谱文件都会被列出（过滤掉 corpus/chunks 等非图谱文件）
                subdirs = [d for d in glob.glob(os.path.join(item_path, "*")) if os.path.isdir(d)]
                if subdirs:
                    # 按名称（时间戳）倒序排序，最新的排在前面
                    subdirs.sort(key=lambda x: os.path.basename(x), reverse=True)

                    for subdir in subdirs:
                        subdir_name = os.path.basename(subdir)
                        subdir_json_files = sorted(glob.glob(os.path.join(subdir, "*.json")))
                        # 过滤出图谱文件：名字以 graph_ 或 multi_stage 开头
                        graph_candidates = [
                            f for f in subdir_json_files
                            if os.path.basename(f).startswith(("graph_", "multi_stage"))
                        ]
                        # 按优先级排序：multi_stage_deduplicated.json > 其他 deduplicated > graph_doc_* > 其余
                        def _pipeline_sort_key(path):
                            name = os.path.basename(path)
                            if name == "multi_stage_deduplicated.json":
                                return (0, name)
                            if "deduplicated" in name:
                                return (1, name)
                            if name.startswith("graph_doc_"):
                                return (2, name)
                            return (3, name)
                        graph_candidates.sort(key=_pipeline_sort_key)

                        for graph_file in graph_candidates:
                            graph_files.append({
                                "type": "pipeline",
                                "filename": os.path.basename(graph_file),
                                "schema_name": "pipeline",
                                "prompt_name": "pipeline",
                                "path": graph_file,
                                "timestamp": subdir_name
                            })
                
                # 只有存在图谱文件的目录才返回
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

    @app.get("/quality-eval.html")
    async def read_quality_eval():
        """返回图谱质量评估页面（stage4 提示词渲染）"""
        file_path = os.path.join(frontend_dir, "quality-eval.html")
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
    DEFAULT_SCHEMA_NAME, DEFAULT_PROMPT_NAME,
    # 分阶段 prompt 支持
    list_staged_prompts, get_staged_prompt_files, get_staged_prompt_content,
    update_staged_prompt_content, create_staged_prompt_file, delete_staged_prompt_file,
    STAGE_FILES,
    # 分阶段 prompt 文件夹管理
    create_staged_prompt_folder, delete_staged_prompt_folder
)

# 新增 输入 dataset_name, schema_name, prompt_name, schema_content, prompt_content,等混合模式   返回 graph_path， time_now 为当前时间，session_id由前端输入
from tune_prompt import tune_prompt_mixed
from run_end2end_pipeline import run_end2end_pipeline as run_pipeline_func
from datetime import datetime

# 导入 corpus 管理模块
from corpus_saving import (
    list_corpus_items, check_corpus, save_corpus, delete_corpus,
    create_folder, rename_folder, delete_folder, collect_corpus_from_folder
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


@app.get("/api/prompts4Gen")
async def get_available_prompts():
    """
    获取可用的 prompt 列表
    返回 prompts/ 目录下所有包含 stage1 和 stage2 文件的子文件夹，
    以及单个 .txt prompt 文件
    """
    prompts_dir = os.path.join(PROJECT_DIR, "prompts")
    if not os.path.exists(prompts_dir):
        return {"prompts": []}

    prompts_list = []
    for item_name in os.listdir(prompts_dir):
        item_path = os.path.join(prompts_dir, item_name)
        
        # 跳过 customized_example 目录
        if item_name == "customized_example":
            continue
            
        if os.path.isdir(item_path):
            # 检查是否包含 stage1 和 stage2 文件（分阶段 prompt 文件夹）
            stage1_path = os.path.join(item_path, "stage1_entity_recognition.txt")
            stage2_path = os.path.join(item_path, "stage2_relation_extraction.txt")
            if os.path.exists(stage1_path) and os.path.exists(stage2_path):
                prompts_list.append({"name": item_name, "type": "folder"})
        elif item_name.endswith('.txt'):
            # 单个 prompt 文件
            prompts_list.append({"name": item_name[:-4], "type": "file"})  # 去掉 .txt 后缀

    return {"prompts": prompts_list}


@app.get("/api/schemas4Gen")
async def get_available_schemas():
    """
    获取可用的 schema 文件列表
    """
    schemas_dir = os.path.join(PROJECT_DIR, "schemas")
    if not os.path.exists(schemas_dir):
        return {"schemas": []}

    schemas_list = []
    for file_name in os.listdir(schemas_dir):
        if file_name.endswith(".json"):
            schema_name = file_name[:-5]
            schemas_list.append({"name": schema_name, "file": file_name})

    return {"schemas": schemas_list}


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


# ==================== Schema 文件夹管理 API ====================

@app.get("/api/schemas/folders")
async def list_schema_folders_api():
    """获取 schemas 目录下的文件夹结构"""
    try:
        schemas_dir = os.path.join(PROJECT_DIR, "schemas")
        if not os.path.exists(schemas_dir):
            return {"folders": []}
        
        folders = []
        for item in os.listdir(schemas_dir):
            item_path = os.path.join(schemas_dir, item)
            if os.path.isdir(item_path):
                # 统计文件夹内的 schema 文件
                schemas = []
                for f in os.listdir(item_path):
                    if f.endswith('.json'):
                        schemas.append(f[:-5])  # 去掉 .json 后缀
                folders.append({
                    "name": item,
                    "schemas": schemas,
                    "schema_count": len(schemas)
                })
        
        return {"folders": folders}
    except Exception as e:
        import traceback
        error_detail = f"获取schema文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/schemas/folder/{folder_name}")
async def get_schema_folder_api(folder_name: str):
    """获取指定文件夹内的所有 schema"""
    try:
        folder_path = os.path.join(PROJECT_DIR, "schemas", folder_name)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            raise HTTPException(status_code=404, detail=f"文件夹 '{folder_name}' 不存在")
        
        schemas = []
        for f in os.listdir(folder_path):
            if f.endswith('.json'):
                schemas.append(f[:-5])
        
        return {"folder_name": folder_name, "schemas": schemas}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取文件夹内容失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/schemas/folder/{folder_name}/{schema_name}")
async def get_schema_in_folder_api(folder_name: str, schema_name: str):
    """获取指定文件夹内的 schema 内容"""
    try:
        schema_path = os.path.join(PROJECT_DIR, "schemas", folder_name, f"{schema_name}.json")
        if not os.path.exists(schema_path):
            raise HTTPException(status_code=404, detail=f"Schema '{schema_name}' 不存在于文件夹 '{folder_name}'")
        
        with open(schema_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        return {"folder_name": folder_name, "schema_name": schema_name, "schema_content": content}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"获取schema内容失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaFolderCreateRequest(BaseModel):
    folder_name: str

@app.post("/api/schemas/folder/create")
async def create_schema_folder_api(request: SchemaFolderCreateRequest):
    """创建新的 schema 文件夹"""
    try:
        folder_path = os.path.join(PROJECT_DIR, "schemas", request.folder_name)
        if os.path.exists(folder_path):
            raise HTTPException(status_code=400, detail=f"文件夹 '{request.folder_name}' 已存在")
        
        os.makedirs(folder_path, exist_ok=True)
        return {"message": f"文件夹 '{request.folder_name}' 创建成功", "folder_name": request.folder_name}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"创建文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaFolderDeleteRequest(BaseModel):
    folder_name: str

@app.post("/api/schemas/folder/delete")
async def delete_schema_folder_api(request: SchemaFolderDeleteRequest):
    """删除 schema 文件夹"""
    try:
        folder_path = os.path.join(PROJECT_DIR, "schemas", request.folder_name)
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"文件夹 '{request.folder_name}' 不存在")
        
        import shutil
        shutil.rmtree(folder_path)
        return {"message": f"文件夹 '{request.folder_name}' 删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"删除文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaFolderUpdateRequest(BaseModel):
    old_folder_name: str
    new_folder_name: str

@app.post("/api/schemas/folder/update")
async def update_schema_folder_api(request: SchemaFolderUpdateRequest):
    """重命名 schema 文件夹"""
    try:
        old_path = os.path.join(PROJECT_DIR, "schemas", request.old_folder_name)
        new_path = os.path.join(PROJECT_DIR, "schemas", request.new_folder_name)
        
        if not os.path.exists(old_path):
            raise HTTPException(status_code=404, detail=f"文件夹 '{request.old_folder_name}' 不存在")
        if os.path.exists(new_path):
            raise HTTPException(status_code=400, detail=f"文件夹 '{request.new_folder_name}' 已存在")
        
        os.rename(old_path, new_path)
        return {"message": f"文件夹重命名成功", "old_name": request.old_folder_name, "new_name": request.new_folder_name}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"重命名文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaCreateRequest(BaseModel):
    schema_name: str
    schema_content: dict
    folder_name: Optional[str] = None

@app.post("/api/schema/create")
async def create_schema_api(request: SchemaCreateRequest):
    """创建新的 schema 文件"""
    try:
        if request.folder_name:
            schema_path = os.path.join(PROJECT_DIR, "schemas", request.folder_name, f"{request.schema_name}.json")
        else:
            schema_path = os.path.join(PROJECT_DIR, "schemas", f"{request.schema_name}.json")
        
        if os.path.exists(schema_path):
            raise HTTPException(status_code=400, detail=f"Schema '{request.schema_name}' 已存在")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(schema_path), exist_ok=True)
        
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(request.schema_content, f, ensure_ascii=False, indent=2)
        
        return {"message": f"Schema '{request.schema_name}' 创建成功"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"创建schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaDeleteRequest(BaseModel):
    schema_name: str
    folder_name: Optional[str] = None

@app.post("/api/schema/delete")
async def delete_schema_api(request: SchemaDeleteRequest):
    """删除 schema 文件"""
    try:
        if request.folder_name:
            schema_path = os.path.join(PROJECT_DIR, "schemas", request.folder_name, f"{request.schema_name}.json")
        else:
            schema_path = os.path.join(PROJECT_DIR, "schemas", f"{request.schema_name}.json")
        
        if not os.path.exists(schema_path):
            raise HTTPException(status_code=404, detail=f"Schema '{request.schema_name}' 不存在")
        
        os.remove(schema_path)
        return {"message": f"Schema '{request.schema_name}' 删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"删除schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaUpdateRequest(BaseModel):
    schema_name: str
    schema_content: dict
    folder_name: Optional[str] = None

@app.post("/api/schema/update")
async def update_schema_api(request: SchemaUpdateRequest):
    """更新 schema 文件内容"""
    try:
        if request.folder_name:
            schema_path = os.path.join(PROJECT_DIR, "schemas", request.folder_name, f"{request.schema_name}.json")
        else:
            schema_path = os.path.join(PROJECT_DIR, "schemas", f"{request.schema_name}.json")
        
        if not os.path.exists(schema_path):
            raise HTTPException(status_code=404, detail=f"Schema '{request.schema_name}' 不存在")
        
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(request.schema_content, f, ensure_ascii=False, indent=2)
        
        return {"message": f"Schema '{request.schema_name}' 更新成功"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"更新schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaFolderCreateSchemaRequest(BaseModel):
    folder_name: str
    schema_name: str
    schema_content: dict

@app.post("/api/schemas/folder/create-schema")
async def create_schema_in_folder_api(request: SchemaFolderCreateSchemaRequest):
    """在指定文件夹内创建 schema"""
    try:
        folder_path = os.path.join(PROJECT_DIR, "schemas", request.folder_name)
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"文件夹 '{request.folder_name}' 不存在")
        
        schema_path = os.path.join(folder_path, f"{request.schema_name}.json")
        if os.path.exists(schema_path):
            raise HTTPException(status_code=400, detail=f"Schema '{request.schema_name}' 已存在于文件夹中")
        
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(request.schema_content, f, ensure_ascii=False, indent=2)
        
        return {"message": f"Schema '{request.schema_name}' 创建成功", "folder_name": request.folder_name}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"创建schema失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class SchemaFolderDeleteSchemaRequest(BaseModel):
    folder_name: str
    schema_name: str

@app.post("/api/schemas/folder/delete-schema")
async def delete_schema_in_folder_api(request: SchemaFolderDeleteSchemaRequest):
    """删除文件夹内的 schema"""
    try:
        schema_path = os.path.join(PROJECT_DIR, "schemas", request.folder_name, f"{request.schema_name}.json")
        if not os.path.exists(schema_path):
            raise HTTPException(status_code=404, detail=f"Schema '{request.schema_name}' 不存在于文件夹 '{request.folder_name}'")
        
        os.remove(schema_path)
        return {"message": f"Schema '{request.schema_name}' 删除成功"}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"删除schema失败: {str(e)}\n{traceback.format_exc()}"
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


@app.get("/api/stage4-prompts")
async def get_stage4_prompts_api():
    """获取 Stage4 评分相关的 prompt 模板"""
    try:
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "staged")
        
        result = {}
        
        node_accuracy_path = os.path.join(prompts_dir, "stage4_node_accuracy.txt")
        if os.path.exists(node_accuracy_path):
            with open(node_accuracy_path, 'r', encoding='utf-8') as f:
                result["stage4_node_accuracy"] = f.read()
        else:
            result["stage4_node_accuracy"] = None
        
        triple_support_path = os.path.join(prompts_dir, "stage4_triple_support.txt")
        if os.path.exists(triple_support_path):
            with open(triple_support_path, 'r', encoding='utf-8') as f:
                result["stage4_triple_support"] = f.read()
        else:
            result["stage4_triple_support"] = None
        
        return result
    except Exception as e:
        import traceback
        error_detail = f"获取Stage4 prompts失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/evaluation-prompts")
async def get_evaluation_prompts_api():
    """获取所有评估相关的 prompt 模板（包括元图谱评估、社区质量评估等）"""
    try:
        prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
        
        result = {}
        
        meta_graph_path = os.path.join(prompts_dir, "meta_graph_quality_evaluation.txt")
        if os.path.exists(meta_graph_path):
            with open(meta_graph_path, 'r', encoding='utf-8') as f:
                result["meta_graph_quality_evaluation"] = f.read()
        else:
            result["meta_graph_quality_evaluation"] = None
        
        community_path = os.path.join(prompts_dir, "community_quality_evaluation.txt")
        if os.path.exists(community_path):
            with open(community_path, 'r', encoding='utf-8') as f:
                result["community_quality_evaluation"] = f.read()
        else:
            result["community_quality_evaluation"] = None
        
        return result
    except Exception as e:
        import traceback
        error_detail = f"获取评估prompts失败: {str(e)}\n{traceback.format_exc()}"
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

        # 与前端约定：参数1=实体识别风格，参数2=关系抽取风格（对应两个下拉框）
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


# ==================== 分阶段 Prompt API ====================

class UpdateStagedPromptRequest(BaseModel):
    folder_name: str
    stage_file: str
    content: str

class DeleteStagedPromptRequest(BaseModel):
    folder_name: str
    stage_file: str

@app.get("/api/prompts/staged")
async def list_staged_prompts_api():
    """列出所有分阶段 prompt 文件夹"""
    try:
        staged_prompts = list_staged_prompts()
        return {"staged_prompts": staged_prompts}
    except Exception as e:
        import traceback
        error_detail = f"列出分阶段 prompt 失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/api/prompt/staged/{folder_name}")
async def get_staged_prompt_files_api(folder_name: str):
    """获取分阶段 prompt 文件夹中的文件列表"""
    try:
        files_info = get_staged_prompt_files(folder_name)
        return files_info
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"获取分阶段 prompt 文件列表失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/api/prompt/staged/{folder_name}/{stage_file}")
async def get_staged_prompt_content_api(folder_name: str, stage_file: str):
    """获取分阶段 prompt 文件夹中特定 stage 文件的内容"""
    try:
        content = get_staged_prompt_content(folder_name, stage_file)
        return {
            "folder_name": folder_name,
            "stage_file": stage_file,
            "content": content
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"获取分阶段 prompt 内容失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/prompt/staged/update")
async def update_staged_prompt_api(request: UpdateStagedPromptRequest):
    """更新分阶段 prompt 文件夹中特定 stage 文件的内容"""
    try:
        update_staged_prompt_content(request.folder_name, request.stage_file, request.content)
        return {"message": f"Prompt '{request.folder_name}/{request.stage_file}' 更新成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"更新分阶段 prompt 失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/prompt/staged/delete")
async def delete_staged_prompt_api(request: DeleteStagedPromptRequest):
    """删除分阶段 prompt 文件夹中的 stage 文件"""
    try:
        delete_staged_prompt_file(request.folder_name, request.stage_file)
        return {"message": f"文件 '{request.stage_file}' 删除成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"删除分阶段 prompt 文件失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)



# ==================== 分阶段 Prompt 文件夹管理 API ====================

class CreateStagedFolderRequest(BaseModel):
    folder_name: str
    source_folder: str = None

class DeleteStagedFolderRequest(BaseModel):
    folder_name: str

@app.post("/api/prompt/staged/folder/create")
async def create_staged_folder_api(request: CreateStagedFolderRequest):
    """创建分阶段 prompt 文件夹"""
    try:
        create_staged_prompt_folder(request.folder_name, request.source_folder)
        return {"message": f"Staged prompt 文件夹 '{request.folder_name}' 创建成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"创建 staged prompt 文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/api/prompt/staged/folder/delete")
async def delete_staged_folder_api(request: DeleteStagedFolderRequest):
    """删除分阶段 prompt 文件夹"""
    try:
        delete_staged_prompt_folder(request.folder_name)
        return {"message": f"Staged prompt 文件夹 '{request.folder_name}' 删除成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"删除 staged prompt 文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

# ==================== Corpus 的增删查 API ====================

@app.get("/api/corpus")
async def list_corpus_api():
    """列出所有可用的corpus，同时兼容单级目录和两级目录结构"""
    try:
        result = list_corpus_items()
        return result
    except Exception as e:
        import traceback
        error_detail = f"列出corpus失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/corpus/folders-for-generation")
async def list_corpus_folders_api():
    """
    列出所有可用于生成图谱的文件夹（仅文件夹级别，包含单级语料文件夹和主题文件夹）
    """
    try:
        result = list_corpus_items()
        folders = []
        for item in result.get("items", []):
            folders.append({
                "name": item["id"],
                "type": item["type"]  # direct, folder, mixed
            })
        return {"folders": folders}
    except Exception as e:
        import traceback
        error_detail = f"列出文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/corpus/{corpus_name:path}")
async def get_corpus_api(corpus_name: str):
    """获取指定corpus的内容，支持路径形式如 '主题/子文件夹'"""
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


# ==================== Corpus 文件夹管理 API ====================

class CreateFolderRequest(BaseModel):
    parent_path: str = ""
    folder_name: str


@app.post("/api/corpus/folder/create")
async def create_corpus_folder_api(request: CreateFolderRequest):
    """在 input 目录下创建新文件夹"""
    try:
        target = create_folder(request.parent_path, request.folder_name)
        return {"message": f"文件夹 '{request.folder_name}' 创建成功", "path": target}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"创建文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class RenameFolderRequest(BaseModel):
    old_path: str
    new_name: str


@app.post("/api/corpus/folder/rename")
async def rename_corpus_folder_api(request: RenameFolderRequest):
    """重命名 input 目录下的文件夹"""
    try:
        new_path = rename_folder(request.old_path, request.new_name)
        return {"message": f"文件夹已重命名为 '{request.new_name}'", "path": new_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileExistsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"重命名文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


class DeleteFolderRequest(BaseModel):
    folder_path: str


@app.post("/api/corpus/folder/delete")
async def delete_corpus_folder_api(request: DeleteFolderRequest):
    """删除 input 目录下的文件夹"""
    try:
        delete_folder(request.folder_path)
        return {"message": f"文件夹 '{request.folder_path}' 删除成功"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"删除文件夹失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


# ==================== 混合模式生成图谱 API ====================

class TunePromptMixedRequest(BaseModel):
    dataset_name: str
    session_id: str
    stage1_style: Optional[str] = None
    stage2_style: Optional[str] = None
    user_design_stage1_style: Optional[str] = None
    user_design_stage2_style: Optional[str] = None
    schema_name: Optional[str] = None  # Schema 文件名称
    prompt_folder: Optional[str] = None  # Prompt 文件夹或文件名称
    fast_mode: Optional[bool] = False  # 快速模式：关闭验证和评估
    simple_output: Optional[bool] = False  # 兼容旧前端
    output_mode: Optional[str] = "balanced"  # 输出模式：full(完整) | balanced(均衡) | simple(精简)
    output_name: Optional[str] = None  # 自定义输出目录名称


@app.post("/api/tune_prompt_mixed")
async def tune_prompt_mixed_api(request: TunePromptMixedRequest):
    """
    使用 run_end2end_pipeline 生成图谱（后台任务模式）
    立即返回任务状态，前端通过 /api/generation_status/{session_id} 轮询结果
    """
    # 检查是否达到最大并发生成数
    if not generation_semaphore.acquire(blocking=False):
        # 获取当前正在运行的任务信息
        running_tasks = []
        for sid, info in active_generations.items():
            elapsed = (datetime.now() - info['start_time']).seconds
            running_tasks.append(f"{sid}（已运行 {elapsed // 60} 分 {elapsed % 60} 秒）")
        running_info = "；".join(running_tasks) if running_tasks else ""
        raise HTTPException(
            status_code=409,
            detail=f"当前已达到最大并发生成数量（{MAX_CONCURRENT_GENERATIONS}），正在运行的任务: {running_info}，请稍后再试"
        )

    try:
        # 收集文件夹内的语料文件（支持单级语料文件夹和主题文件夹）
        try:
            corpus_file = collect_corpus_from_folder(request.dataset_name)
        except Exception as e:
            generation_semaphore.release()
            raise HTTPException(
                status_code=400,
                detail=f"数据集 '{request.dataset_name}' 处理失败: {str(e)}"
            )

        # 加载配置文件并修改 corpus_path
        try:
            import yaml
        except ImportError:
            generation_semaphore.release()
            raise HTTPException(
                status_code=500,
                detail="需要安装 PyYAML 才能运行 pipeline: pip install pyyaml"
            )

        # 根据参数选择配置文件（优先使用 output_mode，兼容旧的 simple_output）
        mode = request.output_mode or ("simple" if request.simple_output else "full")
        if mode == "simple":
            config_path = SIMPLE_PIPELINE_CONFIG_PATH
        elif mode == "balanced":
            config_path = BALANCED_PIPELINE_CONFIG_PATH
        else:
            config_path = PIPELINE_CONFIG_PATH
        logger.info(f"使用配置文件: {config_path} (output_mode={mode})")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 动态设置 corpus_path（基于用户选择的数据集）
        config['dataset']['name'] = request.dataset_name
        config['dataset']['corpus_path'] = corpus_file

        # 动态生成输出目录
        time_now = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 如果提供了自定义输出名称，则使用该名称；否则使用时间戳
        if request.output_name:
            import re
            safe_output_name = re.sub(r'[\\/*?:"<>|]', "_", request.output_name).strip()
            if not safe_output_name:
                safe_output_name = time_now
            base_output_dir = os.path.join(OUTPUT_DIR, request.dataset_name, safe_output_name)
        else:
            base_output_dir = os.path.join(OUTPUT_DIR, request.dataset_name, time_now)
        
        output_dir = get_unique_output_dir(base_output_dir)

        # 修改配置
        config['output'] = config.get('output', {})
        config['output']['output_dir'] = output_dir

        # 处理 Schema 选择
        if request.schema_name:
            config['schema']['name'] = request.schema_name
            logger.info(f"使用 Schema: {request.schema_name}")

        # 处理 Prompt 选择（支持文件夹和单文件）
        if request.prompt_folder:
            prompt_path = os.path.join(PROJECT_DIR, "prompts", request.prompt_folder)

            # 检查是否是分阶段 prompt 文件夹
            if os.path.isdir(prompt_path):
                stage1_path = os.path.join(prompt_path, "stage1_entity_recognition.txt")
                stage2_path = os.path.join(prompt_path, "stage2_relation_extraction.txt")
                stage3_path = os.path.join(prompt_path, "stage3_attribute_extraction.txt")

                if os.path.exists(stage1_path) and os.path.exists(stage2_path):
                    config.setdefault('extraction', {})['prompt_paths'] = {}
                    config['extraction']['prompt_paths']['stage1'] = stage1_path
                    config['extraction']['prompt_paths']['stage2'] = stage2_path
                    if os.path.exists(stage3_path):
                        config['extraction']['prompt_paths']['stage3'] = stage3_path
                    logger.info(f"使用分阶段 Prompt 文件夹: {request.prompt_folder}")
                else:
                    logger.warning(f"Prompt 文件夹 {request.prompt_folder} 缺少 stage1/stage2 文件")
            else:
                prompt_file_path = f"{prompt_path}.txt"
                if os.path.exists(prompt_file_path):
                    with open(prompt_file_path, 'r', encoding='utf-8') as f:
                        prompt_content = f.read()
                    config['extraction']['use_staged_extraction'] = False
                    config['extraction']['prompt_content'] = prompt_content
                    logger.info(f"使用单文件 Prompt: {request.prompt_folder}.txt")
                else:
                    logger.warning(f"Prompt 文件不存在: {prompt_file_path}")

        # 快速模式：关闭耗时功能
        if request.fast_mode:
            config['extraction']['use_stage4_validation'] = False
            config['extraction']['save_stage_outputs'] = False
            config['deduplication']['enable_cid'] = False
            config['meta_graph']['enable'] = False
            config['meta_graph_evaluation']['enable'] = False
            config['community_evaluation']['enable'] = False
            logger.info("启用快速模式：已关闭 Stage4 验证、元图谱、CID 查询和社区评估")

        need_examples = (
            request.stage1_style is not None
            or request.stage2_style is not None
            or request.user_design_stage1_style is not None
            or request.user_design_stage2_style is not None
        )
        if need_examples:
            config.setdefault('extraction', {})['examples'] = {}
            if request.stage1_style is not None:
                config['extraction']['examples']['stage1_style'] = request.stage1_style
            if request.stage2_style is not None:
                config['extraction']['examples']['stage2_style'] = request.stage2_style
            if request.user_design_stage1_style is not None and request.user_design_stage1_style.strip():
                try:
                    import json
                    json.loads(request.user_design_stage1_style)
                    config['extraction']['examples']['user_design_stage1_style'] = request.user_design_stage1_style.strip()
                except json.JSONDecodeError as e:
                    generation_semaphore.release()
                    raise HTTPException(
                        status_code=400,
                        detail=f"user_design_stage1_style JSON 解析失败: {str(e)}，请检查格式是否正确"
                    )
            if request.user_design_stage2_style is not None and request.user_design_stage2_style.strip():
                try:
                    import json
                    json.loads(request.user_design_stage2_style)
                    config['extraction']['examples']['user_design_stage2_style'] = request.user_design_stage2_style.strip()
                except json.JSONDecodeError as e:
                    generation_semaphore.release()
                    raise HTTPException(
                        status_code=400,
                        detail=f"user_design_stage2_style JSON 解析失败: {str(e)}，请检查格式是否正确"
                    )
            logger.info(
                f"设置 examples: stage1_style={request.stage1_style}, stage2_style={request.stage2_style}, "
                f"user_design_stage1={'yes' if request.user_design_stage1_style and request.user_design_stage1_style.strip() else 'no'}, "
                f"user_design_stage2={'yes' if request.user_design_stage2_style and request.user_design_stage2_style.strip() else 'no'}"
            )

        # 注册任务到活跃列表
        active_generations[request.session_id] = {
            'start_time': datetime.now(),
            'cancelled': False,
            'dataset_name': request.dataset_name
        }

        logger.info(f"开始运行 pipeline: dataset={request.dataset_name}, output_dir={output_dir}")

        # 从已构建的 config 字典中提取各配置段，直接传给 run_end2end_pipeline
        pipeline_dataset_cfg = config.get("dataset", {})
        pipeline_schema_cfg = config.get("schema", {})
        pipeline_extraction_cfg = config.get("extraction", {})
        pipeline_dedup_cfg = config.get("deduplication", {})
        pipeline_output_cfg = config.get("output", {})
        pipeline_meta_graph_cfg = config.get("meta_graph", {})
        pipeline_meta_graph_evaluation_cfg = config.get("meta_graph_evaluation", {})
        pipeline_community_clustering_cfg = config.get("community_clustering", {})
        pipeline_community_evaluation_cfg = config.get("community_evaluation", {})

        # 设置日志级别（如果有配置）
        pipeline_logging_cfg = config.get("logging", {})
        if pipeline_logging_cfg:
            from run_end2end_pipeline import _setup_logging_from_config
            _setup_logging_from_config(pipeline_logging_cfg)

        # 在后台线程中运行 pipeline，不阻塞事件循环
        def _run_pipeline_background():
            try:
                # 直接调用 run_end2end_pipeline，避免 sys.argv 全局变量竞态条件
                run_pipeline_func(
                    dataset_cfg=pipeline_dataset_cfg,
                    schema_cfg=pipeline_schema_cfg,
                    extraction_cfg=pipeline_extraction_cfg,
                    dedup_cfg=pipeline_dedup_cfg,
                    output_cfg=pipeline_output_cfg,
                    meta_graph_cfg=pipeline_meta_graph_cfg,
                    meta_graph_evaluation_cfg=pipeline_meta_graph_evaluation_cfg,
                    community_clustering_cfg=pipeline_community_clustering_cfg,
                    community_evaluation_cfg=pipeline_community_evaluation_cfg,
                )

                # Pipeline 完成，查找生成的图谱文件
                graph_files = []
                if os.path.exists(output_dir):
                    for f in os.listdir(output_dir):
                        if f.endswith(".json") and ("deduplicated" in f or "meta" in f):
                            graph_files.append(f)

                # 存储成功结果
                generation_results[request.session_id] = {
                    "status": "completed",
                    "message": "图谱生成成功",
                    "output_dir": output_dir,
                    "graph_files": graph_files,
                    "time_now": time_now,
                    "session_id": request.session_id
                }
                logger.info(f"Pipeline 完成: session_id={request.session_id}, graph_files={graph_files}")

            except Exception as e:
                import traceback
                error_detail = f"生成图谱失败: {str(e)}\n{traceback.format_exc()}"
                logger.error(f"Pipeline 失败: session_id={request.session_id}, error={error_detail}")
                # 清理不完整的输出目录，避免留下无图谱数据的目录影响后续查找
                if os.path.exists(output_dir):
                    has_graph = any(
                        fname.startswith("graph_") or fname.startswith("multi_stage")
                        for fname in os.listdir(output_dir)
                        if fname.endswith(".json")
                    )
                    if not has_graph:
                        import shutil
                        shutil.rmtree(output_dir, ignore_errors=True)
                        logger.info(f"已清理不完整的输出目录: {output_dir}")

                # 存储失败结果
                generation_results[request.session_id] = {
                    "status": "failed",
                    "error": error_detail,
                    "session_id": request.session_id
                }
            finally:
                # 从活跃列表中移除任务
                if request.session_id in active_generations:
                    del active_generations[request.session_id]
                # 释放信号量
                generation_semaphore.release()

        # 启动后台线程
        bg_thread = threading.Thread(target=_run_pipeline_background, daemon=True)
        bg_thread.start()

        # 立即返回，不等待 pipeline 完成
        return {
            "message": "图谱生成任务已启动",
            "status": "running",
            "session_id": request.session_id,
            "time_now": time_now,
            "output_dir": output_dir
        }

    except HTTPException:
        raise
    except Exception as e:
        # 确保异常时也释放信号量
        generation_semaphore.release()
        import traceback
        error_detail = f"启动生成任务失败: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/generation_status/{session_id}")
async def get_generation_status(session_id: str):
    """
    查询指定 session_id 的图谱生成任务状态
    返回: running / completed / failed / not_found
    """
    # 检查是否正在运行
    if session_id in active_generations:
        info = active_generations[session_id]
        elapsed = (datetime.now() - info['start_time']).seconds
        return {
            "status": "running",
            "session_id": session_id,
            "elapsed_seconds": elapsed,
            "cancelled": info['cancelled'],
            "dataset_name": info.get('dataset_name', '')
        }

    # 检查是否已有结果
    if session_id in generation_results:
        result = generation_results[session_id]
        return result

    return {"status": "not_found", "session_id": session_id}


class CancelGenerationRequest(BaseModel):
    session_id: str


@app.post("/api/cancel_generation")
async def cancel_generation_api(request: CancelGenerationRequest):
    """
    取消正在进行的图谱生成任务
    注意：这只能取消后续未开始的 chunk 处理，已经在进行的 LLM 调用无法中断
    """
    session_id = request.session_id
    if session_id in active_generations:
        active_generations[session_id]['cancelled'] = True
        return {"message": f"已标记取消任务: {session_id}", "status": "cancelling"}
    else:
        return {"message": f"未找到活跃任务: {session_id}", "status": "not_found"}


@app.get("/api/active_generations")
async def get_active_generations():
    """获取当前正在进行的生成任务列表"""
    return {
        "active_generations": [
            {
                "session_id": sid,
                "start_time": info['start_time'].isoformat(),
                "cancelled": info['cancelled']
            }
            for sid, info in active_generations.items()
        ],
        "count": len(active_generations)
    }


@app.post("/api/force_stop")
async def force_stop_api():
    """
    强制停止服务（用于中断长时间运行的任务）
    警告：这会立即终止所有正在进行的生成任务
    """
    import signal
    import threading
    
    def shutdown():
        import time
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
    
    # 在后台线程中延迟关闭服务
    threading.Thread(target=shutdown, daemon=True).start()
    
    return {"message": "服务将在 1 秒后停止，请稍后刷新页面"}



# ==================== 图谱后处理 API ====================

class MergeNodesRequest(BaseModel):
    dataset_name: str
    timestamp: Optional[str] = None
    filename: Optional[str] = None
    target_name: str
    source_names: List[str]
    output_suffix: str = "_merged"


class AddRelationRequest(BaseModel):
    dataset_name: str
    timestamp: Optional[str] = None
    filename: Optional[str] = None
    start_node: Dict[str, Any]
    end_node: Dict[str, Any]
    relation: str
    source: str = "manual addition"
    evidence: str = "user manually added"
    output_suffix: str = "_edited"


@app.post("/api/graph/{dataset_name}/merge-nodes")
async def merge_nodes_api(dataset_name: str, request: MergeNodesRequest):
    """
    手动合并节点：将 source_names 的所有出现替换为 target_name
    """
    try:
        graph_path = find_graph_file(
            dataset_name,
            graph_type="pipeline",
            timestamp=request.timestamp,
            filename=request.filename,
        )
        if not graph_path:
            raise HTTPException(status_code=404, detail="未找到图谱文件")

        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        if not isinstance(graph_data, list):
            raise HTTPException(status_code=400, detail="图谱格式错误，期望为三元组列表")

        source_names_set = set(request.source_names)
        replaced_count = 0
        merged_duplicates = 0

        # 第一轮：替换节点名称
        new_triples = []
        for triple in graph_data:
            modified = False
            for node_key in ["start_node", "end_node"]:
                node = triple.get(node_key, {})
                props = node.get("properties", {})
                name = props.get("name", "")
                if isinstance(name, list):
                    if any(n in source_names_set for n in name):
                        original_names = [n for n in name if n != request.target_name]
                        props["name"] = request.target_name
                        if original_names:
                            existing = props.get("merged_from", [])
                            if isinstance(existing, str):
                                existing = [existing]
                            props["merged_from"] = list(dict.fromkeys(existing + original_names))
                        modified = True
                elif name in source_names_set:
                    props["name"] = request.target_name
                    existing = props.get("merged_from", [])
                    if isinstance(existing, str):
                        existing = [existing]
                    props["merged_from"] = list(dict.fromkeys(existing + [name]))
                    modified = True
            if modified:
                replaced_count += 1
            new_triples.append(triple)

        # 第二轮：合并重复三元组
        seen = {}
        deduped_triples = []
        for triple in new_triples:
            start_name = triple.get("start_node", {}).get("properties", {}).get("name", "")
            end_name = triple.get("end_node", {}).get("properties", {}).get("name", "")
            relation = triple.get("relation", "")
            # name 可能是列表（去重后），转换为 tuple 以便作为 dict key
            if isinstance(start_name, list):
                start_name = tuple(start_name)
            if isinstance(end_name, list):
                end_name = tuple(end_name)
            key = (start_name, relation, end_name)
            if key in seen:
                existing = seen[key]
                existing_cids = existing.get("chunk_id", [])
                if isinstance(existing_cids, str):
                    existing_cids = [existing_cids]
                new_cids = triple.get("chunk_id", [])
                if isinstance(new_cids, str):
                    new_cids = [new_cids]
                merged_cids = list(dict.fromkeys([*(existing_cids or []), *(new_cids or [])]))
                existing["chunk_id"] = merged_cids
                merged_duplicates += 1
            else:
                seen[key] = triple
                deduped_triples.append(triple)

        # 默认覆盖原文件，如果指定了后缀则生成新文件
        output_suffix = request.output_suffix.strip()
        if output_suffix:
            base_path = os.path.splitext(graph_path)[0]
            output_path = f"{base_path}{output_suffix}.json"
        else:
            output_path = graph_path
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(deduped_triples, f, ensure_ascii=False, indent=2)

        logger.info(
            f"节点合并完成: {graph_path} -> {output_path}, "
            f"替换了 {replaced_count} 个三元组, 合并了 {merged_duplicates} 个重复"
        )

        return {
            "message": "节点合并成功",
            "input_file": graph_path,
            "output_file": output_path,
            "replaced_triples": replaced_count,
            "merged_duplicates": merged_duplicates,
            "total_triples_after": len(deduped_triples),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"节点合并失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"节点合并失败: {str(e)}")


@app.post("/api/graph/{dataset_name}/add-relation")
async def add_relation_api(dataset_name: str, request: AddRelationRequest):
    """
    手动增加关系：向图谱中添加新的三元组
    """
    try:
        graph_path = find_graph_file(
            dataset_name,
            graph_type="pipeline",
            timestamp=request.timestamp,
            filename=request.filename,
        )
        if not graph_path:
            raise HTTPException(status_code=404, detail="未找到图谱文件")

        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        if not isinstance(graph_data, list):
            raise HTTPException(status_code=400, detail="图谱格式错误，期望为三元组列表")

        new_triple = {
            "start_node": {
                "label": "entity",
                "properties": request.start_node,
            },
            "relation": request.relation,
            "end_node": {
                "label": "entity",
                "properties": request.end_node,
            },
            "source": request.source,
            "evidence": request.evidence,
            "chunk_id": ["manual"],
        }

        # 检查是否已存在相同三元组
        start_name = request.start_node.get("name", "")
        end_name = request.end_node.get("name", "")
        for triple in graph_data:
            s_name = triple.get("start_node", {}).get("properties", {}).get("name", "")
            e_name = triple.get("end_node", {}).get("properties", {}).get("name", "")
            rel = triple.get("relation", "")
            # name 可能是列表（去重后），统一比较方式
            def _norm_name(n):
                return tuple(n) if isinstance(n, list) else n
            if _norm_name(s_name) == _norm_name(start_name) and _norm_name(e_name) == _norm_name(end_name) and rel == request.relation:
                raise HTTPException(
                    status_code=409,
                    detail=f"三元组已存在: {start_name} --{request.relation}--> {end_name}"
                )

        graph_data.append(new_triple)

        # 默认覆盖原文件，如果指定了后缀则生成新文件
        output_suffix = request.output_suffix.strip()
        if output_suffix:
            base_path = os.path.splitext(graph_path)[0]
            output_path = f"{base_path}{output_suffix}.json"
        else:
            output_path = graph_path
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        logger.info(f"关系添加完成: {graph_path} -> {output_path}, 新增 1 个三元组")

        return {
            "message": "关系添加成功",
            "input_file": graph_path,
            "output_file": output_path,
            "added_triples": 1,
            "total_triples_after": len(graph_data),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"关系添加失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"关系添加失败: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "vis_graph_v1:app",
        host="0.0.0.0",
        port=6998,
        workers=1,  # 使用单进程模式
        timeout_keep_alive=600,
        timeout_graceful_shutdown=120,
        limit_concurrency=100,  # 增加并发数
    )
