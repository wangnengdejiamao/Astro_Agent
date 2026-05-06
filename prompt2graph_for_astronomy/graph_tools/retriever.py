from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Set

import networkx as nx

from dotenv import load_dotenv
# load_dotenv("/Dspace/pku-projects/dev-projects/lab-agents/agents/.env")
# graphrag_dir = os.environ.get("GRAPHRAG_DIR")
# if not graphrag_dir:
#     raise ValueError("GRAPHRAG_DIR 环境变量未设置")

from .graph_util import GraphAnalyzer 
from .utils.logger import logger


# 定义缓存目录（允许通过环境变量覆盖）   # TODO
_cache_dir_env = os.environ.get("GRAPH_RETRIEVER_CACHE_DIR") or os.environ.get("GRAPH_CACHE_ROOT")
project_dir = os.environ.get("PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not _cache_dir_env or _cache_dir_env == "1":
    # 添加默认缓存目录
    _cache_dir_env = os.path.join(project_dir, "cache")
    logger.warning(f"GRAPH_RETRIEVER_CACHE_DIR 或 GRAPH_CACHE_ROOT 环境变量未设置或无效，使用默认缓存目录: {_cache_dir_env}")
else:
    # 确保使用绝对路径
    if not os.path.isabs(_cache_dir_env):
        _cache_dir_env = os.path.join(project_dir, _cache_dir_env)
        logger.info(f"将相对缓存目录路径转换为绝对路径: {_cache_dir_env}")

CACHE_DIR = _cache_dir_env
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"GraphRetriever 缓存目录: {CACHE_DIR}")

@dataclass
class GraphRetriever:
    """
    面向对象封装的图检索工具，负责子图抽取与 chunk 文本缓存。
    """

    cache_dir: str = CACHE_DIR

    def __post_init__(self):
        # 实例级缓存，避免多进程间的冲突
        self._analyzer_cache: Dict[str, GraphAnalyzer] = {}
        self._chunk_mapping_cache: Dict[str, Dict[str, str]] = {}
        self._chunk_full_mapping_cache: Dict[str, Dict[str, str]] = {}

    def get_subgraph_by_start_node(self, start_node_name: str, graph_path: str, depth: int = 2) -> List[Dict]:
        analyzer = self._get_analyzer(graph_path)
        subgraph = analyzer.find_subgraph_by_node_name(start_node_name, depth=depth)
        return self._convert_graph_to_triples(subgraph)

    def get_subgraph_by_start_and_end(
        self,
        start_node_name: str,
        end_node_name: str,
        graph_path: str,
        depth: int = 2,
    ) -> List[Dict]:
        analyzer = self._get_analyzer(graph_path)
        subgraph = analyzer.find_subgraph_by_node_name(start_node_name, depth=depth)
        if subgraph.number_of_nodes() == 0:
            logger.warning("未能找到与起始节点匹配的子图")
            return []

        end_ids = self._resolve_node_ids(analyzer, end_node_name, subgraph)
        if not end_ids:
            logger.warning(f"终止节点 '{end_node_name}' 未在子图中找到（包含别名匹配）")
            return []

        start_ids = self._resolve_node_ids(analyzer, start_node_name, subgraph)
        if not start_ids:
            logger.warning(f"起始节点 '{start_node_name}' 未在子图中找到（包含别名匹配）")
            return []
        path_edges = self._collect_path_edges(subgraph, start_ids, end_ids, depth)

        if not path_edges:
            for u, v in subgraph.edges():
                if u in end_ids or v in end_ids:
                    path_edges.add((u, v))

        path_subgraph = nx.MultiDiGraph()
        for node_id in {node for edge in path_edges for node in edge}:
            if node_id in subgraph:
                path_subgraph.add_node(node_id, **subgraph.nodes[node_id])

        for u, v in path_edges:
            if subgraph.has_edge(u, v):
                for key, edge_data in subgraph[u][v].items():
                    path_subgraph.add_edge(u, v, key=key, **edge_data)

        return self._convert_graph_to_triples(path_subgraph)

    def retrieve_chunk_by_id(self, chunk_id: str, chunk_path: str) -> str:
        if not chunk_id or not chunk_path:
            return ""

        mapping = self._chunk_mapping_cache.get(chunk_path)
        if mapping is None:
            mapping = self._load_chunk_mapping(chunk_path)
            self._chunk_mapping_cache[chunk_path] = mapping
        return mapping.get(chunk_id, "")

    def retrieve_chunk_and_title_by_id(
        self, chunk_id: str, chunk_path: str
    ) -> Dict[str, str]:
        """
        根据 chunk_id 检索 chunk 的标题和文本内容。

        Args:
            chunk_id: chunk 的唯一标识符
            chunk_path: chunk 文件路径

        Returns:
            包含 'title' 和 'text' 键的字典，如果未找到则返回空字典
        """
        if not chunk_id or not chunk_path:
            return {"title": "", "text": ""}

        mapping = self._chunk_full_mapping_cache.get(chunk_path)
        if mapping is None:
            mapping = self._load_chunk_full_mapping(chunk_path)
            self._chunk_full_mapping_cache[chunk_path] = mapping
        return mapping.get(chunk_id, {"title": "", "text": ""})

    def _get_analyzer(self, graph_path: str) -> GraphAnalyzer:
        if not graph_path:
            raise ValueError("graph_path 不能为空")
        analyzer = self._analyzer_cache.get(graph_path)
        if analyzer is None:
            analyzer = GraphAnalyzer(graph_path)
            self._analyzer_cache[graph_path] = analyzer
        return analyzer

    @staticmethod
    def _convert_graph_to_triples(graph: nx.MultiDiGraph) -> List[Dict]:
        triples: List[Dict] = []
        for u, v, data in graph.edges(data=True):
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            triple = {
                "start_node": {
                    "label": u_data.get("label", "entity"),
                    "properties": u_data.get("properties", {}),
                },
                "relation": data.get("relation", "related_to"),
                "end_node": {
                    "label": v_data.get("label", "entity"),
                    "properties": v_data.get("properties", {}),
                },
            }
            # 支持新版结构：chunk id在三元组级别
            if "chunk id" in data:
                triple["chunk id"] = data["chunk id"]
            triples.append(triple)
        return triples

    @staticmethod
    def _resolve_node_ids(
        analyzer: GraphAnalyzer,
        node_name: str,
        scoped_graph: nx.MultiDiGraph,
        node_label: Optional[str] = None,
    ) -> Set[str]:
        """
        使用 GraphAnalyzer 的别名索引解析节点名称，并限制在给定子图内。
        """
        cleaned = (node_name or "").strip()
        if not cleaned:
            return set()

        resolved: Set[str] = set()
        # 直接命中节点ID
        if cleaned in scoped_graph:
            resolved.add(cleaned)

        # 借助 GraphAnalyzer 的别名索引
        analyzer_matches = analyzer.get_node_ids_by_name(cleaned, node_label=node_label)
        if analyzer_matches:
            resolved.update(analyzer_matches)

        # 仅保留子图中的节点
        return {node_id for node_id in resolved if node_id in scoped_graph}

    @staticmethod
    def _collect_path_edges(
        graph: nx.MultiDiGraph,
        start_ids: set,
        end_ids: set,
        depth: int,
    ) -> set:
        edges = set()
        for start_id in start_ids:
            for end_id in end_ids:
                try:
                    for path in nx.all_simple_paths(graph, start_id, end_id, cutoff=depth):
                        for i in range(len(path) - 1):
                            edges.add((path[i], path[i + 1]))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return edges

    def _load_chunk_mapping(self, chunk_path: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        cache_file = self._chunk_cache_path(chunk_path)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as cache_f:
                    cached = json.load(cache_f)
                if isinstance(cached, dict):
                    mapping = {str(k): str(v) for k, v in cached.items()}
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                mapping = {}

        if not mapping and os.path.exists(chunk_path):
            mapping = self._parse_chunk_file(chunk_path)
            if mapping:
                try:
                    with open(cache_file, "w", encoding="utf-8") as cache_f:
                        json.dump(mapping, cache_f, ensure_ascii=False)
                except OSError:
                    pass
        return mapping

    @staticmethod
    def _parse_chunk_file(chunk_path: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        with open(chunk_path, "r", encoding="utf-8") as source:
            for raw_line in source:
                line = raw_line.strip()
                if not line or not line.startswith("id:"):
                    continue
                try:
                    id_part, chunk_part = line.split("\tChunk:", 1)
                except ValueError:
                    continue
                chunk_id = id_part.split("id:", 1)[1].strip()
                chunk_str = chunk_part.strip()
                if not chunk_id or not chunk_str:
                    continue
                try:
                    chunk_data = ast.literal_eval(chunk_str)
                except (SyntaxError, ValueError):
                    continue
                text = chunk_data.get("text", "")
                if isinstance(text, str):
                    cleaned = text.replace("\\r\\n", "\n").replace("\r\n", "\n").strip()
                    mapping[chunk_id] = cleaned
        return mapping

    def _load_chunk_full_mapping(self, chunk_path: str) -> Dict[str, Dict[str, str]]:
        """
        加载包含标题和文本的完整 chunk 映射。

        Returns:
            字典，键为 chunk_id，值为包含 'title' 和 'text' 的字典
        """
        mapping: Dict[str, Dict[str, str]] = {}
        cache_file = self._chunk_full_cache_path(chunk_path)

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as cache_f:
                    cached = json.load(cache_f)
                if isinstance(cached, dict):
                    mapping = {
                        str(k): {
                            "title": str(v.get("title", "")),
                            "text": str(v.get("text", "")),
                        }
                        for k, v in cached.items()
                    }
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                mapping = {}

        if not mapping and os.path.exists(chunk_path):
            mapping = self._parse_chunk_file_full(chunk_path)
            if mapping:
                try:
                    with open(cache_file, "w", encoding="utf-8") as cache_f:
                        json.dump(mapping, cache_f, ensure_ascii=False)
                except OSError:
                    pass
        return mapping

    @staticmethod
    def _parse_chunk_file_full(chunk_path: str) -> Dict[str, Dict[str, str]]:
        """
        解析 chunk 文件，提取包含标题和文本的完整信息。

        Returns:
            字典，键为 chunk_id，值为包含 'title' 和 'text' 的字典
        """
        mapping: Dict[str, Dict[str, str]] = {}
        with open(chunk_path, "r", encoding="utf-8") as source:
            for raw_line in source:
                line = raw_line.strip()
                if not line or not line.startswith("id:"):
                    continue
                try:
                    id_part, chunk_part = line.split("\tChunk:", 1)
                except ValueError:
                    continue
                chunk_id = id_part.split("id:", 1)[1].strip()
                chunk_str = chunk_part.strip()
                if not chunk_id or not chunk_str:
                    continue
                try:
                    chunk_data = ast.literal_eval(chunk_str)
                except (SyntaxError, ValueError):
                    continue
                title = chunk_data.get("title", "")
                text = chunk_data.get("text", "")
                if isinstance(title, str) and isinstance(text, str):
                    cleaned_text = text.replace("\\r\\n", "\n").replace("\r\n", "\n").strip()
                    cleaned_title = title.strip()
                    mapping[chunk_id] = {"title": cleaned_title, "text": cleaned_text}
        return mapping

    def _chunk_cache_path(self, chunk_path: str) -> str:
        digest = hashlib.md5(chunk_path.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"chunk_cache_{digest}.json")

    def _chunk_full_cache_path(self, chunk_path: str) -> str:
        digest = hashlib.md5(chunk_path.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"chunk_full_cache_{digest}.json")


__all__ = ["GraphRetriever"]

