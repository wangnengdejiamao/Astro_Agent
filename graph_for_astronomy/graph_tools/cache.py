
from __future__ import annotations
import json
import os
import pickle
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from utils.logger import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
# load_dotenv("/Dspace/pku-projects/dev-projects/lab-agents/agents/.env")
load_dotenv()
graphrag_dir = os.environ.get("GRAPHRAG_DIR")
if not graphrag_dir:
    raise ValueError("GRAPHRAG_DIR 环境变量未设置")

from .graph_util import GraphAnalyzer  # noqa: E402


class GraphIndexCache:
    """
    缓存 schema_type 倒排索引和节点数据，避免重复读取大型图文件。
    """

    def __init__(self):
        cache_dir_env = os.environ.get("GRAPH_INDEX_CACHE_DIR") or os.environ.get("GRAPH_CACHE_ROOT")
        self.cache_dir = cache_dir_env or os.path.join(current_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._schema_index: Dict[str, Dict[str, List[str]]] = {}
        self._node_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def ensure_loaded(self, dataset_name: str, force_analyzer: bool = False, profile: bool = False) -> GraphAnalyzer | None:
        if dataset_name in self._schema_index and dataset_name in self._node_cache:
            return None if not force_analyzer else self._get_analyzer(dataset_name)

        if self._is_cache_valid(dataset_name):
            if self._load_cache(dataset_name, profile=profile):
                return None if not force_analyzer else self._get_analyzer(dataset_name)

        analyzer = self._get_analyzer(dataset_name)
        self._build_index(dataset_name, analyzer)
        return analyzer

    def get_nodes_by_schema(self, dataset_name: str, schema_type: str) -> List[Dict[str, Any]]:
        node_ids = self._schema_index.get(dataset_name, {}).get(schema_type, [])
        node_data = self._node_cache.get(dataset_name, {})
        return [node_data[node_id] for node_id in node_ids if node_id in node_data]

    def _get_analyzer(self, dataset_name: str) -> GraphAnalyzer:
        graph_path = os.path.join(graphrag_dir, f"{dataset_name}_concise.json")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"图谱文件不存在: {graph_path}")
        return GraphAnalyzer(graph_path)

    def _build_index(self, dataset_name: str, analyzer: GraphAnalyzer):
        schema_index = defaultdict(list)
        node_cache: Dict[str, Dict[str, Any]] = {}

        for node_id, node_data in analyzer.graph.nodes(data=True):
            props = node_data.get("properties", {})
            schema_type = props.get("schema_type", "")
            if schema_type:
                schema_index[schema_type].append(node_id)
            node_cache[node_id] = {
                "node_id": node_id,
                "name": props.get("name", ""),
                "properties": props,
                "label": node_data.get("label", ""),
                "schema_type": schema_type,
            }

        self._schema_index[dataset_name] = dict(schema_index)
        self._node_cache[dataset_name] = node_cache
        self._save_cache(dataset_name)

    def _is_cache_valid(self, dataset_name: str) -> bool:
        schema_path, node_path = self._cache_paths(dataset_name)
        if not os.path.exists(schema_path) or not os.path.exists(node_path):
            return False

        graph_path = os.path.join(graphrag_dir, f"{dataset_name}_concise.json")
        if not os.path.exists(graph_path):
            return False

        try:
            graph_mtime = os.path.getmtime(graph_path)
            cache_mtime = min(os.path.getmtime(schema_path), os.path.getmtime(node_path))
            return cache_mtime >= graph_mtime
        except OSError:  # pragma: no cover
            return False

    def _load_cache(self, dataset_name: str, profile: bool = False) -> bool:
        schema_path, node_path = self._cache_paths(dataset_name)
        if not os.path.exists(schema_path) or not os.path.exists(node_path):
            return False

        try:
            start = time.time()
            with open(schema_path, "r", encoding="utf-8") as schema_file:
                self._schema_index[dataset_name] = json.load(schema_file)
            mid = time.time()
            with open(node_path, "rb") as node_file:
                self._node_cache[dataset_name] = pickle.load(node_file)
            total = time.time() - start
            if profile:
                logger.debug(
                    "加载缓存 %s 成功 (schema: %.4fs, node: %.4fs, total: %.4fs)",
                    dataset_name,
                    mid - start,
                    total - (mid - start),
                    total,
                )
            return True
        except Exception as exc:  # pragma: no cover
            logger.warning("加载缓存失败 (%s): %s", dataset_name, exc)
            self._schema_index.pop(dataset_name, None)
            self._node_cache.pop(dataset_name, None)
            return False

    def _save_cache(self, dataset_name: str):
        schema_path, node_path = self._cache_paths(dataset_name)
        try:
            with open(schema_path, "w", encoding="utf-8") as schema_file:
                json.dump(self._schema_index[dataset_name], schema_file, ensure_ascii=False, indent=2)
            with open(node_path, "wb") as node_file:
                pickle.dump(self._node_cache[dataset_name], node_file)
        except Exception as exc:  # pragma: no cover
            logger.warning("保存缓存失败 (%s): %s", dataset_name, exc)

    def _cache_paths(self, dataset_name: str) -> Tuple[str, str]:
        schema_path = os.path.join(self.cache_dir, f"{dataset_name}_schema_index.json")
        node_path = os.path.join(self.cache_dir, f"{dataset_name}_node_data.pkl")
        return schema_path, node_path


__all__ = ["GraphIndexCache"]

