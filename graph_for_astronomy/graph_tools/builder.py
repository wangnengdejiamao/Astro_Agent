from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np

from .embedding import EmbeddingService
from .milvus import CollectionType, MilvusCollectionManager
from .utils.logger import logger

from dotenv import load_dotenv
# load_dotenv("/Dspace/pku-projects/dev-projects/lab-agents/agents/.env")
load_dotenv()
project_root = os.environ.get("PROJECT_ROOT")
if not project_root:
    raise ValueError("PROJECT_ROOT 环境变量未设置")

graphrag_dir = os.environ.get("GRAPHRAG_DIR")
if not graphrag_dir:
    raise ValueError("GRAPHRAG_DIR 环境变量未设置")


try:
    from tqdm import tqdm  # noqa: F401

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover
    TQDM_AVAILABLE = False

from .graph_util import GraphAnalyzer  # noqa: E402
from .utils.graph_processor import extract_aliases  # noqa: E402


class GraphIndexBuilder:
    """
    针对不同 collection 构建向量索引。
    """

    def __init__(self, embedding_service: EmbeddingService, collection_manager: MilvusCollectionManager):
        self.embedding_service = embedding_service
        self.collection_manager = collection_manager

    def build_node_names(self, dataset_name: str, force_rebuild: bool = False):
        def builder(collection, dataset: str):
            analyzer = self._load_analyzer(dataset)
            node_names = self._collect_unique_node_names(analyzer)
            self._insert_in_batches(collection, dataset, node_names)

        self.collection_manager.ensure_dataset_vectors(
            CollectionType.NODE_NAMES, dataset_name, builder, force_rebuild=force_rebuild
        )

    def build_triple_strings(self, dataset_name: str, force_rebuild: bool = False):
        def builder(collection, dataset: str):
            analyzer = self._load_analyzer(dataset)
            triples = self._collect_unique_triples(analyzer)
            self._insert_in_batches(collection, dataset, triples)

        self.collection_manager.ensure_dataset_vectors(
            CollectionType.TRIPLE_STRINGS, dataset_name, builder, force_rebuild=force_rebuild
        )

    def build_schema_names(self, dataset_name: str, force_rebuild: bool = False):
        def builder(collection, dataset: str):
            analyzer = self._load_analyzer(dataset)
            schemas = self._collect_unique_schema_types(analyzer)
            self._insert_in_batches(collection, dataset, schemas)

        self.collection_manager.ensure_dataset_vectors(
            CollectionType.SCHEMA_NAMES, dataset_name, builder, force_rebuild=force_rebuild
        )

    def build_community_names(self, dataset_name: str, force_rebuild: bool = False):
        def builder(collection, dataset: str):
            analyzer = self._load_analyzer(dataset)
            communities = self._collect_unique_communities(analyzer)
            self._insert_in_batches(collection, dataset, communities)

        self.collection_manager.ensure_dataset_vectors(
            CollectionType.COMMUNITY_NAMES, dataset_name, builder, force_rebuild=force_rebuild
        )

    def _insert_in_batches(self, collection, dataset_name: str, items: List[str]):
        if not items:
            logger.warning("数据集 %s 无可用数据用于构建向量索引。", dataset_name)
            return

        logger.info("开始构建向量索引: dataset=%s, items=%d", dataset_name, len(items))
        total = len(items)
        batch_size = self.embedding_service.batch_size
        max_workers = self.embedding_service.max_workers
        batches = [items[i : i + batch_size] for i in range(0, total, batch_size)]
        progress = tqdm(total=total, desc=f"构建向量 ({dataset_name})") if TQDM_AVAILABLE else None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(self.embedding_service.compute_embeddings_batch, batch): batch for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                embeddings = future.result()
                valid_items = []
                valid_embeddings = []
                for item, emb in zip(batch, embeddings):
                    if emb is not None:
                        valid_items.append(item)
                        valid_embeddings.append(emb)
                if valid_items:
                    self.collection_manager.insert_vectors(collection, dataset_name, valid_items, valid_embeddings)
                if progress:
                    progress.update(len(batch))

        if progress:
            progress.close()

    @staticmethod
    def _collect_unique_node_names(analyzer: GraphAnalyzer) -> List[str]:
        """
        收集所有唯一的节点名称，为每个别名生成单独的条目。
        例如，如果节点的 name 是 ["A", "B"]，会生成两个条目：A 和 B。
        """
        unique_nodes = set()
        for _, node_data in analyzer.graph.nodes(data=True):
            props = node_data.get("properties", {})
            name = props.get("name", "")
            # 使用 extract_aliases 统一处理，确保每个别名都单独生成 embedding
            aliases = extract_aliases(name)
            for alias in aliases:
                if alias:  # 确保别名非空
                    unique_nodes.add(alias)
        return list(unique_nodes)

    @staticmethod
    def _collect_unique_triples(analyzer: GraphAnalyzer) -> List[str]:
        """
        收集所有唯一的三元组字符串，为每个别名组合生成单独的条目。
        例如，如果 start_node 的 name 是 ["A", "B"]，end_node 的 name 是 "C"，
        relation 是 "is_compatible_with"，会生成两个三元组：
        - "A is_compatible_with C"
        - "B is_compatible_with C"
        """
        triples = set()
        for u, v, edge_data in analyzer.graph.edges(data=True):
            props_u = analyzer.graph.nodes[u].get("properties", {})
            props_v = analyzer.graph.nodes[v].get("properties", {})
            # 使用 extract_aliases 获取所有别名
            aliases_u = extract_aliases(props_u.get("name", ""))
            aliases_v = extract_aliases(props_v.get("name", ""))
            
            if not aliases_u or not aliases_v:
                continue
            
            relation = edge_data.get("relation", "related_to")
            # 为每个别名组合生成单独的三元组
            for alias_u in aliases_u:
                for alias_v in aliases_v:
                    if alias_u and alias_v:  # 确保别名非空
                        triples.add(f"{alias_u} {relation} {alias_v}")
        return list(triples)

    @staticmethod
    def _collect_unique_schema_types(analyzer: GraphAnalyzer) -> List[str]:
        schemas = set()
        for _, node_data in analyzer.graph.nodes(data=True):
            schema_type = node_data.get("properties", {}).get("schema_type")
            if isinstance(schema_type, str):
                schema_type = schema_type.strip()
                if schema_type:
                    schemas.add(schema_type)
            elif schema_type:
                schema_str = str(schema_type).strip()
                if schema_str:
                    schemas.add(schema_str)
        logger.info(
            "数据集 %s 的 schema_type 统计: 总节点 %d, 唯一 schema %d",
            os.path.basename(analyzer.graph_path).replace("_concise.json", ""),
            analyzer.graph.number_of_nodes(),
            len(schemas),
        )
        return list(schemas)

    @staticmethod
    def _collect_unique_communities(analyzer: GraphAnalyzer) -> List[str]:
        communities = set()
        for _, node_data in analyzer.graph.nodes(data=True):
            if node_data.get("label") == "community":
                name = node_data.get("properties", {}).get("name", "")
                if isinstance(name, str) and name:
                    communities.add(name)
        return list(communities)

    @staticmethod
    def _load_analyzer(dataset: str) -> GraphAnalyzer:
        """
        尝试在多个候选目录中定位图文件，兼容不同部署目录结构。
        """
        repo_root = os.path.dirname(project_root)
        candidate_paths = [
            os.path.join(graphrag_dir, "output", "graphs", f"{dataset}_concise.json"),
            os.path.join(project_root, "graph_construction", "output", f"{dataset}_concise.json"),
            os.path.join(project_root, "graph_construction", "output", dataset, f"{dataset}_concise.json"),
            os.path.join(repo_root, "graph_construction", "output", f"{dataset}_concise.json"),
            os.path.join(repo_root, "graph_construction", "output", dataset, f"{dataset}_concise.json"),
        ]

        for path in candidate_paths:
            if os.path.exists(path):
                logger.info("使用图文件: %s", path)
                return GraphAnalyzer(path)

        raise FileNotFoundError(
            f"未找到数据集 {dataset} 的图文件，请确认已在以下路径之一生成: {candidate_paths}"
        )


__all__ = ["GraphIndexBuilder"]

