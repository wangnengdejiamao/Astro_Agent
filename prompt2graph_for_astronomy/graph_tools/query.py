from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

from .cache import GraphIndexCache
from .similarity import GraphSimilarityService


class GraphQueryService:
    """
    提供 schema_type 节点查询以及模糊匹配功能。
    """

    def __init__(self, cache: GraphIndexCache, similarity_service: GraphSimilarityService):
        self.cache = cache
        self.similarity_service = similarity_service

    async def get_nodes_by_schema_type(
        self,
        dataset_name: str,
        node_type: str,
        profile: bool = False,
    ) -> Tuple[List[Dict[str, Any]], int]:
        try:
            def ensure():
                return self.cache.ensure_loaded(dataset_name, force_analyzer=False, profile=profile)

            await asyncio.to_thread(ensure)
            nodes = self.cache.get_nodes_by_schema(dataset_name, node_type)
            return nodes, len(nodes)
        except FileNotFoundError as exc:
            return [], 0
        except Exception:  # pragma: no cover
            return [], 0

    async def get_nodes_by_schema_type_fuzzy(
        self,
        dataset_name: str,
        query_schema_type: str,
        profile: bool = False,
        fuzzy: bool = True,
        fuzzy_threshold: float = 0.9,
    ) -> Tuple[List[Dict[str, Any]], int]:
        schema_type = query_schema_type
        if fuzzy:
            # 直接 await 异步方法，不再使用 asyncio.to_thread
            similar = await self.similarity_service.get_similar_schema_names(dataset_name, query_schema_type, 1)
            if similar:
                schema_type, confidence = similar[0]
                if confidence < fuzzy_threshold:
                    raise ValueError(
                        f"最相近的 schema_type '{schema_type}' 置信度为 {confidence:.3f}，低于阈值 {fuzzy_threshold}"
                    )
        return await self.get_nodes_by_schema_type(dataset_name, schema_type, profile=profile)


__all__ = ["GraphQueryService"]

