from __future__ import annotations

import asyncio
from typing import List, Tuple

from pymilvus import Collection

from .embedding import EmbeddingService
from .milvus import COLLECTION_CONFIGS, CollectionType, MilvusCollectionManager

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility  # noqa: F401

    MILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    MILVUS_AVAILABLE = False

class GraphSimilarityService:
    """
    基于 Milvus 的相似度搜索（schema / community / node / triple）。
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        collection_manager: MilvusCollectionManager,
        search_metric: str,
        index_builder,
    ):
        self.embedding_service = embedding_service
        self.collection_manager = collection_manager
        self.index_builder = index_builder
        self.search_params = {"metric_type": search_metric, "params": {}}

    async def get_similar_schema_names(self, dataset_name: str, query_schema: str, k: int = 1) -> List[Tuple[str, float]]:
        return await self._search(CollectionType.SCHEMA_NAMES, dataset_name, query_schema, k)

    async def get_similar_community_names(self, dataset_name: str, query_community: str, k: int = 1) -> List[Tuple[str, float]]:
        return await self._search(CollectionType.COMMUNITY_NAMES, dataset_name, query_community, k)

    async def get_similar_node_names(self, dataset_name: str, query_node: str, k: int = 1) -> List[Tuple[str, float]]:
        return await self._search(CollectionType.NODE_NAMES, dataset_name, query_node, k)

    async def get_similar_triple_strings(
        self,
        dataset_name: str,
        query_triple_string: str,
        k: int = 1,
        rebuild_collection: bool = False,
    ) -> List[Tuple[str, float]]:
        return await self._search(
            CollectionType.TRIPLE_STRINGS, dataset_name, query_triple_string, k, rebuild_collection=rebuild_collection
        )

    async def _search(
        self,
        collection_type: CollectionType,
        dataset_name: str,
        query_text: str,
        k: int,
        rebuild_collection: bool = False,
    ) -> List[Tuple[str, float]]:
        if not MILVUS_AVAILABLE:
            raise RuntimeError("Milvus 不可用，请安装 pymilvus 并配置连接")
        if not self.embedding_service.available:
            raise RuntimeError("Embedding 模型不可用")

        collection = self.collection_manager.get_collection(collection_type)
        text_field = COLLECTION_CONFIGS[collection_type].text_field

        await self._ensure_dataset_ready(collection, collection_type, dataset_name, rebuild_collection)

        embedding = self.embedding_service.compute_embedding(query_text)
        if embedding is None:
            raise RuntimeError("无法计算查询文本的 embedding")

        # 将同步的 collection.search() 包装在 asyncio.to_thread 中
        results = await asyncio.to_thread(
            collection.search,
            data=[embedding.tolist()],
            anns_field="embedding",
            param=self.search_params,
            limit=k,
            expr=f'dataset_name == "{dataset_name}"',
            output_fields=[text_field],
        )

        similar_items: List[Tuple[str, float]] = []
        if results:
            for hit in results[0]:
                item = hit.entity.get(text_field)
                similarity = max(0.0, min(1.0, hit.distance))
                similar_items.append((item, similarity))
        return similar_items

    async def _ensure_dataset_ready(
        self,
        collection: Collection,
        collection_type: CollectionType,
        dataset_name: str,
        rebuild_collection: bool,
    ):
        need_rebuild = rebuild_collection
        if not rebuild_collection:
            # 将同步的 collection.query() 包装在 asyncio.to_thread 中
            existing = await asyncio.to_thread(
                collection.query,
                expr=f'dataset_name == "{dataset_name}"',
                output_fields=["id"],
                limit=1
            )
            need_rebuild = len(existing) == 0

        if need_rebuild:
            builder = {
                CollectionType.SCHEMA_NAMES: self.index_builder.build_schema_names,
                CollectionType.NODE_NAMES: self.index_builder.build_node_names,
                CollectionType.TRIPLE_STRINGS: self.index_builder.build_triple_strings,
                CollectionType.COMMUNITY_NAMES: self.index_builder.build_community_names,
            }[collection_type]
            # 将同步的 builder 调用包装在 asyncio.to_thread 中
            await asyncio.to_thread(builder, dataset_name, force_rebuild=rebuild_collection)
            await asyncio.to_thread(collection.load)


__all__ = ["GraphSimilarityService"]

