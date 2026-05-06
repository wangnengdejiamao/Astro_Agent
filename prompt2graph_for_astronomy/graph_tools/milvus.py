from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from utils.logger import logger

try:
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility  # noqa: F401

    MILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    MILVUS_AVAILABLE = False

class CollectionType(str, Enum):
    NODE_NAMES = "node_names"
    TRIPLE_STRINGS = "triple_strings"
    SCHEMA_NAMES = "schema_names"
    COMMUNITY_NAMES = "community_names"


@dataclass(frozen=True)
class CollectionConfig:
    env_name: str
    default_name: str
    text_field: str


COLLECTION_CONFIGS: Dict[CollectionType, CollectionConfig] = {
    CollectionType.NODE_NAMES: CollectionConfig("MILVUS_COLLECTION_NODE_NAMES", "graph_node_names", "node_name"),
    CollectionType.TRIPLE_STRINGS: CollectionConfig(
        "MILVUS_COLLECTION_TRIPLE_STRINGS", "graph_triple_strings", "triple_string"
    ),
    CollectionType.SCHEMA_NAMES: CollectionConfig(
        "MILVUS_COLLECTION_SCHEMA_NAMES", "graph_schema_names", "schema_name"
    ),
    CollectionType.COMMUNITY_NAMES: CollectionConfig(
        "MILVUS_COLLECTION_COMMUNITY_NAMES", "graph_community_names", "community_name"
    ),
}


class MilvusCollectionManager:
    """
    管理 Milvus collection 的创建、连接与数据插入。
    """

    def __init__(
        self,
        host: str,
        port: str,
        vector_dim: int,
        index_type: str,
        metric_type: str,
        nlist: int,
    ):
        self.host = host
        self.port = port
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.metric_type = metric_type
        self.nlist = nlist
        self._connected = False
        self._collection_name_cache: Dict[CollectionType, str] = {}

    def ensure_connection(self):
        if not MILVUS_AVAILABLE:
            raise RuntimeError("Milvus 不可用，请安装 pymilvus")

        if self._connected:
            return

        try:
            connections.connect("default", host=self.host, port=self.port)
            self._connected = True
            logger.info("Milvus 连接成功: %s:%s", self.host, self.port)
        except Exception as exc:  # pragma: no cover
            logger.error("Milvus 连接失败: %s", exc)
            raise

    def rebuild_collections(self, collection_type: str = "all", force: bool = True) -> List[str]:
        if not MILVUS_AVAILABLE:
            raise RuntimeError("Milvus 不可用，无法重建 collection")

        self.ensure_connection()

        if collection_type == "all":
            targets = list(CollectionType)
        else:
            try:
                targets = [CollectionType(collection_type)]
            except ValueError:
                raise ValueError(f"不支持的 collection 类型: {collection_type}")

        rebuilt: List[str] = []
        for target in targets:
            collection_name = self.get_collection_name(target)
            if utility.has_collection(collection_name):
                collection = Collection(name=collection_name)
                if not force:
                    collection.load()
                    if collection.num_entities > 0:
                        logger.warning(
                            "Collection %s 已包含数据，跳过重建（force=False）。", collection_name
                        )
                        continue
                logger.info("删除旧的 collection: %s", collection_name)
                collection.drop()

            logger.info("创建新的 collection: %s", collection_name)
            self._create_collection(target)
            rebuilt.append(collection_name)
        return rebuilt

    def get_collection_name(self, collection_type: CollectionType) -> str:
        if collection_type in self._collection_name_cache:
            return self._collection_name_cache[collection_type]
        config = COLLECTION_CONFIGS[collection_type]
        name = os.getenv(config.env_name, config.default_name)
        self._collection_name_cache[collection_type] = name
        return name

    def get_collection(self, collection_type: CollectionType) -> Collection:
        self.ensure_connection()
        name = self.get_collection_name(collection_type)
        if not utility.has_collection(name):
            self._create_collection(collection_type)
        collection = Collection(name=name)
        collection.load()
        return collection

    def ensure_dataset_vectors(
        self,
        collection_type: CollectionType,
        dataset_name: str,
        build_fn,
        force_rebuild: bool = False,
    ):
        collection = self.get_collection(collection_type)
        if force_rebuild:
            collection.delete(f'dataset_name == "{dataset_name}"')
            collection.flush()
        else:
            existing = collection.query(expr=f'dataset_name == "{dataset_name}"', output_fields=["id"], limit=1)
            if existing:
                return
        build_fn(collection, dataset_name)
        collection.flush()
        collection.load()

    def insert_vectors(
        self,
        collection: Collection,
        dataset_name: str,
        texts: List[str],
        embeddings: List,
    ):
        if not texts or not embeddings or len(texts) != len(embeddings):
            return
        collection.insert(
            [
                [dataset_name] * len(texts),
                texts,
                [vector.tolist() for vector in embeddings],
            ]
        )

    def _create_collection(self, collection_type: CollectionType):
        schema_fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dataset_name", dtype=DataType.VARCHAR, max_length=256),
        ]

        text_field = COLLECTION_CONFIGS[collection_type].text_field
        max_length = 2000 if collection_type != CollectionType.SCHEMA_NAMES else 500
        schema_fields.append(FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=max_length))
        schema_fields.append(FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim))

        schema = CollectionSchema(schema_fields)
        name = self.get_collection_name(collection_type)
        collection = Collection(name=name, schema=schema)

        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {"nlist": self.nlist},
            },
        )
        collection.create_index(field_name="dataset_name", index_params={"index_type": "TRIE"})


__all__ = [
    "CollectionType",
    "CollectionConfig",
    "COLLECTION_CONFIGS",
    "MilvusCollectionManager",
]

