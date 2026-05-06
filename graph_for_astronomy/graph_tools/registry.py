from __future__ import annotations

import os

from .builder import GraphIndexBuilder
from .cache import GraphIndexCache
from .embedding import EmbeddingService
from .milvus import MilvusCollectionManager
from .query import GraphQueryService
from .similarity import GraphSimilarityService

EMBEDDING_SERVICE_HOST = os.getenv("EMBEDDING_SERVICE_HOST", "localhost")
EMBEDDING_SERVICE_PORT = int(os.getenv("EMBEDDING_SERVICE_PORT", "8035"))
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "600"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))
EMBEDDING_MAX_WORKERS = int(os.getenv("EMBEDDING_MAX_WORKERS", "4"))

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
VECTOR_DIM = int(os.getenv("VECTOR_DIMENSION", "2560"))
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE")
MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "128"))


embedding_service = EmbeddingService(
    host=EMBEDDING_SERVICE_HOST,
    port=EMBEDDING_SERVICE_PORT,
    timeout=EMBEDDING_TIMEOUT,
    max_retries=EMBEDDING_MAX_RETRIES,
    batch_size=EMBEDDING_BATCH_SIZE,
    max_workers=EMBEDDING_MAX_WORKERS,
)

collection_manager = MilvusCollectionManager(
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    vector_dim=VECTOR_DIM,
    index_type=MILVUS_INDEX_TYPE,
    metric_type=MILVUS_METRIC_TYPE,
    nlist=MILVUS_NLIST,
)

index_builder = GraphIndexBuilder(embedding_service, collection_manager)
similarity_service = GraphSimilarityService(
    embedding_service=embedding_service,
    collection_manager=collection_manager,
    search_metric=MILVUS_METRIC_TYPE,
    index_builder=index_builder,
)
cache = GraphIndexCache()
query_service = GraphQueryService(cache, similarity_service)


__all__ = [
    "embedding_service",
    "collection_manager",
    "index_builder",
    "similarity_service",
    "cache",
    "query_service",
]

