from __future__ import annotations

import logging
from typing import List, Tuple

from .utils.logger import logger

try:
    from .cache import GraphIndexCache
except Exception as exc:  # optional for local JSON visualization
    logger.warning("GraphIndexCache unavailable: %s", exc)
    GraphIndexCache = None  # type: ignore

try:
    from .path import GraphPathFinder
except Exception as exc:
    logger.warning("GraphPathFinder unavailable: %s", exc)
    GraphPathFinder = None  # type: ignore

try:
    from .query import GraphQueryService
except Exception as exc:
    logger.warning("GraphQueryService unavailable: %s", exc)
    GraphQueryService = None  # type: ignore

try:
    from .registry import (
        cache,
        collection_manager,
        embedding_service,
        index_builder,
        query_service,
        similarity_service,
    )
except Exception as exc:
    logger.warning("Graph registry unavailable: %s", exc)
    cache = collection_manager = embedding_service = index_builder = query_service = similarity_service = None

try:
    from .retriever import GraphRetriever
    _graph_retriever = GraphRetriever()
except Exception as exc:
    logger.warning("GraphRetriever unavailable: %s", exc)
    GraphRetriever = None  # type: ignore
    _graph_retriever = None

try:
    from .similarity import GraphSimilarityService
except Exception as exc:
    logger.warning("GraphSimilarityService unavailable: %s", exc)
    GraphSimilarityService = None  # type: ignore


def _require_service(service, service_name: str):
    if service is None:
        raise RuntimeError(f"{service_name} is unavailable; install optional vector-search dependencies first.")
    return service


async def get_nodes_by_schema_type(dataset_name: str, node_type: str, profile: bool = False):
    return await _require_service(query_service, "query_service").get_nodes_by_schema_type(dataset_name, node_type, profile=profile)


async def get_nodes_by_schema_type_fuzzy(
    dataset_name: str,
    query_schema_type: str,
    profile: bool = False,
    fuzzy: bool = True,
    fuzzy_threshold: float = 0.9,
):
    return await _require_service(query_service, "query_service").get_nodes_by_schema_type_fuzzy(
        dataset_name,
        query_schema_type,
        profile=profile,
        fuzzy=fuzzy,
        fuzzy_threshold=fuzzy_threshold,
    )


async def get_similar_schema_names(dataset_name: str, query_schema: str, k: int = 1) -> List[Tuple[str, float]]:
    return await _require_service(similarity_service, "similarity_service").get_similar_schema_names(dataset_name, query_schema, k=k)


async def get_similar_community_names(dataset_name: str, query_community: str, k: int = 1) -> List[Tuple[str, float]]:
    return await _require_service(similarity_service, "similarity_service").get_similar_community_names(dataset_name, query_community, k=k)


async def get_similar_node_names(dataset_name: str, query_node: str, k: int = 1) -> List[Tuple[str, float]]:
    return await _require_service(similarity_service, "similarity_service").get_similar_node_names(dataset_name, query_node, k=k)


async def get_similar_triple_strings(
    dataset_name: str,
    query_triple_string: str,
    k: int = 1,
    rebuild_collection: bool = False,
) -> List[Tuple[str, float]]:
    return await _require_service(similarity_service, "similarity_service").get_similar_triple_strings(
        dataset_name,
        query_triple_string,
        k=k,
        rebuild_collection=rebuild_collection,
    )


def build_node_names_vector_index(dataset_name: str, force_rebuild: bool = False):
    _require_service(index_builder, "index_builder").build_node_names(dataset_name, force_rebuild=force_rebuild)


def build_triple_strings_vector_index(dataset_name: str, force_rebuild: bool = False):
    _require_service(index_builder, "index_builder").build_triple_strings(dataset_name, force_rebuild=force_rebuild)


def build_schema_names_vector_index(dataset_name: str, force_rebuild: bool = False):
    _require_service(index_builder, "index_builder").build_schema_names(dataset_name, force_rebuild=force_rebuild)


def build_community_names_vector_index(dataset_name: str, force_rebuild: bool = False):
    _require_service(index_builder, "index_builder").build_community_names(dataset_name, force_rebuild=force_rebuild)


def rebuild_collection(collection_type: str = "all", force: bool = True):
    return _require_service(collection_manager, "collection_manager").rebuild_collections(collection_type=collection_type, force=force)


def set_log_level(level: int):
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def get_subgraph_from_graph_by_start_node(start_node_name: str, graph_path: str, depth: int = 2):
    return _require_service(_graph_retriever, "GraphRetriever").get_subgraph_by_start_node(start_node_name, graph_path, depth)


def get_subgraph_from_graph_by_start_and_end(
    start_node_name: str,
    end_node_name: str,
    graph_path: str,
    depth: int = 2,
):
    return _require_service(_graph_retriever, "GraphRetriever").get_subgraph_by_start_and_end(start_node_name, end_node_name, graph_path, depth)


def retrieve_chunk_by_id(chunk_id: str, chunk_path: str):
    return _require_service(_graph_retriever, "GraphRetriever").retrieve_chunk_by_id(chunk_id, chunk_path)


def retrieve_chunk_and_title_by_id(chunk_id: str, chunk_path: str):
    return _require_service(_graph_retriever, "GraphRetriever").retrieve_chunk_and_title_by_id(chunk_id, chunk_path)


__all__ = [
    "EmbeddingService",
    "MilvusCollectionManager",
    "GraphIndexBuilder",
    "GraphIndexCache",
    "GraphSimilarityService",
    "GraphQueryService",
    "GraphRetriever",
    "GraphPathFinder",
    "CollectionType",
    "CollectionConfig",
    "COLLECTION_CONFIGS",
    "embedding_service",
    "collection_manager",
    "index_builder",
    "similarity_service",
    "cache",
    "query_service",
    "get_nodes_by_schema_type",
    "get_nodes_by_schema_type_fuzzy",
    "get_similar_schema_names",
    "get_similar_community_names",
    "get_similar_node_names",
    "get_similar_triple_strings",
    "build_node_names_vector_index",
    "build_triple_strings_vector_index",
    "build_schema_names_vector_index",
    "build_community_names_vector_index",
    "rebuild_collection",
    "set_log_level",
    "get_subgraph_from_graph_by_start_node",
    "get_subgraph_from_graph_by_start_and_end",
    "retrieve_chunk_by_id",
    "retrieve_chunk_and_title_by_id",
]
