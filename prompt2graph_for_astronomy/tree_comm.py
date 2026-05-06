from __future__ import annotations

import json
import os
import pickle
import socket
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import json_repair
from sentence_transformers import SentenceTransformer
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm 可选
    tqdm = None
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from utils import call_llm_api
from utils.logger import logger


warnings.filterwarnings('ignore')

try:
    from config import get_config
except ImportError:
    get_config = None


class LocalEmbeddingServiceEncoder:
    """Simple encoder client that talks to the local embedding_service via TCP"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: int | None = None,
        max_batch_size: int | None = None,
        vector_dim: int | None = None,
        max_retries: int | None = None,
    ):
        self.host = host or os.getenv("EMBEDDING_SERVICE_HOST", "localhost")
        self.port = int(port or os.getenv("EMBEDDING_SERVICE_PORT", "8035"))
        self.timeout = int(timeout or os.getenv("EMBEDDING_TIMEOUT", "600"))
        self.max_batch_size = int(max_batch_size or os.getenv("EMBEDDING_MAX_BATCH_SIZE", "8"))
        self.vector_dim = int(vector_dim or os.getenv("VECTOR_DIMENSION", "2560"))
        self.max_retries = int(max_retries or os.getenv("EMBEDDING_MAX_RETRIES", "3"))

    def encode(self, texts, batch_size: int | None = None, convert_to_tensor: bool = False, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = batch_size or self.max_batch_size
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._request_embeddings_with_retry(batch)
            all_embeddings.extend(embeddings)

        array = np.array(all_embeddings, dtype=np.float32)
        if convert_to_tensor:
            return torch.tensor(array)
        return array

    def _request_embeddings_with_retry(self, texts):
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._request_embeddings(texts)
            except Exception as exc:
                last_error = exc
                logger.warning(f"[LocalEmbeddingService] 请求失败 (尝试 {attempt}/{self.max_retries}): {exc}")
                time.sleep(1)
        raise RuntimeError(f"无法从本地嵌入服务获取结果: {last_error}")

    def _request_embeddings(self, texts):
        payload = pickle.dumps(texts)
        header = len(payload).to_bytes(4, byteorder="big")

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as client:
            client.sendall(header)
            client.sendall(payload)

            resp_header = self._recv_exact(client, 4)
            resp_len = int.from_bytes(resp_header, byteorder="big")
            resp_payload = self._recv_exact(client, resp_len)

        embeddings = pickle.loads(resp_payload)
        if not isinstance(embeddings, list):
            raise ValueError("嵌入服务返回格式错误")

        for emb in embeddings:
            if not isinstance(emb, (list, tuple)) or len(emb) != self.vector_dim:
                raise ValueError(f"嵌入向量维度不匹配，期望 {self.vector_dim}")
        return embeddings

    def _recv_exact(self, sock, expected_len: int) -> bytes:
        data = b""
        while len(data) < expected_len:
            chunk = sock.recv(expected_len - len(data))
            if not chunk:
                raise ConnectionError("连接中断，未收到完整数据")
            data += chunk
        return data


class FastTreeComm:
    def __init__(self, graph, embedding_model="all-MiniLM-L6-v2", struct_weight=0.3, config=None):
        """
        :param graph: Input graph (NetworkX DiGraph)
        :param embedding_model: Sentence embedding model
        :param struct_weight: Structural similarity weight (float between 0 and 1)
        :param config: Configuration object (optional)
        """
        if config is None and get_config is not None:
            try:
                config = get_config()
            except:
                config = None
        self.config = config
        self.graph = graph

        if config:
            # Priority: use embeddings.model_name if available (for unified model path management)
            # Otherwise fall back to tree_comm.embedding_model or default
            if embedding_model == "all-MiniLM-L6-v2" and hasattr(config, 'embeddings') and hasattr(config.embeddings, 'model_name'):
                embedding_model = config.embeddings.model_name
            else:
                embedding_model = embedding_model or config.tree_comm.embedding_model
            struct_weight = struct_weight if struct_weight != 0.3 else config.tree_comm.struct_weight
        
        if embedding_model == "local_embedding_service":
            self.model = LocalEmbeddingServiceEncoder(
                host=getattr(config.tree_comm, "embedding_service_host", None) if config and hasattr(config, "tree_comm") else None,
                port=getattr(config.tree_comm, "embedding_service_port", None) if config and hasattr(config, "tree_comm") else None,
                timeout=getattr(config.tree_comm, "embedding_service_timeout", None) if config and hasattr(config, "tree_comm") else None,
                max_batch_size=getattr(config.tree_comm, "embedding_service_batch_size", None) if config and hasattr(config, "tree_comm") else None,
                vector_dim=getattr(config.tree_comm, "embedding_service_vector_dim", None) if config and hasattr(config, "tree_comm") else None,
            )
        else:
            self.model = SentenceTransformer("./models/iic_nlp_gte_sentence-embedding_english-base")
        self.semantic_cache = {}
        self.struct_weight = struct_weight
        self.node_list = list(graph.nodes())
        self.node_names = {}
        for n in graph.nodes():
            props = graph.nodes[n].get("properties") or {}
            name = props.get("name", n)
            self.node_names[n] = str(name)
        self.neighbor_cache = {n: set(graph.neighbors(n)) for n in graph.nodes()}
        self.edge_relations = {(u, v): data.get("relation", "related_to") 
                          for u, v, data in graph.edges(data=True)}
        
        self.triple_strings_cache = {}
        self.degree_cache = {n: self.graph.degree(n) for n in self.node_list}

        self.adjacency_sparse = self._build_sparse_adjacency()
        
        self.llm_client = call_llm_api.LLMCompletionCall()
        env_embed_batch = os.getenv("TREECOMM_EMBED_BATCH_SIZE")
        config_batch = getattr(getattr(config, "tree_comm", None), "embedding_service_batch_size", None) if config else None
        if env_embed_batch:
            self.embed_batch_size = max(1, int(env_embed_batch))
        elif config_batch:
            self.embed_batch_size = max(1, int(config_batch))
        else:
            self.embed_batch_size = int(getattr(self.model, "max_batch_size", 32))
        self.embed_batch_size = max(1, self.embed_batch_size)
        self.max_triple_chars = int(os.getenv("TREECOMM_MAX_TRIPLE_CHARS", "2048"))
        # 默认开启进度条，除非显式设置 0
        self.show_progress = os.getenv("TREECOMM_SHOW_PROGRESS", "1") != "0"

        self._precompute_all_triples()

    def _progress(self, iterable, **kwargs):
        if self.show_progress and tqdm:
            if "total" not in kwargs and hasattr(iterable, "__len__"):
                kwargs["total"] = len(iterable)
            return tqdm(iterable, **kwargs)
        return iterable

    def _build_sparse_adjacency(self):
        n = len(self.node_list)
        node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        row, col = [], []
        
        for node in self.node_list:
            i = node_to_idx[node]
            for neighbor in self.graph.neighbors(node):
                if neighbor in node_to_idx:
                    j = node_to_idx[neighbor]
                    row.append(i)
                    col.append(j)
        
        data = [1] * len(row)
        return sp.csr_matrix((data, (row, col)), shape=(n, n))

    def _precompute_all_triples(self):
        iterator = self._progress(self.node_list, desc="预计算三元组", unit="node")
        for node_id in iterator:
            self.triple_strings_cache[node_id] = self._get_triple_strings(node_id)
        
        return

    def _get_triple_strings(self, node_id):
        """extract all neighbors for one node, enhance the structural perception with 1-hop neighbors"""
        if node_id in self.triple_strings_cache:
            return self.triple_strings_cache[node_id]
            
        node_name = str(self.graph.nodes[node_id]["properties"]["name"])
        triples = []
        
        for neighbor in self.graph.neighbors(node_id):
            rel = self.graph.edges[node_id, neighbor, 0].get("relation", "related_to")
            neighbor_name = str(self.graph.nodes[neighbor]["properties"]["name"])
            triples.append(f"{node_name} {rel} {neighbor_name}")
            
        result = list(set(triples))
        self.triple_strings_cache[node_id] = result
        return result

    def get_triple_embedding(self, node_id):
        """leverage triple-level embedding to represent one node"""
        if node_id not in self.semantic_cache:
            triples = self.triple_strings_cache.get(node_id, [])
            text = ", ".join(triples) if triples else str(self.graph.nodes[node_id]["properties"]["name"])
            if self.max_triple_chars and len(text) > self.max_triple_chars:
                text = text[:self.max_triple_chars]
            self.semantic_cache[node_id] = self.model.encode(text)
        return self.semantic_cache[node_id]
    
    def get_triple_embeddings_batch(self, node_ids):
        """Batch processing for GPU acceleration with optimized caching"""
        uncached_ids = [nid for nid in node_ids if nid not in self.semantic_cache]
        
        if uncached_ids:
            texts = []
            for nid in uncached_ids:
                triples = self.triple_strings_cache.get(nid, [])
                text = " ".join(triples) if triples else self.node_names[nid]
                if self.max_triple_chars and len(text) > self.max_triple_chars:
                    text = text[:self.max_triple_chars]
                texts.append(text)
            
            batch_size = self.embed_batch_size
            iterator = range(0, len(texts), batch_size)
            if self.show_progress and tqdm and len(texts) > batch_size:
                iterator = tqdm(
                    iterator,
                    total=(len(texts) + batch_size - 1) // batch_size,
                    desc="Encoding triples",
                    unit="batch",
                )
            
            encoded_batches = []
            with torch.no_grad():
                for start in iterator:
                    batch_texts = texts[start : start + batch_size]
                    encoded = self.model.encode(
                        batch_texts,
                        convert_to_tensor=True,
                        batch_size=batch_size,
                    )
                    encoded_batches.append(encoded)
            
            embeddings = torch.cat(encoded_batches, dim=0) if encoded_batches else torch.empty((0,))
                
            for nid, emb in zip(uncached_ids, embeddings):
                self.semantic_cache[nid] = emb.cpu().numpy()
        return np.array([self.semantic_cache[nid] for nid in node_ids])

    def _compute_jaccard_matrix_vectorized(self, level_nodes):

        node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        level_indices = [node_to_idx[node] for node in level_nodes if node in node_to_idx]

        if not level_indices:
            return np.zeros((len(level_nodes), len(level_nodes)))

        sub_adj = self.adjacency_sparse[level_indices][:, level_indices]
        intersection = sub_adj.dot(sub_adj.T).toarray()
        row_sums = np.array(sub_adj.sum(axis=1)).flatten()

        union = row_sums[:, None] + row_sums - intersection
        jaccard_matrix = intersection / (union + 1e-9)
        np.fill_diagonal(jaccard_matrix, 1.0)

        return jaccard_matrix

    def _compute_sim_matrix(self, level_nodes):
        start_time = time.time()
        
        node_count = len(level_nodes)
        if node_count <= 1:
            return np.eye(node_count)

        embeddings = self.get_triple_embeddings_batch(level_nodes)
        
        embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        semantic_sim_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)

        structural_sim_matrix = self._compute_jaccard_matrix_vectorized(level_nodes)
        
        sim_matrix = (self.struct_weight * structural_sim_matrix + 
                     (1 - self.struct_weight) * semantic_sim_matrix)
        return sim_matrix

    def _fast_clustering(self, level_nodes, n_clusters=None):
        if len(level_nodes) <= 2:
            return {0: level_nodes}
        
        if n_clusters is None:
            base_clusters = len(level_nodes) // 10
            n_clusters = min(max(2, base_clusters), len(level_nodes) // 2, 200)
        
        embeddings = self.get_triple_embeddings_batch(level_nodes)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        clusters = defaultdict(list)
        for node, label in zip(level_nodes, cluster_labels):
            clusters[label].append(node)
        
        return dict(clusters)

    def detect_communities(self, level_nodes, max_iter=1, merge_threshold=0.5, max_total_communities=None):
        if len(level_nodes) <= 1:
            return {0: level_nodes} if level_nodes else {}

        # 从配置中读取 max_total_communities，如果没有配置则使用默认值
        if max_total_communities is None:
            if self.config and hasattr(self.config.tree_comm, 'max_total_communities'):
                max_total_communities = self.config.tree_comm.max_total_communities
            else:
                # 原有的默认逻辑：节点数的1/3，最少5个，最多200个
                max_total_communities = min(max(5, len(level_nodes) // 3), 200)

        initial_clusters = self._fast_clustering(level_nodes)
        final_communities = {}
        comm_id = 0
        
        # 按簇大小排序，优先处理大簇（确保大簇能得到细分机会）
        sorted_clusters = sorted(initial_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        processed_cluster_ids = set()
        
        cluster_iter = self._progress(sorted_clusters, desc="细化社区", unit="cluster")
        for cluster_id, cluster_nodes in cluster_iter:
            processed_cluster_ids.add(cluster_id)
            
            if len(cluster_nodes) <= 3:
                final_communities[comm_id] = cluster_nodes
                comm_id += 1
            else:
                # 检查是否还有剩余配额进行细分
                if len(final_communities) >= max_total_communities:
                    # 配额已满，将剩余簇直接作为社区，不再细分
                    final_communities[comm_id] = cluster_nodes
                    comm_id += 1
                else:
                    sub_communities = self._refine_cluster(cluster_nodes, max_iter, merge_threshold)
                    for sub_comm in sub_communities.values():
                        final_communities[comm_id] = sub_comm
                        comm_id += 1
                        
                        # 如果社区数量已经达到上限，停止细分
                        if len(final_communities) >= max_total_communities:
                            break
                    
                    # 如果达到上限，将剩余未处理的簇直接添加为社区
                    if len(final_communities) >= max_total_communities:
                        for remaining_cluster_id, remaining_nodes in sorted_clusters:
                            if remaining_cluster_id not in processed_cluster_ids:
                                if len(final_communities) < max_total_communities:
                                    final_communities[comm_id] = remaining_nodes
                                    comm_id += 1
                                else:
                                    break
                        break
        
        logger.info(f"Generated {len(final_communities)} communities from {len(level_nodes)} nodes")
        return final_communities

    def _refine_cluster(self, cluster_nodes, max_iter, merge_threshold):
        if len(cluster_nodes) <= 3:
            return {0: cluster_nodes}

        initial_clusters = self._fast_clustering(cluster_nodes)
        
        if len(initial_clusters) == 1:
            return initial_clusters
        
        cluster_centers = {}
        for cluster_id, nodes in initial_clusters.items():
            center = self._compute_community_center(nodes)
            cluster_centers[cluster_id] = center
        
        center_nodes = list(cluster_centers.values())
        center_sim_matrix = self._compute_sim_matrix(center_nodes)
        
        center_to_idx = {center: idx for idx, center in enumerate(center_nodes)}

        current_clusters = initial_clusters.copy()
        current_centers = cluster_centers.copy()
        
        for iteration in range(max_iter):
            changed = False
            
            cluster_ids = list(current_clusters.keys())
            n_clusters = len(cluster_ids)
            
            cluster_similarities = []
            
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    cluster1_id = cluster_ids[i]
                    cluster2_id = cluster_ids[j]
                    
                    center1 = current_centers[cluster1_id]
                    center2 = current_centers[cluster2_id]
                    idx1 = center_to_idx[center1]
                    idx2 = center_to_idx[center2]
                    center_sim = center_sim_matrix[idx1][idx2]
                    
                    if center_sim >= merge_threshold:
                        cluster_similarities.append({
                            'cluster1': cluster1_id,
                            'cluster2': cluster2_id,
                            'similarity': center_sim
                        })
            
            cluster_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            merged_clusters = set()
            new_clusters = {}
            new_centers = {}
            next_cluster_id = 0
            
            for sim_info in cluster_similarities:
                cluster1_id = sim_info['cluster1']
                cluster2_id = sim_info['cluster2']
                
                if cluster1_id not in merged_clusters and cluster2_id not in merged_clusters:

                    if self._should_merge_clusters(
                        current_clusters[cluster1_id], 
                        current_clusters[cluster2_id],
                        sim_info
                    ):
                        merged_nodes = current_clusters[cluster1_id] + current_clusters[cluster2_id]
                        new_clusters[next_cluster_id] = merged_nodes
                        
                        new_center = self._compute_community_center(merged_nodes)
                        new_centers[next_cluster_id] = new_center
                        center_to_idx[new_center] = len(center_to_idx)
                        
                        merged_clusters.add(cluster1_id)
                        merged_clusters.add(cluster2_id)
                        next_cluster_id += 1
                        changed = True
            
            for cluster_id, nodes in current_clusters.items():
                if cluster_id not in merged_clusters:
                    new_clusters[next_cluster_id] = nodes
                    new_centers[next_cluster_id] = current_centers[cluster_id]
                    next_cluster_id += 1
            
            if not changed:
                break
            
            current_clusters = new_clusters
            current_centers = new_centers
            
            if len(current_clusters) == 1:
                break
        
        return current_clusters
    
    def _should_merge_clusters(self, cluster1_nodes, cluster2_nodes, sim_info):

        if sim_info['similarity'] < 0.5:
            return False
        
        merged_size = len(cluster1_nodes) + len(cluster2_nodes)
        if merged_size > 100:
            return False
        
        return True

    def _compute_community_center(self, community_nodes):
        """Compute community center using the top keyword as the center node"""
        if len(community_nodes) == 1:
            return community_nodes[0]
        return self.extract_keywords_from_community(community_nodes)[0]

    def _build_batch_prompt(self, community_batch):
        batch_data = []
        for comm_id, members in community_batch:
            member_names = [self.node_names[n] for n in members]
            center_node = self._compute_community_center(members)
            center_name = self.node_names[center_node]
            
            comm_info = {
                "id": int(comm_id),
                "size": len(members),
                "members": member_names[:20],
                # 额外提供 center 信息，prompt 可选择使用或忽略
                "center": center_name,
            }
            batch_data.append(comm_info)

        communities_json = json.dumps(batch_data, ensure_ascii=False)
        prompt_template = _load_llm_prompt_template()
        return prompt_template.replace("{communities_json}", communities_json)

    def _call_llm_api_batch(self, content: str) -> List[Dict]:
        if not self.llm_client:
            return []
        response_text = self.llm_client.call_api(content)
        response_json = json_repair.loads(response_text)

        return response_json
        

    def create_super_nodes(self, comm_to_nodes: Dict[str, List[str]], level: int = 4, batch_size: int = 5):
        super_nodes = {}
        communities = [(comm_id, members) for comm_id, members in comm_to_nodes.items() 
                      if len(members) >= 2]
        
        batch_iter = self._progress(
            range(0, len(communities), batch_size),
            desc="生成社区节点",
            unit="batch"
        )
        for i in batch_iter:
            batch = communities[i:i+batch_size]
            
            if self.llm_client:
                try:
                    batch_prompt = self._build_batch_prompt(batch)
                    llm_results = self._call_llm_api_batch(batch_prompt)
                    
                    llm_dict = {str(item.get("id", "")): item for item in llm_results}
                except Exception as e:
                    logger.error(f"Batch LLM processing failed: {e}")
                    llm_dict = {}
            else:
                llm_dict = {}
            
            for comm_id, members in batch:
                try:
                    llm_info = llm_dict.get(str(comm_id), {})
                    comm_name = llm_info.get("name", f"Community_{comm_id}")
                    comm_summary = llm_info.get("summary", f"Community of {len(members)} members")
                    
                    super_node_id = f"comm_{level}_{comm_id}"
                    member_names = [self.node_names[n] for n in members]
                    
                    self.graph.add_node(
                        super_node_id,
                        label="community",
                        level=level,
                        properties={
                            "name": comm_name,
                            "description": comm_summary,
                            "members": member_names
                        }
                    )
                    
                    for node in members:
                        self.graph.add_edge(node, super_node_id, relation="member_of")
                    
                    super_nodes[super_node_id] = member_names
                    
                except Exception as e:
                    logger.error(f"Error creating super node for community {comm_id}: {e}")
        
        logger.info(f"Created {len(super_nodes)} super nodes")
        return super_nodes

    def extract_keywords_from_community(self, community_nodes: List[str], top_k: int = 5) -> List[str]:
        if len(community_nodes) <= top_k:
            return community_nodes

        structural_scores = {node: self.degree_cache.get(node, 0) for node in community_nodes}
        
        node_embeddings = self.get_triple_embeddings_batch(community_nodes)
        avg_embedding = np.mean(node_embeddings, axis=0)
        
        semantic_scores = cosine_similarity(node_embeddings, [avg_embedding]).flatten()
        
        max_degree = max(structural_scores.values()) if structural_scores else 1
        norm_structural = {n: s / max_degree for n, s in structural_scores.items()}
        norm_semantic = dict(zip(community_nodes, semantic_scores))
        
        combined_scores = {
            node: (self.struct_weight * norm_structural[node] +
                   (1 - self.struct_weight) * norm_semantic[node])
            for node in community_nodes
        }
        
        top_nodes = sorted(community_nodes, key=lambda x: combined_scores[x], reverse=True)[:top_k]
        return top_nodes

    def create_super_nodes_with_keywords(self, comm_to_nodes: Dict[str, List[str]], level: int = 4, batch_size: int = 5):
        super_nodes = self.create_super_nodes(comm_to_nodes, level, batch_size)
        
        keyword_mapping = {}
        keyword_iter = self._progress(
            list(comm_to_nodes.items()),
            desc="提取社区关键词",
            unit="community"
        )
        for comm_id, members in keyword_iter:
            if len(members) < 2:
                continue
                
            try:
                keywords = self.extract_keywords_from_community(members)
                super_node_id = f"comm_{level}_{comm_id}"
                
                for keyword in keywords:
                    keyword_node_id = f"kw_{comm_id}_{keyword}"
                    keyword_name = self.node_names[keyword]
                    
                    self.graph.add_node(
                        keyword_node_id,
                        label="keyword",
                        level=3,
                        properties={"name": keyword_name}
                    )
                    
                    self.graph.add_edge(keyword, keyword_node_id, relation="represented_by")
                    self.graph.add_edge(keyword_node_id, super_node_id, relation="keyword_of")
                    
                    for member in members:
                        if member == keyword:
                            self.graph.add_edge(member, keyword_node_id, relation="kw_filter_by")
                    
                    keyword_mapping[keyword_node_id] = keyword
                    
            except Exception as e:
                logger.error(f"Error creating keywords for community {comm_id}: {e}")
        
        return super_nodes, keyword_mapping


def _load_llm_prompt_template(prompt_path: Optional[str] = None) -> str:
    """
    加载用于 TreeComm 社区报告的 LLM prompt 模板。
    为了控制变量，默认与 Leiden 聚类共用同一份：
    prompts/lowlevel_leiden_community_report.txt
    """
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    default_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "lowlevel_leiden_community_report.txt",
    )
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Prompt not found: {default_path}")
    with open(default_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_entity_graph_from_json(path: str) -> nx.MultiDiGraph:
    """
    从底层知识图谱 JSON（prompt2graph 输出的 260129.json 等）构建实体图：
    - 仅保留 start_node/end_node label 为 \"entity\" 的三元组
    - 节点属性中填充 properties.name（供 FastTreeComm 使用）
    - 使用 MultiDiGraph，并为每个三元组添加一条有向边 u -> v（relation 作为边属性）
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        triples = data
    else:
        triples = data.get("edges", data.get("triples", []))

    G = nx.MultiDiGraph()

    for edge in triples:
        start = edge.get("start_node", {}) or {}
        end = edge.get("end_node", {}) or {}

        if start.get("label") != "entity" or end.get("label") != "entity":
            continue

        start_props = start.get("properties", {}) or {}
        end_props = end.get("properties", {}) or {}

        start_name = start_props.get("name")
        end_name = end_props.get("name")
        if not start_name or not end_name:
            continue

        # 使用实体名作为节点 ID，保持直观；同时在 properties 中写回 name
        u = str(start_name)
        v = str(end_name)

        if u not in G:
            G.add_node(u, label="entity", properties={"name": start_name, **start_props})
        if v not in G:
            G.add_node(v, label="entity", properties={"name": end_name, **end_props})

        rel = edge.get("relation", "related_to")

        # 记录该三元组关联的 chunk_ids，后续聚合到社区级别用于“与原文吻合度”评估
        chunk_ids = edge.get("chunk_id", [])
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        elif not isinstance(chunk_ids, list):
            chunk_ids = [chunk_ids] if chunk_ids else []
        chunk_ids = [str(c) for c in chunk_ids if c is not None]

        # 单向边已足够，既能提供结构信息，也能生成三元组文本
        G.add_edge(u, v, relation=rel, chunk_ids=chunk_ids)

    logger.info("Loaded entity graph from %s: %d nodes, %d edges", path, G.number_of_nodes(), G.number_of_edges())
    return G


def run_tree_comm_clustering(
    input_path: str,
    output_dir: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    struct_weight: float = 0.3,
) -> Dict:
    """
    使用 FastTreeComm 在底层实体图上做社区聚类，并输出简单的 community 映射结果。

    Args:
        input_path: 底层图谱 JSON 路径（如 output/.../260129.json）
        output_dir: 输出目录，默认与输入同目录
        embedding_model: sentence-transformer 模型名，或 \"local_embedding_service\"
        struct_weight: 结构相似度权重（0~1）

    Returns:
        dict，包含：
        - node_communities: [{\"node\": 实体名, \"community_id\": cid}, ...]
        - communities: {cid: [实体名, ...]}
        - stats: 基本统计信息
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_json = os.path.join(output_dir, f"{base}_tree_comm_communities.json")
    # out_txt = os.path.join(output_dir, f"{base}_tree_comm_communities.txt")
    out_report_json = os.path.join(output_dir, f"{base}_tree_comm_community_report.json")
    out_report_txt = os.path.join(output_dir, f"{base}_tree_comm_community_report.txt")

    graph = _load_entity_graph_from_json(input_path)
    if graph.number_of_nodes() == 0:
        logger.warning("Graph is empty, skip tree_comm clustering")
        return {"node_communities": [], "communities": {}, "stats": {"nodes": 0, "edges": 0, "communities": 0}}

    detector = FastTreeComm(graph, embedding_model=embedding_model, struct_weight=struct_weight)
    level_nodes = list(graph.nodes())
    comm_to_nodes = detector.detect_communities(level_nodes)

    # 构建 node -> community_id 映射（取第一个匹配的社区）
    node_to_comm: Dict[str, int] = {}
    for cid, members in comm_to_nodes.items():
        for n in members:
            node_to_comm[n] = cid

    node_communities = [{"node": n, "community_id": cid} for n, cid in sorted(node_to_comm.items(), key=lambda x: (x[1], x[0]))]

    # 与 community_clustering.py 统一输出格式：node_communities, communities, stats, hierarchy
    communities_dict = {str(cid): sorted(members) for cid, members in comm_to_nodes.items()}
    stats_dict = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "communities": len(comm_to_nodes),
    }
    # TreeComm 非层次聚类，hierarchy 为空或全部 -1
    hierarchy_dict = {str(cid): -1 for cid in comm_to_nodes.keys()}

    communities_output = {
        "node_communities": node_communities,
        "communities": communities_dict,
        "stats": stats_dict,
        "hierarchy": hierarchy_dict,
    }

    result = {
        "node_communities": node_communities,
        "communities": communities_dict,
        "stats": stats_dict,
        "hierarchy": hierarchy_dict,
    }

    # 写 communities JSON（与 Leiden 统一格式）
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(communities_output, f, ensure_ascii=False, indent=2)
    logger.info("TreeComm communities written to %s", out_json)

    # # 写一个简单的文本版
    # lines: List[str] = []
    # lines.append(f"# TreeComm 社区聚类结果\n")
    # lines.append(f"- 节点数: {result['stats']['nodes']}")
    # lines.append(f"- 边数: {result['stats']['edges']}")
    # lines.append(f"- 社区数: {result['stats']['communities']}\n")

    # for cid, members in sorted(comm_to_nodes.items(), key=lambda x: x[0]):
    #     lines.append(f"## 社区 {cid}  (大小: {len(members)})")
    #     for n in sorted(members):
    #         lines.append(f"- {n}")
    #     lines.append("")

    # with open(out_txt, "w", encoding="utf-8") as f:
    #     f.write("\n".join(lines))
    # logger.info("TreeComm text report written to %s", out_txt)

    # --- 生成基于 LLM 的社区报告（名称 + 摘要），便于与 Leiden 报告对比 ---
    try:
        # 使用已有的社区划分生成社区超节点（内部会调用 LLM）
        super_nodes = detector.create_super_nodes(comm_to_nodes, level=4, batch_size=5)

        # 汇总每个社区内部边对应的 chunk_ids
        from collections import defaultdict as _dd  # 局部别名，避免顶部 import 冲突
        comm_to_chunk_ids: Dict[int, set] = _dd(set)
        for u, v, data in graph.edges(data=True):
            cid_u = node_to_comm.get(u)
            cid_v = node_to_comm.get(v)
            if cid_u is None or cid_v is None or cid_u != cid_v:
                continue
            chunk_ids = data.get("chunk_ids") or []
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            for c in chunk_ids:
                if c is not None:
                    comm_to_chunk_ids[cid_u].add(str(c))

        community_reports = []
        for cid, members in sorted(comm_to_nodes.items(), key=lambda x: x[0]):
            super_node_id = f"comm_4_{cid}"
            node_data = graph.nodes.get(super_node_id, {}) or {}
            props = node_data.get("properties", {}) or {}

            name = props.get("name", f"Community_{cid}")
            summary = props.get("description", f"Community of {len(members)} members")
            member_names = [detector.node_names.get(n, n) for n in members]
            chunk_ids = sorted(comm_to_chunk_ids.get(cid, set()))

            community_reports.append(
                {
                    "community_id": int(cid),
                    "name": name,
                    "summary": summary,
                    "size": len(members),
                    "members": member_names,
                    "relation_counts": {},
                    "top_relations": [],
                    "super_node_id": super_node_id,
                    "chunk_ids": chunk_ids,
                }
            )

        # 与 community_clustering 统一：顶层 key 为 community_report + stats
        report_obj = {
            "community_report": community_reports,
            "stats": result["stats"],
        }

        with open(out_report_json, "w", encoding="utf-8") as f:
            json.dump(report_obj, f, ensure_ascii=False, indent=2)
        logger.info("TreeComm community report written to %s", out_report_json)

        # 简单文本版，方便肉眼对比
        r_lines: List[str] = []
        r_lines.append("# TreeComm 社区报告（LLM 生成名称与摘要）")
        r_lines.append("")
        r_lines.append(f"- 节点数: {result['stats']['nodes']}")
        r_lines.append(f"- 边数: {result['stats']['edges']}")
        r_lines.append(f"- 社区数: {result['stats']['communities']}")
        r_lines.append("")

        for r in community_reports:
            r_lines.append(f"## 社区 {r['community_id']}: {r['name']}  (大小: {r['size']})")
            r_lines.append("")
            r_lines.append(f"摘要: {r['summary']}")
            r_lines.append("")
            r_lines.append("成员（部分）：")
            for m in r["members"][:20]:
                r_lines.append(f"- {m}")
            r_lines.append("")

        with open(out_report_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(r_lines))
        logger.info("TreeComm community text report written to %s", out_report_txt)
    except Exception as e:
        logger.error(f"Failed to create TreeComm community reports: {e}")

    return result


if __name__ == "__main__":  # 简单 CLI，用于直接在底层图谱上跑 TreeComm 聚类
    import argparse

    parser = argparse.ArgumentParser(description="FastTreeComm 双重感知社区检测（底层实体图）")
    parser.add_argument("input", help="底层图谱 JSON 路径，如 output/.../260129.json")
    parser.add_argument("-o", "--output-dir", default=None, help="输出目录（默认与输入同目录）")
    parser.add_argument("--embedding-model", default="prompt2graph/models/iic_nlp_gte_sentence-embedding_english-base", help="sentence-transformer 模型名，或 local_embedding_service")
    parser.add_argument("--struct-weight", type=float, default=0.3, help="结构相似度权重（0~1，默认 0.3）")
    args = parser.parse_args()

    res = run_tree_comm_clustering(
        input_path=args.input,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        struct_weight=args.struct_weight,
    )
    print(json.dumps(res["stats"], ensure_ascii=False, indent=2))

