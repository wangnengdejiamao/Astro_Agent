
#!/usr/bin/env python3
from __future__ import annotations
"""
知识图谱构建器基类
提供单阶段与多阶段构建器共用的方法：schema 加载、图结构维护、去重、输出格式化等。
"""

import ast
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent import futures
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx

from utils.logger import logger


class GraphBuilder(ABC):
    """知识图谱构建器基类，单阶段与多阶段构建器继承此类并实现 _process_chunk_impl"""

    def __init__(
        self,
        schema_path: str = None,
        schema_content: Union[dict, str] = None,
        pubchem_db_path: str = None,
    ):
        """
        初始化 GraphBuilder 基类
        Args:
            schema_path: schema 文件路径（如果提供 schema_content 则忽略）
            schema_content: schema 内容字典或 JSON 字符串（优先使用）
            pubchem_db_path: PubChem 本地数据库路径（可选）
        """
        self.schema = self.load_schema(schema_path, schema_content)
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.token_len = 0
        self.lock = threading.Lock()
        self.pubchem_db_path = pubchem_db_path
        self.pubchem_client_class = None
        self._thread_local = threading.local()
        # CID 查询中间结果：有 CID 的实体 {实体名: cid}，无 CID 的实体名集合（线程安全）
        self._cid_report_lock = threading.Lock()
        self._entities_with_cid: Dict[str, int] = {}
        self._entities_without_cid: set = set()

        if pubchem_db_path:
            try:
                from scorers.pubchem_scorer import PubChemClient
                self.pubchem_client_class = PubChemClient
                logger.info(f"已初始化 PubChem 客户端，将在构建图谱时查询CID: {pubchem_db_path}")
            except Exception as e:
                logger.warning(f"无法初始化 PubChem 客户端 {pubchem_db_path}: {e}，将跳过CID查询")
                self.pubchem_client_class = None

    def load_schema(self, schema_path: str = None, schema_content: Union[dict, str] = None) -> Dict[str, Any]:
        """加载 schema，优先使用 content，否则从文件加载"""
        if schema_content:
            try:
                if isinstance(schema_content, dict):
                    return schema_content
                if isinstance(schema_content, str):
                    return json.loads(schema_content)
                return dict(schema_content)
            except Exception as e:
                logger.error(f"解析 schema_content 失败: {e}")
                return {}

        if not schema_path or not os.path.exists(schema_path):
            logger.warning(f"Schema 文件不存在: {schema_path}，使用空 schema")
            return {}
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载 schema 失败: {e}")
            return {}

    def token_cal(self, text: str) -> int:
        """计算 token 数量"""
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _validate_triple_format(self, triple: list) -> tuple | None:
        """验证、规范化三元组格式"""
        try:
            if len(triple) > 3:
                triple = triple[:3]
            elif len(triple) < 3:
                return None

            subj_raw, pred_raw, obj_raw = triple

            def normalize(value):
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    return ", ".join(str(item) for item in value if item) if value else None
                return str(value)

            subj = normalize(subj_raw)
            pred = normalize(pred_raw)
            obj = normalize(obj_raw)
            if not subj or not pred or not obj:
                return None
            return subj.strip(), pred.strip(), obj.strip()
        except Exception as e:
            logger.error(f"Error validating triple {triple}: {e}")
            return None

    def _query_cid_for_entity(self, entity_name: str) -> int | None:
        """为实体查询 CID（线程安全），并记录到 CID 中间结果供后续保存 JSON"""
        if not self.pubchem_db_path or not self.pubchem_client_class:
            return None
        try:
            client = self._get_thread_local_pubchem_client()
            if not client:
                return None
            result = client.lookup(entity_name)
            if result.get("match", False):
                cid = result.get("cid")
                if cid is not None:
                    logger.debug(f"为实体 '{entity_name}' 查询到 CID: {cid}")
                    with self._cid_report_lock:
                        self._entities_with_cid[entity_name] = cid
                    return cid
            with self._cid_report_lock:
                self._entities_without_cid.add(entity_name)
            logger.debug(f"实体 '{entity_name}' 在 PubChem 数据库中未找到匹配")
        except Exception as e:
            logger.debug(f"查询实体 '{entity_name}' 的CID时出错: {e}")
            with self._cid_report_lock:
                self._entities_without_cid.add(entity_name)
        return None

    def _get_thread_local_pubchem_client(self):
        """获取线程本地的 PubChem 客户端（线程安全）"""
        if not hasattr(self._thread_local, 'pubchem_client'):
            if self.pubchem_client_class and self.pubchem_db_path:
                try:
                    self._thread_local.pubchem_client = self.pubchem_client_class(
                        db_path=self.pubchem_db_path,
                        check_same_thread=False
                    )
                    logger.debug(f"为线程 {threading.current_thread().name} 创建了 PubChem 客户端")
                except Exception as e:
                    logger.warning(f"无法为当前线程创建 PubChem 客户端: {e}")
                    self._thread_local.pubchem_client = None
            else:
                self._thread_local.pubchem_client = None
        return getattr(self._thread_local, 'pubchem_client', None)

    def _update_properties_with_cid(self, properties: dict, entity_name: str, context: str = ""):
        """更新属性字典，如果缺少 CID 则查询并添加"""
        if "cid" not in properties or properties.get("cid") is None:
            cid = self._query_cid_for_entity(entity_name)
            if cid is not None:
                properties["cid"] = cid
                if context:
                    logger.debug(f"{context}实体 '{entity_name}' 查询到 CID: {cid}")

    def _find_or_create_entity(
        self, entity_name: str, nodes_to_add: list, entity_type: str | None = None
    ) -> str:
        """查找或创建实体节点，若启用 PubChem 则查询 CID 并写入属性"""
        with self.lock:
            entity_node_id = next(
                (
                    n
                    for n, d in self.graph.nodes(data=True)
                    if d.get("label") == "entity" and d["properties"].get("name") == entity_name
                ),
                None,
            )
            if entity_node_id:
                node_data = self.graph.nodes[entity_node_id]
                properties = node_data.get("properties", {})
                self._update_properties_with_cid(properties, entity_name, "为已存在的")
                node_data["properties"] = properties
                return entity_node_id

            pending_node_id = None
            pending_node_data = None
            for pid, pdata in nodes_to_add:
                if (
                    pdata.get("label") == "entity"
                    and pdata.get("properties", {}).get("name") == entity_name
                ):
                    pending_node_id = pid
                    pending_node_data = pdata
                    break

            if pending_node_id:
                properties = pending_node_data.get("properties", {})
                self._update_properties_with_cid(properties, entity_name, "为待添加的")
                pending_node_data["properties"] = properties
                return pending_node_id

            entity_node_id = f"entity_{self.node_counter}"
            properties = {"name": entity_name}
            if entity_type:
                properties["schema_type"] = entity_type
            self._update_properties_with_cid(properties, entity_name, "为新创建的")

            nodes_to_add.append((
                entity_node_id,
                {
                    "label": "entity",
                    "properties": properties,
                    "level": 2,
                },
            ))
            self.node_counter += 1
            return entity_node_id

    @abstractmethod
    def _process_chunk_impl(self, chunk_id: str, chunk_text: str) -> None:
        """子类实现：处理单个 chunk 的文本，将提取结果写入 self.graph"""
        pass

    def process_chunk(self, chunk_id: str, chunk_data: Dict[str, str]) -> None:
        """处理单个 chunk（从 chunk_data 取文本并调用子类实现）"""
        chunk_text = chunk_data.get("text", "")
        if not chunk_text:
            return
        self._process_chunk_impl(chunk_id, chunk_text)

    def process_all_chunks(self, chunk_file: str) -> None:
        """读取 chunk 文件并并发处理"""
        if not os.path.exists(chunk_file):
            logger.error(f"Chunk file not found: {chunk_file}")
            return

        logger.debug(f"======== Processing chunks from {chunk_file} ========")
        chunks_to_process: Dict[str, Dict[str, str]] = {}
        seen_chunk_ids = set()
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if "\t" not in line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) != 2 or not parts[0].startswith("id: ") or not parts[1].startswith("Chunk: "):
                        continue
                    chunk_id = parts[0][4:]
                    chunk_data_str = parts[1][7:]
                    if chunk_id in seen_chunk_ids:
                        continue
                    try:
                        chunk_data = ast.literal_eval(chunk_data_str)
                        if isinstance(chunk_data, dict):
                            chunks_to_process[chunk_id] = chunk_data
                            seen_chunk_ids.add(chunk_id)
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"解析 chunk {chunk_id} 失败: {e}")
        except Exception as e:
            logger.error(f"读取 chunk 文件失败: {type(e).__name__}: {e}")
            return

        chunks_list = list(chunks_to_process.items())
        total_chunks = len(chunks_list)
        if total_chunks == 0:
            logger.warning("No chunks to process")
            return

        # IO 密集型任务（LLM API 调用），使用更多线程
        # 从 32 增加到 64，充分利用网络带宽和 API 并发能力
        max_workers = min((os.cpu_count() or 1) * 8, 64)
        start_process = time.time()
        processed_count = 0
        failed_count = 0

        logger.debug(f"Starting processing {total_chunks} chunks with {max_workers} workers...")
        try:
            with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                all_futures = [
                    executor.submit(self.process_chunk, chunk_id, chunk_data)
                    for chunk_id, chunk_data in chunks_list
                ]
                for future in futures.as_completed(all_futures):
                    try:
                        future.result()
                        processed_count += 1
                        if processed_count % 50 == 0 or processed_count == total_chunks:
                            elapsed = time.time() - start_process
                            avg_time = elapsed / processed_count if processed_count else 0
                            remaining = (total_chunks - processed_count) * avg_time
                            logger.debug(
                                f"Progress: {processed_count}/{total_chunks} "
                                f"({processed_count/total_chunks*100:.1f}%) "
                                f"[{failed_count} failed] ETA: {remaining/60:.1f} minutes"
                            )
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Chunk处理失败: {type(e).__name__}: {e}")
        except Exception as e:
            logger.error(f"Processing phase failed: {type(e).__name__}: {e}")
            return

        elapsed_time = time.time() - start_process
        logger.debug(f"Processing Time: {elapsed_time:.2f}s")
        logger.info(f"Successfully processed: {processed_count}/{total_chunks} chunks; Failed: {failed_count}")

    def triple_deduplicate(self):
        """去重三元组并合并 chunk id"""
        dedup_start = time.time()
        logger.debug("[Level3] 开始三元组去重...")
        chunk_id_field = "chunk_id"
        new_graph = nx.MultiDiGraph()

        all_nodes = list(self.graph.nodes(data=True))
        for node, data in all_nodes:
            new_graph.add_node(node, **data)

        all_edges = list(self.graph.edges(keys=True, data=True))
        seen_triples: dict = {}

        def _normalize_chunk_ids(chunk_value):
            if chunk_value is None:
                return []
            if isinstance(chunk_value, list):
                return chunk_value
            return [chunk_value]

        for u, v, _key, data in all_edges:
            relation = data.get("relation")
            triple_key = (u, v, relation)
            if triple_key not in seen_triples:
                normalized = data.copy()
                chunk_ids = _normalize_chunk_ids(normalized.get(chunk_id_field))
                if chunk_ids:
                    normalized[chunk_id_field] = chunk_ids
                seen_triples[triple_key] = normalized
                new_graph.add_edge(u, v, **normalized)
            else:
                existing = seen_triples[triple_key]
                existing_ids = _normalize_chunk_ids(existing.get(chunk_id_field))
                new_ids = _normalize_chunk_ids(data.get(chunk_id_field))
                for cid in new_ids:
                    if cid not in existing_ids:
                        existing_ids.append(cid)
                if existing_ids:
                    existing[chunk_id_field] = existing_ids
                for edge_key in new_graph[u][v]:
                    if new_graph[u][v][edge_key].get("relation") == relation:
                        new_graph[u][v][edge_key].update(existing)
                        break

        self.graph = new_graph
        logger.debug(f"[Level3] 三元组去重完成，用时 {time.time() - dedup_start:.2f}s")

    def format_output(self) -> List[Dict[str, Any]]:
        """格式化输出为列表（精简格式，保留评分但去掉解释）"""
        output: List[Dict[str, Any]] = []
        for u, v, data in self.graph.edges(data=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            relation = data.get("relation", "")

            # 保留评分字段，去掉详细解释字段
            score_dict: Dict[str, float] = {}
            
            if relation == "has_attribute":
                node_accuracy_score = data.get("node_accuracy_score")
                triple_support_score = data.get("triple_support_score")
                accuracy_score = data.get("accuracy_score")
                usefulness_score = data.get("usefulness_score")
                if node_accuracy_score is not None:
                    score_dict["node_accuracy_score"] = node_accuracy_score
                if triple_support_score is not None:
                    score_dict["triple_support_score"] = triple_support_score
                if accuracy_score is not None:
                    score_dict["accuracy_score"] = accuracy_score
                if usefulness_score is not None:
                    score_dict["usefulness_score"] = usefulness_score
            else:
                start_accuracy_score = data.get("start_accuracy_score")
                end_accuracy_score = data.get("end_accuracy_score")
                triple_support_score = data.get("triple_support_score")
                accuracy_score = data.get("accuracy_score")
                usefulness_score = data.get("usefulness_score")
                if start_accuracy_score is not None:
                    score_dict["start_accuracy_score"] = start_accuracy_score
                if end_accuracy_score is not None:
                    score_dict["end_accuracy_score"] = end_accuracy_score
                if triple_support_score is not None:
                    score_dict["triple_support_score"] = triple_support_score
                if accuracy_score is not None:
                    score_dict["accuracy_score"] = accuracy_score
                if usefulness_score is not None:
                    score_dict["usefulness_score"] = usefulness_score

            relationship: Dict[str, Any] = {
                "start_node": {"label": u_data["label"], "properties": u_data["properties"]},
                "relation": relation,
                "end_node": {"label": v_data["label"], "properties": v_data["properties"]},
                "source": data.get("source", ""),
                "evidence": data.get("evidence", ""),
                "chunk_id": data.get("chunk_id"),
            }
            # 只在有评分时添加 score 字段
            if score_dict:
                relationship["score"] = score_dict
            output.append(relationship)
        return output

    @property
    def use_staged_extraction(self) -> bool:
        """是否使用多阶段提取，子类可覆盖"""
        return False

    def _save_cid_report(self, output_dir: str) -> Optional[str]:
        """将 CID 查询中间结果保存为 JSON。返回保存路径；未启用 PubChem 或无数据时返回 None"""
        if not self.pubchem_db_path:
            return None
        with self._cid_report_lock:
            if not self._entities_with_cid and not self._entities_without_cid:
                return None
            report = {
                "entities_with_cid": [
                    {"name": name, "cid": cid}
                    for name, cid in sorted(self._entities_with_cid.items(), key=lambda x: x[0])
                ],
                "entities_without_cid": sorted(self._entities_without_cid),
                "summary": {
                    "count_with_cid": len(self._entities_with_cid),
                    "count_without_cid": len(self._entities_without_cid),
                },
            }
        path = os.path.join(output_dir, "cid_query_report.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"CID 查询中间结果已保存: {path} (有 CID: {report['summary']['count_with_cid']}, 无 CID: {report['summary']['count_without_cid']})")
        return path

    def build_knowledge_graph(
        self,
        chunk_file: str,
        output_graph_path: str,
        stage_output_dir: Optional[str] = None,
    ) -> str:
        """主流程：读取 chunk -> 构建 -> 去重 -> 保存"""
        logger.info("======== 开始图谱构建流程 ========")
        if getattr(self, "save_stage_outputs", False):
            self.stage_output_dir = stage_output_dir
        else:
            self.stage_output_dir = None

        extraction_mode = "Staged" if self.use_staged_extraction else "Single-stage"
        logger.debug(f"========{'Start Building (' + extraction_mode + ')':^30}========")
        logger.debug(f"{'➖' * 30}")
        if self.use_staged_extraction and getattr(self, "enable_stage4_validation", False):
            logger.debug("阶段4验证: 已启用 (会消耗大量 token)")

        self.process_all_chunks(chunk_file)
        if not self.use_staged_extraction:
            logger.debug(f"All Process finished, token cost: {self.token_len}")

        logger.debug(f"🚀🚀🚀🚀 {'Processing Level 3':^20} 🚀🚀🚀🚀")
        logger.debug(f"{'➖' * 20}")
        self.triple_deduplicate()

        output = self.format_output()
        output_dir = os.path.dirname(output_graph_path)
        os.makedirs(output_dir, exist_ok=True)
        with open(output_graph_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info(f"Graph saved to {output_graph_path}")

        self._save_cid_report(output_dir)

        logger.info("======== 图谱构建流程完成 ========")

        pubchem_client = getattr(self, "pubchem_client", None)
        if pubchem_client and hasattr(pubchem_client, 'close'):
            try:
                pubchem_client.close()
            except Exception as e:
                logger.debug(f"关闭PubChem客户端时出错: {e}")

        return output_graph_path
