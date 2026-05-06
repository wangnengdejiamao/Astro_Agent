
#!/usr/bin/env python3
from __future__ import annotations
"""
多阶段（Staged）知识图谱构建器
通过 Stage1 实体识别、Stage2 关系提取、Stage3 属性提取（及可选的 Stage4 验证）构建图谱。
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from graph_builder import GraphBuilder
from utils.logger import logger


class StagedGraphBuilder(GraphBuilder):
    """多阶段图谱构建器：Stage1→Stage2→Stage3（→Stage4）合并后写入图"""

    def __init__(
        self,
        config: Dict[str, Any] = None,
        *,
        schema_path: str = None,
        schema_content: dict | str = None,
        prompt_paths: Dict[str, str] = None,
        examples: Dict[str, str] = None,
        enable_stage4_validation: bool = False,
        save_stage_outputs: bool = False,
        pubchem_db_path: str = None,
        stage4_min_triple_score: float = 0.5,
        stage4_min_node_score: float = 0.5,
        stage4_use_chunk_scoring: bool = True,
        stage4_use_node_accuracy_scoring: bool = True,
        stage4_use_triple_support_scoring: bool = True,
    ):
        """
        初始化 StagedGraphBuilder。
        推荐传入 config 字典；也可使用关键字参数（与 config 同时传入时，kwargs 覆盖 config）。
        config 可包含: schema_path, schema_content, prompt_paths, examples,
        enable_stage4_validation, save_stage_outputs, pubchem_db_path,
        stage4 (dict with min_triple_score, min_node_score, use_chunk_scoring, use_node_accuracy_scoring, use_triple_support_scoring)。

        examples 格式: {"stage1": "stage1 examples content", "stage2": "stage2 examples content"}
        """
        if config:
            schema_path = config.get("schema_path", schema_path)
            schema_content = config.get("schema_content", schema_content)
            prompt_paths = config.get("prompt_paths", prompt_paths)
            examples = config.get("examples", examples)
            enable_stage4_validation = config.get("enable_stage4_validation", enable_stage4_validation)
            save_stage_outputs = config.get("save_stage_outputs", save_stage_outputs)
            pubchem_db_path = config.get("pubchem_db_path", pubchem_db_path)
            stage4_cfg = config.get("stage4") or {}
            stage4_min_triple_score = stage4_cfg.get("min_triple_score", stage4_min_triple_score)
            stage4_min_node_score = stage4_cfg.get("min_node_score", stage4_min_node_score)
            stage4_use_chunk_scoring = stage4_cfg.get("use_chunk_scoring", stage4_use_chunk_scoring)
            stage4_use_node_accuracy_scoring = stage4_cfg.get("use_node_accuracy_scoring", stage4_use_node_accuracy_scoring)
            stage4_use_triple_support_scoring = stage4_cfg.get("use_triple_support_scoring", stage4_use_triple_support_scoring)
        super().__init__(
            schema_path=schema_path,
            schema_content=schema_content,
            pubchem_db_path=pubchem_db_path,
        )
        self.prompt_paths = prompt_paths or {}
        self.examples = examples or {}
        self.enable_stage4_validation = enable_stage4_validation
        self.save_stage_outputs = save_stage_outputs
        self.stage_output_dir: Optional[str] = None
        self.stage4_min_triple_score = stage4_min_triple_score
        self.stage4_min_node_score = stage4_min_node_score
        self.stage4_use_chunk_scoring = stage4_use_chunk_scoring
        self.stage4_use_node_accuracy_scoring = stage4_use_node_accuracy_scoring
        self.stage4_use_triple_support_scoring = stage4_use_triple_support_scoring

        from staged_extraction.stage1_entity_recognition import Stage1EntityRecognition
        from staged_extraction.stage2_relation_extraction import Stage2RelationExtraction
        from staged_extraction.stage3_attribute_extraction import Stage3AttributeExtraction
        from staged_extraction.stage4_validation import Stage4Validation

        self.stage1 = Stage1EntityRecognition(
            schema=self.schema,
            prompt_path=self.prompt_paths.get("stage1")
        )
        self.stage2 = Stage2RelationExtraction(
            schema=self.schema,
            prompt_path=self.prompt_paths.get("stage2")
        )
        self.stage3 = Stage3AttributeExtraction(
            schema=self.schema,
            prompt_path=self.prompt_paths.get("stage3")
        )
        if self.enable_stage4_validation:
            self.stage4 = Stage4Validation()

    @property
    def use_staged_extraction(self) -> bool:
        return True

    def _save_stage_output(self, chunk_id: str, stage_name: str, data: Dict[str, Any]) -> None:
        """在开启开关时，将多阶段中间结果保存为 JSON 文件。"""
        if not (self.save_stage_outputs and self.stage_output_dir):
            return
        try:
            os.makedirs(self.stage_output_dir, exist_ok=True)
            file_path = os.path.join(
                self.stage_output_dir, f"{chunk_id}_{stage_name}.json"
            )
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(
                f"Chunk {chunk_id}: 已保存 {stage_name} 中间结果到 {file_path}"
            )
        except Exception as e:
            logger.warning(
                f"Chunk {chunk_id}: 无法保存 {stage_name} 中间结果: {type(e).__name__}: {e}"
            )

    def _process_chunk_staged(self, chunk_id: str, chunk_text: str) -> None:
        """多阶段提取处理 chunk"""
        try:
            stage1_variables = {"examples": self.examples.get("stage1", "")} if self.examples.get("stage1") else None
            stage1_output = self.stage1.extract(chunk_text, variables=stage1_variables)
            if not stage1_output.get("entities"):
                logger.warning(f"Chunk {chunk_id}: No entities found in Stage 1")
                return
            self._save_stage_output(chunk_id, "stage1", stage1_output)

            stage2_variables = {"examples": self.examples.get("stage2", "")} if self.examples.get("stage2") else None
            stage2_output = self.stage2.extract(chunk_text, stage1_output, variables=stage2_variables)
            self._save_stage_output(chunk_id, "stage2", stage2_output)

            stage3_output = self.stage3.extract(chunk_text, stage1_output)
            self._save_stage_output(chunk_id, "stage3", stage3_output)

            if self.enable_stage4_validation:
                logger.info(f"Chunk {chunk_id}: 开始 Stage4 验证")
                triples_for_validation = stage2_output.get("triples", [])
                attributes_for_validation = stage3_output.get("attributes", {})

                validation_result = self.stage4.validate_and_filter(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    triples=triples_for_validation,
                    attributes=attributes_for_validation,
                    min_triple_score=self.stage4_min_triple_score,
                    min_node_score=self.stage4_min_node_score,
                    use_chunk_scoring=self.stage4_use_chunk_scoring,
                    use_node_accuracy_scoring=self.stage4_use_node_accuracy_scoring,
                    use_triple_support_scoring=self.stage4_use_triple_support_scoring,
                )

                self._save_stage_output(chunk_id, "stage4", validation_result["scores"])

                stage2_output = {"triples": validation_result["triples"]}
                stage3_output = {"attributes": validation_result["attributes"]}

                # 传递 triple_details 到图构建阶段，用于将 score 写入边属性
                triple_scores = validation_result["scores"].get("triple_details", {})
                attribute_scores = validation_result["scores"].get("attribute_details", {})

                self.token_len += self.stage4.token_len

                logger.info(
                    f"Chunk {chunk_id}: Stage 4 validation removed "
                    f"{len(validation_result['scores']['removed']['triples'])} triples, "
                    f"{len(validation_result['scores']['removed']['attributes'])} attributes"
                )

                self._merge_and_build_graph_staged(
                    chunk_id,
                    chunk_text,
                    stage1_output,
                    stage2_output,
                    stage3_output,
                    triple_scores,
                    attribute_scores
                )
            else:
                self._merge_and_build_graph_staged(
                    chunk_id,
                    chunk_text,
                    stage1_output,
                    stage2_output,
                    stage3_output,
                    {},
                    {}
                )

            logger.info(f"Chunk {chunk_id}: Processed successfully (staged)")
        except Exception as e:
            logger.error(f"Chunk {chunk_id} processing failed (staged): {e}")

    def _merge_and_build_graph_staged(
        self,
        chunk_id: str,
        chunk_text: str,
        stage1_output: Dict[str, Any],
        stage2_output: Dict[str, Any],
        stage3_output: Dict[str, Any],
        triple_scores: Dict[str, Dict[str, Any]] = {},
        attribute_scores: Dict[str, List[Dict[str, Any]]] = {}
    ) -> None:
        """合并各阶段结果并构建图（多阶段提取）"""
        entity_types = {
            entity["canonical_name"]: entity["schema_type"]
            for entity in stage1_output.get("entities", [])
        }

        attr_nodes, attr_edges = self._process_attributes_staged(
            stage3_output.get("attributes", {}),
            chunk_id,
            chunk_text,
            entity_types,
            attribute_scores
        )

        triple_nodes, triple_edges = self._process_triples_staged(
            stage2_output.get("triples", []),
            chunk_id,
            chunk_text,
            entity_types,
            triple_scores
        )

        all_nodes = attr_nodes + triple_nodes
        all_edges = attr_edges + triple_edges

        with self.lock:
            for node_id, node_data in all_nodes:
                self.graph.add_node(node_id, **node_data)
            for edge_data in all_edges:
                u, v, relation, edge_attrs = edge_data
                self.graph.add_edge(u, v, relation=relation, **edge_attrs)

        logger.debug(f"Chunk {chunk_id}: Added {len(all_nodes)} nodes, {len(all_edges)} edges (staged)")

    def _process_attributes_staged(
        self,
        attributes: Dict[str, List[Dict[str, str]]],
        chunk_id: str,
        chunk_text: str,
        entity_types: Dict[str, str],
        attribute_scores: Dict[str, List[Dict[str, Any]]] = {}
    ) -> Tuple[List, List]:
        """处理属性，转换为图节点和边（多阶段格式）"""
        nodes_to_add = []
        edges_to_add = []

        for entity_name, attr_list in attributes.items():
            entity_type = entity_types.get(entity_name)
            entity_scores = attribute_scores.get(entity_name, [])

            for idx, attr in enumerate(attr_list):
                attr_key = attr.get("key", "")
                attr_value = attr.get("value", "")
                source_text = attr.get("source", "")
                evidence_text = attr.get("evidence", "")

                if not attr_key or not attr_value:
                    continue

                attr_str = f"{attr_key}: {attr_value}"

                with self.lock:
                    attr_node_id = f"attr_{self.node_counter}"
                    self.node_counter += 1

                nodes_to_add.append((
                    attr_node_id,
                    {
                        "label": "attribute",
                        "properties": {"name": attr_str},
                        "level": 1
                    }
                ))

                entity_node_id = self._find_or_create_entity(
                    entity_name,
                    nodes_to_add,
                    entity_type
                )

                edge_attrs = {
                    "source": source_text,
                    "evidence": evidence_text,
                    "chunk_id": chunk_id
                }

                # 保留评分字段，去掉详细解释字段
                if idx < len(entity_scores):
                    scores = entity_scores[idx]
                    if "node_score" in scores:
                        edge_attrs["node_accuracy_score"] = scores["node_score"]
                    if "attr_score" in scores:
                        edge_attrs["triple_support_score"] = scores["attr_score"]
                    if "accuracy_score" in scores:
                        edge_attrs["accuracy_score"] = scores["accuracy_score"]
                    if "usefulness_score" in scores:
                        edge_attrs["usefulness_score"] = scores["usefulness_score"]

                edges_to_add.append((
                    entity_node_id,
                    attr_node_id,
                    "has_attribute",
                    edge_attrs
                ))

        return nodes_to_add, edges_to_add

    def _process_triples_staged(
        self,
        triples: List[Dict[str, str]],
        chunk_id: str,
        chunk_text: str,
        entity_types: Dict[str, str],
        triple_scores: Dict[str, Dict[str, Any]] = {}
    ) -> Tuple[List, List]:
        """处理三元组，转换为图节点和边（多阶段格式）"""
        nodes_to_add = []
        edges_to_add = []

        for triple in triples:
            subject = triple.get("subject", "").strip()
            relation = triple.get("relation", "").strip()
            obj = triple.get("object", "").strip()
            source_text = triple.get("source", "")
            evidence_text = triple.get("evidence", "")

            if not subject or not relation or not obj:
                continue

            subj_type = entity_types.get(subject)
            obj_type = entity_types.get(obj)

            subj_node_id = self._find_or_create_entity(subject, nodes_to_add, subj_type)
            obj_node_id = self._find_or_create_entity(obj, nodes_to_add, obj_type)

            edge_attrs = {
                "source": source_text,
                "evidence": evidence_text,
                "chunk_id": chunk_id
            }

            triple_key = f"{subject}|{relation}|{obj}"
            # 保留评分字段，去掉详细解释字段
            if triple_key in triple_scores:
                scores = triple_scores[triple_key]
                if "start_accuracy_score" in scores:
                    edge_attrs["start_accuracy_score"] = scores["start_accuracy_score"]
                if "end_accuracy_score" in scores:
                    edge_attrs["end_accuracy_score"] = scores["end_accuracy_score"]
                if "triple_support_score" in scores:
                    edge_attrs["triple_support_score"] = scores["triple_support_score"]
                if "accuracy_score" in scores:
                    edge_attrs["accuracy_score"] = scores["accuracy_score"]
                if "usefulness_score" in scores:
                    edge_attrs["usefulness_score"] = scores["usefulness_score"]

            edges_to_add.append((
                subj_node_id,
                obj_node_id,
                relation,
                edge_attrs
            ))

        return nodes_to_add, edges_to_add

    def _process_chunk_impl(self, chunk_id: str, chunk_text: str) -> None:
        """多阶段实现：调用 _process_chunk_staged"""
        self._process_chunk_staged(chunk_id, chunk_text)
