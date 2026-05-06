
#!/usr/bin/env python3
from __future__ import annotations
"""
单阶段知识图谱构建器
通过一次 LLM 调用从文本中提取实体、关系和属性，并构建图谱。
"""

import json
import os
from typing import Any, Dict, List, Tuple, Union

import json_repair

from graph_builder import GraphBuilder
from utils import call_llm_api
from utils.logger import logger


class SingleGraphBuilder(GraphBuilder):
    """单阶段图谱构建器：一次 prompt 提取 triples + attributes + entity_types，再写入图"""

    def __init__(
        self,
        schema_path: str = None,
        prompt_path: str = None,
        schema_content: Union[dict, str] = None,
        prompt_content: str = None,
        pubchem_db_path: str = None,
    ):
        """
        初始化 SingleGraphBuilder
        Args:
            schema_path: schema 文件路径（若提供 schema_content 则忽略）
            prompt_path: prompt 模板文件路径（若提供 prompt_content 则忽略）
            schema_content: schema 内容（优先使用）
            prompt_content: prompt 内容（优先使用）
            pubchem_db_path: PubChem 本地数据库路径（可选）
        """
        super().__init__(
            schema_path=schema_path,
            schema_content=schema_content,
            pubchem_db_path=pubchem_db_path,
        )
        self.prompt_path = prompt_path
        self.prompt_content = prompt_content
        try:
            self.llm_client = call_llm_api.LLMCompletionCall()
        except Exception:
            logger.error("无法初始化 LLM 客户端")
            raise

    def extract_with_llm(self, prompt: str) -> str:
        """调用 LLM 并返回修复后的 JSON 字符串"""
        response = self.llm_client.call_api(prompt)
        parsed_dict = json_repair.loads(response)
        return json.dumps(parsed_dict, ensure_ascii=False)

    def _get_construction_prompt(self, chunk: str) -> str:
        """根据模板生成构建 prompt"""
        recommend_schema = json.dumps(self.schema, ensure_ascii=False)
        prompt_template = None

        if self.prompt_content:
            prompt_template = self.prompt_content
            logger.info("使用提供的 prompt_content")
        elif self.prompt_path and os.path.exists(self.prompt_path):
            try:
                with open(self.prompt_path, "r", encoding="utf-8") as f:
                    prompt_template = f.read()
                logger.info(f"从文件读取 prompt 模板: {self.prompt_path}")
            except Exception as e:
                logger.error(f"读取 prompt 文件失败: {e}")

        if not prompt_template:
            prompt_template = """请从以下文本中提取实体、关系和属性。Schema:
{schema}
文本:
{chunk}

要求：
- 每个三元组必须是一个四元组：[主体, 谓词, 客体, evidence]，其中 evidence 是支持该三元组的原文片段。
- 每个属性必须是一个三元组：[实体, 属性名: 属性值, evidence]，其中 evidence 是支持该属性的原文片段。
- evidence 必须直接来自输入文本，不可编造。

请返回 JSON 格式，包含:
- "triples": 列表，每个元素为 [subj, pred, obj, evidence]
- "attributes": 字典，键为实体名，值为列表，每个元素为 ("属性名: 属性值", evidence) 或 ["属性名: 属性值", evidence]
- "entity_types": 实体类型字典
"""

        prompt = prompt_template.replace("{schema}", recommend_schema)
        prompt = prompt.replace("{chunk}", chunk)
        return prompt

    def _validate_extraction_format(self, parsed_json: Any) -> Tuple[bool, str]:
        """验证提取的 JSON 格式"""
        if not isinstance(parsed_json, dict):
            return False, f"响应不是一个字典 (dict)，而是 {type(parsed_json)}"

        required_keys = ["attributes", "triples", "entity_types"]
        for key in required_keys:
            if key not in parsed_json:
                return False, f"JSON中缺少必需的键: '{key}'"

        triples = parsed_json.get("triples")
        if not isinstance(triples, list):
            return False, f"'triples' 字段必须是一个列表 (list)，但它是 {type(triples)}"
        for i, triple in enumerate(triples):
            if not isinstance(triple, list):
                return False, f"Triples列表中的第 {i} 项不是一个列表"
            if len(triple) not in [3, 5]:
                return False, f"Triples列表中的第 {i} 项长度不为 5 (而是 {len(triple)})"
            if not all(isinstance(el, str) for el in triple):
                return False, f"Triples列表中的第 {i} 项包含非字符串元素"

        attributes = parsed_json["attributes"]
        if not isinstance(attributes, dict):
            return False, "'attributes' 字段必须是一个字典（dict）"
        for entity, attr_list in attributes.items():
            if not isinstance(attr_list, list):
                return False, f"实体 '{entity}' 的属性值必须是列表"
            for j, attr_item in enumerate(attr_list):
                if not (isinstance(attr_item, str) or (isinstance(attr_item, (list, tuple)) and len(attr_item) in [2, 3])):
                    return False, f"实体 '{entity}' 的第 {j} 个属性必须是3元组或字符串: [\"k: v\", source, evidence]"
                for k, part in enumerate(attr_item):
                    if not isinstance(part, str):
                        return False, f"属性项第 {k} 部分必须是字符串"

        if not isinstance(parsed_json.get("entity_types"), dict):
            return False, "'entity_types' 字段必须是一个字典 (dict)"

        return True, "Success"

    def _validate_and_parse_llm_response(
        self, prompt: str, llm_response: str, max_retries: int = 5
    ) -> dict | None:
        """验证、解析并在必要时重试纠错"""
        if llm_response is None:
            return None

        original_prompt = prompt
        current_response_str = llm_response

        for attempt in range(max_retries + 1):
            try:
                if attempt == 0:
                    self.token_len += self.token_cal(original_prompt + current_response_str)

                parsed_dict = json_repair.loads(current_response_str)

                if attempt == 0:
                    logger.debug(f"解析后的JSON键: {list(parsed_dict.keys()) if isinstance(parsed_dict, dict) else 'Not a dict'}")
                    logger.debug(f"响应预览 (前500字符): {current_response_str[:500]}")

                is_valid, error_msg = self._validate_extraction_format(parsed_dict)

                if is_valid:
                    if attempt > 0:
                        logger.info(f"LLM响应验证成功 (经过 {attempt + 1} 次尝试)")
                    return parsed_dict
                raise ValueError(f"JSON格式验证失败: {error_msg}")
            except Exception as e:
                logger.error(f"解析或验证LLM响应失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == 0:
                    logger.error(f"原始响应 (前1000字符): {current_response_str[:1000]}")
                    try:
                        parsed_debug = json_repair.loads(current_response_str)
                        logger.error(f"解析后的键: {list(parsed_debug.keys()) if isinstance(parsed_debug, dict) else type(parsed_debug)}")
                    except Exception as parse_err:
                        logger.error(f"JSON解析失败: {parse_err}")
                if attempt >= max_retries:
                    logger.error(f"达到最大重试次数 ({max_retries + 1})，放弃此 chunk")
                    break

                correction_prompt = f"""上一次尝试提取JSON失败了。

原始提示:
---
{original_prompt}
---

失败的响应:
---
{current_response_str}
---

失败原因: {str(e)}

请严格按照原始提示的格式要求修正这个错误。
你必须严格返回一个符合所有规则的JSON对象。
特别注意：
1. 'triples' 必须是一个列表。
2. 列表中的每一项 必须 是一个包含5个字符串元素的列表。
3. 列表中的每一项的第4个元素必须是原文片段。

请重新生成完整且正确的JSON：
"""
                current_response_str = self.extract_with_llm(correction_prompt)
                self.token_len += self.token_cal(correction_prompt + current_response_str)
        return None

    def _process_attributes(
        self, extracted_attr: dict, chunk_id: int, chunk_text: str, entity_types: dict | None = None
    ) -> Tuple[List, List]:
        """处理单阶段格式的属性，返回 (nodes_to_add, edges_to_add)"""
        nodes_to_add: List = []
        edges_to_add: List = []
        for entity, attributes in extracted_attr.items():
            for attr_item in attributes:
                # 支持字符串或3元组
                if isinstance(attr_item, str):
                    attr_str = attr_item
                    source_text = ""
                    llm_reasoning = ""
                elif isinstance(attr_item, (list, tuple)) and len(attr_item) >= 1:
                    attr_str = attr_item[0]
                    source_text = attr_item[1] if len(attr_item) > 1 else ""
                    llm_reasoning = attr_item[2] if len(attr_item) > 2 else ""
                else:
                    continue
                if attr_str is None or str(attr_str).strip() == "":
                    continue
                attr_str = attr_str.strip()
                if attr_str == "N/A" or attr_str.endswith(": N/A"):
                    continue

                attr_key, attr_value = None, None
                if ": " in attr_str or ":" in attr_str:
                    key_part, _, value_part = attr_str.partition(":")
                    attr_key = key_part.strip().lower()
                    attr_value = value_part.strip()
                else:
                    attr_value = attr_str

                entity_norm = str(entity).strip().lower()
                attr_value_norm = (attr_value or "").strip().lower()
                entity_type_name = (entity_types.get(entity) or "").strip().lower() if entity_types else ""

                redundant_attr = False
                if not attr_value_norm:
                    redundant_attr = True
                elif attr_value_norm == entity_norm:
                    redundant_attr = True
                elif entity_type_name and attr_value_norm == entity_type_name:
                    redundant_attr = True
                elif len(attr_value_norm) < 3:
                    redundant_attr = True
                elif attr_key in {"description", "alias", "abbreviation", "synonym"}:
                    if entity_norm and attr_value_norm.replace(" ", "") == entity_norm.replace(" ", ""):
                        redundant_attr = True
                if redundant_attr:
                    continue

                attr_node_id = f"attr_{self.node_counter}"
                nodes_to_add.append(
                    (
                        attr_node_id,
                        {"label": "attribute", "properties": {"name": attr_str}, "level": 1},
                    )
                )
                self.node_counter += 1

                entity_type = entity_types.get(entity) if entity_types else None
                entity_node_id = self._find_or_create_entity(entity, nodes_to_add, entity_type)

                edge_attrs = {
                    "source": source_text,
                    "evidence": llm_reasoning,
                    "chunk_id": chunk_id
                }
                edges_to_add.append((
                    entity_node_id,
                    attr_node_id,
                    "has_attribute",
                    edge_attrs
                ))
        return nodes_to_add, edges_to_add

    def _process_triples(
        self, extracted_triples: list, chunk_id: int, chunk_text: str, entity_types: dict | None = None
    ) -> Tuple[List, List]:
        """处理单阶段格式的三元组，返回 (nodes_to_add, edges_to_add)"""
        nodes_to_add: List = []
        edges_to_add: List = []
        for triple in extracted_triples:
            # 支持3元素或5元素的triples
            if len(triple) == 3:
                subj, pred, obj = triple
                source_text = ""
                llm_reasoning = ""
            elif len(triple) == 5:
                subj, pred, obj, source_text, llm_reasoning = triple
            else:
                logger.warning(f"Invalid triple format: {triple}")
                continue
            validated_triple = self._validate_triple_format([subj, pred, obj])
            if not validated_triple:
                continue
            subj, pred, obj = validated_triple
            subj_type = entity_types.get(subj) if entity_types else None
            obj_type = entity_types.get(obj) if entity_types else None
            subj_node_id = self._find_or_create_entity(subj, nodes_to_add, subj_type)
            obj_node_id = self._find_or_create_entity(obj, nodes_to_add, obj_type)

            edge_attrs = {
                "source": source_text,
                "evidence": llm_reasoning,
                "chunk_id": chunk_id
            }
            edges_to_add.append((
                subj_node_id,
                obj_node_id,
                pred,
                edge_attrs
            ))
        return nodes_to_add, edges_to_add

    def process_level1_level2(self, chunk: str, chunk_id: int) -> None:
        """单阶段：生成 prompt、调用 LLM、解析并处理属性与三元组"""
        prompt = self._get_construction_prompt(chunk)
        llm_response = self.extract_with_llm(prompt)
        parsed_response = self._validate_and_parse_llm_response(prompt, llm_response)

        if not parsed_response:
            return

        extracted_attr = parsed_response.get("attributes", {})
        extracted_triples = parsed_response.get("triples", [])
        entity_types = parsed_response.get("entity_types", {})

        attr_nodes, attr_edges = self._process_attributes(extracted_attr, chunk_id, chunk, entity_types)
        triple_nodes, triple_edges = self._process_triples(extracted_triples, chunk_id, chunk, entity_types)

        all_nodes = attr_nodes + triple_nodes
        all_edges = attr_edges + triple_edges

        with self.lock:
            for node_id, node_data in all_nodes:
                self.graph.add_node(node_id, **node_data)
            for edge_data in all_edges:
                u, v, relation, edge_attrs = edge_data
                self.graph.add_edge(u, v, relation=relation, **edge_attrs)
        logger.info(f"[SingleGraphBuilder] Chunk {chunk_id}: 创建节点 {len(all_nodes)} 个、边 {len(all_edges)} 条")

    def _process_chunk_impl(self, chunk_id: str, chunk_text: str) -> None:
        """单阶段实现：调用 process_level1_level2"""
        self.process_level1_level2(chunk_text, chunk_id)
