"""
阶段4：验证和增强
对提取的三元组和属性进行质量评估，删除低质量项
"""

import json
import os
import sys
import ast
from typing import Any, Dict, List, Optional, Tuple

# 添加项目根目录到 Python 路径，以便导入模块
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import json_repair
import tiktoken

from utils import call_llm_api
from utils.logger import logger
from scorers.llm_scorer import LLMScorer


class Stage4Validation:
    """阶段4：验证和增强"""

    def __init__(self):
        """初始化验证器"""
        try:
            self.llm_client = call_llm_api.LLMCompletionCall()
        except Exception:
            logger.error("无法初始化 LLM 客户端")
            raise
        self.llm_scorer = LLMScorer()
        self.token_len = 0

        # 加载外部 prompt 模板（从项目根目录下的 prompts/staged/ 中读取）
        self.node_accuracy_template = self._load_prompt_template(
            "stage4_node_accuracy.txt"
        )
        self.triple_support_template = self._load_prompt_template(
            "stage4_triple_support.txt"
        )

    def _load_prompt_template(self, filename: str) -> str:
        """从 prompts/staged 目录加载 Stage4 使用的 prompt 模板。"""
        # 当前文件位置: prompt2graph/staged_extraction/stage4_validation.py
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        template_path = os.path.join(
            project_root, "prompts", "staged", filename
        )
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Stage4Validation: 已加载 prompt 模板: {template_path}")
            return content
        except FileNotFoundError:
            logger.error(f"Stage4Validation: prompt 模板文件不存在: {template_path}")
            raise
        except Exception as e:
            logger.error(
                f"Stage4Validation: 加载 prompt 模板失败 {template_path}: {type(e).__name__}: {e}"
            )
            raise

    def _token_cal(self, text: str) -> int:
        """计算 token 数量"""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _extract_with_llm(self, prompt: str) -> str:
        """调用 LLM 并返回修复后的 JSON 字符串"""
        response = self.llm_client.call_api(prompt)
        parsed_dict = json_repair.loads(response)
        return json.dumps(parsed_dict, ensure_ascii=False)

    def _score_node_accuracy(
        self,
        node_name: str,
        source_text: str,
        evidence_text: str,
        chunk_text: str,
        max_retries: int = 3,
    ) -> Tuple[float, str]:
        """
        评估提取的节点准确性。

        返回:
            (score, reasoning)
        """
        # 截断上下文，避免 prompt 过长
        truncated_chunk = (
            chunk_text[:2000] if len(chunk_text) > 2000 else chunk_text
        )
        scoring_prompt = (
            self.node_accuracy_template.replace("__NODE_NAME__", node_name)
            .replace("__SOURCE_TEXT__", source_text)
            .replace("__EVIDENCE_TEXT__", evidence_text)
            .replace("__CHUNK_TEXT__", truncated_chunk)
        )

        for attempt in range(max_retries):
            try:
                response = self._extract_with_llm(scoring_prompt)
                parsed = json_repair.loads(response)
                score = float(parsed.get("score", 0.5))
                score = max(0.0, min(1.0, score))
                reasoning = str(parsed.get("reasoning", "")).strip()
                self.token_len += self._token_cal(scoring_prompt + response)
                logger.debug(
                    f"节点准确性评分: {node_name} = {score:.2f}, reason={reasoning or 'N/A'}"
                )
                return score, reasoning
            except Exception as e:
                logger.warning(
                    f"节点准确性评分失败 ({node_name})，尝试 {attempt + 1}/{max_retries}: {e}"
                )
                if attempt >= max_retries - 1:
                    logger.error(
                        f"节点准确性评分达到最大重试次数 ({max_retries})，返回 -1"
                    )
                    return -1.0, ""
        return -1.0, ""

    def _score_triple_support(
        self,
        start_node: str,
        relation: str,
        end_node: str,
        source_text: str,
        evidence_text: str,
        chunk_text: str,
        max_retries: int = 3,
    ) -> Tuple[float, str, float, float, str, str]:
        """
        评估source和evidence字段是否能够支持所提取的三元组关系。
        LLM 返回 accuracy_score、usefulness_score、accuracy_reasoning、usefulness_reasoning，
        本函数计算 score = min(accuracy_score, usefulness_score) 并返回。

        Returns:
            (score, combined_reasoning, accuracy_score, usefulness_score, accuracy_reasoning, usefulness_reasoning)
            失败时 score 为 -1.0，其余为默认空值。
        """
        truncated_chunk = chunk_text
        scoring_prompt = (
            self.triple_support_template.replace("__START_NODE__", start_node)
            .replace("__RELATION__", relation)
            .replace("__END_NODE__", end_node)
            .replace("__SOURCE_TEXT__", source_text)
            .replace("__EVIDENCE_TEXT__", evidence_text)
            .replace("__CHUNK_TEXT__", truncated_chunk)
        )

        for attempt in range(max_retries):
            try:
                response = self._extract_with_llm(scoring_prompt)
                parsed = json_repair.loads(response)
                acc_s = float(parsed.get("accuracy_score", 0.5))
                use_s = float(parsed.get("usefulness_score", 0.5))
                acc_s = max(0.0, min(1.0, acc_s))
                use_s = max(0.0, min(1.0, use_s))
                score = min(acc_s, use_s)
                acc_r = str(parsed.get("accuracy_reasoning", "")).strip()
                use_r = str(parsed.get("usefulness_reasoning", "")).strip()
                combined_reasoning = f"[Accuracy] {acc_r} [Usefulness] {use_r}" if (acc_r or use_r) else ""
                self.token_len += self._token_cal(scoring_prompt + response)
                logger.debug(
                    f"三元组支持/重要性评分: ({start_node}, {relation}, {end_node}) = {score:.2f} (acc={acc_s:.2f}, use={use_s:.2f})"
                )
                return score, combined_reasoning, acc_s, use_s, acc_r, use_r
            except Exception as e:
                logger.warning(
                    f"三元组支持/重要性评分失败 ({start_node}, {relation}, {end_node})，"
                    f"尝试 {attempt + 1}/{max_retries}: {e}"
                )
                if attempt >= max_retries - 1:
                    logger.error(
                        f"三元组支持/重要性评分达到最大重试次数 ({max_retries})，返回 -1"
                    )
                    return -1.0, "", -1.0, -1.0, "", ""
        return -1.0, "", -1.0, -1.0, "", ""

    def validate_and_filter(
        self,
        chunk_id: str,
        chunk_text: str,
        triples: List[Dict[str, Any]],
        attributes: Dict[str, List[Dict[str, Any]]],
        min_triple_score: float = 0.5,
        min_node_score: float = 0.5,
        use_chunk_scoring: bool = True,
        use_node_accuracy_scoring: bool = True,
        use_triple_support_scoring: bool = True
    ) -> Dict[str, Any]:
        """
        验证和过滤三元组及属性，删除低质量项

        Args:
            chunk_id: chunk ID
            chunk_text: chunk文本
            triples: 三元组列表，每个元素包含 subject, relation, object, source, evidence
            attributes: 属性字典，格式 {entity: [{"key": ..., "value": ..., "source": ..., "evidence": ...}, ...]}
            min_triple_score: 三元组支持度最低分数阈值（低于此分数将被删除）
            min_node_score: 节点准确性最低分数阈值（低于此分数将被删除）
            use_chunk_scoring: 是否使用chunk级别的LLM打分（检查bad_cases）
            use_node_accuracy_scoring: 是否启用节点准确性打分（False时跳过节点打分，默认分数为-1.0）
            use_triple_support_scoring: 是否启用三元组支持度打分（False时跳过三元组打分，默认分数为-1.0）

        Returns:
            {
                "scores": {
                    "triple_details": {...},   # 所有三元组的详细评分信息（键为 "subject|relation|object"）
                    "attribute_details": {...},  # 所有属性的详细评分信息（键为实体名）
                    "removed": {"triples": [...], "attributes": [...]}
                },
                "triples": [...],   # 删除后的三元组列表（供下游直接使用）
                "attributes": {...}  # 删除后的属性字典（供下游直接使用）
            }
        """
        scores = {
            "triple_details": {},
            "attribute_details": {},
            "removed": {"triples": [], "attributes": []},
        }
        valid_triples: List[Dict[str, Any]] = []
        valid_attributes: Dict[str, List[Dict[str, Any]]] = {}

        # 1. 对三元组进行打分和过滤
        triple_details: Dict[str, Dict[str, Any]] = {}
        valid_triple_keys: set = set()
        for triple in triples:
            subject = triple.get("subject", "").strip()
            relation = triple.get("relation", "").strip()
            obj = triple.get("object", "").strip()
            source_text = triple.get("source", "")
            evidence_text = triple.get("evidence", "")

            if not subject or not relation or not obj:
                scores["removed"]["triples"].append(
                    {**triple, "reason": "missing required fields"}
                )
                continue

            # 评估起始节点准确性
            if use_node_accuracy_scoring:
                start_score, start_reason = self._score_node_accuracy(
                    subject, source_text, evidence_text, chunk_text
                )
            else:
                start_score, start_reason = -1.0, ""
            # 评估结束节点准确性
            if use_node_accuracy_scoring:
                end_score, end_reason = self._score_node_accuracy(
                    obj, source_text, evidence_text, chunk_text
                )
            else:
                end_score, end_reason = -1.0, ""
            # 评估三元组支持度 + 重要性（LLM 返回 accuracy/usefulness，代码中计算 min）
            if use_triple_support_scoring:
                triple_score, triple_reason, acc_score, use_score, acc_reason, use_reason = self._score_triple_support(
                    subject, relation, obj, source_text, evidence_text, chunk_text
                )
            else:
                triple_score, triple_reason = -1.0, ""
                acc_score, use_score, acc_reason, use_reason = -1.0, -1.0, "", ""

            # 使用字符串键（格式：subject|relation|object）以便 JSON 序列化
            triple_key = f"{subject}|{relation}|{obj}"
            detail: Dict[str, Any] = {
                "start_node": subject,
                "relation": relation,
                "end_node": obj,
                "source": source_text,
                "evidence": evidence_text,
            }
            if use_node_accuracy_scoring:
                detail["start_accuracy_score"] = start_score
                detail["end_accuracy_score"] = end_score
                detail["start_reason"] = start_reason
                detail["end_reason"] = end_reason
            if use_triple_support_scoring:
                detail["triple_support_score"] = triple_score
                detail["triple_reason"] = triple_reason
                detail["accuracy_score"] = acc_score
                detail["usefulness_score"] = use_score
                detail["accuracy_reasoning"] = acc_reason
                detail["usefulness_reasoning"] = use_reason
            triple_details[triple_key] = detail

            # 判断是否保留：根据开关决定是否检查对应分数
            # 如果开关为 False，则跳过对应分数的检查
            should_keep = True
            if use_node_accuracy_scoring:
                if start_score < min_node_score or start_score < 0:
                    should_keep = False
                if end_score < min_node_score or end_score < 0:
                    should_keep = False
            if use_triple_support_scoring:
                if triple_score < min_triple_score or triple_score < 0:
                    should_keep = False

            if should_keep:
                valid_triple_keys.add(triple_key)
            else:
                scores["removed"]["triples"].append(
                    {
                        **triple,
                        "reason": f"low scores (start={start_score:.2f}, "
                        f"end={end_score:.2f}, triple={triple_score:.2f})",
                    }
                )

        scores["triple_details"] = triple_details

        # 2. 对属性进行打分和过滤（每个属性单独评估）
        attribute_details: Dict[str, List[Dict[str, Any]]] = {}
        for entity_name, attr_list in attributes.items():
            for attr in attr_list:
                attr_key = attr.get("key", "")
                attr_value = attr.get("value", "")
                source_text = attr.get("source", "")
                evidence_text = attr.get("evidence", "")

                if not attr_key or not attr_value:
                    scores["removed"]["attributes"].append(
                        {**attr, "entity": entity_name, "reason": "missing key/value"}
                    )
                    continue

                # 评估实体节点准确性（每个属性单独评估，因 source/evidence 不同）
                if use_node_accuracy_scoring:
                    node_score, node_reason = self._score_node_accuracy(
                        entity_name, source_text, evidence_text, chunk_text
                    )
                else:
                    node_score, node_reason = -1.0, ""
                if entity_name not in attribute_details:
                    attribute_details[entity_name] = []

                # 评估属性支持度（把属性视作 has_attribute 关系的三元组，LLM 返回 accuracy/usefulness，代码中计算 min）
                if use_triple_support_scoring:
                    attr_score, attr_reason, acc_score, use_score, acc_reason, use_reason = self._score_triple_support(
                        entity_name,
                        "has_attribute",
                        f"{attr_key}: {attr_value}",
                        source_text,
                        evidence_text,
                        chunk_text,
                    )
                else:
                    attr_score, attr_reason = -1.0, ""
                    acc_score, use_score, acc_reason, use_reason = -1.0, -1.0, "", ""

                # 判断是否保留：根据开关决定是否检查对应分数
                should_keep_attr = True
                if use_node_accuracy_scoring:
                    if node_score < min_node_score or node_score < 0:
                        should_keep_attr = False
                if use_triple_support_scoring:
                    if attr_score < min_triple_score or attr_score < 0:
                        should_keep_attr = False

                # 记录详细 attribute 评分信息（仅包含 use_xxx_scoring=True 的字段）
                attr_detail: Dict[str, Any] = {
                    "key": attr_key,
                    "value": attr_value,
                    "source": source_text,
                    "evidence": evidence_text,
                }
                if use_node_accuracy_scoring:
                    attr_detail["node_score"] = node_score
                    attr_detail["node_reason"] = node_reason
                if use_triple_support_scoring:
                    attr_detail["attr_score"] = attr_score
                    # has_attribute 边的支持度推理（用于可视化展示）
                    attr_detail["triple_reason"] = attr_reason
                    attr_detail["accuracy_score"] = acc_score
                    attr_detail["usefulness_score"] = use_score
                    attr_detail["accuracy_reasoning"] = acc_reason
                    attr_detail["usefulness_reasoning"] = use_reason
                attribute_details[entity_name].append(attr_detail)

                if should_keep_attr:
                    if entity_name not in valid_attributes:
                        valid_attributes[entity_name] = []
                    valid_attributes[entity_name].append(attr)
                else:
                    scores["removed"]["attributes"].append(
                        {
                            **attr,
                            "entity": entity_name,
                            "reason": f"low scores (node={node_score:.2f}, "
                            f"attr={attr_score:.2f})",
                            "reasons": {
                                "node_reason": node_reason,
                                "triple_reason": attr_reason,
                            },
                        }
                    )

        scores["attribute_details"] = attribute_details

        # 3. Chunk级别的LLM打分（检查bad_cases）
        if use_chunk_scoring and valid_triple_keys:
            # 从原始 triples 中提取保留的三元组，转换为LLMScorer期望的格式
            triples_for_scorer = []
            triple_key_to_triple = {}
            for triple in triples:
                subject = triple.get("subject", "").strip()
                relation = triple.get("relation", "").strip()
                obj = triple.get("object", "").strip()
                if not subject or not relation or not obj:
                    continue
                triple_key = f"{subject}|{relation}|{obj}"
                if triple_key in valid_triple_keys:
                    triples_for_scorer.append(
                        {
                            "start_node": subject,
                            "relation": relation,
                            "end_node": obj,
                            "source": triple.get("source", ""),
                            "evidence": triple.get("evidence", ""),
                        }
                    )
                    triple_key_to_triple[triple_key] = triple

            if triples_for_scorer:
                chunk_score_result = self.llm_scorer.score_chunk_llm(
                    chunk_id, chunk_text, triples_for_scorer
                )
                bad_cases = chunk_score_result.get("bad_cases", [])

                # 根据bad_cases进一步过滤
                if bad_cases:
                    bad_case_reasons = {
                        f"{bc.get('start_node', '')}|{bc.get('relation', '')}|{bc.get('end_node', '')}": bc.get("reason", "unknown")
                        for bc in bad_cases
                    }
                    for triple_key in list(valid_triple_keys):
                        if triple_key in bad_case_reasons:
                            valid_triple_keys.remove(triple_key)
                            if triple_key in triple_key_to_triple:
                                scores["removed"]["triples"].append(
                                    {
                                        **triple_key_to_triple[triple_key],
                                        "reason": "bad_case from chunk scoring: "
                                        + bad_case_reasons[triple_key],
                                    }
                                )

        # 从原始 triples 中收集保留的三元组（按 valid_triple_keys），并附加 reason 字段
        for triple in triples:
            subject = triple.get("subject", "").strip()
            relation = triple.get("relation", "").strip()
            obj = triple.get("object", "").strip()
            if not subject or not relation or not obj:
                continue
            triple_key = f"{subject}|{relation}|{obj}"
            if triple_key in valid_triple_keys:
                # 从 triple_details 中获取 reason 字段并附加到三元组上
                if triple_key in triple_details:
                    detail = triple_details[triple_key]
                    if "start_accuracy_score" in detail:
                        triple["start_accuracy_score"] = detail["start_accuracy_score"]
                    if "end_accuracy_score" in detail:
                        triple["end_accuracy_score"] = detail["end_accuracy_score"]
                    if "start_reason" in detail:
                        triple["start_reason"] = detail["start_reason"]
                    if "end_reason" in detail:
                        triple["end_reason"] = detail["end_reason"]
                    if "triple_support_score" in detail:
                        triple["triple_support_score"] = detail["triple_support_score"]
                    if "triple_reason" in detail:
                        triple["triple_reason"] = detail["triple_reason"]
                    if "accuracy_score" in detail:
                        triple["accuracy_score"] = detail["accuracy_score"]
                    if "usefulness_score" in detail:
                        triple["usefulness_score"] = detail["usefulness_score"]
                    if "accuracy_reasoning" in detail:
                        triple["accuracy_reasoning"] = detail["accuracy_reasoning"]
                    if "usefulness_reasoning" in detail:
                        triple["usefulness_reasoning"] = detail["usefulness_reasoning"]
                valid_triples.append(triple)

        kept_triples_count = len(valid_triples)
        total_attributes_count = sum(len(attr_list) for attr_list in attributes.values())
        kept_attributes_count = sum(len(v) for v in valid_attributes.values())
        logger.info(
            f"Stage 4 validation for chunk {chunk_id}: "
            f"kept {kept_triples_count}/{len(triples)} triples, "
            f"{kept_attributes_count}/{total_attributes_count} attributes"
        )

        return {
            "scores": scores,
            "triples": valid_triples,
            "attributes": valid_attributes,
        }


if __name__ == "__main__":
    """
    简单调试入口（无需命令行参数）：

    手动在下方填写：
        CHUNK_ID                    : 需要调试的 chunk id（例如 "jicblhdj"）
        STAGE2_OUTPUT_PATH          : Stage2 关系抽取结果 JSON 路径（包含 "triples"）
        STAGE3_OUTPUT_PATH          : Stage3 属性抽取结果 JSON 路径（包含 "attributes"）
        CHUNK_PATH                  : chunk 文本所在的 txt 文件路径（例如 output/paper_mini/chunks.txt）
        USE_CHUNK_SCORING           : 是否使用 chunk 级别的 LLM 打分（检查 bad_cases）
        USE_NODE_ACCURACY_SCORING   : 是否启用节点准确性打分（False 时跳过节点打分，默认分数为 1.0）
        USE_TRIPLE_SUPPORT_SCORING  : 是否启用三元组支持度打分（False 时跳过三元组打分，默认分数为 1.0）
        OUTPUT_PATH                 : 结果保存路径（设为 None 则不保存，否则保存完整结果 JSON）

    然后直接在命令行执行：
        python3 staged_extraction/stage4_validation.py

    validate_and_filter 返回：
        {
          "scores": { "triple_details": {...}, "attribute_details": {...}, "removed": {...} },
          "triples": [...],    # 删除后的三元组（供下游使用）
          "attributes": {...}  # 删除后的属性（供下游使用）
        }
    本地仅保存 result["scores"] 到 OUTPUT_PATH。
    """

    # === 请根据实际情况修改这几个路径 ===
    CHUNK_ID = "jicblhdj"
    STAGE2_OUTPUT_PATH = "output/paper_mini/staged/jicblhdj_stage2.json"
    STAGE3_OUTPUT_PATH = "output/paper_mini/staged/jicblhdj_stage3.json"
    CHUNK_PATH = "output/paper_mini/chunks.txt"
    USE_CHUNK_SCORING = False  # 如果只想看 node/attribute 打分，可改为 False
    USE_NODE_ACCURACY_SCORING = False  # 是否启用节点准确性打分
    USE_TRIPLE_SUPPORT_SCORING = True  # 是否启用三元组支持度打分
    OUTPUT_PATH = "output/paper_mini/staged/jicblhdj_stage4_result.json"  # 结果保存路径（可选，设为 None 则不保存）

    # 1. 读取 Stage2 / Stage3 中间结果
    if not os.path.isfile(STAGE2_OUTPUT_PATH):
        raise FileNotFoundError(f"Stage2 输出文件不存在: {STAGE2_OUTPUT_PATH}")
    if not os.path.isfile(STAGE3_OUTPUT_PATH):
        raise FileNotFoundError(f"Stage3 输出文件不存在: {STAGE3_OUTPUT_PATH}")
    if not os.path.isfile(CHUNK_PATH):
        raise FileNotFoundError(f"chunk 文本文件不存在: {CHUNK_PATH}")

    with open(STAGE2_OUTPUT_PATH, "r", encoding="utf-8") as f2:
        stage2_data = json.load(f2)
    with open(STAGE3_OUTPUT_PATH, "r", encoding="utf-8") as f3:
        stage3_data = json.load(f3)

    triples = stage2_data.get("triples") or []
    attributes = stage3_data.get("attributes") or {}

    if not isinstance(triples, list):
        raise ValueError("Stage2 输出中的字段 'triples' 必须是 list。")
    if not isinstance(attributes, dict):
        raise ValueError("Stage3 输出中的字段 'attributes' 必须是 dict。")

    # 2. 从 chunk txt 文件中提取对应 chunk_id 的文本
    chunk_text = ""
    with open(CHUNK_PATH, "r", encoding="utf-8") as cf:
        for line in cf:
            line = line.rstrip("\n\r")
            if "\n" in line or "\r" in line:
                # 防止意外多行，简单清理
                line = line.strip()
            if "\t" not in line:
                continue
            parts = line.split("\t", 1)
            if (
                len(parts) != 2
                or not parts[0].startswith("id: ")
                or not parts[1].startswith("Chunk: ")
            ):
                continue
            line_chunk_id = parts[0][4:]
            if line_chunk_id != CHUNK_ID:
                continue
            chunk_data_str = parts[1][7:]
            try:
                chunk_obj = ast.literal_eval(chunk_data_str)
            except Exception as e:
                raise ValueError(
                    f"解析 chunk 行失败 (id={line_chunk_id}): {e}"
                ) from e
            if isinstance(chunk_obj, dict):
                chunk_text = str(chunk_obj.get("text", "")) or ""
            break

    if not chunk_text:
        raise ValueError(f"在 {CHUNK_PATH} 中未找到 chunk_id={CHUNK_ID} 对应的文本。")

    # 3. 调用 Stage4 验证
    validator = Stage4Validation()
    result = validator.validate_and_filter(
        chunk_id=CHUNK_ID,
        chunk_text=chunk_text,
        triples=triples,
        attributes=attributes,
        use_chunk_scoring=USE_CHUNK_SCORING,
        use_node_accuracy_scoring=USE_NODE_ACCURACY_SCORING,
        use_triple_support_scoring=USE_TRIPLE_SUPPORT_SCORING
    )

    # 4. 仅保存 scores（triple_details + attribute_details + removed）到本地
    if OUTPUT_PATH:
        output_dir = os.path.dirname(OUTPUT_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result["scores"], f, ensure_ascii=False, indent=2)
        logger.info(f"Stage 4 验证 scores 已保存到: {OUTPUT_PATH}")
    # 可选：打印完整结果（含 triples/attributes）用于调试
    # print(json.dumps(result, ensure_ascii=False, indent=2))
