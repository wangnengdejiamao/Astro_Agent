"""
上层元图谱构建器：基于底层三元组图谱，通过 LLM 端到端抽取三元组间语义关系，构建以三元组为节点、语义关系为边的上层图谱。

输入：base_graph_path（底层图谱 JSON）、chunks_path（chunks.txt）
输出：meta_graph（节点=底层三元组，边=LLM 抽取的语义关系）

抽取策略：按 chunk_id 分组，每次仅将单个 chunk 及其关联三元组送入 LLM，降低上下文长度；各 chunk 抽取结果汇总后去重得到最终 meta-edges。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import json_repair
except ImportError:  # pragma: no cover - 可选依赖
    json_repair = None  # type: ignore

from utils import call_llm_api
from utils.logger import logger


# --- 默认参数 ---
DEFAULT_MAX_CHARS_PER_CHUNK = 2000
DEFAULT_MAX_TOTAL_CHUNK_CHARS = 40000


@dataclass
class TripleNode:
    """上层图谱的节点：代表一个底层实体-实体三元组"""

    id: str
    subject: str
    relation: str
    object: str
    schema_type_subj: Optional[str] = None
    schema_type_obj: Optional[str] = None
    source: Optional[str] = None
    evidence: Optional[str] = None
    chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "schema_type_subj": self.schema_type_subj,
            "schema_type_obj": self.schema_type_obj,
            "source": self.source,
            "evidence": self.evidence,
            "chunk_ids": self.chunk_ids,
        }


@dataclass
class MetaEdge:
    """上层图谱的边：三元组之间的语义关系（均由 LLM 抽取）"""

    source_triple_id: str
    target_triple_id: str
    relation: str
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_triple_id": self.source_triple_id,
            "target_triple_id": self.target_triple_id,
            "relation": self.relation,
            "evidence": self.evidence,
        }


def _triple_id(s: str, r: str, o: str) -> str:
    """生成三元组的唯一 ID（短 MD5）"""
    key = f"{s}|{r}|{o}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:16]


def _get_entity_name(node: Dict) -> Optional[str]:
    if not node or "properties" not in node:
        return None
    name = node.get("properties", {}).get("name")
    if isinstance(name, list) and name:
        return str(name[0]) if name[0] else None
    return str(name) if name else None


def _get_schema_type(node: Dict) -> Optional[str]:
    if not node or "properties" not in node:
        return None
    return node.get("properties", {}).get("schema_type")


def _is_entity_to_entity(edge: Dict) -> bool:
    """仅保留 entity→entity 的边，排除属性等"""
    start = edge.get("start_node", {})
    end = edge.get("end_node", {})
    return start.get("label") == "entity" and end.get("label") == "entity"


def parse_base_graph(edges: List[Dict]) -> List[TripleNode]:
    """
    从底层图谱边列表中解析出 entity-to-entity 三元组，构建 TripleNode 列表。
    """
    nodes: List[TripleNode] = []
    seen_ids: set[str] = set()

    for edge in edges:
        if not _is_entity_to_entity(edge):
            continue

        subj = _get_entity_name(edge.get("start_node"))
        obj = _get_entity_name(edge.get("end_node"))
        rel = edge.get("relation", "")
        if not subj or not obj or not rel:
            continue

        tid = _triple_id(subj, rel, obj)
        if tid in seen_ids:
            continue
        seen_ids.add(tid)

        chunk_ids = edge.get("chunk_id", [])
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]

        node = TripleNode(
            id=tid,
            subject=subj,
            relation=rel,
            object=obj,
            schema_type_subj=_get_schema_type(edge.get("start_node")),
            schema_type_obj=_get_schema_type(edge.get("end_node")),
            source=edge.get("source"),
            evidence=edge.get("evidence"),
            chunk_ids=chunk_ids,
        )
        nodes.append(node)

    return nodes


def _load_prompt_template(prompt_path: Optional[str] = None) -> str:
    """加载端到端抽取用的 LLM prompt 模板"""
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    default_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "meta_graph_e2e_extract.txt",
    )
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Prompt not found: {default_path}")
    with open(default_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_chunks_map(
    chunks_path: str,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> Dict[str, str]:
    """
    从 chunks.txt 加载 chunk_id -> chunk_text 的映射。
    可对每个 chunk 截断以控制长度。
    """
    mapping: Dict[str, str] = {}
    if not os.path.exists(chunks_path):
        logger.warning("Chunks file not found: %s", chunks_path)
        return mapping

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if "\t" not in line:
                    continue
                parts = line.split("\t", 1)
                if (
                    len(parts) != 2
                    or not parts[0].startswith("id: ")
                    or not parts[1].startswith("Chunk: ")
                ):
                    continue
                chunk_id = parts[0][4:]
                chunk_data_str = parts[1][7:]
                if not chunk_id:
                    continue
                try:
                    chunk_data = ast.literal_eval(chunk_data_str)
                except (ValueError, SyntaxError) as e:
                    logger.warning("解析 chunk %s 失败: %s", chunk_id, e)
                    continue
                if isinstance(chunk_data, dict):
                    text = (
                        chunk_data.get("text")
                        or chunk_data.get("chunk")
                        or ""
                    )
                    if text:
                        text = str(text)
                        if len(text) > max_chars_per_chunk:
                            text = text[:max_chars_per_chunk] + "... [truncated]"
                        mapping[str(chunk_id)] = text
    except Exception as e:
        logger.warning("Failed to read chunks file %s: %s", chunks_path, e)

    logger.info("Loaded %d chunks from %s", len(mapping), chunks_path)
    return mapping


def _group_triples_by_chunk_id(
    nodes: List[TripleNode],
    chunks_map: Dict[str, str],
) -> Dict[str, List[Tuple[TripleNode, int]]]:
    """
    按 chunk_id 将三元组分组。每个 chunk_id 对应该 chunk 中出现的三元组列表 (TripleNode, 全局idx)。
    三元组可属于多个 chunk，会出现在多个组中。
    """
    chunk_to_triples: Dict[str, List[Tuple[TripleNode, int]]] = {}
    for idx, n in enumerate(nodes):
        if not n.chunk_ids:
            continue
        for cid in n.chunk_ids:
            if cid not in chunks_map:
                continue
            if cid not in chunk_to_triples:
                chunk_to_triples[cid] = []
            chunk_to_triples[cid].append((n, idx))
    return chunk_to_triples


def _extract_edges_for_one_chunk(
    chunk_id: str,
    chunk_text: str,
    triples_with_global_idx: List[Tuple[TripleNode, int]],
    template: str,
    llm_client: Any,
) -> List[MetaEdge]:
    """
    对单个 chunk 及其关联的三元组调用 LLM，抽取三元组间语义关系。
    triples_with_global_idx: [(TripleNode, global_idx), ...]
    返回的 MetaEdge 使用 triple id（非 idx），便于后续汇总去重。
    """
    if len(triples_with_global_idx) < 2:
        return []

    # 本地 idx 0,1,2,... -> triple id
    id_by_local: Dict[int, str] = {i: n.id for i, (n, _) in enumerate(triples_with_global_idx)}

    triples_json_list = []
    for i, (n, _) in enumerate(triples_with_global_idx):
        triples_json_list.append({
            "idx": i,
            "id": n.id,
            "subject": n.subject,
            "relation": n.relation,
            "object": n.object,
            "source": (n.source or "")[:500],
            "chunk_ids": n.chunk_ids,
        })

    triples_json = json.dumps(triples_json_list, ensure_ascii=False, indent=2)
    chunks_text = f"[Chunk {chunk_id}]\n{chunk_text}"

    prompt = template.replace("{triples_json}", triples_json)
    prompt = prompt.replace("{chunks_text}", chunks_text)

    try:
        response = llm_client.call_api(prompt)
        if json_repair:
            parsed = json_repair.loads(response)
        else:
            parsed = json.loads(response)

        if not isinstance(parsed, list):
            logger.warning("Chunk %s: LLM 返回格式错误，期望 JSON array，得到 %s", chunk_id, type(parsed))
            return []

        edges: List[MetaEdge] = []
        for item in parsed:
            try:
                i = int(item.get("triple_i", -1))
                j = int(item.get("triple_j", -1))
                rel = (item.get("relation") or "").strip()
                reasoning = (item.get("reasoning") or "").strip()
            except (TypeError, ValueError):
                continue

            if i < 0 or j <= i or not rel:
                continue

            src_id = id_by_local.get(i)
            tgt_id = id_by_local.get(j)
            if not src_id or not tgt_id:
                continue

            edges.append(
                MetaEdge(
                    source_triple_id=src_id,
                    target_triple_id=tgt_id,
                    relation=rel,
                    evidence=reasoning or None,
                )
            )
        return edges

    except Exception as e:
        logger.warning("Chunk %s: LLM 调用失败: %s", chunk_id, e)
        return []


def extract_semantic_edges_e2e(
    nodes: List[TripleNode],
    chunks_map: Dict[str, str],
    prompt_path: Optional[str] = None,
) -> List[MetaEdge]:
    """
    按 chunk 分批调用 LLM：每个 chunk 仅与关联三元组一同输入，减少上下文长度。
    汇总各 chunk 抽取的 meta-edge 并去重，得到整份文档的最终结果。
    """
    if not nodes:
        return []

    chunk_to_triples = _group_triples_by_chunk_id(nodes, chunks_map)
    if not chunk_to_triples:
        logger.warning("无三元组的 chunk_id 在 chunks_map 中，跳过 LLM 抽取")
        return []

    template = _load_prompt_template(prompt_path)
    llm_client = call_llm_api.LLMCompletionCall()

    # 按 chunks_map 顺序遍历 chunk，保证输出可复现
    chunk_ids_ordered = [cid for cid in chunks_map if cid in chunk_to_triples]
    seen_edges: Set[Tuple[str, str]] = set()
    all_edges: List[MetaEdge] = []
    num_chunks_with_calls = 0

    for chunk_id in chunk_ids_ordered:
        triples_for_chunk = chunk_to_triples[chunk_id]
        if len(triples_for_chunk) < 2:
            continue

        num_chunks_with_calls += 1
        chunk_text = chunks_map[chunk_id]
        edges = _extract_edges_for_one_chunk(
            chunk_id=chunk_id,
            chunk_text=chunk_text,
            triples_with_global_idx=triples_for_chunk,
            template=template,
            llm_client=llm_client,
        )

        for e in edges:
            key = (e.source_triple_id, e.target_triple_id)
            if key not in seen_edges:
                seen_edges.add(key)
                all_edges.append(e)

    logger.info("按 chunk 分批抽取完成：%d 个 chunk 参与 LLM 调用，共 %d 条 meta-edge（去重后）", num_chunks_with_calls, len(all_edges))
    return all_edges


def build_meta_graph(
    base_graph_path: str,
    chunks_path: str,
    output_path: Optional[str] = None,
    prompt_path: Optional[str] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_total_chunk_chars: int = DEFAULT_MAX_TOTAL_CHUNK_CHARS,
) -> Dict[str, Any]:
    """
    从底层图谱构建上层元图谱（端到端 LLM 抽取，无规则筛选）。

    Args:
        base_graph_path: 底层图谱 JSON 路径
        chunks_path: chunks.txt 源文件路径，提供原文上下文
        output_path: 输出路径，不指定则只返回 dict 不写文件
        prompt_path: LLM prompt 模板路径，默认 prompts/meta_graph_e2e_extract.txt
        max_chars_per_chunk: 每个 chunk 最大字符数
        max_total_chunk_chars: 传入 LLM 的 chunk 总字符上限

    Returns:
        {
            "meta_graph": {"nodes": [...], "edges": [...]},
            "stats": {"num_triple_nodes": N, "num_edges": M}
        }
    """
    with open(base_graph_path, "r", encoding="utf-8") as f:
        edges_raw = json.load(f)

    if not isinstance(edges_raw, list):
        edges_raw = edges_raw.get("edges", edges_raw.get("triples", []))

    nodes = parse_base_graph(edges_raw)
    logger.info("Parsed %d entity-to-entity triples from base graph", len(nodes))

    chunks_map = _load_chunks_map(chunks_path, max_chars_per_chunk=max_chars_per_chunk)

    edges = extract_semantic_edges_e2e(
        nodes,
        chunks_map,
        prompt_path=prompt_path,
    )
    logger.info("E2E LLM extraction: %d meta-edges", len(edges))

    result = {
        "meta_graph": {
            "nodes": [n.to_dict() for n in nodes],
            "edges": [e.to_dict() for e in edges],
        },
        "stats": {
            "num_triple_nodes": len(nodes),
            "num_edges": len(edges),
        },
    }

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Meta graph saved to %s", output_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="端到端构建上层元图谱（LLM 直接抽取三元组间语义关系）")
    parser.add_argument("base_graph", help="底层图谱 JSON 路径")
    parser.add_argument("chunks", help="chunks.txt 路径")
    parser.add_argument("-o", "--output", default=None, help="输出 meta graph JSON 路径")
    parser.add_argument("--prompt", default=None, help="LLM prompt 模板路径")
    parser.add_argument("--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    parser.add_argument("--max-total-chunk-chars", type=int, default=DEFAULT_MAX_TOTAL_CHUNK_CHARS)
    args = parser.parse_args()

    result = build_meta_graph(
        base_graph_path=args.base_graph,
        chunks_path=args.chunks,
        output_path=args.output or args.base_graph.replace(".json", "_meta.json"),
        prompt_path=args.prompt,
        max_chars_per_chunk=args.max_chars_per_chunk,
        max_total_chunk_chars=args.max_total_chunk_chars,
    )
    print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
