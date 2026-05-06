"""
二级图谱（meta-graph）质量评估：基于 LLM 从原文一致性角度对 meta-graph 打分并给出 evidence，结果保存为 JSON。

评估维度：仅 source_alignment——所抽取的三元组间关系能否在原文中找到依据（用抽取出的关系反推原文是否支持，而非评估 evidence 文本是否与原文一致）。
输入：meta_graph.json、底层图谱 JSON、chunks.txt。
"""

from __future__ import annotations

import ast
import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    import json_repair
except ImportError:
    json_repair = None

from utils import call_llm_api
from utils.logger import logger


DEFAULT_MAX_CHARS_PER_CHUNK = 2000
DEFAULT_MAX_TOTAL_CHUNK_CHARS = 50000
DEFAULT_MAX_EDGES_IN_SUMMARY = 80
DEFAULT_MAX_NODES_IN_SUMMARY = 150


def _load_prompt_template(prompt_path: Optional[str] = None) -> str:
    """加载 meta-graph 评估用 prompt 模板"""
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    default_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "meta_graph_quality_evaluation.txt",
    )
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Prompt not found: {default_path}")
    with open(default_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_chunks_map(
    chunks_path: str,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
) -> Dict[str, str]:
    """从 chunks.txt 加载 chunk_id -> chunk_text。格式：id: <id>\\tChunk: {...}"""
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
                except (ValueError, SyntaxError):
                    continue
                if isinstance(chunk_data, dict):
                    text = chunk_data.get("text") or chunk_data.get("chunk") or ""
                    if text:
                        text = str(text)
                        if len(text) > max_chars_per_chunk:
                            text = text[:max_chars_per_chunk] + "... [truncated]"
                        mapping[str(chunk_id)] = text
    except Exception as e:
        logger.warning("Failed to read chunks file %s: %s", chunks_path, e)
    logger.info("Loaded %d chunks from %s", len(mapping), chunks_path)
    return mapping


def _build_chunks_text(
    chunks_map: Dict[str, str],
    referenced_chunk_ids: Optional[List[str]] = None,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHUNK_CHARS,
) -> str:
    """将 chunks 拼接成一段文本，优先包含 referenced_chunk_ids，总长度受 max_total_chars 限制"""
    if not chunks_map:
        return "(No chunk context available)"
    ids = list(referenced_chunk_ids) if referenced_chunk_ids else list(chunks_map.keys())
    if not ids:
        return "(No chunk context available)"
    lines: List[str] = []
    total = 0
    for cid in ids[:100]:
        text = chunks_map.get(cid)
        if not text:
            continue
        block = f"[Chunk {cid}]\n{text}\n\n"
        if total + len(block) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 100:
                lines.append(f"[Chunk {cid}]\n{text[:remaining]}... [truncated]\n\n")
            break
        lines.append(block)
        total += len(block)
    return "".join(lines).strip() if lines else "(No chunk context)"


def _build_meta_graph_summary(
    nodes: List[Dict],
    edges: List[Dict],
    max_nodes: int = DEFAULT_MAX_NODES_IN_SUMMARY,
    max_edges: int = DEFAULT_MAX_EDGES_IN_SUMMARY,
) -> Dict[str, Any]:
    """构建供 LLM 使用的 meta-graph 摘要：节点 id->triple 映射，以及边列表（含 source/target 的 triple 内容）"""
    id_to_triple: Dict[str, str] = {}
    for n in nodes[:max_nodes]:
        tid = n.get("id")
        if not tid:
            continue
        s = n.get("subject", "")
        r = n.get("relation", "")
        o = n.get("object", "")
        id_to_triple[str(tid)] = f"[{s} --{r}--> {o}]"

    edges_summary: List[Dict[str, Any]] = []
    for e in edges[:max_edges]:
        src_id = e.get("source_triple_id")
        tgt_id = e.get("target_triple_id")
        rel = e.get("relation", "")
        ev = e.get("evidence", "")
        if not src_id or not tgt_id:
            continue
        src_triple = id_to_triple.get(str(src_id), f"[id:{src_id}]")
        tgt_triple = id_to_triple.get(str(tgt_id), f"[id:{tgt_id}]")
        edges_summary.append({
            "source_triple_id": src_id,
            "target_triple_id": tgt_id,
            "source_triple": src_triple,
            "target_triple": tgt_triple,
            "relation": rel,
            "evidence": (ev or "")[:800],
        })

    return {
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "id_to_triple_sample": dict(list(id_to_triple.items())[:50]),
        "edges": edges_summary,
    }


def run_meta_graph_evaluation(
    meta_graph_path: str,
    base_graph_path: str,
    chunks_path: str,
    output_path: Optional[str] = None,
    prompt_path: Optional[str] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_total_chunk_chars: int = DEFAULT_MAX_TOTAL_CHUNK_CHARS,
    max_edges_in_summary: int = DEFAULT_MAX_EDGES_IN_SUMMARY,
    max_nodes_in_summary: int = DEFAULT_MAX_NODES_IN_SUMMARY,
) -> Dict[str, Any]:
    """
    对 meta-graph 做五维质量评估（LLM 打分 + evidence），结果写入 JSON。

    Args:
        meta_graph_path: meta-graph JSON 路径（含 meta_graph.nodes, meta_graph.edges）
        base_graph_path: 底层图谱 JSON 路径（用于参考，meta_graph 已含节点信息时可仅用于校验）
        chunks_path: chunks.txt 路径
        output_path: 评估结果 JSON 输出路径，不指定则用 meta_graph_path 同目录下的 *_meta_evaluation.json
        prompt_path: LLM prompt 模板路径
        max_chars_per_chunk: 每个 chunk 最大字符数
        max_total_chunk_chars: 传入 LLM 的 chunk 总字符上限
        max_edges_in_summary: 传入 LLM 的 meta-edge 数量上限
        max_nodes_in_summary: 传入 LLM 的 triple 节点数量上限

    Returns:
        含 evaluation 与 stats 的字典
    """
    with open(meta_graph_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    meta_graph = meta_data.get("meta_graph", meta_data)
    nodes = meta_graph.get("nodes", [])
    edges = meta_graph.get("edges", [])
    stats_from_meta = meta_data.get("stats", {})

    if not nodes and not edges:
        logger.warning("Meta-graph is empty, nothing to evaluate")
        result = {
            "evaluation": None,
            "stats": {
                "num_triple_nodes": 0,
                "num_edges": 0,
                "avg_scores": None,
                "source_meta_graph": meta_graph_path,
                "source_base_graph": base_graph_path,
                "source_chunks": chunks_path,
            },
        }
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    chunks_map = _load_chunks_map(chunks_path, max_chars_per_chunk=max_chars_per_chunk)
    referenced_ids: List[str] = []
    for n in nodes:
        for cid in n.get("chunk_ids", []):
            if cid and cid not in referenced_ids:
                referenced_ids.append(cid)
    chunks_text = _build_chunks_text(
        chunks_map,
        referenced_chunk_ids=referenced_ids,
        max_total_chars=max_total_chunk_chars,
    )

    meta_summary = _build_meta_graph_summary(
        nodes,
        edges,
        max_nodes=max_nodes_in_summary,
        max_edges=max_edges_in_summary,
    )
    meta_summary_json = json.dumps(meta_summary, ensure_ascii=False, indent=2)

    template = _load_prompt_template(prompt_path)
    prompt = template.replace("{meta_graph_summary_json}", meta_summary_json)
    prompt = prompt.replace("{chunks_text}", chunks_text)

    llm_client = call_llm_api.LLMCompletionCall()
    
    max_retries = 3
    retry_delay = 5
    last_error = None
    parsed = None
    for attempt in range(max_retries):
        try:
            response = llm_client.call_api(prompt)
            if json_repair:
                parsed = json_repair.loads(response)
            else:
                parsed = json.loads(response)
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(f"LLM meta-graph 评估调用失败 (尝试 {attempt + 1}/{max_retries}): {e}, {retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"LLM meta-graph 评估调用失败 (已重试 {max_retries} 次): {e}")
    
    if parsed is None:
        result = {
            "evaluation": None,
            "stats": {
                "num_triple_nodes": len(nodes),
                "num_edges": len(edges),
                "error": str(last_error),
                "source_meta_graph": meta_graph_path,
                "source_base_graph": base_graph_path,
                "source_chunks": chunks_path,
            },
        }
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    evaluation_raw = parsed.get("evaluation", parsed)
    # 期望 evaluation 是一个按 edge 给分的列表；若不是列表则回退为空列表
    if isinstance(evaluation_raw, list):
        edge_evaluations = evaluation_raw
    else:
        edge_evaluations = []

    # 计算整图的平均 source_alignment 分数，方便汇总分析
    total = 0.0
    count = 0
    for item in edge_evaluations:
        scores = item.get("scores") or {}
        v = scores.get("source_alignment")
        if v is not None and isinstance(v, (int, float)):
            total += float(v)
            count += 1
    avg_source_alignment = round(total / count, 2) if count else None

    result = {
        "evaluation": {
            "edges": edge_evaluations,
        },
        "stats": {
            "num_triple_nodes": len(nodes),
            "num_edges": len(edges),
            "avg_scores": {"source_alignment": avg_source_alignment},
            "source_meta_graph": meta_graph_path,
            "source_base_graph": base_graph_path,
            "source_chunks": chunks_path,
            "source_stats": stats_from_meta,
        },
    }

    if output_path is None:
        base = os.path.splitext(os.path.basename(meta_graph_path))[0]
        if base.endswith("_meta"):
            base = base[:-5]
        out_dir = os.path.dirname(meta_graph_path)
        output_path = os.path.join(out_dir, f"{base}_meta_evaluation.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Meta-graph evaluation saved to %s", output_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="二级图谱（meta-graph）质量评估（LLM 原文一致性 source_alignment 打分 + evidence）"
    )
    parser.add_argument("meta_graph", help="meta-graph JSON 路径（*_meta.json）")
    parser.add_argument("base_graph", help="底层图谱 JSON 路径")
    parser.add_argument("chunks", help="chunks.txt 路径")
    parser.add_argument("-o", "--output", default=None, help="评估结果 JSON 输出路径")
    parser.add_argument("--prompt", default=None, help="Prompt 模板路径")
    parser.add_argument("--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    parser.add_argument("--max-total-chunk-chars", type=int, default=DEFAULT_MAX_TOTAL_CHUNK_CHARS)
    parser.add_argument("--max-edges-in-summary", type=int, default=DEFAULT_MAX_EDGES_IN_SUMMARY)
    parser.add_argument("--max-nodes-in-summary", type=int, default=DEFAULT_MAX_NODES_IN_SUMMARY)
    args = parser.parse_args()

    run_meta_graph_evaluation(
        meta_graph_path=args.meta_graph,
        base_graph_path=args.base_graph,
        chunks_path=args.chunks,
        output_path=args.output,
        prompt_path=args.prompt,
        max_chars_per_chunk=args.max_chars_per_chunk,
        max_total_chunk_chars=args.max_total_chunk_chars,
        max_edges_in_summary=args.max_edges_in_summary,
        max_nodes_in_summary=args.max_nodes_in_summary,
    )
