"""
社区与社区报告质量评估：基于 LLM 对社区报告中的每个社区在五个维度上打分并给出 evidence，结果保存为 JSON。

维度：主题一致性、可解释性、覆盖度、冗余度、与原文的吻合度。
输入：社区报告 JSON（包含各社区的 chunk_ids）、chunks.txt（图谱抽取原文）。
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


# 默认 chunk 长度限制
DEFAULT_MAX_CHARS_PER_CHUNK = 2000
DEFAULT_MAX_TOTAL_CHUNK_CHARS = 50000


def _load_prompt_template(prompt_path: Optional[str] = None) -> str:
    """加载评估用 prompt 模板"""
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    default_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "community_quality_evaluation.txt",
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


def _build_chunks_text(chunks_map: Dict[str, str], max_total_chars: int = DEFAULT_MAX_TOTAL_CHUNK_CHARS) -> str:
    """将 chunks 拼接成一段文本，总长度受 max_total_chars 限制"""
    if not chunks_map:
        return "(No chunk context available)"
    lines: List[str] = []
    total = 0
    for cid, text in list(chunks_map.items())[:100]:
        block = f"[Chunk {cid}]\n{text}\n\n"
        if total + len(block) > max_total_chars:
            remaining = max_total_chars - total
            if remaining > 100:
                lines.append(f"[Chunk {cid}]\n{text[:remaining]}... [truncated]\n\n")
            break
        lines.append(block)
        total += len(block)
    return "".join(lines).strip() if lines else "(No chunk context)"


def run_community_evaluation(
    report_path: str,
    chunks_path: str,
    output_path: Optional[str] = None,
    prompt_path: Optional[str] = None,
    max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK,
    max_total_chunk_chars: int = DEFAULT_MAX_TOTAL_CHUNK_CHARS,
) -> Dict[str, Any]:
    """
    对社区报告中的每个社区做四维质量评估（LLM 打分 + evidence），结果写入 JSON。

    Args:
        report_path: 社区报告 JSON 路径（含 community_report 数组）
        chunks_path: chunks.txt 路径
        output_path: 评估结果 JSON 输出路径，不指定则用 report_path 同目录下的 *_evaluation.json
        prompt_path: LLM prompt 模板路径
        max_chars_per_chunk: 每个 chunk 最大字符数
        max_total_chunk_chars: 传入 LLM 的 chunk 总字符上限

    Returns:
        含 evaluation 数组与 stats 的字典
    """
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = data.get("community_report", data.get("community_reports", []))
    if not isinstance(report, list):
        report = []
    stats_from_report = data.get("stats", {})

    if not report:
        logger.warning("Community report is empty, nothing to evaluate")
        result = {
            "evaluation": [],
            "stats": {
                "num_communities": 0,
                "avg_scores": {},
                "source_report": report_path,
                "source_stats": stats_from_report,
            },
        }
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    chunks_map = _load_chunks_map(chunks_path, max_chars_per_chunk=max_chars_per_chunk)
    chunks_text = _build_chunks_text(chunks_map, max_total_chars=max_total_chunk_chars)

    # 精简报告内容再喂给 LLM，避免过长（保留 chunk_ids 方便评估与原文吻合度）
    report_for_llm = []
    for r in report:
        report_for_llm.append({
            "community_id": r.get("community_id"),
            "name": r.get("name", ""),
            "summary": r.get("summary", ""),
            "size": r.get("size", 0),
            "members": r.get("members", [])[:30],
            "top_relations": r.get("top_relations", [])[:10],
            "chunk_ids": r.get("chunk_ids", []),
        })

    template = _load_prompt_template(prompt_path)
    community_report_json = json.dumps(report_for_llm, ensure_ascii=False, indent=2)
    prompt = template.replace("{community_report_json}", community_report_json)
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
                logger.warning(f"LLM 社区评估调用失败 (尝试 {attempt + 1}/{max_retries}): {e}, {retry_delay}秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error(f"LLM 社区评估调用失败 (已重试 {max_retries} 次): {e}")
    
    if parsed is None:
        result = {
            "evaluation": [],
            "stats": {"num_communities": len(report), "error": str(last_error), "source_report": report_path},
        }
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    evaluation = parsed.get("evaluation", [])
    if not isinstance(evaluation, list):
        evaluation = []

    # 计算平均分
    dims = ("topic_coherence", "interpretability", "coverage", "redundancy", "source_alignment")
    sums: Dict[str, float] = {d: 0.0 for d in dims}
    counts: Dict[str, int] = {d: 0 for d in dims}
    for item in evaluation:
        s = item.get("scores") or {}
        for d in dims:
            v = s.get(d)
            if v is not None and isinstance(v, (int, float)):
                sums[d] += float(v)
                counts[d] += 1
    avg_scores = {
        d: round(sums[d] / counts[d], 2) if counts[d] else None
        for d in dims
    }

    result = {
        "evaluation": evaluation,
        "stats": {
            "num_communities": len(evaluation),
            "avg_scores": avg_scores,
            "source_report": report_path,
            "chunks_path": chunks_path,
            "source_stats": stats_from_report,
        },
    }

    if output_path is None:
        base = os.path.splitext(os.path.basename(report_path))[0]
        out_dir = os.path.dirname(report_path)
        output_path = os.path.join(out_dir, f"{base}_evaluation.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Community evaluation saved to %s", output_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="社区报告质量评估（LLM 五维打分 + evidence）")
    parser.add_argument("report", help="社区报告 JSON 路径（*_community_report.json）")
    parser.add_argument("chunks", help="chunks.txt 路径")
    parser.add_argument("-o", "--output", default=None, help="评估结果 JSON 输出路径")
    parser.add_argument("--prompt", default=None, help="Prompt 模板路径")
    parser.add_argument("--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    parser.add_argument("--max-total-chunk-chars", type=int, default=DEFAULT_MAX_TOTAL_CHUNK_CHARS)
    args = parser.parse_args()

    run_community_evaluation(
        report_path=args.report,
        chunks_path=args.chunks,
        output_path=args.output,
        prompt_path=args.prompt,
        max_chars_per_chunk=args.max_chars_per_chunk,
        max_total_chunk_chars=args.max_total_chunk_chars,
    )
