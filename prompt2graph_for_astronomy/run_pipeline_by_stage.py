#!/usr/bin/env python3
"""
分 Stage 批量跑 Pipeline

特点：
- Stage 1-3: 64 并发
- Stage 4: 48 并发（Stage 4 是 token 黑洞，略降低）
- 每 Stage 结束后自动检查缺失并重跑（最多 3 轮重试）
- 重试之间有退避延迟，让 API 冷却
- 最后汇总构建图谱

用法：
    python3 run_pipeline_by_stage.py --config configs/test_paper25.yml --output output/paper25/20260423101417

或者只跑某个 stage：
    python3 run_pipeline_by_stage.py --config configs/test_paper25.yml --output output/paper25/20260423101417 --stage 4
"""

import argparse
import json
import os
import sys
import time
from concurrent import futures
from typing import Any, Dict, List, Optional, Tuple

import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from staged_extraction.stage1_entity_recognition import Stage1EntityRecognition
from staged_extraction.stage2_relation_extraction import Stage2RelationExtraction
from staged_extraction.stage3_attribute_extraction import Stage3AttributeExtraction
from staged_extraction.stage4_validation import Stage4Validation
from graph_builder import GraphBuilder
from utils.logger import logger

try:
    from community_clustering import run_community_clustering
    HAS_COMMUNITY_CLUSTERING = True
except ImportError:
    HAS_COMMUNITY_CLUSTERING = False


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    """解析路径：如果是相对路径，基于项目根目录"""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


# ==================== 配置 ====================
STAGE1_WORKERS = 64
STAGE2_WORKERS = 64
STAGE3_WORKERS = 64
STAGE4_WORKERS = 48

MAX_RETRY_ROUNDS = 3          # 每 stage 最多重试轮数
RETRY_BACKOFF_SECONDS = 5     # 重试轮次之间的退避时间
STAGE4_RETRY_BACKOFF = 10     # Stage 4 重试退避更长（因为 token 消耗大）


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(schema_name: str) -> Dict[str, Any]:
    schema_path = _resolve_path(os.path.join("schemas", f"{schema_name}.json"))
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema 文件不存在: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_chunks(chunks_path: str) -> Dict[str, str]:
    """读取 chunks.txt，返回 {chunk_id: chunk_text}"""
    chunks = {}
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if "\t" not in line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2 or not parts[0].startswith("id: ") or not parts[1].startswith("Chunk: "):
                continue
            chunk_id = parts[0][4:]
            chunk_data_str = parts[1][7:]
            try:
                chunk_data = eval(chunk_data_str)
                if isinstance(chunk_data, dict):
                    chunks[chunk_id] = chunk_data.get("text", "")
            except Exception:
                continue
    return chunks


def save_stage_output(stage_dir: str, chunk_id: str, stage_name: str, data: Dict[str, Any]):
    os.makedirs(stage_dir, exist_ok=True)
    file_path = os.path.join(stage_dir, f"{chunk_id}_{stage_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_stage_output(stage_dir: str, chunk_id: str, stage_name: str) -> Optional[Dict[str, Any]]:
    file_path = os.path.join(stage_dir, f"{chunk_id}_{stage_name}.json")
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_stage_with_retry(
    stage_name: str,
    chunk_ids: List[str],
    chunks_map: Dict[str, str],
    process_fn,
    max_workers: int,
    stage_dir: str,
    max_retry_rounds: int = MAX_RETRY_ROUNDS,
    retry_backoff: int = RETRY_BACKOFF_SECONDS,
):
    """
    跑一个 Stage，带自动重试机制。
    process_fn(chunk_id, chunk_text) -> 结果 dict
    """
    print(f"\n{'='*60}")
    print(f"Round: {stage_name} (workers={max_workers})")
    print(f"{'='*60}")

    total = len(chunk_ids)
    remaining = set(chunk_ids)
    round_num = 0

    while remaining and round_num <= max_retry_rounds:
        if round_num > 0:
            print(f"\n>>> 第 {round_num} 轮重试: {len(remaining)} 个 chunks 待重跑")
            print(f">>> 等待 {retry_backoff} 秒让 API 冷却...")
            time.sleep(retry_backoff)

        current_batch = sorted(list(remaining))
        succeeded = set()
        failed = set()

        start_time = time.time()
        with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cid = {
                executor.submit(process_fn, cid, chunks_map[cid]): cid
                for cid in current_batch
            }
            for future in futures.as_completed(future_to_cid):
                cid = future_to_cid[future]
                try:
                    result = future.result()
                    if result is not None:
                        save_stage_output(stage_dir, cid, stage_name.lower().replace(" ", ""), result)
                        succeeded.add(cid)
                    else:
                        failed.add(cid)
                except Exception as e:
                    logger.error(f"Chunk {cid}: {stage_name} 失败: {e}")
                    failed.add(cid)

        elapsed = time.time() - start_time
        remaining = failed
        round_num += 1

        print(f"  本轮成功: {len(succeeded)}, 失败: {len(failed)}, 耗时: {elapsed:.1f}s")

    if remaining:
        print(f"\n⚠️ 警告: {stage_name} 最终仍有 {len(remaining)} 个 chunks 失败，已记录到日志")
        with open(os.path.join(stage_dir, f"{stage_name.lower().replace(' ', '_')}_failed.txt"), "w") as f:
            for cid in sorted(remaining):
                f.write(cid + "\n")
    else:
        print(f"✅ {stage_name} 全部完成！")

    return total - len(remaining)


def build_graph_from_stages(
    chunk_ids: List[str],
    chunks_map: Dict[str, str],
    stage_dir: str,
    schema: Dict[str, Any],
    enable_stage4: bool,
    stage4_cfg: Dict[str, Any],
    output_path: str,
    community_clustering_cfg: Optional[Dict[str, Any]] = None,
):
    """读取所有 stage 结果，构建图谱，并可选执行社区聚类"""
    print(f"\n{'='*60}")
    print("Round 5: 汇总构建图谱")
    print(f"{'='*60}")

    from staged_graph_builder import StagedGraphBuilder

    builder = StagedGraphBuilder(
        schema_content=schema,
        enable_stage4_validation=enable_stage4,
        save_stage_outputs=False,  # 已经保存过了
        stage4_min_triple_score=stage4_cfg.get("min_triple_score", 0.5),
        stage4_min_node_score=stage4_cfg.get("min_node_score", 0.5),
        stage4_use_chunk_scoring=stage4_cfg.get("use_chunk_scoring", True),
        stage4_use_node_accuracy_scoring=stage4_cfg.get("use_node_accuracy_scoring", True),
        stage4_use_triple_support_scoring=stage4_cfg.get("use_triple_support_scoring", True),
    )

    # 逐个 chunk 读取 staged 结果并构建图
    for cid in chunk_ids:
        chunk_text = chunks_map.get(cid, "")

        stage1 = load_stage_output(stage_dir, cid, "stage1")
        stage2 = load_stage_output(stage_dir, cid, "stage2")
        stage3 = load_stage_output(stage_dir, cid, "stage3")
        stage4 = load_stage_output(stage_dir, cid, "stage4") if enable_stage4 else None

        if not stage1 or not stage1.get("entities"):
            continue
        if not stage2:
            continue
        if not stage3:
            continue

        # 如果有 stage4，用 stage4 的评分 + stage2/3 的原始数据（stage4 文件只保存了 scores）
        if stage4 and enable_stage4:
            triple_scores = stage4.get("triple_details", {})
            attr_scores = stage4.get("attribute_details", {})
        else:
            triple_scores = {}
            attr_scores = {}
        stage2_for_build = stage2
        stage3_for_build = stage3

        builder._merge_and_build_graph_staged(
            cid, chunk_text, stage1, stage2_for_build, stage3_for_build, triple_scores, attr_scores
        )

    # 去重
    builder.triple_deduplicate()

    # 格式化并保存
    output = builder.format_output()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"✅ 图谱已保存: {output_path} ({len(output)} triples)")

    # 保存 CID 查询报告（如果有）
    cid_report = builder._save_cid_report(os.path.dirname(output_path) or ".")
    if cid_report:
        print(f"✅ CID 报告已保存: {cid_report}")

    # ==================== 社区聚类 ====================
    clustering_cfg = community_clustering_cfg or {}
    if clustering_cfg.get("method") == "leiden" and HAS_COMMUNITY_CLUSTERING:
        print(f"\n{'='*60}")
        print("Round 6: Leiden 社区聚类")
        print(f"{'='*60}")
        llm_prompt_path = clustering_cfg.get("llm_prompt_path")
        try:
            cluster_result = run_community_clustering(
                input_path=output_path,
                output_dir=os.path.dirname(output_path) or ".",
                max_cluster_size=clustering_cfg.get("max_cluster_size", 1000),
                use_lcc=clustering_cfg.get("use_lcc", False),
                seed=clustering_cfg.get("seed", 42),
                llm_prompt_path=_resolve_path(llm_prompt_path) if llm_prompt_path else None,
            )
            stats = cluster_result.get("stats", {})
            print(
                f"✅ 社区聚类完成: 社区数={stats.get('communities', 0)}, "
                f"节点数={stats.get('nodes', 0)}, 边数={stats.get('edges', 0)}"
            )
        except Exception as e:
            print(f"⚠️ 社区聚类失败: {e}")
    elif clustering_cfg.get("method") == "leiden" and not HAS_COMMUNITY_CLUSTERING:
        print("⚠️ 跳过社区聚类: graspologic 未安装 (pip install graspologic)")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="分 Stage 批量跑 Pipeline")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--output", required=True, help="输出目录（如 output/paper25/20260423101417）")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="只跑某个 stage（1-4）或汇总（5），不指定则跑全部")
    parser.add_argument("--chunks", default=None, help="chunks.txt 路径，默认从 output 目录找")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个 chunks（测试用）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output
    stage_dir = os.path.join(output_dir, "staged")

    # 加载 schema
    schema_cfg = cfg.get("schema", {})
    schema = load_schema(schema_cfg.get("name", "260114"))

    # 加载 chunks
    chunks_path = args.chunks
    if not chunks_path:
        # 自动找 chunks_doc_0000.txt
        for fname in os.listdir(output_dir):
            if fname.startswith("chunks_doc_") and fname.endswith(".txt"):
                chunks_path = os.path.join(output_dir, fname)
                break
    if not chunks_path or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"找不到 chunks 文件: {chunks_path}")

    chunks_map = load_chunks(chunks_path)
    chunk_ids = sorted(list(chunks_map.keys()))
    if args.limit:
        chunk_ids = chunk_ids[:args.limit]
        print(f"加载了 {len(chunk_ids)} 个 chunks（受 --limit={args.limit} 限制）")
    else:
        print(f"加载了 {len(chunk_ids)} 个 chunks")

    # 加载 examples（stage1/stage2 的 example prompt）
    extraction_cfg = cfg.get("extraction", {})
    examples = {}
    # 简化：从 staged_customized 目录加载
    for stage_key in ["stage1", "stage2"]:
        example_file = _resolve_path(os.path.join("prompts", "staged_customized", f"{stage_key}_example.json"))
        if os.path.exists(example_file):
            with open(example_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data:
                    first_key = list(data.keys())[0]
                    examples[stage_key] = json.dumps(data[first_key], ensure_ascii=False, indent=2)

    # 初始化 Stage 处理器
    prompt_paths = extraction_cfg.get("prompt_paths") or {}
    stage1_prompt = _resolve_path(prompt_paths["stage1"]) if prompt_paths.get("stage1") else None
    stage2_prompt = _resolve_path(prompt_paths["stage2"]) if prompt_paths.get("stage2") else None
    stage3_prompt = _resolve_path(prompt_paths["stage3"]) if prompt_paths.get("stage3") else None
    stage1_proc = Stage1EntityRecognition(schema=schema, prompt_path=stage1_prompt)
    stage2_proc = Stage2RelationExtraction(schema=schema, prompt_path=stage2_prompt)
    stage3_proc = Stage3AttributeExtraction(schema=schema, prompt_path=stage3_prompt)

    enable_stage4 = extraction_cfg.get("use_stage4_validation", False)
    stage4_cfg = extraction_cfg.get("stage4", {})
    stage4_proc = Stage4Validation() if enable_stage4 else None

    # ==================== Round 1: Stage 1 ====================
    if args.stage is None or args.stage == 1:
        def process_stage1(cid, text):
            vars_ = {"examples": examples.get("stage1", "")} if examples.get("stage1") else None
            return stage1_proc.extract(text, variables=vars_)

        run_stage_with_retry(
            "Stage 1", chunk_ids, chunks_map, process_stage1,
            STAGE1_WORKERS, stage_dir
        )

    # ==================== Round 2: Stage 2 ====================
    if args.stage is None or args.stage == 2:
        def process_stage2(cid, text):
            stage1_data = load_stage_output(stage_dir, cid, "stage1")
            if not stage1_data:
                return None
            vars_ = {"examples": examples.get("stage2", "")} if examples.get("stage2") else None
            return stage2_proc.extract(text, stage1_data, variables=vars_)

        run_stage_with_retry(
            "Stage 2", chunk_ids, chunks_map, process_stage2,
            STAGE2_WORKERS, stage_dir
        )

    # ==================== Round 3: Stage 3 ====================
    if args.stage is None or args.stage == 3:
        def process_stage3(cid, text):
            stage1_data = load_stage_output(stage_dir, cid, "stage1")
            if not stage1_data:
                return None
            return stage3_proc.extract(text, stage1_data)

        run_stage_with_retry(
            "Stage 3", chunk_ids, chunks_map, process_stage3,
            STAGE3_WORKERS, stage_dir
        )

    # ==================== Round 4: Stage 4 ====================
    if enable_stage4 and (args.stage is None or args.stage == 4):
        def process_stage4(cid, text):
            stage2_data = load_stage_output(stage_dir, cid, "stage2")
            stage3_data = load_stage_output(stage_dir, cid, "stage3")
            if not stage2_data or not stage3_data:
                return None

            result = stage4_proc.validate_and_filter(
                chunk_id=cid,
                chunk_text=text,
                triples=stage2_data.get("triples", []),
                attributes=stage3_data.get("attributes", {}),
                min_triple_score=stage4_cfg.get("min_triple_score", 0.5),
                min_node_score=stage4_cfg.get("min_node_score", 0.5),
                use_chunk_scoring=stage4_cfg.get("use_chunk_scoring", True),
                use_node_accuracy_scoring=stage4_cfg.get("use_node_accuracy_scoring", True),
                use_triple_support_scoring=stage4_cfg.get("use_triple_support_scoring", True),
            )
            # 只保存 scores（与原始 pipeline 的 _save_stage_output 格式一致）
            return result["scores"]

        run_stage_with_retry(
            "Stage 4", chunk_ids, chunks_map, process_stage4,
            STAGE4_WORKERS, stage_dir,
            retry_backoff=STAGE4_RETRY_BACKOFF
        )

    # ==================== Round 5: 汇总构建图谱 ====================
    if args.stage is None or args.stage == 5:
        graph_output = os.path.join(output_dir, "graph_doc_0000.json")
        community_clustering_cfg = cfg.get("community_clustering", {})
        build_graph_from_stages(
            chunk_ids, chunks_map, stage_dir, schema,
            enable_stage4, stage4_cfg, graph_output,
            community_clustering_cfg=community_clustering_cfg,
        )

    print(f"\n{'='*60}")
    print("全部完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
