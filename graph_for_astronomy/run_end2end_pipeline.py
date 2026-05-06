#!/usr/bin/env python3
"""
端到端管线脚本：

从原始语料 JSON（如 `input/paper_mini/corpus_cleaned.json`）
构建多阶段 / 单阶段抽取 + 实体消歧后的知识图谱。

推荐使用 YAML 配置文件集中管理参数，
命令行仅需提供一个 `config.yml` 路径，避免冗长的参数解析。

核心步骤：
1. chunk 切分（调用 `get_chunks.get_chunks`）
2. 构图（调用 `get_lowlevel_graph.build_lowlevel_graph`，默认多阶段提取）
3. 实体消歧（调用 `entity_deduplication.deduplicate_entities`）
4. 多篇语料时图谱合并（调用 `graph_merger.merge_graphs`）
5. 可选：上层元图谱构建（调用 `meta_graph_builder.build_meta_graph`）
6. 可选：元图谱质量评估（调用 `meta_graph_evaluation.run_meta_graph_evaluation`）
7. 可选：导入 Neo4j（`graph_to_neo4j.update_current_graph_and_import`）
8. 必选：社区聚类（Leiden 或 TreeComm，由 `community_clustering.method` 控制）
9. 可选：社区质量评估（调用 `community_evaluation.run_community_evaluation`）
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from pathlib import Path

from get_chunks import get_chunks
from get_lowlevel_graph import build_lowlevel_graph
from entity_deduplication import deduplicate_entities
from graph_merger import merge_graphs
from graph_to_neo4j import update_current_graph_and_import, import_meta_graph, import_community_evaluation
from community_clustering import run_community_clustering
from community_evaluation import run_community_evaluation
from meta_graph_builder import build_meta_graph
from meta_graph_evaluation import run_meta_graph_evaluation
from utils.logger import setup_logger, logger
try:
    from tree_comm import run_tree_comm_clustering
except Exception as exc:
    run_tree_comm_clustering = None  # type: ignore
    logger.warning("TreeComm clustering unavailable; install optional dependencies to enable it: %s", exc)
import logging
import neo4j


load_dotenv()


def _resolve_path(path: str) -> str:
    """将相对路径解析为以项目根目录为基准的绝对路径。"""
    if os.path.isabs(path):
        return path
    project_root = os.getenv("PROJECT_DIR") or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, path)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_unique_output_dir(base_dir: str) -> str:
    """
    生成唯一的输出目录路径，如果目录已存在则自动添加后缀
    例如: base_dir 已存在，则返回 base_dir_1，如果还存在则返回 base_dir_2，以此类推
    """
    if not os.path.exists(base_dir):
        return base_dir
    
    base_name = base_dir
    counter = 1
    while True:
        new_dir = f"{base_name}_{counter}"
        if not os.path.exists(new_dir):
            return new_dir
        counter += 1


def _load_corpus(corpus_path: str) -> List[dict]:
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"语料文件必须是 list，但实际类型为: {type(data)}")
    return data


def _load_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    配置示例结构见 docs/end2end_pipeline.md。
    """
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise ImportError(
            "需要安装 PyYAML 才能使用配置文件方式运行管线：\n"
            "  pip install pyyaml"
        ) from e

    cfg_path_abs = _resolve_path(config_path)
    if not os.path.exists(cfg_path_abs):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path_abs}")

    with open(cfg_path_abs, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容应为映射（dict），实际类型: {type(cfg)}")

    return cfg


def _setup_logging_from_config(logging_cfg: Dict[str, Any]) -> None:
    """根据配置设置日志级别。

    Args:
        logging_cfg: logging 配置段，例如 {"level": "INFO"} 或 {"level": "DEBUG"}
    """
    level_str = logging_cfg.get("level", "INFO").upper()
    
    # 映射字符串到 logging 级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    
    log_level = level_map.get(level_str, logging.INFO)
    
    # 设置所有相关模块的日志级别
    # 使用setup_logger重新配置已存在的logger
    module_loggers = [
        "graphrag",
        "prompt2graph",
        "utils.call_llm_api",
        "get_lowlevel_graph",
        "entity_deduplication",
        "graph_merger",
        "graph_to_neo4j",
        "community_clustering",
        "community_evaluation",
        "meta_graph_builder",
        "meta_graph_evaluation",
        "tree_comm",
        "utils.logger",
        "__main__",
    ]
    
    for module_name in module_loggers:
        setup_logger(module_name, level=log_level)
    
    # 设置根 logger（用于未指定名称的 logger）
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    # 同时更新所有已存在的handler
    for handler in root_logger.handlers:
        handler.setLevel(log_level)


def _example_value_to_prompt_str(val: Any) -> str:
    """将示例对象（多为 dict）转为可注入 prompt 的字符串。"""
    if isinstance(val, str):
        return val
    return json.dumps(val, ensure_ascii=False, indent=2)


def _load_examples_from_config(cfg: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    从配置中加载 examples。
    读取 prompts/staged_customized/stage1_example.json 和 stage2_example.json，
    根据配置中的 style 名称获取对应的 example 内容。

    cfg 格式：
    {
        "examples": {
            "stage1_style": "风格名称" 或 null（使用第一个）,
            "stage2_style": "风格名称" 或 null（使用第一个）,
            "user_design_stage1_style": "可选，合法 JSON 字符串，非空则优先于 stage1_style",
            "user_design_stage2_style": "可选，合法 JSON 字符串，非空则优先于 stage2_style"
        }
    }
    返回格式：{"stage1": "example content", "stage2": "example content"}
    """
    examples_cfg = cfg.get("examples")
    if not examples_cfg:
        return None

    examples: Dict[str, str] = {}
    project_root = os.getenv("PROJECT_DIR") or os.path.dirname(os.path.abspath(__file__))

    for stage_key, style_key in [("stage1", "stage1_style"), ("stage2", "stage2_style")]:
        user_design_raw = examples_cfg.get(f"user_design_{stage_key}_style")
        if user_design_raw is not None and str(user_design_raw).strip():
            try:
                parsed = json.loads(str(user_design_raw).strip())
                examples[stage_key] = _example_value_to_prompt_str(parsed)
                logger.info(f"{stage_key}: 使用自定义 user_design JSON")
                continue
            except json.JSONDecodeError as e:
                logger.warning(
                    f"{stage_key}: 自定义 user_design JSON 无效: {e}"
                )

        style_name = examples_cfg.get(style_key)
        example_file = os.path.join(project_root, "prompts", "staged_customized", f"{stage_key}_example.json")
        if not os.path.exists(example_file):
            logger.warning(f"Example file not found: {example_file}, skip {stage_key}")
            continue

        try:
            with open(example_file, "r", encoding="utf-8") as f:
                examples_data = json.load(f)

            if not style_name:
                first_key = next(iter(examples_data.keys()), None)
                if first_key:
                    examples[stage_key] = _example_value_to_prompt_str(examples_data[first_key])
                    logger.info(f"{stage_key}: 使用默认风格 '{first_key}'")
            else:
                if style_name not in examples_data:
                    logger.warning(f"Style '{style_name}' not found in {example_file}, skip {stage_key}")
                    continue
                examples[stage_key] = _example_value_to_prompt_str(examples_data[style_name])
                logger.info(f"{stage_key}: 使用风格 '{style_name}'")
        except Exception as e:
            logger.error(f"Failed to load {stage_key} examples: {e}")
            continue

    return examples if examples else None


def run_end2end_pipeline(
    dataset_cfg: Dict[str, Any],
    schema_cfg: Dict[str, Any],
    extraction_cfg: Dict[str, Any],
    dedup_cfg: Dict[str, Any],
    output_cfg: Dict[str, Any],
    meta_graph_cfg: Optional[Dict[str, Any]] = None,
    meta_graph_evaluation_cfg: Optional[Dict[str, Any]] = None,
    community_clustering_cfg: Optional[Dict[str, Any]] = None,
    community_evaluation_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    从原始语料构建知识图谱（支持单阶段 / 多阶段抽取）+ 实体消歧后的知识图谱。

    默认行为：
    - **开启多阶段提取**（extraction.use_staged_extraction=True）
    - **开启实体消歧**（基于 abbreviation + CID）
    - 当语料中包含多篇文档时，对每篇单独构图+消歧，并使用 `graph_merger` 自动合并。

    Args:
        dataset_cfg: `dataset` 配置段，对应 YAML 中的 `dataset`。
        schema_cfg: `schema` 配置段，对应 YAML 中的 `schema`。
        extraction_cfg: `extraction` 配置段，对应 YAML 中的 `extraction`。
        dedup_cfg: `deduplication` 配置段，对应 YAML 中的 `deduplication`。
        output_cfg: `output` 配置段，对应 YAML 中的 `output`。
        meta_graph_cfg: 上层元图谱配置（可选），`enable=true` 时构建 meta graph。
        meta_graph_evaluation_cfg: 元图谱质量评估配置（可选），`enable=true` 且 meta_graph 已构建时对 meta graph 做 source_alignment 打分。
        community_clustering_cfg: 社区聚类配置（步骤本身必选），`method=leiden|tree_comm` 控制使用 Leiden 或 TreeComm。
        community_evaluation_cfg: 社区质量评估配置（可选），`enable=true` 时对社区报告做 LLM 五维打分。

    Returns:
        最终消歧后的图谱文件绝对路径。
    """
    # 读取基础配置
    corpus_path = dataset_cfg.get("corpus_path")
    dataset_name = dataset_cfg.get("name")
    is_chunked = bool(dataset_cfg.get("is_chunked", False))
    if not corpus_path or not dataset_name:
        raise ValueError("dataset_cfg 中必须包含 `name` 和 `corpus_path`。")

    use_staged_extraction = extraction_cfg.get("use_staged_extraction", True)
    use_stage4_validation = bool(extraction_cfg.get("use_stage4_validation", False))
    save_stage_outputs = bool(extraction_cfg.get("save_stage_outputs", False))
    stage4_cfg = extraction_cfg.get("stage4") or {}
    stage4_min_triple_score = float(stage4_cfg.get("min_triple_score", 0.5))
    stage4_min_node_score = float(stage4_cfg.get("min_node_score", 0.5))
    stage4_use_chunk_scoring = bool(stage4_cfg.get("use_chunk_scoring", True))
    stage4_use_node_accuracy_scoring = bool(stage4_cfg.get("use_node_accuracy_scoring", True))
    stage4_use_triple_support_scoring = bool(stage4_cfg.get("use_triple_support_scoring", True))

    # schema 配置：path 优先于 name
    schema_path = schema_cfg.get("path")
    schema_name = schema_cfg.get("name")

    enable_abbreviation = bool(dedup_cfg.get("enable_abbreviation", True))
    enable_cid = bool(dedup_cfg.get("enable_cid", True))
    intermediate_output = bool(dedup_cfg.get("intermediate_output", False))
    pubchem_db_path = dedup_cfg.get("pubchem_db_path", "pubchem_names_full.db")

    output_graph_name = output_cfg.get("graph_name", "multi_stage_deduplicated.json")

    logger.info("======== 开始端到端管线 ========")
    corpus_path_abs = _resolve_path(corpus_path)
    corpus = _load_corpus(corpus_path_abs)
    num_docs = len(corpus)

    if num_docs == 0:
        raise ValueError(f"语料文件为空: {corpus_path_abs}")

    # 解析 schema 路径（无论单阶段 / 多阶段都需要）
    if schema_path:
        schema_path_abs = _resolve_path(schema_path)
    elif schema_name:
        schema_path_abs = _resolve_path(os.path.join("schemas", f"{schema_name}.json"))
    else:
        raise ValueError("schema_cfg 中必须提供 `name` 或 `path` 之一。")

    # 单阶段提取时解析 prompt 路径：
    # - 优先使用 extraction.prompt_path
    # - 其次使用 extraction.prompt_name
    # - 如均未提供，回退为 prompts/default.txt
    prompt_path_abs: Optional[str] = None
    if not use_staged_extraction:
        prompt_path_cfg = extraction_cfg.get("prompt_path")
        prompt_name_cfg = extraction_cfg.get("prompt_name")
        if prompt_path_cfg:
            prompt_path_abs = _resolve_path(prompt_path_cfg)
        elif prompt_name_cfg:
            prompt_path_abs = _resolve_path(os.path.join("prompts", f"{prompt_name_cfg}.txt"))
        else:
            prompt_path_abs = _resolve_path(os.path.join("prompts", "default.txt"))

    project_root = os.getenv("PROJECT_DIR") or os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(project_root, "output", dataset_name)
    _ensure_dir(dataset_root)

    # 如果配置中指定了 output_dir，则优先使用；否则使用时间戳
    configured_output_dir = output_cfg.get("output_dir")
    if configured_output_dir:
        output_dir = _get_unique_output_dir(configured_output_dir)
    else:
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        base_output_dir = os.path.join(dataset_root, run_id)
        output_dir = _get_unique_output_dir(base_output_dir)
    _ensure_dir(output_dir)

    # 为每篇文档构建图谱并做实体消歧
    dedup_graph_paths: List[str] = []
    num_docs = len(corpus)

    for idx, doc in enumerate(corpus):
        doc_title = doc.get("title", f"doc_{idx}")
        doc_id = f"{idx:04d}"

        chunk_output_path = os.path.join(output_dir, f"chunks_doc_{doc_id}.txt")

        if is_chunked:
            # 已切好 chunk：跳过第一阶段，直接使用已有 chunk 文件
            if not os.path.exists(chunk_output_path):
                raise FileNotFoundError(
                    f"is_chunked=True 但 chunk 文件不存在: {chunk_output_path}"
                )
            logger.info(f"跳过 chunk 切分，使用已有文件: {chunk_output_path}")
        else:
            # 1. 为该文档写出一个临时 corpus 文件（单文档）
            single_corpus_path = os.path.join(output_dir, f"corpus_doc_{doc_id}.json")
            with open(single_corpus_path, "w", encoding="utf-8") as f:
                json.dump([doc], f, ensure_ascii=False, indent=2)

            # 2. 调用 get_chunks 生成 chunk 文件
            get_chunks(
                corpus_path=single_corpus_path,
                dataset_name=f"{dataset_name}_doc_{doc_id}",
                output_path=chunk_output_path,
            )

        # 3. 调用 build_lowlevel_graph 进行抽取（单阶段 / 多阶段）
        lowlevel_graph_path = os.path.join(output_dir, f"graph_doc_{doc_id}.json")
        examples = _load_examples_from_config(extraction_cfg) if use_staged_extraction else None
        graph_config = {
            "schema_path": schema_path_abs,
            "schema_content": None,
            "prompt_path": None if use_staged_extraction else prompt_path_abs,
            "prompt_content": None,
            "use_staged_extraction": use_staged_extraction,
            "enable_stage4_validation": use_stage4_validation,
            "prompt_paths": extraction_cfg.get("prompt_paths") if use_staged_extraction else None,
            "examples": examples,
            "pubchem_db_path": pubchem_db_path,
            "save_stage_outputs": save_stage_outputs,
            "stage4": {
                "min_triple_score": stage4_min_triple_score,
                "min_node_score": stage4_min_node_score,
                "use_chunk_scoring": stage4_use_chunk_scoring,
                "use_node_accuracy_scoring": stage4_use_node_accuracy_scoring,
                "use_triple_support_scoring": stage4_use_triple_support_scoring,
            },
        }
        build_lowlevel_graph(
            chunk_path=chunk_output_path,
            output_graph_path=lowlevel_graph_path,
            config=graph_config,
        )

        # 4. 对该文档图谱做实体消歧
        dedup_output_path = os.path.join(output_dir, f"graph_doc_{doc_id}_deduplicated.json")
        dedup_config = {
            "output_path": dedup_output_path,
            "intermediate_output": intermediate_output,
            "enable_abbreviation": enable_abbreviation,
            "enable_cid": enable_cid,
            "pubchem_db_path": pubchem_db_path,
        }
        final_dedup_path = deduplicate_entities(
            graph_path=lowlevel_graph_path,
            config=dedup_config,
        )

        dedup_graph_paths.append(final_dedup_path)
        logger.info(f"[Doc {idx+1}/{num_docs}] 标题: {doc_title[:50]}... -> 消歧后图谱: {final_dedup_path}")

    # 5. 如果只有一篇文档，直接将该图谱作为最终输出
    final_output_path = os.path.join(output_dir, output_graph_name)
    first_chunk_path = os.path.join(output_dir, "chunks_doc_0000.txt")

    if len(dedup_graph_paths) == 1:
        # 单篇语料，无需合并，直接复制/重命名
        src = dedup_graph_paths[0]
        if os.path.abspath(src) != os.path.abspath(final_output_path):
            # 读写一次，避免跨文件系统 rename 问题
            with open(src, "r", encoding="utf-8") as f_src, open(
                final_output_path, "w", encoding="utf-8"
            ) as f_dst:
                f_dst.write(f_src.read())
        logger.info(f"单篇语料，无需合并。最终图谱: {final_output_path}")
        final_graph_path = final_output_path
    else:
        # 6. 多篇语料：使用 graph_merger 逐步合并
        logger.info(f"检测到 {len(dedup_graph_paths)} 篇语料，将使用 graph_merger 进行图谱合并。")
        current_path = dedup_graph_paths[0]
        for i in range(1, len(dedup_graph_paths)):
            next_path = dedup_graph_paths[i]
            is_last = i == len(dedup_graph_paths) - 1
            merged_output = final_output_path if is_last else os.path.join(
                output_dir, f"merged_step_{i:02d}.json"
            )
            current_path = merge_graphs(current_path, next_path, merged_output)
            logger.debug(f"合并第 {i} 步: {next_path} -> {current_path}")
        logger.info(f"多篇语料合并完成。最终图谱: {current_path}")
        final_graph_path = current_path

    # 7. 可选：上层元图谱构建
    meta_graph_cfg = meta_graph_cfg or {}
    meta_output = None
    meta_eval_output = None
    if meta_graph_cfg.get("enable", False):
        chunks_path = meta_graph_cfg.get("chunks_path")
        if not chunks_path:
            chunks_path = first_chunk_path
            if len(dedup_graph_paths) > 1:
                logger.warning(
                    "多篇语料下未指定 meta_graph.chunks_path，使用首篇 chunks: %s，"
                    "meta graph 对来自其他文档的三元组可能缺少 chunk 上下文。",
                    chunks_path,
                )
        chunks_path_abs = _resolve_path(chunks_path) if not os.path.isabs(chunks_path) else chunks_path
        base_name = os.path.splitext(os.path.basename(final_graph_path))[0]
        meta_output = os.path.join(output_dir, f"{base_name}_meta.json")
        prompt_path = meta_graph_cfg.get("prompt_path")
        max_chars = int(meta_graph_cfg.get("max_chars_per_chunk", 2000))
        max_total = int(meta_graph_cfg.get("max_total_chunk_chars", 50000))
        build_meta_graph(
            base_graph_path=final_graph_path,
            chunks_path=chunks_path_abs,
            output_path=meta_output,
            prompt_path=_resolve_path(prompt_path) if prompt_path else None,
            max_chars_per_chunk=max_chars,
            max_total_chunk_chars=max_total,
        )
        logger.info("上层元图谱已构建: %s", meta_output)

        # 6a. 可选：元图谱质量评估（需 meta_graph 已构建）
        meta_eval_cfg = meta_graph_evaluation_cfg or {}
        if meta_eval_cfg.get("enable", False):
            meta_eval_chunks = meta_eval_cfg.get("chunks_path") or first_chunk_path
            meta_eval_chunks_abs = _resolve_path(meta_eval_chunks) if not os.path.isabs(meta_eval_chunks) else meta_eval_chunks
            meta_eval_output = meta_eval_cfg.get("output_path")
            if not meta_eval_output:
                meta_eval_output = os.path.join(output_dir, f"{base_name}_meta_evaluation.json")
            else:
                meta_eval_output = _resolve_path(meta_eval_output) if not os.path.isabs(meta_eval_output) else meta_eval_output
            meta_eval_prompt = meta_eval_cfg.get("prompt_path")
            meta_eval_max_chars = int(meta_eval_cfg.get("max_chars_per_chunk", 2000))
            meta_eval_max_total = int(meta_eval_cfg.get("max_total_chunk_chars", 50000))

            logger.info("开始元图谱质量评估: meta_graph=%s, chunks=%s, output=%s", meta_output, meta_eval_chunks_abs, meta_eval_output)
            run_meta_graph_evaluation(
                meta_graph_path=meta_output,
                base_graph_path=final_graph_path,
                chunks_path=meta_eval_chunks_abs,
                output_path=meta_eval_output,
                prompt_path=_resolve_path(meta_eval_prompt) if meta_eval_prompt else None,
                max_chars_per_chunk=meta_eval_max_chars,
                max_total_chunk_chars=meta_eval_max_total,
            )
            logger.info("元图谱质量评估已完成，输出见 %s", meta_eval_output)

    # 8. 可选：社区聚类（Leiden 或 TreeComm，由 method 控制）
    clustering_cfg = community_clustering_cfg or {}
    
    # 检查是否禁用社区聚类
    if clustering_cfg.get("enable") is False:
        logger.info("社区聚类已禁用，跳过此步骤")
        cluster_result = {"stats": {"communities": 0, "nodes": 0, "edges": 0}}
        community_report_path = None
        community_method = "disabled"
    else:
        community_method = (clustering_cfg.get("method") or "leiden").lower().strip()
        base_name = os.path.splitext(os.path.basename(final_graph_path))[0]

        if community_method == "tree_comm":
            embedding_model = clustering_cfg.get("embedding_model", "all-MiniLM-L6-v2")
            struct_weight = float(clustering_cfg.get("struct_weight", 0.3))
            logger.info(
                "开始 TreeComm 社区聚类: input=%s, output_dir=%s, embedding_model=%s, struct_weight=%s",
                final_graph_path,
                output_dir,
                embedding_model,
                struct_weight,
            )
            cluster_result = run_tree_comm_clustering(
                input_path=final_graph_path,
                output_dir=output_dir,
                embedding_model=embedding_model,
                struct_weight=struct_weight,
            )
            community_report_path = os.path.join(output_dir, f"{base_name}_tree_comm_community_report.json")
        else:
            # Leiden（默认）
            max_cluster_size = int(clustering_cfg.get("max_cluster_size", 1000))
            use_lcc = bool(clustering_cfg.get("use_lcc", False))
            seed = clustering_cfg.get("seed")
            if seed is not None:
                seed = int(seed)
            else:
                seed = 42
            llm_prompt_path = clustering_cfg.get("llm_prompt_path")

            logger.info(
                "开始 Leiden 社区聚类: input=%s, output_dir=%s, max_cluster_size=%d, use_lcc=%s, seed=%s",
                final_graph_path,
                output_dir,
                max_cluster_size,
                use_lcc,
                seed,
            )
            cluster_result = run_community_clustering(
                input_path=final_graph_path,
                output_dir=output_dir,
                max_cluster_size=max_cluster_size,
                use_lcc=use_lcc,
                seed=seed,
                llm_prompt_path=_resolve_path(llm_prompt_path) if llm_prompt_path else None,
            )
            community_report_path = os.path.join(output_dir, f"{base_name}_community_report.json")

    stats = cluster_result.get("stats", {})
    logger.info(
        "%s 社区聚类已完成: 社区数=%d, 节点数=%d, 边数=%d；输出见 %s",
        "TreeComm" if community_method == "tree_comm" else "Leiden",
        stats.get("communities", 0),
        stats.get("nodes", 0),
        stats.get("edges", 0),
        output_dir,
    )

    # 9. 可选：社区质量评估（LLM 五维打分）
    eval_cfg = community_evaluation_cfg or {}
    community_eval_output = None
    if eval_cfg.get("enable", False):
        report_path = community_report_path
        chunks_path = eval_cfg.get("chunks_path") or first_chunk_path
        chunks_path_abs = _resolve_path(chunks_path) if not os.path.isabs(chunks_path) else chunks_path
        output_path = eval_cfg.get("output_path")
        if not output_path:
            report_base = os.path.splitext(os.path.basename(report_path))[0]
            output_path = os.path.join(output_dir, f"{report_base}_evaluation.json")
        else:
            output_path = _resolve_path(output_path) if not os.path.isabs(output_path) else output_path
        community_eval_output = output_path
        prompt_path = eval_cfg.get("prompt_path")
        max_chars = int(eval_cfg.get("max_chars_per_chunk", 2000))
        max_total = int(eval_cfg.get("max_total_chunk_chars", 50000))

        logger.info(
            "开始社区质量评估: report=%s, chunks=%s, output=%s",
            report_path,
            chunks_path_abs,
            output_path,
        )
        run_community_evaluation(
            report_path=report_path,
            chunks_path=chunks_path_abs,
            output_path=output_path,
            prompt_path=_resolve_path(prompt_path) if prompt_path else None,
            max_chars_per_chunk=max_chars,
            max_total_chunk_chars=max_total,
        )
        logger.info("社区质量评估已完成，输出见 %s", output_path)

    # 10. 可选：导入 Neo4j（一级图谱 + 二级图谱 + 评分）
    neo4j_cfg: Dict[str, Any] = output_cfg.get("neo4j", {})
    if neo4j_cfg and neo4j_cfg.get("enable", True):
        uri = neo4j_cfg.get("uri", "bolt://localhost:7687")
        user = neo4j_cfg.get("user", "neo4j")
        password = neo4j_cfg.get("password", "password")
        batch_size = int(neo4j_cfg.get("batch_size", 500))
        clear_first = bool(neo4j_cfg.get("clear_first", True))
        current_graph_rel = neo4j_cfg.get("current_graph", "current_graph.json")
        current_graph_path = Path(_resolve_path(current_graph_rel))

        final_graph_file = Path(final_graph_path)

        logger.info(f"开始将图谱导入 Neo4j: {uri}, 用户: {user}")
        driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        try:
            # 10.1 一级图谱导入
            update_current_graph_and_import(
                new_graph_path=final_graph_file,
                driver=driver,
                current_graph_path=current_graph_path,
                clear_first=clear_first,
                batch_size=batch_size,
            )

            # 10.2 二级图谱导入（如果有）
            if meta_graph_cfg and meta_graph_cfg.get("enable", False):
                meta_graph_file = Path(meta_output)
                meta_eval_file = Path(meta_eval_output) if meta_eval_output and Path(meta_eval_output).is_file() else None
                if meta_graph_file.is_file():
                    import_meta_graph(
                        meta_graph_path=meta_graph_file,
                        driver=driver,
                        meta_eval_path=meta_eval_file,
                        batch_size=batch_size,
                    )

            # 10.3 社区评分导入（如果有）
            if community_eval_output and Path(community_eval_output).is_file():
                import_community_evaluation(
                    community_eval_path=Path(community_eval_output),
                    driver=driver,
                )
        finally:
            driver.close()

    logger.info("======== 端到端管线完成 ========")
    return final_graph_path


def main():
    """命令行入口：仅接收一个 YAML 配置文件路径。"""
    parser = argparse.ArgumentParser(
        description="从 YAML 配置构建（单阶段 / 多阶段）抽取 + 实体消歧 + 图谱合并的端到端管线"
    )
    parser.add_argument(
        "config",
        help="YAML 配置文件路径，例如 `configs/paper_mini_pipeline.yml`",
    )

    args = parser.parse_args()

    cfg = _load_config(args.config)

    # 首先设置日志级别（在其他操作之前）
    logging_cfg: Dict[str, Any] = cfg.get("logging", {})
    _setup_logging_from_config(logging_cfg)

    dataset_cfg: Dict[str, Any] = cfg.get("dataset", {})
    schema_cfg: Dict[str, Any] = cfg.get("schema", {})
    extraction_cfg: Dict[str, Any] = cfg.get("extraction", {})
    dedup_cfg: Dict[str, Any] = cfg.get("deduplication", {})
    output_cfg: Dict[str, Any] = cfg.get("output", {})
    meta_graph_cfg: Dict[str, Any] = cfg.get("meta_graph", {})
    meta_graph_evaluation_cfg: Dict[str, Any] = cfg.get("meta_graph_evaluation", {})
    community_clustering_cfg: Dict[str, Any] = cfg.get("community_clustering", {})
    community_evaluation_cfg: Dict[str, Any] = cfg.get("community_evaluation", {})

    run_end2end_pipeline(
        dataset_cfg=dataset_cfg,
        schema_cfg=schema_cfg,
        extraction_cfg=extraction_cfg,
        dedup_cfg=dedup_cfg,
        output_cfg=output_cfg,
        meta_graph_cfg=meta_graph_cfg,
        meta_graph_evaluation_cfg=meta_graph_evaluation_cfg,
        community_clustering_cfg=community_clustering_cfg,
        community_evaluation_cfg=community_evaluation_cfg,
    )


if __name__ == "__main__":
    main()
