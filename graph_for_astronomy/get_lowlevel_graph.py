
#!/usr/bin/env python3
from __future__ import annotations
"""
从 chunks 文件构建知识图谱并保存（低层版本）
直接依赖 chunk_path、schema_path、prompt_path

支持两种构建方式：
- 单阶段：SingleGraphBuilder，一次 LLM 调用提取实体/关系/属性
- 多阶段：StagedGraphBuilder，Stage1 实体 → Stage2 关系 → Stage3 属性（可选 Stage4 验证）

支持通过 config 字典传入参数，避免超长入参。
"""

import os
import sys
from typing import Any, Dict, Union, Optional

from graph_builder import GraphBuilder
from single_graph_builder import SingleGraphBuilder
from staged_graph_builder import StagedGraphBuilder
from utils.logger import logger


def _default_build_config() -> Dict[str, Any]:
    """返回 build_lowlevel_graph 的默认 config 结构（用于合并与文档）。"""
    return {
        "chunk_path": None,
        "output_graph_path": None,
        "schema_path": None,
        "schema_content": None,
        "prompt_path": None,
        "prompt_content": None,
        "use_staged_extraction": False,
        "enable_stage4_validation": False,
        "prompt_paths": None,
        "examples": None,
        "pubchem_db_path": None,
        "save_stage_outputs": False,
        "stage4": {
            "min_triple_score": 0.5,
            "min_node_score": 0.5,
            "use_chunk_scoring": True,
            "use_node_accuracy_scoring": True,
            "use_triple_support_scoring": True,
        },
    }


def _merge_config(config: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """将 config 与 kwargs 合并，kwargs 覆盖 config。若 config 为 None 则仅用 kwargs。"""
    base = dict(config) if config else {}
    stage4 = dict(base.get("stage4") or _default_build_config()["stage4"])
    for k, v in kwargs.items():
        if k == "stage4_min_triple_score":
            stage4["min_triple_score"] = v
        elif k == "stage4_min_node_score":
            stage4["min_node_score"] = v
        elif k == "stage4_use_chunk_scoring":
            stage4["use_chunk_scoring"] = v
        elif k == "stage4_use_node_accuracy_scoring":
            stage4["use_node_accuracy_scoring"] = v
        elif k == "stage4_use_triple_support_scoring":
            stage4["use_triple_support_scoring"] = v
        elif k not in ("stage4_min_triple_score", "stage4_min_node_score", "stage4_use_chunk_scoring",
                       "stage4_use_node_accuracy_scoring", "stage4_use_triple_support_scoring"):
            base[k] = v
    base["stage4"] = stage4
    return base


def build_lowlevel_graph(
    chunk_path: str,
    output_graph_path: str = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    构建知识图谱，支持路径或内容字符串，支持单阶段和多阶段提取。

    推荐使用 config 字典传入参数；也可继续使用关键字参数（与 config 同时传入时，kwargs 覆盖 config）。

    Config 结构示例：
        {
            "schema_path": str | None,
            "schema_content": dict | str | None,
            "prompt_path": str | None,
            "prompt_content": str | None,
            "use_staged_extraction": bool,
            "enable_stage4_validation": bool,
            "prompt_paths": dict | None,
            "pubchem_db_path": str | None,
            "save_stage_outputs": bool,
            "stage4": {
                "min_triple_score": float,
                "min_node_score": float,
                "use_chunk_scoring": bool,
                "use_node_accuracy_scoring": bool,
                "use_triple_support_scoring": bool,
            },
        }
    
    Args:
        chunk_path: chunks 文件路径（必需）
        output_graph_path: 输出图 JSON 保存路径（必需，可放在 config 中）
        config: 可选配置字典，与 kwargs 合并后使用
        **kwargs: 其余参数，与 config 中同名键合并时以 kwargs 为准
    """
    merged = _merge_config(config, {"chunk_path": chunk_path, "output_graph_path": output_graph_path, **kwargs})
    chunk_path = merged.get("chunk_path") or chunk_path
    output_graph_path = merged.get("output_graph_path") or output_graph_path
    schema_path = merged.get("schema_path")
    schema_content = merged.get("schema_content")
    prompt_path = merged.get("prompt_path")
    prompt_content = merged.get("prompt_content")
    use_staged_extraction = bool(merged.get("use_staged_extraction", False))
    enable_stage4_validation = bool(merged.get("enable_stage4_validation", False))
    prompt_paths = merged.get("prompt_paths")
    examples = merged.get("examples")
    pubchem_db_path = merged.get("pubchem_db_path")
    save_stage_outputs = bool(merged.get("save_stage_outputs", False))
    stage4 = merged.get("stage4") or {}

    if not output_graph_path:
        raise ValueError("output_graph_path 是必需参数")
    if not schema_content and not schema_path:
        raise ValueError("必须提供 schema_path 或 schema_content 之一")
    
    if use_staged_extraction:
        if prompt_path or prompt_content:
            logger.warning("多阶段提取时，prompt_path 和 prompt_content 将被忽略")
        builder: GraphBuilder = StagedGraphBuilder(
            config={
                "schema_path": schema_path,
                "schema_content": schema_content,
                "prompt_paths": prompt_paths or {},
                "examples": examples,
                "enable_stage4_validation": enable_stage4_validation,
                "save_stage_outputs": save_stage_outputs,
                "pubchem_db_path": pubchem_db_path,
                "stage4": stage4,
            }
        )
    else:
        if not prompt_content and not prompt_path:
            raise ValueError("单阶段提取必须提供 prompt_path 或 prompt_content 之一")
        builder = SingleGraphBuilder(
        schema_path=schema_path, 
        prompt_path=prompt_path,
        schema_content=schema_content,
        prompt_content=prompt_content,
            pubchem_db_path=pubchem_db_path,
        )

    stage_output_dir: Optional[str] = None
    if use_staged_extraction and save_stage_outputs and output_graph_path:
        stage_output_dir = os.path.join(
            os.path.dirname(output_graph_path), "staged"
        )

    return builder.build_knowledge_graph(
        chunk_path,
        output_graph_path,
        stage_output_dir=stage_output_dir,
    )


# 向后兼容：对外仍可 from get_lowlevel_graph import GraphBuilder（指向基类）
# 若需要具体实现，可 from get_lowlevel_graph import SingleGraphBuilder, StagedGraphBuilder
__all__ = [
    "build_lowlevel_graph",
    "_default_build_config",
    "GraphBuilder",
    "SingleGraphBuilder",
    "StagedGraphBuilder",
]


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("用法: python get_lowlevel_graph.py <chunk_path> <schema_path> <prompt_path> <output_graph_path>")
        sys.exit(1)

    chunk_path = sys.argv[1]
    schema_path = sys.argv[2]
    prompt_path = sys.argv[3]
    output_graph_path = sys.argv[4]

    schema_content = os.environ.get('SCHEMA_CONTENT', None)
    prompt_content = os.environ.get('PROMPT_CONTENT', None)
    
    build_lowlevel_graph(
        chunk_path=chunk_path,
        output_graph_path=output_graph_path,
        config={
            "schema_path": schema_path if not schema_content else None,
            "prompt_path": prompt_path if not prompt_content else None,
            "schema_content": schema_content,
            "prompt_content": prompt_content,
        },
    )
