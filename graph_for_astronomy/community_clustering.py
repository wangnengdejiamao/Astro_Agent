"""
底层知识图谱社区聚类：使用 Leiden 算法对实体节点进行社区检测。

参考 graphrag 的 cluster_graph 实现，基于 graspologic.partition.hierarchical_leiden。
输出：每个节点的 community_id，以及 community report。
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import json_repair

try:
    from graspologic.partition import hierarchical_leiden
    from graspologic.utils import largest_connected_component
    HAS_GRASPOLOGIC = True
except ImportError:
    HAS_GRASPOLOGIC = False

from utils import call_llm_api
from utils.logger import logger


def _get_entity_name(node: Dict) -> Optional[str]:
    """从节点字典提取实体名称"""
    if not node or "properties" not in node:
        return None
    name = node.get("properties", {}).get("name")
    if isinstance(name, list) and name:
        return str(name[0]) if name[0] else None
    return str(name) if name else None


def _is_entity_to_entity(edge: Dict) -> bool:
    """仅保留 entity→entity 的边"""
    start = edge.get("start_node", {})
    end = edge.get("end_node", {})
    return start.get("label") == "entity" and end.get("label") == "entity"


def load_graph_from_json(path: str) -> Tuple[nx.Graph, List[Dict]]:
    """
    从 JSON 加载底层知识图谱，构建实体级无向图。

    支持格式：
    - 列表：直接为三元组列表
    - 字典：包含 "edges" 或 "triples" 键

    每个三元组需包含：start_node, end_node, relation
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        edges_raw = data
    else:
        edges_raw = data.get("edges", data.get("triples", []))

    G = nx.Graph()
    triples_for_report: List[Dict] = []

    for edge in edges_raw:
        if not _is_entity_to_entity(edge):
            continue

        subj = _get_entity_name(edge.get("start_node"))
        obj = _get_entity_name(edge.get("end_node"))
        rel = edge.get("relation", "")
        if not subj or not obj or not rel:
            continue

        # 汇总边的权重与对应的 chunk_ids（用于后续社区与原文对齐评估）
        existing_attrs = G.edges.get((subj, obj), {}) or {}
        prev_weight = existing_attrs.get("weight", 0)
        prev_chunk_ids = existing_attrs.get("chunk_ids") or []
        if isinstance(prev_chunk_ids, str):
            prev_chunk_ids = [prev_chunk_ids]

        chunk_ids = edge.get("chunk_id", [])
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        elif not isinstance(chunk_ids, list):
            chunk_ids = [chunk_ids] if chunk_ids else []
        chunk_ids = [str(c) for c in chunk_ids if c is not None]

        merged_chunk_ids = list({*(str(c) for c in prev_chunk_ids if c), *chunk_ids})

        G.add_edge(subj, obj, weight=prev_weight + 1, chunk_ids=merged_chunk_ids)
        triples_for_report.append(
            {
                "subject": subj,
                "relation": rel,
                "object": obj,
                "chunk_ids": chunk_ids,
            }
        )

    logger.info(f"成功加载图谱 {path}，包含 {G.number_of_nodes()} 个节点、{G.number_of_edges()} 条边")
    return G, triples_for_report


def cluster_with_leiden(
    graph: nx.Graph,
    max_cluster_size: int = 1000,
    use_lcc: bool = False,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, int], Dict[int, int]]:
    """
    使用 Leiden 层次聚类算法对图进行社区检测。

    参考 graphrag/index/operations/cluster_graph.py

    Returns:
        node_to_community: 节点名 -> 社区ID
        hierarchy: 社区ID -> 父社区ID (-1 表示根)
    """
    if not HAS_GRASPOLOGIC:
        raise ImportError("需要安装 graspologic: pip install graspologic")

    if graph.number_of_nodes() == 0:
        logger.warning("图无节点")
        return {}, {}

    if use_lcc:
        graph = largest_connected_component(graph)
        graph = nx.Graph(graph)  # 确保是 nx.Graph

    community_mapping = hierarchical_leiden(
        graph,
        max_cluster_size=max_cluster_size,
        random_seed=seed,
    )

    # 使用 graspologic 提供的 final_level_hierarchical_clustering：
    # 对每个节点取 is_final_cluster=True 时的 cluster，保证所有节点都被包含。
    node_to_community = community_mapping.final_level_hierarchical_clustering()

    hierarchy: Dict[int, int] = {}
    for partition in community_mapping:
        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )

    return node_to_community, hierarchy


def build_community_report(
    node_to_community: Dict[str, int],
    triples: List[Dict],
    graph: nx.Graph,
) -> List[Dict[str, Any]]:
    """
    构建社区报告：每个社区的成员、规模、内部关系统计等。
    """
    comm_to_nodes: Dict[int, List[str]] = defaultdict(list)
    for node, comm_id in node_to_community.items():
        comm_to_nodes[comm_id].append(node)

    comm_to_relations: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    comm_to_chunk_ids: Dict[int, set] = defaultdict(set)
    for t in triples:
        subj, rel, obj = t["subject"], t["relation"], t["object"]
        comm_id_s = node_to_community.get(subj)
        comm_id_o = node_to_community.get(obj)
        if comm_id_s is not None and comm_id_o is not None and comm_id_s == comm_id_o:
            comm_to_relations[comm_id_s][rel] += 1

            # 汇总该社区内部三元组关联的 chunk_ids
            chunk_ids = t.get("chunk_ids") or []
            if isinstance(chunk_ids, str):
                chunk_ids = [chunk_ids]
            for cid in chunk_ids:
                if cid is not None:
                    comm_to_chunk_ids[comm_id_s].add(str(cid))

    reports = []
    for comm_id in sorted(comm_to_nodes.keys()):
        members = comm_to_nodes[comm_id]
        rel_counts = dict(comm_to_relations[comm_id])
        top_relations = sorted(rel_counts.items(), key=lambda x: -x[1])[:10]
        chunk_ids = sorted(comm_to_chunk_ids.get(comm_id, set()))

        reports.append({
            "community_id": comm_id,
            "name": f"Community_{comm_id}",
            "summary": f"社区 {comm_id}: {len(members)} 个实体, 主要关系: {', '.join(r for r, _ in top_relations[:5])}",
            "size": len(members),
            "members": sorted(members),
            "relation_counts": rel_counts,
            "top_relations": top_relations,
            "super_node_id": None,
            "chunk_ids": chunk_ids,
        })

    return reports


def _load_llm_prompt_template(prompt_path: Optional[str] = None) -> str:
    """加载用于 Leiden 社区报告的 LLM prompt 模板"""
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    default_path = os.path.join(
        os.path.dirname(__file__),
        "prompts",
        "lowlevel_leiden_community_report.txt",
    )
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Prompt not found: {default_path}")
    with open(default_path, "r", encoding="utf-8") as f:
        return f.read()


def enrich_communities_with_llm(
    reports: List[Dict[str, Any]],
    llm_prompt_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    使用 LLM 为每个社区生成 name 和 summary。
    - 输入：初步的 report（包含 community_id, size, members, top_relations）
    - 输出：在原 report 上添加/覆盖 name, summary 字段
    """
    if not reports:
        return reports

    prompt_template = _load_llm_prompt_template(llm_prompt_path)

    # 为 LLM 准备更精简的社区信息，避免超长
    communities_for_llm = []
    for r in reports:
        comm_id = r["community_id"]
        size = r["size"]
        members = r.get("members", [])
        top_rel = r.get("top_relations", [])

        # 只取前若干成员，避免 prompt 过长
        max_members = 20
        short_members = members[:max_members]

        communities_for_llm.append(
            {
                "id": int(comm_id),
                "size": int(size),
                "members": short_members,
                "top_relations": [[rel, int(cnt)] for rel, cnt in top_rel],
            }
        )

    communities_json = json.dumps(communities_for_llm, ensure_ascii=False)
    prompt = prompt_template.replace("{communities_json}", communities_json)

    llm_client = call_llm_api.LLMCompletionCall()
    try:
        response_text = llm_client.call_api(prompt)
        parsed = json_repair.loads(response_text)
        if not isinstance(parsed, list):
            raise ValueError("LLM 返回格式错误：顶层不是 list")
    except Exception as e:
        logger.error("调用 LLM 生成 Leiden 社区报告失败，将保留规则 summary: %s", e)
        return reports

    # 构建 id -> {name, summary}
    id_to_llm = {}
    for item in parsed:
        try:
            comm_id = int(item.get("id"))
        except Exception:
            continue
        name = item.get("name")
        summary = item.get("summary")
        if name or summary:
            id_to_llm[comm_id] = {"name": name, "summary": summary}

    # 回填到 reports
    for r in reports:
        comm_id = int(r["community_id"])
        llm_info = id_to_llm.get(comm_id)
        if not llm_info:
            # 没有匹配的 LLM 结果时，至少给一个默认 name
            if "name" not in r:
                r["name"] = f"Community_{comm_id}"
            continue

        name = llm_info.get("name") or f"Community_{comm_id}"
        summary = llm_info.get("summary") or r.get("summary")

        r["name"] = name
        r["summary"] = summary

    return reports


def run_community_clustering(
    input_path: str,
    output_dir: Optional[str] = None,
    max_cluster_size: int = 1000,
    use_lcc: bool = False,
    seed: Optional[int] = 42,
    llm_prompt_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对底层知识图谱执行社区聚类，输出 community id 和 community report。

    Args:
        input_path: 底层图谱 JSON 路径
        output_dir: 输出目录，默认与输入同目录
        max_cluster_size: Leiden 最大簇大小
        use_lcc: 是否只使用最大连通分量
        seed: 随机种子

    Returns:
        包含 node_communities, community_report, stats 的字典
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    out_communities_json = os.path.join(output_dir, f"{base_name}_communities.json")
    out_report_json = os.path.join(output_dir, f"{base_name}_community_report.json")
    out_report_txt = os.path.join(output_dir, f"{base_name}_community_report.txt")

    logger.info("======== 开始 Leiden 社区聚类 ========")
    logger.info(f"输入图谱: {input_path}, 输出目录: {output_dir}, max_cluster_size={max_cluster_size}, use_lcc={use_lcc}, seed={seed}")

    G, triples = load_graph_from_json(input_path)
    if G.number_of_nodes() == 0:
        logger.warning("图为空，跳过聚类")
        return {"node_communities": {}, "community_report": [], "stats": {"nodes": 0, "edges": 0, "communities": 0}}

    logger.info("--- 步骤1: Leiden 聚类 ---")
    node_to_community, hierarchy = cluster_with_leiden(
        G, max_cluster_size=max_cluster_size, use_lcc=use_lcc, seed=seed
    )
    num_communities = len(set(node_to_community.values()))
    logger.info(f"Leiden 聚类完成: 检测到 {num_communities} 个社区")

    logger.info("--- 步骤2: 构建社区报告 ---")
    report = build_community_report(node_to_community, triples, G)

    logger.info("--- 步骤3: LLM 丰富 name 与 summary ---")
    try:
        report = enrich_communities_with_llm(report, llm_prompt_path=llm_prompt_path)
        logger.info("LLM 丰富社区报告完成")
    except Exception as e:
        logger.error("enrich_communities_with_llm 失败，将继续使用原始报告: %s", e)

    logger.info("--- 步骤4: 写入输出文件 ---")
    # 构建 node_communities、communities、stats、hierarchy（与 tree_comm 统一格式）
    node_communities = [
        {"node": n, "community_id": comm_id}
        for n, comm_id in sorted(node_to_community.items())
    ]

    comm_to_nodes: Dict[int, List[str]] = {}
    for n, cid in node_to_community.items():
        comm_to_nodes.setdefault(cid, []).append(n)
    communities = {str(cid): sorted(members) for cid, members in comm_to_nodes.items()}
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "communities": len(report),
    }
    hierarchy_dict = {str(k): v for k, v in hierarchy.items()}

    result = {
        "node_communities": node_communities,
        "communities": communities,
        "stats": stats,
        "hierarchy": hierarchy_dict,
        "community_report": report,
    }

    # 写 communities JSON（与 tree_comm 统一格式）
    communities_output = {
        "node_communities": node_communities,
        "communities": communities,
        "stats": stats,
        "hierarchy": hierarchy_dict,
    }
    with open(out_communities_json, "w", encoding="utf-8") as f:
        json.dump(communities_output, f, ensure_ascii=False, indent=2)
    logger.info(f"社区划分结果已保存到: {out_communities_json}")

    with open(out_report_json, "w", encoding="utf-8") as f:
        json.dump({"community_report": report, "stats": result["stats"]}, f, ensure_ascii=False, indent=2)
    logger.info(f"社区报告已保存到: {out_report_json}")

    # 写可读文本报告
    lines = [
        "# 社区聚类报告 (Leiden)",
        "",
        f"## 统计",
        f"- 节点数: {result['stats']['nodes']}",
        f"- 边数: {result['stats']['edges']}",
        f"- 社区数: {result['stats']['communities']}",
        "",
        "## 各社区详情",
        "",
    ]
    for r in report:
        title = r.get("name") or f"社区 {r['community_id']}"
        lines.append(f"### 社区 {r['community_id']} - {title} (规模: {r['size']})")
        lines.append("")
        if r.get("summary"):
            lines.append(f"摘要: {r['summary']}")
            lines.append("")
        lines.append("**成员:**")
        for m in r["members"]:
            lines.append(f"  - {m}")
        lines.append("")
        lines.append("**主要关系:**")
        for rel, cnt in r["top_relations"]:
            lines.append(f"  - {rel}: {cnt}")
        lines.append("")

    with open(out_report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"社区文本报告已保存到: {out_report_txt}")

    logger.info(
        "======== Leiden 社区聚类流程完成 ======== 社区数=%d, 节点数=%d, 边数=%d",
        stats["communities"],
        stats["nodes"],
        stats["edges"],
    )
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="底层知识图谱 Leiden 社区聚类")
    parser.add_argument("input", nargs="?", default="output/paper_mini/20260203103057/260129.json", help="图谱 JSON 路径")
    parser.add_argument("-o", "--output-dir", default=None, help="输出目录")
    parser.add_argument("--max-cluster-size", type=int, default=1000, help="Leiden 最大簇大小")
    parser.add_argument("--use-lcc", action="store_true", help="仅使用最大连通分量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--llm-prompt",
        default=None,
        help="自定义 LLM prompt 路径（默认为 prompts/lowlevel_leiden_community_report.txt）",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        for base in [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]:
            p = os.path.join(base, input_path)
            if os.path.exists(p):
                input_path = p
                break

    if not os.path.exists(input_path):
        logger.error("输入文件不存在: %s", input_path)
        logger.info("用法: python community_clustering.py <input_json_path> [-o output_dir]")
        exit(1)

    result = run_community_clustering(
        input_path=input_path,
        output_dir=args.output_dir,
        max_cluster_size=args.max_cluster_size,
        use_lcc=args.use_lcc,
        seed=args.seed,
        llm_prompt_path=args.llm_prompt,
    )
    print(json.dumps(result["stats"], indent=2, ensure_ascii=False))
