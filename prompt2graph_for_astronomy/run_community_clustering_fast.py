#!/usr/bin/env python3
"""
快速社区聚类：执行 Leiden 聚类 + 基础报告 + 分批 LLM 丰富
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from community_clustering import (
    load_graph_from_json,
    cluster_with_leiden,
    build_community_report,
    enrich_communities_with_llm,
    _load_llm_prompt_template,
)
from utils import call_llm_api
import json_repair
from utils.logger import logger


def enrich_communities_with_llm_batched(
    reports,
    llm_prompt_path=None,
    batch_size=20,
):
    """
    分批调用 LLM 为每个社区生成 name 和 summary。
    避免一次性发送过多社区导致 prompt 过长 / 超时。
    """
    if not reports:
        return reports

    prompt_template = _load_llm_prompt_template(llm_prompt_path)
    llm_client = call_llm_api.LLMCompletionCall()
    call_delay = getattr(llm_client, 'call_delay', 0.0)

    total = len(reports)
    batches = (total + batch_size - 1) // batch_size
    id_to_llm = {}

    logger.info(f"开始分批 LLM 丰富社区报告: 共 {total} 个社区, {batches} 批, 每批 {batch_size} 个")

    for batch_idx in range(batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch_reports = reports[start:end]

        communities_for_llm = []
        for r in batch_reports:
            comm_id = r["community_id"]
            size = r["size"]
            members = r.get("members", [])
            top_rel = r.get("top_relations", [])
            max_members = 20
            short_members = members[:max_members]
            communities_for_llm.append({
                "id": int(comm_id),
                "size": int(size),
                "members": short_members,
                "top_relations": [[rel, int(cnt)] for rel, cnt in top_rel],
            })

        communities_json = json.dumps(communities_for_llm, ensure_ascii=False)
        prompt = prompt_template.replace("{communities_json}", communities_json)

        logger.info(f"  LLM 丰富第 {batch_idx + 1}/{batches} 批 ({start}-{end})")

        try:
            response_text = llm_client.call_api(prompt)
            parsed = json_repair.loads(response_text)
            if not isinstance(parsed, list):
                logger.warning(f"第 {batch_idx + 1} 批 LLM 返回格式不是 list，跳过")
                continue

            for item in parsed:
                try:
                    comm_id = int(item.get("id"))
                except Exception:
                    continue
                name = item.get("name")
                summary = item.get("summary")
                if name or summary:
                    id_to_llm[comm_id] = {"name": name, "summary": summary}

            logger.info(f"  第 {batch_idx + 1}/{batches} 批完成，已获取 {len(parsed)} 个结果")
        except Exception as e:
            logger.error(f"第 {batch_idx + 1}/{batches} 批 LLM 丰富失败: {e}")

        # 速率限制冷却
        if call_delay > 0 and batch_idx < batches - 1:
            time.sleep(call_delay)

    # 回填到 reports
    enriched_count = 0
    for r in reports:
        comm_id = int(r["community_id"])
        llm_info = id_to_llm.get(comm_id)
        if llm_info:
            name = llm_info.get("name") or r.get("name") or f"Community_{comm_id}"
            summary = llm_info.get("summary") or r.get("summary") or ""
            r["name"] = name
            r["summary"] = summary
            enriched_count += 1
        else:
            if "name" not in r:
                r["name"] = f"Community_{comm_id}"

    logger.info(f"LLM 丰富完成: {enriched_count}/{total} 个社区被丰富")
    return reports


def run_fast_clustering(
    input_path: str,
    output_dir: str,
    max_cluster_size=1000,
    use_lcc=False,
    seed=42,
    llm_prompt_path=None,
    enable_llm_enrichment=True,
    llm_batch_size=20,
):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    out_communities_json = os.path.join(output_dir, f"{base_name}_communities.json")
    out_report_json = os.path.join(output_dir, f"{base_name}_community_report.json")
    out_report_txt = os.path.join(output_dir, f"{base_name}_community_report.txt")

    logger.info("======== Leiden 社区聚类 ========")
    start_time = time.time()

    G, triples = load_graph_from_json(input_path)
    if G.number_of_nodes() == 0:
        logger.warning("图为空，跳过聚类")
        return {"stats": {"nodes": 0, "edges": 0, "communities": 0}}

    logger.info(f"加载图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 步骤1: Leiden 聚类
    t1 = time.time()
    node_to_community, hierarchy = cluster_with_leiden(
        G, max_cluster_size=max_cluster_size, use_lcc=use_lcc, seed=seed
    )
    num_communities = len(set(node_to_community.values()))
    logger.info(f"Leiden 聚类完成: {num_communities} 个社区 (耗时 {time.time()-t1:.2f}s)")

    # 步骤2: 构建社区报告
    t2 = time.time()
    report = build_community_report(node_to_community, triples, G)
    logger.info(f"社区报告构建完成: {len(report)} 个社区 (耗时 {time.time()-t2:.2f}s)")

    # 步骤3: LLM 丰富 name 与 summary（分批）
    if enable_llm_enrichment:
        t3 = time.time()
        try:
            report = enrich_communities_with_llm_batched(
                report,
                llm_prompt_path=llm_prompt_path,
                batch_size=llm_batch_size,
            )
            logger.info(f"LLM 丰富社区报告完成 (耗时 {time.time()-t3:.2f}s)")
        except Exception as e:
            logger.error(f"enrich_communities_with_llm_batched 失败，将使用原始报告: {e}")
    else:
        logger.info("跳过 LLM 丰富 (--skip-llm)")

    # 保存结果
    node_communities = [
        {"node": n, "community_id": comm_id}
        for n, comm_id in sorted(node_to_community.items())
    ]
    comm_to_nodes = {}
    for n, cid in node_to_community.items():
        comm_to_nodes.setdefault(cid, []).append(n)
    communities = {str(cid): sorted(members) for cid, members in comm_to_nodes.items()}
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "communities": len(report),
    }
    hierarchy_dict = {str(k): v for k, v in hierarchy.items()}

    # 写 communities JSON
    communities_output = {
        "node_communities": node_communities,
        "communities": communities,
        "stats": stats,
        "hierarchy": hierarchy_dict,
    }
    with open(out_communities_json, "w", encoding="utf-8") as f:
        json.dump(communities_output, f, ensure_ascii=False, indent=2)
    logger.info(f"社区划分结果已保存: {out_communities_json}")

    # 写 report JSON
    with open(out_report_json, "w", encoding="utf-8") as f:
        json.dump({"community_report": report, "stats": stats}, f, ensure_ascii=False, indent=2)
    logger.info(f"社区报告已保存: {out_report_json}")

    # 写可读文本报告
    lines = [
        "# 社区聚类报告 (Leiden)",
        "",
        "## 统计",
        f"- 节点数: {stats['nodes']}",
        f"- 边数: {stats['edges']}",
        f"- 社区数: {stats['communities']}",
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
    logger.info(f"社区文本报告已保存: {out_report_txt}")

    total_time = time.time() - start_time
    logger.info(
        "======== Leiden 社区聚类完成 ======== 社区数=%d, 节点数=%d, 边数=%d, 总耗时=%.2fs",
        stats["communities"], stats["nodes"], stats["edges"], total_time,
    )

    return {
        "node_communities": node_communities,
        "communities": communities,
        "stats": stats,
        "hierarchy": hierarchy_dict,
        "community_report": report,
        "elapsed_seconds": total_time,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="图谱 JSON 路径")
    parser.add_argument("-o", "--output-dir", default=None, help="输出目录")
    parser.add_argument("--max-cluster-size", type=int, default=1000)
    parser.add_argument("--use-lcc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-prompt", default=None, help="LLM prompt 路径")
    parser.add_argument("--skip-llm", action="store_true", help="跳过 LLM 丰富")
    parser.add_argument("--llm-batch-size", type=int, default=20, help="LLM 丰富每批社区数")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(args.input)
    result = run_fast_clustering(
        args.input, output_dir,
        max_cluster_size=args.max_cluster_size,
        use_lcc=args.use_lcc,
        seed=args.seed,
        llm_prompt_path=args.llm_prompt,
        enable_llm_enrichment=not args.skip_llm,
        llm_batch_size=args.llm_batch_size,
    )
    print(f"\n社区聚类完成: {result['stats']}")
