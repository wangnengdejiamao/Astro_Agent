
#!/usr/bin/env python3
from __future__ import annotations
"""
将 prompt2graph 生成的图谱 JSON 导入 Neo4j。

用法:
  python graph_to_neo4j.py <graph.json> [options
  python graph_to_neo4j.py output/paper_mini/260120_v2.json             # 导入图谱
  python graph_to_neo4j.py output/paper_mini/260120_v2.json --clear     # 清空后导入
  python graph_to_neo4j.py --clear                                      # 仅清空数据库（不导入）
"""


import argparse
import json
import logging
import re
import sys
from pathlib import Path

import neo4j
from graph_merger import merge_graphs

# 设置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Cypher 标识符合法字符
_IDENT_PATTERN = re.compile(r"[^a-zA-Z0-9_]")
_LABEL_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_rel_type(rel: str) -> str:
    """将关系类型转为合法的 Cypher 标识符（只保留字母、数字、下划线）。"""
    s = _IDENT_PATTERN.sub("_", (rel or "").strip())
    return s.upper() if s else "RELATED"


def _sanitize_label(lbl: str) -> str:
    """将标签转为合法的 Cypher 标签。"""
    s = _LABEL_PATTERN.sub("_", (lbl or "").strip())
    return s or "Node"


def _normalize_chunk_id(chunk_id: str | list) -> str:
    """将 chunk_id 统一为字符串。"""
    if isinstance(chunk_id, list):
        return ",".join(str(x) for x in chunk_id)
    return str(chunk_id) if chunk_id is not None else ""


def _normalize_text(value) -> str:
    """将 source/evidence 等字段统一规范为字符串。

    - 如果是列表: 使用 " | " 连接各元素
    - 如果是 None: 返回空字符串
    - 其他类型: 转为字符串并 strip
    """
    if value is None:
        return ""
    # 列表: 连接为一个字符串
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if v is not None]
        return " | ".join(p for p in parts if p) if parts else ""
    # 其他: 直接转为字符串
    return str(value).strip()


def _node_key(node: dict) -> tuple[str, str]:
    """(label, name) 作为节点去重键。"""
    raw_label = (node.get("label") or "entity").strip() or "entity"
    label = _sanitize_label(raw_label)
    props = node.get("properties") or {}
    name_raw = props.get("name")
    
    # 处理 name 可能是字符串或列表的情况
    if isinstance(name_raw, list):
        # 如果是列表，使用第一个元素，或连接所有元素
        name = ", ".join(str(x).strip() for x in name_raw if x) if name_raw else "_unnamed_"
    elif name_raw:
        name = str(name_raw).strip() or "_unnamed_"
    else:
        name = "_unnamed_"
    
    return (label, name)


def _props_for_neo4j(props: dict) -> dict:
    """过滤掉 None、保持可 JSON 序列化的属性。"""
    out = {}
    for k, v in (props or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            out[k] = json.dumps(v) if v else ""
        else:
            out[k] = v
    return out


def load_triples(path: Path) -> list[dict]:
    """加载图谱 JSON，返回三元组列表。"""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("图谱 JSON 应为三元组数组")
    return data


def clear_database(driver: neo4j.Driver) -> None:
    """
    清空 Neo4j 数据库中的所有节点、关系和属性。
    
    Args:
        driver: Neo4j 驱动实例
    """
    with driver.session() as session:
        try:
            # 使用 execute_write 执行清空操作（Neo4j 5.x+）
            def clear_tx(tx):
                result = tx.run("MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
                record = result.single()
                deleted_count = record["deleted"] if record else 0
                return deleted_count
            
            try:
                deleted_count = session.execute_write(clear_tx)
            except AttributeError:
                # Neo4j 4.x 使用 write_transaction
                try:
                    deleted_count = session.write_transaction(clear_tx)
                except AttributeError:
                    # 回退到直接执行
                    session.run("MATCH (n) DETACH DELETE n")
                    deleted_count = None
            
            if deleted_count is not None:
                logger.info(f"已清空数据库: 删除了 {deleted_count} 个节点及其所有关系")
            else:
                logger.info("已清空数据库: 所有节点和关系已删除")
        except Exception as e:
            logger.error(f"清空数据库时出错: {e}")
            raise


def run_import(
    triples: list[dict],
    driver: neo4j.Driver,
    *,
    batch_size: int = 500,
) -> tuple[int, int, int]:
    """
    将三元组导入 Neo4j。
    - 节点按 (label, name) MERGE，避免重复。
    - 关系使用 MERGE，避免重复关系（相同起止节点和关系类型）。
    返回 (节点数, 关系数, 跳过数)。
    """
    seen_nodes: set[tuple[str, str]] = set()
    created_nodes = 0
    created_rels = 0
    skipped = 0

    with driver.session() as session:
        # 批量处理：每个 batch 在一个事务中执行
        for i in range(0, len(triples), batch_size):
            batch = triples[i : i + batch_size]
            
            # 使用事务执行批量导入（兼容 Neo4j 4.x 和 5.x+）
            def import_batch(tx):
                nonlocal created_nodes, created_rels, skipped, seen_nodes
                for t in batch:
                    start = t.get("start_node") or {}
                    end = t.get("end_node") or {}
                    rel_type_raw = (t.get("relation") or "related").strip()
                    rel_type = _sanitize_rel_type(rel_type_raw)
                    source = _normalize_text(t.get("source"))
                    evidence = _normalize_text(t.get("evidence"))
                    chunk_id = _normalize_chunk_id(t.get("chunk_id") or "")

                    skey = _node_key(start)
                    ekey = _node_key(end)
                    if skey == ekey and rel_type in ("SELF", "SELF_REF"):
                        skipped += 1
                        continue

                    slabel = skey[0]
                    sname = skey[1]
                    elabel = ekey[0]
                    ename = ekey[1]
                    
                    # 获取属性，但排除 name（已在 MERGE 中使用）
                    sprops = _props_for_neo4j(start.get("properties"))
                    eprops = _props_for_neo4j(end.get("properties"))
                    sprops.pop("name", None)  # 避免覆盖 MERGE 键
                    eprops.pop("name", None)

                    # 使用字符串拼接构建查询，避免 f-string 与 Cypher 大括号冲突
                    # MERGE 节点：先匹配 name，再设置其他属性
                    sq = (
                        "MERGE (s:" + slabel + " {name: $sname}) "
                        "SET s += $sprops"
                    )
                    tx.run(sq, sname=sname, sprops=sprops)
                    if skey not in seen_nodes:
                        seen_nodes.add(skey)
                        created_nodes += 1

                    eq = (
                        "MERGE (e:" + elabel + " {name: $ename}) "
                        "SET e += $eprops"
                    )
                    tx.run(eq, ename=ename, eprops=eprops)
                    if ekey not in seen_nodes:
                        seen_nodes.add(ekey)
                        created_nodes += 1

                    # 创建关系：使用 CREATE 保留所有三元组信息
                    # 如果同一关系出现多次（不同 source/evidence），会创建多条边
                    rq = (
                        "MATCH (s:" + slabel + " {name: $sname}) "
                        "MATCH (e:" + elabel + " {name: $ename}) "
                        "CREATE (s)-[r:" + rel_type + " {source: $source, evidence: $evidence, chunk_id: $chunk_id}]->(e)"
                    )
                    tx.run(
                        rq,
                        sname=sname,
                        ename=ename,
                        source=source,
                        evidence=evidence,
                        chunk_id=chunk_id,
                    )
                    created_rels += 1
            
            # 使用 execute_write 执行批量导入（Neo4j 5.x+）
            # 兼容旧版本：如果 execute_write 不存在，尝试使用 write_transaction
            try:
                # Neo4j 5.x+ 使用 execute_write
                session.execute_write(import_batch)
            except AttributeError:
                # Neo4j 4.x 使用 write_transaction
                try:
                    session.write_transaction(import_batch)
                except AttributeError:
                    # 如果都不支持，直接使用 session.run（自动事务）
                    logger.warning("Neo4j driver 版本较旧，使用自动事务模式")
                    for t in batch:
                        start = t.get("start_node") or {}
                        end = t.get("end_node") or {}
                        rel_type_raw = (t.get("relation") or "related").strip()
                        rel_type = _sanitize_rel_type(rel_type_raw)
                        source = _normalize_text(t.get("source"))
                        evidence = _normalize_text(t.get("evidence"))
                        chunk_id = _normalize_chunk_id(t.get("chunk_id") or "")

                        skey = _node_key(start)
                        ekey = _node_key(end)
                        if skey == ekey and rel_type in ("SELF", "SELF_REF"):
                            skipped += 1
                            continue

                        slabel = skey[0]
                        sname = skey[1]
                        elabel = ekey[0]
                        ename = ekey[1]
                        
                        sprops = _props_for_neo4j(start.get("properties"))
                        eprops = _props_for_neo4j(end.get("properties"))
                        sprops.pop("name", None)
                        eprops.pop("name", None)

                        sq = (
                            "MERGE (s:" + slabel + " {name: $sname}) "
                            "SET s += $sprops"
                        )
                        session.run(sq, sname=sname, sprops=sprops)
                        if skey not in seen_nodes:
                            seen_nodes.add(skey)
                            created_nodes += 1

                        eq = (
                            "MERGE (e:" + elabel + " {name: $ename}) "
                            "SET e += $eprops"
                        )
                        session.run(eq, ename=ename, eprops=eprops)
                        if ekey not in seen_nodes:
                            seen_nodes.add(ekey)
                            created_nodes += 1

                        rq = (
                            "MATCH (s:" + slabel + " {name: $sname}) "
                            "MATCH (e:" + elabel + " {name: $ename}) "
                            "CREATE (s)-[r:" + rel_type + " {source: $source, evidence: $evidence, chunk_id: $chunk_id}]->(e)"
                        )
                        session.run(
                            rq,
                            sname=sname,
                            ename=ename,
                            source=source,
                            evidence=evidence,
                            chunk_id=chunk_id,
                        )
                        created_rels += 1

    return (created_nodes, created_rels, skipped)


def update_current_graph_and_import(
    new_graph_path: Path,
    driver: neo4j.Driver,
    *,
    current_graph_path: Path,
    clear_first: bool = True,
    batch_size: int = 500,
) -> None:
    """
    将新的图谱 JSON 与 current_graph 合并（支持增量），并导入 Neo4j。

    行为与命令行工具中的逻辑保持一致：
    - clear_first=True: current_graph 被 new_graph 覆盖，作为新的全量图谱；
    - clear_first=False: 如果 current_graph 已存在且非空，则与 new_graph 调用 merge_graphs 合并；
                         否则直接用 new_graph 初始化 current_graph。
    然后：
    - 清空 Neo4j 数据库；
    - 读取 current_graph 并按批次调用 run_import 导入节点和关系。
    """
    if not new_graph_path.is_file():
        raise FileNotFoundError(f"新图谱文件不存在: {new_graph_path}")

    logger.info("======== 开始Neo4j导入流程 ========")
    # 更新 current_graph.json
    if clear_first:
        # 重置 current_graph = new_graph
        current_graph_path.write_text(
            new_graph_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        logger.info(f"已重置 current_graph 为新图谱: {current_graph_path}")
    else:
        if current_graph_path.is_file() and current_graph_path.stat().st_size > 0:
            # 先将 current_graph 与新图谱合并到临时文件
            tmp_merged = current_graph_path.with_suffix(".merged.tmp.json")
            merged_path_str = merge_graphs(
                str(current_graph_path), str(new_graph_path), str(tmp_merged)
            )
            merged_path = Path(merged_path_str)
            current_graph_path.write_text(
                merged_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            try:
                tmp_merged.unlink()
            except OSError:
                pass
            logger.info(f"已合并 current_graph 与新图谱，并更新到: {current_graph_path}")
        else:
            # 如果 current_graph 不存在或为空，则直接以新图谱为当前图谱
            current_graph_path.write_text(
                new_graph_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            logger.info(f"current_graph 不存在或为空，已直接使用新图谱初始化: {current_graph_path}")

    # 此时 current_graph.json 已经是最新合并后的全量图谱
    # 先清空 Neo4j，再基于 current_graph 导入，保证 Neo4j 与 current_graph 一致
    clear_database(driver)

    current_triples = load_triples(current_graph_path)
    logger.info(f"开始导入 current_graph，共 {len(current_triples)} 条三元组: {current_graph_path}")

    nodes, rels, skipped = run_import(
        current_triples,
        driver,
        batch_size=batch_size,
    )
    logger.info(f"导入完成: 节点 {nodes}, 关系 {rels}, 跳过 {skipped}")
    logger.info("======== Neo4j导入流程完成 ========")


def import_meta_graph(
    meta_graph_path: Path,
    driver: neo4j.Driver,
    meta_eval_path: Path | None = None,
    batch_size: int = 500,
) -> tuple[int, int, int]:
    """
    将二级图谱（meta graph）导入 Neo4j。

    Args:
        meta_graph_path: 二级图谱 JSON 文件路径
        driver: Neo4j 驱动
        meta_eval_path: 元图谱评分文件路径（可选）
        batch_size: 批处理大小

    Returns:
        (节点数, 关系数, 跳过数)
    """
    with open(meta_graph_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    meta_graph = meta_data.get("meta_graph", meta_data)
    nodes = meta_graph.get("nodes", [])
    edges = meta_graph.get("edges", [])

    eval_data = None
    if meta_eval_path and meta_eval_path.is_file():
        try:
            with open(meta_eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"加载元图谱评分文件失败: {meta_eval_path}, 错误: {e}")
            eval_data = None

    eval_map: dict[str, dict] = {}
    if eval_data and isinstance(eval_data, dict):
        eval_edges = eval_data.get("evaluation", {}).get("edges", [])
        for item in eval_edges:
            edge_id = item.get("edge_id") or item.get("meta_edge_id")
            if edge_id:
                eval_map[edge_id] = item

    with driver.session() as session:
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]

            def import_batch(tx):
                created_nodes = 0
                for node in batch:
                    node_id = node.get("id", "")
                    subject = _normalize_text(node.get("subject"))
                    relation = _normalize_text(node.get("relation"))
                    object_text = _normalize_text(node.get("object"))
                    schema_type_subj = _normalize_text(node.get("schema_type_subj"))
                    schema_type_obj = _normalize_text(node.get("schema_type_obj"))
                    source = _normalize_text(node.get("source"))
                    evidence = _normalize_text(node.get("evidence"))
                    chunk_ids = node.get("chunk_ids", [])
                    if isinstance(chunk_ids, list):
                        chunk_ids_str = ",".join(str(x) for x in chunk_ids)
                    else:
                        chunk_ids_str = str(chunk_ids)

                    props = {
                        "triple_id": node_id,
                        "subject": subject,
                        "relation": relation,
                        "object": object_text,
                        "schema_type_subj": schema_type_subj,
                        "schema_type_obj": schema_type_obj,
                        "source": source,
                        "evidence": evidence,
                        "chunk_ids": chunk_ids_str,
                    }

                    sq = (
                        "MERGE (n:TripleNode {triple_id: $triple_id}) "
                        "SET n += $props"
                    )
                    tx.run(sq, triple_id=node_id, props=props)
                    created_nodes += 1
                return created_nodes

            try:
                session.execute_write(import_batch)
            except AttributeError:
                def import_batch_legacy(tx):
                    created_nodes = 0
                    for node in batch:
                        node_id = node.get("id", "")
                        subject = _normalize_text(node.get("subject"))
                        relation = _normalize_text(node.get("relation"))
                        object_text = _normalize_text(node.get("object"))
                        schema_type_subj = _normalize_text(node.get("schema_type_subj"))
                        schema_type_obj = _normalize_text(node.get("schema_type_obj"))
                        source = _normalize_text(node.get("source"))
                        evidence = _normalize_text(node.get("evidence"))
                        chunk_ids = node.get("chunk_ids", [])
                        if isinstance(chunk_ids, list):
                            chunk_ids_str = ",".join(str(x) for x in chunk_ids)
                        else:
                            chunk_ids_str = str(chunk_ids)

                        props = {
                            "triple_id": node_id,
                            "subject": subject,
                            "relation": relation,
                            "object": object_text,
                            "schema_type_subj": schema_type_subj,
                            "schema_type_obj": schema_type_obj,
                            "source": source,
                            "evidence": evidence,
                            "chunk_ids": chunk_ids_str,
                        }

                        sq = (
                            "MERGE (n:TripleNode {triple_id: $triple_id}) "
                            "SET n += $props"
                        )
                        tx.run(sq, triple_id=node_id, props=props)
                        created_nodes += 1
                    return created_nodes
                try:
                    session.write_transaction(import_batch_legacy)
                except AttributeError:
                    session.run("MATCH (n:TripleNode) DETACH DELETE n")
                    for node in batch:
                        node_id = node.get("id", "")
                        subject = _normalize_text(node.get("subject"))
                        relation = _normalize_text(node.get("relation"))
                        object_text = _normalize_text(node.get("object"))
                        chunk_ids = node.get("chunk_ids", [])
                        if isinstance(chunk_ids, list):
                            chunk_ids_str = ",".join(str(x) for x in chunk_ids)
                        else:
                            chunk_ids_str = str(chunk_ids)
                        props = {
                            "triple_id": node_id,
                            "subject": subject,
                            "relation": relation,
                            "object": object_text,
                            "chunk_ids": chunk_ids_str,
                        }
                        sq = "MERGE (n:TripleNode {triple_id: $triple_id}) SET n += $props"
                        session.run(sq, triple_id=node_id, props=props)

        for i in range(0, len(edges), batch_size):
            batch = edges[i : i + batch_size]

            def import_edges_batch(tx):
                created_rels = 0
                for edge in batch:
                    source_id = edge.get("source_triple_id") or edge.get("source_id", "")
                    target_id = edge.get("target_triple_id") or edge.get("target_id", "")
                    rel_type_raw = (edge.get("relation") or "related").strip()
                    rel_type = _sanitize_rel_type(rel_type_raw)
                    evidence = _normalize_text(edge.get("evidence"))

                    eval_item = eval_map.get(edge.get("id", ""))
                    source_alignment = None
                    if eval_item:
                        scores = eval_item.get("scores", {})
                        source_alignment = scores.get("source_alignment")

                    props = {"evidence": evidence}
                    if source_alignment is not None:
                        props["source_alignment"] = source_alignment

                    rq = (
                        "MATCH (s:TripleNode {triple_id: $source_id}) "
                        "MATCH (t:TripleNode {triple_id: $target_id}) "
                        "CREATE (s)-[r:" + rel_type + " $props]->(t)"
                    )
                    tx.run(rq, source_id=source_id, target_id=target_id, props=props)
                    created_rels += 1
                return created_rels

            try:
                session.execute_write(import_edges_batch)
            except AttributeError:
                def import_edges_batch_legacy(tx):
                    created_rels = 0
                    for edge in batch:
                        source_id = edge.get("source_triple_id") or edge.get("source_id", "")
                        target_id = edge.get("target_triple_id") or edge.get("target_id", "")
                        rel_type_raw = (edge.get("relation") or "related").strip()
                        rel_type = _sanitize_rel_type(rel_type_raw)
                        evidence = _normalize_text(edge.get("evidence"))

                        eval_item = eval_map.get(edge.get("id", ""))
                        source_alignment = None
                        if eval_item:
                            scores = eval_item.get("scores", {})
                            source_alignment = scores.get("source_alignment")

                        props = {"evidence": evidence}
                        if source_alignment is not None:
                            props["source_alignment"] = source_alignment

                        rq = (
                            "MATCH (s:TripleNode {triple_id: $source_id}) "
                            "MATCH (t:TripleNode {triple_id: $target_id}) "
                            "CREATE (s)-[r:" + rel_type + " $props]->(t)"
                        )
                        tx.run(rq, source_id=source_id, target_id=target_id, props=props)
                        created_rels += 1
                    return created_rels
                try:
                    session.write_transaction(import_edges_batch_legacy)
                except AttributeError:
                    for edge in batch:
                        source_id = edge.get("source_triple_id") or edge.get("source_id", "")
                        target_id = edge.get("target_triple_id") or edge.get("target_id", "")
                        rel_type_raw = (edge.get("relation") or "related").strip()
                        rel_type = _sanitize_rel_type(rel_type_raw)
                        evidence = _normalize_text(edge.get("evidence"))

                        eval_item = eval_map.get(edge.get("id", ""))
                        source_alignment = None
                        if eval_item:
                            scores = eval_item.get("scores", {})
                            source_alignment = scores.get("source_alignment")

                        props = {"evidence": evidence}
                        if source_alignment is not None:
                            props["source_alignment"] = source_alignment

                        rq = (
                            "MATCH (s:TripleNode {triple_id: $source_id}) "
                            "MATCH (t:TripleNode {triple_id: $target_id}) "
                            "CREATE (s)-[r:" + rel_type + " $props]->(t)"
                        )
                        session.run(rq, source_id=source_id, target_id=target_id, props=props)

    logger.info(f"二级图谱导入完成: {len(nodes)} 节点, {len(edges)} 关系")
    return (len(nodes), len(edges), 0)


def import_community_evaluation(
    community_eval_path: Path,
    driver: neo4j.Driver,
) -> int:
    """
    将社区评分导入 Neo4j。

    Args:
        community_eval_path: 社区评分 JSON 文件路径
        driver: Neo4j 驱动

    Returns:
        导入的社区数量
    """
    with open(community_eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    evaluation = eval_data.get("evaluation", [])
    stats = eval_data.get("stats", {})
    avg_scores = stats.get("avg_scores", {})

    if not evaluation:
        logger.warning("社区评分为空，跳过导入")
        return 0

    community_ids = []
    with driver.session() as session:
        def import_communities(tx):
            imported = 0
            for item in evaluation:
                community_id = item.get("community_id", "")
                if not community_id:
                    continue

                name = _normalize_text(item.get("name"))
                summary = _normalize_text(item.get("summary"))
                size = item.get("size", 0)

                scores = item.get("scores", {})
                topic_coherence = scores.get("topic_coherence")
                interpretability = scores.get("interpretability")
                coverage = scores.get("coverage")
                redundancy = scores.get("redundancy")
                source_alignment = scores.get("source_alignment")

                props = {
                    "community_id": community_id,
                    "name": name,
                    "summary": summary,
                    "size": size,
                    "topic_coherence": topic_coherence,
                    "interpretability": interpretability,
                    "coverage": coverage,
                    "redundancy": redundancy,
                    "source_alignment": source_alignment,
                }

                sq = (
                    "MERGE (c:Community {community_id: $community_id}) "
                    "SET c += $props"
                )
                tx.run(sq, community_id=community_id, props=props)
                community_ids.append(community_id)
                imported += 1
            return imported

        try:
            count = session.execute_write(import_communities)
        except AttributeError:
            def import_communities_legacy(tx):
                imported = 0
                for item in evaluation:
                    community_id = item.get("community_id", "")
                    if not community_id:
                        continue

                    name = _normalize_text(item.get("name"))
                    summary = _normalize_text(item.get("summary"))
                    size = item.get("size", 0)

                    scores = item.get("scores", {})
                    props = {
                        "community_id": community_id,
                        "name": name,
                        "summary": summary,
                        "size": size,
                        "topic_coherence": scores.get("topic_coherence"),
                        "interpretability": scores.get("interpretability"),
                        "coverage": scores.get("coverage"),
                        "redundancy": scores.get("redundancy"),
                        "source_alignment": scores.get("source_alignment"),
                    }

                    sq = (
                        "MERGE (c:Community {community_id: $community_id}) "
                        "SET c += $props"
                    )
                    tx.run(sq, community_id=community_id, props=props)
                    community_ids.append(community_id)
                    imported += 1
                return imported
            try:
                count = session.write_transaction(import_communities_legacy)
            except AttributeError:
                for item in evaluation:
                    community_id = item.get("community_id", "")
                    if not community_id:
                        continue

                    name = _normalize_text(item.get("name"))
                    summary = _normalize_text(item.get("summary"))
                    size = item.get("size", 0)

                    scores = item.get("scores", {})
                    props = {
                        "community_id": community_id,
                        "name": name,
                        "summary": summary,
                        "size": size,
                        "topic_coherence": scores.get("topic_coherence"),
                        "interpretability": scores.get("interpretability"),
                        "coverage": scores.get("coverage"),
                        "redundancy": scores.get("redundancy"),
                        "source_alignment": scores.get("source_alignment"),
                    }

                    sq = (
                        "MERGE (c:Community {community_id: $community_id}) "
                        "SET c += $props"
                    )
                    session.run(sq, community_id=community_id, props=props)
                    community_ids.append(community_id)
                count = len(community_ids)

    logger.info(f"社区评分导入完成: {count} 个社区")
    if avg_scores:
        logger.info(f"平均分数: {avg_scores}")
    return count


def main() -> None:
    ap = argparse.ArgumentParser(
        description="将图谱 JSON 导入 Neo4j，或仅清空数据库",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
    示例:
    python graph_to_neo4j.py output/paper_mini/part1.json
    python graph_to_neo4j.py output/paper_mini/part1.json --clear
    python graph_to_neo4j.py --clear
    """,
    )
    ap.add_argument(
        "json_path",
        type=Path,
        nargs="?",
        default=None,
        help="本次新增的图谱 JSON 文件路径（三元组数组）。如果只指定 --clear 而不提供文件，则仅清空数据库和 current_graph",
    )
    ap.add_argument(
        "--uri",
        default="bolt://localhost:7687",
        help="Neo4j Bolt URI",
    )
    ap.add_argument(
        "--user",
        default="neo4j",
        help="Neo4j 用户名",
    )
    ap.add_argument(
        "--password",
        default="password",
        help="Neo4j 密码",
    )
    ap.add_argument(
        "--clear",
        action="store_true",
        help="清空图库（DETACH DELETE 所有节点与关系）。如果未指定 json_path，则仅执行清空操作，并重置 current_graph",
    )
    ap.add_argument(
        "--current-graph",
        type=Path,
        default=Path("current_graph.json"),
        help="维护当前 Neo4j 图谱状态的 JSON 文件路径（默认: current_graph.json）",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="批处理大小",
    )
    args = ap.parse_args()

    # 验证参数组合
    if not args.json_path and not args.clear:
        ap.error("必须提供 json_path 或指定 --clear（或两者都指定）")

    current_graph_path: Path = args.current_graph
    driver = neo4j.GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        # 情况 1：只指定了 --clear 而没有 JSON 文件 → 仅清空数据库 & 重置 current_graph
        if args.clear and not args.json_path:
            print("仅执行清空操作...")
            clear_database(driver)
            # 重置 current_graph.json（写入空列表）
            current_graph_path.write_text("[]", encoding="utf-8")
            print("数据库清空完成！")
            return

        # 情况 2/3：提供了 JSON 文件（可能带或不带 --clear）
        path = args.json_path
        if not path.is_file():
            print(f"错误: 文件不存在 {path}", file=sys.stderr)
            sys.exit(1)

        # 读取新图谱（仅用于日志）
        new_triples = load_triples(path)
        print(f"已加载新增图谱 {len(new_triples)} 条三元组: {path}")

        # 使用统一的封装函数处理 current_graph 更新与导入
        update_current_graph_and_import(
            new_graph_path=path,
            driver=driver,
            current_graph_path=current_graph_path,
            clear_first=args.clear,
            batch_size=args.batch_size,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    main()
