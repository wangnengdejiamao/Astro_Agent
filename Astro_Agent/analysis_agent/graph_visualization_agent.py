"""Knowledge-graph visualization and report agent.

This module turns the local white-dwarf KG into evidence products for the
research agent:

- overview PNG for the largest/highest-degree graph backbone,
- community-level PNG and interactive HTML,
- machine-readable graph statistics,
- Markdown graph report, optionally expanded by an LLM provider loaded from
  private .env files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import networkx as nx
import plotly.graph_objects as go

from .llm_client import LLMClient, load_default_env, load_model_config


REPO_ROOT = Path(__file__).resolve().parents[2]
KG_WORKSPACE = Path(os.getenv("ASTRO_AGENT_KG_WORKSPACE", str(REPO_ROOT / ".local_kg")))
DEFAULT_KG_DB = KG_WORKSPACE / "output" / "white_dwarf_kg" / "kg_index.sqlite"
DEFAULT_SUMMARY = KG_WORKSPACE / "output" / "white_dwarf_kg" / "production_full" / "summary.json"
DEFAULT_OUT = REPO_ROOT / "Astro_Agent" / "output" / "analysis_agent" / "kg_graph_report"


def configure_matplotlib_fonts() -> None:
    """Use a local Chinese-capable font when available; never fail the run."""
    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


configure_matplotlib_fonts()


@dataclass
class EdgeRow:
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    title: str
    source: str
    evidence: str


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> str:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_text(path: Path, text: str) -> str:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    return str(path)


def read_edges(db_path: Path, limit_edges: int | None = None) -> list[EdgeRow]:
    if not db_path.exists():
        raise FileNotFoundError(f"KG SQLite index not found: {db_path}")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    limit_sql = f" LIMIT {int(limit_edges)}" if limit_edges else ""
    rows = con.execute(
        """
        SELECT subject, subject_type, relation, object, object_type, title, source, evidence
        FROM kg_edges
        WHERE subject != '' AND object != '' AND relation != ''
        """
        + limit_sql
    ).fetchall()
    con.close()
    return [
        EdgeRow(
            subject=str(row["subject"]),
            subject_type=str(row["subject_type"] or "Unknown"),
            relation=str(row["relation"]),
            object=str(row["object"]),
            object_type=str(row["object_type"] or "Unknown"),
            title=str(row["title"] or ""),
            source=str(row["source"] or ""),
            evidence=str(row["evidence"] or ""),
        )
        for row in rows
    ]


def build_graph(edges: Iterable[EdgeRow]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for row in edges:
        graph.add_node(row.subject, node_type=row.subject_type)
        graph.add_node(row.object, node_type=row.object_type)
        graph.add_edge(
            row.subject,
            row.object,
            relation=row.relation,
            title=row.title,
            source=row.source[:500],
            evidence=row.evidence[:500],
        )
    return graph


def simplify_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    simple = nx.Graph()
    for node, data in graph.nodes(data=True):
        simple.add_node(node, **data)
    for u, v, data in graph.edges(data=True):
        if simple.has_edge(u, v):
            simple[u][v]["weight"] += 1
            simple[u][v]["relations"].add(data.get("relation", ""))
        else:
            simple.add_edge(u, v, weight=1, relations={data.get("relation", "")})
    for _, _, data in simple.edges(data=True):
        data["relations"] = sorted(r for r in data["relations"] if r)
    return simple


def detect_communities(simple: nx.Graph, seed: int = 42) -> tuple[dict[str, int], list[list[str]], str]:
    if simple.number_of_nodes() == 0:
        return {}, [], "empty"
    try:
        comms = nx.community.louvain_communities(simple, weight="weight", seed=seed, resolution=1.0)
        method = "networkx_louvain"
    except Exception:
        comms = list(nx.community.asyn_lpa_communities(simple, weight="weight", seed=seed))
        method = "networkx_async_label_propagation"
    ordered = [sorted(c) for c in comms if c]
    ordered.sort(key=len, reverse=True)
    node_to_comm = {}
    for cid, members in enumerate(ordered):
        for node in members:
            node_to_comm[node] = cid
    return node_to_comm, ordered, method


def node_type_counts(graph: nx.MultiDiGraph) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, data in graph.nodes(data=True):
        counts[str(data.get("node_type") or "Unknown")] += 1
    return counts


def relation_counts(graph: nx.MultiDiGraph) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, _, data in graph.edges(data=True):
        counts[str(data.get("relation") or "Unknown")] += 1
    return counts


def top_nodes(simple: nx.Graph, n: int = 50) -> list[dict[str, Any]]:
    degree = dict(simple.degree(weight="weight"))
    try:
        pagerank = nx.pagerank(simple, weight="weight", max_iter=200)
    except Exception:
        pagerank = {node: 0.0 for node in simple.nodes}
    rows = []
    for node in simple.nodes:
        rows.append(
            {
                "node": node,
                "node_type": simple.nodes[node].get("node_type", "Unknown"),
                "weighted_degree": float(degree.get(node, 0.0)),
                "pagerank": float(pagerank.get(node, 0.0)),
            }
        )
    rows.sort(key=lambda item: (item["pagerank"], item["weighted_degree"]), reverse=True)
    return rows[:n]


def community_summary(
    simple: nx.Graph,
    communities: list[list[str]],
    node_to_comm: dict[str, int],
    max_communities: int = 30,
) -> list[dict[str, Any]]:
    degree = dict(simple.degree(weight="weight"))
    summaries = []
    for cid, members in enumerate(communities[:max_communities]):
        type_counts = Counter(simple.nodes[node].get("node_type", "Unknown") for node in members)
        top = sorted(members, key=lambda node: degree.get(node, 0.0), reverse=True)[:12]
        rels: Counter[str] = Counter()
        internal_edges = 0
        for u, v, data in simple.subgraph(members).edges(data=True):
            internal_edges += 1
            for rel in data.get("relations", []):
                rels[rel] += int(data.get("weight", 1))
        summaries.append(
            {
                "community_id": cid,
                "size": len(members),
                "internal_edges": internal_edges,
                "node_type_counts": dict(type_counts.most_common()),
                "top_nodes": top,
                "top_relations": dict(rels.most_common(10)),
            }
        )
    return summaries


def graph_statistics(graph: nx.MultiDiGraph, simple: nx.Graph, community_rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    components = sorted(nx.connected_components(simple), key=len, reverse=True)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "simple_edges": simple.number_of_edges(),
        "density": nx.density(simple),
        "connected_components": len(components),
        "largest_component_nodes": len(components[0]) if components else 0,
        "node_type_counts": dict(node_type_counts(graph).most_common()),
        "relation_counts": dict(relation_counts(graph).most_common()),
        "community_method": method,
        "communities": len(community_rows),
        "top_communities": community_rows[:20],
        "top_nodes": top_nodes(simple, n=40),
    }


def subgraph_by_top_degree(simple: nx.Graph, max_nodes: int) -> nx.Graph:
    nodes = sorted(simple.nodes, key=lambda node: simple.degree(node, weight="weight"), reverse=True)[:max_nodes]
    return simple.subgraph(nodes).copy()


def color_for_type(node_type: str) -> str:
    palette = {
        "Paper": "#4c78a8",
        "AnalysisMethod": "#f58518",
        "WhiteDwarfCategory": "#54a24b",
        "Survey": "#e45756",
        "AstronomicalSource": "#72b7b2",
        "PhysicalModel": "#b279a2",
        "ObservationInstrument": "#ff9da6",
        "PhysicalParameter": "#9d755d",
        "Result": "#bab0ac",
    }
    return palette.get(node_type, "#8c8c8c")


def draw_overview_png(simple: nx.Graph, out_path: Path, max_nodes: int = 900, seed: int = 42) -> str:
    sub = subgraph_by_top_degree(simple, max_nodes)
    if sub.number_of_nodes() == 0:
        return ""
    pos = nx.spring_layout(sub, seed=seed, k=1.1 / math.sqrt(max(sub.number_of_nodes(), 1)), iterations=80, weight="weight")
    degree = dict(sub.degree(weight="weight"))
    sizes = [8 + 45 * math.log1p(degree.get(node, 0.0)) for node in sub.nodes]
    colors = [color_for_type(sub.nodes[node].get("node_type", "Unknown")) for node in sub.nodes]
    plt.figure(figsize=(24, 18))
    nx.draw_networkx_edges(sub, pos, alpha=0.055, width=0.35, edge_color="#222222")
    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color=colors, alpha=0.86, linewidths=0.0)
    labels = {row["node"]: row["node"][:45] for row in top_nodes(sub, n=35)}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=6, font_color="#111111")
    plt.title(f"White Dwarf KG Backbone: top {sub.number_of_nodes()} nodes by weighted degree", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return str(out_path)


def draw_community_png(simple: nx.Graph, node_to_comm: dict[str, int], out_path: Path) -> str:
    comm_graph = nx.Graph()
    for node, cid in node_to_comm.items():
        comm_graph.add_node(cid, size=comm_graph.nodes[cid].get("size", 0) + 1 if cid in comm_graph else 1)
    for u, v, data in simple.edges(data=True):
        cu = node_to_comm.get(u)
        cv = node_to_comm.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        weight = int(data.get("weight", 1))
        if comm_graph.has_edge(cu, cv):
            comm_graph[cu][cv]["weight"] += weight
        else:
            comm_graph.add_edge(cu, cv, weight=weight)
    if comm_graph.number_of_nodes() == 0:
        return ""
    pos = nx.spring_layout(comm_graph, seed=42, k=0.9, iterations=120, weight="weight")
    sizes = [20 + 14 * math.sqrt(comm_graph.nodes[node].get("size", 1)) for node in comm_graph.nodes]
    weights = [0.2 + math.log1p(data.get("weight", 1)) * 0.28 for _, _, data in comm_graph.edges(data=True)]
    plt.figure(figsize=(18, 14))
    nx.draw_networkx_edges(comm_graph, pos, width=weights, alpha=0.22, edge_color="#444444")
    nx.draw_networkx_nodes(comm_graph, pos, node_size=sizes, node_color="#3f6f8f", alpha=0.88)
    nx.draw_networkx_labels(comm_graph, pos, labels={n: str(n) for n in comm_graph.nodes}, font_size=7, font_color="white")
    plt.title("Community-Level White Dwarf Knowledge Graph", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    return str(out_path)


def write_interactive_html(simple: nx.Graph, out_path: Path, max_nodes: int = 1200) -> str:
    sub = subgraph_by_top_degree(simple, max_nodes)
    if sub.number_of_nodes() == 0:
        return ""
    pos = nx.spring_layout(sub, seed=42, k=1.2 / math.sqrt(max(sub.number_of_nodes(), 1)), iterations=80, weight="weight")
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in sub.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    degree = dict(sub.degree(weight="weight"))
    node_x = [pos[node][0] for node in sub.nodes]
    node_y = [pos[node][1] for node in sub.nodes]
    hover = [
        f"{node}<br>type={sub.nodes[node].get('node_type', 'Unknown')}<br>weighted_degree={degree.get(node, 0):.1f}"
        for node in sub.nodes
    ]
    colors = [sub.nodes[node].get("node_type", "Unknown") for node in sub.nodes]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=edge_x, y=edge_y, mode="lines", line={"width": 0.35, "color": "rgba(80,80,80,0.18)"}, hoverinfo="none"))
    fig.add_trace(
        go.Scattergl(
            x=node_x,
            y=node_y,
            mode="markers",
            marker={
                "size": [4 + 2.2 * math.log1p(degree.get(node, 0.0)) for node in sub.nodes],
                "color": [hash(c) % 20 for c in colors],
                "colorscale": "Turbo",
                "opacity": 0.86,
            },
            text=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=f"Interactive White Dwarf KG Backbone ({sub.number_of_nodes()} nodes)",
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return str(out_path)


def build_base_report(stats: dict[str, Any], summary_json: dict[str, Any] | None, artifacts: dict[str, str], llm_text: str = "") -> str:
    lines = [
        "# White Dwarf Knowledge Graph Report",
        "",
        "## Artifacts",
    ]
    for key, path in artifacts.items():
        lines.append(f"- {key}: `{path}`")
    lines.extend(
        [
            "",
            "## Global Statistics",
            f"- Nodes: {stats['nodes']}",
            f"- Directed KG edges: {stats['edges']}",
            f"- Simple undirected edges: {stats['simple_edges']}",
            f"- Connected components: {stats['connected_components']}",
            f"- Largest component nodes: {stats['largest_component_nodes']}",
            f"- Community method: {stats['community_method']}",
            f"- Communities summarized: {stats['communities']}",
            "",
            "## Node Types",
        ]
    )
    for key, value in list(stats["node_type_counts"].items())[:20]:
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Relation Types")
    for key, value in list(stats["relation_counts"].items())[:20]:
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Top Nodes")
    for row in stats["top_nodes"][:20]:
        lines.append(f"- {row['node']} ({row['node_type']}): pagerank={row['pagerank']:.4g}, degree={row['weighted_degree']:.1f}")
    lines.append("")
    lines.append("## Top Communities")
    for row in stats["top_communities"][:12]:
        top_nodes_str = ", ".join(row["top_nodes"][:8])
        top_types = ", ".join(f"{k}={v}" for k, v in list(row["node_type_counts"].items())[:5])
        lines.append(f"- Community {row['community_id']}: size={row['size']}, edges={row['internal_edges']}, types=[{top_types}], top_nodes=[{top_nodes_str}]")
    if summary_json:
        lines.extend(["", "## Source Build Summary"])
        for key in ("papers", "papers_in_graph", "categories", "chunks_scanned", "relationships", "unique_nodes_estimate"):
            if key in summary_json:
                lines.append(f"- {key}: {summary_json[key]}")
    if llm_text:
        lines.extend(["", "## LLM Scientific Interpretation", "", llm_text.strip()])
    return "\n".join(lines) + "\n"


def llm_interpretation(stats: dict[str, Any], provider: str, max_output_tokens: int = 5000) -> str:
    cfg = load_model_config(provider)
    client = LLMClient(cfg)
    if not client.available:
        return f"LLM skipped: missing API key environment variable `{cfg.api_key_env}`."
    compact = {
        "global": {k: stats[k] for k in ("nodes", "edges", "simple_edges", "connected_components", "largest_component_nodes", "community_method", "communities")},
        "node_type_counts": stats["node_type_counts"],
        "relation_counts": stats["relation_counts"],
        "top_nodes": stats["top_nodes"][:25],
        "top_communities": stats["top_communities"][:12],
    }
    system = (
        "You are a senior astrophysics knowledge-graph analyst. "
        "Write a rigorous Chinese report for a white-dwarf literature KG. "
        "Do not invent numbers. Distinguish graph-derived facts from interpretation."
    )
    user = (
        "请根据下面的知识图谱统计，写一份科研图谱报告。要求：\n"
        "1. 说明图谱规模、主要节点类型和关系类型。\n"
        "2. 解释最大的社区代表哪些天文学主题。\n"
        "3. 给出这个图谱如何服务未知源分析 agent，包括数据获取、方法迁移、SED/光谱/时域拟合、QA 审稿。\n"
        "4. 指出图谱偏差与下一步改进。\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )
    try:
        return client.complete(system, user, temperature=0.2, max_output_tokens=max_output_tokens)
    except Exception as exc:
        return (
            "LLM report generation failed, so this report falls back to deterministic graph statistics.\n\n"
            f"- provider: {cfg.provider}\n"
            f"- model: {cfg.model}\n"
            f"- api_key_env: {cfg.api_key_env}\n"
            f"- error: {type(exc).__name__}: {exc}\n"
        )


def run_graph_agent(
    kg_db: Path = DEFAULT_KG_DB,
    output_root: Path = DEFAULT_OUT,
    summary_path: Path = DEFAULT_SUMMARY,
    max_plot_nodes: int = 900,
    interactive_nodes: int = 1200,
    use_llm: bool = False,
    provider: str = "deepseek",
) -> dict[str, Any]:
    load_default_env()
    output_root = ensure_dir(output_root)
    edges = read_edges(kg_db)
    graph = build_graph(edges)
    simple = simplify_graph(graph)
    node_to_comm, comms, method = detect_communities(simple)
    comm_rows = community_summary(simple, comms, node_to_comm, max_communities=60)
    stats = graph_statistics(graph, simple, comm_rows, method)
    summary_json = read_summary(summary_path)

    artifacts = {
        "overview_png": draw_overview_png(simple, output_root / "kg_backbone_overview.png", max_nodes=max_plot_nodes),
        "community_png": draw_community_png(simple, node_to_comm, output_root / "kg_community_overview.png"),
        "interactive_html": write_interactive_html(simple, output_root / "kg_interactive_backbone.html", max_nodes=interactive_nodes),
        "stats_json": write_json(output_root / "kg_graph_stats.json", stats),
    }
    llm_text = llm_interpretation(stats, provider=provider) if use_llm else ""
    report = build_base_report(stats, summary_json, artifacts, llm_text=llm_text)
    artifacts["report_md"] = write_text(output_root / "kg_graph_report.md", report)
    write_json(output_root / "kg_graph_artifacts.json", artifacts)
    return {"output_root": str(output_root), "artifacts": artifacts, "llm_used": bool(use_llm), "provider": provider}


def read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate large KG images and graph report.")
    parser.add_argument("--kg-db", default=str(DEFAULT_KG_DB))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--max-plot-nodes", type=int, default=900)
    parser.add_argument("--interactive-nodes", type=int, default=1200)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--provider", default="deepseek")
    args = parser.parse_args()
    result = run_graph_agent(
        kg_db=Path(args.kg_db),
        output_root=Path(args.output_root),
        summary_path=Path(args.summary),
        max_plot_nodes=args.max_plot_nodes,
        interactive_nodes=args.interactive_nodes,
        use_llm=args.use_llm,
        provider=args.provider,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
