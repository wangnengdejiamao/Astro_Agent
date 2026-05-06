
import json
import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter
from collections import deque
import sys
import os

# 添加项目根目录到路径
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .utils.graph_processor import (
    load_graph_from_json,
    extract_aliases,
    build_canonical_key,
)
from .utils.community_display import get_community_members_from_graph
from .utils.logger import logger


class GraphAnalyzer:
    """
    图分析器类，提供图的统计信息、社区信息、节点信息、边信息的展示
    以及高效的子图查找功能
    """
    
    def __init__(self, graph_path: str):
        """
        初始化图分析器
        
        Args:
            graph_path: 图JSON文件的路径
        """
        self.graph_path = graph_path
        self.graph: Optional[nx.MultiDiGraph] = None
        self.keyword_index: Dict[str, Set[str]] = {}  # 关键词到节点ID的反向索引
        self.name_index: Dict[str, Set[str]] = {}  # 节点名称/别名到节点ID的索引
        self.labeled_alias_index: Dict[Tuple[str, str], Set[str]] = {}  # (label, alias) -> 节点ID
        self.node_mapping: Dict[Tuple[str, Any], Set[str]] = {}  # 规范key -> 节点ID集合
        self.node_id_to_names: Dict[str, List[str]] = {}  # 节点ID到名称列表的映射
        self._load_graph()
        self._build_indices()
    
    def _load_graph(self):
        """加载图数据"""
        logger.info(f"正在加载图: {self.graph_path}")
        self.graph = load_graph_from_json(self.graph_path)
        logger.info(f"图加载完成: {self.graph.number_of_nodes()} 个节点, {self.graph.number_of_edges()} 条边")
    
    def _build_indices(self):
        """构建索引以提高搜索效率"""
        logger.info("正在构建索引...")
        self.keyword_index = defaultdict(set)
        self.name_index = defaultdict(set)
        self.labeled_alias_index = defaultdict(set)
        self.node_mapping = defaultdict(set)
        self.node_id_to_names = defaultdict(list)
        
        for node_id, node_data in self.graph.nodes(data=True): # 这里data=True 表示返回节点和数据字典
            props = node_data.get("properties", {})
            raw_name = props.get("name", "")
            node_label = node_data.get("label", "")
            aliases = extract_aliases(raw_name)
            
            # 构建别名映射与索引（兼容 list[str] 或 str 的 name 字段）
            for alias in aliases:
                normalized = alias.strip()
                if not normalized:
                    continue
                key = normalized.lower()
                self.name_index[key].add(node_id)
                self.labeled_alias_index[(node_label, key)].add(node_id)
                if normalized not in self.node_id_to_names[node_id]:
                    self.node_id_to_names[node_id].append(normalized)
            
            # 构建节点规范 key 映射（参考 graph_processor.node_mapping）
            canonical_key = build_canonical_key(node_label, aliases, raw_name)
            self.node_mapping[canonical_key].add(node_id)
            
            # 构建关键词索引（从所有属性中提取关键词）
            all_text = []
            for key, value in props.items():
                if isinstance(value, str):
                    all_text.append(value.lower())
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            all_text.append(item.lower())
            
            # 将文本分割成关键词（简单的分词）
            for text in all_text:
                words = text.split()
                for word in words:
                    if len(word) > 2:  # 忽略太短的词
                        self.keyword_index[word].add(node_id)
        
        logger.info(f"索引构建完成: {len(self.keyword_index)} 个关键词, {len(self.name_index)} 个节点名称")
    
    def get_primary_name(self, node_id: str) -> str:
        """
        获取节点的首选名称，若不存在则返回节点ID
        """
        if not node_id or node_id not in self.graph:
            return ""
        names = self.node_id_to_names.get(node_id, [])
        if names:
            return names[0]
        props = self.graph.nodes[node_id].get("properties", {})
        name = props.get("name")
        if isinstance(name, str):
            return name
        if isinstance(name, list) and name:
            return str(name[0])
        return str(node_id)
    
    def get_node_ids_by_name(self, node_name: str, node_label: Optional[str] = None) -> Set[str]:
        """
        根据节点名称或别名（大小写不敏感）返回匹配的节点ID集合
        
        Args:
            node_name: 待查找的名称/别名
            node_label: 可选的节点类型，用于缩小匹配范围
        """
        if not node_name:
            return set()
        
        normalized = node_name.strip().lower()
        if not normalized:
            return set()
        
        # 先尝试 label + alias 精确命中
        if node_label:
            labeled_key = (node_label, normalized)
            if labeled_key in self.labeled_alias_index:
                return set(self.labeled_alias_index[labeled_key])
        
        # 退化为全局别名匹配
        if normalized in self.name_index:
            return set(self.name_index[normalized])
        
        # 最后使用规范 key 映射做一次扫描（适配 list alias）
        matches: Set[str] = set()
        for (label, alias_tuple), node_ids in self.node_mapping.items():
            if node_label and label != node_label:
                continue
            if isinstance(alias_tuple, tuple) and normalized in alias_tuple:
                matches.update(node_ids)
                break  # 命中即可返回，避免全量扫描
        return matches
    
    def resolve_identifier_to_name(self, identifier: Optional[str]) -> str:
        """
        将节点标识（ID或名称）统一解析为展示名称
        """
        if not identifier:
            return ""
        resolved = str(identifier).strip()
        if not resolved:
            return ""
        
        if self.graph and resolved in self.graph:
            return self.get_primary_name(resolved)
        
        node_ids = self.get_node_ids_by_name(resolved)
        if node_ids:
            # 返回第一个匹配节点的名称（保持稳定性）
            return self.get_primary_name(next(iter(node_ids)))
        
        return resolved
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        获取图的统计信息
        
        Returns:
            包含图的各项统计信息的字典
        """
        if not self.graph:
            return {}
        
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "is_multigraph": self.graph.is_multigraph(),
            "density": nx.density(self.graph),
        }
        
        # 连通性分析
        if not self.graph.is_directed():
            undirected = self.graph.to_undirected()
            stats["connected_components"] = nx.number_connected_components(undirected)
        else:
            stats["weakly_connected_components"] = nx.number_weakly_connected_components(self.graph)
            stats["strongly_connected_components"] = nx.number_strongly_connected_components(self.graph)
        
        # 度统计
        degrees = dict(self.graph.degree())
        if degrees:
            stats["avg_degree"] = sum(degrees.values()) / len(degrees)
            stats["max_degree"] = max(degrees.values())
            stats["min_degree"] = min(degrees.values())
        
        return stats
    
    def get_community_stats(self) -> Dict[str, Any]:
        """
        获取社区的统计信息
        
        Returns:
            包含社区统计信息的字典
        """
        if not self.graph:
            return {}
        
        communities = get_community_members_from_graph(self.graph)
        
        stats = {
            "total_communities": len(communities),
            "communities": {}
        }
        
        if communities:
            member_counts = [comm.get("member_count", 0) for comm in communities.values()]
            stats["avg_members_per_community"] = sum(member_counts) / len(member_counts)
            stats["max_members"] = max(member_counts)
            stats["min_members"] = min(member_counts)
            
            # 每个社区的详细信息
            for comm_name, comm_info in communities.items():
                stats["communities"][comm_name] = {
                    "name": comm_info.get("name", ""),
                    "description": comm_info.get("description", ""),
                    "member_count": comm_info.get("member_count", 0),
                    "members": comm_info.get("members", [])[:10]  # 只保存前10个成员
                }
        
        return stats

    def get_community_members(self, community_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        获取指定社区或全部社区的成员信息。

        Args:
            community_name: 社区名称，None 表示返回全部社区。

        Returns:
            {community_name: {node_id, members, member_nodes, ...}, ...}
        """
        if not self.graph:
            return {}

        communities = get_community_members_from_graph(self.graph, community_name=community_name)
        # 返回浅拷贝，避免外部修改内部结构
        return {name: dict(info) for name, info in communities.items()}
    
    def get_node_stats(self) -> Dict[str, Any]:
        """
        获取节点的统计信息
        
        Returns:
            包含节点统计信息的字典
        """
        if not self.graph:
            return {}
        
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "nodes_by_label": {},
            "nodes_by_level": {}
        }
        
        # 按label分类统计
        label_counter = Counter()
        level_counter = Counter()
        
        for node_id, node_data in self.graph.nodes(data=True):
            label = node_data.get("label", "unknown")
            level = node_data.get("level", 0)
            label_counter[label] += 1
            level_counter[level] += 1
        
        stats["nodes_by_label"] = dict(label_counter)
        stats["nodes_by_level"] = dict(level_counter)
        
        return stats
    
    def get_edge_stats(self) -> Dict[str, Any]:
        """
        获取边的统计信息
        
        Returns:
            包含边统计信息的字典
        """
        if not self.graph:
            return {}
        
        stats = {
            "total_edges": self.graph.number_of_edges(),
            "edges_by_relation": {}
        }
        
        # 按关系类型统计
        relation_counter = Counter()
        for u, v, edge_data in self.graph.edges(data=True):
            relation = edge_data.get("relation", "unknown")
            relation_counter[relation] += 1
        
        stats["edges_by_relation"] = dict(relation_counter)
        
        return stats
    
    def get_structure_info(self) -> Dict[str, Any]:
        """
        获取图的结构信息
        
        Returns:
            包含图结构信息的字典
        """
        if not self.graph:
            return {}
        
        info = {
            "graph_type": "MultiDiGraph" if self.graph.is_multigraph() and self.graph.is_directed() else "Graph",
            "node_types": set(),
            "relation_types": set(),
            "max_path_length": 0,
        }
        
        # 收集所有节点类型和关系类型
        for node_id, node_data in self.graph.nodes(data=True):
            label = node_data.get("label", "")
            if label:
                info["node_types"].add(label)
        
        for u, v, edge_data in self.graph.edges(data=True):
            relation = edge_data.get("relation", "")
            if relation:
                info["relation_types"].add(relation)
        
        info["node_types"] = list(info["node_types"])
        info["relation_types"] = list(info["relation_types"])
        
        # 计算最大路径长度（使用BFS找到最长的简单路径）
        try:
            # 对于大型图，只采样部分节点计算
            nodes = list(self.graph.nodes())
            if len(nodes) > 1000:
                import random
                sample_nodes = random.sample(nodes, min(100, len(nodes)))
            else:
                sample_nodes = nodes
            
            max_length = 0
            for start in sample_nodes[:10]:  # 只检查前10个节点
                lengths = nx.single_source_shortest_path_length(self.graph, start)
                if lengths:
                    max_length = max(max_length, max(lengths.values()))
            
            info["max_path_length"] = max_length
        except:
            info["max_path_length"] = "N/A"
        
        return info
    
    def find_subgraph(self, 
                     depth: int = 2,
                     target_keywords: Optional[List[str]] = None,
                     target_node_names: Optional[List[str]] = None,
                     relation_filters: Optional[List[str]] = None) -> nx.MultiDiGraph:
        """
        查找子图（高效搜索）
        
        Args:
            depth: 搜索深度（从匹配节点开始向外扩展的层数）
            target_keywords: 目标关键词列表，用于匹配节点属性中的文本
            target_node_names: 目标节点名称列表，精确匹配节点名称
            relation_filters: 关系类型过滤器，只保留指定类型的关系
        
        Returns:
            包含匹配节点的子图
        """
        if not self.graph:
            return nx.MultiDiGraph()
        
        # 找到所有匹配的起始节点
        start_nodes = set()
        
        # 通过关键词匹配
        if target_keywords:
            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                # 在关键词索引中查找
                if keyword_lower in self.keyword_index:
                    start_nodes.update(self.keyword_index[keyword_lower])
                # 也在名称索引中查找（部分匹配）
                for name, node_ids in self.name_index.items():
                    if keyword_lower in name:
                        start_nodes.update(node_ids)
        
        # 通过节点名称匹配
        if target_node_names:
            for name in target_node_names:
                name_lower = name.lower()
                if name_lower in self.name_index:
                    start_nodes.update(self.name_index[name_lower])
        
        if not start_nodes:
            logger.warning("警告: 未找到匹配的起始节点")
            return nx.MultiDiGraph()
        
        logger.info(f"找到 {len(start_nodes)} 个起始节点，开始构建子图（深度={depth}）...")
        
        # 使用BFS扩展子图
        visited_nodes = set(start_nodes)
        queue = deque([(node, 0) for node in start_nodes])  # (node_id, current_depth)
        
        while queue:
            current_node, current_depth = queue.popleft()
            
            if current_depth >= depth:
                continue
            
            # 扩展邻居节点
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, current_depth + 1))
            
            # 对于有向图，也考虑入边
            if self.graph.is_directed():
                for predecessor in self.graph.predecessors(current_node):
                    if predecessor not in visited_nodes:
                        visited_nodes.add(predecessor)
                        queue.append((predecessor, current_depth + 1))
        
        # 构建子图
        subgraph = self.graph.subgraph(visited_nodes).copy()
        
        # 应用关系过滤器
        if relation_filters:
            edges_to_remove = []
            for u, v, edge_data in subgraph.edges(data=True):
                relation = edge_data.get("relation", "")
                if relation not in relation_filters:
                    edges_to_remove.append((u, v))
            
            for u, v in edges_to_remove:
                if subgraph.has_edge(u, v):
                    subgraph.remove_edge(u, v)
        
        logger.info(f"子图构建完成: {subgraph.number_of_nodes()} 个节点, {subgraph.number_of_edges()} 条边")
        return subgraph
    
    def find_subgraph_by_node_id(self, node_id: str, depth: int = 2) -> nx.MultiDiGraph:
        """
        通过节点ID查找子图
        
        Args:
            node_id: 节点ID
            depth: 搜索深度
        
        Returns:
            包含匹配节点的子图
        """
        if not self.graph:
            logger.warning("警告: 图未加载")
            return nx.MultiDiGraph()
        
        if node_id not in self.graph:
            logger.warning(f"警告: 节点 {node_id} 不在图中")
            logger.info(f"图中节点总数: {self.graph.number_of_nodes()}")
            # 尝试查找相似的节点
            similar_nodes = [nid for nid in self.graph.nodes() if str(node_id) in str(nid) or str(nid) in str(node_id)]
            if similar_nodes:
                logger.info(f"找到相似节点: {similar_nodes[:5]}")
            return nx.MultiDiGraph()
        
        logger.info(f"找到节点 {node_id}，开始扩展子图（深度={depth}）...")
        start_nodes = {node_id}
        subgraph = self._expand_subgraph(start_nodes, depth)
        logger.info(f"子图扩展完成: {subgraph.number_of_nodes()} 个节点, {subgraph.number_of_edges()} 条边")
        return subgraph
    
    def find_subgraph_by_node_name(
        self,
        node_name: str,
        depth: int = 2,
        node_label: Optional[str] = None,
    ) -> nx.MultiDiGraph:
        """
        通过节点名称/别名查找子图（精确匹配，兼容别名列表）
        
        Args:
            node_name: 节点名称（将进行精确匹配，不进行子串匹配）
            depth: 搜索深度
            node_label: 可选指定节点类型，若提供则优先匹配对应 label
        
        Returns:
            包含匹配节点的子图
        """
        if not self.graph:
            return nx.MultiDiGraph()
        
        cleaned_name = (node_name or "").strip()
        if not cleaned_name:
            logger.warning("未提供有效的节点名称")
            return nx.MultiDiGraph()
        
        node_name_lower = cleaned_name.lower()
        
        # 使用别名索引（含 label）定位节点
        start_nodes = self.get_node_ids_by_name(cleaned_name, node_label=node_label)
        
        # 若 label 精确命中失败，再尝试无 label 的别名索引
        if not start_nodes and node_label:
            start_nodes = self.get_node_ids_by_name(cleaned_name, node_label=None)
        
        # 最后兜底：遍历图，兼容未入索引的别名
        if not start_nodes:
            for node_id, node_data in self.graph.nodes(data=True):
                if node_label and node_data.get("label") != node_label:
                    continue
                props = node_data.get("properties", {})
                aliases = extract_aliases(props.get("name", ""))
                for alias in aliases:
                    if alias.lower() == node_name_lower:
                        start_nodes.add(node_id)
                        break
        
        if not start_nodes:
            logger.warning(f"未找到名称为 '{node_name}' 的节点（精确匹配）")
            return nx.MultiDiGraph()
        
        logger.info(f"找到 {len(start_nodes)} 个精确匹配的节点，开始扩展子图（深度={depth}）...")
        return self._expand_subgraph(start_nodes, depth)
    
    def _expand_subgraph(self, start_nodes: Set[str], depth: int) -> nx.MultiDiGraph:
        """扩展子图（BFS）"""
        visited_nodes = set(start_nodes)
        queue = deque([(node, 0) for node in start_nodes])
        
        logger.info(f"开始扩展子图，起始节点: {len(start_nodes)} 个，深度: {depth}")
        
        while queue:
            current_node, current_depth = queue.popleft()
            
            if current_depth >= depth:
                continue
            
            # 扩展邻居节点（出边）
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, current_depth + 1))
            
            # 对于有向图，也考虑入边（前驱节点）
            if self.graph.is_directed():
                for predecessor in self.graph.predecessors(current_node):
                    if predecessor not in visited_nodes:
                        visited_nodes.add(predecessor)
                        queue.append((predecessor, current_depth + 1))
        
        logger.info(f"子图扩展完成，共访问 {len(visited_nodes)} 个节点")
        subgraph = self.graph.subgraph(visited_nodes).copy()
        logger.info(f"子图包含 {subgraph.number_of_nodes()} 个节点和 {subgraph.number_of_edges()} 条边")
        return subgraph

    def fuzzy_search_nodes(self, query: str, max_edit_distance: int = 1) -> Set[str]:
        """
        模糊搜索节点（包含匹配 + 编辑距离，大小写不敏感）
        
        Args:
            query: 查询字符串
            max_edit_distance: 最大编辑距离阈值
        
        Returns:
            匹配的节点ID集合
        """
        if not self.graph or not query:
            return set()
        
        query_lower = query.lower().strip()
        if not query_lower:
            return set()
        
        matched_nodes = set()
        
        for name, node_ids in self.name_index.items():
            if query_lower in name:
                matched_nodes.update(node_ids)
            elif len(name) >= 3 and self._edit_distance_within(name, query_lower, max_edit_distance):
                matched_nodes.update(node_ids)
        
        logger.info(f"模糊搜索 '{query}' 找到 {len(matched_nodes)} 个匹配节点")
        return matched_nodes

    def _edit_distance_within(self, s1: str, s2: str, max_distance: int) -> bool:
        """
        判断两个字符串的编辑距离是否在阈值内（使用动态规划）
        """
        if abs(len(s1) - len(s2)) > max_distance:
            return False
        
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        return dp[m][n] <= max_distance

    def find_subgraph_by_fuzzy_name(self, query: str, depth: int = 2) -> nx.MultiDiGraph:
        """
        通过模糊名称查找子图（包含匹配 + 编辑距离）
        
        Args:
            query: 查询字符串
            depth: 搜索深度
        
        Returns:
            包含匹配节点的聚合子图
        """
        if not self.graph:
            return nx.MultiDiGraph()
        
        matched_nodes = self.fuzzy_search_nodes(query)
        
        if not matched_nodes:
            logger.warning(f"模糊搜索 '{query}' 未找到匹配节点")
            return nx.MultiDiGraph()
        
        logger.info(f"模糊匹配找到 {len(matched_nodes)} 个节点，开始聚合子图（深度={depth}）...")
        return self._expand_subgraph(matched_nodes, depth)

    def _get_edge_score(self, edge_data: Dict[str, Any], score_type: str) -> Optional[float]:
        """
        根据 score_type 获取边的评分值

        Args:
            edge_data: 边数据字典
            score_type: 评分类型，支持：
                - "accuracy_score": 准确率评分
                - "triple_support_score": 三元组支持度
                - "usefulness_score": 有用性评分
                - "min_accuracy_usefulness": accuracy_score 和 usefulness_score 的较小值
                - "min_accuracy_triple": accuracy_score 和 triple_support_score 的较小值
                - "min_usefulness_triple": usefulness_score 和 triple_support_score 的较小值
                - "min_all": 三个评分的最小值

        Returns:
            评分值，如果找不到返回 None
        """
        if score_type == "accuracy_score":
            return edge_data.get("accuracy_score")
        elif score_type == "triple_support_score":
            return edge_data.get("triple_support_score")
        elif score_type == "usefulness_score":
            return edge_data.get("usefulness_score")
        elif score_type == "min_accuracy_usefulness":
            acc = edge_data.get("accuracy_score")
            util = edge_data.get("usefulness_score")
            if acc is not None and util is not None:
                return min(acc, util)
            return acc or util
        elif score_type == "min_accuracy_triple":
            acc = edge_data.get("accuracy_score")
            triple = edge_data.get("triple_support_score")
            if acc is not None and triple is not None:
                return min(acc, triple)
            return acc or triple
        elif score_type == "min_usefulness_triple":
            util = edge_data.get("usefulness_score")
            triple = edge_data.get("triple_support_score")
            if util is not None and triple is not None:
                return min(util, triple)
            return util or triple
        elif score_type == "min_all":
            scores = []
            for key in ["accuracy_score", "triple_support_score", "usefulness_score"]:
                val = edge_data.get(key)
                if val is not None:
                    scores.append(val)
            return min(scores) if scores else None
        return None

    def convert_to_echarts_format(self, graph: Optional[nx.MultiDiGraph] = None, max_nodes: int = 1000,
                                   threshold_high: int = 2000, threshold_medium: int = 500,
                                   force_subgraph_mode: bool = False,
                                   score_threshold: Optional[float] = None,
                                   score_type: Optional[str] = None) -> Dict[str, Any]:
        """
        将NetworkX图转换为ECharts可视化格式

        Args:
            graph: 要转换的图，如果为None则使用self.graph
            max_nodes: 最大节点数限制
            threshold_high: 高阈值，总节点数大于此值时只展示community节点
            threshold_medium: 中阈值，总节点数在此值和high之间时展示community和keywords
            force_subgraph_mode: 强制使用子图模式，不进行节点过滤
            score_threshold: 评分阈值，低于此值的边会标记为dimmed（可选，不传则不过滤）
            score_type: 评分字段名，支持 "accuracy_score"、"triple_support_score"、"usefulness_score"，
                       以及组合字段如 "min_accuracy_usefulness"（可选，需要与score_threshold配合使用）
        
        Returns:
            ECharts格式的数据
        """
        if graph is None:
            graph = self.graph
        
        if not graph or graph.number_of_nodes() == 0:
            return {"nodes": [], "links": [], "categories": [], "stats": {}}
        
        # 总图信息（整个图的统计信息）
        total_nodes = graph.number_of_nodes()
        total_edges = graph.number_of_edges()
        
        nodes_dict = {}
        links = []
        categories_set = set()
        
        # 先建立community到members的映射（在过滤节点之前，以便保留完整的映射关系）
        community_to_members = {}  # community_id -> set of member_ids
        if graph.is_directed():
            for u, v, edge_data in graph.edges(data=True):
                if edge_data.get("relation") == "member_of":
                    # u是member, v是community
                    if v not in community_to_members:
                        community_to_members[v] = set()
                    community_to_members[v].add(u)
        
        # 根据总节点数决定展示规则
        # 总节点数 > threshold_high：只展示community节点
        # 总节点数在 threshold_medium-threshold_high 之间：展示community和keywords
        # 总节点数 ≤ threshold_medium：展示所有节点
        all_nodes = list(graph.nodes())
        is_subgraph = force_subgraph_mode or total_nodes <= threshold_medium  # 判断是否为子图模式
        
        if total_nodes > threshold_high and not is_subgraph:
            # 只展示community节点
            filtered_nodes = [node_id for node_id in all_nodes 
                        if graph.nodes[node_id].get("label") == "community"]
            if filtered_nodes:
                all_nodes = filtered_nodes
                logger.info(f"总节点数({total_nodes})大于{threshold_high}，只展示community节点: {len(all_nodes)} 个")
            else:
                logger.warning(f"总节点数({total_nodes})大于{threshold_high}，但未找到community节点，将展示所有节点")
                is_subgraph = True  # 回退到子图模式（显示所有节点）
        elif total_nodes > threshold_medium and total_nodes <= threshold_high and not is_subgraph:
            # 展示community和keywords
            filtered_nodes = [node_id for node_id in all_nodes 
                        if graph.nodes[node_id].get("label") in ["community", "keyword"]]
            if filtered_nodes:
                all_nodes = filtered_nodes
                logger.info(f"总节点数({total_nodes})在{threshold_medium}-{threshold_high}之间，展示community和keywords: {len(all_nodes)} 个")
            else:
                logger.warning(f"总节点数({total_nodes})在{threshold_medium}-{threshold_high}之间，但未找到community/keyword节点，将展示所有节点")
                is_subgraph = True  # 回退到子图模式（显示所有节点）
            
            # 先选择高度节点
            sampled_nodes = set(sorted_nodes[:max_nodes])
            
            # 确保所有边的端点都在采样中（避免孤立节点）
            edge_nodes = set()
            for u, v in graph.edges():
                edge_nodes.add(u)
                edge_nodes.add(v)
            
            # 如果采样后的节点不包含所有边的端点，补充缺失的节点
            missing_nodes = edge_nodes - sampled_nodes
            if missing_nodes:
                # 移除一些低度节点，为缺失的节点腾出空间
                low_degree_nodes = sorted(sampled_nodes, key=lambda n: node_degrees[n])
                nodes_to_remove = min(len(missing_nodes), len(low_degree_nodes))
                for node in low_degree_nodes[:nodes_to_remove]:
                    if node not in edge_nodes:  # 只移除不在边中的节点
                        sampled_nodes.remove(node)
                        if len(sampled_nodes) + len(missing_nodes) <= max_nodes:
                            break
                
                # 添加缺失的节点（如果还有空间）
                remaining_space = max_nodes - len(sampled_nodes)
                if remaining_space > 0:
                    sampled_nodes.update(list(missing_nodes)[:remaining_space])
            
            all_nodes = list(sampled_nodes)[:max_nodes]
        
        # 建立member到community的反向映射（使用之前建立的community_to_members）
        member_to_community = {}  # member_id -> community_id
        for comm_id, members in community_to_members.items():
            for member_id in members:
                member_to_community[member_id] = comm_id
        
        # 处理节点
        for node_id in all_nodes:
            node_data = graph.nodes[node_id]
            props = node_data.get("properties", {})
            label = node_data.get("label", "entity")
            name = props.get("name", str(node_id))
            
            # 处理名称（可能是列表）
            if isinstance(name, list):
                name = "\n".join(str(n) for n in name) if name else str(node_id)
            elif not isinstance(name, str):
                name = str(name)
            
            # 使用label作为分类（四层结构）
            category = label  # 使用label而不是schema_type
            categories_set.add(category)
            
            # 计算节点大小（基于度）
            degree = graph.degree(node_id)
            symbol_size = min(max(20 + degree * 2, 15), 60)
            
            # 确定节点属于哪个community（如果有）
            community_id = None
            if node_id in member_to_community:
                community_id = member_to_community[node_id]
            
            nodes_dict[node_id] = {
                "id": node_id,
                "name": name,  # 保留完整名称，不截断（前端可以处理显示）
                "category": category,
                "symbolSize": symbol_size,
                "value": degree,
                "properties": props,
                "label": label,
                "community_id": community_id  # 存储所属的community ID
            }
        
        # 处理边（只包含两个端点都在nodes_dict中的边）
        for u, v, edge_data in graph.edges(data=True):
            if u not in nodes_dict or v not in nodes_dict:
                continue
            
            relation = edge_data.get("relation", "related_to")
            link_data = {
                "source": u,
                "target": v,
                "name": relation,
                "value": 1
            }
            
            # 保留 chunk id 信息（可能是字符串或列表）
            if "chunk_id" in edge_data:
                chunk_id_value = edge_data["chunk_id"]
                if isinstance(chunk_id_value, list) and len(chunk_id_value) > 0:
                    link_data["chunk_id"] = chunk_id_value[0]  # 取第一个
                    link_data["chunk_ids"] = chunk_id_value  # 保留所有
                elif isinstance(chunk_id_value, str) and chunk_id_value.strip():
                    link_data["chunk_id"] = chunk_id_value.strip()
                    link_data["chunk_ids"] = [chunk_id_value.strip()]
                elif chunk_id_value:
                    link_data["chunk_id"] = str(chunk_id_value)
                    link_data["chunk_ids"] = [str(chunk_id_value)]
            
            # 保留 evidence 信息
            if "evidence" in edge_data:
                link_data["evidence"] = edge_data["evidence"]
            
            # 保留 source 信息
            if "source" in edge_data:
                link_data["source_text"] = edge_data["source"]
            
            # 保留 score 相关字段
            if "start_accuracy_score" in edge_data:
                link_data["start_accuracy_score"] = edge_data["start_accuracy_score"]
            if "end_accuracy_score" in edge_data:
                link_data["end_accuracy_score"] = edge_data["end_accuracy_score"]
            # 保留评分字段（数值），去掉解释字段
            if "triple_support_score" in edge_data:
                link_data["triple_support_score"] = edge_data["triple_support_score"]
            if "accuracy_score" in edge_data:
                link_data["accuracy_score"] = edge_data["accuracy_score"]
            if "usefulness_score" in edge_data:
                link_data["usefulness_score"] = edge_data["usefulness_score"]
            if "node_accuracy_score" in edge_data:
                link_data["node_accuracy_score"] = edge_data["node_accuracy_score"]
            if "start_accuracy_score" in edge_data:
                link_data["start_accuracy_score"] = edge_data["start_accuracy_score"]
            if "end_accuracy_score" in edge_data:
                link_data["end_accuracy_score"] = edge_data["end_accuracy_score"]

            links.append(link_data)

        # 根据评分阈值标记边（低分边标记为dimmed）
        if score_threshold is not None and score_type is not None:
            dimmed_count = 0
            for link in links:
                score_value = self._get_edge_score(link, score_type)
                if score_value is not None and score_value < score_threshold:
                    link["dimmed"] = True
                    link["status"] = "filtered"
                    dimmed_count += 1
                else:
                    link["dimmed"] = False
                    link["status"] = "active"
            logger.info(f"评分筛选: {dimmed_count}/{len(links)} 条边被标记为 dimmed (threshold={score_threshold}, type={score_type})")
        else:
            for link in links:
                link["dimmed"] = False
                link["status"] = "active"

        # 对于子图，过滤掉没有连接的孤立节点
        # 这些节点可能是BFS扩展时被包含，但在子图中没有实际连接
        if is_subgraph:
            connected_nodes = set()
            for link in links:
                connected_nodes.add(link["source"])
                connected_nodes.add(link["target"])
            
            # 找出在子图中没有连接的节点
            isolated_nodes = set(nodes_dict.keys()) - connected_nodes
            if isolated_nodes:
                logger.info(f"子图中发现 {len(isolated_nodes)} 个孤立节点，将被过滤")
                # 检查这些节点在子图（graph参数）中的度
                for node_id in list(isolated_nodes):
                    # 如果节点在子图中度为0（没有连接），说明它不应该被显示
                    subgraph_degree = graph.degree(node_id) if node_id in graph else 0
                    if subgraph_degree == 0:
                        # 在子图中没有连接的节点，移除
                        del nodes_dict[node_id]
                        logger.info(f"  移除孤立节点: {node_id}")
        
        # 定义四层结构的颜色（基于label）
        # Level 1: attribute, Level 2: entity, Level 3: keyword, Level 4: community
        base_colors = {
            "attribute": {"hue": 200, "saturation": 70, "lightness": 65},  # 浅蓝色
            "entity": {"hue": 220, "saturation": 70, "lightness": 60},    # 蓝色
            "keyword": {"hue": 240, "saturation": 60, "lightness": 70},   # 浅紫色
            "community": {"hue": 260, "saturation": 80, "lightness": 40}  # 深紫色（较深）
        }
        
        # 为每个community分配独特的颜色
        community_colors = {}  # community_id -> color (用于members)
        community_node_colors = {}  # community_id -> color (用于community节点，加深版本)
        community_hue_map = {}  # community_id -> base_hue
        
        # 先为所有community分配基础色系
        for comm_id in community_to_members.keys():
            if comm_id in nodes_dict:
                # 为每个community分配独特的色相
                comm_hue = 260 + (hash(comm_id) % 60)  # 260-320度之间（紫色系）
                comm_saturation = 70
                comm_lightness = 60  # 成员使用60%的亮度
                
                community_hue_map[comm_id] = comm_hue
                # 成员使用正常亮度
                community_colors[comm_id] = f"hsl({comm_hue}, {comm_saturation}%, {comm_lightness}%)"
                # community节点使用相同色相和饱和度，但降低亮度（加深）
                community_node_colors[comm_id] = f"hsl({comm_hue}, {comm_saturation}%, {comm_lightness - 25}%)"  # 降低25%亮度
        
        # 定义四层结构的形状（基于label）
        label_symbols = {
            "attribute": "circle",      # Level 1: 圆形
            "entity": "rect",           # Level 2: 矩形
            "keyword": "triangle",      # Level 3: 三角形
            "community": "diamond"       # Level 4: 菱形
        }
        
        # 创建分类列表（基于label四层，图例只显示形状，不显示颜色）
        categories = []
        for label_name in ["attribute", "entity", "keyword", "community"]:
            if label_name in categories_set:
                bc = base_colors.get(label_name, {"hue": 220, "saturation": 70, "lightness": 60})
                # 图例使用默认颜色（用于显示，实际节点颜色由节点数据决定）
                color = f"hsl({bc['hue']}, {bc['saturation']}%, {bc['lightness']}%)"
                categories.append({
                    "name": label_name,
                    "itemStyle": {"color": color},
                    "symbol": label_symbols.get(label_name, "circle")  # 图例也显示对应形状
                })
        
        # 为节点分配颜色和形状（考虑community成员关系）
        for node_id, node in nodes_dict.items():
            label = node.get("label", "entity")
            comm_id = node.get("community_id")
            
            # 设置形状
            if label in label_symbols:
                node["symbol"] = label_symbols[label]
            else:
                node["symbol"] = "circle"  # 默认圆形
            
            # 设置颜色
            if label == "community":
                # community节点使用与members相同的颜色，但加深
                if node_id in community_node_colors:
                    node["itemStyle"] = {"color": community_node_colors[node_id]}
                else:
                    # 默认community深色（如果没有members）
                    node["itemStyle"] = {"color": f"hsl(260, 70%, 35%)"}
            elif comm_id and comm_id in community_colors:
                # community的成员使用与community相同的颜色（相同色相和饱和度）
                node["itemStyle"] = {"color": community_colors[comm_id]}
                # 保持category为label，这样图例仍然显示四层结构
            else:
                # 普通节点使用基础颜色
                if label in base_colors:
                    bc = base_colors[label]
                    node["itemStyle"] = {"color": f"hsl({bc['hue']}, {bc['saturation']}%, {bc['lightness']}%)"}
                else:
                    # 默认颜色
                    node["itemStyle"] = {"color": f"hsl(220, 70%, 60%)"}
        
        nodes_list = list(nodes_dict.values())

        # 再次过滤links，确保source和target都在最终的nodes_list中
        node_ids_in_list = {node["id"] for node in nodes_list}
        filtered_links = [
            link for link in links
            if link["source"] in node_ids_in_list and link["target"] in node_ids_in_list
        ]

        # 为节点添加 dimmed 和 status 字段
        # 规则：如果节点的所有边都是 dimmed，则该节点也是 dimmed
        if score_threshold is not None and score_type is not None:
            node_to_edges = {node["id"]: [] for node in nodes_list}
            for link in filtered_links:
                src, tgt = link["source"], link["target"]
                if src in node_to_edges:
                    node_to_edges[src].append(link)
                if tgt in node_to_edges:
                    node_to_edges[tgt].append(link)

            for node in nodes_list:
                node_id = node["id"]
                connected_edges = node_to_edges.get(node_id, [])
                if connected_edges:
                    all_dimmed = all(edge.get("dimmed", False) for edge in connected_edges)
                    if all_dimmed:
                        node["dimmed"] = True
                        node["status"] = "filtered"
                    else:
                        node["dimmed"] = False
                        node["status"] = "active"
                else:
                    node["dimmed"] = False
                    node["status"] = "active"

        # 计算评分统计
        stats = {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "displayed_nodes": len(nodes_list),
            "displayed_edges": len(filtered_links)
        }
        if score_threshold is not None and score_type is not None:
            dimmed_edges = sum(1 for link in filtered_links if link.get("dimmed", False))
            stats["dimmed_edges"] = dimmed_edges
            stats["active_edges"] = len(filtered_links) - dimmed_edges
            stats["score_threshold"] = score_threshold
            stats["score_type"] = score_type

        return {
            "nodes": nodes_list,
            "links": filtered_links,
            "categories": categories,
            "stats": stats
        }
    
    def display_all_stats(self):
        """展示所有统计信息"""
        logger.info("\n" + "="*80)
        logger.info("图统计信息")
        logger.info("="*80)
        graph_stats = self.get_graph_stats()
        for key, value in graph_stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "="*80)
        logger.info("社区统计信息")
        logger.info("="*80)
        community_stats = self.get_community_stats()
        logger.info(f"  总社区数: {community_stats.get('total_communities', 0)}")
        if community_stats.get('total_communities', 0) > 0:
            logger.info(f"  平均每个社区的成员数: {community_stats.get('avg_members_per_community', 0):.2f}")
            logger.info(f"  最大成员数: {community_stats.get('max_members', 0)}")
            logger.info(f"  最小成员数: {community_stats.get('min_members', 0)}")
        
        logger.info("\n" + "="*80)
        logger.info("节点统计信息")
        logger.info("="*80)
        node_stats = self.get_node_stats()
        logger.info(f"  总节点数: {node_stats.get('total_nodes', 0)}")
        logger.info("  按标签分类:")
        for label, count in node_stats.get('nodes_by_label', {}).items():
            logger.info(f"    {label}: {count}")
        logger.info("  按层级分类:")
        for level, count in sorted(node_stats.get('nodes_by_level', {}).items()):
            logger.info(f"    Level {level}: {count}")
        
        logger.info("\n" + "="*80)
        logger.info("边统计信息")
        logger.info("="*80)
        edge_stats = self.get_edge_stats()
        logger.info(f"  总边数: {edge_stats.get('total_edges', 0)}")
        logger.info("  按关系类型分类:")
        for relation, count in sorted(edge_stats.get('edges_by_relation', {}).items(), 
                                      key=lambda x: x[1], reverse=True):
            logger.info(f"    {relation}: {count}")
        
        logger.info("\n" + "="*80)
        logger.info("图结构信息")
        logger.info("="*80)
        structure_info = self.get_structure_info()
        for key, value in structure_info.items():
            if isinstance(value, list):
                logger.info(f"  {key}:")
                for item in value[:10]:  # 只显示前10个
                    logger.info(f"    - {item}")
                if len(value) > 10:
                    logger.info(f"    ... 还有 {len(value) - 10} 个")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("="*80 + "\n")


if __name__ == "__main__":
# 获取graph的信息
    # graph_path = "/Dspace/pku-projects/dev-projects/lab-agents/graph_construction/output/paper_mini_highlevel_concise.json"
    graph_path = "/Dspace/pku-projects/dev-projects/lab-agents/graph_construction/output/paper_mini_highlevel_concise.json"
    
    # graph_path = "/Dspace/pku-projects/projects/lab-agents/graphrag/output/graphs/electrolytes_concise.json"
    

    # 创建图分析器
    analyzer = GraphAnalyzer(graph_path)

    # 展示所有统计信息
    analyzer.display_all_stats()

    # 示例：查找子图
    logger.info("\n" + "="*80)
    logger.info("子图查找示例")
    logger.info("="*80)
    subgraph = analyzer.find_subgraph_by_node_name(node_name="NCM811", depth=1)
    
    
    # subgraph = analyzer.find_subgraph(
    #     depth=2,
    #     target_keywords=["lithium", "battery"],
    #     relation_filters=None  # 不过滤关系类型
    # )
    print(subgraph.nodes(data=True))
    # [('entity_10', {'label': 'entity', 'properties': {'name': 'Electrolyte with Additive', 'schema_type': 'Electrolyte'}, 'level': 2}), ('attribute_9', {'label': 'attribute', 'properties': {'name': 'description: Cathode material'}, 'level': 1}), ('entity_177', {'label': 'entity', 'properties': {'name': 'Cathode Electrolyte Interface', 'schema_type': 'Property'}, 'level': 2}), ('entity_8', {'label': 'entity', 'properties': {'name': ['NCM811', 'LiNi0.8Co0.1Mn0.1O2'], 'schema_type': 'Cathode'}, 'level': 2}), ('entity_305', {'label': 'entity', 'properties': {'name': 'Li\\|NCM811 Cell', 'schema_type': 'TestCondition'}, 'level': 2}), ('community_6', {'label': 'community', 'properties': {'name': 'NCM811-Cathode-Performance', 'description': 'Focuses on NCM811 cathode performance, lithium metal anode stability, and interface characterization in lithium metal batteries.', 'members': ['Coulombic Efficiency', ['NCM811', 'LiNi0.8Co0.1Mn0.1O2'], 'LMA', 'NCM811 cathode', 'Li anode', 'Lithium symmetric cell', 'Galvanostatic Test', 'EIS test', 'Lithium Metal Battery', 'Cathode Electrolyte Interface', 'Lithium Metal Anode', 'Li|NCM811 Cell', 'Li|NCM811 cell with additive', 'Capacity Retention (with additive)', 'Average Coulombic Efficiency (with additive)', 'Li|NCM811 blank cell', 'Average Coulombic Efficiency (blank)', 'Li-Cu Cell', 'Coulombic Efficiency (with additive)', 'Specific Capacity', '1C Rate', 'Li foil', 'Coin Cell', 'Li\\|NCM811 Cell']}, 'level': 4}), ('keyword_13', {'label': 'keyword', 'properties': {'name': ['NCM811', 'LiNi0.8Co0.1Mn0.1O2']}, 'level': 3}), ('attribute_12', {'label': 'attribute', 'properties': {'name': 'loading: 2-5 mg/cm²'}, 'level': 1}), ('attribute_11', {'label': 'attribute', 'properties': {'name': 'loading: 3 mgcm-2'}, 'level': 1})]
    logger.info(f"子图包含 {subgraph.number_of_nodes()} 个节点和 {subgraph.number_of_edges()} 条边")
    
    
    
    
    
    
    
    
    
    
    