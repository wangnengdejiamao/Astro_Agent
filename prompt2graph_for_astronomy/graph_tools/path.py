from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union


NodeType = Union[str, Dict]
EdgeList = Iterable[Dict]


@dataclass
class GraphPathFinder:
    """
    使用 BFS 在三元组格式的边集合中查找最短路径的工具类。
    """

    def find_min_path(self, graph: EdgeList, start_node: NodeType, end_node: NodeType) -> List[Dict]:
        if not graph:
            return []

        start_key = self._extract_node_key(start_node)
        end_key = self._extract_node_key(end_node)

        if not start_key or not end_key or start_key == end_key:
            return []

        adjacency = self._build_adjacency(graph)
        return self._bfs_shortest_path(adjacency, start_key, end_key)

    @staticmethod
    def _extract_node_key(node: NodeType) -> Optional[str]:
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            properties = node.get("properties") or {}
            name = properties.get("name")
            if isinstance(name, list):
                return name[0] if name else None
            if name:
                return name
            node_id = node.get("id") or properties.get("id")
            if node_id:
                return node_id
        return None

    def _build_adjacency(self, graph: EdgeList) -> Dict[str, List[tuple]]:
        adjacency: Dict[str, List[tuple]] = {}
        for edge in graph:
            start_key = self._extract_node_key(edge.get("start_node"))
            end_key = self._extract_node_key(edge.get("end_node"))
            if not start_key or not end_key:
                continue
            adjacency.setdefault(start_key, []).append((end_key, edge))
        return adjacency

    @staticmethod
    def _bfs_shortest_path(adjacency: Dict[str, List[tuple]], start_key: str, end_key: str) -> List[Dict]:
        queue = deque([(start_key, [])])
        visited = {start_key}

        while queue:
            current, path = queue.popleft()
            for next_node, edge in adjacency.get(current, []):
                next_path = path + [edge]
                if next_node == end_key:
                    return next_path
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, next_path))
        return []


__all__ = ["GraphPathFinder", "NodeType", "EdgeList"]

