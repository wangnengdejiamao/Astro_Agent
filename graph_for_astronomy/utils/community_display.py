"""
工具函数：展示community及其成员
"""
import json
from typing import Dict, List, Set
import networkx as nx


def extract_communities_from_json(json_path: str) -> Dict[str, Dict]:
    """
    从JSON文件中提取所有community节点及其成员信息
    
    Returns:
        {
            "community_node_id": {
                "name": "Community Name",
                "description": "Community description",
                "members": ["member1", "member2", ...],
                "member_count": 5
            }
        }
    """
    communities = {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # 查找所有community节点
    for edge in graph_data:
        start_node = edge.get("start_node", {})
        end_node = edge.get("end_node", {})
        relation = edge.get("relation", "")
        
        # 检查是否是member_of关系（entity -> community）
        if relation == "member_of":
            comm_node = end_node
            member_node = start_node
            
            if comm_node.get("label") == "community":
                comm_props = comm_node.get("properties", {})
                comm_name = comm_props.get("name", "Unknown Community")
                
                # 使用community的members列表作为唯一标识
                if comm_name not in communities:
                    communities[comm_name] = {
                        "name": comm_name,
                        "description": comm_props.get("description", ""),
                        "members": [],
                        "member_nodes": []  # 存储完整的节点信息
                    }
                
                # 添加成员
                member_name = member_node.get("properties", {}).get("name", "")
                if member_name:
                    communities[comm_name]["members"].append(member_name)
                    communities[comm_name]["member_nodes"].append({
                        "name": member_name,
                        "label": member_node.get("label", ""),
                        "properties": member_node.get("properties", {})
                    })
    
    # 设置成员数量
    for comm in communities.values():
        comm["member_count"] = len(comm["members"])
    
    return communities


def get_community_members_from_graph(graph: nx.MultiDiGraph, community_name: str = None) -> Dict[str, List]:
    """
    从NetworkX图中提取community及其成员
    
    Args:
        graph: NetworkX MultiDiGraph对象
        community_name: 可选，指定community名称，如果为None则返回所有communities
    
    Returns:
        {
            "community_name": {
                "node_id": "comm_4_0",
                "name": "Community Name",
                "description": "...",
                "members": ["member1", "member2", ...],
                "member_nodes": [...]
            }
        }
    """
    communities = {}
    
    # 遍历所有节点，找到community节点
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("label") == "community":
            props = node_data.get("properties", {})
            comm_name = props.get("name", "Unknown")
            
            # 如果指定了community_name，只返回匹配的
            if community_name and comm_name != community_name:
                continue
            
            # 获取所有通过member_of边连接的节点
            members = []
            member_nodes = []
            
            # 方法1: 从properties中直接获取members列表
            member_names = props.get("members", [])
            
            # 方法2: 遍历图边，找到所有指向此community的member_of边
            for u, v, edge_data in graph.edges(node_id, data=True):
                # 注意：这里是反向查找，因为member_of是 member -> community
                pass
            
            # 正向查找：找到所有指向此community的节点
            for u, v, edge_data in graph.in_edges(node_id, data=True):
                if edge_data.get("relation") == "member_of":
                    member_node_data = graph.nodes[u]
                    member_name = member_node_data.get("properties", {}).get("name", "")
                    if member_name:
                        members.append(member_name)
                        member_nodes.append({
                            "node_id": u,
                            "name": member_name,
                            "label": member_node_data.get("label", ""),
                            "properties": member_node_data.get("properties", {})
                        })
            
            # 如果通过边找不到成员，使用properties中的members列表
            if not members and member_names:
                members = member_names
            
            communities[comm_name] = {
                "node_id": node_id,
                "name": comm_name,
                "description": props.get("description", ""),
                "members": members,
                "member_nodes": member_nodes,
                "member_count": len(members) if members else len(member_names)
            }
    
    return communities


def display_community_summary(communities: Dict[str, Dict]):
    """
    以易读格式展示community摘要
    """
    print(f"\n{'='*60}")
    print(f"找到 {len(communities)} 个社区")
    print(f"{'='*60}\n")
    
    for idx, (comm_name, comm_info) in enumerate(communities.items(), 1):
        print(f"[{idx}] {comm_name}")
        print(f"    描述: {comm_info.get('description', '无描述')[:100]}...")
        print(f"    成员数量: {comm_info.get('member_count', 0)}")
        print(f"    成员列表: {', '.join(comm_info.get('members', [])[:10])}")
        if comm_info.get('member_count', 0) > 10:
            print(f"    ... 还有 {comm_info.get('member_count', 0) - 10} 个成员")
        print()


def display_community_detail(communities: Dict[str, Dict], community_name: str):
    """
    详细展示指定community的完整信息
    """
    if community_name not in communities:
        print(f"未找到社区: {community_name}")
        return
    
    comm = communities[community_name]
    print(f"\n{'='*60}")
    print(f"社区详情: {comm['name']}")
    print(f"{'='*60}")
    print(f"节点ID: {comm.get('node_id', 'N/A')}")
    print(f"描述: {comm.get('description', '无描述')}")
    print(f"成员数量: {comm.get('member_count', 0)}")
    print(f"\n成员列表:")
    print(f"{'-'*60}")
    
    for idx, member in enumerate(comm.get('members', []), 1):
        print(f"  {idx:3d}. {member}")
    
    print(f"{'='*60}\n")


def visualize_community_structure(communities: Dict[str, Dict], max_display: int = 5):
    """
    以树状结构可视化community
    """
    print(f"\n{'='*60}")
    print(f"社区结构概览 (显示前 {max_display} 个)")
    print(f"{'='*60}\n")
    
    for idx, (comm_name, comm_info) in enumerate(list(communities.items())[:max_display], 1):
        print(f"📦 {comm_name}")
        print(f"   └─ 描述: {comm_info.get('description', '无')[:80]}")
        print(f"   └─ 成员 ({comm_info.get('member_count', 0)} 个):")
        
        members = comm_info.get('members', [])
        for i, member in enumerate(members[:10], 1):
            print(f"      {'├─' if i < len(members) and i < 10 else '└─'} {member}")
        
        if len(members) > 10:
            print(f"      └─ ... 还有 {len(members) - 10} 个成员")
        print()


# 示例使用
if __name__ == "__main__":
    # 示例1: 从JSON文件提取
    json_path = "output/graphs/paper_mini_new.json"
    communities = extract_communities_from_json(json_path)
    
    # 显示摘要
    display_community_summary(communities)
    
    # 显示第一个社区的详细信息
    if communities:
        first_comm_name = list(communities.keys())[0]
        display_community_detail(communities, first_comm_name)
        
        # 可视化结构
        visualize_community_structure(communities, max_display=3)
    
    # 示例2: 从NetworkX图提取
    # from utils.graph_processor import load_graph_from_json
    # graph = load_graph_from_json(json_path)
    # communities = get_community_members_from_graph(graph)
    # display_community_summary(communities)

