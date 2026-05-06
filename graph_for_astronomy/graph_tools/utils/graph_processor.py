import networkx as nx
import json

from .logger import logger

# 日志输出计数器，用于限制 merge_name_properties 的日志输出
_merge_log_count = 0
_MAX_MERGE_LOGS = 1  # 最多输出 1 条日志


def extract_aliases(name_field):
    """将 name 字段统一为别名列表，方便节点判重
    "NCM811" → ["NCM811"]
    "LiNi0.8Co0.1Mn0.1O2" → ["LiNi0.8Co0.1Mn0.1O2"]
    ["NCM811", "LiNi0.8Co0.1Mn0.1O2"] → ["NCM811", "LiNi0.8Co0.1Mn0.1O2"]
    "A, B, C" → ["A", "B", "C"]  # 支持逗号分隔的字符串
    None → []
    "" → []
    """
    aliases = []
    if isinstance(name_field, list):
        for item in name_field:
            if item is None:
                continue
            alias = str(item).strip()
            if alias:
                aliases.append(alias)
    elif isinstance(name_field, str):
        alias = name_field.strip()
        if alias:
            # 如果字符串包含逗号，尝试分割（但要注意可能是单个名称中包含逗号）
            # 简单策略：如果包含 ", "（逗号+空格），则分割
            if ", " in alias:
                # 分割并清理
                parts = [part.strip() for part in alias.split(", ") if part.strip()]
                aliases.extend(parts)
            else:
                aliases.append(alias)
    elif name_field not in (None, ""):
        alias = str(name_field).strip()
        if alias:
            # 同样处理逗号分隔的情况
            if ", " in alias:
                parts = [part.strip() for part in alias.split(", ") if part.strip()]
                aliases.extend(parts)
            else:
                aliases.append(alias)
    return aliases


def build_canonical_key(label, aliases, raw_name):
    """构建节点的规范化 key，用于 node_mapping
    label, ("NCM811", "LiNi0.8Co0.1Mn0.1O2") → (label, ("NCM811", "LiNi0.8Co0.1Mn0.1O2"))
    label, "NCM811" → (label, "NCM811")
    label, None → (label, "")
    label, "" → (label, "")
    
    """
    if aliases:
        normalized = tuple(sorted(alias.lower() for alias in aliases))
        return label, normalized
    # 没有别名时，退化为原始 name 的 JSON 表示
    return label, json.dumps(raw_name, ensure_ascii=False, sort_keys=True)


def determine_level(label):
    if label == "attribute":
        return 1
    if label == "entity":
        return 2
    if label == "keyword":
        return 3
    if label == "community":
        return 4
    return 2


def merge_name_properties(existing_value, new_aliases):
    """在已有节点属性中追加新的别名
    已有节点 {"name": "NCM811"}，现在三元组里 name 是 ["NCM811", "LiNi0.8Co0.1Mn0.1O2"]，函数会返回 (["NCM811", "LiNi0.8Co0.1Mn0.1O2"], True)
    
    """
    if not new_aliases:
        return existing_value, False
    existing_aliases = extract_aliases(existing_value)
    changed = False
    for alias in new_aliases:
        if alias not in existing_aliases:
            existing_aliases.append(alias)
            changed = True
    if not changed:
        return existing_value, False
    # 减少日志输出：只输出前 N 条作为示例
    global _merge_log_count
    if _merge_log_count < _MAX_MERGE_LOGS:
        if len(existing_aliases) == 1:
            logger.info(f"![merge_name_properties] existing_aliases: {existing_aliases}")
        else:
            logger.info(f"![merge_name_properties] existing_aliases: {existing_aliases}")
        _merge_log_count += 1
    
    if len(existing_aliases) == 1:
        return existing_aliases[0], True
    return existing_aliases, True


def load_graph_from_json(input_path: str) -> nx.MultiDiGraph:
    """
    Load a knowledge graph from JSON format
    
    Expected JSON format:
    [
        {
            "start_node": {
                "label": "entity",
                "properties": {"name": "Entity Name", "description": "..."}
            },
            "relation": "relation_type",
            "end_node": {
                "label": "entity", 
                "properties": {"name": "Entity Name", "description": "..."}
            }
        }
    ]
    """
    graph = nx.MultiDiGraph()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        relationships = json.load(f)
    
    # Track nodes to avoid duplicates and assign consistent IDs
    node_mapping = {}  # canonical key -> node_id             # ("entity", ("lini...", "ncm811")) -> "entity_0"
    alias_mapping = {}  # (label, alias_lower) -> node_id     # ("entity", "ncm811") -> "entity_0"
    node_counter = 0

    def get_or_create_node(node_data):
        nonlocal node_counter
        node_label = node_data["label"]
        properties = node_data.get("properties", {})
        raw_name = properties.get("name", "")
        aliases = extract_aliases(raw_name)

        node_id = None
        for alias in aliases:
            alias_key = (node_label, alias.lower())
            if alias_key in alias_mapping:
                node_id = alias_mapping[alias_key]
                break

        canonical_key = build_canonical_key(node_label, aliases, raw_name)
        if node_id is None and canonical_key in node_mapping:
            node_id = node_mapping[canonical_key]

        if node_id is None:
            node_id = f"{node_label}_{node_counter}"
            node_counter += 1
            node_attrs = {
                "label": node_label,
                "properties": properties,
                "level": determine_level(node_label)
            }
            graph.add_node(node_id, **node_attrs)
            node_mapping[canonical_key] = node_id
        else:
            # 确保 node_mapping 能通过规范 key 命中当前 node
            node_mapping.setdefault(canonical_key, node_id)
            existing_properties = graph.nodes[node_id].setdefault("properties", {})
            current_name = existing_properties.get("name")
            updated_name, changed = merge_name_properties(current_name, aliases)
            if changed:
                existing_properties["name"] = updated_name
                graph.nodes[node_id]["properties"] = existing_properties

        for alias in aliases:
            alias_key = (node_label, alias.lower())
            alias_mapping[alias_key] = node_id

        return node_id
    
    for rel in relationships:
        # 跳过格式不正确的三元组
        if not isinstance(rel, dict):
            continue
        if "start_node" not in rel or "end_node" not in rel or "relation" not in rel:
            continue
        
        start_node_data = rel["start_node"]
        end_node_data = rel["end_node"]
        relation = rel["relation"]

        start_id = get_or_create_node(start_node_data)
        end_id = get_or_create_node(end_node_data)
        edge_attrs = {"relation": relation}
        
        # 支持 chunk_id 字段
        if "chunk_id" in rel:
            chunk_id_value = rel["chunk_id"]
            if isinstance(chunk_id_value, list) and len(chunk_id_value) > 0:
                edge_attrs["chunk_id"] = str(chunk_id_value[0])
            elif isinstance(chunk_id_value, str) and chunk_id_value.strip():
                edge_attrs["chunk_id"] = chunk_id_value.strip()
            elif chunk_id_value:
                edge_attrs["chunk_id"] = str(chunk_id_value)
        
        # 支持 evidence 和 source 字段
        if "evidence" in rel:
            edge_attrs["evidence"] = rel["evidence"]
        if "source" in rel:
            edge_attrs["source"] = rel["source"]
        
        # 支持 score 对象格式 {"node_accuracy_score": ..., "triple_support_score": ...}
        if "score" in rel and isinstance(rel["score"], dict):
            for score_key, score_value in rel["score"].items():
                edge_attrs[score_key] = score_value
        
        # 支持单独的 score 相关字段（扁平格式）
        if "start_accuracy_score" in rel:
            edge_attrs["start_accuracy_score"] = rel["start_accuracy_score"]
        if "end_accuracy_score" in rel:
            edge_attrs["end_accuracy_score"] = rel["end_accuracy_score"]
        if "triple_support_score" in rel:
            edge_attrs["triple_support_score"] = rel["triple_support_score"]
        if "triple_reason" in rel:
            edge_attrs["triple_reason"] = rel["triple_reason"]
        if "accuracy_score" in rel:
            edge_attrs["accuracy_score"] = rel["accuracy_score"]
        if "usefulness_score" in rel:
            edge_attrs["usefulness_score"] = rel["usefulness_score"]
        if "node_accuracy_score" in rel:
            edge_attrs["node_accuracy_score"] = rel["node_accuracy_score"]
        
        # 支持 reason 相关字段
        if "start_reason" in rel:
            edge_attrs["start_reason"] = rel["start_reason"]
        if "end_reason" in rel:
            edge_attrs["end_reason"] = rel["end_reason"]
        if "accuracy_reasoning" in rel:
            edge_attrs["accuracy_reasoning"] = rel["accuracy_reasoning"]
        if "usefulness_reasoning" in rel:
            edge_attrs["usefulness_reasoning"] = rel["usefulness_reasoning"]
        if "reasoning" in rel:
            edge_attrs["reasoning"] = rel["reasoning"]
        if "node_reason" in rel:
            edge_attrs["node_reason"] = rel["node_reason"]
        
        graph.add_edge(start_id, end_id, **edge_attrs)
    
    return graph


def save_graph_to_json(graph: nx.MultiDiGraph, output_path: str):
    """
    Save a knowledge graph to JSON format
    
    Output format:
    [
        {
            "start_node": {
                "label": "entity",
                "properties": {"name": "Entity Name", "description": "..."}
            },
            "relation": "relation_type", 
            "end_node": {
                "label": "entity",
                "properties": {"name": "Entity Name", "description": "..."}
            },
            "source": "...",
            "evidence": "...",
            "chunk_id": "...",
            "start_accuracy_score": 0.8,
            "end_accuracy_score": 0.9,
            "triple_support_score": 0.85,
            "accuracy_score": 0.8,
            "usefulness_score": 0.9
        }
    ]
    """
    output = []
    
    for u, v, data in graph.edges(data=True):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        
        relationship = {
            "start_node": {
                "label": u_data["label"],
                "properties": u_data["properties"],
            },
            "relation": data["relation"],
            "end_node": {
                "label": v_data["label"],
                "properties": v_data["properties"],
            },
        }
        
        # 保存 source 和 evidence
        if "source" in data:
            relationship["source"] = data["source"]
        if "evidence" in data:
            relationship["evidence"] = data["evidence"]
        if "chunk_id" in data:
            relationship["chunk_id"] = data["chunk_id"]
        
        # 保存 score 相关字段
        if "start_accuracy_score" in data:
            relationship["start_accuracy_score"] = data["start_accuracy_score"]
        if "end_accuracy_score" in data:
            relationship["end_accuracy_score"] = data["end_accuracy_score"]
        if "triple_support_score" in data:
            relationship["triple_support_score"] = data["triple_support_score"]
        if "accuracy_score" in data:
            relationship["accuracy_score"] = data["accuracy_score"]
        if "usefulness_score" in data:
            relationship["usefulness_score"] = data["usefulness_score"]
        
        output.append(relationship)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


# Legacy function for backward compatibility
def load_graph(input_path: str) -> nx.MultiDiGraph:
    """
    Load graph from either JSON or GraphML format (legacy support)
    """
    if input_path.endswith('.json'):
        return load_graph_from_json(input_path)
    elif input_path.endswith('.graphml'):
        return load_graph_from_graphml(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")


def load_graph_from_graphml(input_path: str) -> nx.MultiDiGraph:
    """
    Load graph from GraphML format (legacy function)
    """
    graph_data = nx.read_graphml(input_path)
    
    for node_id, data in graph_data.nodes(data=True):
        # Handle properties (d1)
        if "d1" in data:
            try:
                data["properties"] = json.loads(data["d1"])
                del data["d1"]
            except json.JSONDecodeError:
                logger.warning(f"Warning: Could not parse properties for node {node_id}")
                data["properties"] = {"name": str(data["d1"])}
                del data["d1"]
        
        # Handle level (d2)
        if "d2" in data:
            try:
                data["level"] = int(data["d2"])
                del data["d2"]
            except (ValueError, TypeError):
                data["level"] = 2  # Default level if conversion fails
                del data["d2"]
        
        # Handle label (d0)
        if "d0" in data:
            data["label"] = str(data["d0"])
            del data["d0"]
    
    for u, v, data in graph_data.edges(data=True):
        # Handle relation (d3)
        if "d3" in data:
            data["relation"] = str(data["d3"]).strip('"')
            del data["d3"]
    
    return graph_data


def save_graph(graph: nx.MultiDiGraph, output_path: str):
    """
    Save graph to either JSON or GraphML format based on file extension
    """
    if output_path.endswith('.json'):
        save_graph_to_json(graph, output_path)
    elif output_path.endswith('.graphml'):
        save_graph_to_graphml(graph, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")


def save_graph_to_graphml(graph: nx.MultiDiGraph, output_path: str):
    """
    Save graph to GraphML format (legacy function)
    """
    # Create a copy of the graph to avoid modifying the original
    graph_copy = graph.copy()
    
    for n, data in graph_copy.nodes(data=True):
        for k, v in list(data.items()):  
            if isinstance(v, dict):
                graph_copy.nodes[n][k] = json.dumps(v, ensure_ascii=False)

    for u, v, data in graph_copy.edges(data=True):
        for k, v in list(data.items()):
            if isinstance(v, dict):
                graph_copy.edges[u, v][k] = json.dumps(v, ensure_ascii=False)

    nx.write_graphml(graph_copy, output_path)
    
    
if __name__ == "__main__":
    graph = load_graph_from_json("/Dspace/pku-projects/dev-projects/lab-agents/graph_construction/output/paper_mini_lowlevel_new1_cut_abbreviate.json")
    print(type(graph))
    print(graph.nodes(data=True))
    # ('entity_6', {'label': 'entity', 'properties': {'name': ['NCM811', 'LiNi0.8Co0.1Mn0.1O2'], 'schema_type': 'Cathode'}, 'level': 2}), ('attribute_7', {'label': 'attribute', 'properties': {'name': 'description: Cathode material'}, 'level': 1}), ('entity_8', {'label': 'entity', 'properties': {'name': 'Electrolyte with Additive', 'schema_type': 'Electrolyte'}, 'level': 2})
    
    # [('entity_0', {'label': 'entity', 'properties': {'name': 'Cycling Stability', 'schema_type': 'PerformanceMetric'}, 'level': 2}), ('attribute_1', {'label': 'attribute', 'properties': {'name': 'value: 57.7'}, 'level': 1}), ('attribute_2', {'label': 'attribute', 'properties': {'name': 'units: % retention'}, 'level': 1}), ('entity_3', {'label': 'entity', 'properties': {'name': 'Coulombic Efficiency', 'schema_type': 'PerformanceMetric'}, 'level': 2}), ('attribute_4', {'label': 'attribute', 'properties': {'name': 'value: 95.6'}, 'level': 1}), ('attribute_5', {'label': 'attribute', 'properties': {'name': 'description: Performance metric for battery efficiency'}, 'level': 1})
    
    # print(graph.edges(data=True))
    
    # [('entity_0', 'attribute_1', {'relation': 'has_attribute', 'chunk id': 'jnfjijki'}), ('entity_0', 'attribute_2', {'relation': 'has_attribute', 'chunk id': 'jnfjijki'})
    
    # [('entity_0', 'attribute_1', {'relation': 'has_attribute', 'chunk id': 'jnfjijki'}), ('entity_0', 'attribute_2', {'relation': 'has_attribute', 'chunk id': 'jnfjijki'})
    # print(graph.nodes(data=True))
    # save_graph(graph, "/Dspace/pku-projects/dev-projects/lab-agents/graph_construction/output/electrolytes_new.graphml")