#!/usr/bin/env python3
"""
知识图谱合并模块
实现多张知识图谱的合并：
1. 对两张图谱的实体进行消歧（基于abbreviation和CID）
2. 合并相同的三元组（合并source和evidence字段）
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple, Optional

from scorers import pubchem_scorer
from utils.logger import logger

def normalize_name(name: str) -> str:
    """规范化名称（与pubchem_scorer.py保持一致）"""
    return pubchem_scorer.normalize_name(name)


class GraphMerger:
    """知识图谱合并器"""
    
    def __init__(self):
        """初始化图谱合并器"""
        pass
    
    def _merge_names(self, *names: Any) -> Any:
        """
        合并多个name字段（可能是字符串或列表），去重后返回
        
        Args:
            *names: 可变参数，每个name可能是字符串或列表
            
        Returns:
            合并后的name：如果只有一个唯一name返回字符串，否则返回列表
        """
        all_names = []
        for n in names:
            if isinstance(n, list):
                all_names.extend(n)
            else:
                if n:  # 非空字符串
                    all_names.append(n)
        
        # 基于规范化后的name去重
        unique_names = []
        seen_normalized = set()
        for n in all_names:
            norm = normalize_name(n)
            if norm not in seen_normalized:
                unique_names.append(n)
                seen_normalized.add(norm)
        
        # 如果只有一个唯一name，返回字符串；否则返回列表
        if len(unique_names) == 1:
            return unique_names[0]
        else:
            return unique_names
    
    def _compute_entity_key(self, name: Any) -> Tuple[str, ...]:
        """
        根据name计算entity_key
        
        Args:
            name: 可能是字符串或列表
            
        Returns:
            entity_key（规范化name的元组）
        """
        if isinstance(name, list):
            normalized_names = [normalize_name(n) for n in name if n]
            return tuple(sorted(set(normalized_names)))
        else:
            return (normalize_name(name),) if name else ()
    
    def _merge_entity_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> None:
        """
        合并两个实体的属性（props2合并到props1中）
        
        Args:
            props1: 目标属性字典（会被修改）
            props2: 源属性字典
        """
        for prop_key, prop_value in props2.items():
            if prop_key not in props1:
                props1[prop_key] = prop_value
            elif prop_key == "cid":
                # CID应该相同，如果不相同则保留props1的并警告
                if props1.get("cid") != prop_value:
                    logger.warning(f"实体的CID不同: {props1.get('cid')} vs {prop_value}")
    
    def _get_name_list(self, name: Any) -> List[str]:
        """
        将name（可能是字符串或列表）转换为列表形式
        
        Args:
            name: 可能是字符串或列表
            
        Returns:
            name列表
        """
        if isinstance(name, list):
            return name
        elif name:
            return [name]
        else:
            return []
    
    def _index_names(self, name: Any, index: Dict[str, List], entity_key: Tuple[str, ...]) -> None:
        """
        将name的所有变体索引到字典中
        
        Args:
            name: 可能是字符串或列表
            index: 索引字典（normalized_name -> [entity_keys]）
            entity_key: 要索引的entity_key
        """
        names_list = self._get_name_list(name)
        for n in names_list:
            if n:
                index[normalize_name(n)].append(entity_key)
    
    def _collect_list_values(self, *values: Any) -> List[Any]:
        """
        收集多个值（可能是字符串或列表）到一个列表中
        
        Args:
            *values: 可变参数，每个值可能是字符串、列表或其他类型
            
        Returns:
            收集后的列表
        """
        result = []
        for val in values:
            if isinstance(val, list):
                result.extend(val)
            else:
                if val:  # 非空值
                    result.append(val)
        return result
    
    def _normalize_list_field(self, values: List[Any]) -> Any:
        """
        规范化列表字段：去重后，如果只有一个值返回该值，否则返回列表
        
        Args:
            values: 值列表
            
        Returns:
            如果只有一个唯一值返回该值，否则返回去重后的列表
        """
        unique_values = list(set(values))
        if len(unique_values) == 1:
            return unique_values[0]
        else:
            return unique_values
    
    def _update_entity_key_in_dict(self, entities: Dict[Tuple[str, ...], Dict[str, Any]], 
                                   old_key: Tuple[str, ...], 
                                   new_key: Tuple[str, ...],
                                   entity_mappings: List[Dict[Tuple[str, ...], Tuple[str, ...]]] = None) -> Tuple[str, ...]:
        """
        更新字典中的entity_key，并更新相关的映射关系
        
        Args:
            entities: 实体字典
            old_key: 旧的entity_key
            new_key: 新的entity_key
            entity_mappings: 需要更新的映射字典列表（可选）
            
        Returns:
            最终使用的key（可能是new_key或old_key）
        """
        if new_key != old_key:
            entities[new_key] = entities.pop(old_key)
            # 更新所有映射字典中指向old_key的映射
            if entity_mappings:
                for mapping in entity_mappings:
                    for k, v in mapping.items():
                        if v == old_key:
                            mapping[k] = new_key
            return new_key
        return old_key
    
    def load_graph(self, graph_path: str) -> List[Dict[str, Any]]:
        """
        加载图谱JSON文件
        
        Args:
            graph_path: 图谱JSON文件路径
            
        Returns:
            图谱三元组列表
        """
        if not os.path.exists(graph_path):
            logger.error(f"图谱文件不存在: {graph_path}")
            return []
        
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            logger.info(f"成功加载图谱 {graph_path}，包含 {len(graph_data)} 个三元组")
            return graph_data
        except Exception as e:
            logger.error(f"读取图谱文件失败: {e}")
            return []
    
    def extract_entities(self, graph_data: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
        """
        从图谱中提取所有实体节点（只提取label为"entity"的节点）
        
        Args:
            graph_data: 图谱三元组列表
            
        Returns:
            字典，键为实体的唯一标识（基于name的规范化元组），值为实体信息
        """
        entities = {}
        
        for triple in graph_data:
            # 提取start_node和end_node（只提取label为"entity"的节点）
            for node_key in ["start_node", "end_node"]:
                node = triple.get(node_key, {})
                if node and node.get("label") == "entity":
                    properties = node.get("properties", {})
                    name = properties.get("name", "")
                    
                    if not name:
                        continue
                    
                    # 计算entity_key
                    entity_key = self._compute_entity_key(name)
                    if not entity_key:  # 如果name为空，跳过
                        continue
                    
                    # 如果实体不存在，添加它
                    if entity_key not in entities:
                        entities[entity_key] = {
                            "properties": properties.copy()
                        }
                    else:
                        # 如果实体已存在，合并properties（特别是name字段）
                        existing_props = entities[entity_key]["properties"]
                        existing_name = existing_props.get("name", "")
                        new_name = properties.get("name", "")
                        
                        # 合并name字段
                        if existing_name != new_name:
                            merged_name = self._merge_names(existing_name, new_name)
                            entities[entity_key]["properties"]["name"] = merged_name
                            
                            # 更新entity_key以反映新的name
                            new_key = self._compute_entity_key(merged_name)
                            entity_key = self._update_entity_key_in_dict(entities, entity_key, new_key)
                        
                        # 合并其他属性（保留已有的，添加新的）
                        self._merge_entity_properties(entities[entity_key]["properties"], properties)
        
        logger.info(f"从图谱中提取了 {len(entities)} 个唯一实体（仅entity类型）")
        return entities
    
    def find_entity_matches(self, entities1: Dict[Tuple[str, ...], Dict[str, Any]], 
                           entities2: Dict[Tuple[str, ...], Dict[str, Any]]) -> Dict[Tuple[str, ...], Tuple[str, ...]]:
        """
        找到两张图谱中匹配的实体（基于abbreviation和CID）
        
        Args:
            entities1: 第一张图谱的实体字典
            entities2: 第二张图谱的实体字典
            
        Returns:
            映射字典：entities2的key -> entities1的key（匹配的实体）
        """
        matches = {}  # entities2_key -> entities1_key
        
        # 构建entities1的索引：基于规范化name和CID
        name_index1 = defaultdict(list)  # normalized_name -> [entity_keys]
        cid_index1 = defaultdict(list)    # cid -> [entity_keys]
        
        for entity_key, entity_info in entities1.items():
            props = entity_info["properties"]
            name = props.get("name", "")
            
            # 索引name
            self._index_names(name, name_index1, entity_key)
            
            # 索引CID
            cid = props.get("cid")
            if cid is not None:
                cid_index1[cid].append(entity_key)
        
        # 查找entities2中与entities1匹配的实体
        for entity_key2, entity_info2 in entities2.items():
            props2 = entity_info2["properties"]
            name2 = props2.get("name", "")
            cid2 = props2.get("cid")
            
            matched_key1 = None
            
            # 方法1：基于abbreviation（name匹配）
            if name2:
                names_to_check = self._get_name_list(name2)
                
                for n in names_to_check:
                    if n:
                        normalized = normalize_name(n)
                        if normalized in name_index1:
                            # 找到匹配，使用第一个匹配的实体
                            matched_key1 = name_index1[normalized][0]
                            break
            
            # 方法2：基于CID（如果方法1没找到匹配）
            if not matched_key1 and cid2 is not None:
                if cid2 in cid_index1:
                    matched_key1 = cid_index1[cid2][0]
            
            if matched_key1:
                matches[entity_key2] = matched_key1
        
        logger.info(f"找到 {len(matches)} 对匹配的实体")
        return matches
    
    def merge_entities(self, entities1: Dict[Tuple[str, ...], Dict[str, Any]], 
                      entities2: Dict[Tuple[str, ...], Dict[str, Any]],
                      matches: Dict[Tuple[str, ...], Tuple[str, ...]]) -> Tuple[Dict[Tuple[str, ...], Dict[str, Any]], Dict[Tuple[str, ...], Tuple[str, ...]], Dict[Tuple[str, ...], Tuple[str, ...]]]:
        """
        合并两张图谱的实体
        
        Args:
            entities1: 第一张图谱的实体字典
            entities2: 第二张图谱的实体字典
            matches: 实体匹配映射（entities2_key -> entities1_key）
            
        Returns:
            合并后的实体字典，entities1的key映射（entities1的key -> 合并后的key），以及entities2的key映射（entities2的key -> 合并后的key）
        """
        merged_entities = entities1.copy()
        entity_mapping1 = {key: key for key in entities1.keys()}  # entities1的key -> 合并后的key（初始为恒等映射）
        entity_mapping2 = {}  # entities2的key -> 合并后的key
        
        # 处理匹配的实体：合并到entities1中
        for entity_key2, entity_key1 in matches.items():
            entity1 = merged_entities[entity_key1]
            entity2 = entities2[entity_key2]
            
            props1 = entity1["properties"]
            props2 = entity2["properties"]
            
            # 合并name字段（取并集）
            name1 = props1.get("name", "")
            name2 = props2.get("name", "")
            merged_name = self._merge_names(name1, name2)
            merged_entities[entity_key1]["properties"]["name"] = merged_name
            
            # 合并其他属性（保留entities1的，如果entities2有新的则添加）
            self._merge_entity_properties(merged_entities[entity_key1]["properties"], props2)
            
            # 更新entity_key以反映新的name
            new_key = self._compute_entity_key(merged_name)
            if new_key != entity_key1:
                # 更新entities1的key映射：entity_key1是entities1中的原始key，现在映射到new_key
                entity_mapping1[entity_key1] = new_key
            entity_key1 = self._update_entity_key_in_dict(
                merged_entities, entity_key1, new_key, [entity_mapping1, entity_mapping2]
            )
            
            entity_mapping2[entity_key2] = entity_key1
        
        # 处理entities2中未匹配的实体：直接添加到合并后的实体中
        for entity_key2, entity_info2 in entities2.items():
            if entity_key2 not in matches:
                # entity_key2已经是通过extract_entities计算好的正确key，无需重新计算
                if entity_key2 not in merged_entities:
                    merged_entities[entity_key2] = entity_info2
                    entity_mapping2[entity_key2] = entity_key2
                else:
                    # 如果key已存在，合并实体（可能是在处理匹配实体时更新了key导致的冲突）
                    entity1 = merged_entities[entity_key2]
                    props1 = entity1["properties"]
                    props2 = entity_info2["properties"]
                    name2 = props2.get("name", "")
                    
                    # 合并name
                    name1 = props1.get("name", "")
                    merged_name = self._merge_names(name1, name2)
                    merged_entities[entity_key2]["properties"]["name"] = merged_name
                    
                    # 如果合并后name变化，需要更新entity_key
                    new_key = self._compute_entity_key(merged_name)
                    if new_key != entity_key2:
                        # 更新entities1的key映射：如果entity_key2在entities1中（冲突情况）
                        if entity_key2 in entity_mapping1:
                            entity_mapping1[entity_key2] = new_key
                    final_key = self._update_entity_key_in_dict(
                        merged_entities, entity_key2, new_key, [entity_mapping1, entity_mapping2]
                    )
                    entity_mapping2[entity_key2] = final_key
        
        logger.info(f"实体合并完成: 图谱1有 {len(entities1)} 个实体，图谱2有 {len(entities2)} 个实体，合并后 {len(merged_entities)} 个实体")
        return merged_entities, entity_mapping1, entity_mapping2
    
    def get_entity_key_from_node(self, node: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
        """
        从节点中获取实体key（用于查找）
        
        Args:
            node: 节点字典
            
        Returns:
            实体key（规范化name的元组），如果不是entity则返回None）
        """
        if node.get("label") != "entity":
            return None
        
        properties = node.get("properties", {})
        name = properties.get("name", "")
        
        if not name:
            return None
        
        return self._compute_entity_key(name)
    
    def update_graph_entities(self, graph_data: List[Dict[str, Any]], 
                             merged_entities: Dict[Tuple[str, ...], Dict[str, Any]],
                             entity_mapping: Dict[Tuple[str, ...], Tuple[str, ...]],
                             graph_id: str = "") -> List[Dict[str, Any]]:
        """
        使用合并后的实体更新图谱
        
        Args:
            graph_data: 原始图谱数据
            merged_entities: 合并后的实体字典
            entity_mapping: 实体key映射（原始key -> 合并后的key）
            graph_id: 图谱标识（用于日志）
            
        Returns:
            更新后的图谱数据
        """
        updated_graph = []
        
        for triple in graph_data:
            new_triple = triple.copy()
            
            # 更新start_node和end_node（只更新label为"entity"的节点）
            for node_key in ["start_node", "end_node"]:
                node = triple.get(node_key, {})
                if node and node.get("label") == "entity":
                    entity_key = self.get_entity_key_from_node(node)
                    
                    if entity_key:
                        # 查找合并后的实体key
                        merged_key = entity_mapping.get(entity_key, entity_key)
                        
                        # 如果merged_key不在merged_entities中，尝试直接使用entity_key
                        if merged_key not in merged_entities:
                            merged_key = entity_key
                        
                        merged_entity = merged_entities.get(merged_key)
                        
                        if merged_entity:
                            new_triple[node_key] = {
                                "label": "entity",
                                "properties": merged_entity["properties"].copy()
                            }
            
            updated_graph.append(new_triple)
        
        logger.info(f"图谱 {graph_id} 更新完成: {len(updated_graph)} 个三元组")
        return updated_graph
    
    def normalize_triple_key(self, triple: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        规范化三元组的key（用于比较）
        
        Args:
            triple: 三元组字典
            
        Returns:
            (start_node_key, relation, end_node_key) 的元组
        """
        start_node = triple.get("start_node", {})
        end_node = triple.get("end_node", {})
        relation = triple.get("relation", "")
        
        # 获取节点的key（用于比较）
        def get_node_key(node):
            if node.get("label") == "entity":
                props = node.get("properties", {})
                name = props.get("name", "")
                return self._compute_entity_key(name) if name else ()
            else:
                # 对于attribute节点，使用name作为key
                props = node.get("properties", {})
                name = props.get("name", "")
                return name if name else ""
        
        start_key = get_node_key(start_node)
        end_key = get_node_key(end_node)
        
        return (str(start_key), relation, str(end_key))
    
    def merge_triples(self, graph1: List[Dict[str, Any]], 
                     graph2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        合并两张图谱的三元组（相同的三元组合并source和evidence）
        
        Args:
            graph1: 第一张图谱的三元组列表
            graph2: 第二张图谱的三元组列表
            
        Returns:
            合并后的三元组列表
        """
        # 构建索引：基于三元组key -> 三元组列表
        triple_index = defaultdict(list)
        
        # 索引graph1
        for triple in graph1:
            key = self.normalize_triple_key(triple)
            triple_index[key].append(("graph1", triple))
        
        # 索引graph2
        for triple in graph2:
            key = self.normalize_triple_key(triple)
            triple_index[key].append(("graph2", triple))
        
        merged_triples = []
        processed_keys = set()
        
        # 处理所有唯一的三元组key
        for key, triple_list in triple_index.items():
            if key in processed_keys:
                continue
            
            processed_keys.add(key)
            
            # 选择第一个三元组作为基础（保留结构）
            base_triple = triple_list[0][1].copy()
            
            # 收集所有三元组的source、evidence和chunk_id
            sources = []
            evidences = []
            chunk_ids = []
            
            for graph_id, triple in triple_list:
                if "source" in triple:
                    sources.extend(self._collect_list_values(triple["source"]))
                if "evidence" in triple:
                    evidences.extend(self._collect_list_values(triple["evidence"]))
                if "chunk_id" in triple:
                    chunk_ids.extend(self._collect_list_values(triple["chunk_id"]))
            
            # 去重并合并
            if sources:
                base_triple["source"] = self._normalize_list_field(sources)
            else:
                base_triple["source"] = base_triple.get("source", "")
            
            if evidences:
                base_triple["evidence"] = self._normalize_list_field(evidences)
            else:
                base_triple["evidence"] = base_triple.get("evidence", "")
            
            if chunk_ids:
                base_triple["chunk_id"] = self._normalize_list_field(chunk_ids)
            
            merged_triples.append(base_triple)
        
        logger.info(f"三元组合并完成: 图谱1有 {len(graph1)} 个三元组，图谱2有 {len(graph2)} 个三元组，合并后 {len(merged_triples)} 个三元组")
        return merged_triples
    
    def merge(self, graph_path1: str, graph_path2: str, output_path: str = None) -> str:
        """
        合并两张知识图谱
        
        Args:
            graph_path1: 第一张图谱JSON文件路径
            graph_path2: 第二张图谱JSON文件路径
            output_path: 输出合并后的图谱JSON文件路径（可选，默认自动生成）
            
        Returns:
            输出文件路径
        """
        logger.info("======== 开始图谱合并流程 ========")
        
        # 1. 加载两张图谱
        graph1 = self.load_graph(graph_path1)
        graph2 = self.load_graph(graph_path2)
        
        if not graph1 or not graph2:
            logger.error("图谱数据为空，无法继续")
            return ""
        
        # 2. 提取两张图谱的所有实体
        logger.info("--- 步骤1: 提取实体 ---")
        entities1 = self.extract_entities(graph1)
        entities2 = self.extract_entities(graph2)
        
        # 3. 找到匹配的实体
        logger.info("--- 步骤2: 查找匹配的实体 ---")
        matches = self.find_entity_matches(entities1, entities2)
        
        # 4. 合并实体
        logger.info("--- 步骤3: 合并实体 ---")
        merged_entities, entity_mapping1, entity_mapping2 = self.merge_entities(entities1, entities2, matches)
        
        # 5. 更新两张图谱中的实体引用
        logger.info("--- 步骤4: 更新图谱中的实体引用 ---")
        # 对于graph1，使用entity_mapping1（entities1的key -> 合并后的key）
        updated_graph1 = self.update_graph_entities(graph1, merged_entities, entity_mapping1, "图谱1")
        
        # 对于graph2，使用entity_mapping2（entities2的key -> 合并后的key）
        updated_graph2 = self.update_graph_entities(graph2, merged_entities, entity_mapping2, "图谱2")
        
        # 6. 合并三元组
        logger.info("--- 步骤5: 合并三元组 ---")
        merged_triples = self.merge_triples(updated_graph1, updated_graph2)
        
        # 7. 保存合并后的图谱
        if not output_path:
            base_name1 = os.path.splitext(os.path.basename(graph_path1))[0]
            base_name2 = os.path.splitext(os.path.basename(graph_path2))[0]
            output_dir = os.path.dirname(graph_path1) or "."
            output_path = os.path.join(output_dir, f"{base_name1}_merged_{base_name2}.json")
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_triples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合并后的图谱已保存到: {output_path}")
        logger.info("======== 图谱合并流程完成 ========")
        
        return output_path


def merge_graphs(graph_path1: str, graph_path2: str, output_path: str = None) -> str:
    """
    合并两张知识图谱的便捷函数
    
    Args:
        graph_path1: 第一张图谱JSON文件路径
        graph_path2: 第二张图谱JSON文件路径
        output_path: 输出合并后的图谱JSON文件路径（可选）
        
    Returns:
        输出文件路径
    """
    merger = GraphMerger()
    return merger.merge(graph_path1, graph_path2, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="知识图谱合并工具")
    parser.add_argument("graph_path1", help="第一张图谱JSON文件路径")
    parser.add_argument("graph_path2", help="第二张图谱JSON文件路径")
    parser.add_argument("output_path", nargs="?", default=None, help="输出合并后的图谱JSON文件路径（可选）")
    
    args = parser.parse_args()
    
    merge_graphs(args.graph_path1, args.graph_path2, args.output_path)
