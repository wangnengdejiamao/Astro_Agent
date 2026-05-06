#!/usr/bin/env python3
"""
实体消歧（去重）模块
对已创建的知识图谱进行实体消歧：
1. 使用实体的name查询本地PubChem数据库获取CID
2. 添加CID字段到properties里
3. 将所有相同CID的实体合并
4. 输出消歧后的图谱
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple

from utils.logger import logger

# 导入 normalize_name 函数（从 pubchem_scorer 模块）
import re
try:
    from scorers.pubchem_scorer import PubChemClient
    from scorers import pubchem_scorer
except Exception as exc:
    PubChemClient = None  # type: ignore
    pubchem_scorer = None  # type: ignore
    logger.warning("PubChem scorer unavailable; CID-based deduplication will be skipped: %s", exc)

def normalize_name(name: str) -> str:
    """规范化名称（与pubchem_scorer.py保持一致）"""
    if pubchem_scorer is not None:
        return pubchem_scorer.normalize_name(name)
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


class EntityDeduplicator:
    """实体消歧器"""
    
    def __init__(self, db_path: str = "pubchem_names_full.db", init_pubchem: bool = True):
        """
        初始化实体消歧器
        
        Args:
            db_path: PubChem本地数据库文件路径
            init_pubchem: 是否初始化PubChem客户端（如果只需要abbreviation消歧，可以设为False）
        """
        self.pubchem_client = None
        if init_pubchem:
            if PubChemClient is None:
                logger.warning("PubChem客户端不可用，跳过CID初始化")
                return
            try:
                self.pubchem_client = PubChemClient(db_path=db_path)
                logger.info(f"已初始化 PubChem 本地数据库客户端: {db_path}")
            except Exception as e:
                logger.error(f"无法初始化 PubChem 本地数据库客户端: {e}")
                raise
    
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
            logger.info(f"成功加载图谱，包含 {len(graph_data)} 个三元组")
            return graph_data
        except Exception as e:
            logger.error(f"读取图谱文件失败: {e}")
            return []
    
    def extract_entities(self, graph_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        从图谱中提取所有实体节点（只提取label为"entity"的节点，不包括"attribute"）
        
        Args:
            graph_data: 图谱三元组列表
            
        Returns:
            字典，键为实体name，值为实体信息（只包含properties）
        """
        entities = {}
        
        for triple in graph_data:
            # 提取start_node和end_node（只提取label为"entity"的节点）
            for node_key in ["start_node", "end_node"]:
                node = triple.get(node_key, {})
                if node and node.get("label") == "entity":
                    properties = node.get("properties", {})
                    name = properties.get("name", "")
                    if name and name not in entities:
                        entities[name] = {
                            "properties": properties.copy()
                        }
        
        logger.info(f"从图谱中提取了 {len(entities)} 个唯一实体（仅entity类型）")
        return entities
    
    def query_cids_for_entities(self, entities: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        为所有实体查询CID并添加到properties中
        如果实体已有CID字段（从图谱构建时添加），则直接使用，不再查询
        
        Args:
            entities: 实体字典
            
        Returns:
            更新后的实体字典（包含CID信息）
        """
        logger.info("开始为实体查询CID（优先使用图谱中已有的CID）...")
        
        updated_entities = {}
        query_stats = {
            "total": len(entities),
            "found": 0,
            "not_found": 0,
            "from_graph": 0,  # 从图谱中已有的CID
            "from_query": 0   # 通过查询获得的CID
        }
        
        for entity_key, entity_info in entities.items():
            name = entity_info["properties"].get("name", "")
            if not name:
                continue
            
            # 优先检查图谱中是否已有CID
            existing_cid = entity_info["properties"].get("cid")
            if existing_cid is not None:
                # 图谱中已有CID，直接使用
                updated_entity = {
                    "properties": entity_info["properties"].copy()
                }
                updated_entity["cid"] = existing_cid
                updated_entities[entity_key] = updated_entity
                query_stats["found"] += 1
                query_stats["from_graph"] += 1
                continue
            
            # 图谱中没有CID，需要查询（需要PubChem客户端）
            if not self.pubchem_client:
                logger.warning("图谱中无CID且未初始化PubChem客户端，跳过CID查询")
                query_stats["not_found"] += 1
                continue
            
            # 处理name可能是list的情况（abbreviation合并后）
            if isinstance(name, list):
                # 如果name是list，使用第一个名称来查询CID
                # 如果第一个找不到，尝试其他名称
                query_name = None
                result = None
                for n in name:
                    if n and isinstance(n, str) and n.strip():
                        query_name = n
                        result = self.pubchem_client.lookup(n)
                        if result.get("match", False):
                            break
                
                if not query_name:
                    query_stats["not_found"] += 1
                    continue
            else:
                # name是字符串，直接查询
                query_name = name
                result = self.pubchem_client.lookup(name)
            
            if result.get("match", False):
                cid = result.get("cid")
                updated_entity = {
                    "properties": entity_info["properties"].copy()
                }
                updated_entity["properties"]["cid"] = cid
                updated_entity["cid"] = cid
                updated_entities[entity_key] = updated_entity
                query_stats["found"] += 1
                query_stats["from_query"] += 1
            else:
                query_stats["not_found"] += 1
        
        logger.info(
            f"CID查询完成: 总计找到 {query_stats['found']} 个 "
            f"(图谱中已有: {query_stats['from_graph']}, 新查询: {query_stats['from_query']}), "
            f"未找到 {query_stats['not_found']} 个"
        )
        return updated_entities
    
    def group_entities_by_cid(self, entities: Dict[str, Dict[str, Any]]) -> Dict[int, List[str]]:
        """
        按CID分组实体
        
        Args:
            entities: 包含CID信息的实体字典
            
        Returns:
            字典，键为CID，值为该CID对应的实体key列表
        """
        cid_groups = defaultdict(list)
        no_cid_entities = []
        
        for entity_key, entity_info in entities.items():
            cid = entity_info.get("cid")
            if cid is not None:
                cid_groups[cid].append(entity_key)
            else:
                no_cid_entities.append(entity_key)
        
        logger.info(f"按CID分组完成: {len(cid_groups)} 个CID组，{len(no_cid_entities)} 个无CID实体")
        return dict(cid_groups)
    
    def merge_entities(self, entities: Dict[str, Dict[str, Any]], 
                      cid_groups: Dict[int, List[str]]) -> Dict[str, Dict[str, Any]]:
        """
        合并相同CID的实体
        
        Args:
            entities: 实体字典
            cid_groups: CID分组字典
            
        Returns:
            合并后的实体字典，以及实体key映射（旧key -> 新key）
        """
        merged_entities = {}
        entity_mapping = {}  # 旧key -> 新key的映射
        
        # 处理有CID的实体组
        for cid, entity_keys in cid_groups.items():
            if len(entity_keys) == 1:
                # 只有一个实体，不需要合并
                entity_key = entity_keys[0]
                merged_entities[entity_key] = entities[entity_key]
                entity_mapping[entity_key] = entity_key
            else:
                # 多个实体需要合并
                # 选择第一个实体作为主实体
                main_key = entity_keys[0]
                main_entity = {
                    "properties": entities[main_key]["properties"].copy()
                }
                
                # 收集所有实体的name（处理name可能是list的情况）
                all_names = []
                
                for entity_key in entity_keys:
                    name = entities[entity_key]["properties"].get("name", "")
                    if name:
                        if isinstance(name, list):
                            all_names.extend(name)
                        else:
                            all_names.append(name)
                
                # 去重name列表
                unique_names = []
                seen_names = set()
                for name in all_names:
                    normalized = normalize_name(name)
                    if normalized not in seen_names:
                        unique_names.append(name)
                        seen_names.add(normalized)
                
                # 更新主实体的properties
                # 如果只有一个name，保持原样；如果有多个，将name改为列表
                if len(unique_names) == 1:
                    main_entity["properties"]["name"] = unique_names[0]
                else:
                    main_entity["properties"]["name"] = unique_names
                    main_entity["properties"]["original_names"] = all_names
                
                # 合并其他属性（如果有的话）
                for entity_key in entity_keys[1:]:
                    entity = entities[entity_key]
                    for prop_key, prop_value in entity["properties"].items():
                        if prop_key in ["name", "original_names"]:
                            # name 已经在上面处理过了，跳过
                            continue
                        
                        if prop_key not in main_entity["properties"]:
                            main_entity["properties"][prop_key] = prop_value
                        else:
                            # 统一处理属性值：将字符串和列表都转换为列表，合并后去重
                            main_value = main_entity["properties"][prop_key]
                            
                            # 将主值转换为列表
                            if isinstance(main_value, list):
                                main_list = main_value
                            else:
                                main_list = [main_value] if main_value is not None else []
                            
                            # 将新值转换为列表
                            if isinstance(prop_value, list):
                                new_list = prop_value
                            else:
                                new_list = [prop_value] if prop_value is not None else []
                            
                            # 合并并去重
                            merged_list = main_list + new_list
                            # 去重：保持顺序，使用 set 检查是否已存在
                            seen = set()
                            unique_merged = []
                            for item in merged_list:
                                # 使用 repr 或 str 作为去重键，处理不可哈希类型
                                item_key = repr(item) if not isinstance(item, (str, int, float, bool, type(None))) else item
                                if item_key not in seen:
                                    seen.add(item_key)
                                    unique_merged.append(item)
                            
                            # 如果只有一个值，保持为原始类型；否则保持为列表
                            if len(unique_merged) == 1:
                                main_entity["properties"][prop_key] = unique_merged[0]
                            else:
                                main_entity["properties"][prop_key] = unique_merged
                
                # 使用主key作为合并后的key
                merged_entities[main_key] = main_entity
                
                # 建立映射：所有被合并的实体都映射到主key
                for entity_key in entity_keys:
                    entity_mapping[entity_key] = main_key
                
                logger.debug(f"合并CID {cid} 的 {len(entity_keys)} 个实体: {unique_names}")
        
        logger.info(f"实体合并完成: 原始 {len(entities)} 个，合并后 {len(merged_entities)} 个")
        return merged_entities, entity_mapping
    
    def update_graph_with_merged_entities(self, graph_data: List[Dict[str, Any]], 
                                          merged_entities: Dict[str, Dict[str, Any]],
                                          entity_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        使用合并后的实体更新图谱
        
        Args:
            graph_data: 原始图谱数据
            merged_entities: 合并后的实体字典
            entity_mapping: 实体key映射（原始name -> 合并后的name）
            
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
                    name = node.get("properties", {}).get("name", "")
                    # 处理name可能是list的情况（从之前的合并中）
                    if isinstance(name, list):
                        # 如果name是list，使用第一个name进行查找
                        name = name[0] if name else ""
                    
                    if name:
                        # entity_key现在是name，查找合并后的key
                        merged_key = entity_mapping.get(name, name)
                        merged_entity = merged_entities.get(merged_key)
                        
                        if merged_entity:
                            new_triple[node_key] = {
                                "label": "entity",
                                "properties": merged_entity["properties"].copy()
                            }
            
            updated_graph.append(new_triple)
        
        logger.info(f"图谱更新完成: {len(updated_graph)} 个三元组")
        return updated_graph
    
    def extract_abbreviation_relations(self, graph_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从图谱中提取所有abbreviation关系（has_attribute关系，end_node的name以"abbreviation: "开头）
        
        Args:
            graph_data: 图谱三元组列表
            
        Returns:
            abbreviation关系的三元组列表（保留完整的三元组信息作为证据）
        """
        abbreviation_relations = []
        
        for triple in graph_data:
            if triple.get("relation") == "has_attribute":
                end_node = triple.get("end_node", {})
                if end_node and end_node.get("label") == "attribute":
                    attr_name = end_node.get("properties", {}).get("name", "")
                    if attr_name and attr_name.startswith("abbreviation: "):
                        abbreviation_relations.append(triple)
        
        logger.info(f"从图谱中提取了 {len(abbreviation_relations)} 个abbreviation关系")
        return abbreviation_relations
    
    def build_abbreviation_groups(self, abbreviation_relations: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        根据abbreviation关系构建同义实体组
        
        Args:
            abbreviation_relations: abbreviation关系的三元组列表
            
        Returns:
            字典，键为主实体name，值为该组内所有实体name的列表（包括主实体）
        """
        # 构建双向的同义关系图
        synonym_map = defaultdict(set)
        
        for triple in abbreviation_relations:
            start_node = triple.get("start_node", {})
            end_node = triple.get("end_node", {})
            
            if start_node and end_node:
                entity_name = start_node.get("properties", {}).get("name", "")
                attr_name = end_node.get("properties", {}).get("name", "")
                
                if entity_name and attr_name.startswith("abbreviation: "):
                    # 提取abbreviation值（去掉"abbreviation: "前缀）
                    abbrev_value = attr_name[len("abbreviation: "):].strip()
                    
                    # 建立双向关系：entity_name <-> abbrev_value
                    synonym_map[entity_name].add(abbrev_value)
                    synonym_map[abbrev_value].add(entity_name)
        
        # 使用并查集算法合并所有连通的实体
        # 首先找到所有相关的实体
        all_entities = set()
        for entity, synonyms in synonym_map.items():
            all_entities.add(entity)
            all_entities.update(synonyms)
        
        # 并查集：找到每个实体的根
        parent = {entity: entity for entity in all_entities}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                # 选择字典序较小的作为根（主实体）
                if root_x < root_y:
                    parent[root_y] = root_x
                else:
                    parent[root_x] = root_y
        
        # 合并所有同义实体
        for entity, synonyms in synonym_map.items():
            for synonym in synonyms:
                union(entity, synonym)
        
        # 按根分组
        groups = defaultdict(list)
        for entity in all_entities:
            root = find(entity)
            groups[root].append(entity)
        
        # 转换为字典格式，键为主实体（根），值为所有同义实体列表
        abbreviation_groups = {}
        for root, entities in groups.items():
            if len(entities) > 1:  # 只保留有多个实体的组
                # 去重并排序
                unique_entities = sorted(list(set(entities)))
                abbreviation_groups[root] = unique_entities
        
        logger.info(f"构建了 {len(abbreviation_groups)} 个abbreviation同义实体组")
        return abbreviation_groups
    
    def merge_entities_by_abbreviation(self, entities: Dict[str, Dict[str, Any]], 
                                      abbreviation_groups: Dict[str, List[str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """
        根据abbreviation关系合并同义实体
        
        Args:
            entities: 实体字典
            abbreviation_groups: abbreviation分组字典
            
        Returns:
            合并后的实体字典，以及实体key映射（旧key -> 新key）
        """
        merged_entities = {}
        entity_mapping = {}  # 旧key -> 新key的映射
        
        # 处理有abbreviation关系的实体组
        for main_entity, synonym_list in abbreviation_groups.items():
            # 找到主实体（在entities中存在的第一个）
            main_key = None
            for entity_name in synonym_list:
                if entity_name in entities:
                    main_key = entity_name
                    break
            
            if not main_key:
                # 如果所有同义实体都不在entities中，跳过
                logger.warning(f"abbreviation组 {main_entity} 的 {len(synonym_list)} 个实体都不在entities中，跳过")
                continue
            
            # 创建合并后的实体
            main_entity_data = entities[main_key].copy()
            merged_entity = {
                "properties": main_entity_data["properties"].copy()
            }
            
            for entity_name in synonym_list:
                if entity_name in entities:
                    # 合并其他属性
                    entity_data = entities[entity_name]
                    for prop_key, prop_value in entity_data["properties"].items():
                        if prop_key not in merged_entity["properties"]:
                            merged_entity["properties"][prop_key] = prop_value
                        elif prop_key == "cid" and merged_entity["properties"].get("cid") != prop_value:
                            # 如果CID不同，保留第一个（或可以合并）
                            pass
            
            # 合并后的name为全部同义实体列表（已去重）
            merged_entity["properties"]["name"] = synonym_list
            
            # 使用主key作为合并后的key
            merged_entities[main_key] = merged_entity
            
            # 建立映射：所有被合并的实体都映射到主key
            for entity_name in synonym_list:
                if entity_name in entities:
                    entity_mapping[entity_name] = main_key
            
            # logger.debug(f"合并abbreviation组 {main_entity} 的 {len(synonym_list)} 个实体: {unique_names}")
        
        # 处理无abbreviation关系的实体（保持原样）
        for entity_key, entity_info in entities.items():
            if entity_key not in entity_mapping:
                merged_entities[entity_key] = entity_info
                entity_mapping[entity_key] = entity_key
        
        logger.info(f"基于abbreviation的实体合并完成: 原始 {len(entities)} 个，合并后 {len(merged_entities)} 个")
        return merged_entities, entity_mapping
    
    def remove_abbreviation_triples(self, graph_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从图谱中删除abbreviation相关的三元组（has_attribute关系，end_node的name以"abbreviation: "开头）
        
        Args:
            graph_data: 图谱三元组列表
            
        Returns:
            删除abbreviation三元组后的图谱数据
        """
        filtered_graph = []
        removed_count = 0
        
        for triple in graph_data:
            if triple.get("relation") == "has_attribute":
                end_node = triple.get("end_node", {})
                if end_node and end_node.get("label") == "attribute":
                    attr_name = end_node.get("properties", {}).get("name", "")
                    if attr_name and attr_name.startswith("abbreviation: "):
                        removed_count += 1
                        continue
            
            filtered_graph.append(triple)
        
        logger.info(f"删除了 {removed_count} 个abbreviation相关的三元组")
        return filtered_graph
    
    def save_intermediate_result(self, entities_with_cid: Dict[str, Dict[str, Any]], 
                                cid_groups: Dict[int, List[str]], 
                                abbreviation_relations: List[Dict[str, Any]] = None,
                                abbreviation_groups: Dict[str, List[str]] = None,
                                output_path: str = None):
        """
        保存中间结果（entities_with_cid已经只包含能查询到CID的实体）
        
        Args:
            entities_with_cid: 包含CID信息的实体字典
            cid_groups: CID分组字典
            abbreviation_relations: abbreviation关系的三元组列表（作为证据）
            abbreviation_groups: abbreviation分组字典
            output_path: 输出文件路径
        """
        intermediate_data = {
            "entities_with_cid": entities_with_cid,
            "cid_groups": {
                str(cid): entity_keys for cid, entity_keys in cid_groups.items()
            },
            "statistics": {
                "entities_with_cid": len(entities_with_cid),
                "cid_groups_count": len(cid_groups),
                "entities_to_merge_by_cid": sum(len(keys) for keys in cid_groups.values() if len(keys) > 1)
            }
        }
        
        # 添加abbreviation相关信息
        if abbreviation_relations is not None:
            intermediate_data["abbreviation_relations"] = abbreviation_relations
            intermediate_data["statistics"]["abbreviation_relations_count"] = len(abbreviation_relations)
        
        if abbreviation_groups is not None:
            intermediate_data["abbreviation_groups"] = {
                main_entity: synonyms for main_entity, synonyms in abbreviation_groups.items()
            }
            intermediate_data["statistics"]["abbreviation_groups_count"] = len(abbreviation_groups)
            intermediate_data["statistics"]["entities_to_merge_by_abbreviation"] = sum(
                len(synonyms) for synonyms in abbreviation_groups.values() if len(synonyms) > 1
            )
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
            logger.info(f"中间结果已保存到: {output_path}（包含 {len(entities_with_cid)} 个有CID的实体）")
    
    def deduplicate(self, graph_path: str, output_path: str = None, 
                   intermediate_output: bool = False,
                   enable_abbreviation: bool = True,
                   enable_cid: bool = True) -> str:
        """
        执行完整的实体消歧流程（包括基于CID和基于abbreviation的消歧）
        
        Args:
            graph_path: 输入图谱JSON文件路径
            output_path: 输出消歧后的图谱JSON文件路径（可选，默认自动生成）
            intermediate_output: 是否保存中间结果（如果为True，自动生成路径）
            enable_abbreviation: 是否启用基于abbreviation的消歧（默认True）
            enable_cid: 是否启用基于CID的消歧（默认True）
            
        Returns:
            输出文件路径
        """
        logger.info("======== 开始实体消歧流程 ========")
        logger.info(f"消歧开关: Abbreviation={enable_abbreviation}, CID={enable_cid}")
        
        if not enable_abbreviation and not enable_cid:
            logger.warning("警告：两种消歧方式都被禁用，将不会进行任何消歧操作")
        
        # 1. 加载图谱
        graph_data = self.load_graph(graph_path)
        if not graph_data:
            logger.error("图谱数据为空，无法继续")
            return ""
        
        # 2. 提取所有实体
        entities = self.extract_entities(graph_data)
        
        # 初始化变量
        abbreviation_relations = []
        abbreviation_groups = {}
        abbreviation_mapping = {name: name for name in entities.keys()}
        entities_with_cid = {}
        cid_groups = {}
        cid_mapping = {}
        
        # 3. 基于abbreviation的消歧（如果启用）
        if enable_abbreviation:
            logger.info("--- 步骤1: 提取abbreviation关系 ---")
            abbreviation_relations = self.extract_abbreviation_relations(graph_data)
            
            # 构建abbreviation分组
            if abbreviation_relations:
                abbreviation_groups = self.build_abbreviation_groups(abbreviation_relations)
            
            # 基于abbreviation合并实体
            logger.info("--- 步骤2: 基于abbreviation合并实体 ---")
            if abbreviation_groups:
                entities, abbreviation_mapping = self.merge_entities_by_abbreviation(entities, abbreviation_groups)
        else:
            logger.info("--- 跳过abbreviation消歧（已禁用） ---")
        
        # 4. 基于CID的消歧（如果启用）
        if enable_cid:
            if not self.pubchem_client:
                logger.error("CID消歧已启用，但PubChem客户端未初始化")
                return ""
            
            # 为实体查询CID
            logger.info("--- 步骤3: 为实体查询CID ---")
            entities_with_cid = self.query_cids_for_entities(entities)
            
            # 按CID分组
            cid_groups = self.group_entities_by_cid(entities_with_cid)
            
            # 合并相同CID的实体
            logger.info("--- 步骤4: 基于CID合并实体 ---")
            merged_entities, cid_mapping = self.merge_entities(entities_with_cid, cid_groups)
        else:
            logger.info("--- 跳过CID消歧（已禁用） ---")
            # 如果没有启用CID消歧，使用abbreviation合并后的实体（或原始实体）
            merged_entities = entities
            cid_mapping = {name: name for name in entities.keys()}
        
        # 5. 保存中间结果（如果启用了中间结果输出）
        if intermediate_output:
            # 自动生成中间结果文件路径：与graph_path同路径，文件名为graph_path的文件名加"_intermediate"
            base_name = os.path.splitext(graph_path)[0]
            intermediate_output_path = f"{base_name}_intermediate.json"
            self.save_intermediate_result(
                entities_with_cid if enable_cid else {}, 
                cid_groups if enable_cid else {}, 
                abbreviation_relations if enable_abbreviation and abbreviation_relations else None,
                abbreviation_groups if enable_abbreviation and abbreviation_groups else None,
                intermediate_output_path
            )
        
        # 6. 合并映射关系
        # 根据启用的消歧方式合并映射
        final_mapping = {}
        if enable_abbreviation and enable_cid:
            # 两种方式都启用：先应用abbreviation映射，再应用CID映射
            for original_name in abbreviation_mapping.keys():
                abbrev_mapped = abbreviation_mapping.get(original_name, original_name)
                final_mapped = cid_mapping.get(abbrev_mapped, abbrev_mapped)
                final_mapping[original_name] = final_mapped
        elif enable_abbreviation:
            # 只启用abbreviation
            final_mapping = abbreviation_mapping
        elif enable_cid:
            # 只启用CID
            final_mapping = cid_mapping
        else:
            # 两种都禁用，使用原始映射
            final_mapping = {name: name for name in entities.keys()}
        
        # 7. 删除abbreviation相关的三元组（如果启用了abbreviation消歧）
        if enable_abbreviation:
            logger.info("--- 步骤5: 删除abbreviation三元组 ---")
            graph_without_abbrev = self.remove_abbreviation_triples(graph_data)
        else:
            graph_without_abbrev = graph_data
        
        # 8. 更新图谱（使用合并后的实体）
        logger.info("--- 步骤6: 更新图谱 ---")
        updated_graph = self.update_graph_with_merged_entities(graph_without_abbrev, merged_entities, final_mapping)

        # 9. 去重属性三元组（确保一个实体-属性组合只出现一次）
        logger.info("--- 步骤7: 去重属性三元组 ---")
        updated_graph = self.deduplicate_attribute_triples(updated_graph)

        # 10. 保存消歧后的图谱
        if not output_path:
            base_name = os.path.splitext(graph_path)[0]
            output_path = f"{base_name}_deduplicated.json"
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(updated_graph, f, ensure_ascii=False, indent=2)
        
        logger.info(f"消歧后的图谱已保存到: {output_path}")
        logger.info("======== 实体消歧流程完成 ========")
        
        return output_path
    
    def deduplicate_attribute_triples(self, graph_data: List[Dict]) -> List[Dict]:
        """
        去重属性三元组（确保一个实体-属性组合只出现一次）
        
        Args:
            graph_data: 图谱数据列表
            
        Returns:
            去重后的图谱数据
        """
        seen = set()
        deduplicated = []
        
        for rel in graph_data:
            # 获取start_node的name
            start_node = rel.get("start_node", {})
            start_name = start_node.get("properties", {}).get("name", "")
            if isinstance(start_name, list):
                start_name = start_name[0] if start_name else ""
            
            # 获取end_node的name
            end_node = rel.get("end_node", {})
            end_name = end_node.get("properties", {}).get("name", "")
            if isinstance(end_name, list):
                end_name = end_name[0] if end_name else ""
            
            relation = rel.get("relation", "")
            
            # 只对属性三元组进行去重
            if relation == "has_attribute":
                key = (start_name, end_name)
                if key not in seen:
                    seen.add(key)
                    deduplicated.append(rel)
                else:
                    logger.debug(f"去重属性三元组: {start_name} - {end_name}")
            else:
                deduplicated.append(rel)
        
        logger.info(f"属性三元组去重: {len(graph_data)} -> {len(deduplicated)}")
        return deduplicated
    
    def close(self):
        """关闭数据库连接"""
        if self.pubchem_client:
            self.pubchem_client.close()
    
    def __del__(self):
        """析构函数"""
        self.close()


def deduplicate_entities(
    graph_path: str,
    output_path: str = None,
    config: dict = None,
    *,
    intermediate_output: bool = False,
    enable_abbreviation: bool = True,
    enable_cid: bool = True,
    db_path: str = "pubchem_names_full.db",
) -> str:
    """
    实体消歧的便捷函数。
    推荐传入 config 字典；也可使用关键字参数（与 config 同时传入时，kwargs 覆盖 config）。
    config 可包含: output_path, intermediate_output, enable_abbreviation, enable_cid, pubchem_db_path（或 db_path）。

    Args:
        graph_path: 输入图谱JSON文件路径
        output_path: 输出消歧后的图谱JSON文件路径（可选）
        config: 可选配置字典
        intermediate_output: 是否保存中间结果
        enable_abbreviation: 是否启用基于abbreviation的消歧
        enable_cid: 是否启用基于CID的消歧
        db_path: PubChem本地数据库文件路径（config 中可用 pubchem_db_path 或 db_path）

    Returns:
        输出文件路径
    """
    if config:
        output_path = config.get("output_path", output_path)
        intermediate_output = config.get("intermediate_output", intermediate_output)
        enable_abbreviation = config.get("enable_abbreviation", enable_abbreviation)
        enable_cid = config.get("enable_cid", enable_cid)
        db_path = config.get("pubchem_db_path") or config.get("db_path", db_path)
    deduplicator = EntityDeduplicator(db_path=db_path, init_pubchem=enable_cid)
    try:
        return deduplicator.deduplicate(
            graph_path,
            output_path,
            intermediate_output,
            enable_abbreviation,
            enable_cid,
        )
    finally:
        if deduplicator.pubchem_client:
            deduplicator.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实体消歧工具")
    parser.add_argument("graph_path", help="输入图谱JSON文件路径")
    parser.add_argument("output_path", nargs="?", default=None, help="输出消歧后的图谱JSON文件路径（可选）")
    parser.add_argument("--intermediate-output", action="store_true",
                       help="保存中间结果（文件路径自动生成）")
    parser.add_argument("--enable-abbreviation", action="store_true", default=True,
                       help="启用基于abbreviation的消歧（默认启用）")
    parser.add_argument("--disable-abbreviation", dest="enable_abbreviation", action="store_false",
                       help="禁用基于abbreviation的消歧")
    parser.add_argument("--enable-cid", action="store_true", default=True,
                       help="启用基于CID的消歧（默认启用）")
    parser.add_argument("--disable-cid", dest="enable_cid", action="store_false",
                       help="禁用基于CID的消歧")
    parser.add_argument("--db-path", type=str, default="pubchem_names_full.db",
                       help="PubChem本地数据库文件路径（默认 pubchem_names_full.db，仅在启用CID消歧时需要）")
    
    args = parser.parse_args()
    
    deduplicate_entities(
        graph_path=args.graph_path,
        output_path=args.output_path,
        intermediate_output=args.intermediate_output,
        enable_abbreviation=args.enable_abbreviation,
        enable_cid=args.enable_cid,
        db_path=args.db_path
    )
