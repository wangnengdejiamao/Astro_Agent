#!/usr/bin/env python3
"""
Chunk级别的打分函数
输入：chunk文档文件和对应的三元组图谱json文件
输出：包含每个chunk分数和bad_case的json文件
"""

import ast
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Any

# 添加父目录到路径，以便导入 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorers.llm_scorer import LLMScorer
from scorers.pubchem_scorer import PubChemScorer
from utils.logger import logger


class ChunkScorer:
    """Chunk级别的打分器"""
    
    def __init__(self, pubchem_db_path: str = "pubchem_names_full.db", enable_llm: bool = True, enable_pubchem: bool = True):
        """
        初始化 Chunk 打分器
        
        Args:
            pubchem_db_path: PubChem 本地数据库文件路径（默认 pubchem_names_full.db）
            enable_llm: 是否启用 LLM 打分器
            enable_pubchem: 是否启用 PubChem 打分器
        """
        self.llm_scorer = None
        self.pubchem_scorer = None
        
        if enable_llm:
            try:
                self.llm_scorer = LLMScorer()
            except Exception as e:
                logger.warning(f"无法初始化 LLM 打分器: {e}")
        
        if enable_pubchem:
            try:
                self.pubchem_scorer = PubChemScorer(db_path=pubchem_db_path)
            except Exception as e:
                logger.warning(f"无法初始化 PubChem 打分器: {e}")
    
    def load_chunks(self, chunk_path: str) -> Dict[str, Dict[str, str]]:
        """
        加载chunk文件
        
        Args:
            chunk_path: chunk文件路径
            
        Returns:
            字典，键为chunk_id，值为包含title和text的字典
        """
        chunks = {}
        if not os.path.exists(chunk_path):
            logger.error(f"Chunk文件不存在: {chunk_path}")
            return chunks
        
        try:
            with open(chunk_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "\t" not in line:
                        continue
                    
                    parts = line.split("\t", 1)
                    if len(parts) != 2 or not parts[0].startswith("id: ") or not parts[1].startswith("Chunk: "):
                        continue
                    
                    chunk_id = parts[0][4:].strip()
                    chunk_data_str = parts[1][7:].strip()
                    
                    try:
                        chunk_data = ast.literal_eval(chunk_data_str)
                        if isinstance(chunk_data, dict):
                            chunks[chunk_id] = {
                                "title": chunk_data.get("title", ""),
                                "text": chunk_data.get("text", "")
                            }
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"解析chunk {chunk_id}失败: {e}")
        except Exception as e:
            logger.error(f"读取chunk文件失败: {e}")
        
        logger.info(f"成功加载 {len(chunks)} 个chunks")
        return chunks
    
    def load_graph(self, graph_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        加载图谱json文件并按chunk_id分组
        
        Args:
            graph_path: 图谱json文件路径
            
        Returns:
            字典，键为chunk_id，值为该chunk的所有三元组列表
        """
        triples_by_chunk = defaultdict(list)
        
        if not os.path.exists(graph_path):
            logger.error(f"图谱文件不存在: {graph_path}")
            return triples_by_chunk
        
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            
            for triple in graph_data:
                chunk_id = triple.get("chunk_id")
                if chunk_id:
                    # 提取三元组信息
                    start_node = triple.get("start_node", {}).get("properties", {}).get("name", "")
                    relation = triple.get("relation", "")
                    end_node = triple.get("end_node", {}).get("properties", {}).get("name", "")
                    
                    if start_node and relation and end_node:
                        triples_by_chunk[chunk_id].append({
                            "start_node": start_node,
                            "relation": relation,
                            "end_node": end_node,
                            "source": triple.get("source", ""),
                            "evidence": triple.get("evidence", "")
                        })
        except Exception as e:
            logger.error(f"读取图谱文件失败: {e}")
        
        with open("triples_by_chunk.json", "w", encoding="utf-8") as f:
            json.dump(triples_by_chunk, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功加载图谱，涉及 {len(triples_by_chunk)} 个chunks")
        return dict(triples_by_chunk)
    
    def score_chunk(self, chunk_id: str, chunk_text: str, triples: List[Dict, Any],
                   enable_llm_score: bool = True, enable_pubchem_score: bool = True) -> Dict[str, Any]:
        """
        对单个chunk进行打分（根据开关参数决定执行哪些打分）
        
        Args:
            chunk_id: chunk ID
            chunk_text: chunk文本内容
            triples: 该chunk的所有三元组列表
            enable_llm_score: 是否启用LLM打分
            enable_pubchem_score: 是否启用PubChem打分
            
        Returns:
            包含分数和bad_case的字典
        """
        result = {
            "chunk_id": chunk_id
        }
        
        if not triples:
            result.update({
                "score": 0.0 if enable_llm_score else None,
                "total_triples": 0,
                "bad_cases": [] if enable_llm_score else None,
                "pubchem_score": 0.0 if enable_pubchem_score else None,
                "pubchem_entity_count": 0 if enable_pubchem_score else None,
                "pubchem_found_count": 0 if enable_pubchem_score else None,
                "pubchem_found_entities": [] if enable_pubchem_score else None,
                "pubchem_not_found_entities": [] if enable_pubchem_score else None
            })
            return result
        
        # LLM打分
        if enable_llm_score and self.llm_scorer:
            llm_result = self.llm_scorer.score_chunk_llm(chunk_id, chunk_text, triples)
            result.update(llm_result)
        else:
            result.update({
                "score": None,
                "total_triples": len(triples),
                "bad_cases": None
            })
        
        # PubChem打分
        if enable_pubchem_score and self.pubchem_scorer:
            t_pubchem_start = time.time()
            pubchem_result = self.pubchem_scorer.score_chunk_pubchem(triples)
            t_pubchem_cost = time.time() - t_pubchem_start
            # 计算每次 PubChem lookup 的平均耗时（按唯一实体数估算）
            entity_count = pubchem_result.get("pubchem_entity_count") or 0
            avg_lookup_time = t_pubchem_cost / entity_count if entity_count > 0 else None
            pubchem_result["pubchem_avg_lookup_time"] = avg_lookup_time
            result.update(pubchem_result)
        else:
            result.update({
                "pubchem_score": None,
                "pubchem_entity_count": None,
                "pubchem_found_count": None,
                "pubchem_found_entities": None,
                "pubchem_not_found_entities": None,
                "pubchem_time_cost": None,
                "pubchem_avg_lookup_time": None,
            })
        
        logger.debug(f"Chunk {chunk_id} 评分完成: LLM={enable_llm_score}, PubChem={enable_pubchem_score}")
        return result
    
    def score_all_chunks(self, chunk_path: str, graph_path: str, output_path: str = None,
                        enable_llm_score: bool = True, enable_pubchem_score: bool = True) -> Dict[str, Any]:
        """
        对所有chunks进行打分
        
        Args:
            chunk_path: chunk文件路径
            graph_path: 图谱json文件路径
            output_path: 输出json文件路径（可选）
            enable_llm_score: 是否启用LLM打分（默认True）
            enable_pubchem_score: 是否启用PubChem打分（默认True）
            
        Returns:
            包含所有chunk评分结果的字典
        """
        logger.info("开始加载数据...")
        logger.info(f"打分开关: LLM={enable_llm_score}, PubChem={enable_pubchem_score}")
        
        chunks = self.load_chunks(chunk_path)
        triples_by_chunk = self.load_graph(graph_path)
        
        # 获取所有需要评分的chunk_id（在chunks和triples中都存在的）
        chunk_ids = set(chunks.keys()) & set(triples_by_chunk.keys())
        
        if not chunk_ids:
            logger.warning("没有找到需要评分的chunks（chunks和图谱中没有匹配的chunk_id）")
            return {"results": []}
        
        logger.info(f"开始对 {len(chunk_ids)} 个chunks进行评分...")
        
        results = []
        for chunk_id in sorted(chunk_ids):
            chunk_text = chunks[chunk_id].get("text", "")
            triples = triples_by_chunk[chunk_id]
            
            # 记录单个chunk打分开始时间
            t_start = time.time()
            result = self.score_chunk(
                chunk_id,
                chunk_text,
                triples,
                enable_llm_score=enable_llm_score,
                enable_pubchem_score=enable_pubchem_score,
            )
            t_cost = time.time() - t_start  # 单位：秒
            results.append(result)
            
            # 构建进度信息
            t_cost_ms = t_cost * 1000.0  # 转为毫秒
            progress_info = f"进度: {len(results)}/{len(chunk_ids)} - Chunk {chunk_id}, time={t_cost_ms:.1f}ms"
            if enable_llm_score:
                progress_info += f", score={result.get('score', -1):.2f}"
            if enable_pubchem_score:
                progress_info += f", pubchem_score={result.get('pubchem_score', 0):.2f}"
                avg_lookup_time = result.get("pubchem_avg_lookup_time")
                if avg_lookup_time is not None:
                    progress_info += f", pubchem_avg_lookup_time={avg_lookup_time * 1000.0:.2f}ms"
            logger.info(progress_info)
        
        # 计算统计信息
        output_data = {
            "total_chunks": len(results),
            "chunks_with_triples": len([r for r in results if r.get("total_triples", 0) > 0]),
            "total_triples": sum(r.get("total_triples", 0) for r in results),
            "results": results
        }
        
        # LLM打分统计
        if enable_llm_score:
            valid_scores = [r.get("score", 0) for r in results if r.get("score") is not None and r.get("score", -1) >= 0]
            output_data["total_bad_cases"] = sum(len(r.get("bad_cases", [])) for r in results if r.get("bad_cases") is not None)
            output_data["average_score"] = sum(valid_scores) / max(1, len(valid_scores)) if valid_scores else 0.0
        else:
            output_data["total_bad_cases"] = None
            output_data["average_score"] = None
        
        # PubChem打分统计
        if enable_pubchem_score:
            valid_pubchem_scores = [r.get("pubchem_score", 0) for r in results if r.get("pubchem_score") is not None]
            output_data["average_pubchem_score"] = sum(valid_pubchem_scores) / max(1, len(valid_pubchem_scores)) if valid_pubchem_scores else 0.0
            output_data["total_entities"] = sum(r.get("pubchem_entity_count", 0) for r in results if r.get("pubchem_entity_count") is not None)
            output_data["total_found_entities"] = sum(r.get("pubchem_found_count", 0) for r in results if r.get("pubchem_found_count") is not None)
            output_data["overall_pubchem_score"] = output_data["total_found_entities"] / max(1, output_data["total_entities"]) if output_data["total_entities"] > 0 else 0.0
        else:
            output_data["average_pubchem_score"] = None
            output_data["total_entities"] = None
            output_data["total_found_entities"] = None
            output_data["overall_pubchem_score"] = None
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"评分结果已保存到: {output_path}")
        
        # 构建完成信息
        completion_info = "评分完成！"
        if enable_llm_score:
            completion_info += f" 平均分数: {output_data['average_score']:.2f}"
        if enable_pubchem_score:
            completion_info += f", 平均 PubChem 分数: {output_data['average_pubchem_score']:.2f}"
        logger.info(completion_info)
        
        return output_data


def score_chunks(chunk_path: str, graph_path: str, output_path: str = None, 
                pubchem_db_path: str = "pubchem_names_full.db", 
                enable_llm_score: bool = True, enable_pubchem_score: bool = True) -> Dict[str, Any]:
    """
    Chunk级别打分的便捷函数
    
    Args:
        chunk_path: chunk文件路径
        graph_path: 图谱json文件路径
        output_path: 输出json文件路径（可选）
        pubchem_db_path: PubChem 本地数据库文件路径（默认 pubchem_names_full.db）
        enable_llm_score: 是否启用LLM打分（默认True）
        enable_pubchem_score: 是否启用PubChem打分（默认True）
        
    Returns:
        包含所有chunk评分结果的字典
    """
    scorer = ChunkScorer(
        pubchem_db_path=pubchem_db_path,
        enable_llm=enable_llm_score,
        enable_pubchem=enable_pubchem_score
    )
    return scorer.score_all_chunks(chunk_path, graph_path, output_path,
                                  enable_llm_score=enable_llm_score,
                                  enable_pubchem_score=enable_pubchem_score)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk级别的打分工具")
    parser.add_argument("chunk_path", help="chunk文件路径")
    parser.add_argument("graph_path", help="图谱json文件路径")
    parser.add_argument("output_path", nargs="?", default=None, help="输出json文件路径（可选）")
    parser.add_argument("--pubchem-db", type=str, default="pubchem_names_full.db",
                       help="PubChem 本地数据库文件路径（默认 pubchem_names_full.db）")
    parser.add_argument("--enable-llm-score", action="store_true", default=True,
                       help="启用LLM打分（默认启用）")
    parser.add_argument("--disable-llm-score", dest="enable_llm_score", action="store_false",
                       help="禁用LLM打分")
    parser.add_argument("--enable-pubchem-score", action="store_true", default=True,
                       help="启用PubChem打分（默认启用）")
    parser.add_argument("--disable-pubchem-score", dest="enable_pubchem_score", action="store_false",
                       help="禁用PubChem打分")
    
    args = parser.parse_args()
    
    chunk_path = args.chunk_path
    graph_path = args.graph_path
    output_path = args.output_path
    pubchem_db_path = args.pubchem_db
    enable_llm_score = args.enable_llm_score
    enable_pubchem_score = args.enable_pubchem_score
    
    if not output_path:
        # 如果没有指定输出路径，自动生成
        base_name = os.path.splitext(graph_path)[0]
        output_path = f"{base_name}_chunk_scores.json"
    
    if enable_pubchem_score:
        logger.info(f"使用本地 PubChem 数据库: {pubchem_db_path}")
    
    score_chunks(chunk_path, graph_path, output_path, 
                pubchem_db_path=pubchem_db_path,
                enable_llm_score=enable_llm_score,
                enable_pubchem_score=enable_pubchem_score)

