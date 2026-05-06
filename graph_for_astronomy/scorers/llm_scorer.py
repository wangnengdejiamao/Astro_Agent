#!/usr/bin/env python3
"""
LLM 打分模块
使用 LLM 对 chunk 进行质量评估和 bad_case 识别
"""

from typing import Dict, List, Any
import sys
import os

import json_repair

# 添加父目录到路径，以便导入 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import call_llm_api
from utils.logger import logger


class LLMScorer:
    """LLM 打分器"""
    
    def __init__(self):
        """初始化 LLM 打分器"""
        try:
            self.llm_client = call_llm_api.LLMCompletionCall()
        except Exception:
            logger.error("无法初始化 LLM 客户端")
            raise
    
    def score_chunk_llm(self, chunk_id: str, chunk_text: str, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用LLM对单个chunk进行打分（检查三元组准确性和bad_cases）
        
        Args:
            chunk_id: chunk ID
            chunk_text: chunk文本内容
            triples: 该chunk的所有三元组列表
            
        Returns:
            包含LLM分数和bad_case的字典
        """
        if not triples:
            return {
                "score": 0.0,
                "total_triples": 0,
                "bad_cases": []
            }
        
        # 构建三元组列表字符串
        triples_str = ""
        for i, triple in enumerate(triples, 1):
            triples_str += f"{i}. ({triple['start_node']}, {triple['relation']}, {triple['end_node']})\n"
            triples_str += f"   Source: {triple.get('source', '')}\n"
            triples_str += f"   Evidence: {triple.get('evidence', '')}\n\n"
        
        scoring_prompt = f"""You are a knowledge graph quality assessment expert. Please evaluate the accuracy of triple relationships extracted from the chunk.

**Chunk ID**: {chunk_id}

**Chunk Text Content**:
{chunk_text}

**Triple Relationships Extracted from This Chunk**:
{triples_str}

**Evaluation Task**:
Please check each triple relationship for the following "bad cases":

1. **Missing Information**: The chunk text does not contain information about this triple relationship, i.e., no evidence supporting this relationship can be found in the chunk.
2. **Contradiction**: The chunk contains contrastive words (such as "However", "In contrast", "But", "While", "Although", "Nevertheless", "On the other hand", etc.), and there are contradictory relationships. For example, it first says A improves B, but then contradicts by saying A degrades B.

**Output Requirements**:
Please return a JSON object in the following format:
{{
    "score": <a floating-point number between 0.0 and 1.0, representing the proportion of non-bad_case triples to total triples>,
    "bad_cases": [
        {{
            "start_node": "<start node name>",
            "relation": "<relation name>",
            "end_node": "<end node name>",
            "reason": "<reason for being identified as bad_case, specify whether it's missing information or contradiction, and briefly explain>"
        }},
        ...
    ]
}}

**Scoring Guidelines**:
- score = (total triples - bad_case count) / total triples
- If all triples are bad_cases, score = 0.0
- If no triples are bad_cases, score = 1.0
- The bad_cases list only contains triples identified as bad_cases. If there are no bad_cases, return an empty list.

**Important Notes**:
1. Carefully check whether each triple has clear support in the chunk text
2. Pay special attention to content after contrastive words to determine if there are contradictions
3. If the triple information exists in the chunk and has no contradictions, it should NOT be marked as a bad_case
4. For abbreviations or aliases, if there is a clear correspondence in the chunk, it should be considered valid

Please return ONLY the JSON object, do not include any other text."""

        try:
            response = self.llm_client.call_api(scoring_prompt)
            parsed = json_repair.loads(response)
            
            score = float(parsed.get("score", 0.0))
            score = max(0.0, min(1.0, score))  # 确保在有效范围内
            
            bad_cases = parsed.get("bad_cases", [])
            if not isinstance(bad_cases, list):
                bad_cases = []
            
            result = {
                "score": score,
                "total_triples": len(triples),
                "bad_cases": bad_cases
            }
            
            logger.debug(f"Chunk {chunk_id} LLM评分完成: score={score:.2f}, bad_cases={len(bad_cases)}")
            return result
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} LLM评分失败: {e}")
            # 返回默认结果
            return {
                "score": -1.0,  # -1表示评分失败
                "total_triples": len(triples),
                "bad_cases": [],
                "error": str(e)
            }

