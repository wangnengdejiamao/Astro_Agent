#!/usr/bin/env python3
from __future__ import annotations
"""
单阶段知识图谱构建器
一次性 LLM 调用提取实体、关系、属性
"""

import json
import os
from typing import Any, Dict, List, Optional

from graph_builder import GraphBuilder
from utils.logger import logger


class SingleGraphBuilder(GraphBuilder):
    """单阶段图谱构建器：一次 LLM 调用提取所有信息"""
    
    def __init__(
        self,
        schema_path: str = None,
        schema_content: dict = None,
        prompt_path: str = None,
        prompt_content: str = None,
        pubchem_db_path: str = None,
    ):
        super().__init__(schema_path, schema_content, pubchem_db_path)
        self.prompt = self._load_prompt(prompt_path, prompt_content)
    
    def _load_prompt(self, prompt_path: str = None, prompt_content: str = None) -> str:
        """加载 prompt"""
        if prompt_content:
            return prompt_content
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        # 默认 prompt
        return """请从以下文本中提取知识图谱信息，以 JSON 格式返回：

{
  "entities": [
    {"name": "实体名称", "type": "实体类型", "description": "描述"}
  ],
  "relations": [
    {"source": "源实体", "target": "目标实体", "type": "关系类型", "description": "描述"}
  ]
}

文本：
{chunk}
"""
    
    def _process_chunk_impl(self, chunk: str, chunk_id: str, **kwargs) -> Dict[str, Any]:
        """处理单个 chunk，调用 LLM 提取实体和关系"""
        import requests
        
        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        model = os.getenv("LLM_MODEL", "deepseek-v4-pro")
        
        if not api_key:
            logger.warning("No API key found, returning empty graph")
            return {"entities": [], "relations": []}
        
        prompt = self.prompt.replace("{chunk}", chunk)
        
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "你是一个知识图谱构建专家。请从文本中提取实体和关系，以JSON格式返回。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4000
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 提取 JSON
                try:
                    # 尝试直接解析
                    data = json.loads(content)
                    return data
                except json.JSONDecodeError:
                    # 尝试从 markdown 代码块中提取
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                        return data
                    # 尝试找到 JSON 对象
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                        return data
                    
                    logger.warning(f"Failed to parse LLM response for chunk {chunk_id}")
                    return {"entities": [], "relations": []}
            else:
                logger.warning(f"LLM API error: {response.status_code} - {response.text[:200]}")
                return {"entities": [], "relations": []}
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")
            return {"entities": [], "relations": []}
    
    def process_chunks(
        self,
        chunks: List[str],
        output_path: str = None,
        max_workers: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """处理多个 chunks"""
        all_entities = []
        all_relations = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i}"
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk_id}")
            
            result = self._process_chunk_impl(chunk, chunk_id, **kwargs)
            
            if "entities" in result:
                all_entities.extend(result["entities"])
            if "relations" in result:
                all_relations.extend(result["relations"])
        
        # 去重
        seen_entities = set()
        unique_entities = []
        for e in all_entities:
            key = (e.get("name", ""), e.get("type", ""))
            if key not in seen_entities:
                seen_entities.add(key)
                unique_entities.append(e)
        
        seen_relations = set()
        unique_relations = []
        for r in all_relations:
            key = (r.get("source", ""), r.get("target", ""), r.get("type", ""))
            if key not in seen_relations:
                seen_relations.add(key)
                unique_relations.append(r)
        
        graph = {
            "entities": unique_entities,
            "relations": unique_relations
        }
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph, f, ensure_ascii=False, indent=2)
            logger.info(f"Graph saved to {output_path}")
        
        return graph
