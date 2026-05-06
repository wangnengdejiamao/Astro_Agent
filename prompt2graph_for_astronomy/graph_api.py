"""
统一图谱数据接口
提供 Neo4j 和 Milvus 的联合查询能力
"""
from typing import List, Dict, Any, Optional
import os
from neo4j import GraphDatabase
import json


class UnifiedGraphAPI:
    """统一图谱查询接口"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", 
                 neo4j_password: str = "password"):
        """
        初始化统一图谱接口
        
        Args:
            neo4j_uri: Neo4j 连接地址
            neo4j_user: Neo4j 用户名
            neo4j_password: Neo4j 密码
        """
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
    def close(self):
        """关闭连接"""
        self.neo4j_driver.close()
    
    def query_subgraph(self, node_name: str, depth: int = 1) -> Dict[str, Any]:
        """
        查询节点的子图
        
        Args:
            node_name: 节点名称
            depth: 遍历深度
            
        Returns:
            子图数据，包含 nodes 和 links
        """
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH path = (start:Node)-[r*1..{depth}]-(connected)
                WHERE start.name = $node_name OR $node_name IN start.aliases
                WITH DISTINCT start, connected, r
                RETURN collect(DISTINCT {
                    id: id(start),
                    name: start.name,
                    label: labels(start)[0],
                    schema_type: start.schema_type,
                    aliases: start.aliases
                }) as nodes,
                collect(DISTINCT {
                    source: id(startNode(last(r))),
                    target: id(endNode(last(r))),
                    relation: type(last(r)),
                    evidence: last(r).evidence
                }) as links
            """, node_name=node_name, depth=depth)
            
            record = result.single()
            return {
                "nodes": record["nodes"],
                "links": record["links"]
            }
    
    def search_by_semantic(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        语义搜索（结合 Milvus）
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            匹配的节点或三元组列表
        """
        # 这里调用 Milvus 进行向量检索
        # 然后到 Neo4j 查询完整信息
        # TODO: 集成 Milvus 检索
        pass
    
    def get_node_details(self, node_name: str) -> Optional[Dict]:
        """
        获取节点详细信息
        
        Args:
            node_name: 节点名称
            
        Returns:
            节点详细信息，包含所有关系和属性
        """
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (n:Node)
                WHERE n.name = $node_name OR $node_name IN n.aliases
                OPTIONAL MATCH (n)-[r]->(m)
                RETURN n as node, 
                       collect({
                           relation: type(r),
                           target: m.name,
                           target_schema: m.schema_type,
                           evidence: r.evidence
                       }) as outgoing_relations,
                       collect({
                           relation: type(r2),
                           source: m2.name,
                           source_schema: m2.schema_type,
                           evidence: r2.evidence
                       }) as incoming_relations
            """, node_name=node_name)
            
            record = result.single()
            if not record:
                return None
                
            node = record["node"]
            return {
                "name": node["name"],
                "schema_type": node.get("schema_type"),
                "aliases": node.get("aliases", []),
                "outgoing": record["outgoing_relations"],
                "incoming": record["incoming_relations"]
            }
    
    def find_paths(self, start_node: str, end_node: str, 
                   max_depth: int = 5) -> List[List[Dict]]:
        """
        查找两个节点之间的路径
        
        Args:
            start_node: 起始节点
            end_node: 目标节点
            max_depth: 最大深度
            
        Returns:
            路径列表，每条路径是边列表
        """
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH path = shortestPath(
                    (start:Node)-[*1..{max_depth}]-(end:Node)
                )
                WHERE (start.name = $start_node OR $start_node IN start.aliases)
                  AND (end.name = $end_node OR $end_node IN end.aliases)
                RETURN [edge IN relationships(path) | {
                    from: startNode(edge).name,
                    to: endNode(edge).name,
                    relation: type(edge),
                    evidence: edge.evidence
                }] as path_edges
                LIMIT 10
            """, start_node=start_node, end_node=end_node, max_depth=max_depth)
            
            return [record["path_edges"] for record in result]
    
    def export_triples(self, dataset_name: Optional[str] = None) -> List[Dict]:
        """
        导出所有三元组
        
        Args:
            dataset_name: 数据集名称，None 表示全部
            
        Returns:
            三元组列表
        """
        query = """
            MATCH (a:Node)-[r]->(b:Node)
            RETURN {
                subject: a.name,
                subject_schema: a.schema_type,
                relation: type(r),
                object: b.name,
                object_schema: b.schema_type,
                evidence: r.evidence,
                source: r.source,
                chunk_id: r.chunk_id
            } as triple
        """
        
        with self.neo4j_driver.session() as session:
            result = session.run(query)
            return [record["triple"] for record in result]


# FastAPI 接口封装
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Unified Graph API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 API 实例
graph_api = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化连接"""
    global graph_api
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    graph_api = UnifiedGraphAPI(neo4j_uri, neo4j_user, neo4j_password)


@app.on_event("shutdown")
async def shutdown_event():
    """关闭时释放连接"""
    if graph_api:
        graph_api.close()


@app.get("/api/graph/node/{node_name}")
async def get_node(node_name: str):
    """获取节点详情"""
    result = graph_api.get_node_details(node_name)
    if not result:
        raise HTTPException(status_code=404, detail="Node not found")
    return result


@app.get("/api/graph/subgraph/{node_name}")
async def get_subgraph(node_name: str, depth: int = 1):
    """获取子图"""
    return graph_api.query_subgraph(node_name, depth)


@app.get("/api/graph/paths")
async def get_paths(start: str, end: str, max_depth: int = 5):
    """查找路径"""
    paths = graph_api.find_paths(start, end, max_depth)
    return {"paths": paths, "count": len(paths)}


@app.get("/api/graph/export")
async def export_graph():
    """导出所有三元组"""
    triples = graph_api.export_triples()
    return {"triples": triples, "count": len(triples)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5010)
