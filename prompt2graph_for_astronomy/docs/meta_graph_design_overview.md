### 上层元图谱构建思路串讲（基于 `meta_graph_builder.py`）

---

## 1. 功能背景与初衷

- 底层已经有一张**实体级知识图谱**（如 `260129.json`）：
  - 节点：实体（化合物、电极、电解液、性能指标等）
  - 边：实体间关系（`improves`, `forms`, `inhibits`, `decomposes_to` 等）
- 这些三元组支持多跳推理，但很多**高层任务**其实想直接“拿三元组当对象”：
  - 发现完整的**机制链**（多个三元组首尾衔接）
  - 比较、聚合、筛掉**冗余或弱相关的三元组**
  - 在 triple 层做聚类 / GraphRAG / agent 推理

因此，我们需要一张“**以三元组为节点**”的上层图谱（Meta-Graph）：

- TripleNode：底层一个三元组
- MetaEdge：两个三元组之间的语义关系（由 LLM 端到端抽取）

`meta_graph_builder.py` 就是用来构建这张上层元图谱的。**当前实现采用端到端 LLM 抽取**，完全舍弃规则筛选，输入仅为底层图谱 JSON 和 chunks.txt。

---

## 2. 整体流程概览（端到端版本）

### 2.1 流程图

```text
┌───────────────────────────────────────────────────────────┐
│  输入：base_graph_path（底层图谱 JSON）                   │
│        chunks_path（chunks.txt 原文上下文）               │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│  1）parse_base_graph                                      │
│    - 解析 entity→entity 边                               │
│    - 构建 TripleNode 列表                                │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│  2）_load_chunks_map                                      │
│    - 从 chunks.txt 加载 chunk_id → chunk_text            │
│    - 每个 chunk 可截断（max_chars_per_chunk）            │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│  3）extract_semantic_edges_e2e                            │
│    - 将 triples + chunks 上下文拼入 prompt               │
│    - 单次或分批调用 LLM，直接抽取三元组间语义关系        │
│    - 输出：enables / supports / contradicts / elaborates │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│  4）组装并输出 meta_graph                                 │
│    - nodes: TripleNode 列表                              │
│    - edges: LLM 抽取的语义边（无规则边）                 │
│    - stats: 统计信息                                     │
└───────────────────────────────────────────────────────────┘
```

### 2.2 数据结构

- **TripleNode**
  - 来自底层一条实体-实体三元组
  - 字段：`id`, `subject`, `relation`, `object`, `schema_type_subj`, `schema_type_obj`, `source`, `evidence`, `chunk_ids`
- **MetaEdge**
  - 描述两个 TripleNode 之间的语义关系
  - 字段：`source_triple_id`, `target_triple_id`, `relation`, `extraction_source`, `evidence`
  - 当前实现中 `extraction_source` 恒为 `"llm"`

---

## 3. 第一步：从底层图谱抽取 TripleNode（`parse_base_graph`）

**输入**：底层图谱 JSON（边列表）

**过滤规则**：
- 只保留 `start_node.label == "entity"` 且 `end_node.label == "entity"` 的边
- 排除 `has_attribute` → attribute 的属性边

**输出**：`List[TripleNode]`

**设计动机**：
- 上层图谱关注「实体之间的事实」，属性值、单位等信息不直接参与 triple-level 推理。

---

## 4. 第二步：加载 Chunks 上下文（`_load_chunks_map`）

**输入**：`chunks_path`（chunks.txt）

**格式**：每行形如 `id: <chunk_id>\tChunk: {...}`，其中 dict 含 `text` 或 `chunk` 字段。

**输出**：`Dict[chunk_id, chunk_text]`，每个 chunk 可按 `max_chars_per_chunk` 截断。

**用途**：为 LLM 提供原文上下文，提升语义判断准确性。

---

## 5. 第三步：端到端 LLM 抽取（`extract_semantic_edges_e2e`）

### 5.1 思路

- **不再**基于规则（chains_to、same_subject、same_source）筛选候选对；
- **不再**用 embedding 或任务相关性过滤；
- 直接将**所有 triple + 相关 chunks** 交给 LLM，由 LLM 根据原文上下文**自行判断**哪些三元组对存在语义关系，并输出关系类型与推理。

### 5.2 Prompt 设计

- 模板文件：`prompts/meta_graph_e2e_extract.txt`
- 占位符：
  - `{triples_json}`：编号后的三元组列表（idx, id, subject, relation, object, source, chunk_ids）
  - `{chunks_text}`：拼接后的 chunk 文本
- 输出格式：JSON 数组，每项为 `{triple_i, triple_j, relation, reasoning}`，仅包含有关系的三元组对。

### 5.3 分批策略

- 若三元组数量超过 `max_triples_per_batch`，则分批处理；
- 每批内 LLM 输出该批内 triple 之间的语义边；
- **注意**：跨批的三元组对不会被考虑，可能漏掉部分关系。

### 5.4 关系类型

与 prompt 约定一致：
- **enables**：因果/机制链
- **supports**：证据支持
- **contradicts**：矛盾
- **elaborates**：细化/补充

---

## 6. 输出结构与使用示例

### 6.1 输出 JSON 结构

```json
{
  "meta_graph": {
    "nodes": [...TripleNode 列表...],
    "edges": [...LLM 抽取的语义边...]
  },
  "stats": {
    "num_triple_nodes": 63,
    "num_edges": 15
  }
}
```

### 6.2 命令行示例

```bash
# 端到端：只需 base_graph + chunks
python meta_graph_builder.py \
  output/paper_mini/20260205114356/260129.json \
  output/paper_mini/chunks.txt \
  -o output/paper_mini/260129_meta.json

# 指定 prompt、chunk 截断、批大小
python meta_graph_builder.py \
  output/paper_mini/260129.json \
  output/paper_mini/chunks.txt \
  -o output/paper_mini/260129_meta.json \
  --prompt prompts/meta_graph_e2e_extract.txt \
  --max-chars-per-chunk 2000 \
  --max-total-chunk-chars 40000 \
  --max-triples-per-batch 80
```

### 6.3 Python 调用

```python
from meta_graph_builder import build_meta_graph

result = build_meta_graph(
    base_graph_path="output/paper_mini/260129.json",
    chunks_path="output/paper_mini/chunks.txt",
    output_path="output/paper_mini/260129_meta.json",
    prompt_path=None,
    max_chars_per_chunk=2000,
    max_total_chunk_chars=40000,
    max_triples_per_batch=80,
)
print(result["stats"])
```

---

## 7. 依赖与预期收益

### 7.1 主要依赖

- `utils.call_llm_api`：统一的 LLM 调用封装
- `json_repair`：容错解析 LLM 返回的 JSON

**不再依赖**：`sentence-transformers`（已移除 embedding 语义过滤）

### 7.2 预期收益

通过 `meta_graph_builder.py` 构建的上层元图谱，可以：

- 在 triple 层显式表示**机制链**、**方法–评价链**、**性能–表征链**；
- 利用 `enables/supports/contradicts` 等关系做事实验证与证据聚合；
- 为 GraphRAG、agent 多跳问答提供更语义化、更结构化的「事实空间」；
- 方便做 triple-level 的聚类、社区检测和可视化分析。

端到端设计将「候选筛选 + 关系抽取」合并为单步 LLM 推理，简化管线并更好地利用原文上下文。
