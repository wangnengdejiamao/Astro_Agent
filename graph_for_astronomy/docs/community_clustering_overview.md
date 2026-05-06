### 底层知识图谱社区聚类与社区报告方案串讲  

本说明文档总结并对比 `community_clustering.py` 与 `tree_comm.py` 中两种底层图谱社区聚类方案，以及它们对应的社区报告生成流程，重点突出 **Leiden 纯拓扑聚类** 和 **FastTreeComm 语义+结构双重感知聚类** 的差异与适用场景。

---

## 一、整体对比概览

- **Leiden 方案（`community_clustering.py`）**
  - **输入图**: 由底层 JSON 图谱 (`260129.json`) 解析出的 **实体-实体无向图**。
  - **算法核心**: `graspologic.partition.hierarchical_leiden`（Leiden 层次社区检测），**纯基于图结构**。
  - **社区粒度控制**: `max_cluster_size`；可选只用最大连通分量 `use_lcc`。
  - **社区报告**:
    - 先用规则统计社区内成员、关系频次。
    - 再共享一个 LLM prompt (`prompts/lowlevel_leiden_community_report.txt`) 为每个社区生成 `name` 与 `summary`。

- **TreeComm 方案（`tree_comm.py` / `FastTreeComm`）**
  - **输入图**: 从底层 JSON 图谱构建的 **实体-实体有向 MultiDiGraph**，保留 `relation` 等属性。
  - **算法核心**: 结合
    - 三元组级 **embedding 语义相似度**（SentenceTransformer / 本地 embedding 服务）；
    - 邻居集合的 **Jaccard 结构相似度**；
    - KMeans 初始聚类 + 相似簇合并的多阶段细化流程。
  - **社区报告**:
    - 利用 `detect_communities` 得到社区后，
    - 通过 `_build_batch_prompt` + `_call_llm_api_batch` 调用 LLM，
    - 用同一份 `prompts/lowlevel_leiden_community_report.txt` prompt 生成社区 `name` 和 `summary`，
    - 并写入 `*_tree_comm_community_report.json/.txt`。

两种方案都以 **底层实体图谱** 为输入，都输出：

- `*_communities.json`: 每个实体的 `community_id`。
- `*_community_report.json`: 社区维度的摘要信息（含 LLM 生成的 name/summary）。

区别主要在于 **如何划分社区（拓扑 vs 语义+结构）**，以及 **TreeComm 内部更复杂的工作流**。

---

## 二、Leiden 社区聚类方案（`community_clustering.py`）

### 2.1 图构建：从 JSON 到实体无向图

入口函数 `load_graph_from_json`：

- 解析底层图谱 JSON（list 或包含 `"edges"` / `"triples"` 的 dict）。
- 只保留 `start_node.label == "entity"` 且 `end_node.label == "entity"` 的边。
- 用实体名作为 NetworkX 节点 ID，构建 **无向图 `nx.Graph`**，多条边通过权重叠加：

**要点：**
- 边权 = 出现次数，用于 Leiden 内部的加权聚类。
- 同时抽取简化三元组信息 `{"subject", "relation", "object"}`，供后续社区报告用。

### 2.2 社区检测：`cluster_with_leiden`

该函数包装了 `graspologic.partition.hierarchical_leiden`：

1. 选项：`use_lcc` 时先取最大连通分量（LCC），避免孤立子图干扰。
2. 调用 `hierarchical_leiden(graph, max_cluster_size, random_seed=seed)`。
3. 利用返回的 `HierarchicalClusters`：
   - 使用 `final_level_hierarchical_clustering()` 得到 **每个节点的最终社区归属**，避免简单取 `max_level` 时遗漏未细分社区的节点。
   - 同时遍历所有 `partition`，构建 `hierarchy: {cluster_id -> parent_cluster_id}`，用于记录层次结构。

**输出：**

- `node_to_community: {节点名 -> 社区ID}`
- `hierarchy: {社区ID -> 父社区ID}`（`-1` 表示根社区）

在你的当前数据上，一般不会触发进一步细分，`hierarchy` 多数是 `-1`。

### 2.3 规则版社区报告：`build_community_report`

不依赖 LLM 的版本主要做：

- 按 `community_id` 聚合成员实体。
- 扫描所有 `triples_for_report`，仅在 **subject 和 object 同属一个社区** 时计数其 `relation`：
  - 得到 `relation_counts` 和 `top_relations`（前若干高频关系）。
- 生成初始报告项：

```python
{
  "community_id": comm_id,
  "size": len(members),
  "members": [...],
  "relation_counts": {...},
  "top_relations": [...],
  "summary": "社区 X: N 个实体, 主要关系: improves, inhibits, ..."
}
```

这一步给出了**结构化统计**，但 `summary` 较为简陋，为下一步 LLM 提升做基础。

### 2.4 LLM 增强：`enrich_communities_with_llm`

为了与 TreeComm 对齐、并生成更自然的报告，`community_clustering.py` 在上述基础上增加了 LLM 步骤：

1. 从 `reports` 构造简化版社区信息，字段包括：
   - `id`（社区ID）
   - `size`（规模）
   - `members`（最多前 20 个实体）
   - `top_relations`（如 `["improves", 5]`）
2. 序列化为 `communities_json`，填充进模板：

   - 模板文件：`prompts/lowlevel_leiden_community_report.txt`
   - 占位符：`{communities_json}`
   - 模板中约定输出格式为 JSON 数组，每个元素含 `id / name / summary`。

3. 通过 `utils.call_llm_api.LLMCompletionCall().call_api(prompt)` 调用外部 DeepSeek（或你配置的 LLM）。
4. 用 `json_repair.loads` 解析 LLM 输出，按 `id` 回填：
   - `name`：社区名称
   - `summary`：更自然的文本摘要
5. 把 `name`、`summary` 写入最终 `community_report` 中，并在文本版报告中展示：
   - 标题：`### 社区 0 - <name> (规模: X)`
   - 摘要行：`摘要: <summary>`

**关键点：**

- Leiden 方案的社区边界 **完全由图结构+Leiden 决定**；
- LLM 只负责“解释”和“命名”社区，但不改变聚类结果；
- prompt 与 TreeComm 方案共用，便于做公平对比。

---

## 三、TreeComm 语义+结构双重感知社区聚类方案（`tree_comm.py`）

TreeComm 的工作流比 Leiden 复杂许多，主要体现在：

- 更丰富的**节点语义表示**（基于三元组文本 + embedding）；
- 细致的**结构相似度建模**（Jaccard + 稀疏邻接）；
- 多阶段的**聚类 + 簇细化 + 合并**流程；
- 下游还支持**社区超节点**与**关键词节点**的创建（可用于上层图谱）。

下面按流程串讲。

### 3.1 图构建适配：`_load_entity_graph_from_json`

为了让 FastTreeComm 可以直接运行在你的底层图谱上，`tree_comm.py` 内部提供了：

- `_load_entity_graph_from_json(path)`：
  - 读取底层 JSON 图谱。
  - 仅保留 `start_node/ end_node` 为 `entity` 的三元组。
  - 使用实体名作为节点 ID，构建 **有向 MultiDiGraph**：
    - 节点属性 `properties.name`，供后续文本构造使用。
    - 边属性 `relation`，用于生成 triple 文本。

这一步与 Leiden 方案类似，但：

- 使用的是 `nx.MultiDiGraph`；
- 保留了单向关系，有利于反映更细粒度的结构模式。

### 3.2 FastTreeComm 初始化：embedding 与缓存

`FastTreeComm.__init__` 做了大量准备工作：

1. **模型选择**
   - 若 `embedding_model == "local_embedding_service"`：使用 `LocalEmbeddingServiceEncoder` 走 TCP 本地服务；
   - 否则使用 `SentenceTransformer(embedding_model)` 加载本地/在线句向量模型。
2. **节点与三元组缓存**
   - `self.node_list` / `self.node_names`：映射节点 ID → 实体名；
   - `self.triple_strings_cache`：为每个节点缓存邻接三元组句子；
   - `self.degree_cache`：节点度数，用于结构得分。
3. **稀疏邻接矩阵**
   - `_build_sparse_adjacency` 构建 `csr_matrix`，用于后续计算 Jaccard 相似度。
4. **LLM 客户端与批大小**
   - `self.llm_client = call_llm_api.LLMCompletionCall()`；
   - 根据环境/配置确定 embedding batch 大小、是否展示进度条等。
5. **预计算三元组文本**
   - `_precompute_all_triples()`：遍历所有节点，生成并缓存节点周围的 triple 字符串列表，减少后续重复工作。

### 3.3 语义+结构相似度建模

#### 3.3.1 triple-level embedding：`get_triple_embeddings_batch`

- 对节点集合 `node_ids`：
  - 若未缓存 embedding，则：
    - 获取 `triple_strings_cache[nid]`，拼成一段文本（截断至 `max_triple_chars`）；
    - 调用 SentenceTransformer 批量编码；
    - 将结果缓存到 `semantic_cache`。
  - 最终返回一个 `np.ndarray` 形状为 `(n_nodes, dim)` 的向量矩阵。

这一步为每个节点构造了 **上下文感知的语义表示**（不仅仅是一个实体名，而是“实体 + 它的邻接三元组”的描述）。

#### 3.3.2 Jaccard 结构相似度：`_compute_jaccard_matrix_vectorized`

- 利用稀疏邻接子矩阵 `sub_adj`：
  - 交集：`intersection = sub_adj.dot(sub_adj.T)`；
  - 每行和：`row_sums`；
  - 并集：`union = row_sums_i + row_sums_j - intersection`；
  - `jaccard = intersection / (union + 1e-9)`。
- 对角线强制为 1.0。

这一步从图结构角度衡量“节点邻居集合的重叠度”，反映局部拓扑相似性。

#### 3.3.3 综合相似度：`_compute_sim_matrix`

- 对给定一组 `level_nodes`：
  - 先通过 `get_triple_embeddings_batch` 得到 embedding；
  - L2 归一化后，内积得到 `semantic_sim_matrix`；
  - 同时计算结构 Jaccard 的 `structural_sim_matrix`；
  - 按 `struct_weight` 线性融合：

```python
sim_matrix = struct_weight * structural_sim_matrix + (1 - struct_weight) * semantic_sim_matrix
```

通过调节 `struct_weight`，可以在“结构一致性”和“语义相似度”之间平衡。

### 3.4 KMeans 初始聚类与细化：`detect_communities`

整体流程：

1. **初始聚类 `_fast_clustering`**
   - 在 embedding 空间上跑 `KMeans`：
     - 聚类数 `n_clusters` 约为 `len(nodes) // 10`，限定在 `[2, len(nodes)//2, 200]` 范围内。
   - 得到初始簇 `cluster_id -> [nodes]`。
2. **全局社区数控制**
   - 依据配置项或默认规则设置 `max_total_communities`：
     - 默认 ~ `len(nodes) / 3`，且在 `[5, 200]` 之间。
3. **按簇大小排序，大簇优先细化**
   - 对大簇调用 `_refine_cluster`：
     - 再跑一轮 `_fast_clustering` 得到更细子簇；
     - 为每个子簇选择一个“中心节点”作为代表；
     - 基于中心节点的相似度矩阵合并相似簇（受 `merge_threshold`、簇规模等限制）；
     - 重复迭代，直到不能再合并或只剩一个簇。
4. **社区数上限与残余簇处理**
   - 若当前社区数已到上限，则不再细化，直接把剩余簇视为社区；
   - 确保不会产生过多的小社区。

最终 `detect_communities(level_nodes)` 返回的是 `comm_id -> [node_ids]`，即最终社区划分。

### 3.5 TreeComm 社区报告：LLM 与超节点

#### 3.5.1 社区报告（脚本级输出）

在 `run_tree_comm_clustering` 中，我们将 TreeComm 聚类结果写出为：

- `*_tree_comm_communities.json`：包含：
  - `node_communities`: `[{node, community_id}, ...]`
  - `communities`: `{cid: [实体名,...]}`
  - `stats`: 节点/边/社区数。
- `*_tree_comm_communities.txt`：文本版社区列表。

并额外生成 **社区报告**：

- 调用 `detector.create_super_nodes(comm_to_nodes, level=4)`：
  - 内部：
    - 使用 `_build_batch_prompt` 构造批量社区信息；
    - 按同一份 `prompts/lowlevel_leiden_community_report.txt` 的规则调用 LLM；
    - 为每个社区生成一个 `community` 类型的超节点，写入：
      - `properties.name`（社区名称）
      - `properties.description`（社区摘要）
      - `properties.members`（成员实体名列表）
  - 同时为每个成员添加 `member_of` 边指向超节点。
- 根据这些超节点属性构造 JSON 报告 `*_tree_comm_community_report.json`：

```json
{
  "community_reports": [
    {
      "community_id": 0,
      "name": "...",
      "summary": "...",
      "members": [...],
      "size": N,
      "super_node_id": "comm_4_0"
    }
  ],
  "stats": {...}
}
```

#### 3.5.2 关键词节点（可选）

`create_super_nodes_with_keywords` 在社区超节点基础上：

- 通过 `extract_keywords_from_community` 基于度数 + 语义相似度选出代表性实体；
- 为这些实体创建 `keyword` 节点，并连边：
  - `keyword -> super_node (keyword_of)`
  - `member -> keyword (represented_by/kw_filter_by)`

这部分更多用于构建**上层可视化/检索图谱**，而非当前对比的底层社区聚类核心。

---

## 四、两种方案的主要区别与对比建议

### 4.1 聚类依据的差异

- **Leiden (`community_clustering.py`)**
  - 纯基于 **图结构**（度分布、连通性、边权等）。
  - 优点：
    - 算法成熟、鲁棒，复杂度较低；
    - 不依赖 embedding 或 LLM，结果更可控、可复现；
    - 对“拓扑结构清晰”的图效果好（如清晰模块化结构）。
  - 局限：
    - 无法感知“语义相似但拓扑距离较远”的节点；
    - 对抽取噪声、三元组稀疏等情况较敏感。

- **TreeComm (`tree_comm.py` / `FastTreeComm`)**
  - 综合 **三元组语义 embedding** 与 **结构 Jaccard 相似**。
  - 优点：
    - 能发现“语义趋同”的社区，即使图结构稍显分散；
    - 通过 `struct_weight` 可在“结构 vs 语义”之间灵活权衡；
    - 初始 KMeans + 细化合并的流程，可自适应社区数与形状。
  - 局限：
    - 依赖高质量 embedding 模型与算力（GPU 更佳）；
    - 工作流复杂、调参较多，出错点也更多；
    - 结果受 embedding 质量与三元组文本构造影响较大。

### 4.2 社区报告生成的差异

当前版本中，两者都共用同一份 prompt：  
`prompts/lowlevel_leiden_community_report.txt`，因此 **LLM 行为尽可能统一**。

差异主要来自**聚类输入给 LLM 的“社区内容”**不同：

- Leiden 报告输入：
  - 成员实体 + 内部常见关系（`top_relations`）；
  - 强调“结构上相近的实体簇”。
- TreeComm 报告输入：
  - 成员实体（以及 center 信息）是通过语义+结构双重感知聚出的社区；
  - 社区内部往往语义上更集中，LLM 更容易总结出主题。

### 4.3 实践对比建议

1. **先用 Leiden 得到一个“结构基线”**：
   - 查看 `*_communities.json` 中的社区大小分布；
   - 阅读 `*_community_report.json/.txt` 中 LLM 生成的主题。
2. **再用 TreeComm 在相同底层图上聚类**：
   - 对比 `*_tree_comm_communities.json` 中同一实体的社区变化；
   - 对比 `*_tree_comm_community_report.json` 中 `name/summary` 的主题集中度。
3. 重点观察：
   - 某些关键实体（如具体电解液组分、SEI/CEI 相关实体）在两种聚类下的邻居是否更符合直觉；
   - TreeComm 是否能更好地把“语义上类似但拓扑稍远”的节点聚合在一起。

---

## 五、小结

- **Leiden 方案**：  
  - 简洁稳定、纯拓扑；适合作为“结构基线”。
- **TreeComm 方案**：  
  - 工作流复杂但更“智能”，结合语义与结构；适合在科研语料/图谱下挖掘主题社区。
- 两者在本项目中已通过：
  - 统一的底层实体图谱输入；
  - 统一的 LLM prompt 文件；
  实现了较为公平的对比环境，便于你深入分析“图结构聚类 vs 语义增强聚类”的差异。

