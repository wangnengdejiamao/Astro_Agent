## 端到端知识图谱构建管线说明

本文档说明 `prompt2graph` 项目中，从原始语料到“多阶段抽取 + 实体消歧 + 多文档合并”的**端到端流程**，以及关键脚本、函数和参数含义。

---

## 整体流程概览

整体管线由 `run_end2end_pipeline.py` 串联实现，通过 **YAML 配置文件** 管理参数。  
典型输入为原始语料 JSON（如 `input/paper_mini/corpus_cleaned.json`），最终输出为消歧后的知识图谱（如 `output/paper_mini/multi_stage_deduplicated.json`）。

### 流程图

```text
┌───────────────────────────────────────────────────────────────┐
│                        配置加载 (YAML)                        │
│   - 读取 configs/*.yml                                        │
│   - 切分为 dataset / schema / extraction / deduplication /    │
│     output 等配置段                                           │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                    读取原始语料 (dataset)                     │
│   - 加载 corpus_cleaned.json                                  │
│   - 语料为 [ {title, text}, ... ]                             │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                 按篇拆分语料并生成 chunks                     │
│   每篇文档:                                                   │
│     - 写出单文档 corpus_doc_XXXX.json                         │
│     - 调用 get_chunks.get_chunks → chunks_doc_XXXX.txt        │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                 构建低层图谱 (extraction)                     │
│   对每篇文档:                                                 │
│     - 如果 use_staged_extraction=True:                        │
│         调用 build_lowlevel_graph(多阶段，schema+staged       │
│         prompts)                                              │
│     - 否则 (单阶段):                                          │
│         使用 prompt_path / prompt_name / prompts/default.txt  │
│         调用 build_lowlevel_graph(单阶段)                     │
│     - 输出 graph_doc_XXXX.json                                │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│             实体消歧 (deduplication, per-doc)                 │
│   对每篇文档的图谱:                                           │
│     - 调用 deduplicate_entities                               │
│         • enable_abbreviation / enable_cid                    │
│         • 可选 intermediate_output                            │
│     - 输出 graph_doc_XXXX_deduplicated.json                   │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                多文档图谱合并 (graph_merger)                  │
│   - 如果只有 1 篇文档:                                        │
│       直接将该文档消歧后的图谱重命名为最终输出               │
│   - 如果多篇文档:                                             │
│       逐篇调用 merge_graphs 进行实体/三元组合并               │
│       得到最终 graph_name (如 multi_stage_deduplicated.json)  │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│        可选：上层元图谱构建 (meta_graph.enable)               │
│   - 调用 meta_graph_builder.build_meta_graph                  │
│   - 输入：底层图谱 JSON + chunks.txt                          │
│   - 输出：*_meta.json                                         │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│     可选：元图谱质量评估 (meta_graph_evaluation.enable)       │
│   - 需 meta_graph 已构建；调用 meta_graph_evaluation          │
│   - 输入：meta_graph.json + 底层图谱 + chunks.txt             │
│   - 输出：*_meta_evaluation.json                              │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│          可选：维护 current_graph 并导入 Neo4j (output.neo4j) │
│   - 如果 enable=true:                                         │
│       • 使用 update_current_graph_and_import                  │
│         - clear_first=true: 覆盖 current_graph                │
│         - clear_first=false: 与 current_graph 增量合并        │
│       • 清空 Neo4j 并基于 current_graph 全量导入             │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│             社区聚类 (community_clustering.method)            │
│   - method=leiden: run_community_clustering                   │
│       → *_communities.json, *_community_report.json           │
│   - method=tree_comm: run_tree_comm_clustering                │
│       → *_tree_comm_communities.json, *_tree_comm_community_report.json │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│      可选：社区质量评估 (community_evaluation.enable)         │
│   - 对 Leiden 或 TreeComm 社区报告做 LLM 五维打分             │
│   - 输出：*_community_report_evaluation.json 或               │
│          *_tree_comm_community_report_evaluation.json         │
└───────────────────────────────────────────────────────────────┘
```

**处理步骤：**

1. **逐文档拆分语料**
   - 文件：`run_end2end_pipeline.py`
   - 函数：`run_end2end_pipeline`
   - 动作：将输入的 `corpus_cleaned.json` 读入后，判断文档数量：
     - 单篇语料：直接对该文档构图 + 消歧，得到最终图谱；
     - 多篇语料：为每一篇单独构图 + 消歧，最后用图谱合并模块进行汇总。

2. **文本切分为 chunks**
   - 文件：`get_chunks.py`
   - 函数：`get_chunks(corpus_path, dataset_name, output_path, ...)`
   - 动作：对每篇文档（写成单独的 `corpus_doc_XXXX.json`）进行智能切分，生成 chunk 文件：
     - 输入：单文档 corpus 文件路径；
     - 输出：`output/{dataset_name}/chunks_doc_XXXX.txt`，其中每行形如 `id: {chunk_id}\tChunk: {title+text}`。

3. **构建低层知识图谱（多阶段提取）**
   - 文件：`get_lowlevel_graph.py`
   - 函数：`build_lowlevel_graph(...)`
   - 调用入口：`run_end2end_pipeline.py` 中的 `build_lowlevel_graph(...)`
   - 动作：对上一步生成的 chunk 文件进行多阶段抽取，生成低层图谱：
     - 默认 **开启多阶段提取**：`use_staged_extraction=True`
     - 可选开启第 4 阶段验证：`enable_stage4_validation`
   - 输出：`output/{dataset_name}/graph_doc_XXXX.json`

4. **实体消歧（去重）**
   - 文件：`entity_deduplication.py`
   - 主要类/函数：
     - 类：`EntityDeduplicator`
     - 便捷函数：`deduplicate_entities(graph_path, output_path, ..., db_path)`
   - 调用入口：`run_end2end_pipeline.py` 中的 `deduplicate_entities(...)`
   - 动作：
     1. 读取单文档图谱 `graph_doc_XXXX.json`；
     2. 基于 PubChem 本地库查询 CID，并结合 abbreviation 信息；
     3. 对同一 CID / 同义实体进行合并；
     4. 删除 abbreviation 相关的三元组；
     5. 用合并后的实体更新整张图谱。
   - 输出：`output/{dataset_name}/graph_doc_XXXX_deduplicated.json`

5. **多文档图谱合并（仅当语料包含多篇时）**
   - 文件：`graph_merger.py`
   - 主要类/函数：
     - 类：`GraphMerger`
     - 便捷函数：`merge_graphs(graph_path1, graph_path2, output_path)`
   - 调用入口：`run_end2end_pipeline.py` 中的 `merge_graphs(...)`
   - 动作：
     1. 对多篇文档的“消歧后单文档图谱”两两合并；
     2. 基于 abbreviation 和 CID 再次对跨文档实体进行对齐和合并；
     3. 对相同三元组合并其 `source` / `evidence` / `chunk_id` 字段。
   - 最终输出：`output/{dataset_name}/multi_stage_deduplicated.json`

6. **上层元图谱构建（可选）**
   - 文件：`meta_graph_builder.py`
   - 函数：`build_meta_graph(base_graph_path, chunks_path, output_path, ...)`
   - 调用入口：`run_end2end_pipeline.py` 中，当配置 `meta_graph.enable: true` 时触发。
   - 动作：基于底层图谱和 chunks 调用 LLM 抽取三元组间语义关系，输出 `*_meta.json`。

7. **元图谱质量评估（可选）**
   - 文件：`meta_graph_evaluation.py`
   - 函数：`run_meta_graph_evaluation(meta_graph_path, base_graph_path, chunks_path, ...)`
   - 调用入口：`run_end2end_pipeline.py` 中，当 `meta_graph.enable` 且 `meta_graph_evaluation.enable` 时触发。
   - 动作：对 meta-graph 每条 meta-edge 做 source_alignment 打分，输出 `*_meta_evaluation.json`。

8. **将最终图谱导入 Neo4j（可选）**
   - 文件：`graph_to_neo4j.py`
   - 主要函数：
     - `load_triples(path)`：加载三元组 JSON
     - `clear_database(driver)`：清空 Neo4j 数据库
     - `run_import(triples, driver, batch_size=500)`：批量导入三元组
   - 调用入口：`run_end2end_pipeline.py` 中，当配置 `output.neo4j.enable: true` 时自动触发。
   - 动作：
     1. 读取最终合并后的图谱 JSON；
     2. 可选地清空 Neo4j 当前图谱；
     3. 按批次将节点和关系导入 Neo4j。

9. **社区聚类**
   - 由 `community_clustering.method` 控制：
     - **method=leiden**：`community_clustering.run_community_clustering` → `*_communities.json`、`*_community_report.json`
     - **method=tree_comm**：`tree_comm.run_tree_comm_clustering` → `*_tree_comm_communities.json`、`*_tree_comm_community_report.json`

10. **社区质量评估（可选）**
   - 文件：`community_evaluation.py`
   - 函数：`run_community_evaluation(report_path, chunks_path, output_path, ...)`
   - 调用入口：`run_end2end_pipeline.py` 中，当 `community_evaluation.enable: true` 时触发。
   - 动作：对 Leiden 或 TreeComm 社区报告做 LLM 五维打分，输出 `*_evaluation.json`。

---

## 端到端脚本：`run_end2end_pipeline.py`

### 核心函数：`run_end2end_pipeline`

内部仍然由 Python 函数 `run_end2end_pipeline(...)` 实现各步骤，但日常使用推荐通过 **YAML 配置 + 命令行入口** 来调用：

- 命令行入口：`python run_end2end_pipeline.py path/to/config.yml`
- 配置文件中对各步骤参数进行分层管理（见下文“配置文件结构”）。

函数本身现在接收分段配置 dict，而不是一长串扁平参数，签名大致为：

```python
def run_end2end_pipeline(
    dataset_cfg: Dict[str, Any],
    schema_cfg: Dict[str, Any],
    extraction_cfg: Dict[str, Any],
    dedup_cfg: Dict[str, Any],
    output_cfg: Dict[str, Any],
    meta_graph_cfg: Optional[Dict[str, Any]] = None,
    meta_graph_evaluation_cfg: Optional[Dict[str, Any]] = None,
    community_clustering_cfg: Optional[Dict[str, Any]] = None,
    community_evaluation_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    ...
```

各个配置 dict 与 YAML 的层次一一对应：

- `dataset_cfg` ← `config['dataset']`
- `schema_cfg` ← `config['schema']`
- `extraction_cfg` ← `config['extraction']`
- `dedup_cfg` ← `config['deduplication']`
- `output_cfg` ← `config['output']`
- `meta_graph_cfg` ← `config['meta_graph']`
- `meta_graph_evaluation_cfg` ← `config['meta_graph_evaluation']`
- `community_clustering_cfg` ← `config['community_clustering']`
- `community_evaluation_cfg` ← `config['community_evaluation']`

这些配置的具体 key 在下一节详细说明。

---

## 配置文件结构（YAML）

推荐在 `configs/` 目录下创建管线配置文件，例如：`configs/paper_mini_pipeline.yml`。  
一个完整的配置结构如下（项目中已有示例：`configs/example_pipeline.yml`）：

```yaml
logging:
  # 日志级别：INFO（默认，只显示关键信息）或 DEBUG（显示详细耗时和token信息）
  level: INFO

dataset:
  # 数据集名称，用于组织输出目录：output/{name}/...
  name: paper_mini
  # 原始语料路径（相对路径会以 PROJECT_DIR 或当前项目根目录为基准解析）
  corpus_path: input/paper_mini/corpus_cleaned.json
  # 是否已切好 chunk：true 时跳过第一阶段切分，使用 output/{name}/chunks_doc_{doc_id}.txt
  is_chunked: false

schema:
  # 二选一：
  # 1）指定 schema 名称（对应 schemas/{name}.json）
  name: 260114
  # 2）或指定完整路径（优先级高于 name）
  # path: configs/custom_schema.json

extraction:
  # 是否使用多阶段提取（默认 true）
  use_staged_extraction: true
  # 多阶段提取时，可选开启阶段 4 验证（非常耗 token）
  use_stage4_validation: false
  # 多阶段提取时，是否保存4个阶段的中间结果（按 chunk 输出 JSON）
  save_stage_outputs: false

  # 阶段 4 验证参数（仅当 use_stage4_validation: true 时生效）
  # stage4:
  #   min_triple_score: 0.5
  #   min_node_score: 0.5
  #   use_chunk_scoring: true
  #   use_node_accuracy_scoring: true
  #   use_triple_support_scoring: true

  # 单阶段提取下的 prompt 配置（当 use_staged_extraction: false 时生效）：
  # 优先级：prompt_path > prompt_name > prompts/default.txt
  # prompt_name: default_single_stage
  # prompt_path: prompts/custom_prompt.txt

deduplication:
  # 是否启用基于 abbreviation 的实体消歧
  enable_abbreviation: true
  # 是否启用基于 CID 的实体消歧（需要本地 PubChem 数据库）
  enable_cid: true
  # 是否输出中间结果（*_intermediate.json），用于调试/分析
  intermediate_output: false
  # PubChem 本地数据库路径
  pubchem_db_path: pubchem_names_full.db

# 可选：上层元图谱构建
meta_graph:
  enable: false
  chunks_path: null
  prompt_path: null
  max_chars_per_chunk: 2000
  max_total_chunk_chars: 50000

# 可选：元图谱质量评估（需 meta_graph.enable=true）
meta_graph_evaluation:
  enable: false
  chunks_path: null
  output_path: null
  prompt_path: null
  max_chars_per_chunk: 2000
  max_total_chunk_chars: 50000

# 社区聚类（必选）；method: leiden | tree_comm
community_clustering:
  method: leiden
  # Leiden 专用
  max_cluster_size: 1000
  use_lcc: false
  seed: 42
  llm_prompt_path: null
  # TreeComm 专用
  embedding_model: all-MiniLM-L6-v2
  struct_weight: 0.3

# 可选：社区质量评估
community_evaluation:
  enable: false
  chunks_path: null
  output_path: null
  prompt_path: null
  max_chars_per_chunk: 2000
  max_total_chunk_chars: 50000

output:
  # 最终输出图谱文件名：output/{dataset.name}/{graph_name}
  graph_name: multi_stage_deduplicated.json

  # 可选：将最终图谱导入 Neo4j
  neo4j:
    # 是否启用 Neo4j 导入
    enable: false
    # Neo4j 连接信息
    uri: bolt://localhost:7687
    user: neo4j
    password: password
    # 批处理大小
    batch_size: 500
    # 导入前是否清空 Neo4j 中现有图谱
    clear_first: true
```

**各 section 语义概括：**

- **`logging`**：
  - **`level`**：日志级别，可选值：
    - `INFO`（默认）：只显示关键信息（如处理完成、成功/失败统计、最终输出路径等）
    - `DEBUG`：显示详细信息（包括每个 LLM API 调用的耗时、token 消耗、进度百分比等）
    - 其他标准级别：`WARNING`、`ERROR`、`CRITICAL`

- **`dataset`**：
  - **`name`**：数据集名称，用来确定输出目录 `output/{name}/`。
  - **`corpus_path`**：原始语料 JSON 路径，格式为 list\[{"title", "text"}\]。
  - **`is_chunked`**：是否已切好 chunk（默认 `false`）。设为 `true` 时跳过第一阶段 chunk 切分，直接使用已有文件 `output/{name}/chunks_doc_{doc_id}.txt`；若文件不存在会报错。

- **`schema`**：
  - **`name`**：schema 名称（对应 `schemas/{name}.json`）。
  - **`path`**：schema 完整路径（可选，优先级高于 `name`）。

- **`extraction`**：
  - **`use_staged_extraction`**：是否使用多阶段提取（默认 true）。
  - **`use_stage4_validation`**：多阶段提取时是否启用阶段 4 验证。
  - **`save_stage_outputs`**：多阶段提取时是否保存 4 个阶段的中间结果。
    - `false`（默认）：不落盘，仅保留最终图谱；
    - `true`：为每个 chunk 在输出目录下的 `staged/` 子目录中生成：
      - `chunkId_stage1.json`（实体识别结果）
      - `chunkId_stage2.json`（关系抽取结果）
      - `chunkId_stage3.json`（属性抽取结果）
      - `chunkId_stage4.json`（可选验证与过滤结果，仅当启用阶段4时生成）
  - **`stage4`**（仅当 `use_stage4_validation: true` 时生效）：阶段 4 验证参数。
    - **`min_triple_score`**：三元组支持度最低分数阈值（默认 0.5，低于此分数将被删除）。
    - **`min_node_score`**：节点准确性最低分数阈值（默认 0.5，低于此分数将被删除）。
    - **`use_chunk_scoring`**：是否使用 chunk 级别的 LLM 打分检查 bad_cases（默认 true）。
    - **`use_node_accuracy_scoring`**：是否启用节点准确性打分（默认 true）。
    - **`use_triple_support_scoring`**：是否启用三元组支持度打分（默认 true）。
  - **`prompt_name` / `prompt_path`**：
    - 当 `use_staged_extraction: false` 时生效；
    - prompt 选择优先级：`prompt_path` > `prompt_name` > `prompts/default.txt`。

- **`deduplication`**：
  - **`enable_abbreviation`**：是否启用基于 abbreviation 的实体消歧。
  - **`enable_cid`**：是否启用基于 CID 的实体消歧。
  - **`intermediate_output`**：是否输出中间结果（`*_intermediate.json`）。
  - **`pubchem_db_path`**：PubChem 本地数据库路径。

- **`meta_graph`**（可选）：
  - **`enable`**：是否构建上层元图谱（默认 false）。
  - **`chunks_path`**：chunks 路径，不填则使用首篇 `chunks_doc_0000.txt`。
  - **`prompt_path`**：meta-graph 抽取 Prompt 路径。
  - **`max_chars_per_chunk`** / **`max_total_chunk_chars`**：chunk 长度限制。

- **`meta_graph_evaluation`**（可选）：
  - **`enable`**：是否对 meta-graph 做质量评估（需 meta_graph.enable=true，默认 false）。
  - **`chunks_path`**、**`output_path`**、**`prompt_path`**：评估相关路径。
  - **`max_chars_per_chunk`** / **`max_total_chunk_chars`**：chunk 长度限制。

- **`community_clustering`**：
  - **`method`**：社区聚类方式，`leiden`（默认）或 `tree_comm`。
  - Leiden 专用：**`max_cluster_size`**、**`use_lcc`**、**`seed`**、**`llm_prompt_path`**。
  - TreeComm 专用：**`embedding_model`**、**`struct_weight`**。

- **`community_evaluation`**（可选）：
  - **`enable`**：是否对社区报告做 LLM 五维打分（默认 false）。
  - **`chunks_path`**、**`output_path`**、**`prompt_path`**：评估相关路径。

- **`output`**：
  - **`graph_name`**：最终图谱文件名，实际路径为：`output/{dataset.name}/{graph_name}`。
  - **`neo4j`**：Neo4j 相关配置（可选）：
    - `enable`：是否启用将最终图谱导入 Neo4j。
    - `current_graph`：维护当前 Neo4j 图谱状态的 JSON 文件路径，用于增量合并。
    - `uri` / `user` / `password`：Neo4j 连接信息。
    - `batch_size`：导入时每个事务处理的三元组数量。
    - `clear_first`：是否重置 `current_graph`（true: 本次结果覆盖；false: 与 `current_graph` 增量合并）。

---

## 步骤与对应函数映射

- **chunk 切分**
  - 文件：`get_chunks.py`
  - 函数：`get_chunks(corpus_path, dataset_name, output_path, datasets_no_chunk=None)`
  - 调用位置：`run_end2end_pipeline.py` 中的：
    - `get_chunks(single_corpus_path, f"{dataset_name}_doc_{doc_id}", chunk_output_path)`

- **低层图谱构建（多阶段提取）**
  - 文件：`get_lowlevel_graph.py`
  - 函数：`build_lowlevel_graph(chunk_path, output_graph_path, config=None, **kwargs)`
  - 推荐使用 **config 字典** 传入参数，避免超长入参。config 可包含：`schema_path`、`schema_content`、`prompt_path`、`prompt_content`、`use_staged_extraction`、`enable_stage4_validation`、`prompt_paths`、`pubchem_db_path`、`save_stage_outputs`、`stage4`（子字典）等。
  - 调用位置：`run_end2end_pipeline.py` 中构建 `graph_config` 后调用 `build_lowlevel_graph(chunk_path, output_graph_path, config=graph_config)`。

- **实体消歧**
  - 文件：`entity_deduplication.py`
  - 便捷函数：`deduplicate_entities(graph_path, output_path=None, config=None, **kwargs)`
  - 推荐使用 **config 字典** 传入参数。config 可包含：`output_path`、`intermediate_output`、`enable_abbreviation`、`enable_cid`、`pubchem_db_path`（或 `db_path`）。
  - 调用位置：`run_end2end_pipeline.py` 中构建 `dedup_config` 后调用 `deduplicate_entities(graph_path, config=dedup_config)`。

- **多图谱合并**
  - 文件：`graph_merger.py`
  - 便捷函数：`merge_graphs(graph_path1, graph_path2, output_path)`
  - 调用位置：`run_end2end_pipeline.py` 中的：
    - `current_path = merge_graphs(current_path, next_path, merged_output)`

- **上层元图谱构建（可选）**
  - 文件：`meta_graph_builder.py`
  - 函数：`build_meta_graph(base_graph_path, chunks_path, output_path, ...)`
  - 调用位置：`run_end2end_pipeline.py` 中，当 `meta_graph.enable: true` 时调用。

- **元图谱质量评估（可选）**
  - 文件：`meta_graph_evaluation.py`
  - 函数：`run_meta_graph_evaluation(meta_graph_path, base_graph_path, chunks_path, ...)`
  - 调用位置：`run_end2end_pipeline.py` 中，当 meta_graph 已构建且 `meta_graph_evaluation.enable: true` 时调用。

- **导入 Neo4j（可选）**
  - 文件：`graph_to_neo4j.py`
  - 函数：`load_triples(path)`, `clear_database(driver)`, `run_import(triples, driver, batch_size)`
  - 调用位置：`run_end2end_pipeline.py` 中的：
    - 在最终图谱生成后，如果 `output.neo4j.enable: true`，则：
      - 更新/生成 `output.neo4j.current_graph` 文件：
        - 如果 `clear_first: true`，则用本次图谱 **覆盖** `current_graph`；
        - 如果 `clear_first: false`，则调用 `merge_graphs` 将 `current_graph` 与本次图谱 **增量合并**；
      - 清空 Neo4j 数据库（保证与 `current_graph` 一致）；
      - 读取 `current_graph` 并按 `output.neo4j.batch_size` 批量调用 `run_import` 导入节点和关系。

- **社区聚类**
  - `method=leiden`：`community_clustering.run_community_clustering` → `*_communities.json`、`*_community_report.json`
  - `method=tree_comm`：`tree_comm.run_tree_comm_clustering` → `*_tree_comm_communities.json`、`*_tree_comm_community_report.json`

- **社区质量评估（可选）**
  - 文件：`community_evaluation.py`
  - 函数：`run_community_evaluation(report_path, chunks_path, output_path, ...)`
  - 调用位置：`run_end2end_pipeline.py` 中，当 `community_evaluation.enable: true` 时，对 Leiden 或 TreeComm 社区报告做评估。

---

## 命令行使用示例（基于 YAML 配置）

### 1. 最常用场景：多阶段提取 + 实体消歧 + 自动合并

先编写配置文件（例如 `configs/paper_mini_pipeline.yml`，可参考 `configs/example_pipeline.yml`），内容类似：

```yaml
logging:
  level: INFO  # 或 DEBUG（显示详细耗时和token信息）

dataset:
  name: paper_mini
  corpus_path: input/paper_mini/corpus_cleaned.json
  is_chunked: false  # true 时跳过 chunk 切分，使用已有 chunks_doc_XXXX.txt

schema:
  name: 260114

extraction:
  use_staged_extraction: true
  use_stage4_validation: false

deduplication:
  enable_abbreviation: true
  enable_cid: true
  intermediate_output: false
  pubchem_db_path: pubchem_names_full.db

output:
  graph_name: multi_stage_deduplicated.json
  neo4j:
    enable: true
    current_graph: current_graph.json
    uri: bolt://localhost:7687
    user: neo4j
    password: password
    batch_size: 500
    clear_first: true
```

然后在项目根目录运行：

```bash
cd prompt2graph

python run_end2end_pipeline.py configs/paper_mini_pipeline.yml
```

含义：

- 输入语料：`input/paper_mini/corpus_cleaned.json`
- schema：`schemas/260114.json`
- 输出目录：`output/paper_mini/`
- 最终结果：`output/paper_mini/multi_stage_deduplicated.json`
 - 如果 `output.neo4j.enable: true`，还会在流程末尾自动将该图谱导入到配置的 Neo4j 实例中。

### 2. 单阶段提取 + 自定义 prompt

```yaml
extraction:
  use_staged_extraction: false
  # 使用自定义 prompt 文件
  prompt_path: prompts/my_single_stage_prompt.txt
```

运行同样的命令：

```bash
python run_end2end_pipeline.py configs/paper_mini_single_stage.yml
```

如果在单阶段配置中既不提供 `prompt_path` 也不提供 `prompt_name`，则会自动使用 `prompts/default.txt`。

### 3. 控制日志详细程度

默认情况下（`logging.level: INFO`），终端只显示关键信息：
- 每篇文档的处理进度（`[Doc X/Y]`）
- 成功/失败统计
- 最终输出路径
- 实体消歧和合并的关键步骤

如果需要查看详细的耗时和 token 消耗信息，可以设置：

```yaml
logging:
  level: DEBUG
```

DEBUG 级别会显示：
- 每个 LLM API 调用的详细耗时（构建请求、HTTP 请求、解析响应）
- 每个 chunk 的处理进度百分比和 ETA
- 三元组去重、图谱构建的详细步骤
- Token 消耗统计

**建议**：
- 日常使用：`level: INFO`（简洁输出）
- 性能分析/调试：`level: DEBUG`（详细输出）

---

## 与已有脚本的关系

- `prompt2graph.py`
  - 封装了较早期的调用方式（基于 `dataset_name` + `schema_name` + `prompt_name`）；
  - 当前端到端脚本直接调用其中的底层函数（`get_chunks` + `build_lowlevel_graph`），
    并在此基础上追加实体消歧和多文档合并能力。

- `entity_deduplication.py`
  - 原本可单独对一张已构建好的图谱做实体消歧；
  - 在端到端管线中，对每篇文档的图谱都调用一次；
  - 保证最终合并前，每张单文档图谱已经是“局部消歧”版本。

- `graph_merger.py`
  - 用于多张“已消歧图谱”的跨文档合并；
  - 在端到端管线中自动判断是否需要调用：
    - 单篇语料：不调用；
    - 多篇语料：自动多轮 `merge_graphs`。

- `meta_graph_builder.py`
  - **可选下游步骤**：在底层图谱构建完成后，可单独调用以构建上层元图谱。
  - 输入：`base_graph_path`（底层图谱 JSON）、`chunks_path`（chunks.txt）。
  - 输出：以底层三元组为节点、LLM 抽取的语义关系为边的上层图谱。
  - 详见 `docs/meta_graph_design_overview.md`。

- `meta_graph_evaluation.py`
  - **可选下游步骤**：在 meta-graph 构建完成后，可单独调用以对 meta-edge 做 source_alignment 打分。
  - 输入：`meta_graph_path`、`base_graph_path`、`chunks_path`。
  - 输出：`*_meta_evaluation.json`，每条 meta-edge 对应一个 source_alignment 分。

- `community_clustering.py` / `tree_comm.py`
  - **必选下游步骤**：对合并后的图谱做社区聚类，由 `community_clustering.method` 控制调用哪个模块。
  - Leiden：`community_clustering.run_community_clustering` → `*_communities.json`、`*_community_report.json`
  - TreeComm：`tree_comm.run_tree_comm_clustering` → `*_tree_comm_communities.json`、`*_tree_comm_community_report.json`

- `community_evaluation.py`
  - **可选下游步骤**：对 Leiden 或 TreeComm 的社区报告做 LLM 五维打分。
  - 输入：社区 report JSON、chunks_path。
  - 输出：`*_evaluation.json`。

---

## 开发者扩展建议

- 如需修改 chunk 策略：
  - 修改 `get_chunks.py` 中的 `smart_split_text`、`CHUNK_*` 配置；
  - 端到端脚本不需要改动。

- 如需更换/扩展 schema：
  - 在 `schemas/` 下新增对应的 JSON；
  - 调用时传入新的 `--schema-name` 或 `--schema-path`。

- 如需关闭某些消歧策略（例如只用 abbreviation，不用 CID）：
  - 可以在 `run_end2end_pipeline.py` 中对 `deduplicate_entities(...)` 增加参数开关；
  - 或直接在下游脚本中调用 `deduplicate_entities` 并设置 `enable_cid=False`。

