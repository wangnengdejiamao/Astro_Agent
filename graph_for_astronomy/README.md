# Prompt2Graph —— 天文领域知识图谱构建工具

> 本项目是 Prompt2Graph 的精简迁移版，已剔除化学领域专用数据（PubChem 数据库、锂电文献等），适配**天文/天体物理**领域的知识图谱自动化构建。

## 核心能力

- **端到端管线**：从原始天文文献（`corpus_cleaned.json`）一键生成知识图谱
- **Schema 引导抽取**：通过自定义 Schema 控制实体类型（如 `Star`, `Galaxy`, `Nebula`, `Telescope`, `SpectralLine`）与关系
- **多阶段 LLM 抽取**：实体识别 → 关系抽取 → 属性抽取 → 可选验证，任务解耦、便于调优
- **实体消歧**：自动合并同义实体（缩写、别名）
- **社区聚类**：支持 Leiden / TreeComm 算法，发现文献中的主题社区
- **质量评估**：对元图谱与社区报告进行 LLM 多维打分
- **Neo4j 导入**：将最终图谱导入图数据库进行交互式查询与可视化
- **前端可视化**：自带轻量 HTML 前端，可本地预览图谱结构

---

## 目录结构

```
graph_for_astronomy/
├── configs/                    # 管线 YAML 配置示例
│   ├── simple_pipeline.yml     # 最简配置（推荐起步）
│   └── example_pipeline.yml    # 完整功能配置
├── input/                      # 输入语料目录
│   └── astronomy/              # 你的天文语料放这里
│       └── corpus_cleaned.json # 清洗后的文献语料（见下方格式）
├── output/                     # 输出目录（运行后自动生成）
├── prompts/                    # Prompt 模板（含多阶段抽取模板）
│   └── staged/                 # 多阶段各阶段 prompt
├── schemas/                    # Schema 定义（JSON）
│   └── astronomy_schema.json   # 建议：自定义天文 Schema
├── docs/                       # 项目文档（逻辑说明、设计文档）
├── frontend/                   # 可视化前端
│   └── index.html
├── graph_tools/                # 图谱工具函数
├── scorers/                    # 质量打分模块
├── staged_extraction/          # 多阶段抽取实现
├── utils/                      # 公共工具（LLM 调用、日志等）
├── run_end2end_pipeline.py     # 【推荐】端到端管线入口
├── prompt2graph.py             # 底层图谱构建主入口
├── get_chunks.py               # 文本分块
├── get_lowlevel_graph.py       # 单/多阶段图谱构建
├── entity_deduplication.py     # 实体消歧
├── graph_merger.py             # 多文档图谱合并
├── meta_graph_builder.py       # 二级元图谱构建
├── community_clustering.py     # Leiden 社区聚类
├── tree_comm.py                # TreeComm 社区检测
├── graph_to_neo4j.py           # Neo4j 导入
└── README.md                   # 本文件
```

---

## 快速开始

### 1. 环境准备

```bash
# 建议 Python 3.10+
pip install -r requirements.txt
```

> 若缺少 `requirements.txt`，请根据报错手动安装常见依赖：`openai`, `neo4j`, `networkx`, `leidenalg`, `python-dotenv`, `pyyaml`, `numpy`, `graspologic`（可选，TreeComm 需要）。

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填写你的 API Key 与 Neo4j 连接（可选）：

```bash
cp .env.example .env
```

编辑 `.env`：

```ini
# LLM API（以 OpenAI 兼容格式为例）
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 项目根目录（可选，默认当前目录）
PROJECT_DIR=.

# Neo4j（如需导入图数据库）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

### 3. 准备天文语料

将你的文献整理为 `input/astronomy/corpus_cleaned.json`，格式如下：

```json
[
  {
    "doc_id": "doc_0000",
    "title": "JWST Observations of Early Galaxy Formation",
    "abstract": "We report deep JWST imaging...",
    "content": "Full text of the paper or its sections..."
  },
  {
    "doc_id": "doc_0001",
    "title": "Spectroscopic Analysis of Type Ia Supernovae",
    "abstract": "...",
    "content": "..."
  }
]
```

- 每个元素为一篇文献，至少包含 `doc_id` 与 `content`
- `content` 为待抽取的正文内容（可以是完整论文、章节或摘要集合）

### 4. 定义天文 Schema（关键！）

在 `schemas/` 下新建 `astronomy_schema.json`，定义你想抽取的实体类型、关系与属性。例如：

```json
{
  "entity_types": [
    "Star",
    "Galaxy",
    "Nebula",
    "Exoplanet",
    "Telescope",
    "SpectralLine",
    "ChemicalElement",
    "AstronomicalObject"
  ],
  "relation_types": [
    "observed_by",
    "contains",
    "emits",
    "orbits",
    "classified_as",
    "located_in",
    "associated_with"
  ],
  "attributes": {
    "Star": ["mass", "temperature", "spectral_type", "distance"],
    "Galaxy": ["redshift", "morphology", "luminosity"],
    "Exoplanet": ["orbital_period", "radius", "mass"]
  }
}
```

Schema 文件名（不含 `.json`）即配置中的 `schema.name`。

### 5. 修改管线配置

复制 `configs/simple_pipeline.yml` 为 `configs/astronomy.yml`，修改以下关键项：

```yaml
logging:
  level: INFO

dataset:
  name: astronomy
  corpus_path: input/astronomy/corpus_cleaned.json
  is_chunked: false

schema:
  name: astronomy_schema   # 对应 schemas/astronomy_schema.json

extraction:
  use_staged_extraction: true
  use_stage4_validation: false
  save_stage_outputs: false

deduplication:
  enable_abbreviation: true
  enable_cid: false        # 天文领域无需 PubChem CID
  intermediate_output: false

meta_graph:
  enable: true

community_clustering:
  enable: true
  # method: leiden 或 tree_comm

output:
  graph_name: astronomy_graph.json
  neo4j:
    enable: false          # 需要 Neo4j 时改为 true
```

### 6. 运行管线

```bash
python run_end2end_pipeline.py configs/astronomy.yml
```

运行结束后，输出位于 `output/astronomy/{timestamp}/`，包含：

- `chunks_doc_*.txt`：分块后的文本
- `*_deduplicated.json`：消歧后的底层知识图谱
- `*_meta.json`：二级元图谱（若开启）
- `*_communities.json` / `*_community_report.json`：社区聚类结果与报告

### 7. 可视化（可选）

直接用浏览器打开 `frontend/index.html`，或导入 Neo4j 后用 Neo4j Browser 查看。

---

## 天文领域适配要点

| 模块 | 原化学领域 | 天文领域调整 |
|------|-----------|-------------|
| **Schema** | `Salt`, `Solvent`, `Additive` | 改为 `Star`, `Galaxy`, `Nebula`, `Telescope` 等 |
| **Prompt** | 化学术语、反应式 | 需替换为天文术语、观测关系；建议基于 `prompts/staged/` 修改 |
| **PubChem** | 查询化合物 CID | **已删除**，天文领域无需 |
| **语料** | 电化学论文 DOCX | 改为天文文献 JSON，字段保留 `doc_id`, `content` |
| **消歧** | 化学缩写映射 | 保留缩写消歧，需在天文 prompt 中定义常见缩写（如 `SN` → `Supernova`） |

> **提示**：如果 LLM 在天文实体识别上表现不佳，优先调优 `prompts/staged/stage1_entity_recognition.txt` 中的示例与约束，而非修改代码。

---

## 分步调用（高级）

若不想用端到端管线，可单独调用各模块：

```bash
# 1. 文本分块
python get_chunks.py input/astronomy/corpus_cleaned.json astronomy

# 2. 底层图谱（多阶段）
python prompt2graph.py astronomy -s astronomy_schema -o graph.json --staged --chunked

# 3. 实体消歧
python entity_deduplication.py output/astronomy/.../graph_doc_0000.json -o graph_dedup.json

# 4. 社区聚类
python community_clustering.py graph_dedup.json -o output/astronomy/.../

# 5. 导入 Neo4j
python graph_to_neo4j.py graph_dedup.json
```

---

## 注意事项

1. **不要上传 `.env` 文件**：内含 API Key，已提供 `.env.example` 作为模板。
2. **大文件已剔除**：本压缩包不含原始 `.git`、PubChem 数据库（>2 GB）、化学文献输入与历史输出。
3. **首次运行建议**：先用 1–2 篇短文测试 `configs/simple_pipeline.yml`，确认 Schema 与 Prompt 效果后再跑大批量文献。
4. **TreeComm 依赖**：若使用 `method: tree_comm`，需安装 `graspologic` 与 `sentence-transformers`，并在配置中指定 embedding 模型。
5. **LLM 费用**：多阶段抽取每 chunk 调用 3–4 次 LLM，大批量文献前请估算 token 消耗。

---

## 许可证与来源

本项目基于 Prompt2Graph 原仓库精简，移除化学领域专用数据后适配通用科学文献场景。原项目内部使用，请遵守相关数据合规要求。
