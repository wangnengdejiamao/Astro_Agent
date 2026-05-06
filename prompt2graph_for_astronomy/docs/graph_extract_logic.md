# 知识图谱提取串讲报告

本文档整合单阶段与多阶段两种图谱提取方式，涵盖设计思路、数据流、使用方式及实现要点。

---

## 一、概述

### 1.1 目标

从科学论文等文本中提取**实体**、**关系**（三元组）与**属性**，构建结构化知识图谱。输出包含 `source`（原文片段）与 `evidence`（LLM 推理），便于追溯与验证。

### 1.2 两种提取方式

| 方式 | 说明 | 适用场景 |
|------|------|----------|
| **单阶段提取** | 一个长 prompt 一次性完成实体识别、关系提取、属性提取 | 快速验证、简单语料 |
| **多阶段提取** | 拆成 4 个阶段，每阶段独立 prompt，顺序执行 | 追求质量、复杂语料、需可维护性 |

### 1.3 整体流程

```
文本 → 分块 (chunks) → 提取（单阶段 or 多阶段）→ 去重 (triple_deduplicate) → 格式化输出
```

---

## 二、单阶段提取

### 2.1 流程

```
┌─────────────────────────────────────────────────────────┐
│  单一 Prompt（如 260120_v3.txt，约 352 行）             │
│  • 实体规范化规则                                        │
│  • 缩写/同义词识别                                       │
│  • 关系提取规则                                          │
│  • 属性提取规则                                          │
│  • 三元组方向、source/evidence 要求                     │
│  → 一次 API 调用，输出 triples + attributes + entity_types │
└─────────────────────────────────────────────────────────┘
```

### 2.2 特点

- **优点**：实现简单，每 chunk 仅 1 次 LLM 调用，延迟低。
- **缺点**：Prompt 过长，关键规则易被忽略；实体、关系、属性耦合，难以单独优化与排错。

### 2.3 使用方式

```python
from prompt2graph import prompt2graph

output_path = prompt2graph(
    dataset_name="paper_mini",
    schema_name="260114",
    prompt_name="260120",
    is_chunked=True,
    output_graph_name="graph_single.json",
    use_staged_extraction=False
)
```

**必需参数**：`schema_name` 或 `schema_content`；`prompt_name` 或 `prompt_content`。

---

## 三、多阶段提取

### 3.1 流程概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入：Text Chunk + Schema                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段1：实体识别和规范化 (Entity Recognition & Canonicalization) │
│  输出：entities[] + abbreviation_mappings{}                      │
│  Prompt 长度：约 150–200 行                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段2：关系提取 (Relation Extraction)                           │
│  输入：chunk + schema + 阶段1输出                                │
│  输出：triples[]                                                 │
│  Prompt 长度：约 100–150 行                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段3：属性提取 (Attribute Extraction)                          │
│  输入：chunk + schema + 阶段1输出                                │
│  输出：attributes{}                                              │
│  Prompt 长度：约 100–150 行                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  阶段4：验证与增强 (Validation & Enhancement, 可选)              │
│  输入：chunk + 阶段1/2/3 输出                                    │
│  输出：验证/过滤后的 triples + attributes                        │
│  说明：打分、过滤低质量项，消耗 token 较多，默认关闭             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  合并 → 去重 (triple_deduplicate) → 格式化输出                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 各阶段说明

#### 阶段1：实体识别和规范化

- **任务**：识别实体、规范化名称、确定 schema 类型、识别变体（缩写、同义词）。
- **输出示例**：

```json
{
  "entities": [
    {
      "canonical_name": "LiPF6",
      "variants": ["lithium hexafluorophosphate", "LiPF₆"],
      "schema_type": "Salt",
      "source": "LiPF6 (lithium hexafluorophosphate) was added.",
      "evidence": "Formula LiPF6 is canonical. Full name and Unicode variant are variants."
    }
  ],
  "abbreviation_mappings": {
    "EC": "ethylene carbonate",
    "DMC": "dimethyl carbonate"
  }
}
```

#### 阶段2：关系提取

- **任务**：基于阶段1实体抽取三元组，确定方向，填写 `source` / `evidence`。
- **输出示例**：

```json
{
  "triples": [
    {
      "subject": "LiPF6",
      "relation": "dissociates_in",
      "object": "ethylene carbonate",
      "source": "LiPF6 dissociates well in ethylene carbonate (EC).",
      "evidence": "Active voice: subject-verb-object. Both entities from stage 1."
    }
  ]
}
```

#### 阶段3：属性提取

- **任务**：为实体抽取属性（浓度、单位、缩写等），关联到实体，并填写 `source` / `evidence`。
- **输出示例**：

```json
{
  "attributes": {
    "LiPF6": [
      { "key": "concentration", "value": "1 M", "source": "...", "evidence": "..." }
    ],
    "ethylene carbonate": [
      { "key": "abbreviation", "value": "EC", "source": "...", "evidence": "..." }
    ]
  }
}
```

#### 阶段4：验证与增强（可选）

- **任务**：对三元组与属性打分，判断 `source` / `evidence` 是否支持抽取结果；利用 `scorers` 下的 LLM 打分及 chunk 级 bad_case 检测，删除低质量项。
- **说明**：逻辑位于 `staged_extraction/stage4_validation.py`，通过 `enable_stage4_validation` 控制。开启后会显著增加 token 消耗。

### 3.3 设计要点

1. **先实体、后关系**：关系依赖规范化的实体名，阶段2 直接复用阶段1 结果。
2. **属性与关系分离**：属性面向实体自身，规则不同，独立阶段便于优化。
3. **阶段4 可选**：按需开启验证与过滤，权衡质量与成本。

### 3.4 使用方式

```python
from prompt2graph import prompt2graph

# 多阶段提取（不启用阶段4）
output_path = prompt2graph(
    dataset_name="paper_mini",
    schema_name="260114",
    is_chunked=True,
    output_graph_name="graph_staged.json",
    use_staged_extraction=True
)

# 多阶段提取 + 阶段4 验证（耗 token）
output_path = prompt2graph(
    dataset_name="paper_mini",
    schema_name="260114",
    is_chunked=True,
    output_graph_name="graph_staged_validated.json",
    use_staged_extraction=True,
    enable_stage4_validation=True
)
```

**多阶段时**：只需 `schema_name` 或 `schema_content`；`prompt_name` / `prompt_content` 被忽略，使用内置的 staged prompts。

---

## 四、单阶段 vs 多阶段对比

| 维度 | 单阶段 | 多阶段 |
|------|--------|--------|
| Prompt 长度 | 单 prompt 可达 352 行 | 每阶段约 100–200 行 |
| API 调用 | 每 chunk 1 次 | 每 chunk 3 次（若启用阶段4 则更多） |
| 任务耦合 | 实体/关系/属性混合 | 按阶段拆分，职责清晰 |
| 可维护性 | 改一处影响全局 | 可单独调整某阶段 prompt |
| 错误定位 | 较难 | 易定位到具体阶段 |
| 输出格式 | 统一 | 与单阶段兼容 |

---

## 五、输出格式

单阶段与多阶段**共用同一输出结构**，例如：

```json
[
  {
    "start_node": {
      "label": "entity",
      "properties": { "name": "Cu(NO3)2", "schema_type": "Salt" }
    },
    "relation": "has_attribute",
    "end_node": {
      "label": "attribute",
      "properties": { "name": "concentration: 0.1 M" }
    },
    "source": "dissolving 0.1M Cu(NO3)2·3H2O into the blank electrolyte",
    "evidence": "Concentration extracted as attribute per Salt rule.",
    "chunk_id": "kppoapmo"
  }
]
```

边统一包含 `source`、`evidence`、`chunk_id`；若启用阶段4，可能额外带各类 score 字段。

---

## 六、参数说明

### 6.1 `prompt2graph` 主入口

| 参数 | 说明 |
|------|------|
| `dataset_name` | 数据集名（必填） |
| `schema_name` | Schema 名称（与 `schema_content` 二选一） |
| `prompt_name` | 单阶段用的 prompt 名称（多阶段时忽略） |
| `is_chunked` | 是否已分块 |
| `schema_content` | Schema 字典（优先于 `schema_name`） |
| `prompt_content` | 单阶段用的 prompt 内容（多阶段时忽略） |
| `output_graph_name` | 输出 JSON 文件名 |
| `use_staged_extraction` | 是否多阶段提取 |
| `enable_stage4_validation` | 是否启用阶段4 验证（仅多阶段有效，默认 `False`） |

### 6.2 直接调用 `build_lowlevel_graph`

支持通过 `get_lowlevel_graph.build_lowlevel_graph` 调用，参数包括 `chunk_path`、`schema_path` / `schema_content`、`prompt_path` / `prompt_content`、`output_graph_path`、`use_staged_extraction`、`enable_stage4_validation`、`prompt_paths` 等。多阶段下可通过 `prompt_paths` 指定各阶段自定义 prompt 路径。

---

## 七、实现与代码结构

### 7.1 目录结构

```
prompt2graph/
├── prompt2graph.py              # 主入口，支持 use_staged_extraction / enable_stage4_validation
├── get_lowlevel_graph.py        # 图构建与去重，统一处理单阶段/多阶段
├── staged_extraction/
│   ├── __init__.py
│   ├── base.py                  # 各阶段基类
│   ├── stage1_entity_recognition.py
│   ├── stage2_relation_extraction.py
│   ├── stage3_attribute_extraction.py
│   └── stage4_validation.py     # 验证与打分，可选用
├── prompts/
│   ├── 260120_v3.txt            # 单阶段 long prompt 示例
│   └── staged/
│       ├── stage1_entity_recognition.txt
│       ├── stage2_relation_extraction.txt
│       └── stage3_attribute_extraction.txt
├── scorers/
│   └── llm_scorer.py            # 阶段4 所用 chunk 级打分
└── docs/
    └── 图谱提取串讲报告.md      # 本文档
```

### 7.2 数据流（多阶段）

1. 读入 chunk 与 schema。
2. 阶段1 → `entities` + `abbreviation_mappings`。
3. 阶段2 接收阶段1 输出 → `triples`。
4. 阶段3 接收阶段1 输出 → `attributes`。
5. 若 `enable_stage4_validation`：对 triples/attributes 打分、过滤，再合并。
6. 合并为图结构 → `triple_deduplicate` → `format_output` → 写 JSON。

---

## 八、注意事项与性能

### 8.1 性能

- **单阶段**：每 chunk 1 次 API 调用，延迟低，适合大批量或对时延敏感场景。
- **多阶段**：每 chunk 至少 3 次调用；开启阶段4 后另有大量打分调用，token 与耗时明显增加。

### 8.2 使用建议

1. **快速实验 / 简单语料**：单阶段 + 现有 long prompt。
2. **追求质量 / 复杂语料**：多阶段；视需要再开 `enable_stage4_validation`。
3. **调试**：多阶段便于按阶段排查；可单独润色某一阶段的 prompt。

### 8.3 其他

- 各阶段间通过 JSON 传递数据，需保持字段约定一致。
- 若某阶段失败，当前实现会跳过该 chunk 或降级处理，不阻塞整体流程。
- 自定义 staged prompts 可通过 `prompt_paths` 传入 `build_lowlevel_graph`。

---
