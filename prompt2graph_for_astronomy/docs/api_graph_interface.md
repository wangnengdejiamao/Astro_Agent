# 图谱可视化接口文档

本文档描述图谱生成和可视化相关接口。

## 基础信息

| 项目 | 值 |
|------|-----|
| 基础 URL | `http://localhost:6777` |
| 输入目录 | `{PROJECT_DIR}/input/{dataset_name}/corpus_cleaned.json` |
| 输出目录 | `{OUTPUT_DIR}/{dataset_name}/{timestamp}/` |

---

## 1. 生成图谱接口

### 接口地址
```
POST /api/tune_prompt_mixed
```

### 请求参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| dataset_name | string | 是 | 数据集名称，对应 input 目录下的子文件夹名称 |
| session_id | string | 是 | 会话 ID，用于标识本次操作 |
| stage1_style | string | 否 | Stage1 实体识别风格，从 `prompts/staged_customized/stage1_example.json` 中选择，默认为第一个风格 |
| stage2_style | string | 否 | Stage2 关系提取风格，从 `prompts/staged_customized/stage2_example.json` 中选择，默认为第一个风格 |
| user_design_stage1_style | string | 否 | 用户自定义 Stage1 风格（JSON 格式），优先级高于 `stage1_style`，JSON 解析失败时返回 400 错误 |
| user_design_stage2_style | string | 否 | 用户自定义 Stage2 风格（JSON 格式），优先级高于 `stage2_style`，JSON 解析失败时返回 400 错误 |

### 请求示例

```json
{
  "dataset_name": "1-Dicarbonyl_Electrolyte_test",
  "session_id": "abc123",
  "stage1_style": "机制优先风格（强调中间过程实体，适合构建多跳推理图谱）",
  "stage2_style": "多跳机制风格（强调因果链拆解，适合复杂机理分析）"
}
```

### 请求示例（使用自定义 JSON 风格）

```json
{
  "dataset_name": "1-Dicarbonyl_Electrolyte_test",
  "session_id": "abc123",
  "user_design_stage1_style": "{\"input\": \"LiPF6 is dissolved in EC...\", \"output\": {\"entities\": [...]}}",
  "user_design_stage2_style": "{\"input\": \"LiPF6 dissociates in EC...\", \"output\": {\"triples\": [...]}}"
}
```

### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| message | string | 状态信息 |
| output_dir | string | 图谱输出目录绝对路径 |
| graph_files | array | 生成的图谱文件名列表 |
| time_now | string | 时间戳 |
| session_id | string | 请求传入的 session_id |

### 响应示例

```json
{
  "message": "图谱生成成功",
  "output_dir": "/devSpaceIT/huangjiahao/prompt2graph/output/1-Dicarbonyl_Electrolyte_test/20260318120000",
  "graph_files": [
    "multi_stage_deduplicated.json",
    "multi_stage_deduplicated_meta.json"
  ],
  "time_now": "20260318120000",
  "session_id": "abc123"
}
```

### 注意事项

1. **输入检查**: 接口会验证 `input/{dataset_name}/corpus_cleaned.json` 是否存在
2. **配置文件**: 使用 `configs/example_pipeline.yml` 作为默认配置
3. **输出路径**: 自动生成，格式为 `output/{dataset_name}/{YYYYMMDDHHMMSS}/`
4. **风格优先级**: `user_design_stage*_style` > `stage*_style` > 默认第一个风格
5. **风格选择**: `stage1_style` 和 `stage2_style` 为可选参数，用于自定义图谱生成风格，风格名称必须与 `prompts/staged_customized/stage1_example.json` / `stage2_example.json` 中的 key 完全匹配

### 获取 Stage 风格选项

用于前端下拉框动态填充 `stage1_style` / `stage2_style` 可选值。

```
GET /api/stage-examples
```

响应示例：

```json
{
  "参数1": ["默认通用风格（平衡抽取，适用于大多数材料文本）", "机制优先风格（强调中间过程实体，适合构建多跳推理图谱）"],
  "参数2": ["默认通用关系风格（标准三元组抽取，适用于大多数材料文本）", "多跳机制风格（强调因果链拆解，适合复杂机理分析）"]
}
```

响应字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| 参数1 | array | **实体识别风格**下拉选项，对应 `prompts/staged_customized/stage1_example.json` 的 key 列表 |
| 参数2 | array | **关系抽取风格**下拉选项，对应 `prompts/staged_customized/stage2_example.json` 的 key 列表 |

---

### 可用风格列表

#### Stage1 实体识别风格 (stage1_example.json)

| 风格名称 | 说明 |
|---------|------|
| 默认通用风格（平衡抽取，适用于大多数材料文本） | 标准实体识别 |
| 机制优先风格（强调中间过程实体，适合构建多跳推理图谱） | 强调机制和中间实体 |
| 去泛指风格（避免additive/material等泛化节点，提升图谱质量） | 避免泛指词 |

#### Stage2 关系提取风格 (stage2_example.json)

| 风格名称 | 说明 |
|---------|------|
| 默认通用关系风格（标准三元组抽取，适用于大多数材料文本） | 标准三元组 |
| 多跳机制风格（强调因果链拆解，适合复杂机理分析） | 强调因果链 |
| 避免捷径风格（禁止直接连接性能结果，保留中间机制） | 保留中间机制 |
| 被动语态严谨风格（确保主客体方向严格正确） | 严谨方向 |

---

## 2. 获取图谱数据接口

### 接口地址

```
GET /api/graph/{dataset_name}
```

### 路径参数

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| dataset_name | string | 是 | 数据集名称 |

### 查询参数（可选）

| 字段 | 类型 | 说明 |
|------|------|------|
| graph_type | string | 图谱类型：`file`、`content` 或 `pipeline` |
| schema_name | string | schema 名称（graph_type=file 时使用） |
| prompt_name | string | prompt 名称（graph_type=file 时使用） |
| session_id | string | 会话 ID（graph_type=content 时使用） |
| timestamp | string | 时间戳目录（graph_type=pipeline 时使用） |
| threshold_high | int | 高阈值，默认 2000 |
| threshold_medium | int | 中阈值，默认 500 |
| score_threshold | float | **评分阈值**，低于此值的边会标记为 `dimmed`（可选，不传则不过滤） |
| score_type | string | **评分字段**，与 score_threshold 配合使用，详见下方可选值 |

#### score_type 可选值

| 值 | 说明 |
|----|------|
| `accuracy_score` | 准确率评分 |
| `triple_support_score` | 三元组支持度 |
| `usefulness_score` | 有用性评分 |
| `min_accuracy_usefulness` | accuracy_score 和 usefulness_score 的**较小值** |
| `min_accuracy_triple` | accuracy_score 和 triple_support_score 的**较小值** |
| `min_usefulness_triple` | usefulness_score 和 triple_support_score 的**较小值** |
| `min_all` | 三个评分（accuracy、triple_support、usefulness）的**最小值** |

#### 使用示例

```bash
# 不过滤（默认行为）
GET /api/graph/{my_dataset}?graph_type=pipeline

# 筛选准确率 >= 0.7 的边
GET /api/graph/{my_dataset}?graph_type=pipeline&score_threshold=0.7&score_type=accuracy_score

# 筛选 accuracy 和 usefulness 都 >= 0.6 的边
GET /api/graph/{my_dataset}?graph_type=pipeline&score_threshold=0.6&score_type=min_accuracy_usefulness
```

### 响应参数

| 字段 | 类型 | 说明 |
|------|------|------|
| nodes | array | 一级图谱节点列表 |
| links | array | 一级图谱边列表 |
| categories | array | 分类列表 |
| stats | object | 图谱统计信息 |
| graph_file | string | 图谱文件名 |
| graph_path | string | 图谱文件绝对路径 |
| meta_graph | object | 元图谱信息（如果存在） |

#### stats 统计信息字段

| 字段 | 类型 | 说明 |
|------|------|------|
| total_nodes | int | 原始图谱节点总数 |
| total_edges | int | 原始图谱边总数 |
| displayed_nodes | int | 当前展示的节点数 |
| displayed_edges | int | 当前展示的边数 |
| dimmed_edges | int | **低分边数量**（仅当传入 score_threshold 和 score_type 时存在） |
| active_edges | int | **正常边数量**（仅当传入 score_threshold 和 score_type 时存在） |
| score_threshold | float | **使用的评分阈值**（仅当传入时存在） |
| score_type | string | **使用的评分字段**（仅当传入时存在） |

#### nodes 节点字段

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 节点 ID |
| name | string | 节点名称 |
| category | string | 分类（对应 schema_type） |
| symbolSize | int | 节点大小 |
| value | int | 度值 |
| properties | object | 节点属性 |
| label | string | 标签类型 |
| dimmed | boolean | **是否低分节点**，`true` 表示该节点所有边都是低分边，前端应淡出显示 |
| status | string | **节点状态**，可选值：`"active"`（正常）、`"filtered"`（所有边都是低分边） |

> **节点 dimmed 判定规则**：如果一个节点的所有相连边都是 `dimmed=true`，则该节点也被标记为 `dimmed=true`。

#### links 边字段

| 字段 | 类型 | 说明 |
|------|------|------|
| source | string | 起始节点 ID |
| target | string | 目标节点 ID |
| name | string | 关系名称 |
| value | int | 权重值 |
| chunk_id | string | 来源 chunk ID |
| chunk_ids | array | 所有来源 chunk ID 列表 |
| evidence | string | 证据信息 |
| source_text | string | 来源文本 |
| accuracy_score | float | 准确率评分 |
| triple_support_score | float | 三元组支持度 |
| start_accuracy_score | float | 起始节点准确性评分 |
| end_accuracy_score | float | 目标节点准确性评分 |
| usefulness_score | float | 有用性评分 |
| dimmed | boolean | **是否低分边**，`true` 表示评分低于阈值，前端应淡出显示 |
| status | string | **边状态**，可选值：`"active"`（正常）、`"filtered"`（低分被过滤） |

> **前端提示**：当 `dimmed: true` 时，前端应：
> - 设置边的 CSS `opacity: 0.3`（淡出效果）
> - 禁用该边的点击/选中交互事件
> - 当 `dimmed: false` 且 `status: "active"` 时，边正常显示，可交互

#### meta_graph 元图谱字段

| 字段 | 类型 | 说明 |
|------|------|------|
| nodes | array | 元图谱节点列表 |
| links | array | 元图谱边列表 |
| categories | array | 分类列表 |
| meta_graph_file | string | 元图谱文件名 |

### 响应示例

```json
{
  "nodes": [
    {
      "id": "entity_123",
      "name": "BTFE",
      "category": "ChemicalCompound",
      "symbolSize": 30,
      "value": 5,
      "properties": {
        "name": "BTFE",
        "schema_type": "ChemicalCompound"
      },
      "label": "entity",
      "dimmed": false,
      "status": "active"
    },
    {
      "id": "entity_789",
      "name": "LowScoreCompound",
      "category": "ChemicalCompound",
      "symbolSize": 20,
      "value": 2,
      "properties": {
        "name": "LowScoreCompound",
        "schema_type": "ChemicalCompound"
      },
      "label": "entity",
      "dimmed": true,
      "status": "filtered"
    }
  ],
  "links": [
    {
      "source": "entity_123",
      "target": "attribute_456",
      "name": "has_attribute",
      "value": 1,
      "evidence": "从文本中提取的证据信息...",
      "source_text": "BTFE promotes its reduction...",
      "accuracy_score": 0.85,
      "triple_support_score": 0.9,
      "dimmed": false,
      "status": "active"
    },
    {
      "source": "entity_789",
      "target": "attribute_101",
      "name": "related_to",
      "value": 1,
      "evidence": "另一条证据...",
      "accuracy_score": 0.55,
      "dimmed": true,
      "status": "filtered"
    }
  ],
  "categories": [
    {"name": "ChemicalCompound"},
    {"name": "Additive"}
  ],
  "stats": {
    "total_nodes": 150,
    "total_edges": 200,
    "displayed_nodes": 150,
    "displayed_edges": 200,
    "dimmed_edges": 30,
    "active_edges": 170,
    "score_threshold": 0.7,
    "score_type": "accuracy_score"
  },
  "graph_file": "multi_stage_deduplicated.json",
  "graph_path": "/devSpaceIT/huangjiahao/prompt2graph/output/1-Dicarbonyl_Electrolyte_test/20260318124339/multi_stage_deduplicated.json",
  "meta_graph": {
    "nodes": [
      {
        "id": "meta_123",
        "name": "BTFE",
        "category": "ChemicalCompound",
        "properties": {
          "subject": "BTFE",
          "relation": "promotes",
          "object": "CEI layer",
          "evidence": "..."
        }
      }
    ],
    "links": [...],
    "categories": [...],
    "meta_graph_file": "multi_stage_deduplicated_meta.json"
  }
}
```

### 文件查找规则

1. **优先查找**: `output/{dataset_name}/{最新时间戳目录}/multi_stage_deduplicated.json`
2. **元图谱**: `output/{dataset_name}/{最新时间戳目录}/multi_stage_deduplicated_meta.json`
3. 如果找不到子目录，则查找根目录下的 JSON 文件

---

## 3. 测试用例

### 测试 1：生成图谱（使用默认风格）

```bash
curl -X POST "http://localhost:6777/api/tune_prompt_mixed" \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "1-Dicarbonyl_Electrolyte_test", "session_id": "test001"}'
```

### 测试 2：生成图谱（指定风格）

```bash
curl -X POST "http://localhost:6777/api/tune_prompt_mixed" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "1-Dicarbonyl_Electrolyte_test",
    "session_id": "test001",
    "stage1_style": "机制优先风格（强调中间过程实体，适合构建多跳推理图谱）",
    "stage2_style": "多跳机制风格（强调因果链拆解，适合复杂机理分析）"
  }'
```

### 测试 3：生成图谱（使用自定义 JSON 风格）

```bash
curl -X POST "http://localhost:6777/api/tune_prompt_mixed" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "1-Dicarbonyl_Electrolyte_test",
    "session_id": "test001",
    "user_design_stage1_style": "{\"input\": \"LiPF6 is dissolved in EC...\", \"output\": {\"entities\": [{\"canonical_name\": \"LiPF6\", \"variants\": [\"lithium hexafluorophosphate\"], \"schema_type\": \"Salt\", \"source\": \"...\", \"evidence\": \"...\"}]}}",
    "user_design_stage2_style": "{\"input\": \"LiPF6 dissociates in EC...\", \"output\": {\"triples\": [{\"subject\": \"LiPF6\", \"relation\": \"dissociates_in\", \"object\": \"ethylene carbonate\", \"source\": \"...\", \"evidence\": \"...\"}]}}"
  }'
```

### 测试 4：获取图谱数据

```bash
curl "http://localhost:6777/api/graph/1-Dicarbonyl_Electrolyte_test"
```

### 测试 5：检查返回数据

```bash
curl -s "http://localhost:6777/api/graph/1-Dicarbonyl_Electrolyte_test" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('=== 一级图谱 ===')
print(f'节点数: {len(d.get(\"nodes\", []))}')
print(f'边数: {len(d.get(\"links\", []))}')
print('=== 元图谱 ===')
print(f'存在: {\"meta_graph\" in d}')
if 'meta_graph' in d:
    m = d['meta_graph']
    print(f'节点数: {len(m.get(\"nodes\", []))}')
    print(f'边数: {len(m.get(\"links\", []))}')
print('=== 边信息 ===')
links = d.get('links', [])
if links:
    l = links[0]
    print(f'evidence: {\"evidence\" in l}')
    print(f'accuracy_score: {\"accuracy_score\" in l}')
    print(f'triple_support_score: {\"triple_support_score\" in l}')
"
```

### 测试 6：获取图谱数据（带评分筛选）

```bash
# 按准确率筛选 >= 0.7 的边
curl "http://localhost:6777/api/graph/1-Dicarbonyl_Electrolyte_test?graph_type=pipeline&score_threshold=0.7&score_type=accuracy_score"
```

### 测试 7：检查评分筛选结果

```bash
curl -s "http://localhost:6777/api/graph/1-Dicarbonyl_Electrolyte_test?graph_type=pipeline&score_threshold=0.7&score_type=accuracy_score" | python3 -c "
import json, sys
d = json.load(sys.stdin)
stats = d.get('stats', {})
print('=== 评分筛选统计 ===')
print(f'总边数: {stats.get(\"total_edges\", 0)}')
print(f'展示边数: {stats.get(\"displayed_edges\", 0)}')
print(f'正常边数: {stats.get(\"active_edges\", 0)}')
print(f'低分边数: {stats.get(\"dimmed_edges\", 0)}')
print(f'评分阈值: {stats.get(\"score_threshold\", \"N/A\")}')
print(f'评分类型: {stats.get(\"score_type\", \"N/A\")}')

links = d.get('links', [])
dimmed = [l for l in links if l.get('dimmed')]
active = [l for l in links if not l.get('dimmed')]
print(f'\\n=== 边详情 ===')
print(f'正常边: {len(active)} 条')
print(f'低分边（淡出）: {len(dimmed)} 条')
"
```

---

### 前端「图谱质量评估」页面对应关系

页面 `frontend/quality-eval.html` 使用**两个主 Tab**（样式与 `manage.html` 一致），与下面两个接口一一对应：

| Tab | 接口 | 展示内容 |
|-----|------|----------|
| Stage4 评分 Prompt | `GET /api/stage4-prompts` | `stage4_node_accuracy`、`stage4_triple_support` 两列 Markdown 预览 |
| 评估 Prompt | `GET /api/evaluation-prompts` | `meta_graph_quality_evaluation`、`community_quality_evaluation` 两列 Markdown 预览 |

切换至「评估 Prompt」Tab 时首次自动请求 `/api/evaluation-prompts`；两 Tab 内均有「刷新」按钮可重新拉取对应接口。

---

## 4. 获取 Stage4 评分 Prompt 接口

### 接口说明

获取 Stage 4 验证阶段使用的 LLM 评分 prompt 模板，包括节点准确性评分和三元组支持度评分。

### 请求

```http
GET /api/stage4-prompts
```

### 响应

```json
{
    "stage4_node_accuracy": "prompt模板内容（节点准确性评分）",
    "stage4_triple_support": "prompt模板内容（三元组支持度评分）"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| stage4_node_accuracy | string \| null | 节点准确性评分 prompt，如果文件不存在则返回 null |
| stage4_triple_support | string \| null | 三元组支持度评分 prompt，如果文件不存在则 null |

### 示例

```bash
curl -X GET "http://localhost:6777/api/stage4-prompts"
```

### 响应示例

```json
{
    "stage4_node_accuracy": "你是一个图谱质量评估专家...\n\n## 任务\n评估以下节点是否在原文中被准确识别...\n\n## 输入\n节点名称: __NODE_NAME__\n\n原文: __SOURCE_TEXT__\n\n证据: __EVIDENCE_TEXT__\n\n上下文: __CHUNK_TEXT__\n\n## 输出要求\n请返回 JSON 格式的评分结果...",
    "stage4_triple_support": "你是一个图谱质量评估专家...\n\n## 任务\n评估以下三元组是否被原文支持...\n\n## 输入\n起始节点: __START_NODE__\n关系: __RELATION__\n结束节点: __END_NODE__\n\n..."
}
```

---

## 5. 获取评估 Prompt 接口

### 接口说明

获取各种评估任务使用的 LLM 评分 prompt 模板，包括元图谱质量评估和社区质量评估。

### 请求

```http
GET /api/evaluation-prompts
```

### 响应

```json
{
    "meta_graph_quality_evaluation": "prompt模板内容（元图谱质量评估）",
    "community_quality_evaluation": "prompt模板内容（社区质量评估）"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| meta_graph_quality_evaluation | string \| null | 元图谱质量评估 prompt，如果文件不存在则返回 null |
| community_quality_evaluation | string \| null | 社区质量评估 prompt，如果文件不存在则返回 null |

### 示例

```bash
curl -X GET "http://localhost:6777/api/evaluation-prompts"
```

---

## 6. 错误码说明

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 404 | 图谱文件未找到 |
| 500 | 服务器内部错误 |
