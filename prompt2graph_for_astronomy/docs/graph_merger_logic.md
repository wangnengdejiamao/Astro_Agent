# 知识图谱合并模块逻辑串讲报告

## 1. 概述

知识图谱合并模块（`graph_merger.py`）是一个用于合并多张知识图谱的工具，通过实体消歧和三元组合并两个核心步骤，将两张独立的知识图谱整合为一张统一的图谱。

**核心功能**：
1. **实体消歧**：识别两张图谱中的相同实体（基于 abbreviation 同义词和 PubChem CID）
2. **实体合并**：将匹配的实体合并，name 字段取并集
3. **三元组合并**：合并相同的三元组，合并 source 和 evidence 字段

**应用场景**：
- 合并来自不同数据源的知识图谱
- 整合同一主题的不同版本图谱
- 构建统一的知识库

## 2. 核心类与方法

### 2.1 GraphMerger 类

主要的图谱合并器类，负责整个合并流程的协调和执行。

#### 初始化
- **参数**：无
- **功能**：初始化图谱合并器

#### 主要公共方法

- `load_graph(graph_path)` - 加载图谱 JSON 文件
- `extract_entities(graph_data)` - 从图谱中提取所有实体节点
- `find_entity_matches(entities1, entities2)` - 找到两张图谱中匹配的实体
- `merge_entities(entities1, entities2, matches)` - 合并两张图谱的实体
- `update_graph_entities(graph_data, merged_entities, entity_mapping)` - 更新图谱中的实体引用
- `merge_triples(graph1, graph2)` - 合并两张图谱的三元组
- `merge(graph_path1, graph_path2, output_path)` - 主合并流程

## 3. 完整合并流程

### 3.1 流程图

```
加载图谱1
    ↓
加载图谱2
    ↓
提取图谱1的所有实体
    ↓
提取图谱2的所有实体
    ↓
查找匹配的实体（基于abbreviation和CID）
    ↓
合并实体（name取并集，更新entity_key）
    ↓
更新图谱1中的实体引用
    ↓
更新图谱2中的实体引用
    ↓
合并三元组（相同三元组合并source和evidence）
    ↓
保存合并后的图谱
```

### 3.2 详细步骤说明

#### 步骤1：加载两张图谱 (`load_graph`)

**功能**：从 JSON 文件加载知识图谱数据

**输入**：图谱 JSON 文件路径

**输出**：图谱三元组列表

**关键逻辑**：
- 检查文件是否存在
- 使用 UTF-8 编码读取 JSON 文件
- 返回三元组列表（每个三元组包含 `start_node`、`relation`、`end_node`、`source`、`evidence` 等信息）

**数据结构示例**：
```json
[
  {
    "start_node": {
      "label": "entity",
      "properties": {
        "name": "water",
        "schema_type": "Solvent"
      }
    },
    "relation": "has_attribute",
    "end_node": {
      "label": "attribute",
      "properties": {
        "name": "temperature: 25°C"
      }
    },
    "source": "The reaction was performed at 25°C",
    "evidence": "Temperature extracted from text"
  }
]
```

---

#### 步骤2：提取实体 (`extract_entities`)

**功能**：从图谱中提取所有实体节点（仅 `label="entity"` 的节点）

**输入**：图谱三元组列表

**输出**：实体字典 `{entity_key: {properties: {...}}}`

**关键逻辑**：
- 遍历所有三元组的 `start_node` 和 `end_node`
- 只提取 `label == "entity"` 的节点（忽略 `attribute` 类型）
- 使用规范化后的 name 计算 `entity_key`（元组形式）
- 如果同一图谱内存在相同 `entity_key` 的实体，合并它们的 name 字段和其他属性
- 如果合并后 name 变化，更新 `entity_key`

**entity_key 计算规则**：
- 如果 name 是字符串：`entity_key = (normalize_name(name),)`
- 如果 name 是列表：`entity_key = tuple(sorted(set([normalize_name(n) for n in name])))`

**数据结构示例**：
```python
{
    ('water',): {
        "properties": {
            "name": "water",
            "schema_type": "Solvent"
        }
    },
    ('h2o', 'water'): {
        "properties": {
            "name": ["H2O", "water"],
            "schema_type": "Solvent",
            "cid": 962
        }
    }
}
```

**重要特性**：
- 同一图谱内的实体如果 name 规范化后相同，会自动合并
- 合并时 name 字段取并集并去重
- 其他属性保留已有的，添加新的

---

#### 步骤3：查找匹配的实体 (`find_entity_matches`)

**功能**：找到两张图谱中匹配的实体（基于 abbreviation 和 CID）

**输入**：
- `entities1`：第一张图谱的实体字典
- `entities2`：第二张图谱的实体字典

**输出**：匹配映射字典 `{entities2的key: entities1的key}`

**匹配策略**：

**方法1：基于 Abbreviation（name 匹配）**
- 为 entities1 建立 name 索引：`{normalized_name: [entity_keys]}`
- 遍历 entities2 中每个实体的所有 name 变体
- 如果规范化后的 name 在索引中存在，则匹配成功
- 使用第一个匹配的实体

**方法2：基于 CID（PubChem Compound ID）**
- 为 entities1 建立 CID 索引：`{cid: [entity_keys]}`
- 如果方法1未找到匹配，且 entities2 的实体有 CID
- 在 CID 索引中查找，找到则匹配成功

**匹配优先级**：
1. 优先使用 name 匹配（abbreviation）
2. 如果 name 匹配失败，使用 CID 匹配

**示例**：
```python
# entities1
{
    ('water',): {
        "properties": {"name": "water", "cid": 962}
    }
}

# entities2
{
    ('h2o',): {
        "properties": {"name": "H2O", "cid": 962}
    }
}

# 匹配结果
matches = {
    ('h2o',): ('water',)  # 通过CID匹配
}
```

**关键设计**：
- 使用索引结构提高查找效率（O(1) 查找）
- 支持 name 为列表的情况（遍历所有 name 变体）
- 规范化 name 后比较，处理大小写、空格等差异

---

#### 步骤4：合并实体 (`merge_entities`)

**功能**：合并两张图谱的实体，生成合并后的实体字典和映射关系

**输入**：
- `entities1`：第一张图谱的实体字典
- `entities2`：第二张图谱的实体字典
- `matches`：实体匹配映射

**输出**：
- `merged_entities`：合并后的实体字典
- `entity_mapping1`：entities1 的 key 映射（entities1的key -> 合并后的key）
- `entity_mapping2`：entities2 的 key 映射（entities2的key -> 合并后的key）

**合并流程**：

**4.1 处理匹配的实体**

对于每个匹配的实体对 `(entity_key2, entity_key1)`：

1. **合并 name 字段**：
   - 收集两个实体的所有 name 变体
   - 基于规范化后的 name 去重
   - 如果只有一个唯一 name，存储为字符串；否则存储为列表

2. **合并其他属性**：
   - 保留 entities1 的属性
   - 如果 entities2 有新的属性，添加到合并后的实体
   - 如果 CID 不同，保留 entities1 的并警告

3. **更新 entity_key**：
   - 根据合并后的 name 重新计算 `entity_key`
   - 如果 `entity_key` 变化，更新字典中的 key
   - 同步更新所有相关的映射关系

**4.2 处理未匹配的实体**

对于 entities2 中未匹配的实体：

1. **直接添加**：如果 key 不存在，直接添加到合并结果
2. **处理冲突**：如果 key 已存在（可能是在处理匹配实体时更新了 key 导致的），合并实体

**示例**：
```python
# 输入
entities1 = {
    ('water',): {
        "properties": {"name": "water", "schema_type": "Solvent"}
    }
}

entities2 = {
    ('h2o',): {
        "properties": {"name": "H2O", "cid": 962}
    }
}

matches = {
    ('h2o',): ('water',)
}

# 合并后
merged_entities = {
    ('h2o', 'water'): {
        "properties": {
            "name": ["H2O", "water"],
            "schema_type": "Solvent",
            "cid": 962
        }
    }
}

entity_mapping1 = {
    ('water',): ('h2o', 'water')  # entity_key更新了
}

entity_mapping2 = {
    ('h2o',): ('h2o', 'water')
}
```

**重要特性**：
- name 字段取并集，保留所有变体
- 合并后可能更新 `entity_key`（因为 name 变化）
- 维护两个映射关系，用于后续更新图谱中的实体引用

---

#### 步骤5：更新图谱中的实体引用 (`update_graph_entities`)

**功能**：使用合并后的实体更新图谱中的实体节点

**输入**：
- `graph_data`：原始图谱数据
- `merged_entities`：合并后的实体字典
- `entity_mapping`：实体 key 映射（原始key -> 合并后的key）
- `graph_id`：图谱标识（用于日志）

**输出**：更新后的图谱数据

**关键逻辑**：
- 遍历图谱中的所有三元组
- 对于每个三元组的 `start_node` 和 `end_node`：
  - 如果是 `label="entity"` 的节点，提取其 `entity_key`
  - 通过 `entity_mapping` 查找合并后的 `entity_key`
  - 从 `merged_entities` 中获取合并后的实体信息
  - 替换节点中的 `properties`

**更新示例**：
```python
# 原始三元组
{
    "start_node": {
        "label": "entity",
        "properties": {"name": "water"}
    },
    "relation": "has_attribute",
    "end_node": {...}
}

# 更新后（entity_mapping: {('water',): ('h2o', 'water')}）
{
    "start_node": {
        "label": "entity",
        "properties": {
            "name": ["H2O", "water"],
            "schema_type": "Solvent",
            "cid": 962
        }
    },
    "relation": "has_attribute",
    "end_node": {...}
}
```

**重要特性**：
- 只更新 `label="entity"` 的节点
- `attribute` 节点保持不变
- 使用映射关系确保正确找到合并后的实体

---

#### 步骤6：合并三元组 (`merge_triples`)

**功能**：合并两张图谱的三元组，相同的三元组合并 source 和 evidence 字段

**输入**：
- `graph1`：第一张图谱的三元组列表（已更新实体引用）
- `graph2`：第二张图谱的三元组列表（已更新实体引用）

**输出**：合并后的三元组列表

**关键逻辑**：

1. **构建索引**：
   - 为每张图谱的三元组计算规范化 key：`(start_node_key, relation, end_node_key)`
   - 建立索引：`{triple_key: [(graph_id, triple), ...]}`

2. **合并相同三元组**：
   - 对于每个唯一的 `triple_key`：
     - 选择第一个三元组作为基础（保留结构）
     - 收集所有三元组的 source、evidence、chunk_id
     - 去重并合并：
       - 如果只有一个唯一值，存储为字符串/值
       - 如果有多个值，存储为列表

**三元组 key 计算规则**：
- 对于 entity 节点：使用 `entity_key`（规范化 name 的元组）
- 对于 attribute 节点：使用 name 字符串
- 最终 key：`(str(start_node_key), relation, str(end_node_key))`

**合并示例**：
```python
# 图谱1中的三元组
{
    "start_node": {"label": "entity", "properties": {"name": "water"}},
    "relation": "has_attribute",
    "end_node": {"label": "attribute", "properties": {"name": "temperature: 25°C"}},
    "source": "The reaction was performed at 25°C",
    "evidence": "Temperature extracted from text"
}

# 图谱2中的相同三元组
{
    "start_node": {"label": "entity", "properties": {"name": "water"}},
    "relation": "has_attribute",
    "end_node": {"label": "attribute", "properties": {"name": "temperature: 25°C"}},
    "source": "Reaction temperature: 25°C",
    "evidence": "Temperature value identified"
}

# 合并后
{
    "start_node": {"label": "entity", "properties": {"name": "water"}},
    "relation": "has_attribute",
    "end_node": {"label": "attribute", "properties": {"name": "temperature: 25°C"}},
    "source": [
        "The reaction was performed at 25°C",
        "Reaction temperature: 25°C"
    ],
    "evidence": [
        "Temperature extracted from text",
        "Temperature value identified"
    ]
}
```

**重要特性**：
- 基于规范化后的节点 key 判断三元组是否相同
- 保留所有 source 和 evidence 信息
- 自动去重，避免重复信息

---

#### 步骤7：保存合并后的图谱

**功能**：将合并后的图谱保存到 JSON 文件

**输入**：合并后的三元组列表

**输出**：输出文件路径

**关键逻辑**：
- 如果未指定输出路径，自动生成：`{graph1_name}_merged_{graph2_name}.json`
- 创建输出目录（如果不存在）
- 使用 UTF-8 编码保存 JSON 文件
- 使用缩进格式化，便于阅读

---

## 4. 核心算法与数据结构

### 4.1 实体匹配算法

**索引构建**：
- 使用 `defaultdict(list)` 构建倒排索引
- name 索引：`{normalized_name: [entity_keys]}`
- CID 索引：`{cid: [entity_keys]}`

**匹配策略**：
1. 优先使用 name 匹配（支持 abbreviation 同义词）
2. 如果 name 匹配失败，使用 CID 匹配（更可靠但可能缺失）

**时间复杂度**：
- 索引构建：O(n)，n 是 entities1 的实体数量
- 匹配查找：O(m × k)，m 是 entities2 的实体数量，k 是每个实体的 name 数量
- 总体：O(n + m × k)，非常高效

### 4.2 实体合并算法

**name 合并**：
- 收集所有 name 变体到一个列表
- 基于规范化后的 name 去重
- 保留原始格式（大小写等）

**entity_key 更新**：
- 合并后 name 可能变化
- 重新计算 `entity_key`
- 更新字典中的 key 和所有相关映射

**映射维护**：
- `entity_mapping1`：记录 entities1 的 key 变化
- `entity_mapping2`：记录 entities2 的 key 映射
- 确保后续更新图谱时能正确找到合并后的实体

### 4.3 三元组合并算法

**key 规范化**：
- entity 节点：使用 `entity_key`（元组）
- attribute 节点：使用 name（字符串）
- 转换为字符串用于比较

**字段合并**：
- source、evidence、chunk_id 字段收集所有值
- 去重后，如果只有一个值返回该值，否则返回列表

---

## 5. 关键设计决策

### 5.1 实体消歧策略

**为什么使用两种匹配方法？**
- **Abbreviation 匹配**：快速、不需要外部数据，但可能误匹配
- **CID 匹配**：准确、可靠，但需要实体有 CID 且可能缺失

**优先级设计**：
- 先尝试 abbreviation 匹配
- 如果失败，再尝试 CID 匹配
- 这样既保证了效率，又提高了准确性

### 5.2 name 字段处理

**为什么 name 可能是列表？**
- 实体消歧后，一个实体可能有多个 name 变体
- 例如：`["water", "H2O", "dihydrogen monoxide"]`
- 保留所有变体有助于后续查询和展示

**合并策略**：
- 取并集：保留所有 name 变体
- 去重：基于规范化后的 name
- 保留原始格式：不改变大小写等

### 5.3 entity_key 设计

**为什么使用元组作为 key？**
- 元组是可哈希的，可以作为字典的 key
- 规范化后的 name 元组能唯一标识实体
- 支持 name 为列表的情况

**key 更新机制**：
- 合并后 name 变化，需要更新 `entity_key`
- 更新字典中的 key 和所有相关映射
- 确保后续查找能正确找到实体

### 5.4 映射关系维护

**为什么需要两个映射？**
- `entity_mapping1`：entities1 的 key 可能因为合并而更新
- `entity_mapping2`：entities2 的 key 需要映射到合并后的 key
- 两个映射确保两张图谱都能正确更新

---

## 6. 使用示例

### 6.1 命令行使用

```bash
python graph_merger.py graph1.json graph2.json merged.json
```

### 6.2 Python 代码使用

```python
from graph_merger import merge_graphs

# 合并两张图谱
output_path = merge_graphs(
    "output/paper_mini/260120_v2_deduplicated.json",
    "output/paper_mini/260120_deduplicated.json",
    "output/paper_mini/merged.json"
)
```

### 6.3 使用 GraphMerger 类

```python
from graph_merger import GraphMerger

merger = GraphMerger()
output_path = merger.merge(
    "output/paper_mini/260120_v2_deduplicated.json",
    "output/paper_mini/260120_deduplicated.json",
    "output/paper_mini/merged.json"
)
```

---

## 7. 输出结果说明

### 7.1 合并后的实体

合并后的实体包含：
- **name**：所有 name 变体的并集（字符串或列表）
- **其他属性**：合并自两张图谱的所有属性
- **entity_key**：基于合并后的 name 计算

### 7.2 合并后的三元组

合并后的三元组包含：
- **节点信息**：使用合并后的实体信息
- **source**：所有 source 的并集（字符串或列表）
- **evidence**：所有 evidence 的并集（字符串或列表）
- **chunk_id**：所有 chunk_id 的并集（字符串或列表）

### 7.3 统计信息

合并过程中会输出以下统计信息：
- 提取的实体数量
- 匹配的实体对数
- 合并后的实体数量
- 合并后的三元组数量

---

## 8. 注意事项与限制

### 8.1 输入要求

- 输入图谱必须是有效的 JSON 格式
- 三元组必须包含 `start_node`、`relation`、`end_node` 字段
- 实体节点必须有 `name` 字段

### 8.2 性能考虑

- 对于大型图谱，合并过程可能需要较长时间
- 建议在合并前先对单张图谱进行实体消歧
- 索引结构优化了查找效率，但内存占用可能较大

### 8.3 数据质量

- CID 匹配依赖于实体的 CID 字段是否准确
- name 匹配依赖于 name 的规范化是否一致
- 建议在合并前检查数据质量

### 8.4 特殊情况处理

- **CID 冲突**：如果匹配的实体 CID 不同，会警告并保留 entities1 的 CID
- **key 冲突**：如果未匹配的实体 key 已存在，会合并实体
- **空 name**：跳过没有 name 的实体节点

---

## 9. 总结

知识图谱合并模块通过实体消歧和三元组合并两个核心步骤，实现了多张知识图谱的统一整合。主要特点包括：

1. **智能实体匹配**：基于 abbreviation 和 CID 双重策略，提高匹配准确性
2. **完整信息保留**：合并时保留所有 name 变体和 source/evidence 信息
3. **高效算法设计**：使用索引结构优化查找效率
4. **灵活的数据结构**：支持多种图谱格式

通过合理使用合并模块，可以构建统一、完整的知识图谱，为后续的知识查询和分析提供基础。
