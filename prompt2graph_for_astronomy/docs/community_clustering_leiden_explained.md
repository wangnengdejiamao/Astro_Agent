# cluster_with_leiden 函数与 hierarchical_leiden 逐行说明

## 一、Leiden 算法背景（简要）

Leiden 是一种**图社区检测算法**，用于将图中的节点划分成若干社区（cluster），使得**同一社区内节点连接紧密，不同社区之间连接稀疏**。

- **优化目标**：通常最大化**模块度（modularity）**，衡量社区划分的“紧密程度”
- **迭代过程**：逐节点尝试加入邻居所在社区，若能使模块度提升则移动
- **与 Louvain 区别**：Leiden 引入 `randomness` 参数，允许一定随机探索，避免 Louvain 可能产生的“不良连通社区”

---

## 二、hierarchical_leiden 与 leiden 的区别

| 函数 | 行为 |
|------|------|
| `leiden()` | 对整图跑一次 Leiden，得到**一层**社区划分，直接返回 `{节点 -> 社区ID}` |
| `hierarchical_leiden()` | 在 Leiden 基础上做**层次化**：若某社区节点数 ≥ `max_cluster_size`，则对该社区子图再跑一次 Leiden，形成多层级结构 |

**hierarchical_leiden 的层次逻辑**（graspologic 文档）：

1. **Level 0**：对整图跑 Leiden，得到初始社区划分
2. **检查**：若某社区节点数 ≥ `max_cluster_size`，则：
   - 提取该社区对应的子图（只含该社区内的节点和边）
   - 对该子图再跑一次 Leiden
   - 得到的新社区记为 **Level 1**，并记录 `parent_cluster`（父社区 ID）
3. **递归**：对 Level 1 中仍超大的社区重复上述过程，得到 Level 2、3...
4. **终止**：直到所有社区都 < `max_cluster_size`，或无法再细分

**返回值**：`List[HierarchicalCluster]`，每个元素是一个 `HierarchicalCluster` 命名元组，包含：

- `node`：节点 ID
- `cluster`：该节点在当前 level 所属的社区 ID
- `parent_cluster`：该社区由哪个父社区细分而来（Level 0 时为 `None`）
- `level`：层级（0, 1, 2, ...）
- `is_final_cluster`：该节点是否已处于“最终”社区（不再参与后续细分）

---

## 三、cluster_with_leiden 逐行说明

```python
def cluster_with_leiden(
    graph: nx.Graph,
    max_cluster_size: int = 1000,
    use_lcc: bool = False,
    seed: Optional[int] = 42,
) -> Tuple[Dict[str, int], Dict[int, int]]:
```

### 第 97-102 行：依赖与空图检查

```python
    if not HAS_GRASPOLOGIC:
        raise ImportError("需要安装 graspologic: pip install graspologic")

    if graph.number_of_nodes() == 0:
        logger.warning("图无节点")
        return {}, {}
```

- 检查是否已安装 graspologic
- 若图为空，直接返回空字典，避免后续报错

---

### 第 105-107 行：可选的最大连通分量

```python
    if use_lcc:
        graph = largest_connected_component(graph)
        graph = nx.Graph(graph)  # 确保是 nx.Graph
```

- `use_lcc=True` 时：只保留**最大连通分量**（LCC），丢弃孤立子图
- 目的：Leiden 在连通图上效果更好；孤立节点/小连通分量可能干扰聚类
- `nx.Graph(graph)`：`largest_connected_component` 可能返回子图视图，转成普通 `nx.Graph` 保证类型一致

---

### 第 109-113 行：调用 hierarchical_leiden（核心）

```python
    community_mapping = hierarchical_leiden(
        graph,
        max_cluster_size=max_cluster_size,
        random_seed=seed,
    )
```

**hierarchical_leiden 内部大致流程**（graspologic 实现）：

1. 将 `graph` 转为边列表 `(source, target, weight)`，传给 Rust 实现的 `graspologic_native`
2. **Level 0**：对整图跑 Leiden，得到初始划分
3. **层次化**：遍历每个社区，若 `len(社区) >= max_cluster_size`：
   - 提取该社区诱导子图
   - 对子图再跑 Leiden
   - 新社区 ID 映射回全局空间，`parent_cluster` 设为原社区 ID
   - `level += 1`
4. 重复直到无社区可再细分
5. 返回 `List[HierarchicalCluster]`，相当于**每次划分变更的一条记录**

**返回值 `community_mapping`**：  
一个列表，每个元素形如：

```
HierarchicalCluster(node='In(NO3)3', cluster=0, parent_cluster=None, level=0, is_final_cluster=True)
HierarchicalCluster(node='SEI Layer', cluster=0, parent_cluster=None, level=0, is_final_cluster=True)
...
```

同一节点可能在不同 level 出现多次（若其所在社区被细分），`cluster` 会随 level 变化。

---

### 第 115-125 行：解析 community_mapping，构建两个字典

```python
    # 构建 level -> {node -> cluster_id}
    node_id_to_community_map: Dict[int, Dict[str, int]] = {}
    hierarchy: Dict[int, int] = {}

    for partition in community_mapping:
        level = partition.level
        node_id_to_community_map.setdefault(level, {})
        node_id_to_community_map[level][partition.node] = partition.cluster
        hierarchy[partition.cluster] = (
            partition.parent_cluster if partition.parent_cluster is not None else -1
        )
```

**逐条解释**：

- `node_id_to_community_map`：`{level: {节点名: 社区ID}}`
  - 例如 `node_id_to_community_map[0]['In(NO3)3'] = 0` 表示在 Level 0 时，节点 `In(NO3)3` 属于社区 0
  - 同一节点在不同 level 的 `cluster` 可能不同（被细分后）

- `hierarchy`：`{社区ID: 父社区ID}`
  - `partition.cluster` 是当前这条记录中的社区 ID
  - `partition.parent_cluster` 是它的父社区（Level 0 时为 `None`）
  - 用 `-1` 表示根社区（无父社区）

**注意**：循环中会多次覆盖 `hierarchy[partition.cluster]`，因为同一社区在不同 level 的记录中会重复出现，最终保留的是最后一次写入的值，对根社区来说都是 `-1`，对子社区来说是父 ID，逻辑正确。

---

### 第 127-131 行：取每个节点的最终社区归属（重要修正）

**原逻辑的 bug**：若用 `max_level = max(...)` 再取 `node_id_to_community_map[max_level]`，会**漏掉**从未被细分的社区中的节点。因为：
- 被细分的社区：其节点会在 level 1、2... 再次出现
- 未被细分的社区（节点数 < max_cluster_size）：其节点**只在 level 0 出现**，不会出现在更高 level
- 因此 `node_id_to_community_map[max_level]` 只包含被细分过的节点，其余节点会丢失

**正确做法**：使用 graspologic 提供的 `final_level_hierarchical_clustering()`：

```python
    node_to_community = community_mapping.final_level_hierarchical_clustering()
```

该方法内部用 `is_final_cluster=True` 筛选：每个节点在“最终确定归属”时会被标记为 `is_final_cluster`，从而保证：
- 从未被细分的节点：在 level 0 即 `is_final_cluster=True`，会被取到
- 被细分过的节点：在最高 level 才 `is_final_cluster=True`，会取到细分后的社区 ID

---

## 四、图示：层次聚类何时发生

```
整图 (100 节点)
    │
    ▼ Leiden (Level 0)
    ├─ 社区 0: 60 节点  ← 超过 max_cluster_size=1000？否，不细分
    ├─ 社区 1: 25 节点
    └─ 社区 2: 15 节点

若 max_cluster_size=20，社区 0 有 60 节点：
    │
    ▼ 对社区 0 诱导子图再跑 Leiden (Level 1)
    ├─ 社区 10 (parent=0): 20 节点
    ├─ 社区 11 (parent=0): 18 节点
    └─ 社区 12 (parent=0): 22 节点  ← 仍超 20，可继续细分 (Level 2)
```

你的 260129 图谱规模较小，所有社区都 < 1000，因此只有 Level 0，`hierarchy` 中全是 `-1`。

---

## 五、graspologic.hierarchical_leiden 参数速查

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `graph` | - | 输入图（nx.Graph / 边列表 / 邻接矩阵） |
| `max_cluster_size` | 1000 | 超过此规模的社区会被细分 |
| `random_seed` | None | 随机种子，用于可复现 |
| `resolution` | 1.0 | 越大社区越多，越小社区越少 |
| `randomness` | 0.001 | 探索强度，越大越随机 |
| `use_modularity` | True | 使用模块度（否则用 CPM） |

---

## 六、参考文献

- Traag, V.A.; Waltman, L.; Van Eck, N.J. "From Louvain to Leiden: guaranteeing well-connected communities", *Scientific Reports*, Vol. 9, 2019.
- graspologic 源码：`graspologic/partition/leiden.py`
