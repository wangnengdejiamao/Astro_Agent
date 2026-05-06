# A-13 · 知识图谱构建管线

**英文副标题**：Knowledge Graph Pipeline · prompt2graph Multi-Stage Deduplication

## 页面目的
讲 KG 是怎么造出来的：从顶刊文献 → 抽取节点/关系 → 多阶段去重 → 12,740 节点。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `知识图谱构建` + mint bar; subtitle italic gray `Knowledge Graph Pipeline · prompt2graph Multi-Stage Deduplication`.
>
> **Composition**: a left-to-right pipeline with **6 stations**, drawn as a sloping serpentine path (NOT straight), each station is a small mint hexagon with Chinese label below:
>
> 1. `顶刊文献 PDF · 8GB` → 2. `LLM 抽取 · prompt2graph 范式` → 3. `初版三元组 · ~30万` → 4. `多阶段去重 · multi-stage dedup` → 5. `社区检测 · networkx Louvain` → 6. `KG · 12,740 节点 / 83,782 边 / 9 社区`
>
> Right column (offset card): a ring chart showing the 9 node-types from real stats:
> - `Paper 8,661`
> - `AstronomicalSource 3,949`
> - `WhiteDwarfCategory 23`
> - `Result 23`
> - `ObservationInstrument 22`
> - `Survey 20`
> - `AnalysisMethod 14`
> - `PhysicalParameter 14`
> - `PhysicalModel 14`
>
> Use mint color for the dominant slice and varied gray for the rest.
>
> Bottom callout: `KG 给 Agent 注入「方法迁移路径」 · 比 RAG 更显式的领域知识`.
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `知识图谱构建`
- 6 站标题（按 prompt 列出）
- `KG · 12,740 节点 / 83,782 边 / 9 社区`
- 9 个节点类型与计数（按 prompt 列出，**数字一字不改**）
- `KG 给 Agent 注入「方法迁移路径」 · 比 RAG 更显式的领域知识`

## 英文术语锁定
`prompt2graph`、`networkx`、`Louvain`、`Paper / AstronomicalSource / WhiteDwarfCategory / Result / ObservationInstrument / Survey / AnalysisMethod / PhysicalParameter / PhysicalModel`

## 参考图
- `reference_images/kg_backbone.png`（仅作 KG 形态参考）
