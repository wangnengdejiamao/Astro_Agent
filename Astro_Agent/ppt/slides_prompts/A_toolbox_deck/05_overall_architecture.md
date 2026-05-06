# A-05 · 总体架构

**英文副标题**：End-to-End Architecture · One Picture, Three Layers

## 页面目的
**全场最重要的一页**：用一张图把三层平台说清楚，老板看完这页就懂大概。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `总体架构` + mint bar; subtitle italic gray `End-to-End Architecture · One Picture, Three Layers`.
>
> **Composition (offset-axis cards, NOT three columns)**: divide the canvas into three irregular horizontal bands, each band is offset by 80px horizontally to create staircase rhythm. From top to bottom:
>
> 1. **顶层 智能体 `analysis_agent`**：在该带左侧画一个圆角主标签 `Chief Investigator`，右侧水平展开 7 个小芯片节点（`resolve → data_fetcher → rag_navigator → kg_navigator → 三次建模迭代 → qa_gate → drafter`），用细mint箭头串联。
>
> 2. **中层 知识 `RAG + KG`**：左侧两个交叉同心圆，左圆标 `RAG · 白矮星文献 SQLite`、右圆标 `KG · 12,740 节点 / 83,782 边`。圆中央写 `Dual Knowledge Bus`。
>
> 3. **底层 工具 `astro_toolbox`**：8×4 蜂窝芯片墙（每个芯片是一个模块名小标签：`SDSS / DESI / LAMOST / KOA / HST / JWST / SPHEREx / ZTF / WISE / Gaia / TESS / Kepler / GALEX / 2MASS / X-ray / SED / HR / WD-fit / cooling-age / period / RV / orbit / six-dim ...`），mint 描边。
>
> 在画布最右侧一根贯穿三层的粗 mint 箭头，顶端写 `坐标输入 RA / Dec`，底端写 `ApJ 论文草稿 .tex`，中段标 `Coordinates → Manuscript`。
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `总体架构`
- `Chief Investigator` · `resolve` · `data_fetcher` · `rag_navigator` · `kg_navigator` · `三次建模迭代` · `qa_gate` · `drafter`
- `RAG · 白矮星文献 SQLite`、`KG · 12,740 节点 / 83,782 边`、`Dual Knowledge Bus`
- `坐标输入 RA / Dec` → `ApJ 论文草稿 .tex`
- 30+ 模块名（按 prompt 列出）

## 英文术语锁定
`analysis_agent`、`Chief Investigator`、`RAG`、`KG`、`SQLite`、`astro_toolbox`、`SDSS`、`DESI`、`LAMOST`、`KOA`、`HST`、`JWST`、`SPHEREx`、`ZTF`、`WISE`、`Gaia`、`TESS`、`Kepler`、`GALEX`、`2MASS`

## 参考图
- `reference_images/image6.png` 或 `image7.png` —— 构图参考（旧 PPT 的端到端架构图，借鉴分层）
