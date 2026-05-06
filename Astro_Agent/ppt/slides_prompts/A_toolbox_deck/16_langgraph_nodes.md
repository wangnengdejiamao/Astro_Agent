# A-16 · LangGraph 节点流

**英文副标题**：LangGraph State Machine · 8 Auditable Nodes

## 页面目的
让老板看到 Agent 内部不是黑箱——是个有序的状态机。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `LangGraph 节点流` + mint bar; subtitle italic gray `LangGraph State Machine · 8 Auditable Nodes`.
>
> **Composition**: 一个**水平 S 形**节点流图，**8 个节点**用圆角矩形（mint 描边）水平排布，但每两个之间高度有差，形成柔和起伏。每个节点上方写中文名，下方用 mono 字体写落盘 JSON 文件名：
>
> 1. `Resolve` / `01_resolved_target.json`
> 2. `Data Fetcher` / `02_data_fetch.json`
> 3. `RAG Navigator` / `03_rag_results.json`
> 4. `KG Navigator` / `04_kg_results.json`
> 5. `Iter-1 Baseline` / `05_iteration_1_baseline.json`
> 6. `Iter-2 Residuals` / `06_iteration_2_residuals.json`
> 7. `Iter-3 Systematics` / `07_iteration_3_systematics.json`
> 8. `QA Gate` / `08_qa_gate.json`
>
> 节点间用 mint 实线箭头连接；从 `QA Gate` 引出**两条分支**：
> - 向下分支 → `clear_for_draft` → 蓝色出口标 `→ PaperOrchestra Drafter`
> - 向上分支 → `not_clear` → 红色虚线回到 `Iter-1`，标 `Human Review · abnormal_analysis_report.md`
>
> 顶部右侧极小字：`三次迭代为铁律 / 第三次仍未收敛即触发人审`。
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `LangGraph 节点流`
- 8 节点中文名 + 落盘文件名（按 prompt，不许拼错）
- `clear_for_draft → PaperOrchestra Drafter`
- `not_clear → Human Review · abnormal_analysis_report.md`
- `三次迭代为铁律 / 第三次仍未收敛即触发人审`

## 英文术语锁定
`LangGraph`、`Resolve / Data Fetcher / RAG Navigator / KG Navigator / Iter-1 Baseline / Iter-2 Residuals / Iter-3 Systematics / QA Gate`、`clear_for_draft`、`not_clear`、`PaperOrchestra Drafter`、`abnormal_analysis_report.md`

## 参考图
- `reference_images/image7.png`（构图参考——旧 PPT 的工作流图）
