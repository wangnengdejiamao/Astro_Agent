# A-18 · 端到端真实案例 · ZTF J152934.91+292801.87

**英文副标题**：End-to-End Demo · From RA/Dec to Manuscript

## 页面目的
用真实样本目标证明：平台不是 PPT 工程，能跑通。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `端到端真实案例` + mint bar; subtitle italic gray `End-to-End Demo · From RA/Dec to Manuscript`.
>
> **Composition**: 一根**水平时间轴**贯穿画布中间偏下，时间轴上左→右标 11 个里程碑节点，节点高度交替（上下错落）以避免单调。每个节点：
>
> 1. **输入** · `target = ZTF J152934.91+292801.87`
> 2. `01_resolved_target.json` (SIMBAD 交叉匹配)
> 3. `02_data_fetch.json` (`astro_toolbox` 多波段拉取)
> 4. `03_rag_results.json` (4 组方法学查询)
> 5. `04_kg_results.json` (3 组方法迁移查询)
> 6. `05_iteration_1_baseline.json`
> 7. `06_iteration_2_residuals.json`
> 8. `07_iteration_3_systematics.json`
> 9. `08_qa_gate.json` (clear_for_draft ✓)
> 10. `paper_orchestra/figures/*.png` (SED / HR / fold)
> 11. **输出** · `final/paper.tex` (ApJ-style 草稿) + `peer_review.md`
>
> 时间轴上方放 3 个产物缩略卡（错位）：
> - 缩略 SED 图（线性 mint 曲线）
> - 缩略 HR 图（mint 散点）
> - 缩略 fold light curve（mint 折叠曲线）
>
> 时间轴下方一行小字：`真实样本目录：output/analysis_agent/ZTFJ152934_full_agent_run/`
>
> 右下大数字 KPI：`从坐标到草稿 · 端到端 1 次提交`。
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `端到端真实案例`
- 11 个里程碑（按 prompt，不许漏字符）
- `真实样本目录：output/analysis_agent/ZTFJ152934_full_agent_run/`
- `从坐标到草稿 · 端到端 1 次提交`

## 英文术语锁定
`ZTF J152934.91+292801.87`、`SIMBAD`、`astro_toolbox`、`RAG`、`KG`、`clear_for_draft`、`paper.tex`、`peer_review.md`、`ApJ`

## 参考图
无（用真实样本路径，原创时间轴）。
