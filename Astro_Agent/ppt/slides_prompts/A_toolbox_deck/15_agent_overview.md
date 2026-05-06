# A-15 · analysis_agent 总览

**英文副标题**：analysis_agent · A LangGraph Chief Investigator

## 页面目的
亮出整个 Agent 的角色定位——首席研究员，不是 chatbot。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `analysis_agent · 首席研究员` + mint bar; subtitle italic gray `analysis_agent · A LangGraph Chief Investigator`.
>
> **Composition**: 中央放一个 `Chief Investigator` 大圆环徽章（mint 描边，内含一只极简线性「指挥棒」icon），围绕这个圆环呈 **不规则星系状** 散落 7 个职能小圆，**不要**等距等大；每个小圆有一行职能名 + 一行落盘文件名（mono 8pt）：
>
> - `Data Fetcher` / `02_data_fetch.json`
> - `RAG Navigator` / `03_rag_results.json`
> - `KG Navigator` / `04_kg_results.json`
> - `Coder/QA · 三次建模迭代` / `05–07_iteration_*.json`
> - `QA Gate` / `08_qa_gate.json`
> - `Drafter (PaperOrchestra)` / `paper.tex`
> - `Peer Reviewer + Toolbox Evolution` / `peer_review.md / toolbox_evolution_plan.json`
>
> 用极细 mint 线条把每个小圆和中心圆环连起来，线条之间避免对称。
>
> 顶部右侧一行小字：`引擎：LangGraph 状态机 · Pydantic SharedContext · 单一 JSON 状态在所有节点间流动`。
>
> 底部三个标签条：`审计可追溯 · 三次迭代铁律 · QA 门禁`。
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `analysis_agent · 首席研究员`
- 7 个职能名 + 落盘文件名（按 prompt 列出，不许写错）
- `引擎：LangGraph 状态机 · Pydantic SharedContext · 单一 JSON 状态在所有节点间流动`
- `审计可追溯 · 三次迭代铁律 · QA 门禁`

## 英文术语锁定
`analysis_agent`、`LangGraph`、`Chief Investigator`、`Data Fetcher / RAG Navigator / KG Navigator / Coder/QA / QA Gate / Drafter / PaperOrchestra / Peer Reviewer / Toolbox Evolution`、`Pydantic`、`SharedContext`

## 参考图
- `reference_images/image2.png` 或 `image3.png`（构图参考——旧 PPT 的 Agent 架构图）
