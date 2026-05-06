# A-12 · RAG 双轨检索

**英文副标题**：Dual-Track Retrieval · Vector + Keyword + Rerank

## 页面目的
讲清 RAG 的检索链路：双路混合 + Rerank → Top5。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `RAG 双轨检索` + mint bar; subtitle italic gray `Dual-Track Retrieval · Vector + Keyword + Rerank`.
>
> **Composition**: a horizontal funnel/pipeline with five stages, but **bent into a slight S-curve** to keep it non-linear:
>
> 1. `用户问题 / agent query` (输入小气泡，左上)
> 2. `元数据过滤` (mint 漏斗) — 期刊 / 年份 / 方法标签
> 3. **双轨并行**：
>    - 上轨 `向量检索 BGE / Vector Store` → Top20
>    - 下轨 `关键词检索 BM25 / 倒排索引` → Top20
> 4. `合并去重 → 候选集 Top40`
> 5. `Rerank · bge-reranker · Top5`
> 6. `输出 → analysis_agent / Drafter` (右下出口)
>
> 在双轨那段，用上下两条 mint 平行带分别画出，并在中央汇合处画一个 mint 菱形标 `Hybrid Merge`。
>
> Bottom-right small KPI card：
> - `召回率 +20% (vs 纯固定分块)`
> - `多跳准确率 58% → 70%`
> - `幻觉率 28% → 9%`
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `RAG 双轨检索`
- `用户问题 / agent query`、`元数据过滤`、`向量检索 BGE / Vector Store`、`关键词检索 BM25 / 倒排索引`、`合并去重 → 候选集 Top40`、`Rerank · bge-reranker · Top5`、`输出 → analysis_agent / Drafter`、`Hybrid Merge`
- KPI: `召回率 +20% (vs 纯固定分块)`、`多跳准确率 58% → 70%`、`幻觉率 28% → 9%`

## 英文术语锁定
`RAG`、`BGE`、`BM25`、`bge-reranker`、`Vector Store`、`Top20 / Top5 / Top40`、`Drafter`、`analysis_agent`

## 参考图
- `reference_images/image8.png` 或 `image9.png`（构图参考——旧 PPT 的 RAG 链路图）
