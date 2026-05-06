# A-17 · PaperOrchestra 五智能体写作管线

**英文副标题**：PaperOrchestra · 5 Sub-Agents Drafting an ApJ Manuscript

## 页面目的
告诉老板：Agent 不只是出参数，能直接出可发表的 ApJ 草稿。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `PaperOrchestra · 五智能体写作管线` + mint bar; subtitle italic gray `PaperOrchestra · 5 Sub-Agents Drafting an ApJ Manuscript`.
>
> **Composition**: 一个偏右侧的 5-agent 环形（不要居中），每个 agent 是一个 mint 圆角徽章，用细 mint 弧线串成环。中央放一个 `paper.tex` 文档 icon，下方小字 `aastex631 · ApJ-style`。
>
> 5 个 agent（顺时针）：
> 1. **Outline Agent** — 生成大纲与章节骨架
> 2. **Plotting Agent** — 调用 `astro_toolbox` 出 SED / HR / fold / RV 图
> 3. **Literature Review Agent** — 调用 `RAG + KG` 写 Intro & Related Work
> 4. **Section Writing Agent** — 写 Methods / Results / Discussion
> 5. **Content Refinement Agent** — 通读、润色、citation 校正
>
> 画布**左 30%** 留作叙事栏：
> - 一行大字 `从「数据闭环」到「文字闭环」`
> - 三个小卡：
>   - `inputs/idea.md · experimental_log.md`（输入）
>   - `figures/captions.json`（中间产物）
>   - `final/paper.tex · provenance.json`（最终产物）
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `PaperOrchestra · 五智能体写作管线`
- 5 个 agent 中文描述（按 prompt）
- `从「数据闭环」到「文字闭环」`
- 三个产物文件名（按 prompt）
- `aastex631 · ApJ-style`

## 英文术语锁定
`PaperOrchestra`、`Outline Agent / Plotting Agent / Literature Review Agent / Section Writing Agent / Content Refinement Agent`、`astro_toolbox`、`RAG + KG`、`paper.tex`、`aastex631`、`ApJ`、`provenance.json`

## 参考图
- `reference_images/image2.png`（构图参考——旧 PPT 中的 Agent 协同环）
