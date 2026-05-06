# A-08 · 物理分析层

**英文副标题**：Physics Layer · SED / HR / WD Fitting / Cooling Age

## 页面目的
告诉老板：拿到数据后，工具箱直接做物理。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `物理分析层` + mint bar; subtitle italic gray `Physics Layer · SED / HR / WD Fitting / Cooling Age`.
>
> **Composition**: A diagonal "Z-flow" with **four offset stations**, each station is a non-rectangular rounded card tilted slightly. Connecting curves are dashed mint Bézier lines. Stations:
>
> 1. `SED 拟合 sed.SEDFitter` — 子项：`collect_photometry → apply_extinction → plot`，附极小线性 SED 曲线缩略图（暗黑底蓝光谱代表多波段，但保持线性单色 mint）。
> 2. `HR 图分类 hr_diagram.HRDiagram` — 子项：`Gaia CMD 定位 / 自动分类`，缩略图为 HR 散点图。
> 3. `白矮星拟合 wd_fitting` — 子项：`Koester 网格 · DA/DB/DC · 双白矮星复合 fit_dwd`。
> 4. `冷却年龄 cooling_age` — 子项：`Gaia G/Bp/Rp + parallax → cooling-age`。
>
> Right side: a vertical mint pill summarising `物理参数闭环 · Teff / log g / mass / cooling age`。
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `物理分析层`
- `SED 拟合 sed.SEDFitter`、`collect_photometry → apply_extinction → plot`
- `HR 图分类 hr_diagram.HRDiagram`、`Gaia CMD 定位 / 自动分类`
- `白矮星拟合 wd_fitting`、`Koester 网格 · DA/DB/DC · 双白矮星复合 fit_dwd`
- `冷却年龄 cooling_age`、`Gaia G/Bp/Rp + parallax → cooling-age`
- `物理参数闭环 · Teff / log g / mass / cooling age`

## 英文术语锁定
`SED`、`HR`、`Gaia`、`Koester`、`DA / DB / DC`、`Teff`、`log g`、`Bp`、`Rp`、`parallax`、`SEDFitter`、`HRDiagram`、`fit_dwd`、`cooling_age`

## 参考图
无。
