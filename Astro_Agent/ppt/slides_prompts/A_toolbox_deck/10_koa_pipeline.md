# A-10 · KOA 自动化流水线

**英文副标题**：KOA / Keck-LRIS Pipeline · Metadata → 1D Spectrum

## 页面目的
拿一个最复杂的子流水线——KOA Keck/LRIS——证明工具箱能吃硬骨头。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `KOA 自动化流水线` + mint bar; subtitle italic gray `KOA / Keck-LRIS Pipeline · Metadata → 1D Spectrum`.
>
> **Composition**: a horizontal **5-step assembly line**, each step is a tilted parallelogram chip in mint outline, connected by mint chevron arrows. Below each step: small mono font listing the produced filename. Above each step: short Chinese label.
>
> 1. **元数据查询** · 产物 `koa_lris_observed_targets.csv` / `koa_lris_observed_exposures.csv`
> 2. **raw FITS 下载** · 产物 `download/lris/lev0/*.fits.gz` + `koa_file_manifest.csv`
> 3. **PypeIt setup** · 产物 `pypeit_setup/lris_*.pypeit`
> 4. **一维谱抽取** · 产物 `spec1d_*.fits` / `spec2d_*.fits`
> 5. **报告聚合** · 产物 `koa_reduction_summary.csv` / `koa_spectrum_report.txt`
>
> Above the line, in a translucent mint band, write the entry-point CLI: `python -m astro_toolbox.koa_batch <csv> --output-root <dir>`.
>
> Bottom-right callout card: `批处理已扫过 N 万曝光 · 一键串到 1D 谱`（N 数字不要写死，用占位 `N=`）。
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `KOA 自动化流水线`
- 5 步标题：`元数据查询 / raw FITS 下载 / PypeIt setup / 一维谱抽取 / 报告聚合`
- 5 个产物文件名（按 prompt 列出，不许拼错）
- CLI 命令 `python -m astro_toolbox.koa_batch <csv> --output-root <dir>`
- `批处理已扫过 N 万曝光 · 一键串到 1D 谱`

## 英文术语锁定
`KOA`、`Keck`、`LRIS`、`PypeIt`、`spec1d`、`spec2d`、`koa_batch`、`koa_lris_observed_targets.csv`、`koa_lris_observed_exposures.csv`、`koa_file_manifest.csv`、`koa_reduction_summary.csv`

## 参考图
无（原创装配线）。
