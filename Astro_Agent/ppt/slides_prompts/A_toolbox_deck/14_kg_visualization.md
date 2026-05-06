# A-14 · 知识图谱可视化

**英文副标题**：KG Visualization · Backbone & Communities

## 页面目的
直接放真图，用真实数据告诉老板「这是真做出来了」。

## Prompt

> 16:9 white `#FFFFFF`, mint `#12CCB9`. Title top-left `知识图谱可视化` + mint bar; subtitle italic gray `KG Visualization · Backbone & Communities`.
>
> **Composition**: 双图并置 + 左侧叙事卡 — **不是**等宽两栏，而是大小 6:4 错位。
>
> - 画布**左 35%**：一张窄叙事卡，上下两段：
>   - 上段（KPI 三连）：特大 mint 数字
>     - `12,740` 小字 `节点`
>     - `83,782` 小字 `边`
>     - `9` 小字 `社区`
>   - 下段：一段说明 `Top 节点：SDSS · Gaia · 2MASS · parallax · astrometry · GALEX · WISE · Pan-STARRS`
> - 画布**右 65%**：嵌入两张实图，上 `kg_backbone_overview.png`、下 `kg_community_overview.png`，圆角白卡承载，每张图下方一行小字标注：上图 `Backbone · 主干结构`、下图 `Communities · 社区分布`。
>
> **重要**：右侧两张图不要重画，请用 `reference_images/kg_backbone.png` 与 `reference_images/kg_community.png` 直接嵌入；图像模型仅负责加标题/卡片壳子/页脚。
>
> Footer: thin mint line + `林佳茂 · 中山大学物理与天文学院 · 2026`.

## 关键中文文字
- `知识图谱可视化`
- `12,740 节点`、`83,782 边`、`9 社区`
- `Top 节点：SDSS · Gaia · 2MASS · parallax · astrometry · GALEX · WISE · Pan-STARRS`
- `Backbone · 主干结构`、`Communities · 社区分布`

## 英文术语锁定
`Backbone`、`Communities`、`SDSS / Gaia / 2MASS / GALEX / WISE / Pan-STARRS`、`parallax`、`astrometry`

## 参考图（**真实数据图，必须保留**）
- `reference_images/kg_backbone.png` — 上图
- `reference_images/kg_community.png` — 下图
