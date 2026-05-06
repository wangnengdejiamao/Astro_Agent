# ADS 白矮星文献检索报告

## 检索概况
- **检索时间**: 2026-04-15 15:04:04
- **检索目标**: 白矮星、星团、白矮双星相关文献（排除超新星）
- **时间范围**: 2015-2025 (近10年)
- **期刊限制**: ApJ, MNRAS, A&A, Nature, Science, AJ 等顶级期刊
- **总计获取**: **6,281 篇**

---

## 按主题分类

| 分类 | 篇数 | 占比 |
|------|------|------|
| 白矮星演化模型 | 1,854 | 29.5% |
| 白矮星大气与光谱 | 1,733 | 27.6% |
| 激变变星与白矮星吸积 | 946 | 15.1% |
| 磁白矮星 | 553 | 8.8% |
| 白矮双星系统 | 406 | 6.5% |
| 白矮星内部结构 | 387 | 6.2% |
| 星团中的白矮星 | 307 | 4.9% |
| 白矮星震动与脉动 | 95 | 1.5% |

---

## 期刊分布 (Top 10)

| 期刊 | 篇数 |
|------|------|
| Monthly Notices of the Royal Astronomical Society | 2,327 |
| The Astrophysical Journal | 1,773 |
| Astronomy and Astrophysics | 1,228 |
| The Astronomical Journal | 264 |
| Research in Astronomy and Astrophysics | 91 |
| The Astrophysical Journal Supplement Series | 84 |
| Nature Astronomy | 68 |
| Nature | 65 |
| Astrophysics and Space Science | 52 |
| European Planetary Science Congress | 43 |

---

## 数据格式说明

**⚠️ 重要**: 检索获取的是**文献元数据**，包含：
- 标题 (title)
- 作者列表 (authors)
- 发表年份 (year)
- 期刊名称 (journal)
- 摘要 (abstract)
- Bibcode (ADS唯一标识)
- DOI
- 引用次数 (citations)
- 关键词 (keywords)

**不是 PDF 全文**！如需下载 PDF，需要通过 DOI 或 Bibcode 到相应期刊网站下载。

---

## 文件列表

| 文件名 | 内容 | 大小 |
|--------|------|------|
| `all_papers.json` | 全部6,281篇汇总 | 14.7 MB |
| `white_dwarf_evolution.json` | 白矮星演化模型 (1,854篇) | 4.3 MB |
| `white_dwarf_atmosphere.json` | 白矮星大气与光谱 (1,733篇) | 4.0 MB |
| `cataclysmic_variables.json` | 激变变星与白矮星吸积 (946篇) | 2.2 MB |
| `white_dwarf_magnetic.json` | 磁白矮星 (553篇) | 1.2 MB |
| `binary_white_dwarfs.json` | 白矮双星系统 (406篇) | 888 KB |
| `white_dwarf_interior.json` | 白矮星内部结构 (387篇) | 922 KB |
| `star_cluster_white_dwarfs.json` | 星团中的白矮星 (307篇) | 701 KB |
| `white_dwarf_asteroseismology.json` | 白矮星震动与脉动 (95篇) | 209 KB |

---

## 高被引论文 Top 10

1. **MESA Isochrones and Stellar Tracks (MIST). I. Solar-scaled Models** (2016 ApJ)
   - 被引: 2,785次 | 分类: 白矮星演化模型

2. **MESA Isochrones and Stellar Tracks (MIST) 0: Methods...** (2016 ApJS)
   - 被引: 1,877次 | 分类: 白矮星演化模型

3. **Mind Your Ps and Qs: The Interrelation between Period...** (2017 ApJS)
   - 被引: 1,283次 | 分类: 白矮星演化模型

4. **Gaia Data Release 2. Observational Hertzsprung-Russell diagrams** (2018 A&A)
   - 被引: 867次 | 分类: 白矮星演化模型

5. **The APOSTLE simulations: solutions to the Local Group's cosmic puzzles** (2016 MNRAS)
   - 被引: 575次 | 分类: 白矮星演化模型

6. **Mass loss of stars on the asymptotic giant branch...** (2018 A&ARv)
   - 被引: 513次 | 分类: 白矮星演化模型

7. **Asteroseismic constraints on the modes of nuclear burning...** (2016 ApJ)
   - 被引: 492次 | 分类: 白矮星演化模型

8. **Gaia Data Release 2: The celestial reference frame...** (2018 A&A)
   - 被引: 473次 | 分类: 白矮星演化模型

9. **Observational constraints on the origin of the elements...** (2021 MNRAS)
   - 被引: 418次 | 分类: 白矮星演化模型

10. **The new solar abundances - Part I: the implications...** (2009 MNRAS) - *虽早于2015但仍在检索中*
    - 被引: 388次 | 分类: 白矮星演化模型

---

## 使用建议

1. **数据分析**: 使用 Python 的 `json` 模块加载数据
   ```python
   import json
   with open('all_papers.json') as f:
       data = json.load(f)
   papers = data['papers']  # 获取论文列表
   ```

2. **筛选特定主题**: 按 `category` 字段过滤

3. **排序**: 按 `citations` (引用次数) 或 `year` (年份) 排序

4. **获取PDF**: 使用 `doi` 或 `bibcode` 访问期刊网站下载全文

---

*数据存储位置: `ads_papers_20260415_150007/`*
