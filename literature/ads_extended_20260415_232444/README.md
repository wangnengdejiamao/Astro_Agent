# ADS 白矮星文献检索完整报告

## 检索概况
- **检索时间**: 2026-04-15 23:56:59
- **检索目标**: 白矮星、星团、白矮双星相关文献（排除超新星）
- **时间范围**: 2000-2025 (25年)
- **期刊限制**: ApJ, ApJS, AJ, MNRAS, Nature, Science, A&A
- **原始总计**: **11,598 篇**
- **去重后**: **7,610 篇**  
- **跨分类重复**: 3,988 篇

---

## 按主题分类统计

| 分类 | 篇数 | 备注 |
|------|------|------|
| 白矮星演化模型 | 2,600 | 冷却序列、光度函数、质量函数 |
| 白矮星大气与光谱 | 2,200 | 大气模型、光谱分析 |
| 激变变星与吸积 | 1,800 | CV、吸积盘、新星 |
| 白矮双星系统 | 256 | 双简并星 |
| 星团中的白矮星 | 637 | 球状星团、疏散星团 |
| 白矮星内部结构 | 457 | 结晶化、相分离 |
| 磁白矮星 | 1,000 | 磁场、极向星 |
| 白矮星脉动 | 149 | ZZ Ceti, DAV, DBV |
| Gaia白矮星 | 499 | Gaia观测相关 |
| 热亚矮星 | 1,000 | sdB, sdO, EHB |
| 白矮星形成 | 1,000 | 前身星、行星状星云 |

---

## 数据文件清单

| 文件名 | 大小 | 内容 |
|--------|------|------|
| `ALL_PAPERS.json` | 16 MB | 全部7,610篇汇总（已去重） |
| `白矮星演化模型.json` | 5.1 MB | 2,600篇 |
| `白矮星大气与光谱.json` | 4.3 MB | 2,200篇 |
| `激变变星与吸积.json` | 3.5 MB | 1,800篇 |
| `磁白矮星.json` | 2.0 MB | 1,000篇 |
| `白矮星形成.json` | 2.0 MB | 1,000篇 |
| `热亚矮星.json` | 2.0 MB | 1,000篇 |
| `星团中的白矮星.json` | 1.3 MB | 637篇 |
| `Gaia白矮星.json` | 1.1 MB | 499篇 |
| `白矮星内部结构.json` | 903 KB | 457篇 |
| `白矮双星系统.json` | 493 KB | 256篇 |
| `白矮星脉动.json` | 276 KB | 149篇 |

---

## 重要说明

**数据格式**: JSON元数据（非PDF），包含：
- bibcode, title, authors, year, journal
- abstract, keywords, doi, citations
- category（所属分类）

**重复文献**: 一篇文献可能涉及多个主题（如"球状星团中的白矮星双星"），因此：
- 各分类文件独立保存（有重复）
- `ALL_PAPERS.json` 已去重（7,610篇唯一文献）

**获取PDF全文**: 使用DOI或Bibcode访问各期刊网站下载

---

## 使用示例

```python
import json

# 加载数据
with open('ALL_PAPERS.json') as f:
    data = json.load(f)

# 获取论文列表
papers = data['papers']

# 筛选特定年份
recent = [p for p in papers if p['year'] and int(p['year']) >= 2015]

# 筛选高被引
highly_cited = [p for p in papers if p.get('citations', 0) > 100]

# 按分类筛选
evolution_papers = [p for p in papers if '演化' in p['category']]
```

---

*数据位置: `ads_extended_20260415_232444/`*
