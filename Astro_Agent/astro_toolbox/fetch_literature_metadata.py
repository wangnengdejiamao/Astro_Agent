#!/usr/bin/env python3
"""
通过 Crossref API 批量获取白矮星相关文献元数据。
由于 arXiv API 当前限流严重，此脚本优先获取文献清单和元数据，
为后续 PDF 下载做准备。

主题: 星团中的白矮星、白矮星双星、白矮星演化系统
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/Users/a1/Desktop/desi匹配/astro_toolbox/literature")
META_DIR = OUTPUT_DIR / "wd_literature_batch"
META_DIR.mkdir(parents=True, exist_ok=True)

# Crossref 查询策略
QUERIES = [
    ("white dwarf cluster", "wd_cluster"),
    ("white dwarf binary", "wd_binary"),
    ("double white dwarf", "dwd"),
    ("white dwarf evolution", "wd_evolution"),
    ("white dwarf cooling", "wd_cooling"),
    ("white dwarf merger", "wd_merger"),
    ("white dwarf supernova", "wd_sn"),
    ("white dwarf spectroscopy", "wd_spectroscopy"),
    ("pulsating white dwarf", "wd_pulsating"),
    ("magnetic white dwarf", "wd_magnetic"),
    ("white dwarf mass", "wd_mass"),
    ("white dwarf luminosity", "wd_luminosity"),
    ("cataclysmic variable white dwarf", "wd_cv"),
    ("white dwarf planetary", "wd_planetary"),
]

# 天文常见期刊 (用于优先级排序/过滤)
ASTRO_JOURNALS = {
    "The Astrophysical Journal", "The Astrophysical Journal Letters",
    "The Astrophysical Journal Supplement Series",
    "Monthly Notices of the Royal Astronomical Society",
    "Astronomy & Astrophysics", "Astronomy and Astrophysics",
    "Astronomical Journal", "The Astronomical Journal",
    "Physical Review D", "Physical Review Letters",
    "Nature", "Nature Astronomy", "Science",
    "Publications of the Astronomical Society of the Pacific",
    "Astronomy and Computing",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AstroToolbox/1.0; mailto:astro@example.com)"
}

MAX_PER_QUERY = 500  # 每个查询最多获取500条
REQUEST_DELAY = 1.5  # 秒，Crossref 建议不要过快


def crossref_search(query, offset=0, rows=100):
    """调用 Crossref API 搜索"""
    url = (
        f"https://api.crossref.org/works?"
        f"query={urllib.parse.quote(query)}"
        f"&rows={rows}&offset={offset}"
        f"&sort=relevance&order=desc"
        f"&select=DOI,title,author,container-title,issued,link,abstract,URL"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {})
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print(f"    Crossref 429, waiting 5s...")
            time.sleep(5)
            return None
        print(f"    HTTPError {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"    Error: {type(e).__name__}: {e}")
        return None


def extract_arxiv_id(item):
    """从 Crossref 条目中提取 arXiv ID"""
    doi = item.get("DOI", "")
    # arXiv DOI 格式: 10.48550/arXiv.XXXX.XXXXX
    if doi.startswith("10.48550/arXiv.") or doi.startswith("10.48550/arxiv."):
        return doi.split(".", 1)[1] if "." in doi else ""

    # 检查 URL
    url = item.get("URL", "")
    if "arxiv.org/abs/" in url:
        parts = url.split("arxiv.org/abs/")
        if len(parts) > 1:
            return parts[1].split("v")[0].split("?")[0].strip("/")

    # 检查 link 数组
    for link in item.get("link", []):
        lurl = link.get("URL", "")
        if "arxiv.org" in lurl:
            if "/abs/" in lurl:
                return lurl.split("/abs/")[1].split("v")[0].split("?")[0].strip("/")
            if "/pdf/" in lurl:
                return lurl.split("/pdf/")[1].replace(".pdf", "").split("v")[0].strip("/")

    return ""


def extract_pdf_url(item):
    """提取开放获取 PDF URL"""
    for link in item.get("link", []):
        if link.get("content-type") == "application/pdf":
            return link.get("URL", "")
    return ""


def parse_item(item, tag):
    """解析 Crossref 条目为统一格式"""
    title_list = item.get("title", [])
    title = title_list[0] if title_list else ""

    authors = []
    for a in item.get("author", [])[:10]:  # 最多取10个作者
        name = a.get("given", "") + " " + a.get("family", "")
        authors.append(name.strip())

    # 出版年份
    year = None
    issued = item.get("issued", {})
    if issued:
        parts = issued.get("date-parts", [[]])
        if parts and parts[0]:
            year = parts[0][0]

    # 期刊
    journals = item.get("container-title", [])
    journal = journals[0] if journals else ""

    doi = item.get("DOI", "")
    url = item.get("URL", "")
    arxiv_id = extract_arxiv_id(item)
    pdf_url = extract_pdf_url(item)
    if arxiv_id and not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "url": url,
        "arxiv_id": arxiv_id,
        "pdf_url": pdf_url,
        "tag": tag,
        "is_astro_journal": journal in ASTRO_JOURNALS,
        "raw": item,  # 保留原始数据以便后续使用
    }


def fetch_all_metadata(target_total=1200):
    """获取所有查询的元数据"""
    all_items = {}
    total_fetched = 0

    for query, tag in QUERIES:
        print(f"\n[搜索 Crossref] {tag}: {query}")
        offset = 0
        rows = 100
        max_per_query = 250
        query_items = []

        while offset < max_per_query:
            print(f"  offset={offset} ...")
            msg = crossref_search(query, offset=offset, rows=rows)
            if msg is None:
                print("  请求失败，跳过剩余")
                break

            items = msg.get("items", [])
            if not items:
                print("  无更多结果")
                break

            for item in items:
                parsed = parse_item(item, tag)
                if parsed["title"]:
                    query_items.append(parsed)

            offset += len(items)
            total_fetched += len(items)
            print(f"  本批 {len(items)} 条，本查询累计 {len(query_items)} 条")

            if len(items) < rows:
                break

            time.sleep(REQUEST_DELAY)

        # 去重合并
        for it in query_items:
            key = it["doi"] or it["arxiv_id"] or it["title"]
            if key not in all_items:
                all_items[key] = it
            else:
                if tag not in all_items[key].get("tags", []):
                    all_items[key].setdefault("tags", [all_items[key]["tag"]]).append(tag)

        print(f"  全局累计不重复: {len(all_items)} 条")
        if len(all_items) >= target_total:
            print(f"达到目标 {target_total}")
            break

    print(f"\n元数据获取完成: 总请求 {total_fetched} 条, 去重后 {len(all_items)} 条")
    return list(all_items.values())


def main():
    print("=" * 70)
    print("白矮星文献元数据批量获取工具")
    print("数据源: Crossref API")
    print("=" * 70)

    papers = fetch_all_metadata(target_total=1200)

    if not papers:
        print("未获取到任何文献")
        return

    # 排序：天文期刊优先，然后按年份降序
    papers.sort(key=lambda x: (-int(x.get("is_astro_journal") or 0), -(x.get("year") or 0)))

    # 保存完整元数据
    meta_path = META_DIR / "papers_metadata_crossref.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"\n完整元数据已保存: {meta_path}")

    # 生成简化的 CSV 清单
    import csv
    csv_path = META_DIR / "papers_index.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "year", "arxiv_id", "doi", "journal", "title", "authors", "pdf_url", "tag"])
        for i, p in enumerate(papers, 1):
            writer.writerow([
                i,
                p.get("year", ""),
                p.get("arxiv_id", ""),
                p.get("doi", ""),
                p.get("journal", ""),
                p.get("title", ""),
                "; ".join(p.get("authors", [])),
                p.get("pdf_url", ""),
                p.get("tag", ""),
            ])
    print(f"CSV 清单已保存: {csv_path}")

    # 统计
    arxiv_count = sum(1 for p in papers if p.get("arxiv_id"))
    astro_count = sum(1 for p in papers if p.get("is_astro_journal"))
    year_dist = {}
    for p in papers:
        y = p.get("year")
        if y:
            year_dist[y] = year_dist.get(y, 0) + 1

    print(f"\n统计:")
    print(f"  总文献: {len(papers)}")
    print(f"  含 arXiv ID: {arxiv_count} ({arxiv_count/len(papers)*100:.1f}%)")
    print(f"  天文期刊: {astro_count}")
    print(f"  年份分布: {dict(sorted(year_dist.items())[-10:])}")
    print("=" * 70)

    # 尝试下载有 arXiv ID 的 PDF（如果 arXiv 限流已解除）
    if arxiv_count > 0:
        print("\n尝试下载有 arXiv ID 的 PDF...")
        pdf_dir = META_DIR / "pdfs"
        pdf_dir.mkdir(exist_ok=True)
        success = 0
        failed = 0
        for i, p in enumerate(papers, 1):
            arxiv_id = p.get("arxiv_id")
            if not arxiv_id:
                continue
            pdf_path = pdf_dir / f"{arxiv_id}.pdf"
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                success += 1
                continue
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=30)
                data = resp.read()
                if len(data) > 1000:
                    with open(pdf_path, "wb") as f:
                        f.write(data)
                    success += 1
                    print(f"[{i}] ✓ {arxiv_id}")
                else:
                    failed += 1
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"[{i}] arXiv 限流中，停止下载。已下载 {success} 篇。")
                    break
                failed += 1
            except Exception as e:
                failed += 1
            if i % 5 == 0:
                time.sleep(2)
        print(f"PDF 下载: 成功 {success} 篇, 失败/跳过 {failed} 篇")


if __name__ == "__main__":
    main()
