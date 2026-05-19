#!/usr/bin/env python3
"""
通过 NASA ADS API 批量搜索白矮星文献，并下载 arXiv PDF。

主题: 星团中的白矮星、白矮星双星、白矮星演化系统

需要 ~/.ads/dev_key 中保存 ADS API token。
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# 输出目录
OUTPUT_DIR = Path("/Users/a1/Desktop/desi匹配/astro_toolbox/literature")
META_DIR = OUTPUT_DIR / "wd_literature_ads"
META_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR = META_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ADS API Token
ADS_TOKEN_PATH = Path.home() / ".ads" / "dev_key"
ADS_TOKEN = os.environ.get("ADS_DEV_KEY", "")
if not ADS_TOKEN and ADS_TOKEN_PATH.exists():
    ADS_TOKEN = ADS_TOKEN_PATH.read_text().strip()

if not ADS_TOKEN:
    print("错误: 未找到 ADS API token。请先保存到 ~/.ads/dev_key")
    sys.exit(1)

ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"

# 查询策略
QUERIES = [
    ("abs:\"white dwarf\" AND cluster", "wd_cluster"),
    ("abs:\"white dwarf\" AND binary", "wd_binary"),
    ("abs:\"double white dwarf\" OR DWD", "dwd"),
    ("abs:\"white dwarf\" AND (evolution OR cooling)", "wd_evolution"),
    ("abs:\"white dwarf\" AND merger", "wd_merger"),
    ("abs:\"white dwarf\" AND (supernova OR \"type Ia\")", "wd_sn"),
    ("abs:\"white dwarf\" AND (pulsation OR pulsating)", "wd_pulsating"),
    ("abs:\"white dwarf\" AND magnetic", "wd_magnetic"),
    ("abs:\"white dwarf\" AND (GAIA OR spectroscopy OR photometry)", "wd_survey"),
]

# 配置
MAX_PER_QUERY = 500
BATCH_SIZE = 100  # ADS 每次最多 2000，但 100 更稳定
ADS_REQUEST_DELAY = 1.0  # ADS API 间隔（有 token 时限流很宽松）
ARXIV_DOWNLOAD_DELAY = 2.0  # arXiv PDF 下载间隔
MAX_RETRIES = 5
RATE_LIMIT_WAIT = 60

HEADERS = {
    "Authorization": f"Bearer {ADS_TOKEN}",
    "User-Agent": "Mozilla/5.0 (AstroToolbox/1.0; mailto:astro@example.com)",
}


def ads_search(query, start=0, rows=100):
    """调用 ADS API 搜索"""
    fields = "bibcode,title,author,year,arxiv,doi,abstract,property,identifier,pubdate"
    url = (
        f"{ADS_SEARCH_URL}?"
        f"q={urllib.parse.quote(query)}"
        f"&fl={urllib.parse.quote(fields)}"
        f"&rows={rows}&start={start}"
        f"&sort=date%20desc"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(MAX_RETRIES):
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", {})
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"    ADS 429, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    ADS HTTP {e.code}: {e.reason}")
            return {}
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"    Error: {type(e).__name__}: {e}, retry in {wait}s...")
            time.sleep(wait)
    return {}


def extract_arxiv_id(doc):
    """从 ADS 文档中提取 arXiv ID"""
    # 1. 检查 identifier 数组
    for ident in doc.get("identifier", []):
        if ident.startswith("arXiv:"):
            return ident.replace("arXiv:", "").strip()
    # 2. 检查 arxiv 字段
    arxiv_field = doc.get("arxiv", "")
    if arxiv_field:
        return arxiv_field.replace("arXiv:", "").strip()
    # 3. 从 bibcode 提取（如 2026arXiv260412494B）
    bib = doc.get("bibcode", "")
    if "arXiv" in bib:
        # 格式: 2026arXiv260412494B -> 2604.12494
        import re
        m = re.search(r'arXiv(\d{4}\.\d{4,})', bib, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def has_arxiv_openaccess(doc):
    """检查是否有开放获取的 arXiv 预印本"""
    props = doc.get("property", [])
    return "EPRINT_OPENACCESS" in props or "OPENACCESS" in props


def is_article(doc):
    """检查是否为正式论文（过滤会议摘要、数据目录等）"""
    props = doc.get("property", [])
    bib = doc.get("bibcode", "")
    # 排除 VizieR 数据目录、会议摘要等
    if "yCat" in bib or "meeting" in bib.lower():
        return False
    return "ARTICLE" in props


def search_all_queries(target_total=1200):
    """执行所有 ADS 查询并合并去重"""
    all_papers = {}
    total_fetched = 0

    for query, tag in QUERIES:
        print(f"\n[ADS 搜索] {tag}: {query}")
        offset = 0
        query_papers = []

        while offset < MAX_PER_QUERY:
            print(f"  offset={offset} ...")
            resp = ads_search(query, start=offset, rows=BATCH_SIZE)
            if not resp:
                print("  请求失败，跳过")
                break

            docs = resp.get("docs", [])
            num_found = resp.get("numFound", 0)
            if not docs:
                print(f"  无更多结果 (总数: {num_found})")
                break

            for doc in docs:
                if not is_article(doc):
                    continue
                arxiv_id = extract_arxiv_id(doc)
                bibcode = doc.get("bibcode", "")
                key = bibcode  # 按 bibcode 去重
                if key not in all_papers:
                    all_papers[key] = {
                        "bibcode": bibcode,
                        "title": doc.get("title", [""])[0] if doc.get("title") else "",
                        "authors": doc.get("author", []),
                        "year": doc.get("year", ""),
                        "pubdate": doc.get("pubdate", ""),
                        "doi": doc.get("doi", []),
                        "arxiv_id": arxiv_id,
                        "has_arxiv_oa": has_arxiv_openaccess(doc),
                        "abstract": doc.get("abstract", ""),
                        "tags": [tag],
                    }
                else:
                    if tag not in all_papers[key]["tags"]:
                        all_papers[key]["tags"].append(tag)
                query_papers.append(doc)

            offset += len(docs)
            total_fetched += len(docs)
            print(f"  本批 {len(docs)} 条，本查询累计 {len(query_papers)} 条，全局不重复 {len(all_papers)} 条")

            if len(docs) < BATCH_SIZE:
                break

            time.sleep(ADS_REQUEST_DELAY)

        print(f"  查询完成: {tag} -> {len(query_papers)} 条")
        if len(all_papers) >= target_total:
            print(f"达到目标 {target_total}")
            break

    print(f"\nADS 搜索完成: 总请求 {total_fetched} 条, 去重后 {len(all_papers)} 条")
    return list(all_papers.values())


def download_pdf(paper, output_dir, timeout=60):
    """下载单篇 arXiv PDF（不使用代理）"""
    arxiv_id = paper.get("arxiv_id", "")
    if not arxiv_id:
        return False, "no_arxiv_id"

    pdf_path = output_dir / f"{arxiv_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        return True, "exists"

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # 创建不使用代理的 opener
    no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = no_proxy_opener.open(req, timeout=timeout)
            data = resp.read()
            if len(data) < 1000:
                return False, f"too_small ({len(data)} bytes)"
            with open(pdf_path, "wb") as f:
                f.write(data)
            return True, f"ok ({len(data)} bytes)"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, f"404"
            if e.code == 429 or e.code == 503:
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"    arXiv rate limited, wait {wait}s...")
                time.sleep(wait)
                continue
            wait = 10 * (attempt + 1)
            time.sleep(wait)
        except Exception as e:
            wait = 5 * (attempt + 1)
            time.sleep(wait)
    return False, f"failed after {MAX_RETRIES} retries"


def main():
    print("=" * 70)
    print("NASA ADS + arXiv 白矮星文献批量下载")
    print("=" * 70)

    meta_path = META_DIR / "papers_metadata_ads.json"

    # 1. ADS 搜索（如果元数据已存在则跳过）
    if meta_path.exists():
        print(f"检测到已有元数据文件: {meta_path}")
        print("跳过 ADS 搜索，直接加载现有元数据...")
        with open(meta_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        print(f"已加载 {len(papers)} 篇文献")
    else:
        papers = search_all_queries(target_total=1200)
        if not papers:
            print("未找到任何文献")
            return
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"\n元数据已保存: {meta_path} ({len(papers)} 篇)")

    # 统计
    arxiv_count = sum(1 for p in papers if p.get("arxiv_id"))
    oa_count = sum(1 for p in papers if p.get("has_arxiv_oa"))
    print(f"  含 arXiv ID: {arxiv_count} 篇")
    print(f"  arXiv 开放获取: {oa_count} 篇")

    # 3. 下载 arXiv PDF
    print(f"\n开始下载 PDF 到: {PDF_DIR}")
    success_count = 0
    fail_count = 0
    fail_list = []

    for i, p in enumerate(papers, 1):
        status, msg = download_pdf(p, PDF_DIR)
        if status:
            success_count += 1
            indicator = "✓"
        else:
            fail_count += 1
            indicator = "✗"
            fail_list.append({"arxiv_id": p.get("arxiv_id", ""), "title": p["title"], "reason": msg})

        print(f"[{i}/{len(papers)}] {indicator} {p.get('arxiv_id','N/A')} | {p['title'][:50]}... | {msg}")

        if i % 5 == 0:
            time.sleep(ARXIV_DOWNLOAD_DELAY)

    # 4. 保存 manifest
    manifest = {
        "total_papers": len(papers),
        "with_arxiv": arxiv_count,
        "downloaded": success_count,
        "failed": fail_count,
        "output_dir": str(PDF_DIR),
        "failed_list": fail_list,
    }
    manifest_path = META_DIR / "download_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("下载完成!")
    print(f"总计文献: {len(papers)} 篇")
    print(f"含 arXiv: {arxiv_count} 篇")
    print(f"成功下载: {success_count} 篇")
    print(f"下载失败: {fail_count} 篇")
    print(f"PDF 目录: {PDF_DIR}")
    print(f"清单文件: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断。已下载的文件保留。")
        sys.exit(0)
