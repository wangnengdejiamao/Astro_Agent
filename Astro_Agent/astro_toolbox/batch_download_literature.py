#!/usr/bin/env python3
"""
批量下载白矮星相关文献 (arXiv)
主题: 星团中的白矮星、白矮星双星、白矮星演化系统

注意:
- arXiv API 对请求频率有限制, 脚本已内置限速和自动重试
- 请勿使用代理访问 arXiv, 代理IP容易被限流
- 预计下载1000篇需要 2-5 小时
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from xml.etree import ElementTree as ET

# 输出目录
OUTPUT_DIR = Path("/Users/a1/Desktop/desi匹配/astro_toolbox/literature")

# arXiv API 命名空间
ARXIV_NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'arxiv': 'http://arxiv.org/schemas/atom'
}

# 查询策略: 多个查询组合以覆盖不同子主题
QUERIES = [
    # 星团中的白矮星
    ("cat:astro-ph* AND (\"white dwarf\" OR \"white dwarfs\") AND (cluster OR clusters OR \"globular cluster\" OR \"open cluster\")", "wd_cluster"),
    # 白矮星双星 / 双白矮星
    ("cat:astro-ph* AND (\"white dwarf\" OR \"white dwarfs\") AND (binary OR binaries OR \"double white dwarf\" OR DWD)", "wd_binary"),
    # 白矮星演化 / 冷却
    ("cat:astro-ph* AND (\"white dwarf\" OR \"white dwarfs\") AND (evolution OR cooling OR \"stellar evolution\" OR \"cooling sequence\")", "wd_evolution"),
    # 白矮星并合 / 超新星 Ia
    ("cat:astro-ph* AND (\"white dwarf\" OR \"white dwarfs\") AND (merger OR merging OR \"type Ia\" OR supernova)", "wd_merger_sn"),
    # 更宽泛的白矮星巡天/观测
    ("cat:astro-ph* AND (\"white dwarf\" OR \"white dwarfs\") AND (spectroscopy OR photometry OR GAIA OR SDSS OR LAMOST)", "wd_survey"),
]

# 配置
MAX_RETRIES = 5
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 60
API_DELAY_SECONDS = 25      # arXiv API 请求间隔 (保守设置避免限流)
DOWNLOAD_DELAY_SECONDS = 2  # PDF 下载间隔
RATE_LIMIT_WAIT = 120       # 遇到限流时等待秒数


def build_opener(use_proxy=False):
    """构建 urllib opener. arXiv 强烈建议不使用代理."""
    if use_proxy:
        proxy = os.environ.get('https_proxy', 'http://127.0.0.1:7890')
        if proxy:
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy,
                'https': proxy,
            })
            opener = urllib.request.build_opener(proxy_handler)
        else:
            opener = urllib.request.build_opener()
    else:
        opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
    ]
    return opener


def is_rate_limited(data):
    """检查响应是否表示被限流"""
    if not data:
        return False
    text = data.decode('utf-8', errors='ignore').lower()
    return 'rate exceeded' in text or 'too many requests' in text or '429' in text


def arxiv_search(query, start=0, max_results=200):
    """调用 arXiv API 搜索, 自动处理限流"""
    url = (
        f"https://export.arxiv.org/api/query?"
        f"search_query={urllib.parse.quote(query)}"
        f"&start={start}&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    opener = build_opener(use_proxy=False)
    for attempt in range(MAX_RETRIES):
        try:
            resp = opener.open(url, timeout=READ_TIMEOUT)
            data = resp.read()
            if is_rate_limited(data):
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"  arXiv rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code == 503:
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"  arXiv HTTP {e.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HTTPError {e.code}: {e.reason}")
            raise
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  Error ({type(e).__name__}): {e}, retry in {wait}s...")
            time.sleep(wait)
    return None


def parse_arxiv_entries(xml_bytes):
    """解析 arXiv API 返回的 XML"""
    if not xml_bytes:
        return []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}")
        return []

    entries = []
    for entry in root.findall('atom:entry', ARXIV_NS):
        id_el = entry.find('atom:id', ARXIV_NS)
        if id_el is None:
            continue
        arxiv_id = id_el.text.strip().split('/abs/')[-1]
        arxiv_id_clean = arxiv_id.split('v')[0]

        title = entry.find('atom:title', ARXIV_NS)
        title_text = title.text.strip() if title is not None and title.text else ""

        summary = entry.find('atom:summary', ARXIV_NS)
        summary_text = summary.text.strip() if summary is not None and summary.text else ""

        published = entry.find('atom:published', ARXIV_NS)
        published_text = published.text.strip() if published is not None else ""

        authors = []
        for author in entry.findall('atom:author', ARXIV_NS):
            name = author.find('atom:name', ARXIV_NS)
            if name is not None and name.text:
                authors.append(name.text.strip())

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"

        categories = []
        for cat in entry.findall('arxiv:primary_category', ARXIV_NS):
            cat_term = cat.get('term', '')
            if cat_term:
                categories.append(cat_term)
        for cat in entry.findall('atom:category', ARXIV_NS):
            cat_term = cat.get('term', '')
            if cat_term and cat_term not in categories:
                categories.append(cat_term)

        entries.append({
            'arxiv_id': arxiv_id_clean,
            'title': title_text,
            'authors': authors,
            'published': published_text,
            'summary': summary_text,
            'pdf_url': pdf_url,
            'categories': categories,
        })
    return entries


def search_all_queries(target_total=1200):
    """执行所有查询并合并去重"""
    all_papers = {}
    total_found = 0

    for query, tag in QUERIES:
        print(f"\n[搜索] {tag}: {query}")
        papers_for_query = []
        start = 0
        batch_size = 25   # 减小批次大小以降低限流风险
        max_per_query = 500

        while start < max_per_query:
            print(f"  获取 offset={start} ...")
            xml = arxiv_search(query, start=start, max_results=batch_size)
            if xml is None:
                print("  失败，跳过此查询剩余部分")
                break

            entries = parse_arxiv_entries(xml)
            if not entries:
                print("  无更多结果")
                break

            papers_for_query.extend(entries)
            start += len(entries)
            print(f"  本批获取 {len(entries)} 篇, 累计 {len(papers_for_query)} 篇")

            if len(entries) < batch_size:
                break

            time.sleep(API_DELAY_SECONDS)

        print(f"  本查询总计: {len(papers_for_query)} 篇")
        total_found += len(papers_for_query)

        for p in papers_for_query:
            aid = p['arxiv_id']
            if aid not in all_papers:
                all_papers[aid] = p
                all_papers[aid]['tags'] = [tag]
            else:
                if tag not in all_papers[aid]['tags']:
                    all_papers[aid]['tags'].append(tag)

        print(f"  全局累计不重复: {len(all_papers)} 篇")
        if len(all_papers) >= target_total:
            print(f"已达到目标数量 {target_total}")
            break

    print(f"\n搜索完成: 总获取 {total_found} 篇, 去重后 {len(all_papers)} 篇")
    return list(all_papers.values())


def download_pdf(paper, output_dir, opener, timeout=60):
    """下载单篇 PDF"""
    arxiv_id = paper['arxiv_id']
    pdf_path = output_dir / f"{arxiv_id}.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        return True, "exists"

    url = paper['pdf_url']
    for attempt in range(MAX_RETRIES):
        try:
            resp = opener.open(url, timeout=timeout)
            data = resp.read()
            if len(data) < 1000:
                return False, f"too_small ({len(data)} bytes)"
            with open(pdf_path, 'wb') as f:
                f.write(data)
            return True, f"ok ({len(data)} bytes)"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, f"404"
            if e.code == 429 or e.code == 503:
                wait = RATE_LIMIT_WAIT * (attempt + 1)
                print(f"    PDF download rate limited, wait {wait}s...")
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
    print("白矮星文献批量下载工具")
    print("主题: 星团白矮星 / 白矮星双星 / 白矮星演化")
    print("数据源: arXiv (export.arxiv.org)")
    print("=" * 70)

    # 1. 搜索文献
    papers = search_all_queries(target_total=1200)

    if not papers:
        print("未找到任何文献，退出")
        return

    # 2. 保存元数据
    meta_dir = OUTPUT_DIR / "wd_literature_batch"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = meta_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / "papers_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print(f"\n元数据已保存: {meta_path}")
    print(f"  共 {len(papers)} 篇不重复文献")

    # 3. 下载 PDF
    print(f"\n开始下载 PDF 到: {pdf_dir}")
    print("(按 Ctrl+C 可以中断，下次运行会跳过已下载的文件)\n")

    opener = build_opener(use_proxy=False)
    success_count = 0
    fail_count = 0
    fail_list = []

    for i, p in enumerate(papers, 1):
        status, msg = download_pdf(p, pdf_dir, opener)
        if status:
            success_count += 1
            indicator = "✓"
        else:
            fail_count += 1
            indicator = "✗"
            fail_list.append({'arxiv_id': p['arxiv_id'], 'title': p['title'], 'reason': msg})

        print(f"[{i}/{len(papers)}] {indicator} {p['arxiv_id']} | {p['title'][:60]}... | {msg}")

        if i % 5 == 0:
            time.sleep(DOWNLOAD_DELAY_SECONDS)

    # 4. 保存 manifest
    manifest = {
        'total_papers': len(papers),
        'downloaded': success_count,
        'failed': fail_count,
        'output_dir': str(pdf_dir),
        'failed_list': fail_list,
    }
    manifest_path = meta_dir / "download_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("下载完成!")
    print(f"总计文献: {len(papers)} 篇")
    print(f"成功下载: {success_count} 篇")
    print(f"下载失败: {fail_count} 篇")
    print(f"PDF 目录: {pdf_dir}")
    print(f"清单文件: {manifest_path}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断了下载。已下载的文件保留，下次运行会跳过已下载文件。")
        sys.exit(0)
