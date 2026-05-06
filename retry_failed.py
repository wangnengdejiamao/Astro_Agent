#!/usr/bin/env python3
"""
重新下载失败的文献PDF
使用ADS API获取准确的下载链接
"""

import requests
import urllib.parse
import time
import concurrent.futures
import threading
import os
import glob
import json
from datetime import datetime

TOKEN = "gnU0UJDvPnLDotwmhijqk4M8JGHCS653kUQRX2Bx"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}
PDF_DIR = "literature/pdfs"
MAX_WORKERS = 10
REQUEST_DELAY = 1.0
TIMEOUT = 45

print_lock = threading.Lock()
success_count = [0]
failed_count = [0]
still_failed = []

def safe_print(msg):
    with print_lock:
        print(msg)
        with open("retry.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

def get_arxiv_id_from_ads(bibcode):
    """通过ADS API查询arXiv ID"""
    url = f"https://api.adsabs.harvard.edu/v1/search/query?q=bibcode:{bibcode}&fl=bibcode,identifier,doi,property,esources"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            docs = data.get('response', {}).get('docs', [])
            if docs:
                doc = docs[0]
                identifiers = doc.get('identifier', [])
                for ident in identifiers:
                    if 'arXiv' in ident:
                        return ident
                # 从doi中提取
                dois = doc.get('doi', [])
                for doi in dois:
                    if 'arXiv' in doi:
                        parts = doi.split('arXiv.')
                        if len(parts) > 1:
                            return f"arXiv:{parts[1]}"
                
                # 检查property是否有ARXIV
                props = doc.get('property', [])
                esources = doc.get('esources', [])
                if 'ARXIV' in props or 'EPRINT_PDF' in esources:
                    # 尝试用export获取BibTeX中的eprint
                    return query_export_for_arxiv(bibcode)
        return None
    except Exception as e:
        return None

def query_export_for_arxiv(bibcode):
    """通过ADS export获取arXiv ID"""
    url = f"https://api.adsabs.harvard.edu/v1/export/bibtex"
    payload = {"bibcode": [bibcode], "sort": "date desc"}
    try:
        resp = requests.post(url, headers={**HEADERS, "Content-Type": "application/json"}, 
                           json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            bibtex = data.get('export', '')
            import re
            m = re.search(r'eprint\s*=\s*\{([^}]+)\}', bibtex)
            if m:
                return m.group(1)
    except:
        pass
    return None

def download_from_arxiv(arxiv_id):
    """通过arXiv直接下载PDF"""
    arxiv_id = arxiv_id.replace('arXiv:', '').strip()
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('application/pdf'):
            return True, resp.content
    except:
        pass
    return False, None

def download_from_ads_eprint(bibcode):
    """通过ADS EPRINT_PDF下载"""
    url = f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/EPRINT_PDF"
    try:
        resp = requests.get(url, allow_redirects=True, timeout=TIMEOUT)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('application/pdf'):
            return True, resp.content
    except:
        pass
    return False, None

def download_pdf(bibcode):
    """尝试多种方式下载PDF"""
    filepath = f"{PDF_DIR}/{bibcode}.pdf"
    if os.path.exists(filepath):
        return True, "already_exists"
    
    # 方式1: ADS EPRINT_PDF
    success, content = download_from_ads_eprint(bibcode)
    if success:
        with open(filepath, 'wb') as f:
            f.write(content)
        return True, f"ads_eprint:{len(content)}"
    
    # 方式2: 通过ADS API获取arXiv ID，然后直接下载
    arxiv_id = get_arxiv_id_from_ads(bibcode)
    if arxiv_id:
        success, content = download_from_arxiv(arxiv_id)
        if success:
            with open(filepath, 'wb') as f:
                f.write(content)
            return True, f"arxiv:{arxiv_id}:{len(content)}"
    
    return False, f"no_source"

def worker_task(args):
    idx, total, bibcode = args
    time.sleep(REQUEST_DELAY * (idx % MAX_WORKERS) / MAX_WORKERS)
    
    success, info = download_pdf(bibcode)
    
    if success:
        if info == "already_exists":
            safe_print(f"[{idx+1}/{total}] ○ {bibcode} (已存在)")
        else:
            success_count[0] += 1
            safe_print(f"[{idx+1}/{total}] ✓ {bibcode} ({info})")
    else:
        failed_count[0] += 1
        still_failed.append(bibcode)
        safe_print(f"[{idx+1}/{total}] ✗ {bibcode} -> {info}")
    
    if (idx + 1) % 50 == 0:
        safe_print(f"--- 进度 [{idx+1}/{total}] 成功:{success_count[0]} 失败:{failed_count[0]} ---")
    
    time.sleep(REQUEST_DELAY)
    return success, bibcode

def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    
    # 清空日志
    with open("retry.log", "w", encoding="utf-8") as f:
        f.write("")
    
    # 读取失败列表
    with open('literature/failed_downloads.txt', 'r') as f:
        failed = [line.strip() for line in f if line.strip()]
    
    # 过滤掉已有PDF的
    existing = set()
    for p in glob.glob(f"{PDF_DIR}/*.pdf"):
        basename = os.path.basename(p)
        name = basename.replace('.pdf', '')
        existing.add(name)
    
    to_download = [b for b in failed if b not in existing]
    
    safe_print(f"{'=' * 60}")
    safe_print(f"失败文献重新下载")
    safe_print(f"{'=' * 60}")
    safe_print(f"失败总数: {len(failed)}")
    safe_print(f"已有PDF跳过: {len(failed) - len(to_download)}")
    safe_print(f"待重新下载: {len(to_download)}")
    safe_print(f"并发线程: {MAX_WORKERS}")
    safe_print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"{'=' * 60}")
    
    if not to_download:
        safe_print("没有需要重新下载的文献！")
        return
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [(i, len(to_download), bc) for i, bc in enumerate(to_download)]
        futures = {executor.submit(worker_task, t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                safe_print(f"任务异常: {e}")
    
    elapsed = time.time() - start_time
    
    # 保存仍然失败的列表
    if still_failed:
        with open("literature/still_failed.txt", "w", encoding="utf-8") as f:
            for bc in still_failed:
                f.write(bc + "\n")
    
    final_pdfs = len(glob.glob(f"{PDF_DIR}/*.pdf"))
    safe_print(f"\n{'=' * 60}")
    safe_print(f"重新下载完成!")
    safe_print(f"成功下载: {success_count[0]}")
    safe_print(f"仍然失败: {len(still_failed)}")
    safe_print(f"最终PDF总数: {final_pdfs}")
    safe_print(f"耗时: {elapsed/60:.1f} 分钟")
    safe_print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if still_failed:
        safe_print(f"仍然失败列表: literature/still_failed.txt")
    safe_print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
