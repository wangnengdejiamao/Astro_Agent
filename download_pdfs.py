#!/usr/bin/env python3
"""
批量下载缺失的文献PDF
通过ADS link_gateway获取arXiv PDF
优先下载较新的文献（更可能有arXiv预印本）
"""

import json
import glob
import os
import re
import requests
import time
import concurrent.futures
import threading
from datetime import datetime

PDF_DIR = "literature/pdfs"
MAX_WORKERS = 10
REQUEST_DELAY = 0.5
TIMEOUT = 45
MAX_RETRIES = 2

print_lock = threading.Lock()

# 统计
total_success = [0]
total_failed = [0]
total_skipped = [0]

def safe_print(msg):
    with print_lock:
        print(msg)
        with open("download.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()

def get_existing_pdfs():
    """获取已存在的PDF列表 - 匹配所有文件名格式"""
    pdfs = glob.glob(f"{PDF_DIR}/*.pdf")
    existing = set()
    for p in pdfs:
        basename = os.path.basename(p)
        existing.add(basename)
    return existing

def bibcode_to_pdf_exists(bibcode, existing_set):
    """检查某个bibcode是否已有对应PDF"""
    # 直接匹配 bibcode.pdf
    if f"{bibcode}.pdf" in existing_set:
        return True
    return False

def load_all_papers():
    """加载所有文献（去重），按年份倒序排列"""
    json_files = glob.glob('literature/*/*.json')
    all_papers = {}
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        papers = data.get('papers', []) if isinstance(data, dict) else data
        for p in papers:
            if isinstance(p, dict):
                bibcode = p.get('bibcode', '')
                if bibcode and bibcode not in all_papers:
                    all_papers[bibcode] = p
    # 按年份倒序排列（新文献优先）
    sorted_papers = sorted(all_papers.items(), 
                          key=lambda x: x[1].get('year', '0000'), 
                          reverse=True)
    return sorted_papers

def download_pdf(bibcode, retry=0):
    """下载单个文献的PDF"""
    url = f"https://ui.adsabs.harvard.edu/link_gateway/{bibcode}/EPRINT_PDF"
    filepath = f"{PDF_DIR}/{bibcode}.pdf"
    
    if os.path.exists(filepath):
        return True, "already_exists"
    
    try:
        resp = requests.get(url, allow_redirects=True, timeout=TIMEOUT)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('application/pdf'):
            with open(filepath, 'wb') as f:
                f.write(resp.content)
            return True, f"{len(resp.content)}"
        else:
            content_type = resp.headers.get('Content-Type', '')
            if 'html' in content_type.lower() and retry < MAX_RETRIES:
                time.sleep(2 ** retry)
                return download_pdf(bibcode, retry + 1)
            return False, f"status={resp.status_code}"
    except requests.exceptions.Timeout:
        if retry < MAX_RETRIES:
            time.sleep(2 ** retry)
            return download_pdf(bibcode, retry + 1)
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:50]

def worker_task(args):
    """工作线程任务"""
    idx, total, bibcode = args
    time.sleep(REQUEST_DELAY * (idx % MAX_WORKERS) / MAX_WORKERS)
    
    success, info = download_pdf(bibcode)
    
    if success:
        if info == "already_exists":
            total_skipped[0] += 1
            safe_print(f"[{idx+1}/{total}] ○ {bibcode} (已存在)")
        else:
            total_success[0] += 1
            safe_print(f"[{idx+1}/{total}] ✓ {bibcode} ({info} bytes)")
    else:
        total_failed[0] += 1
        safe_print(f"[{idx+1}/{total}] ✗ {bibcode} -> {info}")
    
    # 每100篇报告一次汇总
    if (idx + 1) % 100 == 0:
        safe_print(f"--- 进度 [{idx+1}/{total}] 成功:{total_success[0]} 失败:{total_failed[0]} 跳过:{total_skipped[0]} ---")
    
    time.sleep(REQUEST_DELAY)
    return success, bibcode

def main():
    os.makedirs(PDF_DIR, exist_ok=True)
    
    # 清空日志
    with open("download.log", "w", encoding="utf-8") as f:
        f.write("")
    
    existing = get_existing_pdfs()
    sorted_papers = load_all_papers()
    
    # 筛选缺失PDF的文献
    missing = []
    for bibcode, paper in sorted_papers:
        if not bibcode_to_pdf_exists(bibcode, existing):
            missing.append(bibcode)
    
    total_papers = len(sorted_papers)
    already_have = total_papers - len(missing)
    
    safe_print(f"{'=' * 60}")
    safe_print(f"文献PDF批量下载工具")
    safe_print(f"{'=' * 60}")
    safe_print(f"总文献数: {total_papers}")
    safe_print(f"已有PDF: {already_have}")
    safe_print(f"待下载: {len(missing)}")
    safe_print(f"并发线程: {MAX_WORKERS}")
    safe_print(f"请求间隔: {REQUEST_DELAY}秒")
    safe_print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print(f"{'=' * 60}")
    
    if not missing:
        safe_print("所有文献PDF已下载完成！")
        return
    
    start_time = time.time()
    failed_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tasks = [(i, len(missing), bc) for i, bc in enumerate(missing)]
        futures = {executor.submit(worker_task, t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                success, bibcode = future.result()
                if not success:
                    failed_list.append(bibcode)
            except Exception as e:
                safe_print(f"任务异常: {e}")
    
    elapsed = time.time() - start_time
    
    if failed_list:
        with open("literature/failed_downloads.txt", "w", encoding="utf-8") as f:
            for bc in failed_list:
                f.write(bc + "\n")
    
    final_pdfs = len(glob.glob(f"{PDF_DIR}/*.pdf"))
    safe_print(f"\n{'=' * 60}")
    safe_print(f"下载完成!")
    safe_print(f"成功下载: {total_success[0]}")
    safe_print(f"已存在跳过: {total_skipped[0]}")
    safe_print(f"失败/无arXiv: {total_failed[0]}")
    safe_print(f"最终PDF总数: {final_pdfs}")
    safe_print(f"耗时: {elapsed/60:.1f} 分钟")
    safe_print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if failed_list:
        safe_print(f"失败列表已保存到: literature/failed_downloads.txt")
    safe_print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
