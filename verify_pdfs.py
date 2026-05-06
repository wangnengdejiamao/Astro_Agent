#!/usr/bin/env python3
"""
批量验证PDF内容准确性
"""

import glob
import json
import os
import re
import subprocess
import concurrent.futures
import threading

# 加载文献数据
json_files = glob.glob('literature/*/*.json')
all_papers = {}
for jf in json_files:
    with open(jf, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except:
            continue
    papers = data.get('papers', []) if isinstance(data, dict) else data
    for p in papers:
        if isinstance(p, dict):
            bc = p.get('bibcode', '')
            if bc:
                all_papers[bc] = p

pdfs = glob.glob('literature/pdfs/*.pdf')
lock = threading.Lock()
results = {
    'ok': [],
    'suspicious': [],
    'error': [],
    'scan_only': []
}

def verify_pdf(pdf_path):
    bc = os.path.basename(pdf_path).replace('.pdf', '')
    paper = all_papers.get(bc, {})
    title = paper.get('title', '')
    
    try:
        result = subprocess.run(['pdftotext', '-l', '1', pdf_path, '-'],
                              capture_output=True, text=True, timeout=15)
        text = result.stdout[:1000]
        word_count = len(text.split())
        
        # 检查是否是扫描版（几乎没有文字）
        if word_count < 10:
            with lock:
                results['scan_only'].append((bc, word_count))
            return
        
        # 检查标题匹配率
        title_words = [w.lower() for w in re.findall(r'\b\w+\b', title) if len(w) > 3]
        if len(title_words) >= 3:
            matched = sum(1 for w in title_words[:10] if w in text.lower())
            ratio = matched / min(len(title_words), 10)
        else:
            ratio = 1.0  # 短标题跳过
        
        # 检查期刊名或作者
        journal = paper.get('journal', '')
        authors = paper.get('authors', [])
        journal_match = False
        author_match = False
        
        if journal:
            journal_short = journal.lower()[:10]
            journal_match = journal_short in text.lower()
        
        if authors:
            for a in authors[:2]:
                surname = a.split(',')[0].strip().lower()
                if len(surname) > 2 and surname in text.lower():
                    author_match = True
                    break
        
        if ratio >= 0.3 or journal_match or author_match:
            with lock:
                results['ok'].append(bc)
        else:
            with lock:
                results['suspicious'].append((bc, ratio, title[:60]))
    except Exception as e:
        with lock:
            results['error'].append((bc, str(e)[:40]))

print(f"开始验证 {len(pdfs)} 个PDF...")
print(f"并发数: 10")

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(verify_pdf, pdfs)

print(f"\n验证完成!")
print(f"  正常: {len(results['ok'])} ({len(results['ok'])/len(pdfs)*100:.1f}%)")
print(f"  扫描版(无文字层): {len(results['scan_only'])} ({len(results['scan_only'])/len(pdfs)*100:.1f}%)")
print(f"  可疑(内容不匹配): {len(results['suspicious'])} ({len(results['suspicious'])/len(pdfs)*100:.1f}%)")
print(f"  解析错误: {len(results['error'])} ({len(results['error'])/len(pdfs)*100:.1f}%)")

if results['suspicious']:
    print(f"\n可疑文件列表:")
    for bc, ratio, title in sorted(results['suspicious'], key=lambda x: x[1]):
        print(f"  {bc}: 匹配率={ratio:.2f}, 标题={title}")

if results['scan_only']:
    print(f"\n扫描版PDF列表(前10个):")
    for bc, wc in results['scan_only'][:10]:
        print(f"  {bc}: {wc} words")

# 保存结果
with open('literature/pdf_verification_report.txt', 'w', encoding='utf-8') as f:
    f.write(f"PDF验证报告\n")
    f.write(f"总PDF数: {len(pdfs)}\n")
    f.write(f"正常: {len(results['ok'])}\n")
    f.write(f"扫描版: {len(results['scan_only'])}\n")
    f.write(f"可疑: {len(results['suspicious'])}\n")
    f.write(f"错误: {len(results['error'])}\n\n")
    if results['suspicious']:
        f.write("可疑文件:\n")
        for bc, ratio, title in results['suspicious']:
            f.write(f"  {bc}: 匹配率={ratio:.2f}\n")

print(f"\n报告已保存: literature/pdf_verification_report.txt")
