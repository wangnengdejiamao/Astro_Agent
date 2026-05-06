#!/usr/bin/env python3
"""
从 corpus.json 构建 chunks 并保存到文件
"""

import json
import os
import re
import sys
import uuid
import ast
import threading
from typing import List, Dict, Tuple, Any
from concurrent import futures
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.logger import logger
except ImportError:
    logger = None

try:
    import nanoid
except ImportError:
    if logger:
        logger.warning("警告: nanoid 未安装，将使用 uuid 生成 ID")
    else:
        print("警告: nanoid 未安装，将使用 uuid 生成 ID")
    nanoid = None

# ==================== Chunk长度配置 ====================
CHUNK_MIN_CHARS = 1500      # 最小chunk长度（字符数）
CHUNK_IDEAL_MIN_CHARS = 2500  # 理想最小长度
CHUNK_IDEAL_MAX_CHARS = 4000  # 理想最大长度
CHUNK_MAX_CHARS = 6000       # 最大chunk长度
CHUNK_OVERLAP_RATE = 0.1     # 重叠系数，示例默认10%

def generate_chunk_id() -> str:
    """生成随机chunk ID"""
    if nanoid:
        return nanoid.generate(size=8)
    else:
        # 使用 uuid 作为备选方案
        uuid_str = format(uuid.uuid4().int & (1 << 64) - 1, '016x')[:8]
        result = []
        for char in uuid_str:
            if char.isdigit():
                result.append(chr(ord('a') + int(char)))
            else:
                result.append(chr(ord('k') + ord(char) - ord('a')))
        return ''.join(result)


def find_split_points(text: str) -> List[int]:
    """
    找到文本中的分割点，优先在以下位置分割：
    1. ".\r\n\r\n" - 句号后跟两个换行符
    2. ".\r\n" 后跟大写字母 - 句号+换行+大写字母（新段落开始）
    3. ". " 后跟大写字母 - 句号+空格+大写字母（新句开始）
    """
    split_points = []
    
    # 模式1: ".\r\n\r\n" - 句号后跟两个换行符（段落分隔）
    pattern1 = r'\.\r\n\r\n'
    for match in re.finditer(pattern1, text):
        split_points.append(match.end())
    
    # 模式2: ".\r\n" 后跟大写字母
    pattern2 = r'\.\r\n([A-Z])'
    for match in re.finditer(pattern2, text):
        split_points.append(match.start() + 1)
    
    # 模式3: ". " 后跟大写字母
    pattern3 = r'\. ([A-Z][a-z])'
    for match in re.finditer(pattern3, text):
        pos = match.start() + 1
        if pos > 2:
            prev_char = text[pos - 2:pos]
            if not re.search(r'[a-z]\.\s$', text[max(0, pos-10):pos]):
                split_points.append(pos)
    
    # 去重并排序
    split_points = sorted(list(set(split_points)))
    return split_points

# def find_split_points(text: str) -> List[int]:
#     """
#     找到文本中的合法分割点 —— 必须位于句子末尾之后的空白处（不在句子内部）。
#     返回的是 split index（split 点为切割后 chunk 的起始位置，index 为字符位置）。
#     优先匹配双换行（段落边界），其次匹配以句末标点 [.?!] 结束并跟随空白的边界。
#     """
#     split_points = set()
#     # 优先：句号 + 双换行（段落分隔）
#     for m in re.finditer(r'\.\r?\n\r?\n', text):
#         split_points.add(m.end())

#     # 匹配句末标点（可能带引号、括号）后跟至少一个空白（空格或换行）
#     # 例如: "word." + " " or "word." + "\n"
#     # 保证 split 点位于句子末尾之后（即不会切入句子内部）
#     pattern = r'[\.!?][\"\'\)\]\}]*\s+'
#     for m in re.finditer(pattern, text):
#         split_points.add(m.end())

#     # 移除 0 和末尾之外的不合法点，排序返回
#     pts = sorted(p for p in split_points if 0 < p < len(text))
#     return pts


def find_best_split_point(text: str, start: int, end: int, target_pos: int) -> int:
    """在目标位置附近寻找最佳分割点（句号、段落分隔等）"""
    search_start = max(start, target_pos - 200)
    search_end = min(end, target_pos + 200)
    search_text = text[search_start:search_end]
    
    patterns = [
        r'\.\r\n\r\n',           # 句号+双换行（段落分隔）
        r'\.\r\n',               # 句号+单换行
        r'\.\s+[A-Z]',           # 句号+空格+大写字母
        r'\.\s',                 # 句号+空格
    ]
    
    best_pos = target_pos
    best_score = -1
    
    for pattern in patterns:
        for match in re.finditer(pattern, search_text):
            pos = search_start + match.end()
            distance = abs(pos - target_pos)
            score = 1000 / (distance + 1)
            if score > best_score:
                best_score = score
                best_pos = pos
    
    return best_pos


def smart_split_text(text: str, min_chunks: int = 5, max_chunks: int = 10) -> List[str]:
    """
    智能分割文本（保证分割点在句尾空白处），并根据 CHUNK_OVERLAP_RATE 引入重叠。
    重叠通过向前扩展 chunk 起点来实现，但扩展后的起点会向左对齐到最近的合法 split_point（或 0）。
    """
    split_points = find_split_points(text)

    # 如果没有分割点，按长度均匀分割
    if not split_points:
        text_length = len(text)
        chunk_count = max(min_chunks, 1)
        chunk_size = text_length // chunk_count
        chunks = []
        for i in range(chunk_count):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < chunk_count - 1 else text_length
            chunks.append(text[start:end].strip())
        return [c for c in chunks if c]

    # 构造区间起止（合法起点列表）
    starts = [0] + split_points
    ends = split_points + [len(text)]
    msg = f"找到 {len(split_points)} 个初始分割点\n{split_points}"
    if logger:
        logger.debug(msg)
    else:
        print(msg)

    text_length = len(text)
    
    # 合并或选取 split_points 使区间数量在 [min_chunks, max_chunks]
    # 按照字符长度均匀分割，选择的切割点尽可能接近均分text_length的位置
    if len(split_points) >= max_chunks:
        # 目标chunk数量
        target_chunks = max_chunks
        # 每个chunk的理想字符长度
        ideal_chunk_length = text_length / target_chunks
        
        # 计算理想的分割位置（均分text_length）
        ideal_positions = []
        for i in range(1, target_chunks):
            ideal_pos = int(i * ideal_chunk_length)
            ideal_positions.append(ideal_pos)
        
        # 对于每个理想位置，在split_points中找到最接近的点
        sel = []
        for ideal_pos in ideal_positions:
            # 找到split_points中最接近ideal_pos的点
            closest_point = min(split_points, key=lambda x: abs(x - ideal_pos))
            # 确保不重复且保持顺序
            if closest_point not in sel:
                sel.append(closest_point)
        
        # 排序并去重
        sel = sorted(set(sel))
        starts = [0] + sel
        ends = sel + [text_length]    
    # 生成初始 chunks（无重叠）
    intervals = [(starts[i], ends[i]) for i in range(len(starts))]
    chunks_raw = [text[s:e].strip() for s, e in intervals if s < e]

    if logger:
        logger.debug(f"初始分割为 {len(chunks_raw)} 个 chunks（无重叠）")
        logger.debug(f"start point: {starts}")
        logger.debug(f"end point: {ends}")
        logger.debug(f"重叠前chunks区间: {intervals}")
    else:
        print(f"初始分割为 {len(chunks_raw)} 个 chunks（无重叠）")
        print(f"start point: {starts}")
        print(f"end point: {ends}")
        print(f"重叠前chunks区间: {intervals}")

    # 应用重叠：对于每个 chunk（除了第一个），尝试向左扩展 start 以包含 overlap，
    # 扩展后的 start 必须对齐到最近的合法 split point（包括 0）。
    if CHUNK_OVERLAP_RATE and 0.0 < CHUNK_OVERLAP_RATE < 0.5:
        allowed_starts = [0] + split_points
        new_intervals = []
        for (orig_s, orig_e) in intervals:
            length = orig_e - orig_s
            desired_overlap = int(length * CHUNK_OVERLAP_RATE)
            if orig_s == 0 or desired_overlap <= 0:
                new_intervals.append((orig_s, orig_e))
                continue
            desired_start = max(0, orig_s - desired_overlap)
            # 找到 allowed_starts 中 <= desired_start 的最大点
            candidate = max((p for p in allowed_starts if p <= desired_start), default=0)
            # print(f"original_start: {orig_s}, desired_start: {desired_start}, chosen_start: {candidate}, original_end: {orig_e}")
            # 如果没有满足的，则保留原始 orig_s
            new_intervals.append((candidate, orig_e))
        # 避免区间重叠导致顺序错误：确保 intervals 单调递增且 start < end
        merged_intervals = []
        for s, e in new_intervals:
            if s >= e:
                continue
            merged_intervals.append((s, e))
        chunks = [text[s:e].strip() for s, e in merged_intervals if s < e]
        msg = f"重叠后chunks区间: {merged_intervals}"
        if logger:
            logger.debug(msg)
        else:
            print(msg)
    else:
        chunks = chunks_raw

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _enforce_max_chunk_size(chunks: List[str], max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """
    后处理：将超过 max_chars 的 chunk 按句子边界硬性切分，
    确保每个 chunk 都不超过 max_chars。
    """
    result = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
            continue
        # 需要切分
        start = 0
        text_len = len(chunk)
        while start < text_len:
            # 如果剩余部分已经不超过 max_chars，直接作为最后一块
            if text_len - start <= max_chars:
                result.append(chunk[start:].strip())
                break
            end = start + max_chars
            # 在 end 附近找最佳分割点（优先句子边界）
            best_end = find_best_split_point(chunk, start, text_len, end)
            # 确保至少前进最小长度，避免无限循环或产生过小块
            if best_end <= start + CHUNK_MIN_CHARS:
                best_end = end
            result.append(chunk[start:best_end].strip())
            start = best_end
    return [c for c in result if c]


def chunk_text(text: Any, datasets_no_chunk: List[str] = None, dataset_name: str = None) -> Tuple[List[str], Dict[str, str], str]:
    """
    将文本分割成 chunks
    
    Args:
        text: 文本内容，可以是字符串或字典（包含 'title' 和 'text'）
        datasets_no_chunk: 不需要分割的数据集列表
        dataset_name: 数据集名称
    
    Returns:
        Tuple of (chunks, chunk2id, title)
    """
    if datasets_no_chunk is None:
        datasets_no_chunk = []
    
    # 提取 title 和 text
    title = ""
    if isinstance(text, dict):
        title = text.get('title', '')
        text_content = text.get('text', '')
        if not text_content:
            return [], {}, title
        full_text = text_content.strip()
    else:
        full_text = str(text).strip()
        if not full_text:
            return [], {}, title
    
    # 如果数据集在 no_chunk 列表中，返回整个文本作为单个 chunk
    if dataset_name and dataset_name in datasets_no_chunk:
        if len(full_text) >= CHUNK_MIN_CHARS:
            chunks = [full_text]
        else:
            return [], {}, title
    else:
        # 根据文本长度动态调整chunk数量范围
        # 核心改动：让 chunk 数量由 CHUNK_MAX_CHARS 主导，而非固定上限 12
        text_length = len(full_text)
        ideal_chunks = max(1, text_length // CHUNK_IDEAL_MAX_CHARS)
        min_chunks = max(3, ideal_chunks // 2)
        max_chunks = max(10, ideal_chunks + 5)
        # 同时设置一个合理的上限，避免极端情况产生过多小块
        max_chunks = min(max_chunks, 300)
        
        chunks = smart_split_text(full_text, min_chunks=min_chunks, max_chunks=max_chunks)
        
        # 后处理：强制确保每个 chunk 不超过 CHUNK_MAX_CHARS
        chunks = _enforce_max_chunk_size(chunks, max_chars=CHUNK_MAX_CHARS)
        
        msg = f"文本长度: {text_length} 字符，分割为 {len(chunks)} 个 chunks（强制上限 {CHUNK_MAX_CHARS} 字符）"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    if not chunks:
        return [], {}, title
    
    # 生成 chunk IDs
    chunk2id = {}
    for idx, chunk in enumerate(chunks):
        try:
            chunk_id = generate_chunk_id()
            chunk2id[chunk_id] = chunk
        except Exception as e:
            msg = f"警告: 生成 chunk ID 失败: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            continue
    
    return chunks, chunk2id, title


def chunk_corpus(input_file: str, output_file: str, dataset_name: str = None, datasets_no_chunk: List[str] = None, output_path: str = None):
    """
    读取corpus.json，对每篇文章进行chunk分割，输出到txt文件
    
    Args:
        input_file: 输入的corpus.json文件路径
        output_file: 输出的txt文件路径
        dataset_name: 数据集名称
        datasets_no_chunk: 不需要分割的数据集列表
    """
    if datasets_no_chunk is None:
        datasets_no_chunk = []
    
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
    except Exception as e:
        msg = f"错误: 无法读取文件 {input_file}: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return
    
    # 处理每篇文章
    output_lines = []
    all_chunks = {}
    stats = {
        'total_articles': len(corpus),
        'total_chunks': 0,
        'chunk_lengths': [],
        'articles_with_issues': []
    }
    
    for idx, article in enumerate(corpus):
        title = article.get('title', '')
        text = article.get('text', '')
        
        if not text:
            continue
        
        text_length = len(text)
        
        # 分割文本
        chunks, chunk2id, doc_title = chunk_text(
            article, 
            datasets_no_chunk=datasets_no_chunk,
            dataset_name=dataset_name
        )
        msg = f"文章 {idx + 1}/{len(corpus)}: '{title[:30]}...' 长度 {text_length} 字符，生成 {len(chunks)} 个 chunks"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        if not chunks or not chunk2id:
            continue
        
        # 检查chunk质量并保存
        article_issues = []
        for chunk_id, chunk in chunk2id.items():
            chunk_len = len(chunk)
            stats['chunk_lengths'].append(chunk_len)
            
            if chunk_len < CHUNK_MIN_CHARS:
                article_issues.append(f"chunk太短: {chunk_len}字符")
            elif chunk_len > CHUNK_MAX_CHARS:
                article_issues.append(f"chunk太长: {chunk_len}字符")
            
            # 保存 chunk 数据
            chunk_data = {
                'title': doc_title or title,
                'text': chunk
            }
            all_chunks[chunk_id] = chunk_data
            
            # 生成输出行
            output_line = f"id: {chunk_id}\tChunk: {chunk_data}"
            output_lines.append(output_line)
        
        if article_issues:
            stats['articles_with_issues'].append({
                'title': title[:50],
                'issues': article_issues,
                'chunk_count': len(chunks)
            })
        
        stats['total_chunks'] += len(chunks)
        
        if (idx + 1) % 10 == 0:
            msg = f"处理进度: {idx + 1}/{len(corpus)} 篇文章, 已生成 {stats['total_chunks']} 个 chunks"
            if logger:
                logger.info(msg)
            else:
                print(msg)
    
    # 写入输出文件
    target_path = output_path or output_file
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    # 输出统计信息
    header_line = "=" * 60
    if logger:
        logger.info(header_line)
        logger.info("处理完成！")
        logger.info(header_line)
        logger.info(f"文章总数: {stats['total_articles']}")
        logger.info(f"总chunk数: {stats['total_chunks']}")
        logger.debug(f"统计详情: {stats}")
    else:
        print(f"\n{header_line}")
        print("处理完成！")
        print(header_line)
        print(f"文章总数: {stats['total_articles']}")
        print(f"总chunk数: {stats['total_chunks']}")
        print(stats)

    if stats['chunk_lengths']:
        avg_len = sum(stats['chunk_lengths']) / len(stats['chunk_lengths'])
        min_len = min(stats['chunk_lengths'])
        max_len = max(stats['chunk_lengths'])
        msg_lines = [
            "Chunk长度统计:",
            f"  - 平均长度: {avg_len:.0f} 字符",
            f"  - 最小长度: {min_len} 字符",
            f"  - 最大长度: {max_len} 字符",
            f"  - 理想范围: {CHUNK_IDEAL_MIN_CHARS}-{CHUNK_IDEAL_MAX_CHARS} 字符",
        ]
        if logger:
            for line in msg_lines:
                logger.info(line)
        else:
            for line in msg_lines:
                print(line)
    
    if stats['articles_with_issues']:
        summary = f"{len(stats['articles_with_issues'])} 篇文章的chunk存在长度问题"
        if logger:
            logger.warning(summary)
            for issue in stats['articles_with_issues'][:5]:
                logger.warning(f"  - {issue['title']}: {', '.join(issue['issues'])}")
        else:
            print(f"\n警告: {summary}")
            for issue in stats['articles_with_issues'][:5]:
                print(f"  - {issue['title']}: {', '.join(issue['issues'])}")
    
    footer = f"输出文件: {output_file}"
    if logger:
        logger.info(footer)
        logger.info(header_line)
    else:
        print(f"\n{footer}")
        print(f"{header_line}\n")


def get_chunks(corpus_path: str, dataset_name: str, output_path: str = None, datasets_no_chunk: List[str] = None):
    """
    从 corpus.json 生成 chunks 并保存

    Args:
        corpus_path: corpus.json 文件路径
        dataset_name: 数据集名称
        output_path: 可选的输出文件路径；未提供时写入默认目录
        datasets_no_chunk: 不需要分割的数据集列表

    Returns:
        输出文件路径
    """
    if logger:
        logger.info("======== 开始Chunk切分流程 ========")
    # output_dir = "/Dspace/pku-projects/dev-projects/lab-agents/graph_construction/output/chunks"
    project_root = os.getenv("PROJECT_DIR") or os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(project_root, 'output', dataset_name)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_file = output_path
    else:
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = os.path.join(dataset_root, run_id)
        os.makedirs(run_dir, exist_ok=True)
        output_file = os.path.join(run_dir, f"{dataset_name}.txt")

    msg_start = f"开始处理数据集: {dataset_name}"
    msg_in = f"输入文件: {corpus_path}"
    msg_out = f"输出文件: {output_file}"
    if logger:
        logger.info(msg_start)
        logger.info(msg_in)
        logger.info(msg_out)
    else:
        print(msg_start)
        print(msg_in)
        print(msg_out)

    chunk_corpus(corpus_path, output_file, dataset_name, datasets_no_chunk, output_path=output_path)

    if logger:
        logger.info("======== Chunk切分流程完成 ========")
    return output_file


if __name__ == '__main__':
    # 示例用法
    import sys
    
    if len(sys.argv) < 4:
        usage = "用法: python get_chunks.py <corpus_path> <dataset_name> <output_path>"
        example = "示例: python get_chunks.py data/uploaded/demo/corpus.json demo /Dspace/pku-projects/dev-projects/lab-agents/prompt2graph/chunks/demo.txt"
        if logger:
            logger.info(usage)
            logger.info(example)
        else:
            print(usage)
            print(example)
        # sys.exit(1)
    
    corpus_path = sys.argv[1]
    dataset_name = sys.argv[2]
    # output_path = sys.argv[3]
    
    # get_chunks(corpus_path, dataset_name, output_path)
    get_chunks(corpus_path, dataset_name)
    
    
    # python3 get_chunks.py /Dspace/pku-projects/dev-projects/lab-agents/graph_construction/output/paper_mini/corpus_cleaned.json paper_mini
    
    # python3 get_chunks.py /Dspace/pku-projects/dev-projects/lab-agents/graph_construction/output/electrolytes/corpus_cleaned.json electrolytes
