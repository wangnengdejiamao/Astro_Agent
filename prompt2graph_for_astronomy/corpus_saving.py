import os
import json
import shutil
from typing import Tuple, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

PROJECT_DIR = os.getenv("PROJECT_DIR")


# 受保护的 corpus 名称列表，这些不能被删除
PROTECTED_CORPUS_NAMES = [
    "1-Dicarbonyl_Electrolyte",
    "2-Fluorinated_Hybrid_Diluent_Modulated_Electrolyte",
    "3-Quasi-Localized_High-Concentration_Electrolytes",
    "paper_mini"
]


def validate_corpus_format(corpus_content: str) -> Tuple[bool, str]:
    """
    校验corpus是否符合规范
    """
    try:
        data = json.loads(corpus_content)
    except json.JSONDecodeError as e:
        return False, f"JSON格式错误: {str(e)}"

    if not isinstance(data, list):
        return False, f"Corpus必须是一个JSON数组，当前类型: {type(data).__name__}"

    if len(data) == 0:
        return False, "Corpus数组不能为空"

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"第{i+1}个元素必须是对象(dict)，当前类型: {type(item).__name__}"

        if "title" not in item:
            return False, f"第{i+1}个元素缺少必需字段: 'title'"

        if "text" not in item:
            return False, f"第{i+1}个元素缺少必需字段: 'text'"

        if not isinstance(item["title"], str):
            return False, f"第{i+1}个元素的'title'字段必须是字符串，当前类型: {type(item['title']).__name__}"

        if not isinstance(item["text"], str):
            return False, f"第{i+1}个元素的'text'字段必须是字符串，当前类型: {type(item['text']).__name__}"

        if not item["title"].strip():
            return False, f"第{i+1}个元素的'title'字段不能为空"

        if not item["text"].strip():
            return False, f"第{i+1}个元素的'text'字段不能为空"

    return True, ""


def list_corpus_items() -> Dict[str, Any]:
    """
    返回所有语料项，同时兼容两种存储规则：
    - 规则B（单级目录）: input/文献文件夹/corpus_cleaned.json
    - 规则A（两级目录）: input/主题文件夹/文献子文件夹/corpus_cleaned.json

    一个文件夹可以同时属于两种规则（既有直接 corpus，又有子文件夹 corpus）。

    Returns:
        {
            "corpus_names": [所有可用语料ID列表，兼容旧格式],
            "items": [结构化数据，用于前端展示]
        }
    """
    input_dir = os.path.join(PROJECT_DIR, "input")
    if not os.path.exists(input_dir):
        return {"corpus_names": [], "items": []}

    corpus_names = []
    items = []

    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue

        # 规则B：单级目录，直接有 corpus_cleaned.json
        direct_corpus = os.path.join(path, "corpus_cleaned.json")
        has_direct = os.path.exists(direct_corpus)

        # 规则A：两级目录，遍历子文件夹
        children = []
        try:
            for sub_name in sorted(os.listdir(path)):
                sub_path = os.path.join(path, sub_name)
                if not os.path.isdir(sub_path):
                    continue
                sub_corpus = os.path.join(sub_path, "corpus_cleaned.json")
                if os.path.exists(sub_corpus):
                    child_id = f"{name}/{sub_name}"
                    corpus_names.append(child_id)
                    # 尝试读取语料标题
                    title = sub_name
                    try:
                        with open(sub_corpus, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list) and len(data) > 0:
                                first = data[0]
                                if isinstance(first, dict) and first.get('title'):
                                    title = first['title']
                    except Exception:
                        pass
                    children.append({
                        "id": child_id,
                        "name": sub_name,
                        "title": title,
                        "parent": name,
                        "type": "subfolder"
                    })
        except PermissionError:
            pass

        if has_direct:
            corpus_names.append(name)

        if has_direct and children:
            # 同时有两种：自身是 direct，又有子文件夹
            items.append({
                "id": name,
                "name": name,
                "parent": None,
                "type": "mixed",
                "has_direct": True,
                "children": children
            })
        elif has_direct:
            items.append({
                "id": name,
                "name": name,
                "parent": None,
                "type": "direct"
            })
        elif children:
            items.append({
                "id": name,
                "name": name,
                "parent": None,
                "type": "folder",
                "children": children
            })

    return {"corpus_names": corpus_names, "items": items}


def collect_corpus_from_folder(folder_name: str) -> str:
    """
    收集指定文件夹内的所有语料 JSON 文件，合并后保存到一个缓存文件。
    返回合并后的文件绝对路径。

    规则：
    - 单级语料文件夹：读取该文件夹内所有直接的 .json 文件
    - 主题文件夹：递归读取该文件夹下所有子文件夹中的 corpus_cleaned.json
    - 混合文件夹：同时读取直接的 .json 文件和递归读取子文件夹中的 corpus_cleaned.json
    """
    input_dir = os.path.join(PROJECT_DIR, "input")
    folder_path = os.path.join(input_dir, folder_name)

    # 安全检查：防止路径穿越
    real_input = os.path.realpath(input_dir)
    real_folder = os.path.realpath(folder_path)
    if not real_folder.startswith(real_input + os.sep) and real_folder != real_input:
        raise ValueError("Invalid folder path")

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_name}")

    all_items = []

    # 1. 收集该文件夹内所有直接的 .json 文件（仅文件，不包含子目录递归）
    try:
        for f in sorted(os.listdir(folder_path)):
            if not f.endswith('.json'):
                continue
            file_path = os.path.join(folder_path, f)
            if not os.path.isfile(file_path):
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    if isinstance(data, list):
                        all_items.extend(data)
                    else:
                        all_items.append(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
    except PermissionError:
        pass

    # 2. 递归收集所有子文件夹中的 corpus_cleaned.json
    # 跳过根目录本身，避免与步骤 1 重复
    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            continue
        if 'corpus_cleaned.json' not in files:
            continue
        corpus_path = os.path.join(root, 'corpus_cleaned.json')
        try:
            with open(corpus_path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    all_items.extend(data)
                else:
                    all_items.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

    if not all_items:
        raise ValueError(f"文件夹 '{folder_name}' 内没有找到有效的语料文件")

    # 保存到缓存目录
    cache_dir = os.path.join(input_dir, ".cache", "merged")
    os.makedirs(cache_dir, exist_ok=True)

    safe_name = folder_name.replace('/', '_').replace('\\', '_')
    merged_path = os.path.join(cache_dir, f"{safe_name}_merged.json")

    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    return merged_path


def check_corpus(corpus_name: str):
    """
    获取指定corpus的内容。corpus_name支持路径形式，如 '主题/子文件夹'。
    """
    # 安全校验：防止路径穿越
    safe_name = corpus_name.replace('..', '').strip('/')
    if not safe_name:
        raise FileNotFoundError(f"Invalid corpus name: {corpus_name}")

    corpus_path = os.path.join(PROJECT_DIR, "input", safe_name, "corpus_cleaned.json")
    # 确保最终路径在 input 目录下
    real_input = os.path.realpath(os.path.join(PROJECT_DIR, "input"))
    real_corpus = os.path.realpath(corpus_path)
    if not real_corpus.startswith(real_input + os.sep) and real_corpus != real_input:
        raise FileNotFoundError(f"Corpus file not found: {corpus_name}!")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file not found: {corpus_name}!")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_corpus(corpus_name: str, corpus_content: str, force_new: bool = False):
    """
    保存corpus到文件
    """
    # 安全校验
    safe_name = corpus_name.replace('..', '').strip('/')
    if not safe_name:
        raise ValueError("Invalid corpus name")

    # 检查是否为受保护的corpus（仅对一级目录有效）
    top_name = safe_name.split('/')[0]
    is_protected = top_name in PROTECTED_CORPUS_NAMES

    corpus_path = os.path.join(PROJECT_DIR, "input", safe_name, "corpus_cleaned.json")

    # 确保路径安全
    real_input = os.path.realpath(os.path.join(PROJECT_DIR, "input"))
    real_corpus = os.path.realpath(corpus_path)
    if not real_corpus.startswith(real_input + os.sep) and real_corpus != real_input:
        raise ValueError("Invalid corpus path")

    # 如果是受保护的corpus且文件已存在，不允许覆盖
    if is_protected and os.path.exists(corpus_path):
        raise ValueError(f"不能覆盖受保护的corpus: {top_name}")

    # 校验corpus格式
    is_valid, error_message = validate_corpus_format(corpus_content)
    if not is_valid:
        raise ValueError(f"Corpus格式不符合规范: {error_message}")

    # 确保目录存在
    os.makedirs(os.path.dirname(corpus_path), exist_ok=True)

    # 对于非受保护的corpus，如果force_new=True，允许覆盖
    if force_new and not is_protected:
        if os.path.exists(corpus_path):
            os.remove(corpus_path)

    if not os.path.exists(corpus_path):
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.write(corpus_content)
    else:
        raise FileExistsError(f"Corpus文件已存在: {corpus_name}，如需覆盖请设置 force_new=True")


def delete_corpus(corpus_name: str):
    """
    删除指定的corpus文件夹
    """
    safe_name = corpus_name.replace('..', '').strip('/')
    if not safe_name:
        raise ValueError("Invalid corpus name")

    top_name = safe_name.split('/')[0]
    if top_name in PROTECTED_CORPUS_NAMES:
        raise ValueError(f"不能删除受保护的corpus: {top_name}")

    corpus_dir = os.path.join(PROJECT_DIR, "input", safe_name)

    # 确保路径安全
    real_input = os.path.realpath(os.path.join(PROJECT_DIR, "input"))
    real_dir = os.path.realpath(corpus_dir)
    if not real_dir.startswith(real_input + os.sep) and real_dir != real_input:
        raise ValueError("Invalid corpus path")

    if os.path.exists(corpus_dir) and os.path.isdir(corpus_dir):
        shutil.rmtree(corpus_dir)
    else:
        raise FileNotFoundError(f"Corpus文件夹不存在: {corpus_name}!")


# ==================== 文件夹管理 ====================

def create_folder(parent_path: str, folder_name: str):
    """
    在指定路径下创建新文件夹

    Args:
        parent_path: 父目录相对路径（空字符串表示 input/ 根目录）
        folder_name: 新文件夹名称
    """
    safe_parent = parent_path.replace('..', '').strip('/')
    safe_name = folder_name.replace('..', '').replace('/', '').strip()

    if not safe_name:
        raise ValueError("文件夹名称不能为空")

    if safe_parent:
        target = os.path.join(PROJECT_DIR, "input", safe_parent, safe_name)
    else:
        target = os.path.join(PROJECT_DIR, "input", safe_name)

    # 路径安全校验
    real_input = os.path.realpath(os.path.join(PROJECT_DIR, "input"))
    real_target = os.path.realpath(target)
    if not real_target.startswith(real_input + os.sep) and real_target != real_input:
        raise ValueError("Invalid folder path")

    if os.path.exists(target):
        raise FileExistsError(f"文件夹已存在: {folder_name}")

    os.makedirs(target, exist_ok=True)
    return target


def rename_folder(old_path: str, new_name: str):
    """
    重命名文件夹

    Args:
        old_path: 原文件夹相对路径
        new_name: 新文件夹名称
    """
    safe_old = old_path.replace('..', '').strip('/')
    safe_new = new_name.replace('..', '').replace('/', '').strip()

    if not safe_old or not safe_new:
        raise ValueError("路径或名称不能为空")

    old_full = os.path.join(PROJECT_DIR, "input", safe_old)
    parent = os.path.dirname(old_full)
    new_full = os.path.join(parent, safe_new)

    # 路径安全校验
    real_input = os.path.realpath(os.path.join(PROJECT_DIR, "input"))
    real_old = os.path.realpath(old_full)
    real_new = os.path.realpath(new_full)
    if not real_old.startswith(real_input + os.sep):
        raise ValueError("Invalid folder path")
    if not real_new.startswith(real_input + os.sep) and real_new != real_input:
        raise ValueError("Invalid folder path")

    # 保护检查
    top_name = safe_old.split('/')[0]
    if top_name in PROTECTED_CORPUS_NAMES:
        raise ValueError(f"不能重命名受保护的corpus文件夹: {top_name}")

    if not os.path.exists(old_full):
        raise FileNotFoundError(f"文件夹不存在: {old_path}")

    if os.path.exists(new_full):
        raise FileExistsError(f"目标文件夹已存在: {new_name}")

    os.rename(old_full, new_full)
    return new_full


def delete_folder(folder_path: str):
    """
    删除指定文件夹

    Args:
        folder_path: 文件夹相对路径
    """
    safe_path = folder_path.replace('..', '').strip('/')
    if not safe_path:
        raise ValueError("路径不能为空")

    full_path = os.path.join(PROJECT_DIR, "input", safe_path)

    # 路径安全校验
    real_input = os.path.realpath(os.path.join(PROJECT_DIR, "input"))
    real_path = os.path.realpath(full_path)
    if not real_path.startswith(real_input + os.sep):
        raise ValueError("Invalid folder path")

    # 保护检查
    top_name = safe_path.split('/')[0]
    if top_name in PROTECTED_CORPUS_NAMES:
        raise ValueError(f"不能删除受保护的corpus文件夹: {top_name}")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    if not os.path.isdir(full_path):
        raise ValueError(f"不是文件夹: {folder_path}")

    shutil.rmtree(full_path)
