import os
import json
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


PROJECT_DIR = os.getenv("PROJECT_DIR")
# print(f"PROJECT_DIR: {PROJECT_DIR}")

def save_schema(schema_name: str, schema_content: str, force_new: bool = False):
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    if force_new:
        if os.path.exists(schema_path):
            os.remove(schema_path)
    if not os.path.exists(schema_path):
        with open(schema_path, 'w') as f:
            f.write(schema_content)

def save_prompt(prompt_name: str, prompt_content: str, force_new: bool = False):
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    if force_new:
        if os.path.exists(prompt_path):
            os.remove(prompt_path)
    if not os.path.exists(prompt_path):
        with open(prompt_path, 'w') as f:
            f.write(prompt_content)
            
def save_schema_and_prompt(schema_name: str,schema_content: str, prompt_name: str, prompt_content: str, force_new: bool = False):
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    
    if force_new:
        if os.path.exists(schema_path):
            os.remove(schema_path)
        if os.path.exists(prompt_path):
            os.remove(prompt_path)
    
    if not os.path.exists(schema_path):
        with open(schema_path, 'w') as f:
            f.write(schema_content)
    if not os.path.exists(prompt_path):
        with open(prompt_path, 'w') as f:
            f.write(prompt_content)


# 增删改查schema和prompt，其中默认 electrolytes 和 electrolytes 为默认schema和prompt 不可以被删除和修改

# 默认的schema和prompt名称，受保护，不能被删除或修改
DEFAULT_SCHEMA_NAME = "electrolytes"
DEFAULT_PROMPT_NAME = "electrolytes"

def list_schemas():
    """列出所有可用的schema名称"""
    schemas_dir = f"{PROJECT_DIR}/schemas"
    if not os.path.exists(schemas_dir):
        return []
    schemas = []
    for filename in os.listdir(schemas_dir):
        if filename.endswith('.json'):
            schema_name = filename[:-5]  # 移除.json后缀
            schemas.append(schema_name)
    return sorted(schemas)

def list_prompts():
    """列出所有可用的prompt名称"""
    prompts_dir = f"{PROJECT_DIR}/prompts"
    if not os.path.exists(prompts_dir):
        return []
    prompts = []
    for filename in os.listdir(prompts_dir):
        if filename.endswith('.txt'):
            prompt_name = filename[:-4]  # 移除.txt后缀
            prompts.append(prompt_name)
    return sorted(prompts)

def get_schema(schema_name: str) -> Optional[dict]:
    """获取指定schema的内容"""
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    if not os.path.exists(schema_path):
        return None
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"读取schema文件失败: {e}")

def get_prompt(prompt_name: str) -> Optional[str]:
    """获取指定prompt的内容"""
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    if not os.path.exists(prompt_path):
        return None
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"读取prompt文件失败: {e}")

def create_schema(schema_name: str, schema_content: dict, force_overwrite: bool = False):
    """创建新的schema
    
    Args:
        schema_name: schema名称
        schema_content: schema内容（字典格式）
        force_overwrite: 如果schema已存在，是否强制覆盖
        
    Raises:
        ValueError: 如果schema_name是默认schema且force_overwrite为False
        ValueError: 如果schema已存在且force_overwrite为False
    """
    if schema_name == DEFAULT_SCHEMA_NAME and not force_overwrite:
        raise ValueError(f"默认schema '{DEFAULT_SCHEMA_NAME}' 不能被修改，如需修改请设置 force_overwrite=True")
    
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    
    if os.path.exists(schema_path) and not force_overwrite:
        raise ValueError(f"Schema '{schema_name}' 已存在，如需覆盖请设置 force_overwrite=True")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema_content, f, indent=4, ensure_ascii=False)

def create_prompt(prompt_name: str, prompt_content: str, force_overwrite: bool = False):
    """创建新的prompt
    
    Args:
        prompt_name: prompt名称
        prompt_content: prompt内容（字符串格式）
        force_overwrite: 如果prompt已存在，是否强制覆盖
        
    Raises:
        ValueError: 如果prompt_name是默认prompt且force_overwrite为False
        ValueError: 如果prompt已存在且force_overwrite为False
    """
    if prompt_name == DEFAULT_PROMPT_NAME and not force_overwrite:
        raise ValueError(f"默认prompt '{DEFAULT_PROMPT_NAME}' 不能被修改，如需修改请设置 force_overwrite=True")
    
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    
    if os.path.exists(prompt_path) and not force_overwrite:
        raise ValueError(f"Prompt '{prompt_name}' 已存在，如需覆盖请设置 force_overwrite=True")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt_content)

def update_schema(schema_name: str, schema_content: dict):
    """更新已存在的schema
    
    Args:
        schema_name: schema名称
        schema_content: 新的schema内容
        
    Raises:
        ValueError: 如果schema_name是默认schema
        FileNotFoundError: 如果schema不存在
    """
    if schema_name == DEFAULT_SCHEMA_NAME:
        raise ValueError(f"默认schema '{DEFAULT_SCHEMA_NAME}' 不能被修改")
    
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema '{schema_name}' 不存在")
    
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema_content, f, indent=4, ensure_ascii=False)

def update_prompt(prompt_name: str, prompt_content: str):
    """更新已存在的prompt
    
    Args:
        prompt_name: prompt名称
        prompt_content: 新的prompt内容
        
    Raises:
        ValueError: 如果prompt_name是默认prompt
        FileNotFoundError: 如果prompt不存在
    """
    if prompt_name == DEFAULT_PROMPT_NAME:
        raise ValueError(f"默认prompt '{DEFAULT_PROMPT_NAME}' 不能被修改")
    
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt '{prompt_name}' 不存在")
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt_content)

def delete_schema(schema_name: str):
    """删除指定的schema
    
    Args:
        schema_name: schema名称
        
    Raises:
        ValueError: 如果schema_name是默认schema
        FileNotFoundError: 如果schema不存在
    """
    if schema_name == DEFAULT_SCHEMA_NAME:
        raise ValueError(f"默认schema '{DEFAULT_SCHEMA_NAME}' 不能被删除")
    
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema '{schema_name}' 不存在")
    
    os.remove(schema_path)

def delete_prompt(prompt_name: str):
    """删除指定的prompt
    
    Args:
        prompt_name: prompt名称
        
    Raises:
        ValueError: 如果prompt_name是默认prompt
        FileNotFoundError: 如果prompt不存在
    """
    if prompt_name == DEFAULT_PROMPT_NAME:
        raise ValueError(f"默认prompt '{DEFAULT_PROMPT_NAME}' 不能被删除")
    
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt '{prompt_name}' 不存在")
    
    os.remove(prompt_path)

def schema_exists(schema_name: str) -> bool:
    """检查schema是否存在"""
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    return os.path.exists(schema_path)

def prompt_exists(prompt_name: str) -> bool:
    """检查prompt是否存在"""
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    return os.path.exists(prompt_path)


# ==================== 分阶段 Prompt 文件夹支持 ====================

STAGE_FILES = [
    "stage1_entity_recognition.txt",
    "stage2_relation_extraction.txt",
    "stage3_attribute_extraction.txt",
    "stage4_node_accuracy.txt",
    "stage4_triple_support.txt",
]

def list_staged_prompts():
    """列出所有分阶段 prompt 文件夹"""
    prompts_dir = f"{PROJECT_DIR}/prompts"
    if not os.path.exists(prompts_dir):
        return []
    
    staged_prompts = []
    for item in os.listdir(prompts_dir):
        item_path = os.path.join(prompts_dir, item)
        if os.path.isdir(item_path) and item != "customized_example":
            # 检查是否包含 stage 文件
            has_stage_files = any(
                os.path.exists(os.path.join(item_path, stage_file))
                for stage_file in STAGE_FILES
            )
            if has_stage_files:
                staged_prompts.append(item)
    
    return sorted(staged_prompts)

def get_staged_prompt_files(prompt_folder: str) -> dict:
    """获取分阶段 prompt 文件夹中的文件列表和内容概览
    
    Args:
        prompt_folder: prompt 文件夹名称
        
    Returns:
        dict: 包含文件夹名称、文件列表和每个文件的概览信息
    """
    folder_path = f"{PROJECT_DIR}/prompts/{prompt_folder}"
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Prompt 文件夹 '{prompt_folder}' 不存在")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"'{prompt_folder}' 不是文件夹")
    
    files_info = []
    for stage_file in STAGE_FILES:
        file_path = os.path.join(folder_path, stage_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 获取文件前100字符作为概览
                    preview = content[:100] + "..." if len(content) > 100 else content
                    files_info.append({
                        "filename": stage_file,
                        "exists": True,
                        "preview": preview,
                        "size": len(content)
                    })
            except Exception as e:
                files_info.append({
                    "filename": stage_file,
                    "exists": True,
                    "preview": f"读取失败: {e}",
                    "size": 0
                })
        else:
            files_info.append({
                "filename": stage_file,
                "exists": False,
                "preview": "",
                "size": 0
            })
    
    return {
        "folder_name": prompt_folder,
        "files": files_info
    }

def get_staged_prompt_content(prompt_folder: str, stage_file: str) -> str:
    """获取分阶段 prompt 文件夹中特定 stage 文件的内容
    
    Args:
        prompt_folder: prompt 文件夹名称
        stage_file: stage 文件名（如 stage1_entity_recognition.txt）
        
    Returns:
        str: 文件内容
    """
    if stage_file not in STAGE_FILES:
        raise ValueError(f"无效的 stage 文件名 '{stage_file}'，必须是以下之一: {STAGE_FILES}")
    
    folder_path = f"{PROJECT_DIR}/prompts/{prompt_folder}"
    file_path = os.path.join(folder_path, stage_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 '{stage_file}' 在文件夹 '{prompt_folder}' 中不存在")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"读取文件失败: {e}")

def update_staged_prompt_content(prompt_folder: str, stage_file: str, content: str):
    """更新分阶段 prompt 文件夹中特定 stage 文件的内容
    
    Args:
        prompt_folder: prompt 文件夹名称
        stage_file: stage 文件名（如 stage1_entity_recognition.txt）
        content: 新的文件内容
        
    Raises:
        ValueError: 如果 prompt_folder 是默认文件夹
        FileNotFoundError: 如果文件不存在
    """
    if prompt_folder == DEFAULT_PROMPT_NAME:
        raise ValueError(f"默认 prompt 文件夹 '{DEFAULT_PROMPT_NAME}' 不能被修改")
    
    if stage_file not in STAGE_FILES:
        raise ValueError(f"无效的 stage 文件名 '{stage_file}'，必须是以下之一: {STAGE_FILES}")
    
    folder_path = f"{PROJECT_DIR}/prompts/{prompt_folder}"
    file_path = os.path.join(folder_path, stage_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 '{stage_file}' 在文件夹 '{prompt_folder}' 中不存在")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise ValueError(f"写入文件失败: {e}")

def create_staged_prompt_file(prompt_folder: str, stage_file: str, content: str = ""):
    """在分阶段 prompt 文件夹中创建新的 stage 文件
    
    Args:
        prompt_folder: prompt 文件夹名称
        stage_file: stage 文件名（如 stage1_entity_recognition.txt）
        content: 文件内容（可选）
        
    Raises:
        ValueError: 如果 stage_file 不是有效的 stage 文件名
    """
    if stage_file not in STAGE_FILES:
        raise ValueError(f"无效的 stage 文件名 '{stage_file}'，必须是以下之一: {STAGE_FILES}")
    
    folder_path = f"{PROJECT_DIR}/prompts/{prompt_folder}"
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, stage_file)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        raise ValueError(f"创建文件失败: {e}")

def delete_staged_prompt_file(prompt_folder: str, stage_file: str):
    """删除分阶段 prompt 文件夹中的 stage 文件
    
    Args:
        prompt_folder: prompt 文件夹名称
        stage_file: stage 文件名
        
    Raises:
        ValueError: 如果 prompt_folder 是默认文件夹
        FileNotFoundError: 如果文件不存在
    """
    if prompt_folder == DEFAULT_PROMPT_NAME:
        raise ValueError(f"默认 prompt 文件夹 '{DEFAULT_PROMPT_NAME}' 不能被删除")
    
    folder_path = f"{PROJECT_DIR}/prompts/{prompt_folder}"
    file_path = os.path.join(folder_path, stage_file)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 '{stage_file}' 在文件夹 '{prompt_folder}' 中不存在")
    
    os.remove(file_path)








# ==================== Schema 文件夹支持 ====================

def list_schema_folders():
    """列出所有schema文件夹"""
    schemas_dir = f"{PROJECT_DIR}/schemas"
    if not os.path.exists(schemas_dir):
        return []
    
    folders = []
    for item in os.listdir(schemas_dir):
        item_path = os.path.join(schemas_dir, item)
        if os.path.isdir(item_path):
            folders.append(item)
    
    return sorted(folders)

def list_schemas_in_folder(folder_name: str):
    """列出指定文件夹中的所有schema文件"""
    folder_path = f"{PROJECT_DIR}/schemas/{folder_name}"
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Schema文件夹 '{folder_name}' 不存在")
    
    schemas = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            schema_name = filename[:-5]  # 移除.json后缀
            schemas.append(schema_name)
    return sorted(schemas)

def get_schema_from_folder(folder_name: str, schema_name: str) -> dict:
    """从指定文件夹获取schema内容"""
    schema_path = f"{PROJECT_DIR}/schemas/{folder_name}/{schema_name}.json"
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema '{schema_name}' 在文件夹 '{folder_name}' 中不存在")
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"读取schema文件失败: {e}")

def create_schema_folder(folder_name: str):
    """创建新的schema文件夹"""
    folder_path = f"{PROJECT_DIR}/schemas/{folder_name}"
    if os.path.exists(folder_path):
        raise ValueError(f"Schema文件夹 '{folder_name}' 已存在")
    
    os.makedirs(folder_path, exist_ok=True)

def create_schema_in_folder(folder_name: str, schema_name: str, schema_content: dict, force_overwrite: bool = False):
    """在指定文件夹中创建schema文件"""
    folder_path = f"{PROJECT_DIR}/schemas/{folder_name}"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Schema文件夹 '{folder_name}' 不存在")
    
    schema_path = os.path.join(folder_path, f"{schema_name}.json")
    
    if os.path.exists(schema_path) and not force_overwrite:
        raise ValueError(f"Schema '{schema_name}' 已存在，如需覆盖请设置 force_overwrite=True")
    
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema_content, f, indent=4, ensure_ascii=False)

def update_schema_in_folder(folder_name: str, schema_name: str, schema_content: dict):
    """更新指定文件夹中的schema文件"""
    schema_path = f"{PROJECT_DIR}/schemas/{folder_name}/{schema_name}.json"
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema '{schema_name}' 在文件夹 '{folder_name}' 中不存在")
    
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema_content, f, indent=4, ensure_ascii=False)

def delete_schema_in_folder(folder_name: str, schema_name: str):
    """删除指定文件夹中的schema文件"""
    schema_path = f"{PROJECT_DIR}/schemas/{folder_name}/{schema_name}.json"
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema '{schema_name}' 在文件夹 '{folder_name}' 中不存在")
    
    os.remove(schema_path)

def delete_schema_folder(folder_name: str):
    """删除schema文件夹及其所有内容"""
    folder_path = f"{PROJECT_DIR}/schemas/{folder_name}"
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Schema文件夹 '{folder_name}' 不存在")
    
    import shutil
    shutil.rmtree(folder_path)





# ==================== 分阶段 Prompt 文件夹 创建/删除 ====================

PROTECTED_STAGED_FOLDERS = {'staged_customized', 'staged'}

def create_staged_prompt_folder(folder_name: str, source_folder: str = None):
    """创建分阶段 prompt 文件夹
    
    Args:
        folder_name: 新文件夹名称
        source_folder: 源文件夹名称（可选），非空则从该文件夹复制所有 stage 文件
    """
    if not folder_name or not folder_name.strip():
        raise ValueError("文件夹名称不能为空")
    
    # 防止路径穿越
    if '..' in folder_name or '/' in folder_name or chr(92) in folder_name:
        raise ValueError("文件夹名称不能包含路径分隔符或 ..")
    
    folder_path = f"{PROJECT_DIR}/prompts/{folder_name}"
    
    if os.path.exists(folder_path):
        raise ValueError(f"Staged prompt 文件夹 '{folder_name}' 已存在")
    
    os.makedirs(folder_path)
    
    if source_folder:
        import shutil
        source_path = f"{PROJECT_DIR}/prompts/{source_folder}"
        if not os.path.exists(source_path):
            # 清理已创建的空文件夹
            os.rmdir(folder_path)
            raise FileNotFoundError(f"源文件夹 '{source_folder}' 不存在")
        
        for stage_file in STAGE_FILES:
            src_file = os.path.join(source_path, stage_file)
            dst_file = os.path.join(folder_path, stage_file)
            if os.path.exists(src_file):
                shutil.copy2(src_file, dst_file)
            else:
                with open(dst_file, 'w', encoding='utf-8') as f:
                    f.write('')
    else:
        for stage_file in STAGE_FILES:
            file_path_full = os.path.join(folder_path, stage_file)
            with open(file_path_full, 'w', encoding='utf-8') as f:
                f.write('')


def delete_staged_prompt_folder(folder_name: str):
    """删除分阶段 prompt 文件夹及其所有内容
    
    Args:
        folder_name: 文件夹名称
    """
    if folder_name in PROTECTED_STAGED_FOLDERS:
        raise ValueError(f"默认 staged prompt 文件夹 '{folder_name}' 不可删除")
    
    folder_path = f"{PROJECT_DIR}/prompts/{folder_name}"
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Staged prompt 文件夹 '{folder_name}' 不存在")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"'{folder_name}' 不是文件夹")
    
    # 安全检查：确认是合法的 staged prompt 文件夹
    has_stage_files = any(
        os.path.exists(os.path.join(folder_path, sf))
        for sf in STAGE_FILES
    )
    if not has_stage_files:
        raise ValueError(f"'{folder_name}' 不是合法的 staged prompt 文件夹")
    
    import shutil
    shutil.rmtree(folder_path)
