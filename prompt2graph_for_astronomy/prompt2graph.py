import os
from datetime import datetime
from get_chunks import get_chunks
from get_lowlevel_graph import build_lowlevel_graph
from dotenv import load_dotenv
load_dotenv()


PROJECT_DIR = os.getenv("PROJECT_DIR")

def prompt2graph(
    dataset_name: str,
    schema_name: str = None,
    prompt_name: str = None,
    is_chunked: bool = False,
    schema_content: dict = None,
    prompt_content: str = None,
    output_graph_name: str = "graph.json",
    use_staged_extraction: bool = False,
    enable_stage4_validation: bool = False,
    pubchem_db_path: str = "pubchem_names.db",
    save_stage_outputs: bool = False,
):
    """
    构建知识图谱，支持通过路径或内容字符串使用
    
    支持两种使用方式：
    1. 传统方式（通过名称/路径）: prompt2graph(dataset_name, schema_name, prompt_name)
    2. 内容字符串方式: prompt2graph(dataset_name, schema_content=..., prompt_content=...)
    
    Args:
        dataset_name: 数据集名称（必需）
        schema_name: schema 名称（如果提供 schema_content 则忽略）
        prompt_name: prompt 名称（如果提供 prompt_content 则忽略，多阶段提取时忽略此参数）
        is_chunked: 是否已经分块
        schema_content: schema 内容字符串（优先使用，如果提供则忽略 schema_name）
        prompt_content: prompt 内容字符串（单阶段提取时使用，多阶段提取时忽略此参数）
        output_graph_name: 输出图谱文件名
        use_staged_extraction: 是否使用多阶段提取（默认False，使用单阶段提取）
        enable_stage4_validation: 是否启用阶段4验证（仅多阶段提取时有效，默认False，会消耗大量token）
        pubchem_db_path: PubChem本地数据库路径（可选，如果提供则在构建图谱时查询CID并添加到实体属性中）
    """
    # dataset_path = f"{PROJECT_DIR}/input/{dataset_name}/corpus.json"
    dataset_path = f"{PROJECT_DIR}/input/{dataset_name}/corpus_cleaned.json"
    dataset_output_root = f"{PROJECT_DIR}/output/{dataset_name}"
    os.makedirs(dataset_output_root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(dataset_output_root, run_id)
    os.makedirs(output_dir, exist_ok=True)
    chunk_path = f"{dataset_output_root}/chunks.txt"
    output_graph_path = f"{output_dir}/{output_graph_name}"
    
    # 1. 构建chunks
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_name}!")
    
    if is_chunked:
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk file not found: {dataset_name}!")
    else:
        get_chunks(dataset_path, dataset_name, chunk_path)
        
    print("Chunked dataset successfully!")
    
    # 2. 构建 low-level graph
    if use_staged_extraction:
        # 使用多阶段提取
        print("Using staged extraction (multi-stage)...")
        if enable_stage4_validation:
            print("Stage 4 validation: enabled (will consume significant tokens)")
        
        # 确定schema路径
        schema_path = None
        if not schema_content and schema_name:
            schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
        graph_config = {
            "schema_path": schema_path,
            "schema_content": schema_content,
            "prompt_path": None,
            "prompt_content": None,
            "use_staged_extraction": True,
            "enable_stage4_validation": enable_stage4_validation,
            "prompt_paths": None,
            "pubchem_db_path": pubchem_db_path,
            "save_stage_outputs": save_stage_outputs,
        }
        build_lowlevel_graph(
            chunk_path=chunk_path,
            output_graph_path=output_graph_path,
            config=graph_config,
        )
    else:
        # 使用单阶段提取（原有方式）
        print("Using single-stage extraction...")
        schema_path = None
        prompt_path = None
        if not schema_content and schema_name:
            schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
        if not prompt_content and prompt_name:
            prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
        if not schema_content and not schema_path:
            raise ValueError("必须提供 schema_name 或 schema_content")
        if not prompt_content and not prompt_path:
            raise ValueError("单阶段提取必须提供 prompt_name 或 prompt_content")
        graph_config = {
            "schema_path": schema_path,
            "schema_content": schema_content,
            "prompt_path": prompt_path,
            "prompt_content": prompt_content,
            "use_staged_extraction": False,
            "pubchem_db_path": pubchem_db_path,
        }
        build_lowlevel_graph(
            chunk_path=chunk_path,
            output_graph_path=output_graph_path,
            config=graph_config,
        )
    
    return output_graph_path


def save_schema_and_prompt(schema_name: str, schema_content: str, prompt_name: str, prompt_content: str):
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    with open(schema_path, 'w') as f:
        f.write(schema_content)
    with open(prompt_path, 'w') as f:
        f.write(prompt_content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="底层图谱构建：从语料/分块文本抽取实体-关系三元组（支持单阶段与多阶段提取）"
    )
    parser.add_argument("dataset", help="数据集名称，对应 input/{dataset}/ 与 output/{dataset}/")
    parser.add_argument("--schema", "-s", default="260114", help="Schema 名称（对应 schemas/{name}.json），默认 260114")
    parser.add_argument("--prompt", "-p", default=None, help="单阶段提取时的 prompt 名称（对应 prompts/{name}.txt）；多阶段时可不填")
    parser.add_argument("--output", "-o", default="graph.json", help="输出图谱文件名，默认 graph.json")
    parser.add_argument("--chunked", action="store_true", help="是否已分块（True 则跳过 get_chunks，使用已有 chunks）")
    parser.add_argument("--staged", action="store_true", help="使用多阶段提取（默认单阶段）")
    parser.add_argument("--stage4", action="store_true", help="多阶段提取时启用阶段 4 验证（耗 token）")
    parser.add_argument("--save-stage-outputs", action="store_true", help="多阶段提取时保存各阶段中间结果到 staged/")
    parser.add_argument("--pubchem-db", default="pubchem_names_full.db", help="PubChem 本地数据库路径")
    args = parser.parse_args()

# prompt2graph(dataset_name="paper_mini", schema_name="260114", 
# prompt_name="260120", is_chunked=True, output_graph_name=output_graph_name, 
# use_staged_extraction=True, save_stage_outputs=True)

    prompt2graph(
        dataset_name=args.dataset,
        schema_name=args.schema,
        prompt_name=args.prompt,
        is_chunked=args.chunked,
        output_graph_name=args.output,
        use_staged_extraction=args.staged,
        enable_stage4_validation=args.stage4,
        save_stage_outputs=args.save_stage_outputs,
        pubchem_db_path=args.pubchem_db,
    )




