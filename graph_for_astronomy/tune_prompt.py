from prompt2graph import prompt2graph
import os
import json
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

PROJECT_DIR = os.getenv("PROJECT_DIR")

def tune_prompt_by_content(dataset_name: str, schema_content: dict, prompt_content: str, time_now: str, session_id: str):
    
    # 判断是否已经保存了chunk
    chunk_path = f"{PROJECT_DIR}/output/{dataset_name}/chunks.txt"
    if not os.path.exists(chunk_path):
        is_chunked = False
    else:
        is_chunked = True
        
    graph_name = f"graph_by_content_{time_now}_session_{session_id}.json"
    
    graph_path = prompt2graph(dataset_name = dataset_name, schema_content = schema_content, prompt_content = prompt_content, is_chunked = is_chunked, output_graph_name = graph_name)
    return graph_path

def tune_prompt_by_file(dataset_name: str, schema_name: str, prompt_name: str):
    schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
    prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_name}!")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_name}!")
    
    
        # 判断是否已经保存了chunk
    chunk_path = f"{PROJECT_DIR}/output/{dataset_name}/chunks.txt"
    if not os.path.exists(chunk_path):
        is_chunked = False
    else:
        is_chunked = True
    
    graph_name = f"graph_by_file_s_{schema_name}_p_{prompt_name}.json"
    
    schema_content = json.load(open(schema_path, 'r'))
    prompt_content = open(prompt_path, 'r').read()
    
    
    graph_path = prompt2graph(dataset_name = dataset_name, schema_content = schema_content, prompt_content = prompt_content, is_chunked = is_chunked, output_graph_name = graph_name)
    return graph_path


def tune_prompt_mixed(dataset_name: str, schema_name: Optional[str], prompt_name: Optional[str], schema_content: Optional[dict], prompt_content: Optional[str], time_now: str, session_id: str):
    """
    混合模式：支持通过文件或内容字符串使用 schema 和 prompt
    
    规则：
    1. 如果只有 schema_name 和 prompt_name（没有 content），则使用 file 模式
    2. 如果既有 content 又有 file，则使用 content 模式（如果有 file，则需要将 file 的内容读取出来）
    3. 如果部分有 content，部分有 name，从文件读取缺失的部分，然后使用 content 模式
    """
    # 情况1: 如果只有 schema_name 和 prompt_name（没有 content），使用 file 模式
    if not schema_content and not prompt_content and schema_name and prompt_name:
        print("使用 file 模式")
        graph_path = tune_prompt_by_file(dataset_name=dataset_name, schema_name=schema_name, prompt_name=prompt_name)
        return graph_path
    
    print("使用 content 模式")
    # 情况2: 如果既有 content 又有 file，使用 content 模式（从文件读取缺失的部分）
    # 情况3: 如果部分有 content，部分有 name，从文件读取缺失的部分，然后使用 content 模式
    final_schema_content = schema_content
    final_prompt_content = prompt_content
    
    # 如果 schema_content 不存在但 schema_name 存在，从文件读取
    if not final_schema_content and schema_name:
        schema_path = f"{PROJECT_DIR}/schemas/{schema_name}.json"
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}!")
        with open(schema_path, 'r', encoding='utf-8') as f:
            final_schema_content = json.load(f)
    
    # 如果 prompt_content 不存在但 prompt_name 存在，从文件读取
    if not final_prompt_content and prompt_name:
        prompt_path = f"{PROJECT_DIR}/prompts/{prompt_name}.txt"
        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}!")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            final_prompt_content = f.read()
    
    # 验证最终是否有足够的参数
    if not final_schema_content or not final_prompt_content:
        raise ValueError("必须提供 schema_name/content 和 prompt_name/content 的组合，且至少能获取到完整的 schema 和 prompt 内容")
    
    # 使用 content 模式
    graph_path = tune_prompt_by_content(
        dataset_name=dataset_name, 
        schema_content=final_schema_content, 
        prompt_content=final_prompt_content, 
        time_now=time_now, 
        session_id=session_id
    )
    return graph_path
        



if __name__ == "__main__":
    def test_tune_prompt_by_content():
        from datetime import datetime
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = "1234567890"
        prompt_content = """
        You are a research scientist specializing in battery materials and electrochemistry.
    Your task is to meticulously analyze the provided scientific text about battery electrolytes and extract key chemical compounds, materials, properties, and their relationships.
    The goal is to build a structured knowledge graph for materials discovery.
    **CRITICAL FORMATTING RULES (Must be followed):**
    1.  **JSON Only:** Return ONLY a raw JSON object containing "attributes", "triples", and "entity_types".
    2.  **Triple Format:** All triples *must* be a 3-element list `[subject, relation, object]`. All 3 elements *must* be strings.
        -   Correct: `["Li-ion", "exhibits", "high energy density"]`
        -   WRONG: `["Li-ion", "exhibits"]` (Missing object)
        -   WRONG: `[["Li-ion"], "exhibits", "high energy density"]` (Subject is a list, not a string)
    3.  **Attribute Format:** Attributes *must* be strings in `"key: value"` format (e.g., `"value: 8.5"`, `"units: mS/cm"`).
    4.  **Entity Consistency:** Entity names must be identical across `attributes` keys, all `triples`, and the `entity_types` keys.

    **Guidelines for Extraction:**
    1. Strictly adhere to the provided schema for extraction.
    ```{schema}```
    2. Extract entities such as specific chemical compounds, solvents, salts, additives, and the properties they exhibit (e.g., ionic conductivity, electrochemical window).
    3. Extract relationships like a compound improving a performance metric, an electrolyte being composed of certain chemicals, or a property being measured by a specific test method.
    4. Pay close attention to numerical data. Extract property values and their units as attributes.
    5. Avoid redundant attributes: do NOT repeat the entity name/schema type as an attribute value (e.g., "description: salt" for entity "Salt"); descriptions must add new information.

    **Schema Evolution (Optional):**
    If you identify new, critical entity or relation types that are fundamental to electrolyte research (e.g., "InterfaceLayer", "DegradationMechanism"), include them in a "new_schema_types" field.
    **Clarification:** This field should only be used for truly new and important concepts. If the existing schema already covers the concepts, or if no new concepts are found, **return an empty object `{}` or omit this "new_schema_types" field entirely.**

    Text Chunk:
        ```{chunk}```

    Example Output:
        {{
            "attributes": {{
            "Ethylene Carbonate": ["abbreviation: EC", "formula: C3H4O3"],
            "Ionic Conductivity": ["value: 8.5", "units: mS/cm"]
            }},
            "triples": [
            ["LiPF6-EC/DMC", "is_composed_of", "Ethylene Carbonate"],
            ["Fluoroethylene 
    Carbonate", "improves", "Coulombic Efficiency"],
            ["LiPF6-EC/DMC", "exhibits_property", "Ionic Conductivity"]
            ],
            "entity_types": {{
            "Ethylene Carbonate": "Solvent",
            "LiPF6-EC/DMC": "Electrolyte",
            "Fluoroethylene Carbonate": "Additive",
            "Ionic Conductivity": "Property",
            "Coulombic Efficiency": "PerformanceMetric"
            
    }},
            "new_schema_types": {{
            "nodes": ["SEI_Component"],
            "relations": ["forms_on"],
            "attributes": ["stability_potential"]
            }}
        }}
        """
        
        schema_content = {
        "Nodes": [
            "ChemicalCompound",
            "Electrolyte",
            "Solvent",
            "Salt",
            "Additive",
            "Property",
            "PerformanceMetric",
            "TestMethod",
            "Anode",
            "Cathode",
            "TestCondition",
            "CharacterizationMethod"
        ],
        "Relations": [
            "is_composed_of",
            "is_solvent_for",
            "is_salt_in",
            "is_additive_in",
            "is_compatible_with",
            "exhibits_property",
            "is_measured_by",
            "improves",
            "degrades"
        ],
        "Attributes": [
            "formula",
            "concentration",
            "value",
            "units",
            "abbreviation",
            "description",
            "loading",
            "source"
        ]
    }
        graph_path = tune_prompt_by_content(dataset_name="paper_mini", schema_content=schema_content, prompt_content=prompt_content, time_now=time_now, session_id=session_id)
        print(graph_path)
    

    def test_tune_prompt_by_file():
        
        graph_path = tune_prompt_by_file(dataset_name="paper_mini", schema_name="electrolytes", prompt_name="electrolytes")
        print(graph_path)

    
    def test_tune_prompt_mixed():
        from datetime import datetime
        time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = "1234567890"
        
        test_prompt_content = """
        You are a research scientist specializing in battery materials and electrochemistry.
    Your task is to meticulously analyze the provided scientific text about battery electrolytes and extract key chemical compounds, materials, properties, and their relationships.
    The goal is to build a structured knowledge graph for materials discovery.
    **CRITICAL FORMATTING RULES (Must be followed):**
    1.  **JSON Only:** Return ONLY a raw JSON object containing "attributes", "triples", and "entity_types".
    2.  **Triple Format:** All triples *must* be a 3-element list `[subject, relation, object]`. All 3 elements *must* be strings.
        -   Correct: `["Li-ion", "exhibits", "high energy density"]`
        -   WRONG: `["Li-ion", "exhibits"]` (Missing object)
        -   WRONG: `[["Li-ion"], "exhibits", "high energy density"]` (Subject is a list, not a string)
    3.  **Attribute Format:** Attributes *must* be strings in `"key: value"` format (e.g., `"value: 8.5"`, `"units: mS/cm"`).
    4.  **Entity Consistency:** Entity names must be identical across `attributes` keys, all `triples`, and the `entity_types` keys.

    **Guidelines for Extraction:**
    1. Strictly adhere to the provided schema for extraction.
    ```{schema}```
    2. Extract entities such as specific chemical compounds, solvents, salts, additives, and the properties they exhibit (e.g., ionic conductivity, electrochemical window).
    3. Extract relationships like a compound improving a performance metric, an electrolyte being composed of certain chemicals, or a property being measured by a specific test method.
    4. Pay close attention to numerical data. Extract property values and their units as attributes.
    5. Avoid redundant attributes: do NOT repeat the entity name/schema type as an attribute value (e.g., "description: salt" for entity "Salt"); descriptions must add new information.

    **Schema Evolution (Optional):**
    If you identify new, critical entity or relation types that are fundamental to electrolyte research (e.g., "InterfaceLayer", "DegradationMechanism"), include them in a "new_schema_types" field.
    **Clarification:** This field should only be used for truly new and important concepts. If the existing schema already covers the concepts, or if no new concepts are found, **return an empty object `{}` or omit this "new_schema_types" field entirely.**

    Text Chunk:
        ```{chunk}```

    Example Output:
        {{
            "attributes": {{
            "Ethylene Carbonate": ["abbreviation: EC", "formula: C3H4O3"],
            "Ionic Conductivity": ["value: 8.5", "units: mS/cm"]
            }},
            "triples": [
            ["LiPF6-EC/DMC", "is_composed_of", "Ethylene Carbonate"],
            ["Fluoroethylene 
    Carbonate", "improves", "Coulombic Efficiency"],
            ["LiPF6-EC/DMC", "exhibits_property", "Ionic Conductivity"]
            ],
            "entity_types": {{
            "Ethylene Carbonate": "Solvent",
            "LiPF6-EC/DMC": "Electrolyte",
            "Fluoroethylene Carbonate": "Additive",
            "Ionic Conductivity": "Property",
            "Coulombic Efficiency": "PerformanceMetric"
            
    }},
            "new_schema_types": {{
            "nodes": ["SEI_Component"],
            "relations": ["forms_on"],
            "attributes": ["stability_potential"]
            }}
        }}
        """
        
        test_schema_content = {
        "Nodes": [
            "ChemicalCompound",
            "Electrolyte",
            "Solvent",
            "Salt",
            "Additive",
            "Property",
            "PerformanceMetric",
            "TestMethod",
            "Anode",
            "Cathode",
            "TestCondition",
            "CharacterizationMethod"
        ],
        "Relations": [
            "is_composed_of",
            "is_solvent_for",
            "is_salt_in",
            "is_additive_in",
            "is_compatible_with",
            "exhibits_property",
            "is_measured_by",
            "improves",
            "degrades"
        ],
        "Attributes": [
            "formula",
            "concentration",
            "value",
            "units",
            "abbreviation",
            "description",
            "loading",
            "source"
        ]
    }
        
        print("=" * 60)
        print("测试所有 schema_name/prompt_name 和 schema_content/prompt_content 的组合情况")
        print("=" * 60)
        
        # 测试情况1: 只有 schema_name 和 prompt_name（没有 content）- 使用 file 模式
        print("\n[测试1] 只有 schema_name 和 prompt_name（没有 content）- 使用 file 模式")
        try:
            test1_path = tune_prompt_mixed(
                dataset_name="paper_mini", 
                schema_name="electrolytes", 
                prompt_name="electrolytes", 
                schema_content=None, 
                prompt_content=None, 
                time_now=time_now, 
                session_id=session_id
            )
            print(f"✓ 测试1成功: {test1_path}")
        except Exception as e:
            print(f"✗ 测试1失败: {e}")
        
        # 测试情况2: 只有 schema_content 和 prompt_content（没有 name）- 使用 content 模式
        print("\n[测试2] 只有 schema_content 和 prompt_content（没有 name）- 使用 content 模式")
        try:
            test2_path = tune_prompt_mixed(
                dataset_name="paper_mini", 
                schema_name=None, 
                prompt_name=None, 
                schema_content=test_schema_content, 
                prompt_content=test_prompt_content, 
                time_now=time_now, 
                session_id=session_id
            )
            print(f"✓ 测试2成功: {test2_path}")
        except Exception as e:
            print(f"✗ 测试2失败: {e}")
        
        # 测试情况3: schema_content + prompt_name - 从文件读取 prompt，使用 content 模式
        print("\n[测试3] schema_content + prompt_name - 从文件读取 prompt，使用 content 模式")
        try:
            test3_path = tune_prompt_mixed(
                dataset_name="paper_mini", 
                schema_name=None, 
                prompt_name="electrolytes", 
                schema_content=test_schema_content, 
                prompt_content=None, 
                time_now=time_now, 
                session_id=session_id
            )
            print(f"✓ 测试3成功: {test3_path}")
        except Exception as e:
            print(f"✗ 测试3失败: {e}")
        
        # 测试情况4: schema_name + prompt_content - 从文件读取 schema，使用 content 模式
        print("\n[测试4] schema_name + prompt_content - 从文件读取 schema，使用 content 模式")
        try:
            test4_path = tune_prompt_mixed(
                dataset_name="paper_mini", 
                schema_name="electrolytes", 
                prompt_name=None, 
                schema_content=None, 
                prompt_content=test_prompt_content, 
                time_now=time_now, 
                session_id=session_id
            )
            print(f"✓ 测试4成功: {test4_path}")
        except Exception as e:
            print(f"✗ 测试4失败: {e}")
        
        # 测试情况5: schema_content + prompt_name + schema_name + prompt_content - 既有 content 又有 file，使用 content 模式（忽略 file）
        print("\n[测试5] schema_content + prompt_content + schema_name + prompt_name - 既有 content 又有 file，使用 content 模式（忽略 file）")
        try:
            test5_path = tune_prompt_mixed(
                dataset_name="paper_mini", 
                schema_name="electrolytes", 
                prompt_name="electrolytes", 
                schema_content=test_schema_content, 
                prompt_content=test_prompt_content, 
                time_now=time_now, 
                session_id=session_id
            )
            print(f"✓ 测试5成功: {test5_path}")
        except Exception as e:
            print(f"✗ 测试5失败: {e}")
        
        # 测试情况6: schema_content + prompt_name + schema_name - 部分 content，部分 name，使用 content 模式
        print("\n[测试6] schema_content + prompt_name + schema_name - 部分 content，部分 name，使用 content 模式")
        try:
            test6_path = tune_prompt_mixed(
                dataset_name="paper_mini", 
                schema_name="electrolytes", 
                prompt_name="electrolytes", 
                schema_content=test_schema_content, 
                prompt_content=None, 
                time_now=time_now, 
                session_id=session_id
            )
            print(f"✓ 测试6成功: {test6_path}")
        except Exception as e:
            print(f"✗ 测试6失败: {e}")
        
        # 测试情况7: schema_name + prompt_content + prompt_name - 部分 content，部分 name，使用 content 模式
        print("\n[测试7] schema_name + prompt_content + prompt_name - 部分 content，部分 name，使用 content 模式")
        try:
            test7_path = tune_prompt_mixed(
                dataset_name="paper_mini", 
                schema_name="electrolytes", 
                prompt_name="electrolytes", 
                schema_content=None, 
                prompt_content=test_prompt_content, 
                time_now=time_now, 
                session_id=session_id
            )
            print(f"✓ 测试7成功: {test7_path}")
        except Exception as e:
            print(f"✗ 测试7失败: {e}")
        
        print("\n" + "=" * 60)
        print("所有测试完成")
        print("=" * 60)
    
    
    # test_tune_prompt_by_content()
    # test_tune_prompt_by_file()
    test_tune_prompt_mixed()




# output_graph_path = f"{output_dir}/graph_s_{schema_name}_p_{prompt_name}.json"

# TODO: 前端页面有shema编辑和prompt编辑，编辑后点击 “生成graph” 按钮时,将

# 对应的前端代码

