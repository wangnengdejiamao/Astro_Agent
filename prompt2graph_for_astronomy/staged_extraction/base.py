"""
分阶段提取器基类
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import json_repair

from utils import call_llm_api
from utils.logger import logger


class BaseStageExtractor(ABC):
    """分阶段提取器基类，封装 LLM 调用、重试与验证逻辑"""

    def __init__(
        self,
        schema: Dict[str, Any],
        prompt_path: Optional[str] = None,
        default_prompt_filename: str = "",
    ):
        """
        Args:
            schema: Schema 定义
            prompt_path: 自定义 prompt 文件路径（可选）
            default_prompt_filename: 默认 prompt 文件名（如 stage1_entity_recognition.txt）
        """
        self.schema = schema
        self.llm_client = call_llm_api.LLMCompletionCall()
        self.prompt_template = self._load_prompt_template(prompt_path, default_prompt_filename)

    def _load_prompt_template(
        self, prompt_path: Optional[str], default_prompt_filename: str
    ) -> str:
        """加载 prompt 模板：优先使用 prompt_path，否则使用默认路径"""
        if prompt_path and os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prompts",
            "staged",
            default_prompt_filename,
        )
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Prompt file not found: {default_path}")
        with open(default_path, "r", encoding="utf-8") as f:
            return f.read()

    def apply_variables(self, prompt: str, variables: Optional[Dict[str, str]] = None) -> str:
        """
        将变量注入到 prompt 模板中。
        支持的变量格式为 ```{var_name}```。

        Args:
            prompt: 原始 prompt 模板
            variables: 变量字典，key 为变量名（如 "examples"），value 为要注入的内容

        Returns:
            替换后的 prompt
        """
        if not variables:
            return prompt
        result = prompt
        for key, value in variables.items():
            placeholder = f"```{key}```"
            result = result.replace(placeholder, value)
        return result

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """调用 LLM 并解析 JSON"""
        response = self.llm_client.call_api(prompt)
        return json_repair.loads(response)

    def _run_extraction(
        self,
        prompt: str,
        validate_fn: Callable[[Dict[str, Any]], bool],
        fallback: Dict[str, Any],
        stage_name: str,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        执行提取：调用 LLM、验证、重试
        
        Args:
            prompt: 构建好的 prompt
            validate_fn: 验证函数，接受解析结果，返回是否合法
            fallback: 验证失败或异常时的默认返回值
            stage_name: 阶段名称（用于日志）
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                parsed = self._call_llm(prompt)
                if validate_fn(parsed):
                    return parsed
                logger.warning(
                    f"{stage_name} output validation failed "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            except Exception as e:
                logger.error(
                    f"{stage_name} extraction failed "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
            if attempt >= max_retries - 1:
                logger.error(f"{stage_name} failed after max retries")
                return fallback
        return fallback

    @abstractmethod
    def _build_prompt(self, chunk: str, **kwargs: Any) -> str:
        """构建 prompt，子类实现"""
        pass

    @abstractmethod
    def _validate_output(self, output: Dict[str, Any]) -> bool:
        """验证输出格式，子类实现"""
        pass

    @abstractmethod
    def extract(self, chunk: str, **kwargs: Any) -> Dict[str, Any]:
        """执行提取，子类实现并调用 _run_extraction"""
        pass
