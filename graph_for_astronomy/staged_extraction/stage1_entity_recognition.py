"""
阶段1：实体识别和规范化
"""

import json
from typing import Any, Dict, Optional

from .base import BaseStageExtractor


class Stage1EntityRecognition(BaseStageExtractor):
    """阶段1：实体识别和规范化"""

    _DEFAULT_PROMPT = "stage1_entity_recognition.txt"

    def __init__(self, schema: Dict[str, Any], prompt_path: Optional[str] = None):
        super().__init__(schema, prompt_path, self._DEFAULT_PROMPT)

    def _build_prompt(self, chunk: str, **kwargs: Any) -> str:
        schema_str = json.dumps(self.schema, ensure_ascii=False, indent=2)
        prompt = (
            self.prompt_template.replace("{schema}", schema_str).replace("{chunk}", chunk)
        )
        variables = kwargs.get("variables")
        if variables:
            prompt = self.apply_variables(prompt, variables)
        return prompt

    def _validate_output(self, output: Dict[str, Any]) -> bool:
        if not isinstance(output, dict):
            return False
        if "entities" not in output or "abbreviation_mappings" not in output:
            return False
        if not isinstance(output["entities"], list):
            return False
        if not isinstance(output["abbreviation_mappings"], dict):
            return False
        required = ["canonical_name", "variants", "schema_type", "source", "evidence"]
        for entity in output["entities"]:
            if not isinstance(entity, dict) or any(k not in entity for k in required):
                return False
        return True

    def extract(self, chunk: str, max_retries: int = 3, **kwargs: Any) -> Dict[str, Any]:
        prompt = self._build_prompt(chunk, **kwargs)
        return self._run_extraction(
            prompt,
            self._validate_output,
            {"entities": [], "abbreviation_mappings": {}},
            "Stage 1",
            max_retries,
        )
