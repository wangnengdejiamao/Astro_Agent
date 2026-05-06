"""
阶段3：属性提取
"""

import json
from typing import Any, Dict, Optional

from .base import BaseStageExtractor


class Stage3AttributeExtraction(BaseStageExtractor):
    """阶段3：属性提取"""

    _DEFAULT_PROMPT = "stage3_attribute_extraction.txt"

    def __init__(self, schema: Dict[str, Any], prompt_path: Optional[str] = None):
        super().__init__(schema, prompt_path, self._DEFAULT_PROMPT)

    def _build_prompt(self, chunk: str, **kwargs: Any) -> str:
        stage1_output = kwargs.get("stage1_output") or {}
        schema_str = json.dumps(self.schema, ensure_ascii=False, indent=2)
        entities_str = json.dumps(
            stage1_output.get("entities", []), ensure_ascii=False, indent=2
        )
        abbrev_str = json.dumps(
            stage1_output.get("abbreviation_mappings", {}),
            ensure_ascii=False,
            indent=2,
        )
        return (
            self.prompt_template.replace("{schema}", schema_str)
            .replace("{chunk}", chunk)
            .replace("{entities}", entities_str)
            .replace("{abbreviation_mappings}", abbrev_str)
        )

    def _validate_output(self, output: Dict[str, Any]) -> bool:
        if not isinstance(output, dict) or "attributes" not in output:
            return False
        if not isinstance(output["attributes"], dict):
            return False
        required = ["key", "value", "source", "evidence"]
        for attr_list in output["attributes"].values():
            if not isinstance(attr_list, list):
                return False
            for attr in attr_list:
                if not isinstance(attr, dict) or any(k not in attr for k in required):
                    return False
        return True

    def extract(
        self,
        chunk: str,
        stage1_output: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(chunk, stage1_output=stage1_output)
        return self._run_extraction(
            prompt,
            self._validate_output,
            {"attributes": {}},
            "Stage 3",
            max_retries,
        )
