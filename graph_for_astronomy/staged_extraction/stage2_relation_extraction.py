"""
阶段2：关系提取
"""

import json
from typing import Any, Dict, Optional

from .base import BaseStageExtractor


class Stage2RelationExtraction(BaseStageExtractor):
    """阶段2：关系提取"""

    _DEFAULT_PROMPT = "stage2_relation_extraction.txt"

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
        prompt = (
            self.prompt_template.replace("{schema}", schema_str)
            .replace("{chunk}", chunk)
            .replace("{entities}", entities_str)
            .replace("{abbreviation_mappings}", abbrev_str)
        )
        variables = kwargs.get("variables")
        if variables:
            prompt = self.apply_variables(prompt, variables)
        return prompt

    def _validate_output(self, output: Dict[str, Any]) -> bool:
        if not isinstance(output, dict) or "triples" not in output:
            return False
        if not isinstance(output["triples"], list):
            return False
        required = ["subject", "relation", "object", "source", "evidence"]
        for triple in output["triples"]:
            if not isinstance(triple, dict) or any(k not in triple for k in required):
                return False
        return True

    def extract(
        self,
        chunk: str,
        stage1_output: Dict[str, Any],
        max_retries: int = 3,
        **kwargs: Any
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(chunk, stage1_output=stage1_output, **kwargs)
        return self._run_extraction(
            prompt,
            self._validate_output,
            {"triples": []},
            "Stage 2",
            max_retries,
        )
