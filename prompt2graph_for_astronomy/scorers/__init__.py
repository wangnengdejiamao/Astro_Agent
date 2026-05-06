"""
打分相关模块包：
- LLMScorer：LLM 打分
- PubChemScorer：基于 PubChem 本地库的打分
- ChunkScorer / score_chunks：综合打分入口
"""

from .llm_scorer import LLMScorer  # noqa: F401
try:
    from . import pubchem_scorer  # noqa: F401
    from .pubchem_scorer import PubChemScorer  # noqa: F401
except Exception:
    PubChemScorer = None  # type: ignore
    class _FallbackPubChemScorerModule:
        @staticmethod
        def normalize_name(name):
            import re
            return re.sub(r"\s+", " ", str(name or "").strip().lower())

    pubchem_scorer = _FallbackPubChemScorerModule()  # type: ignore
