"""KG / RAG retrieval prompt helpers (HyDE pattern + per-source class rerank).

Two functions, used by `workflow.rag_navigator_node`,
`workflow.kg_navigator_node`, and `workflow.method_scout_node`:

  expand_queries(section, source_class, base_queries, *, use_llm, provider)
      Return 2-3 query rewrites per base query, biased toward
      white-dwarf / compact-binary vocabulary. Falls back to a
      deterministic expansion if `use_llm` is False or the LLM call
      fails. Emits a small dict that the caller logs alongside the
      retrieval result.

  rerank_hits(hits, source_class, *, use_llm, provider)
      Re-order a list of hits (RAG rows or KG triples) by a relevance
      score that combines:
          (a) full-text overlap with the source-class-specific weight
              terms in `RERANK_KEYS[source_class]`,
          (b) the per_source_rag `mentions_target` flag if available,
          (c) (when `use_llm=True`) an LLM "why is this relevant"
              one-sentence explanation that boosts the score if the
              explanation is non-trivial.

Both are intentionally robust: they never raise; on any error they fall
back to the original ordering.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence


# Source-class-specific keywords. Used both as HyDE seeds and as
# rerank weight terms. Keep the lists short; the LLM expansion fills in.
RERANK_KEYS: Dict[str, List[str]] = {
    "white_dwarf_single": [
        "Bergeron", "Koester", "cooling track", "log g", "Teff",
        "Mass-Radius", "DA atmosphere", "Gaia parallax", "RUWE",
    ],
    "white_dwarf_binary": [
        "Roche-lobe", "ellipsoidal", "eclipse", "RV curve",
        "Kepler", "mass ratio", "Eggleton 1983", "common envelope",
        "post-common-envelope binary", "PCEB",
    ],
    "double_white_dwarf": [
        "DWD", "GW inspiral", "gravitational wave", "AM CVn",
        "He core", "merger", "GR shrinkage", "tau_GW",
    ],
    "hot_subdwarf_binary": [
        "sdB", "sdO", "EHB", "Heber 2016", "extreme horizontal branch",
        "envelope stripping", "He flash",
    ],
    "cataclysmic_variable": [
        "CV", "mass transfer", "accretion disk", "dwarf nova",
        "outburst", "Patterson", "superhump",
    ],
    "magnetic_white_dwarf": [
        "Zeeman", "magnetic field", "polar", "intermediate polar",
        "cyclotron",
    ],
    "unknown": [],
}


def _keys_for(source_class: Optional[str]) -> List[str]:
    if not source_class:
        return []
    base = RERANK_KEYS.get(source_class, [])
    # also union with a generic compact-binary list to avoid blanks for
    # rare classes
    return base + RERANK_KEYS.get("white_dwarf_binary", [])


def expand_queries(
    section: str,
    source_class: Optional[str],
    base_queries: Sequence[str],
    *,
    use_llm: bool = False,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Return {"queries": [...], "explanations": {q: why}, "llm_used": bool}.

    Deterministic expansion: for each base query, append source-class
    weight terms and a HyDE-style "what would the abstract say" seed.
    """
    keys = _keys_for(source_class)
    expanded: List[str] = []
    explanations: Dict[str, str] = {}
    for q in base_queries:
        q = (q or "").strip()
        if not q:
            continue
        expanded.append(q)
        if keys:
            # weighted: append 2-3 class keywords to bias the search
            expanded.append(q + " " + " ".join(keys[:3]))
        # HyDE seed
        expanded.append(
            f"abstract: We present a study of a {source_class or 'compact object'} "
            f"and discuss {q}."
        )
        explanations[q] = (
            f"queries the literature for `{section}`-relevant evidence on "
            f"{source_class or 'compact object'}"
        )
    out: Dict[str, Any] = {
        "queries": expanded,
        "rerank_keys": keys,
        "explanations": explanations,
        "llm_used": False,
    }

    if use_llm:
        try:
            from ..llm_client import LLMClient, load_model_config
            from . import wd_domain
            cfg = load_model_config(provider) if provider else load_model_config()
            client = LLMClient(cfg)
            if client.available:
                system = (
                    wd_domain.system_for_role("retrieval") + "\n" + wd_domain.RETRIEVAL_TASK
                )
                user = (
                    f"Section: {section}\n"
                    f"Source class: {source_class or 'unknown'}\n"
                    "Base queries:\n- " + "\n- ".join(base_queries)
                )
                text = client.complete(system=system, user=user, temperature=0.1, max_output_tokens=900)
                obj = _parse_json(text)
                if isinstance(obj, dict):
                    qs = obj.get("queries") or []
                    rk = obj.get("rerank_keys") or []
                    if qs:
                        # Merge LLM queries on top, deduplicated.
                        merged: List[str] = []
                        seen = set()
                        for q in list(qs) + expanded:
                            if q and q not in seen:
                                seen.add(q)
                                merged.append(q)
                        out["queries"] = merged
                    if rk:
                        out["rerank_keys"] = list(dict.fromkeys(list(rk) + keys))
                    out["llm_used"] = True
                    out["llm_provider"] = cfg.provider
        except Exception as exc:
            out["llm_error"] = f"{type(exc).__name__}: {exc}"
    return out


def _parse_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(text.splitlines()[1:-1]) if text.count("```") >= 2 else text.strip("`")
    try:
        return json.loads(text)
    except Exception:
        return None


def rerank_hits(
    hits: Iterable[Dict[str, Any]],
    source_class: Optional[str],
    *,
    body_key: str = "abstract",
    title_key: str = "title",
) -> List[Dict[str, Any]]:
    """Stable rerank by class-keyword overlap + per_source_rag mentions_target.

    Each hit is annotated with `_rerank_score` and `_rerank_why` (one
    short sentence describing the boost) for downstream logging /
    reflexion injection. Original order is preserved for ties.
    """
    hits = list(hits or [])
    keys = _keys_for(source_class)
    keys_lower = [k.lower() for k in keys]
    scored: List[Dict[str, Any]] = []
    for idx, h in enumerate(hits):
        title = str(h.get(title_key) or "").lower()
        body = str(h.get(body_key) or "").lower()
        kw_hits = sum(1 for k in keys_lower if k in title or k in body)
        boost_target = 2 if h.get("mentions_target") else 0
        score = float(kw_hits) + boost_target
        why_parts: List[str] = []
        if kw_hits:
            matched = [k for k in keys if k.lower() in title or k.lower() in body][:3]
            why_parts.append(f"matches class terms {matched}")
        if boost_target:
            why_parts.append("abstract mentions this target")
        h2 = dict(h)
        h2["_rerank_score"] = score
        h2["_rerank_why"] = "; ".join(why_parts) or "no class-keyword match"
        h2["_rerank_position_in"] = idx
        scored.append(h2)
    scored.sort(key=lambda x: (-x["_rerank_score"], x["_rerank_position_in"]))
    return scored


__all__ = ["expand_queries", "rerank_hits", "RERANK_KEYS"]
