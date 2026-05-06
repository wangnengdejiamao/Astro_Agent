from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAG_DB = PROJECT_ROOT.parent / "rag_pipeline" / "index" / "white_dwarf_rag.sqlite"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "white_dwarf_kg"


SURVEY_NAMES = {
    "2MASS",
    "APOGEE",
    "ASAS-SN",
    "CRTS",
    "DES",
    "DESI",
    "GALEX",
    "Gaia",
    "Kepler",
    "K2",
    "LAMOST",
    "LSST",
    "Pan-STARRS",
    "RAVE",
    "SDSS",
    "TESS",
    "UKIDSS",
    "VISTA",
    "WISE",
    "ZTF",
}


GENERIC_SOURCE_TOKENS = {
    "WD",
    "SDSS",
    "ZTF",
    "Gaia",
    "Gaia DR1",
    "Gaia DR2",
    "Gaia DR3",
    "EDR3",
    "DR3",
    "J2000",
}


OBJECT_PATTERNS = [
    re.compile(r"\bSDSS\s?J\d{4,6}(?:\.\d+)?[+-]\d{4,6}(?:\.\d+)?\b", re.I),
    re.compile(r"\bZTF\s?J\d{6}(?:\.\d+)?[+-]\d{6}(?:\.\d+)?\b", re.I),
    re.compile(r"\bGaia\s+(?:DR1|DR2|DR3|EDR3)\s+\d{8,}\b", re.I),
    re.compile(r"\bWD\s?J?\d{4}[+-]\d{3,4}\b", re.I),
    re.compile(r"\bPG\s?\d{4}[+-]\d{3,4}\b", re.I),
    re.compile(r"\bGD\s?\d{1,4}\b", re.I),
    re.compile(r"\bG\s?\d{1,3}-\d{1,3}\b", re.I),
    re.compile(r"\bKIC\s?\d{5,}\b", re.I),
    re.compile(r"\bTIC\s?\d{5,}\b", re.I),
    re.compile(r"\bJ\d{4,6}(?:\.\d+)?[+-]\d{4,6}(?:\.\d+)?\b"),
]


METHOD_PATTERNS = {
    "光谱拟合": r"\bspectral fit(?:ting)?\b|\bspectrum fit(?:ting)?\b|Balmer line fit|atmosphere fit",
    "SED拟合": r"\bSED fit(?:ting)?\b|spectral energy distribution|infrared excess",
    "测光分析": r"\bphotometr(?:y|ic)\b|light curve|colour|color[- ]magnitude",
    "时序测光": r"time[- ]series photometr|high[- ]speed photometr|fast photometr",
    "周期搜索": r"\bperiodogram\b|Lomb[-\s]?Scargle|phase[- ]fold|period search",
    "径向速度拟合": r"radial velocit(?:y|ies)|RV curve|orbital solution",
    "Gaia视差测量": r"\bGaia\b|parallax|astrometr(?:y|ic)",
    "塞曼分裂测量": r"Zeeman|magnetic field|field strength|spectropolarimetr|polarimetr",
    "回旋辐射拟合": r"cyclotron|polarimetric|AM Her",
    "X射线交叉匹配": r"X[- ]ray|ROSAT|XMM|Chandra|Swift|eROSITA",
    "贝叶斯/MCMC": r"\bMCMC\b|Markov Chain Monte Carlo|Bayesian",
    "星震学": r"asteroseismolog|g[- ]mode|period spacing|pulsat",
    "冷却年龄估计": r"cooling age|cooling sequence|cooling track|cooling model",
    "双星光变建模": r"eclipsing binary|eclipse model|ellipsoidal|binary light[- ]curve",
}


PARAMETER_PATTERNS = {
    "有效温度 Teff": r"\bT\s*[_ ]?\{?\\?rm\s*eff\}?|\bT[_ ]?eff\b|effective temperature",
    "表面重力 logg": r"\blog\s*g\b|\\log\s*g|surface gravity",
    "白矮星质量": r"\bM\s*[_ ]?\{?\\?rm\s*WD\}?|\bM_WD\b|white[- ]dwarf mass|stellar mass|\bmass\b",
    "白矮星半径": r"\bR\s*[_ ]?\{?\\?rm\s*WD\}?|\bR_WD\b|white[- ]dwarf radius|stellar radius|\bradius\b",
    "磁场强度": r"magnetic field|field strength|MG\b|kG\b|Zeeman",
    "轨道周期": r"orbital period|P[_ ]?orb|\bperiod\b",
    "脉动周期": r"pulsation period|period spacing|g[- ]mode",
    "径向速度": r"radial velocit(?:y|ies)|km\s*s",
    "视差": r"parallax|mas\b",
    "距离": r"\bdistance\b|pc\b|kpc\b",
    "冷却年龄": r"cooling age|cooling time",
    "吸积率": r"accretion rate|mass transfer rate|Mdot|\\dot",
    "金属丰度": r"metal abundance|abundance|polluted|Ca\s+II|Mg\s+II|Fe\s+I",
    "红外超额": r"infrared excess|IR excess|dust disk|debris disk",
}


FEATURE_SPECS: list[dict[str, Any]] = [
    {
        "name": "磁性白矮星",
        "aliases": ["magnetic white dwarf", "MWD", "magnetic WD"],
        "patterns": [
            r"magnetic white dwarf",
            r"\bMWDs?\b.*magnetic|magnetic.*\bMWDs?\b",
            r"Zeeman",
            r"field strength",
            r"\b\d+(?:\.\d+)?\s*MG\b",
            r"\bMG\b.*white dwarf",
            r"spectropolarimetr",
            r"cyclotron",
            r"\bpolar\b|AM\s*Her|intermediate polar",
        ],
        "methods": ["塞曼分裂测量", "回旋辐射拟合", "光谱拟合"],
        "parameters": ["磁场强度", "有效温度 Teff", "表面重力 logg"],
    },
    {
        "name": "短周期白矮星双星",
        "aliases": ["short-period white dwarf binary", "compact WD binary"],
        "patterns": [
            r"short[- ]period",
            r"ultra[- ]compact",
            r"period\s+(?:of\s+)?(?:\d+(?:\.\d+)?\s*)?(?:min|minutes|hr|hours)",
            r"\b\d+(?:\.\d+)?\s*(?:min|minutes|hr|hours)\b",
            r"compact binary",
            r"close binary",
            r"double white dwarf",
            r"\bDWD\b",
            r"\bP[_ ]?orb\b",
        ],
        "methods": ["周期搜索", "双星光变建模", "径向速度拟合", "时序测光"],
        "parameters": ["轨道周期", "径向速度", "白矮星质量"],
    },
    {
        "name": "激变变星/CV",
        "aliases": ["cataclysmic variable", "CV", "dwarf nova"],
        "patterns": [
            r"cataclysmic variable",
            r"\bCVs?\b",
            r"dwarf nova",
            r"nova[- ]like",
            r"AM\s*CVn",
            r"accreting white dwarf",
            r"mass transfer",
            r"accretion disk",
            r"emission line.*white dwarf|white dwarf.*emission line",
        ],
        "methods": ["光谱拟合", "周期搜索", "径向速度拟合", "X射线交叉匹配"],
        "parameters": ["轨道周期", "吸积率", "径向速度"],
    },
    {
        "name": "大质量/超大质量白矮星",
        "aliases": ["massive white dwarf", "ultra-massive white dwarf"],
        "patterns": [
            r"massive white dwarf",
            r"ultra[- ]massive",
            r"high[- ]mass white dwarf",
            r"\b1\.[0-4]\s*(?:M|M_\{?\\?odot|M_sun|solar mass)",
            r"\bM\s*[>=]\s*1\.",
        ],
        "methods": ["光谱拟合", "Gaia视差测量", "冷却年龄估计"],
        "parameters": ["白矮星质量", "白矮星半径", "表面重力 logg"],
    },
    {
        "name": "极低质量白矮星/ELM",
        "aliases": ["extremely low-mass white dwarf", "ELM WD"],
        "patterns": [
            r"extremely low[- ]mass",
            r"\bELM\b",
            r"low[- ]mass white dwarf",
            r"\b0\.[12]\d?\s*(?:M|M_\{?\\?odot|M_sun|solar mass)",
        ],
        "methods": ["光谱拟合", "径向速度拟合", "双星光变建模"],
        "parameters": ["白矮星质量", "表面重力 logg", "轨道周期"],
    },
    {
        "name": "脉动白矮星",
        "aliases": ["pulsating white dwarf", "ZZ Ceti", "DAV", "DBV"],
        "patterns": [
            r"pulsating white dwarf",
            r"ZZ\s*Ceti",
            r"\bDAV\b",
            r"\bDBV\b",
            r"pulsation",
            r"g[- ]mode",
            r"asteroseismolog",
        ],
        "methods": ["星震学", "周期搜索", "时序测光"],
        "parameters": ["脉动周期", "有效温度 Teff", "表面重力 logg"],
    },
    {
        "name": "金属污染/行星残骸白矮星",
        "aliases": ["polluted white dwarf", "debris disk white dwarf"],
        "patterns": [
            r"polluted white dwarf",
            r"metal[- ]polluted",
            r"planetary debris",
            r"debris disk",
            r"dust disk",
            r"Ca\s+II",
            r"circumstellar",
            r"tidally disrupted",
        ],
        "methods": ["光谱拟合", "SED拟合", "测光分析"],
        "parameters": ["金属丰度", "红外超额", "吸积率"],
    },
    {
        "name": "食白矮星双星",
        "aliases": ["eclipsing white dwarf binary", "eclipsing WD"],
        "patterns": [
            r"eclipsing white dwarf",
            r"eclipsing binary",
            r"\beclipse",
            r"transit",
            r"eclipse timing",
        ],
        "methods": ["双星光变建模", "时序测光", "周期搜索"],
        "parameters": ["轨道周期", "白矮星半径", "倾角"],
    },
    {
        "name": "X射线/高能白矮星系统",
        "aliases": ["X-ray white dwarf system", "high-energy compact binary"],
        "patterns": [
            r"X[- ]ray",
            r"eROSITA",
            r"ROSAT",
            r"XMM[- ]Newton",
            r"Chandra",
            r"Swift",
            r"hard X[- ]ray",
        ],
        "methods": ["X射线交叉匹配", "光谱拟合", "测光分析"],
        "parameters": ["吸积率", "轨道周期"],
    },
    {
        "name": "双简并/并合候选体",
        "aliases": ["double-degenerate candidate", "WD merger candidate"],
        "patterns": [
            r"double[- ]degenerate",
            r"merger candidate",
            r"Type Ia progenitor",
            r"gravitational wave",
            r"LISA",
            r"double white dwarf",
        ],
        "methods": ["径向速度拟合", "周期搜索", "Gaia视差测量"],
        "parameters": ["轨道周期", "白矮星质量", "径向速度"],
    },
]


FEATURE_BY_NAME = {spec["name"]: spec for spec in FEATURE_SPECS}


@dataclass(frozen=True)
class Paper:
    id: int
    bibcode: str
    title: str
    year: int | None
    journal: str
    abstract: str
    authors: list[str]
    categories: list[str]


def load_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def normalize_name(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_source_name(value: str) -> str:
    value = normalize_name(value)
    value = re.sub(r"^(?:source|object)\s+", "", value, flags=re.I)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,.;:")


def is_good_source_name(value: str) -> bool:
    name = clean_source_name(value)
    if not name or name in GENERIC_SOURCE_TOKENS:
        return False
    if len(name) < 5 or len(name) > 48:
        return False
    if name.lower() in {item.lower() for item in GENERIC_SOURCE_TOKENS}:
        return False
    return any(pattern.search(name) for pattern in OBJECT_PATTERNS)


def extract_sources(text: str, raw_objects: Iterable[Any], limit: int = 12) -> list[str]:
    hits: list[str] = []
    seen = set()
    for raw in raw_objects:
        name = clean_source_name(str(raw))
        key = name.lower()
        if is_good_source_name(name) and key not in seen:
            hits.append(name)
            seen.add(key)
    for pattern in OBJECT_PATTERNS:
        for match in pattern.finditer(text):
            name = clean_source_name(match.group(0))
            key = name.lower()
            if is_good_source_name(name) and key not in seen:
                hits.append(name)
                seen.add(key)
            if len(hits) >= limit:
                return hits
    return hits[:limit]


def detect_terms(text: str, patterns: dict[str, str]) -> set[str]:
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}


def detect_features(text: str) -> set[str]:
    found = set()
    for spec in FEATURE_SPECS:
        if any(re.search(pattern, text, re.I) for pattern in spec["patterns"]):
            found.add(spec["name"])
    return found


def source_regex(source: str) -> re.Pattern[str]:
    escaped = re.escape(source)
    flexible = escaped.replace(r"\ ", r"\s*")
    return re.compile(flexible, re.I)


def source_context(text: str, source: str, window: int = 700, max_windows: int = 4) -> str:
    """Return compact evidence windows around source mentions."""
    matches = list(source_regex(source).finditer(text))
    if not matches:
        compact_source = re.sub(r"\s+", "", source)
        if compact_source != source:
            matches = list(source_regex(compact_source).finditer(text))
    if not matches:
        return ""

    windows = []
    for match in matches[:max_windows]:
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        windows.append(text[start:end])
    return trim_evidence(" ... ".join(windows), max_len=max(window * max_windows, 1200))


def trim_evidence(text: str, max_len: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def node(label: str, name: str, schema_type: str, **properties: Any) -> dict[str, Any]:
    props = {"name": normalize_name(name), "schema_type": schema_type}
    for key, value in properties.items():
        if value not in (None, "", [], {}):
            props[key] = value
    return {"label": label, "properties": props}


def source_node(name: str, **properties: Any) -> dict[str, Any]:
    return node("entity", name, "AstronomicalSource", **properties)


def feature_node(name: str) -> dict[str, Any]:
    spec = FEATURE_BY_NAME.get(name, {})
    return node(
        "community",
        name,
        "SourceFeature",
        aliases=spec.get("aliases", []),
        description=f"面向调研的白矮星源特征：{name}",
    )


def method_node(name: str) -> dict[str, Any]:
    return node("entity", name, "MeasurementMethod")


def parameter_node(name: str) -> dict[str, Any]:
    return node("attribute", name, "PhysicalParameter")


def instrument_node(name: str) -> dict[str, Any]:
    schema_type = "Survey" if name in SURVEY_NAMES else "ObservationInstrument"
    return node("entity", name, schema_type)


def paper_node(paper: Paper) -> dict[str, Any]:
    return node(
        "keyword",
        paper.bibcode,
        "EvidencePaper",
        title=paper.title,
        year=paper.year,
        journal=paper.journal,
        authors=paper.authors[:8],
        categories=paper.categories,
    )


def add_relationship(
    relationships: list[dict[str, Any]],
    seen: set[tuple[str, str, str, str, str]],
    start_node: dict[str, Any],
    relation: str,
    end_node: dict[str, Any],
    source: str,
    evidence: str,
    chunk_id: str | int | None = None,
    score: float = 1.0,
    **properties: Any,
) -> bool:
    start_name = str(start_node["properties"].get("name", "")).lower()
    end_name = str(end_node["properties"].get("name", "")).lower()
    key = (start_node["label"], start_name, relation, end_node["label"], end_name)
    if key in seen:
        return False
    seen.add(key)
    rel = {
        "start_node": start_node,
        "relation": relation,
        "end_node": end_node,
        "source": trim_evidence(source),
        "evidence": evidence,
        "score": {
            "triple_support_score": score,
            "usefulness_score": score,
            "accuracy_score": score,
        },
    }
    if chunk_id is not None:
        rel["chunk_id"] = str(chunk_id)
    if properties:
        rel["properties"] = {k: v for k, v in properties.items() if v not in (None, "", [], {})}
    relationships.append(rel)
    return True


def get_papers(con: sqlite3.Connection) -> dict[int, Paper]:
    rows = con.execute(
        """
        SELECT
            p.id,
            p.bibcode,
            p.title,
            p.year,
            p.journal,
            p.abstract,
            p.authors_json,
            COALESCE(group_concat(c.name, ' ; '), '') AS categories
        FROM papers p
        LEFT JOIN paper_categories pc ON pc.paper_id = p.id
        LEFT JOIN categories c ON c.id = pc.category_id
        GROUP BY p.id
        """
    )
    papers = {}
    for pid, bibcode, title, year, journal, abstract, authors_json, category_blob in rows:
        papers[int(pid)] = Paper(
            id=int(pid),
            bibcode=str(bibcode),
            title=title or "",
            year=year,
            journal=journal or "",
            abstract=abstract or "",
            authors=load_json(authors_json, []),
            categories=[item.strip() for item in (category_blob or "").split(";") if item.strip()],
        )
    return papers


def chunk_rows(con: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    return con.execute(
        """
        SELECT
            c.id,
            c.paper_id,
            c.section,
            c.page_start,
            c.page_end,
            c.text,
            c.instruments_json,
            c.object_ids_json,
            c.methods_json,
            c.method_priority
        FROM chunks c
        WHERE c.source = 'metadata'
           OR c.method_priority = 1
           OR c.section IN ('Abstract', 'Methods', 'Observations', 'Data Reduction', 'Results')
        ORDER BY c.paper_id, c.id
        """
    )


def feature_method_defaults(feature: str) -> list[str]:
    spec = FEATURE_BY_NAME.get(feature, {})
    return list(spec.get("methods", []))


def feature_parameter_defaults(feature: str) -> list[str]:
    spec = FEATURE_BY_NAME.get(feature, {})
    return list(spec.get("parameters", []))


def build_graph(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    con = sqlite3.connect(args.rag_db)
    con.row_factory = sqlite3.Row
    papers = get_papers(con)

    relationships: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    source_profiles: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "features": Counter(),
            "methods": Counter(),
            "parameters": Counter(),
            "instruments": Counter(),
            "papers": Counter(),
            "sections": Counter(),
            "evidence": [],
        }
    )
    feature_profiles: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "sources": Counter(),
            "methods": Counter(),
            "parameters": Counter(),
            "papers": Counter(),
        }
    )

    chunks_scanned = 0
    chunks_used = 0
    papers_used: set[int] = set()

    for row in chunk_rows(con):
        if args.limit_chunks and chunks_scanned >= args.limit_chunks:
            break
        chunks_scanned += 1

        text = row["text"] or ""
        paper_id = int(row["paper_id"])
        paper = papers.get(paper_id)
        if paper is None:
            continue

        objects = load_json(row["object_ids_json"], [])
        sources = extract_sources(text, objects, limit=args.max_sources_per_chunk)
        if not sources:
            continue

        instruments = {
            normalize_name(item)
            for item in load_json(row["instruments_json"], [])
            if normalize_name(item)
        }

        chunk_id = row["id"]
        section = row["section"] or ""
        chunk_had_use = False

        for source in sources:
            context = source_context(text, source, window=args.source_context_chars)
            if not context and len(sources) == 1:
                context = trim_evidence(text, max_len=args.evidence_chars)
            if not context:
                continue

            features = detect_features(context)
            methods = detect_terms(context, METHOD_PATTERNS)
            parameters = detect_terms(context, PARAMETER_PATTERNS)

            # Use feature priors to connect an identified source trait with the
            # methods and parameters normally used to establish that trait.
            for feature in list(features):
                methods.update(feature_method_defaults(feature))
                parameters.update(feature_parameter_defaults(feature))

            if not features:
                continue

            chunk_had_use = True
            evidence = trim_evidence(context, max_len=args.evidence_chars)
            profile = source_profiles[source]
            profile["papers"][paper.bibcode] += 1
            profile["sections"][section] += 1
            if len(profile["evidence"]) < args.max_evidence_per_source:
                profile["evidence"].append(
                    {
                        "bibcode": paper.bibcode,
                        "year": paper.year,
                        "section": section,
                        "chunk_id": chunk_id,
                        "text": evidence,
                    }
                )

            snode = source_node(source, papers=list(profile["papers"].keys())[:8])
            pnode = paper_node(paper)
            add_relationship(
                relationships,
                seen,
                pnode,
                "报道源",
                snode,
                evidence,
                "The literature chunk discusses this astronomical source.",
                chunk_id,
                score=0.88,
            )

            for feature in sorted(features):
                profile["features"][feature] += 1
                feature_profiles[feature]["sources"][source] += 1
                feature_profiles[feature]["papers"][paper.bibcode] += 1
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "具有特征",
                    feature_node(feature),
                    evidence,
                    f"{source} is described with the source feature: {feature}.",
                    chunk_id,
                    score=0.92,
                    section=section,
                    bibcode=paper.bibcode,
                )

            for parameter in sorted(parameters):
                profile["parameters"][parameter] += 1
                for feature in features:
                    feature_profiles[feature]["parameters"][parameter] += 1
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "测得参数",
                    parameter_node(parameter),
                    evidence,
                    f"The chunk reports or constrains {parameter} for {source}.",
                    chunk_id,
                    score=0.86,
                    section=section,
                    bibcode=paper.bibcode,
                )

            for method in sorted(methods):
                profile["methods"][method] += 1
                for feature in features:
                    feature_profiles[feature]["methods"][method] += 1
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "用方法测量",
                    method_node(method),
                    evidence,
                    f"The source trait or parameter is supported by {method}.",
                    chunk_id,
                    score=0.84,
                    section=section,
                    bibcode=paper.bibcode,
                )

            for instrument in sorted(instruments):
                profile["instruments"][instrument] += 1
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "观测于",
                    instrument_node(instrument),
                    evidence,
                    f"The source is linked to observations from {instrument}.",
                    chunk_id,
                    score=0.82,
                    section=section,
                    bibcode=paper.bibcode,
                )

            for method in sorted(methods):
                for parameter in sorted(parameters):
                    add_relationship(
                        relationships,
                        seen,
                        method_node(method),
                        "测量",
                        parameter_node(parameter),
                        evidence,
                        f"{method} is used in the same source-focused evidence as {parameter}.",
                        chunk_id,
                        score=0.78,
                    )

        if chunk_had_use:
            chunks_used += 1
            papers_used.add(paper_id)

    con.close()

    if args.compact_graph:
        relationships = []
        seen = set()
        for source, profile in source_profiles.items():
            if not profile["features"]:
                continue
            evidence_items = profile.get("evidence") or []
            evidence = evidence_items[0]["text"] if evidence_items else ""
            papers_for_source = list(profile["papers"].keys())[: args.max_papers_per_source]
            snode = source_node(source, papers=papers_for_source)

            for feature, count in profile["features"].most_common(args.max_features_per_source):
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "具有特征",
                    feature_node(feature),
                    evidence,
                    f"{source} is repeatedly described with this source feature.",
                    score=0.92,
                    support_count=count,
                    bibcodes=papers_for_source[:6],
                )

            for method, count in profile["methods"].most_common(args.max_source_methods):
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "用方法测量",
                    method_node(method),
                    evidence,
                    f"Top measurement/analysis method associated with {source}.",
                    score=0.84,
                    support_count=count,
                    bibcodes=papers_for_source[:6],
                )

            for parameter, count in profile["parameters"].most_common(args.max_source_parameters):
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "测得参数",
                    parameter_node(parameter),
                    evidence,
                    f"Top physical parameter reported or constrained for {source}.",
                    score=0.86,
                    support_count=count,
                    bibcodes=papers_for_source[:6],
                )

            for instrument, count in profile["instruments"].most_common(args.max_source_instruments):
                add_relationship(
                    relationships,
                    seen,
                    snode,
                    "观测于",
                    instrument_node(instrument),
                    evidence,
                    f"Main survey/instrument linked to {source}.",
                    score=0.78,
                    support_count=count,
                    bibcodes=papers_for_source[:6],
                )

    # Add feature-level aggregation edges so users can start from a trait and
    # immediately see candidate methods/parameters without opening a paper node.
    for feature, profile in feature_profiles.items():
        fnode = feature_node(feature)
        for method, count in profile["methods"].most_common(args.max_feature_methods):
            add_relationship(
                relationships,
                seen,
                fnode,
                "常用测量方法",
                method_node(method),
                f"{count} source-feature evidence chunks",
                "Aggregated from source-focused white-dwarf literature evidence.",
                score=0.8,
                support_count=count,
            )
        for parameter, count in profile["parameters"].most_common(args.max_feature_parameters):
            add_relationship(
                relationships,
                seen,
                fnode,
                "关键参数",
                parameter_node(parameter),
                f"{count} source-feature evidence chunks",
                "Aggregated from source-focused white-dwarf literature evidence.",
                score=0.8,
                support_count=count,
            )
        for source, count in profile["sources"].most_common(args.max_feature_sources):
            add_relationship(
                relationships,
                seen,
                fnode,
                "代表源",
                source_node(source),
                f"{count} evidence chunks connect {source} to {feature}",
                "High-support source for this feature class.",
                score=0.76,
                support_count=count,
            )

    # Convert Counter-heavy source profiles into JSON-friendly summaries.
    profile_json = {}
    for source, profile in source_profiles.items():
        if not profile["features"]:
            continue
        profile_json[source] = {
            "features": dict(profile["features"].most_common()),
            "methods": dict(profile["methods"].most_common()),
            "parameters": dict(profile["parameters"].most_common()),
            "instruments": dict(profile["instruments"].most_common()),
            "papers": dict(profile["papers"].most_common(args.max_papers_per_source)),
            "sections": dict(profile["sections"].most_common()),
            "evidence": profile["evidence"],
        }

    relation_counts = Counter(rel["relation"] for rel in relationships)
    node_keys = set()
    node_type_counts = Counter()
    for rel in relationships:
        for side in ("start_node", "end_node"):
            item = rel[side]
            props = item.get("properties", {})
            key = (item.get("label"), str(props.get("name", "")).lower())
            node_keys.add(key)
            node_type_counts[props.get("schema_type", item.get("label"))] += 1

    top_sources = sorted(
        (
            {
                "source": source,
                "feature_hits": sum(profile["features"].values()),
                "features": dict(profile["features"].most_common(5)),
                "methods": dict(profile["methods"].most_common(5)),
                "papers": list(profile["papers"].keys())[:5],
            }
            for source, profile in source_profiles.items()
            if profile["features"]
        ),
        key=lambda item: item["feature_hits"],
        reverse=True,
    )[: args.max_summary_sources]

    summary = {
        "rag_db": str(Path(args.rag_db).resolve()),
        "kg_mode": "source_feature_literature_graph",
        "description": "Source-centric white-dwarf literature KG focused on object traits, measurement methods, and physical parameters.",
        "papers_total": len(papers),
        "papers_in_graph": len(papers_used),
        "chunks_scanned": chunks_scanned,
        "chunks_used": chunks_used,
        "relationships": len(relationships),
        "unique_nodes_estimate": len(node_keys),
        "source_profiles": len(profile_json),
        "relation_counts": dict(relation_counts.most_common()),
        "node_type_mentions": dict(node_type_counts.most_common()),
        "feature_counts": {
            feature: sum(profile["sources"].values())
            for feature, profile in sorted(feature_profiles.items())
        },
        "top_sources": top_sources,
        "limits": {
            "compact_graph": args.compact_graph,
            "max_sources_per_chunk": args.max_sources_per_chunk,
            "max_feature_sources": args.max_feature_sources,
            "max_features_per_source": args.max_features_per_source,
            "max_source_methods": args.max_source_methods,
            "max_source_parameters": args.max_source_parameters,
            "max_source_instruments": args.max_source_instruments,
            "evidence_chars": args.evidence_chars,
        },
    }
    return relationships, summary, profile_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a source-feature astronomy KG from the local white-dwarf RAG database."
    )
    parser.add_argument("--rag-db", default=str(DEFAULT_RAG_DB), help="Path to white_dwarf_rag.sqlite")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Dataset output root")
    parser.add_argument("--run-name", default=None, help="Output run folder name")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Debug: scan at most N chunks")
    parser.add_argument("--max-sources-per-chunk", type=int, default=8)
    parser.add_argument("--max-evidence-per-source", type=int, default=6)
    parser.add_argument("--max-papers-per-source", type=int, default=20)
    parser.add_argument("--max-feature-methods", type=int, default=10)
    parser.add_argument("--max-feature-parameters", type=int, default=10)
    parser.add_argument("--max-feature-sources", type=int, default=60)
    parser.add_argument("--max-features-per-source", type=int, default=4)
    parser.add_argument("--max-source-methods", type=int, default=1)
    parser.add_argument("--max-source-parameters", type=int, default=2)
    parser.add_argument("--max-source-instruments", type=int, default=0)
    parser.add_argument("--max-summary-sources", type=int, default=30)
    parser.add_argument("--evidence-chars", type=int, default=900)
    parser.add_argument("--source-context-chars", type=int, default=700)
    parser.add_argument("--expanded-graph", dest="compact_graph", action="store_false", help="Write every source evidence edge instead of the compact default graph.")
    parser.set_defaults(compact_graph=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag_db = Path(args.rag_db)
    if not rag_db.exists():
        raise FileNotFoundError(f"RAG database not found: {rag_db}")

    run_name = args.run_name or datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    relationships, summary, profiles = build_graph(args)
    graph_path = output_dir / "multi_stage_deduplicated.json"
    summary_path = output_dir / "summary.json"
    profile_path = output_dir / "source_profiles.json"

    graph_path.write_text(json.dumps(relationships, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"graph: {graph_path}")
    print(f"summary: {summary_path}")
    print(f"source_profiles: {profile_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
