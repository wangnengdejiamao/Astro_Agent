# White Dwarf Knowledge Graph Design

This project is configured for a 3-stage extraction flow:

1. Stage 1: entity recognition and canonicalization.
2. Stage 2: relation extraction.
3. Stage 3: attribute extraction.

Stage 4 validation is disabled for the white-dwarf pipeline because it is expensive and not needed for the first production graph. Quality control is handled by conservative schema constraints, evidence fields, deterministic extraction from the RAG database, and later spot checks.

## Core Graph Patterns

- `AstronomicalSource - 使用 -> ObservationInstrument / Survey`
- `Paper - 提出 / 采用 -> AnalysisMethod`
- `Paper - 使用模型 -> PhysicalModel`
- `PhysicalModel - 拟合 -> PhysicalParameter`
- `AnalysisMethod - 测量 / 推断 / 约束 -> PhysicalParameter`
- `AnalysisMethod - 可迁移到 -> WhiteDwarfCategory`

## White-Dwarf Focus

The graph emphasizes methods and analysis relationships instead of only citation links. It is designed to answer questions such as:

- Which instruments and surveys are used for a given white-dwarf subtype?
- Which papers propose or apply a method?
- Which models fit `T_{\rm eff}`, `\log g`, mass, radius, cooling age, parallax, or magnetic field?
- Which methods bridge subfields, such as moving asteroseismology or light-curve modelling ideas into eclipsing white-dwarf binaries?

## DeepSeek

The LLM client reads:

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL`

It also accepts the OpenAI-compatible aliases:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

Use `https://api.deepseek.com` as the OpenAI-compatible base URL. Do not hard-code API keys in source files.
