# Astro Research Multi-Agent Platform

## Global Directory Structure

```text
Astro_Agent/
├── astro_toolbox/                         # Domain tools: surveys, SED, WD fitting, RV, orbit traceback
├── vendor/
│   ├── PaperOrchestra-main/               # Upstream source, read-only reference
│   └── codex-main/                        # Upstream source, read-only reference
├── analysis_agent/                        # Integrated astronomy platform
│   ├── cli.py                             # User entry point
│   ├── workflow.py                        # LangGraph Chief Investigator state machine
│   ├── schemas.py                         # Pydantic SharedContext contract
│   ├── tools.py                           # astro_toolbox/RAG/KG adapters
│   ├── llm_client.py                      # fox/deepseek provider adapter via env vars
│   ├── paper_orchestra.py                 # Astronomy-adapted PaperOrchestra pipeline
│   ├── paper_agents.py                    # Five embedded paper-agent manifest
│   ├── codex_style.py                     # Codex-derived tool/safety/context rules
│   ├── skills/
│   │   └── astro-paper-orchestra/
│   │       └── SKILL.md                   # Codex-style skill for this domain workflow
│   └── .env.example                       # Secret-free provider configuration template
└── output/
    └── analysis_agent/<run_id>/
        ├── 01_resolved_target.json
        ├── 02_data_fetch.json
        ├── 03_rag_results.json
        ├── 04_kg_results.json
        ├── 05_iteration_1_baseline.json
        ├── 06_iteration_2_residuals.json
        ├── 07_iteration_3_systematics.json
        ├── 08_qa_gate.json
        ├── abnormal_analysis_report.md
        ├── agents_manifest.json
        ├── codex_style_guidance.json
        ├── toolbox_evolution_plan.json
        └── paper_orchestra/               # Created only when QA clears or direct paper test runs
            ├── inputs/
            │   ├── idea.md
            │   ├── experimental_log.md
            │   ├── template.tex
            │   └── conference_guidelines.md
            ├── outline.json
            ├── figures/
            │   ├── captions.json
            │   └── fig_workflow_overview.dot
            ├── refs.bib
            ├── citation_pool.json
            ├── drafts/
            │   ├── intro_relwork.tex
            │   └── paper.tex
            ├── refinement/
            │   └── worklog.json
            ├── final/
            │   └── paper.tex
            └── provenance.json
```

## Integration Contract

- PaperOrchestra is the manuscript pipeline.
- Codex is treated as a compute/code-repair node, not as the owner of scientific truth.
- `astro_toolbox`, RAG, and KG provide deterministic science artifacts.
- `SharedContext` in `schemas.py` is the single JSON state exchanged across all nodes.
- Drafter consumes figures and tables only through `ArtifactBundle` keys such as:
  - `sed_plot_path`
  - `hr_diagram_path`
  - `rv_table_latex`
  - `periodogram_plot_path`

## Safety Gates

QA blocks fitting or drafting when:

- target identity or J2000 coordinates are ambiguous,
- spectrum is inconsistent with the assumed stellar/white-dwarf class,
- a high-redshift quasar or nonstellar interpretation is plausible,
- fitting remains non-converged after three iterations,
- systematic errors are missing,
- Codex repair exceeds the retry limit.
