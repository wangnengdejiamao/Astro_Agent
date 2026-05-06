---
name: astro-paper-orchestra
description: Use when turning local astronomy analysis artifacts into an ApJ-style manuscript with the adapted PaperOrchestra workflow. Applies to unknown-object analyses using astro_toolbox, local RAG, knowledge graph method transfer, three mandatory modeling iterations, QA gating, peer review, and human-in-the-loop pauses.
---

# Astro PaperOrchestra

This skill adapts PaperOrchestra to the local astronomy workflow and folds in
Codex-style tool discipline.

## Core Rule

Never turn uncertified model output into final scientific claims. If the QA gate is
not clear, write an abnormal-analysis report and withhold final parameter tables.

## Workflow

1. Resolve target identity and ICRS coordinates.
2. Run or plan `astro_toolbox` data retrieval and modeling.
3. Search local SQLite RAG for method and citation evidence.
4. Search the local white-dwarf KG for cross-domain method transfer.
5. Enforce the three modeling iterations:
   - Baseline standard fitting.
   - Residual and physical-plausibility review.
   - Errors and systematics review.
6. If QA holds, stop at `abnormal_analysis_report.md`.
7. If QA clears, run the adapted PaperOrchestra package:
   - `inputs/idea.md`
   - `inputs/experimental_log.md`
   - `inputs/template.tex`
   - `inputs/conference_guidelines.md`
   - `outline.json`
   - `figures/captions.json`
   - `refs.bib`
   - `drafts/paper.tex`
   - `refinement/worklog.json`
   - `final/paper.tex`
   - `provenance.json`
8. Peer-review the draft with at least three sharp scientific questions.

## The Five Embedded Paper Agents

- `Outline Agent`: strict paper/figure/literature/section plan.
- `Plotting Agent`: figure/caption plan using astrotool outputs, never invented data.
- `Literature Review Agent`: local RAG/KG based citations and method context.
- `Section Writing Agent`: one coherent ApJ LaTeX draft.
- `Content Refinement Agent`: up to three reviewer/reviser iterations with revert rules.

## Codex-Style Discipline

- Bound every model-facing context item; store large evidence as artifacts.
- Keep data tools deterministic and reserve LLM calls for writing/critique.
- Prefer integration smoke tests over isolated checks for agent logic.
- Lead reviews with concrete risks and file/artifact references.
- Use small, local edits and update documentation when behavior changes.

## Astronomy Gates

Trigger human review when:

- target identity or coordinate units are ambiguous,
- strong or unusual emission lines appear,
- a model remains non-converged after three iterations,
- WD cooling age exceeds plausible cosmological limits without explanation,
- period folding may be an alias,
- Gaia parallax, extinction, flux calibration, or model-grid systematics are missing,
- tool coverage is absent and a new data adapter is required.

## LLM Use

Use LLM calls only for writing, outline refinement, and text critique. Keep data
fetching, RAG search, KG search, unit checks, and QA gates deterministic.

Secrets must come from environment variables:

- `OPENAI_API_KEY` for the `fox` Responses-compatible provider.
- `DEEPSEEK_API_KEY` for DeepSeek.

Never write live API keys to repo files.
