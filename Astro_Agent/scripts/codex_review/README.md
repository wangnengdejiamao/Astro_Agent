# Codex as five-persona reviewer

This directory pairs an alignment prompt with a runner that invokes the
local Codex CLI to act as five reviewers at once:

  * **Audit Officer** — checks every value/citation in paper.tex is
    traceable to a run artifact.
  * **Working Astronomer** — checks the 12 white-dwarf domain priors
    and 12 failure modes from `analysis_agent/prompts/wd_domain.py`.
  * **ApJ Referee** — scores on the same 4×25 rubric as the in-system
    `paper_orchestra.llm_review` step, so alignment with the internal
    reviewer can be measured.
  * **Field Professor** — novelty against the gold paper +
    follow-up-observation suggestions + ApJ acceptance probability.
  * **Toolbox & KG Auditor** — checks skipped/nonconverged workflow
    nodes, KG/RAG retrieval alignment, method-scout/toolbox gaps, and
    citation provenance.

The output is a single JSON document that is written to
`<run>/09c_reviewer.json` and whose `next_actions_for_reflexion` field
is appended to `<run>/09b_reflexion.json`. On the next agent re-run,
the drafter sees those advice strings in its evidence block and is
forced to address each one.

## Quick start

```bash
# Make sure the codex CLI is on PATH (or set $CODEX_BIN).
export CODEX_BIN=codex     # or absolute path

# Dry-run: prints the assembled system+user prompts without calling Codex
python scripts/codex_review/run_codex_review.py --run UPK13c2_stage34 --dry-run

# Real run (will produce 09c_reviewer.json under the run directory)
python scripts/codex_review/run_codex_review.py --run UPK13c2_stage34

# Explicit gold paper (auto-detected if omitted)
python scripts/codex_review/run_codex_review.py --run ZTFJ2130_v21_stage34 \
    --gold scripts/ablation/golds/ZTFJ2130_gold.json
```

## What gets sent to Codex

System prompt: `analysis_agent/prompts/codex_reviewer_alignment.md` (the
five-persona role pack).

User message (assembled on the fly):
- `paper.tex` (truncated at 18 kB)
- `refs.bib` (first 4 kB)
- stage artifacts: paper_qc, physics_checks, published_params,
  source_rag, cluster_membership, hypothesis_plan, novelty,
  comparison_table, qa_gate, reflexion, analysis_plan, plus KG/RAG,
  method_scout/toolbox, iteration, SED, light-curve, MCMC, and extinction
  artifacts used by the Toolbox & KG Auditor
- `analysis_agent/prompts/retrieval.py` so Persona E can compare
  observed rerank keys against `RERANK_KEYS[source_class]`
- The matching gold paper from `scripts/ablation/golds/`

## What gets written back

- `output/analysis_agent/<run>/09c_reviewer.json` — the full reviewer
  JSON (schema `codex_reviewer_v2`).
- `output/analysis_agent/<run>/09b_reflexion.json` — each entry in
  `next_actions_for_reflexion` is appended as an `action_items` row
  with `check_id: codex_reviewer`. Also stamps
  `codex_reviewer_overall`, `codex_reviewer_decision`,
  `codex_reviewer_timestamp`, plus Persona E scores
  `codex_toolbox_coverage_score` and `codex_kg_alignment_score` when
  `toolbox_kg_audit` is present. The runner also appends a small number
  of deterministic `toolbox_kg_audit` action items for node-status,
  per-source RAG, and citation-provenance failures.

## Closing the loop

After Codex writes its critique, re-run the workflow with `use_llm=True`
on the same `output_root`; the drafter will pick up the new
`action_items` automatically (existing reflexion mechanism in
`analysis_agent/paper_orchestra.py` already injects
`reflexion_history[-1].verbal_reflection` into the section evidence
block). The result is a closed alignment loop:

```
agent draft → in-system paper_qc → Codex 5-persona review →
  reflexion advice → agent rewrite → repeat
```

## Tuning the five-persona prompt

Edit `analysis_agent/prompts/codex_reviewer_alignment.md` directly.
Changes take effect on the next invocation — no restart needed. The
prompt file is plain Markdown so you can review the diff in git.
