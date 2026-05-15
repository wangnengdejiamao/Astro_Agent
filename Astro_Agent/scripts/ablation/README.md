# Ablation experiment — Astro_Agent paper writer

This directory contains the experiment harness referenced by the
project plan at `/Users/a1/.claude/plans/noble-zooming-valiant.md`.

## Goal

Quantify the contribution of each design choice in the white-dwarf
paper-writing pipeline (prompt scaffolding, specialist split, KG
retrieval, per-source RAG, physics checks, reflexion, best-of-N) by
running a fixed set of configurations against two gold targets and
scoring the generated `paper.tex` along four axes:

- internal QC (16 checks in `paper_qc.py`)
- physics consistency (4 checks in `physics_checks.py`)
- gold-paper agreement (numeric IoU, claim overlap, bibcode Jaccard)
- simulated reviewer score (Claude playing ApJ referee)

## Independent variables (7 switches)

| Dimension              | Variants                                          |
| ---------------------- | ------------------------------------------------- |
| Prompt scaffolding     | `legacy` vs `wd_domain_v1` (this study)           |
| Specialist split       | monolithic vs physicist + writer + critic         |
| KG retrieval           | off / on                                          |
| Per-source RAG         | off / on                                          |
| Physics checks         | off / on                                          |
| Reflexion rounds       | 0 / 1 / 2 / 3                                     |
| Best-of-N (with RM)    | 1 / 4                                             |

Full factorial would be 2^5 × 4 × 2 = 256 cells. We use a
**leave-one-out** matrix instead:

- 1 baseline (everything on, reflexion=1, N=1)
- 7 single-feature-removed runs
- 4-point reflexion sweep
- 2-point best-of-N sweep

= 14 configurations × 2 gold targets = **28 cells**.

## Targets

| ID         | Designation         | Class                 | Gold reference                                      |
| ---------- | ------------------- | --------------------- | --------------------------------------------------- |
| UPK13c2    | UPK 13-c2           | WD + K-dwarf binary   | internal team gold (UPK13c2_gold.json)              |
| ZTFJ2130   | ZTF J2130+4420      | hot subdwarf + WD     | Kupfer/Bauer/Burdge+ 2020 ApJ (ZTFJ2130_gold.json)  |

Both gold JSON files list:
- `published_params`: list of (parameter, value, error, unit, claim)
- `key_claims`: list of declarative sentences
- `expected_bibcodes`: list of bibcodes the manuscript *should* cite
- `physics_checks_should_pass`: list of physics_check IDs

## Scoring

`score_against_gold.py` computes:

- **numeric_iou** — Jaccard of (value, unit) triples matched within
  5% relative tolerance.
- **claim_overlap** — fraction of gold key_claims for which at least
  one manuscript sentence shares ≥0.30 token-set Jaccard.
- **bibcode_jaccard** — Jaccard over the `\\citep{...}` keys.
- **physics_pass_rate** — fraction of `physics_checks_should_pass`
  that scored "pass" in `02i_physics_checks.json`.

Composite = `0.4·numeric + 0.25·claim + 0.2·bibcode + 0.15·physics`
(renormalised when physics is missing).

## Running the matrix

Prerequisites:
- `Astro_Agent/output/analysis_agent/UPK13c2_stage34/astrotool_run`
  must exist (will be reused so we don't re-fetch surveys).
- `Astro_Agent/output/analysis_agent/ZTFJ2130_v21_stage34/astrotool_run`
  must exist (similar).
- `provider=claude` in `Astro_Agent/.env` (or override per call).

Examples:

```bash
# Dry run (no LLM calls, just print the cell plan):
python scripts/ablation/run_matrix.py --rows baseline,loo_kg_off --targets UPK13c2 --dry-run

# Single cell:
python scripts/ablation/run_matrix.py --rows baseline --targets UPK13c2

# Full 28-cell grid (slow):
python scripts/ablation/run_matrix.py --all
```

Results append to `ablation_results.csv`. The columns are:

```
timestamp, config_id, target_id, output_root,
qc_pass, qc_warn, qc_fail, qc_verdict,
physics_pass_rate, numeric_iou, claim_overlap, bibcode_jaccard,
composite, reviewer_overall, wall_seconds, error
```

The Ablation Dashboard tab in the web UI reads this CSV and the
underlying per-run JSON to draw the heatmap and leave-one-out bar chart.

## Score a single existing manuscript

```bash
python scripts/ablation/score_against_gold.py \
    --paper output/analysis_agent/UPK13c2_stage34/paper_orchestra/final/paper.tex \
    --gold  scripts/ablation/golds/UPK13c2_gold.json \
    --physics output/analysis_agent/UPK13c2_stage34/02i_physics_checks.json
```

## What "passing" the ablation looks like

The baseline config (everything on) should achieve, per target:

- qc_pass ≥ 12 / qc_fail = 0
- numeric_iou ≥ 0.50
- claim_overlap ≥ 0.60
- bibcode_jaccard ≥ 0.40 (lower for ZTFJ2130 because the manuscript
  draws from per-source RAG which may miss the 2020 Kupfer bibcode)
- composite ≥ 0.50

Each leave-one-out config should be strictly worse than baseline along
at least one axis; if not, that feature is a candidate for removal.
