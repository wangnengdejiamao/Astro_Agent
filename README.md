# Astro Agent

An end-to-end astronomy research agent that connects survey-data acquisition,
quantitative modeling, evidence auditing, and manuscript drafting into a single
auditable LangGraph workflow.

This repository ships **code, prompts, configs, and reproducible scripts** only.
Private API keys, downloaded papers, FITS/spectra products, SQLite indexes,
local knowledge-graph workspaces, and personal reports are kept out of version
control and loaded at runtime from local `.env` files.

---

## 1. What This Project Is

Astro Agent is built around the question *"can a science-grade research workflow
be expressed as a graph of deterministic, reviewable agents?"* The system is
two cooperating layers:

- **`Astro_Agent/analysis_agent`** — a LangGraph state machine that resolves a
  target, fetches multi-survey data, runs modeling iterations, audits results
  against physics, retrieves comparable methods from local literature, and (when
  the QA gate clears) produces an ApJ-style manuscript with peer-review notes.
- **`Astro_Agent/astro_toolbox`** — a domain toolbox of survey clients and
  scientific modeling modules (spectra, photometry, light curves, SED, white
  dwarf fitting, RV/period analysis, extinction, kinematic traceback, cluster
  membership, compact-binary diagnostics).

A third local-only system (a literature-derived knowledge graph) can be plugged
in via `ASTRO_AGENT_KG_WORKSPACE`. The graph itself, its corpus, and its
extraction pipeline are **not part of this repository** — only the navigator
node that queries an external workspace is shipped.

---

## 2. System Architecture

```
                    ┌────────────────────────────────────────┐
                    │  CLI  ·  FastAPI server  ·  Web UI     │
                    └───────────────────┬────────────────────┘
                                        │
                ┌───────────────────────▼───────────────────────┐
                │       analysis_agent  (LangGraph state machine) │
                │   resolve → data_fetch → memory_advisor →     │
                │   structure_planner → rag/kg navigator →      │
                │   method_scout → source_research →            │
                │   iteration_1/2/3 → model_supervisor →        │
                │   claude_code_delegate → qa_gate ⇄ replan →   │
                │   drafter → paper_qc ⇄ reflexion → peer_review │
                └───────┬─────────────────────────────┬─────────┘
                        │                             │
            ┌───────────▼──────────┐     ┌────────────▼────────────┐
            │     astro_toolbox    │     │  local RAG + optional   │
            │  (survey + modeling) │     │  KG workspace (private) │
            └──────────────────────┘     └─────────────────────────┘
```

The agent and the toolbox communicate only through structured JSON artifacts
written to a per-run directory. This makes every step inspectable and replayable
without re-running upstream nodes.

---

## 3. Workflow Design

### 3.1 Why a state graph instead of free ReAct loops

Astronomical analysis is **long-horizon, audit-heavy, and partially
deterministic**: target classification dictates which physical model is valid,
and modeling claims must be backed by specific evidence. Free chain-of-thought
agents tend to hallucinate parameter values when evidence is missing. The
LangGraph state machine instead enforces:

- explicit per-node responsibilities and typed state (`AnalysisState` TypedDict),
- conditional edges for replan / reflexion / abnormal exit,
- a hard cap of three modeling iterations and at most two replans / two
  reflexion rewrites,
- file-system checkpoints (`01_resolved_target.json`, `02_data_fetch.json`, …)
  that double as human-readable provenance.

### 3.2 Node responsibilities

| Node | Responsibility | Notes |
|------|----------------|-------|
| `resolve` | SIMBAD cross-identification of name ↔ RA/Dec | offline-tolerant via `--skip-simbad` |
| `data_fetcher` | Parallel calls to 20+ survey clients in `astro_toolbox` | writes a unified `run_summary` |
| `memory_advisor` | Reads a SQLite ledger of past method/tool outcomes | guides planner toward known-good paths |
| `structure_planner` | Routes per SIMBAD class (WD / sdOB / CV / Polar / …) into spectroscopy+SED, HRD+SED photometric fallback, SED-only, or insufficient-data | each branch unlocks a different evidence set |
| `rag_navigator` | BM25 search over a local SQLite literature index, pre-tagged with 46 instruments and 24 method families | precise on domain jargon |
| `kg_navigator` | Optional method-transfer search over an external KG workspace | gracefully degrades when absent |
| `method_scout` | Compares RAG/KG hits to current toolbox capabilities, flags capability gaps | optionally LLM-assisted |
| `source_research` | Per-target evidence pack: SIMBAD-linked references, exact RAG matches, HST/SED/spectral-line QA | gates downstream modeling claims |
| `iteration_1/2/3` | Mandatory baseline → residuals → systematics passes | each must converge or be marked non-converged |
| `model_supervisor` | Audits residuals, grid-boundary fits, missing exports, no-spectrum claims, generates repair actions with `owner / priority / acceptance` | |
| `claude_code_delegate` | Optional handoff of repair actions to a Claude Code subprocess | |
| `qa_gate` | Routes `clear_for_draft` → drafter, `model_mismatch` → replan, otherwise → abnormal report | |
| `drafter` | PaperOrchestra five-agent manuscript pipeline (outline / plotting / lit-review / section-writing / refinement) producing `aastex631` LaTeX | |
| `paper_qc` | ApJ checklist: parameter table, units, citations, figures, tables | |
| `reflexion` | QC-driven targeted rewrite, hard-capped | |
| `peer_reviewer` | Generates four scientific-question review notes | |
| `toolbox_evolution` | Records confirmed capability gaps and required code/doc updates | |

### 3.3 Mandatory three modeling iterations

`iteration_1_baseline → iteration_2_residuals → iteration_3_systematics` is
non-skippable: a single best-fit number without residual diagnostics and
systematic checks is rejected by `qa_gate`. This is the main mechanism by
which the agent refuses to publish under-supported claims.

### 3.4 Model-mismatch self-heal

When SIMBAD identifies a target as e.g. sdOB or CV but the active branch is
white-dwarf fitting, `qa_gate` emits `model_mismatch`, the conditional edge
routes back to `structure_planner`, and a retry counter limits replanning to
two attempts. If the target's pipeline is not implemented, the run terminates
with `replan_blocked` and a human trigger entry instead of producing a paper.

### 3.5 Reflexion loop

`paper_qc` failures (missing parameter table, unsupported claim, citation gap)
feed structured findings into `reflexion`, which performs a *targeted* rewrite
of only the offending section rather than re-running the whole drafter. The
loop is bounded at two rewrites.

---

## 4. Methodological Choices

### 4.1 BM25 + rule-tagged retrieval over dense vectors

Astronomy text is dense in highly specific tokens (`DA white dwarf`, `logg`,
`Balmer lines`, `Lomb-Scargle`, `Bayestar2019`). Empirically BM25 with a
domain-rule pre-tagger (46 instruments × 24 method families) outperforms
generic dense retrieval on method-transfer queries, while remaining cheap and
auditable. The same store is used by both `rag_navigator` and `method_scout`.

### 4.2 Filesystem-as-memory

Every node writes a numbered JSON artifact to the run directory. This:

- removes framework lock-in (no proprietary checkpoint format),
- gives a human-readable provenance chain,
- enables `--astrotool-run <existing_dir>` to resume without re-downloading
  survey products,
- makes diffing two runs a `diff` away.

A separate SQLite **method-success ledger** is maintained across runs and read
by `memory_advisor`; it stores aggregate outcomes only, not raw data.

### 4.3 Evidence-gated parameter claims

`source_research` produces a per-target pack that explicitly lists which
modeling claims are *currently supported* and which are *blocked pending
evidence*. In `photometric_hrd_sed_fallback` (no spectra) the agent may report
provisional Teff / radius / luminosity from SED+Gaia HRD, but blocks final
spectral type, line detections, composition, precise log g, mass, and
cooling-age until stronger evidence is added. Drafter and paper_qc respect
these gates.

### 4.4 Supervisor-issued repair actions

`model_supervisor` does not "fix" results; it emits structured repair tasks
(`owner`, `priority`, `acceptance_criterion`). Repairs are then executed
either by the next iteration node or, optionally, by `claude_code_delegate`
calling Claude Code as a subprocess. This keeps science decisions and code
changes on separately reviewable artifacts.

### 4.5 PaperOrchestra: five-agent manuscript pipeline

Drafting is split into deterministic sub-roles:

- **Outline Agent** — section plan and required evidence per section,
- **Plotting Agent** — figure list keyed to evidence artifacts,
- **Literature Review Agent** — RAG-grounded references,
- **Section Writing Agent** — produces LaTeX per section against the outline,
- **Content Refinement Agent** — consistency, units, claim/evidence linking.

Each sub-role's prompt manifest is in
`paper_orchestra/agents_manifest.json`; the Codex-style tool/context rules
(bounded context window, structured tool I/O, review-first QA) live in
`paper_orchestra/codex_style_guidance.json`.

---

## 5. Astro Toolbox

### 5.1 Survey coverage

Spectroscopy: SDSS DR18, DESI DR1, LAMOST DR8, HST COS/STIS, JWST
NIRSpec/MIRI, GALAH DR4, KOA/Keck.
Photometry: SDSS *ugriz*, Gaia DR3, 2MASS, WISE, GALEX, SPHEREx.
Time-domain: ZTF DR23, TESS, Kepler/K2, Gaia epoch photometry, NEOWISE.
X-ray: ROSAT / XMM / Chandra via HEASARC.

### 5.2 Modeling modules

`sed`, `sed_decoupled` (Lin+2025 UPK 13-c2 two-component decoupling),
`wd_fitting` (Koester / TLUSTY atmospheres + cooling track + mass-radius),
`cooling_age`, `rv_fitting` (cross-correlation / template matching),
`period_analysis` (Lomb–Scargle + folding), `orbit_traceback` (6D phase-space
integration), `hr_diagram`, `cluster_membership` (kinematic + spatial χ²),
`extinction` (Bayestar2019 / SFD98), `compact_binary_report`,
`disk_eclipse_mcmc`, `ingress_measurement`, `binary_orbit`, stellar templates.

### 5.3 Single-target driver

`run_single_target_all_tools.py` orchestrates the toolbox end-to-end and
produces the JSON `run_summary` consumed by `data_fetcher`.

---

## 6. Repository Layout

```text
.
├── Astro_Agent/
│   ├── analysis_agent/          # LangGraph workflow, LLM clients, QA, paper pipeline
│   ├── astro_toolbox/           # survey clients + scientific modeling modules
│   ├── claude_code_toolbox/     # optional Claude Code subprocess wrapper
│   ├── scripts/                 # ablation, prompt tuning, review, helper scripts
│   ├── web/                     # local web UI
│   └── USAGE.md                 # extended usage notes
├── rag_pipeline/                # local literature RAG utilities
├── start_services.sh            # local launcher
├── stop_services.sh             # local stopper
└── README.md
```

Note: a private literature → KG workspace can live anywhere on disk and is
referenced via `ASTRO_AGENT_KG_WORKSPACE`. Its construction code, corpus, and
exports are **not** part of this repository.

---

## 7. Setup

Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib astropy astroquery requests \
    python-dotenv pyyaml networkx fastapi uvicorn
python -m pip install openai langgraph json-repair scikit-learn
```

Optional, depending on which modules you exercise:

```bash
python -m pip install lightkurve galpy dustmaps emcee corner sentence-transformers
```

Copy and edit local env files (never commit them):

```bash
cp Astro_Agent/analysis_agent/.env.example Astro_Agent/analysis_agent/.env
cp Astro_Agent/astro_toolbox/.env.example Astro_Agent/astro_toolbox/.env
```

Common variables:

```text
ASTRO_AGENT_MODEL_PROVIDER=deepseek            # or gemini / kimi
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_API_KEY=...

ADS_DEV_KEY=...
GAIA_TOKEN=...
LAMOST_TOKEN=...
ASTRO_AGENT_KG_WORKSPACE=/absolute/path/to/private/kg_workspace   # optional
```

---

## 8. Running

Plan-only (no downloads):

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" \
    --ra 232.3955 --dec 29.4672
```

Full toolbox-backed analysis:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" \
    --ra 232.3955 --dec 29.4672 --execute
```

LLM-backed writing/review:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" \
    --ra 232.3955 --dec 29.4672 --execute --use-llm --llm-provider deepseek
```

Local HTTP service + web UI:

```bash
python -m uvicorn Astro_Agent.analysis_agent.server:app \
    --host 0.0.0.0 --port 8765 --reload
# then open http://localhost:8765/
```

Toolbox stand-alone:

```bash
python -m Astro_Agent.astro_toolbox.run_single_target_all_tools
```

---

## 9. Outputs

Each run writes to `Astro_Agent/output/analysis_agent/<target>_<timestamp>/`
and contains:

- numbered JSON checkpoints per node,
- `run_summary.json` from the toolbox,
- `source_research/` evidence pack,
- either `paper/<aastex631>.tex` + figures + bibliography, or
  `abnormal_analysis_report.md` when QA blocks publishing,
- `peer_review.md`, `toolbox_evolution.md`,
- supervisor repair-action ledger.

All outputs are gitignored.

---

## 10. Data, Attribution, Safety

This repository contains orchestration code only and does not redistribute
third-party survey data. If you publish results derived from data fetched by
the toolbox, cite the original providers and follow their terms (SIMBAD/CDS,
VizieR, ADS, Gaia, SDSS, DESI, MAST, ZTF, WISE, 2MASS, GALEX, LAMOST, GALAH,
KOA/Keck, HEASARC, …).

Before pushing:

```bash
git status --short
git ls-files --others --exclude-standard
rg -n "sk-[A-Za-z0-9_-]{20,}|BEGIN .*PRIVATE KEY|password\s*=" \
    -g '!**/.git/**' -g '!**/.env' .
find . -path './.git' -prune -o -type f -size +50M -print
```

Only code, public prompts, public configs, docs, and lightweight examples
should be committed. Keep `.env`, PDFs, FITS, SQLite indexes, KG workspaces,
private reports, and run artifacts local.

## License

Research use. Respect the terms of the external data services and model
providers you configure locally.
