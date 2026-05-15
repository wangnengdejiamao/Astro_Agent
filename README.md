# Astro Agent

Astro Agent is an astronomy research agent workspace that combines three pieces:

- `Astro_Agent/analysis_agent`: a LangGraph-style scientific analysis agent for source research, evidence collection, QA, and manuscript drafting.
- `Astro_Agent/astro_toolbox`: survey and modeling tools for spectra, photometry, light curves, SEDs, white-dwarf checks, binaries, extinction, and traceback analysis.
- `graph_for_astronomy`: an astronomy knowledge-graph pipeline adapted from Prompt2Graph, with staged extraction, entity deduplication, community detection, and optional Neo4j import.

The repository is designed for code, prompts, configs, and reproducible scripts. Private `.env` files, downloaded papers, FITS products, SQLite indexes, local outputs, and personal reports are intentionally excluded from version control.

## What Is New

- Expanded Chief Investigator workflow with source research, per-source RAG, evidence manifests, physics checks, novelty detection, reflexion, workflow tracing, and paper QC.
- Added PaperOrchestra support for outline, plotting, literature review, section writing, refinement, LaTeX compilation, figures, comparison tables, and final QA.
- Added new astro toolbox modules for binary orbit estimates, cluster membership, compact-binary reporting, disk-eclipse MCMC, extinction, ingress measurement, decoupled SED fitting, and stellar templates.
- Added prompt tuning, ablation, prompt verification, and Codex review helper scripts.
- Updated web UI and service scripts for local interactive runs.

## Repository Layout

```text
.
├── Astro_Agent/
│   ├── analysis_agent/          # agent workflow, LLM clients, QA, paper pipeline
│   ├── astro_toolbox/           # astronomy data and modeling toolbox
│   ├── claude_code_toolbox/     # optional Claude Code subprocess wrapper
│   ├── scripts/                 # review, ablation, prompt tuning, helper scripts
│   ├── web/                     # local web UI
│   └── USAGE.md                 # longer usage notes
├── graph_for_astronomy/         # astronomy KG extraction and visualization
├── rag_pipeline/                # local RAG utilities and docs
├── start_services.sh            # local service launcher
├── stop_services.sh             # local service stopper
└── README.md
```

## Setup

Use Python 3.10 or newer. Create an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy matplotlib astropy astroquery requests python-dotenv pyyaml networkx fastapi uvicorn
python -m pip install openai langgraph neo4j json-repair scikit-learn
```

Optional packages depend on what you run:

```bash
python -m pip install lightkurve galpy dustmaps emcee corner
python -m pip install python-igraph leidenalg sentence-transformers
```

## Private Configuration

Do not commit real API keys. Copy one of the examples and edit it locally:

```bash
cp Astro_Agent/analysis_agent/.env.example Astro_Agent/analysis_agent/.env
cp Astro_Agent/astro_toolbox/.env.example Astro_Agent/astro_toolbox/.env
cp graph_for_astronomy/.env.example graph_for_astronomy/.env
```

Common variables:

```text
ASTRO_AGENT_MODEL_PROVIDER=deepseek
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_API_KEY=your_private_key_here

ADS_DEV_KEY=your_private_ads_key_here
GAIA_TOKEN=your_private_gaia_token_here
LAMOST_TOKEN=your_private_lamost_token_here
```

The root `.gitignore` excludes `.env`, local outputs, downloaded data, databases, PDFs, FITS files, logs, and local personal reports.

## Run The Agent

Dry-run a target without downloading survey products:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672
```

Run the toolbox-backed analysis:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672 --execute
```

Run with an LLM-backed writing/review pass:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672 --execute --use-llm --llm-provider deepseek
```

Start the local HTTP service and web UI:

```bash
python -m uvicorn Astro_Agent.analysis_agent.server:app --host 0.0.0.0 --port 8765 --reload
```

Then open `http://localhost:8765/`.

## Run The Astro Toolbox

```bash
python -m Astro_Agent.astro_toolbox.run_single_target_all_tools
```

Or import modules directly:

```python
from Astro_Agent.astro_toolbox import sdss, ztf, sed

ra, dec = 190.305, 2.596
spec = sdss.query_spectrum(ra, dec)
lc = ztf.query_lightcurve(ra, dec)

fitter = sed.SEDFitter(ra, dec)
fitter.collect_photometry()
fitter.apply_extinction()
```

Outputs and caches are written under ignored local directories such as `Astro_Agent/output/`, `Astro_Agent/data/`, and toolbox cache folders.

## Run The Knowledge Graph Pipeline

Prepare a local corpus JSON and keep large input/output data out of Git:

```text
graph_for_astronomy/input/<dataset>/corpus_cleaned.json
```

Run a pipeline config:

```bash
python graph_for_astronomy/run_end2end_pipeline.py graph_for_astronomy/configs/simple_pipeline.yml
```

Outputs are written under `graph_for_astronomy/output/`, which is ignored.

## Safety Checklist Before Pushing

Before publishing changes, run:

```bash
git status --short
git ls-files --others --exclude-standard
rg -n "sk-[A-Za-z0-9_-]{20,}|BEGIN .*PRIVATE KEY|password\\s*=" -g '!**/.git/**' -g '!**/.env' .
find . -path './.git' -prune -o -type f -size +50M -print
```

Only code, public prompts, public configs, docs, and lightweight examples should be committed. Keep `.env`, PDFs, FITS files, SQLite indexes, large KG outputs, private technical reports, and generated run artifacts local.

## License

Research use. Respect the terms of the external data services and model providers you configure locally.
