# Astro Agent

Astro Agent is an astronomy research agent workspace that combines two public pieces:

- `Astro_Agent/analysis_agent`: a LangGraph-style scientific analysis agent for source research, evidence collection, QA, and manuscript drafting.
- `Astro_Agent/astro_toolbox`: survey and modeling tools for spectra, photometry, light curves, SEDs, white-dwarf checks, binaries, extinction, and traceback analysis.

The repository is designed for code, prompts, configs, and reproducible scripts. Private `.env` files, downloaded papers, FITS products, SQLite indexes, local outputs, local knowledge-graph workspaces, and personal reports are intentionally excluded from version control.

## What Is New

- Expanded Chief Investigator workflow with source research, per-source RAG, evidence manifests, physics checks, novelty detection, reflexion, workflow tracing, and paper QC.
- Added PaperOrchestra support for outline, plotting, literature review, section writing, refinement, LaTeX compilation, figures, comparison tables, and final QA.
- Added new astro toolbox modules for binary orbit estimates, cluster membership, compact-binary reporting, disk-eclipse MCMC, extinction, ingress measurement, decoupled SED fitting, and stellar templates.
- Added prompt tuning, ablation, prompt verification, and Codex review helper scripts.
- Updated web UI and service scripts for local interactive runs.

## Workflow

1. Resolve a target name or RA/Dec and cross-check identity with public astronomy services.
2. Fetch or reuse local survey products: spectra, photometry, light curves, SED inputs, astrometry, and archive metadata.
3. Run toolbox modules for data quality checks, SED/HRD evidence, binary/orbit estimates, period/RV checks, extinction, and source-specific diagnostics.
4. Search local RAG and a private local KG index for comparable papers and reusable methods.
5. Build an evidence manifest, run physics/consistency checks, then draft or hold the paper depending on QA status.
6. Use review, ablation, prompt-tuning, and Codex-review scripts to improve prompts, tools, and workflow behavior.

The KG is not shipped in this public repository. Users should privately collect papers, convert them into a local corpus, run their own extraction pipeline, build a local SQLite/JSON index, and point the agent to it with `ASTRO_AGENT_KG_WORKSPACE`.

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
ASTRO_AGENT_KG_WORKSPACE=/absolute/path/to/private/kg_workspace
```

The root `.gitignore` excludes `.env`, local outputs, downloaded data, databases, PDFs, FITS files, logs, local knowledge-graph workspaces, and local personal reports.

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

## Data, Tools, And Attribution

This repository provides orchestration code and does not redistribute third-party survey data. If you use downloaded products or generated figures in a paper, cite the original data providers and follow their terms.

Common upstream astronomy services used by the toolbox include SIMBAD/CDS, VizieR, NASA ADS, Gaia, SDSS, DESI, MAST archives for HST/JWST/TESS/Kepler, ZTF, WISE, 2MASS, GALEX, LAMOST, KOA/Keck, GALAH, and related public catalogs.

Core software dependencies include Python, Astropy, Astroquery, NumPy, SciPy, Pandas, Matplotlib, NetworkX, FastAPI/Uvicorn, LangGraph, OpenAI-compatible API clients, and optional packages such as Lightkurve, Galpy, Dustmaps, emcee, corner, python-igraph, Leidenalg, SentenceTransformers, and Neo4j.

LLM providers, archive credentials, and private corpora stay in local `.env` files or ignored workspaces. Do not commit API keys, raw papers, FITS files, SQLite indexes, or generated KG outputs.

## Safety Checklist Before Pushing

Before publishing changes, run:

```bash
git status --short
git ls-files --others --exclude-standard
rg -n "sk-[A-Za-z0-9_-]{20,}|BEGIN .*PRIVATE KEY|password\\s*=" -g '!**/.git/**' -g '!**/.env' .
find . -path './.git' -prune -o -type f -size +50M -print
```

Only code, public prompts, public configs, docs, and lightweight examples should be committed. Keep `.env`, PDFs, FITS files, SQLite indexes, local knowledge-graph workspaces, private technical reports, and generated run artifacts local.

## License

Research use. Respect the terms of the external data services and model providers you configure locally.
