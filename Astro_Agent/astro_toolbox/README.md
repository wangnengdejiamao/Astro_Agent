# Astro_Agent

`Astro_Agent` is a white-dwarf research toolbox plus a Hermes-style Agent layer. The toolbox collects spectra, photometry, light curves, SED products, HRD checks, RV measurements, WD fits, cooling ages, orbit traceback and 6D summary figures. The Agent layer reads those products, keeps reusable memory, applies a white-dwarf science design system, and writes paper-style research reports.

The project is designed for difficult sources: double white dwarfs, cluster white dwarfs, magnetic candidates, emission-line systems, and sources where the catalog label is probably not the full story.

## Current Architecture

```text
astro_toolbox/
├── run_single_target_all_tools.py      # Main data/toolbox pipeline for one target
├── run_existing_astro_output_analysis.py # Offline re-analysis of existing products
├── gui.py                             # Tk GUI, now connected to the real backend
├── tess.py, period_analysis.py         # TESS/ZTF/WISE/Gaia/Kepler light-curve work
├── wd_fitting.py, rv_fitting.py        # WD atmosphere and RV/DWD analysis
├── cooling_age.py, orbit_traceback.py  # WD age and cluster/orbit context
├── six_dim.py                          # Cluster/DWD science summary figures
└── hermes_wd_agent/
    ├── run_wd_agent.py                 # Agent CLI
    ├── workflow/                       # End-to-end WD workflow
    ├── memory/                         # L1 task state, L2 SQLite source archive, L5 preferences
    ├── semantic_library/               # L3 rules, method reliability, paper patterns
    ├── wd_skills/                      # L4 branch skills: single WD, DWD, cluster WD
    ├── prompts/                        # Report and analysis prompts
    └── tools/                          # Existing output reader, toolbox runner, Hermes bridge
```

The important rule is simple:

- `run_single_target_all_tools.py` is the data layer.
- `hermes_wd_agent/` is the scientific reasoning and memory layer.
- `gui.py` starts the data layer first, then automatically runs the WD Agent report/memory step on the output.

## Environment

Use the project virtual environment when it exists:

```bash
cd /Users/a1/Desktop/desi匹配
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install numpy pandas scipy matplotlib astropy astroquery requests lightkurve galpy dustmaps
```

Optional tools:

```bash
.venv/bin/python -m pip install pypeit
```

Local/private credentials should stay outside Git:

```bash
export ADS_DEV_KEY="..."
export GAIA_TOKEN="..."
export LAMOST_TOKEN="..."
export LAMOST_FTP_USER="..."
export LAMOST_FTP_PASSWORD="..."
```

`.env`, `output/`, `data/`, `literature/`, FITS files, generated CSV/PDF/PNG products and logs are ignored by Git.

## Run The Toolbox

From the parent directory:

```bash
cd /Users/a1/Desktop/desi匹配
.venv/bin/python -m astro_toolbox.run_single_target_all_tools \
  --target ZTFJ035315.74+095633.40 \
  --ra 58.3155833333 \
  --dec 9.9426111111 \
  --output-root output/astro_output/ZTFJ035315.74+095633.40/toolbox_run
```

Main products:

```text
output/astro_output/<target>/
├── module_status.csv
├── run_summary.json
├── target_info.json
├── sdss/ desi/ lamost/ ztf/ tess/ wise/ sed/ ...
├── period_analysis/
├── wd_fitting/
├── rv/
├── cooling_age/
├── orbit_traceback/
└── six_dim/
```

## Run The WD Agent

To read existing toolbox products without downloading new data:

```bash
cd /Users/a1/Desktop/desi匹配
.venv/bin/python -m astro_toolbox.hermes_wd_agent.run_wd_agent \
  --ra 58.3155833333 \
  --dec 9.9426111111 \
  --target ZTFJ035315.74+095633.40 \
  --input-output-root DWD_new/astro_output/ZTFJ035315.74+095633.40/fix_validation_20260505 \
  --output-root output/astro_output/ZTFJ035315.74+095633.40/agent_report \
  --no-hermes
```

Agent products:

```text
<agent_output_root>/
├── wd_agent_report.md
├── wd_agent_report_context.json
├── source_memory.md
├── figure_manifest.csv
└── wd_agent_summary.json

output/astro_output/wd_agent_memory/
├── wd_memory.sqlite
├── l1_tasks/*.json
└── generated_skills/*.md
```

`--no-hermes` keeps the run deterministic and uses the local report generator. If the official Hermes package is installed, the bridge in `hermes_wd_agent/tools/hermes_bridge.py` can call it; if not, the local workflow still works.

## Run The GUI

```bash
cd /Users/a1/Desktop/desi匹配
.venv/bin/python -m astro_toolbox.gui
```

The GUI now uses the real backend. A single target run does this:

1. Runs `astro_toolbox.run_single_target_all_tools`.
2. Watches `module_status.csv` and updates the progress table.
3. Recursively scans generated figures for preview.
4. Runs `astro_toolbox.hermes_wd_agent.run_wd_agent --no-hermes`.
5. Writes `wd_agent_report.md`, `source_memory.md` and L1/L2 memory records.

It no longer imports the removed `test_toolbox.AstroQueryAll` test adapter.

## Agent Design System

The Agent design system is already in the repo and should remain separate from the raw data tools:

- L1 task memory: `hermes_wd_agent/memory/astronomy_memory.py`
- L2 source archive: `hermes_wd_agent/memory/wd_memory_db.py`
- L3 semantic rules: `hermes_wd_agent/semantic_library/*.json`
- L4 branch skills: `hermes_wd_agent/wd_skills/*/*.md`
- L5 preferences: `hermes_wd_agent/config/default_preferences.json`
- Report workflow: `hermes_wd_agent/workflow/wd_full_analysis_workflow.py`

Good integration pattern:

1. Keep survey downloads and numerical fitting in `astro_toolbox`.
2. Keep memory, branch choice, report writing and paper logic in `hermes_wd_agent`.
3. Add new science logic first as structured toolbox output, then teach `ExistingOutputReader` to parse it into the Agent report context.

This avoids turning the Agent into a second copy of the toolbox.

## Local Test Commands

Compile check:

```bash
cd /Users/a1/Desktop/desi匹配
PYTHONPYCACHEPREFIX=/private/tmp/astro_pycache \
MPLCONFIGDIR=/private/tmp/astro_mpl \
.venv/bin/python -m compileall -q astro_toolbox
```

Offline Agent smoke test:

```bash
cd /Users/a1/Desktop/desi匹配
rm -rf /private/tmp/astro_agent_smoke
PYTHONPYCACHEPREFIX=/private/tmp/astro_pycache \
MPLCONFIGDIR=/private/tmp/astro_mpl \
.venv/bin/python -m astro_toolbox.hermes_wd_agent.run_wd_agent \
  --ra 58.3155833333 \
  --dec 9.9426111111 \
  --target ZTFJ035315.74+095633.40 \
  --input-output-root DWD_new/astro_output/ZTFJ035315.74+095633.40/fix_validation_20260505 \
  --output-root /private/tmp/astro_agent_smoke/ZTFJ035315.74+095633.40 \
  --memory-root /private/tmp/astro_agent_smoke/memory \
  --no-hermes
```

Expected outputs are `wd_agent_report.md`, `source_memory.md`, `figure_manifest.csv`, `wd_agent_summary.json`, `memory/l1_tasks/*.json` and `memory/wd_memory.sqlite`.

## What To Improve Next

For publication-level DWD/cluster-WD science, the next useful code work is:

- Extend `ExistingOutputReader` so it parses period-analysis CSVs, magnetic-field summaries, RV tables and six-dimensional cluster rows from old output folders, not just figures.
- Add a small target-level "science score" table: DWD evidence, cluster evidence, magnetic evidence, CV/emission evidence, WD-fit reliability and missing-data risk.
- Let the Agent compare WD cooling age, cluster age and orbital/RV evidence in one structured block.
- Keep TESS/ZTF folded-light-curve products in a machine-readable table so the Agent can quote the real period, alias and uncertainty instead of only listing PNG files.

## Git Hygiene

Commit source code, Agent rules, skills, prompts and README files. Do not commit downloaded survey data, local credentials, large FITS products, generated figures, generated CSV outputs or temporary logs.

Typical publish flow:

```bash
git status
git add README.md gui.py hermes_wd_agent
git commit -m "Integrate WD Agent memory workflow"
git push
```
