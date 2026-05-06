# Chief Investigator Analysis Agent

This package fuses three local systems into one auditable astronomy agent:

- `astro_toolbox` + local RAG/KG for science data, methods, and checks.
- PaperOrchestra for the five-agent manuscript pipeline.
- Codex-style operating rules for bounded context, structured tools, skills,
  review-first QA, and integration smoke tests.

It uses LangGraph when available and delegates fixed responsibilities to deterministic nodes:

- `Data Fetcher`: resolves coordinates, cross-matches SIMBAD, and runs or plans
  `astro_toolbox/run_single_target_all_tools.py`.
- `Structure Planner`: decides whether the run is spectroscopy+SED, HRD+SED
  photometric fallback, SED-only fallback, or insufficient-data.
- `RAG Navigator`: searches the local SQLite white-dwarf literature database.
- `KG Navigator`: searches the local prompt2graph knowledge graph export for
  method-transfer paths.
- `Method Scout`: investigates reusable or newer methods from RAG/KG and,
  optionally, an LLM provider such as Kimi.
- `Source Research Package`: for each target, downloads/loads all SIMBAD-linked
  references, checks exact RAG papers and KG source relations, and writes HST,
  SED, and spectral-line QA products before modeling claims are accepted.
- `Coder/QA`: enforces the mandatory three modeling iterations.
- `Model Supervisor`: audits model outputs after every run and creates repair
  actions for bad residuals, missing WD fitting exports, grid-boundary fits, or
  unsupported no-spectrum parameter claims.
- `Claude Code Delegate`: optionally sends Supervisor repair tasks to Claude Code.
- `Drafter`: creates an ApJ-style `aastex631` draft when the QA gate is clear.
- `Peer Reviewer`: writes sharp scientific review questions.
- `Toolbox Evolution`: records missing capabilities and requires code plus
  documentation updates after a confirmed tool gap.

The PaperOrchestra layer contains five explicit sub-agents:

- Outline Agent
- Plotting Agent
- Literature Review Agent
- Section Writing Agent
- Content Refinement Agent

Their astronomy-specific manifest is written to `paper_orchestra/agents_manifest.json`.
Codex-derived tool and context rules are written to
`paper_orchestra/codex_style_guidance.json`.

## Recommended Agent Base

For this project, LangGraph is the best local base because the workflow needs
checkpoint-like state, deterministic routing, human-in-the-loop pauses, and strict
three-iteration control. Chat-style multi-agent frameworks such as AutoGen or CrewAI
are useful for discussion, but the astronomy pipeline here benefits more from an
explicit graph whose state can be inspected after every node.

The current environment already has `langgraph==0.6.11` installed, so no download is
required for the first implementation.

## Usage

Plan a run without downloading survey data:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672
```

Execute the existing `astro_toolbox` modules:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672 --execute
```

Use the configured LLM provider for PaperOrchestra writing calls:

```bash
python -m Astro_Agent.analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672 --execute --use-llm --llm-provider deepseek
```

Use an existing astrotool output directory instead of downloading/running the
toolbox again:

```bash
python -m Astro_Agent.analysis_agent.cli "ZTFJ152934.91+292801.87" \
  --ra 232.39546190293 \
  --dec 29.46718626414 \
  --astrotool-run Astro_Agent/output/ZTFJ152934.91+292801.87 \
  --use-llm \
  --llm-provider deepseek \
  --kg-report \
  --kg-report-llm \
  --kg-report-provider deepseek \
  --source-research-package \
  --download-simbad-pdfs \
  --method-scout-llm \
  --method-scout-provider deepseek \
  --claude-timeout 300 \
  --max-supervision-rounds 3
```

For local debugging when SIMBAD/network access is slow, add `--skip-simbad`.
For a non-final writing smoke test while QA is still on hold, add
`--draft-on-hold`; the manuscript will preserve QA warnings and withhold final
physical parameters.

Provider config is read from environment variables only. See `.env.example`.
The agent also auto-loads private `.env` files from:

```text
./.env
Astro_Agent/analysis_agent/.env
prompt2graph_for_astronomy/.env
```

DeepSeek-compatible provider variables:

```text
ASTRO_AGENT_MODEL_PROVIDER=deepseek
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_API_KEY=<private>
```

Gemini-compatible provider variables:

```text
ASTRO_AGENT_MODEL_PROVIDER=gemini
GOOGLE_GEMINI_BASE_URL=https://code.newcli.com/gemini
GEMINI_MODEL=gemini-3-pro-preview
GEMINI_API_KEY=<private>
```

Kimi/Moonshot-compatible provider variables:

```text
ASTRO_AGENT_MODEL_PROVIDER=kimi
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=kimi-k2-latest
KIMI_API_KEY=<private>
```

To let the Supervisor hand concrete code/repair tasks to Claude Code:

```bash
python -m Astro_Agent.analysis_agent.cli "ZTFJ152934.91+292801.87" \
  --ra 232.39546190293 \
  --dec 29.46718626414 \
  --astrotool-run Astro_Agent/output/ZTFJ152934.91+292801.87 \
  --method-scout-llm \
  --method-scout-provider kimi \
  --enable-claude-code
```

If no spectra are available but SED/HRD products exist, the Structure Planner
switches to `photometric_hrd_sed_fallback`. In that mode the agent may estimate
provisional Teff/radius/luminosity from SED plus Gaia/HRD evidence, but it blocks
final spectral type, line detections, composition, precise logg, mass, and
cooling-age claims until stronger evidence is added.

Generate the white-dwarf KG overview images and report:

```bash
python -m Astro_Agent.analysis_agent.graph_visualization_agent --use-llm --provider deepseek
```

Windows launchers are available at:

```text
Astro_Agent/run_agent_deepseek_ztfj152934.bat
Astro_Agent/run_agent_kimi_method_scout_ztfj152934.bat
Astro_Agent/run_kg_report_deepseek.bat
Astro_Agent/start_analysis_agent_server.bat
```

The agent writes JSON checkpoints, an abnormal-analysis report or paper draft, peer
review output, and a toolbox-evolution plan under:

```text
Astro_Agent/output/analysis_agent/<target>_<timestamp>/
```

## Human Review Rule

The workflow pauses and writes `abnormal_analysis_report.md` if:

- coordinates or target identity cannot be resolved,
- the run is still a dry run,
- one or more required modules errored,
- WD fitting/RV/period evidence is missing for the intended interpretation,
- Model Supervisor still has unresolved repair actions,
- any of the three mandatory modeling iterations remains non-converged,
- rare spectral behavior or physically implausible parameters are detected.

No final numerical parameter table should be trusted until the QA gate is
`clear_for_draft`.
