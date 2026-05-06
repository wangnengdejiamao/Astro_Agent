@echo off
setlocal

cd /d C:\Users\Administrator\Desktop\rag

set ASTRO_AGENT_MODEL_PROVIDER=deepseek
set DEEPSEEK_MODEL=deepseek-v4-pro
set MPLCONFIGDIR=%TEMP%\matplotlib

python -m Astro_Agent.analysis_agent.graph_visualization_agent ^
  --use-llm ^
  --provider deepseek ^
  --output-root Astro_Agent\output\analysis_agent\kg_graph_report_deepseek_v4 ^
  --max-plot-nodes 900 ^
  --interactive-nodes 1200

pause
