@echo off
setlocal

cd /d C:\Users\Administrator\Desktop\rag

set ASTRO_AGENT_MODEL_PROVIDER=deepseek
set DEEPSEEK_MODEL=deepseek-v4-pro
set MPLCONFIGDIR=%TEMP%\matplotlib

python -m Astro_Agent.analysis_agent.cli ZTFJ152934.91+292801.87 ^
  --ra 232.39546190293 ^
  --dec 29.46718626414 ^
  --output-root Astro_Agent\output\analysis_agent\ZTFJ152934_deepseek_agent_run ^
  --astrotool-run Astro_Agent\output\ZTFJ152934.91+292801.87 ^
  --use-llm ^
  --llm-provider deepseek ^
  --method-scout-llm ^
  --method-scout-provider deepseek ^
  --max-supervision-rounds 2 ^
  --draft-on-hold ^
  --kg-report ^
  --kg-report-llm ^
  --kg-report-provider deepseek

pause
