@echo off
setlocal

cd /d C:\Users\Administrator\Desktop\rag

set ASTRO_AGENT_MODEL_PROVIDER=kimi
set KIMI_MODEL=kimi-k2-latest
set MPLCONFIGDIR=%TEMP%\matplotlib

python -m Astro_Agent.analysis_agent.cli ZTFJ152934.91+292801.87 ^
  --ra 232.39546190293 ^
  --dec 29.46718626414 ^
  --output-root Astro_Agent\output\analysis_agent\ZTFJ152934_kimi_supervised_run ^
  --astrotool-run Astro_Agent\output\ZTFJ152934.91+292801.87 ^
  --skip-simbad ^
  --method-scout-llm ^
  --method-scout-provider kimi ^
  --llm-provider kimi ^
  --max-supervision-rounds 2 ^
  --draft-on-hold

pause
