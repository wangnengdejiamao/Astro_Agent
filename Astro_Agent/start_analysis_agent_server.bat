@echo off
setlocal

cd /d C:\Users\Administrator\Desktop\rag\Astro_Agent

set ASTRO_AGENT_MODEL_PROVIDER=deepseek
set DEEPSEEK_MODEL=deepseek-v4-pro
set MPLCONFIGDIR=%TEMP%\matplotlib

python -m uvicorn analysis_agent.server:app --host 127.0.0.1 --port 8765

pause
