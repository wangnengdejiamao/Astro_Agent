# Astro_Agent · 使用与部署指南

## 1. 这个 agent 是什么、能做什么

**Astro_Agent** 是基于 LangGraph 的天文科研全栈智能体平台，三层结构：

| 层 | 组件 | 入口 |
| --- | --- | --- |
| 工具层 | `astro_toolbox/` 30+ 模块（SDSS/DESI/LAMOST/KOA/HST/JWST/ZTF/WISE/Gaia/SED/HR/WD-fit/cooling-age/period/RV/orbit-traceback) | `python -m astro_toolbox.gui` |
| 知识层 | RAG: `white_dwarf_rag.sqlite` · KG: `graph_for_astronomy/output/white_dwarf_kg/` (12,740 节点 / 83,782 边) | `tools.search_rag` / `tools.search_kg` |
| 智能体层 | `analysis_agent/` Chief Investigator (12 个 LangGraph 节点：resolve → data_fetcher → rag/kg navigator → 三次迭代 → qa_gate → drafter (PaperOrchestra 五智能体) → peer_reviewer → toolbox_evolution) | CLI: `python -m analysis_agent.cli` · HTTP: `analysis_agent.server` |

**写文章 / Codex 工具箱已经接入**：
- `paper_orchestra.py` 已封装 PaperOrchestra 五智能体（Outline / Plotting / Literature Review / Section Writing / Content Refinement），不需要再外塞。
- `codex_tool.py` 新增：把 `vendor/codex-main` 与本地 Claude Code CLI 包成 agent 子进程工具，超时受控，输出 JSON 化。
- `codex_style.py` 已经把 codex 的 8 条工程纪律编码进 SharedContext 守则。

## 2. API 配置（已就位）

`.env` 在仓库根（**不入 git**），已写入 DeepSeek pro：

```
ASTRO_AGENT_MODEL_PROVIDER=deepseek
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_WIRE_API=chat_completions
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_API_KEY=sk-fbdd4d9358ad4998be7fd3bd370b79f7

# 同 key 可走 OpenAI / Anthropic 兼容端点
OPENAI_BASE_URL=https://api.deepseek.com
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
```

可换 `flash`：把 `DEEPSEEK_MODEL=deepseek-v4-flash`。

## 3. 命令行用法

**单次跑 agent（dry-run，不下载数据）**
```bash
python -m analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672
```

**真实跑 + 用 LLM 写文章**
```bash
python -m analysis_agent.cli "Gaia DR3 865415642195374464" --ra 232.3955 --dec 29.4672 --execute --use-llm
```

产物：`output/analysis_agent/<target>_<timestamp>/01..08_*.json` + `paper_orchestra/final/paper.tex`。

## 4. 前端 + HTTP API（新增）

```bash
cd /mnt/c/Users/Administrator/Desktop/rag/Astro_Agent
python -m uvicorn analysis_agent.server:app --host 0.0.0.0 --port 8765 --reload
```

浏览器开 `http://localhost:8765/` ，6 个 tab：

| Tab | 功能 |
| --- | --- |
| Agent 全流程 | 输入 target/RA/Dec，一键跑完整 LangGraph |
| RAG 检索 | 直接查 `white_dwarf_rag.sqlite`，可勾「method_only」 |
| 知识图谱 | 多 query 查 KG，返回方法迁移路径 |
| astro_toolbox | 选模块 + 函数 + RA/Dec，直接调单个工具 |
| Codex / Claude | 把任意工程任务派给 Codex CLI 或本地 Claude Code |
| 历史 Runs | 列 `output/analysis_agent/`，可读取产物 |

主题色与 PPT 完全一致：薄荷 `#12CCB9`、白底、左上 mint 短竖条。

### REST 端点速查
```
GET  /api/health
POST /api/agent/run         {target, ra, dec, execute, use_llm}
POST /api/rag/search        {query, method_only, limit}
POST /api/kg/search         {queries: [...], limit}
POST /api/toolbox/run       {module, function, ra, dec, radius_arcsec}
POST /api/codex/exec        {prompt, cwd, timeout}
POST /api/claude/exec       {prompt, cwd, timeout}
GET  /api/runs
GET  /api/runs/{name}/{file}
```

## 5. 文件改动清单

| 路径 | 用途 |
| --- | --- |
| `.env` | DeepSeek pro key + Codex/Claude bin 路径（**不入 git**） |
| `.gitignore` | 保护 .env、output/、data/ 等 |
| `analysis_agent/codex_tool.py` | Codex CLI / Claude Code CLI 子进程适配器 |
| `analysis_agent/server.py` | FastAPI 服务器，9 个端点 |
| `web/index.html` | 单页前端（6 tab，调全部接口） |

## 6. 健康检查

```bash
python -c "from analysis_agent.llm_client import LLMClient; c=LLMClient(); print(c.config); print('available:', c.available)"
```
若 `available: True` 即 DeepSeek key 加载成功。

```bash
curl http://localhost:8765/api/health
```
返回应包含 `"llm_available": true, "rag_db_exists": true, "kg_index_exists": true`。

## 7. 给老板演示的 30 秒动线

1. 终端起 `uvicorn analysis_agent.server:app --port 8765`。
2. 浏览器开 `http://localhost:8765/`。
3. **Agent 全流程** tab → 点「运行」→ 看右侧黑底面板里 8 个 JSON 落盘 + 一段 paper draft。
4. 切到 **知识图谱** → 输入 `SED fitting Bayesian inference parallax cooling age` → 看返回的方法迁移路径。
5. 切到 **astro_toolbox** → 选 `sdss` + `query_spectrum` + RA/Dec → 一键拿到光谱字典。
6. 切到 **Codex / Claude** → 直接把工程任务派出去。
