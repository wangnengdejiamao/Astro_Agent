# Astro Agent — 天文智能分析与知识图谱平台

> 面向天文学研究的综合智能平台，集成**多源数据工具箱**、**论文分析 Agent** 与**知识图谱构建**三大核心能力。

---

## 项目简介

Astro Agent 是一个专为天体物理研究设计的智能工具平台，旨在打通**数据获取 → 分析计算 → 知识沉淀**的全链路：

- **数据层**：统一查询数十个天文数据库（DESI、Gaia、SDSS、HST、JWST 等）
- **分析层**：LLM 驱动的论文解读、光谱分析、参数计算 Agent
- **知识层**：从文献自动抽取天文知识图谱，支持社区发现与可视化

---

## 核心模块

### 1. Astro Agent / `analysis_agent`

天文论文智能分析 Agent，基于多节点工作流编排：

| 能力 | 说明 |
|------|------|
| `paper_orchestra` | 论文阅读与结构化解析 |
| `source_research_pipeline` | 源研究自动调研（SIMBAD、ADS、RAG 检索） |
| `llm_client` | 多模型统一接入（OpenAI、DeepSeek、Kimi） |
| `workflow` | LangGraph 状态机工作流 |
| `nodes/*` | 专用节点：Claude Code 诊断、代码审查、补丁评审、QA 门禁 |

**入口**：`analysis_agent/cli.py`、`analysis_agent/server.py`

### 2. Astro Toolbox / `astro_toolbox`

天文数据查询与处理工具箱，覆盖主流巡天与观测设施：

| 设施 | 功能 |
|------|------|
| `desi.py` | DESI 光谱查询与处理 |
| `gaia_lc.py` | Gaia 光变曲线 |
| `sdss.py` | SDSS 光谱与成像 |
| `hst.py` / `jwst.py` | 空间望远镜数据下载 |
| `kepler.py` / `tess.py` | 时域测光数据 |
| `lamost.py` | LAMOST 光谱 |
| `koa.py` | Keck 观测档案 |
| `ztf.py` | ZTF 暂现源 |
| `galex.py` / `wise.py` / `twomass.py` | 多波段测光 |
| `xray.py` | X 射线数据 |
| `sed.py` | 光谱能量分布拟合 |
| `hr_diagram.py` | 赫罗图绘制 |
| `orbit_traceback.py` | 轨道回溯 |
| `rv_fitting.py` / `rv_correction.py` | 视向速度拟合与改正 |
| `period_analysis.py` | 周期分析 |
| `wd_fitting.py` / `cooling_age.py` | 白矮星拟合与冷却年龄 |
| `six_dim.py` | 六维相空间分析 |

**入口**：`astro_toolbox/gui.py`（GUI）、`run_single_target_all_tools.py`（批量）

### 3. Graph for Astronomy / `graph_for_astronomy`

天文领域知识图谱自动构建工具（原 Prompt2Graph 天文适配版）：

- **端到端管线**：从 `corpus_cleaned.json` 一键生成知识图谱
- **Schema 引导抽取**：自定义天文实体类型（Star、Galaxy、Nebula、Telescope 等）
- **多阶段 LLM 抽取**：实体识别 → 关系抽取 → 属性抽取 → 可选验证
- **实体消歧**：自动合并同义实体与缩写
- **社区聚类**：Leiden / TreeComm 算法发现文献主题社区
- **质量评估**：LLM 多维打分
- **Neo4j 导入**：图数据库交互式查询
- **前端可视化**：轻量 HTML 本地预览

**入口**：`graph_for_astronomy/run_end2end_pipeline.py`

---

## 目录结构

```
Astro_Agent/
├── analysis_agent/          # 论文分析 Agent
│   ├── nodes/               # 工作流节点
│   ├── skills/              # 技能定义
│   ├── cli.py               # 命令行入口
│   ├── server.py            # 服务入口
│   └── workflow.py          # 工作流编排
├── astro_toolbox/           # 天文数据工具箱
│   ├── desi.py, gaia_lc.py, sdss.py, hst.py, jwst.py, ...
│   ├── gui.py               # 图形界面
│   └── config.py            # 全局配置
├── claude_code_toolbox/     # Claude Code 封装
├── configs/                 # 配置文件
├── desi_tool/               # DESI 专用工具
├── scripts/                 # 辅助脚本
├── templates/               # 论文模板（LaTeX）
├── web/                     # Web 界面
└── USAGE.md                 # 使用文档

graph_for_astronomy/         # 天文知识图谱构建
├── configs/                 # 管线 YAML 配置
├── docs/                    # 设计文档
├── frontend/                # 可视化前端
├── graph_tools/             # 图谱工具函数
├── prompts/                 # Prompt 模板
│   └── staged/              # 多阶段抽取模板
├── scorers/                 # 质量打分
├── staged_extraction/       # 多阶段抽取实现
├── utils/                   # 公共工具
├── build_white_dwarf_kg.py  # 白矮星 KG 构建
├── community_clustering.py  # 社区聚类
├── entity_deduplication.py  # 实体消歧
├── graph_builder.py         # 图谱构建器
├── graph_merger.py          # 图谱合并
├── graph_to_neo4j.py        # Neo4j 导入
├── meta_graph_builder.py    # 元图谱构建
├── prompt2graph.py          # 底层主入口
├── run_end2end_pipeline.py  # 【推荐】端到端入口
├── run_frontend_server.py   # 前端服务
└── README.md                # 模块文档
```

---

## 快速开始

### 环境准备

```bash
# Python 3.10+
pip install -r requirements.txt
```

> 核心依赖：`openai`, `langgraph`, `networkx`, `neo4j`, `leidenalg`, `astropy`, `astroquery`, `numpy`, `pandas`, `matplotlib`

### 配置环境变量

复制对应模块的 `.env.example` 为 `.env`，填入 API Key：

```bash
cp Astro_Agent/.env.example Astro_Agent/.env
cp graph_for_astronomy/.env.example graph_for_astronomy/.env
```

### 运行工具箱

```bash
# 单目标全工具查询
python Astro_Agent/astro_toolbox/run_single_target_all_tools.py

# 启动 GUI
python Astro_Agent/astro_toolbox/gui.py
```

### 运行 Agent

```bash
# 命令行模式
python Astro_Agent/analysis_agent/cli.py

# 服务化启动
python Astro_Agent/analysis_agent/server.py
```

### 运行知识图谱管线

```bash
# 准备语料：graph_for_astronomy/input/astronomy/corpus_cleaned.json
# 修改配置：graph_for_astronomy/configs/simple_pipeline.yml

python graph_for_astronomy/run_end2end_pipeline.py graph_for_astronomy/configs/simple_pipeline.yml
```

---

## 注意事项

1. **不要上传 `.env` 文件** — 已加入 `.gitignore`，内含 API Key 与数据库密码
2. **大文件已排除** — `output/`、`cache/`、`data/`、`.fits`、`.sqlite`、`.h5` 等不会进入版本控制
3. **工具箱依赖外部数据接口** — 部分功能需要网络连接与天文数据库账号（如 KOA、MAST）
4. **LLM 调用计费** — Agent 与知识图谱管线涉及多次 LLM API 调用，大批量运行前请估算 token 消耗
5. **知识图谱首次运行建议** — 先用 1–2 篇短文测试 `simple_pipeline.yml`，确认 Schema 与 Prompt 效果后再跑大批量文献

---

## 许可证

内部研究使用，请遵守相关数据合规与 API 服务条款。
