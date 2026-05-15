"""Workflow trace helpers for the Astro_Agent front end.

The analysis workflow already writes durable JSON artifacts.  This module adds a
small presentation layer: a stable step blueprint plus compact input/output
summaries that let the UI show how state moves through the agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PACKAGE_DIR = Path(__file__).resolve().parent
ASTRO_AGENT_DIR = PACKAGE_DIR.parent


STEP_SPECS: List[Dict[str, Any]] = [
    {
        "id": "resolve",
        "label": "目标解析",
        "node": "resolve",
        "owner": "Data Fetcher",
        "purpose": "确认目标身份，将目标名或用户给定坐标统一为 ICRS 十进制度。",
        "input_keys": ["target", "ra_deg", "dec_deg"],
        "output_keys": ["resolved", "ra_deg", "dec_deg"],
        "artifact_files": ["01_resolved_target.json"],
        "science_checks": ["RA 必须在 [0, 360) deg", "Dec 必须在 [-90, 90] deg", "记录坐标来源和单位转换"],
    },
    {
        "id": "data_fetcher",
        "label": "数据获取",
        "node": "data_fetcher",
        "owner": "Data Fetcher",
        "purpose": "执行或规划 SIMBAD 交叉匹配和 astro_toolbox 多巡天查询。",
        "input_keys": ["resolved", "skip_simbad", "astrotool_run", "dry_run", "force"],
        "output_keys": ["data_fetch"],
        "artifact_files": ["02_data_fetch.json"],
        "science_checks": ["区分真实非探测、网络失败和 dry-run", "只把成功产物作为后续物理证据"],
    },
    {
        "id": "structure_planner",
        "label": "结构规划",
        "node": "structure_planner",
        "owner": "Structure Planner",
        "purpose": "根据已可用的数据选择光谱+SED、HRD+SED、SED-only 或数据不足路线。",
        "input_keys": ["data_fetch"],
        "output_keys": ["analysis_plan"],
        "artifact_files": ["02b_analysis_plan.json"],
        "science_checks": ["失败模块不能当作可用证据", "无光谱时禁止声称谱型、谱线、精确 logg 或成分"],
    },
    {
        "id": "rag_navigator",
        "label": "RAG 文献检索",
        "node": "rag_navigator",
        "owner": "RAG Navigator",
        "purpose": "从本地白矮星文献库检索可复用的方法学段落。",
        "input_keys": ["analysis_plan"],
        "output_keys": ["rag_results"],
        "artifact_files": ["03_rag_results.json"],
        "science_checks": ["区分文献支持与假设", "检索缺失时继续运行但标记风险"],
    },
    {
        "id": "kg_navigator",
        "label": "知识图谱检索",
        "node": "kg_navigator",
        "owner": "KG Navigator",
        "purpose": "从本地知识图谱中找方法迁移路径和相关证据。",
        "input_keys": ["analysis_plan", "rag_results"],
        "output_keys": ["kg_results"],
        "artifact_files": ["04_kg_results.json"],
        "science_checks": ["图谱边只作为方法提示，不直接当作目标源观测事实"],
    },
    {
        "id": "kg_graph_report",
        "label": "KG 总览报告",
        "node": "kg_graph_report",
        "owner": "Graph Visualization Agent",
        "purpose": "可选生成知识图谱总览、社区和可视化报告。",
        "input_keys": ["kg_report", "kg_report_llm", "kg_results"],
        "output_keys": ["kg_graph_report"],
        "artifact_files": ["04b_kg_graph_report.json"],
        "science_checks": ["报告失败不能阻塞基础分析，但要暴露错误"],
    },
    {
        "id": "method_scout",
        "label": "方法侦察",
        "node": "method_scout",
        "owner": "Method Scout",
        "purpose": "结合 RAG/KG 和可选 LLM，提取可执行算法规范，生成可验证的工具箱缺口。",
        "input_keys": ["analysis_plan", "rag_results", "kg_results", "method_scout_llm"],
        "output_keys": ["method_scout", "toolbox_gap"],
        "artifact_files": ["04c_method_scout.json", "04e_toolbox_gap.json"],
        "science_checks": ["LLM 建议必须标注证据来源或人审假设", "不能发明观测结果"],
    },
    {
        "id": "source_research_package",
        "label": "源研究包",
        "node": "source_research_package",
        "owner": "Source Research Package",
        "purpose": "可选构建 SIMBAD/RAG/KG/SED/谱线证据包，给论文写作提供可追溯事实。",
        "input_keys": ["source_research_package", "ra_deg", "dec_deg", "data_fetch"],
        "output_keys": ["source_research"],
        "artifact_files": ["04d_source_research_package.json"],
        "science_checks": ["论文声明必须追溯到本地证据包", "谱线计数需有 S/N 和连续谱窗口"],
    },
    {
        "id": "iteration_1_baseline",
        "label": "迭代 1：基线",
        "node": "iteration_1_baseline",
        "owner": "Coder/QA",
        "purpose": "检查基线 astro_toolbox 产物和第一轮物理拟合状态。",
        "input_keys": ["data_fetch", "rag_results"],
        "output_keys": ["iterations"],
        "artifact_files": ["05_iteration_1_baseline.json"],
        "science_checks": ["dry-run 不能认证数值结果", "缺 WD fitting 时禁止最终参数表"],
    },
    {
        "id": "iteration_2_residuals",
        "label": "迭代 2：残差与物理",
        "node": "iteration_2_residuals",
        "owner": "Coder/QA",
        "purpose": "检查残差、谱线、边界解和物理不可能参数区域。",
        "input_keys": ["data_fetch", "kg_results", "iterations"],
        "output_keys": ["iterations"],
        "artifact_files": ["06_iteration_2_residuals.json"],
        "science_checks": ["冷却年龄、logg、红外超额和周期别名必须通过物理守门"],
    },
    {
        "id": "iteration_3_systematics",
        "label": "迭代 3：误差系统学",
        "node": "iteration_3_systematics",
        "owner": "Coder/QA",
        "purpose": "把测光零点、消光、视差、模型网格、周期/RV 系统误差纳入审查。",
        "input_keys": ["data_fetch", "iterations"],
        "output_keys": ["iterations"],
        "artifact_files": ["07_iteration_3_systematics.json"],
        "science_checks": ["系统误差缺失时不能给最终置信区间"],
    },
    {
        "id": "model_supervisor",
        "label": "模型监督",
        "node": "model_supervisor",
        "owner": "Model Supervisor",
        "purpose": "汇总模型风险，生成必须修复的代码/数据/论文任务。",
        "input_keys": ["analysis_plan", "data_fetch", "source_research", "method_scout", "iterations"],
        "output_keys": ["model_supervision"],
        "artifact_files": ["07b_model_supervision.json"],
        "science_checks": ["未解决 repair actions 时 QA 必须 hold", "无光谱 fallback 必须保留参数 caveat"],
    },
    {
        "id": "claude_code_delegate",
        "label": "代码代理委派",
        "node": "claude_code_delegate",
        "owner": "Claude Code Delegate",
        "purpose": "按开关把监督器任务和文献方法工具箱缺口派给本地代码代理。",
        "input_keys": ["enable_claude_code", "model_supervision", "toolbox_gap"],
        "output_keys": ["claude_code", "dynamic_skill_registration"],
        "artifact_files": ["07c_claude_code.json"],
        "science_checks": ["默认不自动改科学结论", "任何代码代理输出都需要人审"],
    },
    {
        "id": "qa_gate",
        "label": "QA 门禁",
        "node": "qa_gate",
        "owner": "QA Gate",
        "purpose": "决定进入论文草稿还是异常报告；所有科学风险在这里集中暴露。",
        "input_keys": ["data_fetch", "iterations", "model_supervision"],
        "output_keys": ["qa", "next_step"],
        "artifact_files": ["08_qa_gate.json"],
        "science_checks": ["human_review_required=True 时不得发布最终参数表"],
    },
    {
        "id": "abnormal_report",
        "label": "异常报告",
        "node": "abnormal_report",
        "owner": "Human Review",
        "purpose": "当 QA hold 时写出暂停原因、已完成迭代和人工复核建议。",
        "input_keys": ["qa", "iterations"],
        "output_keys": ["abnormal_report"],
        "artifact_files": ["abnormal_analysis_report.md", "agents_manifest.json", "codex_style_guidance.json"],
        "science_checks": ["异常报告优先于论文草稿", "明确禁止信任未通过 QA 的数值"],
    },
    {
        "id": "drafter",
        "label": "论文草稿",
        "node": "drafter",
        "owner": "PaperOrchestra",
        "purpose": "QA 通过或允许 draft-on-hold 时，生成 ApJ 风格草稿和方法上下文。",
        "input_keys": ["qa", "rag_results", "kg_results", "artifacts"],
        "output_keys": ["paper", "paper_orchestra"],
        "artifact_files": ["paper/method_context.md"],
        "science_checks": ["只能使用本地产物、RAG/KG 证据和显式引用", "hold 状态下必须保留警告"],
    },
    {
        "id": "memory_advisor",
        "label": "记忆顾问 (CORAL)",
        "node": "memory_advisor",
        "owner": "Memory Advisor",
        "purpose": "查询跨 run 学习账本，给本次结构规划提供历史方法成功率、已实现假设、星团先验。",
        "input_keys": ["target", "resolved"],
        "output_keys": ["memory_advice"],
        "artifact_files": ["02a_memory_advice.json"],
        "science_checks": ["不向前看泄漏：不能用本次 run 自身的数据预测自身的成员性"],
    },
    {
        "id": "extinction",
        "label": "消光查询 (A_V)",
        "node": "extinction",
        "owner": "Extinction",
        "purpose": "查 Bayestar2019 (3D) / SFD98 (2D) 得到 A_V；为下游 SED 拟合提供 dereddening。",
        "input_keys": ["ra_deg", "dec_deg", "published_params"],
        "output_keys": ["extinction"],
        "artifact_files": ["02g_extinction.json"],
        "science_checks": ["distance_pc 优先用 Gaia parallax 而不是 SIMBAD", "fallback 时标 provenance"],
    },
    {
        "id": "sed_decoupled",
        "label": "SED 3 步解耦拟合",
        "node": "sed_decoupled",
        "owner": "SED Decoupled",
        "purpose": "Step1 F_diff (extinction-independent) / Step2 F_low / Step3 F_high χ² 比较多假设。",
        "input_keys": ["extinction", "data_fetch", "flux_high", "flux_low"],
        "output_keys": ["sed_decoupled"],
        "artifact_files": ["02h_sed_decoupled.json"],
        "science_checks": ["F_diff 必须扣完 A_V 后比较", "joint χ² 排序后给 best_hypothesis_joint"],
    },
    {
        "id": "light_curve_geometry",
        "label": "光变形态测量",
        "node": "light_curve_geometry",
        "owner": "Light Curve Geometry",
        "purpose": "对相位折叠光变曲线做梯形食模型拟合，输出 t_ingress / 食占比 τ/P / 形态判别；同时给 Kepler 轨道 a / v_orb / Roche 半径。",
        "input_keys": ["data_fetch", "published_params", "analysis_plan"],
        "output_keys": ["light_curve_geometry"],
        "artifact_files": ["02j_light_curve_geometry.json"],
        "science_checks": ["flat_bottomed vs U_shaped morphology 必须报告", "周期来源优先级：literature > photometric > skip"],
    },
    {
        "id": "eclipse_mcmc",
        "label": "盘食 MCMC (e, ω, α)",
        "node": "eclipse_mcmc",
        "owner": "Eclipse MCMC",
        "purpose": "对 (e, ω, α) 跑 emcee 或确定性网格 MCMC，给后验 16/50/84 百分位。Source class 不合适或形态非 flat-bottomed 时跳过。",
        "input_keys": ["light_curve_geometry"],
        "output_keys": ["eclipse_mcmc"],
        "artifact_files": ["02k_eclipse_mcmc.json"],
        "science_checks": ["只对 morphology=flat_bottomed 或 disk-eclipsing-binary 假设触发", "emcee 缺失时降级到 deterministic_grid"],
    },
    {
        "id": "physics_checks",
        "label": "物理一致性论证",
        "node": "physics_checks",
        "owner": "Physics Checks",
        "purpose": "生成 Rayleigh-Jeans / Ingress-time / Tidal-truncation 等论证段落，喂给 drafter。",
        "input_keys": ["sed_decoupled", "published_params", "analysis_plan"],
        "output_keys": ["physics_checks"],
        "artifact_files": ["02i_physics_checks.json"],
        "science_checks": ["每段必须可由数据复算", "M_tot 默认值要在 assumptions 字段披露"],
    },
    {
        "id": "cluster_membership",
        "label": "开放星团成员性",
        "node": "cluster_membership",
        "owner": "Cluster Membership",
        "purpose": "对 Hunt+2023 开放星团目录计算 χ²_spat / χ²_kin / RV-σ，作为 discussion 的成员性段落证据。",
        "input_keys": ["data_fetch", "published_params"],
        "output_keys": ["cluster_membership"],
        "artifact_files": ["02e_cluster_membership.json"],
        "science_checks": ["要求空间 + 运动学 + 视差三项分别给 σ", "不能把单变量近似作为成员性结论"],
    },
    {
        "id": "ads_live",
        "label": "ADS 实时检索",
        "node": "ads_live",
        "owner": "ADS Live",
        "purpose": "用 ADS_DEV_KEY 在目标坐标 0.005 deg 内、2018+ 年发表的文献中再次检索；合并入 per-source RAG。无 key 时优雅跳过。",
        "input_keys": ["target", "ra_deg", "dec_deg", "source_rag"],
        "output_keys": ["ads_live"],
        "artifact_files": ["02l_ads_live.json"],
        "science_checks": ["无 ADS_DEV_KEY 必须跳过而不报错", "合并后必须更新 source_rag.sqlite"],
    },
    {
        "id": "novelty_detector",
        "label": "新颖性差异化",
        "node": "novelty_detector",
        "owner": "Novelty Detector",
        "purpose": "把 published_params 按参数分组，对 this_work 与 literature 同名参数计算 Δ 和 Δ/σ；标 confirm/consistent/tension/extend/new。",
        "input_keys": ["published_params"],
        "output_keys": ["novelty"],
        "artifact_files": ["02m_novelty.json"],
        "science_checks": ["confirm/tension 阈值 |Δ/σ|<1 / >3", "无误差棒标 no_error_bars"],
    },
    {
        "id": "comparison_table",
        "label": "文献对照表",
        "node": "comparison_table",
        "owner": "Comparison Table",
        "purpose": "在 KNOWN_SYSTEMS 库里找当前 source_class 的 benchmark systems，生成 deluxetable*。",
        "input_keys": ["analysis_plan", "published_params", "eclipse_mcmc", "light_curve_geometry"],
        "output_keys": ["comparison_table"],
        "artifact_files": ["02n_comparison_table.json"],
        "science_checks": ["每个 benchmark 必须给 bibcodes", "目标行必须用本次 run 数据填"],
    },
    {
        "id": "figure_synthesizer",
        "label": "Figure 自合成",
        "node": "figure_synthesizer",
        "owner": "Figure Synthesizer",
        "purpose": "在 paper_orchestra/figures/ 下生成 lightcurve / SED / cluster / corner 4 张 PNG。matplotlib 缺失时 graceful skip。",
        "input_keys": ["data_fetch", "sed_decoupled", "cluster_membership", "eclipse_mcmc", "light_curve_geometry"],
        "output_keys": ["figures"],
        "artifact_files": ["11_figures.json", "paper_orchestra/figures/*.png"],
        "science_checks": ["每张图必须带 caption 字段", "无数据时返回 status 不抛异常"],
    },
    {
        "id": "latex_compile",
        "label": "LaTeX 编译",
        "node": "latex_compile",
        "owner": "LaTeX Compile",
        "purpose": "把 paper_orchestra/final/paper.tex 用 latexmk → PDF；无 latexmk/pdflatex 时跳过。",
        "input_keys": ["paper_orchestra"],
        "output_keys": ["latex_compile"],
        "artifact_files": ["12_latex_compile.json", "paper_orchestra/build/paper.pdf (if success)"],
        "science_checks": ["失败时必须提取 ! 错误行供调试", "无编译器时 status=no_latex_compiler"],
    },
    {
        "id": "paper_qc",
        "label": "论文 QC 检查",
        "node": "paper_qc",
        "owner": "Paper QC",
        "purpose": "对 drafter 产物运行 ApJ checklist：章节齐全、abstract 数值、引文密度、bibcode 解析、括号平衡、不确定度语言、novelty 段。",
        "input_keys": ["paper", "paper_orchestra", "published_params"],
        "output_keys": ["paper_qc"],
        "artifact_files": ["09_paper_qc.json"],
        "science_checks": ["pass 阈值 = 0 fail 且 warn≤2", "失败时不应进入投稿环节"],
    },
    {
        "id": "peer_reviewer",
        "label": "同行审查",
        "node": "peer_reviewer",
        "owner": "Peer Reviewer",
        "purpose": "结合 paper_qc 结果与领域问题，给出审查报告。",
        "input_keys": ["paper", "paper_qc", "qa", "iterations"],
        "output_keys": ["peer_review"],
        "artifact_files": ["paper/peer_review.md"],
        "science_checks": ["审查问题应指向可验证的产物或缺口", "必须复述 paper_qc verdict"],
    },
    {
        "id": "reflexion",
        "label": "反思自评 (Reflexion)",
        "node": "reflexion",
        "owner": "Reflexion Critic",
        "purpose": "把 paper_qc 的 fail/warn 转成 verbal reflection；当存在可处理的失败且未达重试上限时回到 drafter 做定向重写。",
        "input_keys": ["paper_qc"],
        "output_keys": ["reflexion_history"],
        "artifact_files": ["09b_reflexion.json"],
        "science_checks": ["重试次数受 max_reflexion_retries 限制", "每条 action_item 必须有 section 和 advice"],
    },
    {
        "id": "kg_writeback",
        "label": "学习账本写回",
        "node": "kg_writeback",
        "owner": "Learning Ledger",
        "purpose": "把本次 run 的方法-状态、参数、假设结果、星团成员性写入 SQLite 学习账本，供后续同类源查询。",
        "input_keys": ["analysis_plan", "qa", "published_params", "hypothesis_plan", "cluster_membership"],
        "output_keys": ["kg_writeback"],
        "artifact_files": ["10_kg_writeback.json", "_learning_ledger.sqlite (累积)"],
        "science_checks": ["每条 method 必须带 status 和 timestamp", "失败的方法也必须写入账本以便后续避坑"],
    },
    {
        "id": "toolbox_evolution",
        "label": "工具箱演进",
        "node": "toolbox_evolution",
        "owner": "Toolbox Evolution",
        "purpose": "把失败模块和缺失能力转化为后续代码/文档更新计划。",
        "input_keys": ["data_fetch", "qa", "peer_review", "abnormal_report"],
        "output_keys": ["toolbox_evolution"],
        "artifact_files": ["toolbox_evolution_plan.json"],
        "science_checks": ["确认是工具缺口后才要求新增脚本", "更新代码时同步 smoke test 和 README"],
    },
]


def workflow_blueprint() -> List[Dict[str, Any]]:
    return [dict(step, status="blueprint") for step in STEP_SPECS]


def _short_text(value: Any, limit: int = 220) -> str:
    text = str(value).replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _compact(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return _short_text(value, 160)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) else _short_text(value, 260)
    if isinstance(value, list):
        return {
            "type": "list",
            "count": len(value),
            "sample": [_compact(item, depth + 1) for item in value[:3]],
        }
    if isinstance(value, dict):
        preferred = {}
        for key in (
            "status",
            "route",
            "apj_gate",
            "human_review_required",
            "next_step",
            "resolver",
            "target",
            "ra_deg",
            "dec_deg",
            "output_root",
            "rows",
            "n",
            "count",
            "error",
            "message",
        ):
            if key in value:
                preferred[key] = _compact(value[key], depth + 1)
        warnings = value.get("warnings") or value.get("reasons") or value.get("human_review_triggers")
        if warnings:
            preferred["warnings_or_reasons"] = _compact(warnings, depth + 1)
        if preferred:
            preferred["keys"] = sorted(str(k) for k in value.keys())[:24]
            return preferred
        return {
            "type": "dict",
            "keys": sorted(str(k) for k in value.keys())[:24],
            "sample": {str(k): _compact(v, depth + 1) for k, v in list(value.items())[:5]},
        }
    return _short_text(value)


def _collect_issues(value: Any) -> List[str]:
    issues: List[str] = []
    if isinstance(value, dict):
        for key in ("error", "message"):
            if key in value and value[key]:
                issues.append(_short_text(value[key], 360))
        for key in ("warnings", "reasons", "human_review_triggers"):
            raw = value.get(key)
            if isinstance(raw, list):
                issues.extend(_short_text(item, 360) for item in raw if item)
        if value.get("status") in {"error", "failed", "nonconverged"}:
            issues.append(f"status={value.get('status')}")
        for nested in value.values():
            issues.extend(_collect_issues(nested)[:6])
    elif isinstance(value, list):
        for item in value[:8]:
            issues.extend(_collect_issues(item)[:6])
    seen = set()
    out = []
    for item in issues:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out[:12]


def _artifact_url(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(ASTRO_AGENT_DIR.resolve())
    except Exception:
        return None
    return "/api/files/" + str(rel).replace("\\", "/")


def _artifact_entry(path: Path) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        entry["size_bytes"] = path.stat().st_size
        entry["url"] = _artifact_url(path)
    return entry


def _artifact_lookup(run_dir: Path | None, artifacts: Iterable[str]) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for raw in artifacts:
        try:
            path = Path(raw)
        except TypeError:
            continue
        lookup[path.name] = path
        if run_dir:
            with_prefix = str(path).replace(str(run_dir) + "/", "")
            lookup[with_prefix] = path
    if run_dir and run_dir.exists():
        for path in run_dir.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(run_dir))
                lookup[rel] = path
                lookup[path.name] = path
    return lookup


def _load_run_state(run_dir: Path) -> Dict[str, Any]:
    state: Dict[str, Any] = {"output_root": str(run_dir), "artifacts": []}
    file_map = {
        "01_resolved_target.json": "resolved",
        "02_data_fetch.json": "data_fetch",
        "02b_analysis_plan.json": "analysis_plan",
        "03_rag_results.json": "rag_results",
        "04_kg_results.json": "kg_results",
        "04b_kg_graph_report.json": "kg_graph_report",
        "04c_method_scout.json": "method_scout",
        "04e_toolbox_gap.json": "toolbox_gap",
        "04d_source_research_package.json": "source_research",
        "07b_model_supervision.json": "model_supervision",
        "07c_claude_code.json": "claude_code",
        "08_qa_gate.json": "qa",
        "toolbox_evolution_plan.json": "toolbox_evolution",
    }
    iterations = []
    for filename, key in file_map.items():
        path = run_dir / filename
        if path.exists():
            state["artifacts"].append(str(path))
            try:
                state[key] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                state[key] = {"status": "error", "error": f"Could not read {filename}: {exc}"}
    for filename in ("05_iteration_1_baseline.json", "06_iteration_2_residuals.json", "07_iteration_3_systematics.json"):
        path = run_dir / filename
        if path.exists():
            state["artifacts"].append(str(path))
            try:
                iterations.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                iterations.append({"status": "error", "error": f"Could not read {filename}: {exc}"})
    if iterations:
        state["iterations"] = iterations
    for path in run_dir.rglob("*"):
        if path.is_file() and str(path) not in state["artifacts"]:
            state["artifacts"].append(str(path))
    resolved = state.get("resolved") or {}
    state["target"] = resolved.get("target") or run_dir.name
    state["ra_deg"] = resolved.get("ra_deg")
    state["dec_deg"] = resolved.get("dec_deg")
    qa = state.get("qa") or {}
    state["next_step"] = "abnormal" if qa.get("human_review_required") else "paper"
    return state


def _step_status(outputs: Dict[str, Any], artifacts: List[Dict[str, Any]]) -> str:
    if not outputs and not any(item.get("exists") for item in artifacts):
        return "pending"
    issues = _collect_issues(outputs)
    if any("status=error" in issue or "status=failed" in issue for issue in issues):
        return "error"
    if any("status=nonconverged" in issue for issue in issues):
        return "warning"
    if any(issue for issue in issues):
        return "warning"
    if outputs:
        raw_status = None
        for value in outputs.values():
            if isinstance(value, dict) and value.get("status"):
                raw_status = value.get("status")
                break
        if raw_status == "skipped":
            return "skipped"
    return "completed"


def build_trace(state: Dict[str, Any], run_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    run_path = Path(run_dir or state.get("output_root", "")) if (run_dir or state.get("output_root")) else None
    lookup = _artifact_lookup(run_path, state.get("artifacts", []))
    trace: List[Dict[str, Any]] = []
    for spec in STEP_SPECS:
        inputs = {key: _compact(state.get(key)) for key in spec["input_keys"] if key in state}
        outputs = {key: _compact(state.get(key)) for key in spec["output_keys"] if key in state}
        artifacts = []
        for filename in spec["artifact_files"]:
            path = lookup.get(filename)
            if path is None and run_path:
                path = run_path / filename
            if path is not None:
                artifacts.append(_artifact_entry(path))
        issues = _collect_issues({key: state.get(key) for key in spec["output_keys"] if key in state})
        item = {
            **spec,
            "status": _step_status(outputs, artifacts),
            "inputs": inputs,
            "outputs": outputs,
            "artifacts": artifacts,
            "issues": issues,
            "data_flow": {
                "reads": spec["input_keys"],
                "writes": spec["output_keys"],
                "artifact_files": spec["artifact_files"],
            },
        }
        trace.append(item)
    return trace


def build_trace_from_run(run_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(run_dir)
    return build_trace(_load_run_state(path), path)
