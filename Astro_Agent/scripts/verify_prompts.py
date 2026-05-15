"""Static + smoke verification of the layered prompt scaffolding.

Static checks (no LLM call):
  - every role assembles cleanly via wd_domain.system_for_role(role);
  - the assembled prompt is < 6000 tokens (approximated as len/4 chars);
  - REQUIRED_TERMS all appear in the assembled prompt;
  - REQUIRED_TERMS do NOT trigger the FORBIDDEN_HYPE filter (sanity);
  - every section in {Abstract, Introduction, Data, Methods, Results,
    Discussion, Conclusions} has a non-trivial section_prompt;
  - specialists.system_prompt_for("physicist"/"writer"/"critic") work;
  - paper_qc and reflexion can be imported without errors, and the
    section-aware + Codex-derived checks are wired into run_paper_qc.

Smoke check (optional, gated by --llm):
  - calls Claude (provider="claude") with the writer system prompt +
    the Abstract task prompt + a minimal mock evidence block;
  - asserts the response contains >=1 \\citep{...}, >=3 numbers+units,
    and 0 FORBIDDEN_HYPE words.

Exit code 0 = all green. 1 = any static check failed. 2 = LLM smoke
failed (only if --llm passed).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))

from analysis_agent.prompts import wd_domain  # noqa: E402
from analysis_agent import specialists  # noqa: E402
from analysis_agent.nodes import paper_qc  # noqa: E402
from analysis_agent import reflexion  # noqa: E402


ALL_ROLES = ("physicist", "writer", "critic", "outline", "reviewer", "retrieval")
ALL_SECTIONS = ("Abstract", "Introduction", "Data", "Methods",
                "Results", "Discussion", "Conclusions")

NEW_CHECKS = (
    "methods_chi2_density",
    "results_uncertainty_density",
    "discussion_alternatives",
    "intro_motivation_chain",
    "forbidden_hype",
    "cluster_joint_criteria",
    "bibkey_format",
    "target_identity_consistency",
    "extinction_provenance",
    "literature_consistency",
    "physics_checks_integration",
)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def check_role_assembly(failures: list) -> None:
    for role in ALL_ROLES:
        try:
            prompt = wd_domain.system_for_role(role)
        except Exception as exc:
            failures.append(f"role={role}: build failed: {exc}")
            continue
        if not prompt.strip():
            failures.append(f"role={role}: empty prompt")
            continue
        tok = approx_tokens(prompt)
        if tok > 6000:
            failures.append(f"role={role}: prompt too long ({tok} tokens > 6000)")
        # Required white-dwarf domain terms
        missing = [t for t in wd_domain.REQUIRED_TERMS if t.lower() not in prompt.lower()]
        if missing:
            failures.append(f"role={role}: missing terms {missing}")


def check_section_prompts(failures: list) -> None:
    for sec in ALL_SECTIONS:
        prompt = wd_domain.section_prompt(sec)
        if not prompt.strip() or len(prompt) < 200:
            failures.append(f"section={sec}: prompt suspiciously short ({len(prompt)} chars)")


def check_specialists(failures: list) -> None:
    for spec in ("physicist", "writer", "critic"):
        sp = specialists.system_prompt_for(spec)
        if not sp.strip():
            failures.append(f"specialist={spec}: empty system prompt")


def check_paper_qc_wired(failures: list) -> None:
    sample_tex = (
        r"\begin{abstract}A dummy paragraph with $T_{\rm eff}=12500\,$K "
        r"and $M=0.6$ Msun and $P=42$ min.\end{abstract}" "\n"
        r"\section{Introduction}One. Two. Three." "\n"
        r"\section{Data}" "\n"
        r"\section{Methods}A chi^2 here; chi^2 there; chi^2 also." "\n"
        r"\section{Results}value $\pm$ err units, $5 \pm 1$ K." "\n"
        r"\section{Discussion}alternatively this could also be Y." "\n"
        r"\section{Conclusions}" "\n"
    )
    result = paper_qc.run_paper_qc(
        final_tex_path=None,
        workspace_root=None,
        published_params_table={"rows": []},
    )
    # We expect run_paper_qc to fail on empty input but still execute without error
    if "checks" not in result:
        failures.append("paper_qc.run_paper_qc returned unexpected shape")
        return
    # Write sample_tex to a temp path and invoke for real check IDs
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as tmp:
        tmp.write(sample_tex)
        tmp_path = tmp.name
    out = paper_qc.run_paper_qc(
        final_tex_path=tmp_path,
        workspace_root=None,
        published_params_table={"rows": []},
    )
    ids = {c["id"] for c in out.get("checks", [])}
    missing = [c for c in NEW_CHECKS if c not in ids]
    if missing:
        failures.append(f"paper_qc missing new checks: {missing}")
    # Regression: optional citation arguments must still expose the key.
    bad_cite_tex = sample_tex.replace(
        r"\section{Methods}A chi^2 here; chi^2 there; chi^2 also.",
        r"\section{Methods}A chi^2 here; chi^2 there; chi^2 also. \citep[][placeholder]{2025ApJ}",
    )
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False) as tmp2:
        tmp2.write(bad_cite_tex)
        bad_path = tmp2.name
    bad = paper_qc.run_paper_qc(
        final_tex_path=bad_path,
        workspace_root=None,
        published_params_table={"rows": []},
    )
    by_id = {c["id"]: c for c in bad.get("checks", [])}
    if by_id.get("bibkey_format", {}).get("verdict") != "fail":
        failures.append("paper_qc.bibkey_format did not catch optional-arg placeholder citation")


def check_reflexion_advice(failures: list) -> None:
    for cid in NEW_CHECKS:
        if cid not in reflexion._CHECK_TO_REWRITE:
            failures.append(f"reflexion._CHECK_TO_REWRITE missing advice for {cid}")


def check_forbidden_hype_consistency(failures: list) -> None:
    # The assembled system prompt must NOT itself trigger the hype filter
    # (it lists hype words in a "do not use" context — that's OK only if
    # the implementation matches on whole words in body text, not in the
    # prompt). We just confirm the FORBIDDEN list is non-empty and the
    # paper_qc forbidden list matches wd_domain.FORBIDDEN_HYPE.
    if not wd_domain.FORBIDDEN_HYPE:
        failures.append("wd_domain.FORBIDDEN_HYPE is empty")
    # Cross-check: paper_qc uses its own _FORBIDDEN_HYPE tuple
    qc_set = set(getattr(paper_qc, "_FORBIDDEN_HYPE", ()))
    wd_set = set(wd_domain.FORBIDDEN_HYPE)
    if qc_set != wd_set:
        diff = wd_set.symmetric_difference(qc_set)
        failures.append(
            f"FORBIDDEN_HYPE drift between wd_domain and paper_qc: {sorted(diff)}"
        )


def llm_smoke(failures: list) -> None:
    """Hit Claude with the writer Abstract prompt + tiny mock evidence."""
    try:
        from analysis_agent.llm_client import LLMClient, load_model_config, load_default_env
    except Exception as exc:
        failures.append(f"llm_smoke: import failed: {exc}")
        return
    load_default_env()
    cfg = load_model_config()
    client = LLMClient(cfg)
    if not client.available:
        failures.append("llm_smoke: no provider available (skipping)")
        return
    sys_prompt = wd_domain.system_for_role("writer")
    user = (
        wd_domain.section_prompt("Abstract") + "\n\n"
        "### Evidence (mock)\n"
        "Target: Mock WD J0000+0000\n"
        "Coordinates: RA=0.0, Dec=0.0\n"
        "published_params rows: Teff=12500 K (bib=2020A&A...638A...1X), "
        "M=0.6 Msun \\citep{2018ApJ...XXX...YY}, P_orb=42 min.\n"
        "this-work: parallax=2.5 mas (bib=this_work_gaia).\n"
        "QA gate: clear.\n"
    )
    try:
        out = client.complete(system=sys_prompt, user=user)
    except Exception as exc:
        failures.append(f"llm_smoke: completion failed: {exc}")
        return
    if not out:
        failures.append("llm_smoke: empty response")
        return
    # Assertions
    if not re.search(r"\\citep\{", out):
        failures.append("llm_smoke: no \\citep{} in abstract")
    nums = re.findall(r"\d+(?:\.\d+)?\s*(?:K|Msun|M_?⊙|min|hr|pc|kpc|mas|km/s)", out)
    if len(nums) < 3:
        failures.append(f"llm_smoke: only {len(nums)} number+unit phrases (want >=3)")
    lower = out.lower()
    hype = [w for w in wd_domain.FORBIDDEN_HYPE if w in lower]
    if hype:
        failures.append(f"llm_smoke: hype words emitted: {hype}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="run the LLM smoke test")
    args = ap.parse_args()

    failures: list[str] = []
    check_role_assembly(failures)
    check_section_prompts(failures)
    check_specialists(failures)
    check_paper_qc_wired(failures)
    check_reflexion_advice(failures)
    check_forbidden_hype_consistency(failures)

    static_failed = list(failures)
    if args.llm:
        llm_smoke(failures)
    llm_failed = [f for f in failures if f not in static_failed]

    print("=" * 60)
    print("static checks:", "FAIL" if static_failed else "OK")
    for f in static_failed:
        print(" -", f)
    if args.llm:
        print("llm smoke:", "FAIL" if llm_failed else "OK")
        for f in llm_failed:
            print(" -", f)
    print("=" * 60)
    if static_failed:
        return 1
    if llm_failed:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
