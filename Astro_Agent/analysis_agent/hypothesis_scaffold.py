"""Hypothesis-test scaffolding for the analysis agent.

The reference UPK13-c2 paper (Lin et al., 2025 ApJ) succeeded by explicitly
testing two physical hypotheses (WD+MS vs MS+MS+disk) with quantitative
χ² comparisons in three sub-steps each.  This module gives the rest of the
workflow a structured place to declare which hypotheses to test for a given
source class, what observables discriminate them, and which fitting modules
to call.

The output is consumed by:
  - the drafter, which now writes a Methods/Discussion section that mentions
    explicitly which hypotheses were considered and (if implemented) which won;
  - the paper_qc check `hypothesis_test_articulated`, which fails a paper
    that does not name at least one alternative interpretation.

The actual fitting modules referenced here may not exist yet — the scaffold
is the *plan*, and module presence (`module_implemented` flag) is checked
against the filesystem so we never claim to have done work we haven't.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class Hypothesis:
    name: str
    label: str
    description: str
    requires_observables: List[str]
    fitting_module: str
    chi2_label: str
    discriminating_against: List[str] = field(default_factory=list)
    references_bibcodes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hypothesis catalog per SIMBAD source class.  Each class maps to one or more
# competing physical hypotheses.  When the system grows new fitting modules,
# they should add themselves here so the dispatcher and drafter both pick
# them up.
# ---------------------------------------------------------------------------

HYPOTHESES: Dict[str, List[Hypothesis]] = {
    "unknown": [
        Hypothesis(
            name="generic_stellar_source",
            label="Generic stellar source (catalog unknown to SIMBAD)",
            description=(
                "The SIMBAD object_type could not be matched to a known class. "
                "Run general-purpose photometric and time-domain diagnostics "
                "to characterise the source before committing to a physics route."
            ),
            requires_observables=["SED", "time_domain_photometry"],
            fitting_module="astro_toolbox.sed",
            chi2_label="chi2_generic_sed",
        ),
        Hypothesis(
            name="cluster_member_candidate",
            label="Open-cluster member candidate (if target_cluster supplied)",
            description=(
                "If the workflow received a target_cluster hint and the "
                "kinematic χ² < 12 and spatial χ² < 9, the source may be a "
                "member of that cluster.  Otherwise it is a field object."
            ),
            requires_observables=["Gaia_astrometry"],
            fitting_module="astro_toolbox.cluster_membership",
            chi2_label="chi2_cluster",
        ),
        Hypothesis(
            name="disk_eclipsing_binary",
            label="Disk-eclipsing binary (UPK 13-c2 / KH 15D / Bernhard-2 class)",
            description=(
                "A flat-bottomed eclipse with multi-day ingress + an "
                "approximately achromatic optical-to-NIR flux decrement and a "
                "mid-IR excess at W3/W4 implies a misaligned circumbinary "
                "disk occulting one component.  Discriminate against WD+MS "
                "via a difference-spectrum (F_high - F_low) fit."
            ),
            requires_observables=["time_domain_photometry", "SED", "MIR_excess"],
            fitting_module="astro_toolbox.disk_eclipse_msms_fitting",
            chi2_label="chi2_disk_eclipse",
            discriminating_against=["generic_stellar_source"],
            references_bibcodes=["2025ApJ...UPK13c2L", "2022ApJ...931...13M"],
        ),
    ],
    "single_white_dwarf": [
        Hypothesis(
            name="single_DA_WD",
            label="Single DA white dwarf",
            description=(
                "Hot single white dwarf with hydrogen-dominated atmosphere; "
                "no companion required to explain SED + spectroscopy."
            ),
            requires_observables=["optical_spectrum", "Gaia_parallax", "SED"],
            fitting_module="astro_toolbox.wd_fitting",
            chi2_label="chi2_DA",
        ),
        Hypothesis(
            name="DA_plus_M_dwarf",
            label="DA white dwarf + M-dwarf companion",
            description=(
                "Composite DA + M-dwarf SED; required when red excess or "
                "Hα emission is present."
            ),
            requires_observables=["optical_spectrum", "Gaia_parallax", "NIR_photometry"],
            fitting_module="astro_toolbox.wdms_fitting",
            chi2_label="chi2_DA_MS",
            discriminating_against=["single_DA_WD"],
        ),
        Hypothesis(
            name="MS_plus_MS_disk",
            label="Main-sequence binary + circumbinary disk (UPK13-c2-like)",
            description=(
                "When a flat-bottomed eclipse with multi-day ingress is seen, "
                "the WD+MS hypothesis can fail in favor of a late-K MS+MS binary "
                "occulted by a misaligned disk; see Lin et al. (2025 ApJ)."
            ),
            requires_observables=["time_domain_photometry", "SED", "MIR_excess"],
            fitting_module="astro_toolbox.disk_eclipse_msms_fitting",
            chi2_label="chi2_MSMS",
            discriminating_against=["single_DA_WD", "DA_plus_M_dwarf"],
        ),
    ],
    "sdob_binary_or_single": [
        Hypothesis(
            name="sdob_isolated",
            label="Isolated hot subdwarf",
            description=(
                "Single hot subdwarf with no detectable companion; SED fit with "
                "TLUSTY/BSTAR atmosphere; no short-period photometric variation."
            ),
            requires_observables=["optical_or_UV_spectrum", "SED", "time_domain_photometry"],
            fitting_module="astro_toolbox.sdob_fitting",
            chi2_label="chi2_sdob",
        ),
        Hypothesis(
            name="sdob_plus_WD_detached",
            label="sdOB + WD detached binary (post-CE, pre-mass-transfer)",
            description=(
                "Detached sdOB+WD with ellipsoidal or Doppler-beaming modulation; "
                "P_orb in 20 min–10 h; LISA verification-binary candidate."
            ),
            requires_observables=["time_domain_photometry", "SED", "radial_velocity"],
            fitting_module="astro_toolbox.sdob_wd_detached_fitting",
            chi2_label="chi2_sdob_wd_detached",
            discriminating_against=["sdob_isolated", "sdob_plus_WD_rlof"],
        ),
        Hypothesis(
            name="sdob_plus_WD_rlof",
            label="Roche-lobe-filling sdOB + WD (Kupfer 2020 prototype class)",
            description=(
                "sdOB donor fills its Roche lobe; mass-transferring; eclipses "
                "+ X-ray candidate; ZTF J213056.71+442046.5 is the prototype."
            ),
            requires_observables=["time_domain_photometry", "X-ray", "SED"],
            fitting_module="astro_toolbox.sdob_wd_rlof_fitting",
            chi2_label="chi2_sdob_wd_rlof",
            references_bibcodes=["2020ApJ...891...45K", "2022ApJ...931...13M"],
            discriminating_against=["sdob_isolated", "sdob_plus_WD_detached"],
        ),
    ],
    "cataclysmic_variable": [
        Hypothesis(
            name="non_magnetic_CV",
            label="Non-magnetic CV (dwarf nova or nova-like)",
            description=(
                "Accretion via disc to WD primary; broad Balmer emission; "
                "long-term outbursts on day-to-month timescales."
            ),
            requires_observables=["optical_spectrum", "time_domain_photometry"],
            fitting_module="astro_toolbox.cv_fitting",
            chi2_label="chi2_CV_disc",
        ),
        Hypothesis(
            name="polar",
            label="Magnetic polar (AM Her)",
            description=(
                "Strong cyclotron humps + Zeeman splitting; no accretion disc."
            ),
            requires_observables=["optical_spectrum", "polarimetry"],
            fitting_module="astro_toolbox.polar_fitting",
            chi2_label="chi2_polar",
            discriminating_against=["non_magnetic_CV"],
        ),
    ],
    "ultracompact_double_degenerate": [
        Hypothesis(
            name="DWD_inspiral",
            label="Double WD inspiral (LISA source)",
            description=(
                "Two degenerate components in a sub-hour orbit; GW-driven inspiral."
            ),
            requires_observables=["time_domain_photometry", "radial_velocity"],
            fitting_module="astro_toolbox.dwd_fitting",
            chi2_label="chi2_DWD",
        ),
        Hypothesis(
            name="AM_CVn",
            label="AM CVn mass-transferring DWD",
            description=(
                "Helium-rich mass-transferring DWD; He-emission spectrum; "
                "P_orb 5–60 min."
            ),
            requires_observables=["optical_spectrum", "time_domain_photometry"],
            fitting_module="astro_toolbox.amcvn_fitting",
            chi2_label="chi2_AMCVn",
            discriminating_against=["DWD_inspiral"],
        ),
    ],
}


def _module_implemented(module_dotted_path: str) -> bool:
    if not module_dotted_path:
        return False
    repo_root = Path(__file__).resolve().parent.parent
    relative = Path(*module_dotted_path.split("."))
    return (repo_root / relative).with_suffix(".py").exists()


def hypothesis_plan_for(
    source_class: str,
    *,
    available_evidence: Mapping[str, bool],
) -> Dict[str, Any]:
    """Return the structured hypothesis-test plan for a source class.

    available_evidence is the dict produced by structure_planner_node, keyed
    by observable name (spectrum / sed / hr_diagram / rv / period_products
    / hst_spectrum).  Each hypothesis is annotated with whether its required
    observables are satisfied AND whether its fitting module exists.
    """
    out: Dict[str, Any] = {
        "source_class": source_class,
        "hypotheses": [],
    }
    # Map structure_planner evidence keys -> hypothesis observable tokens.
    EVIDENCE_ALIAS = {
        "spectrum": "optical_spectrum",
        "hst_spectrum": "UV_spectrum",
        "sed": "SED",
        "hr_diagram": "HRD",
        "rv": "radial_velocity",
        "period_products": "time_domain_photometry",
    }
    evidence_norm = {EVIDENCE_ALIAS.get(k, k): bool(v) for k, v in (available_evidence or {}).items()}
    # Approximate UV+optical → optical_or_UV_spectrum
    evidence_norm["optical_or_UV_spectrum"] = (
        evidence_norm.get("optical_spectrum", False) or evidence_norm.get("UV_spectrum", False)
    )
    # MIR_excess and X-ray availability are not directly in structure_planner evidence;
    # callers can extend evidence_norm before calling this function.
    for h in HYPOTHESES.get(source_class, []):
        required = h.requires_observables
        observable_status = {tok: evidence_norm.get(tok, False) for tok in required}
        missing = [tok for tok, v in observable_status.items() if not v]
        out["hypotheses"].append({
            **asdict(h),
            "observable_status": observable_status,
            "missing_observables": missing,
            "ready_to_run": not missing,
            "module_implemented": _module_implemented(h.fitting_module),
        })
    out["n_total"] = len(out["hypotheses"])
    out["n_ready"] = sum(1 for h in out["hypotheses"] if h["ready_to_run"])
    out["n_implemented"] = sum(1 for h in out["hypotheses"] if h["module_implemented"])
    return out


def render_markdown(plan: Mapping[str, Any]) -> str:
    """Pretty-print the hypothesis plan for the drafter's experimental log."""
    lines = [
        "## Hypothesis Test Plan",
        f"- source class: `{plan.get('source_class')}`",
        f"- hypotheses: {plan.get('n_total')} total, "
        f"{plan.get('n_ready')} have all required observables, "
        f"{plan.get('n_implemented')} have an implemented fitting module",
        "",
        "| Hypothesis | required observables | module | implemented? | ready? |",
        "|---|---|---|---|---|",
    ]
    for h in plan.get("hypotheses", []):
        missing = h.get("missing_observables") or []
        obs = ", ".join(h["requires_observables"])
        ready = "yes" if h.get("ready_to_run") else f"no (missing: {','.join(missing)})"
        impl = "yes" if h.get("module_implemented") else "no"
        lines.append(
            f"| **{h.get('label')}** (`{h.get('name')}`) | {obs} | `{h.get('fitting_module')}` | {impl} | {ready} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["hypothesis_plan_for", "render_markdown"]
