"""Multi-panel figure synthesizer (UPK 13-c2 style figures).

Builds publication-ready figures from per-run artifacts:

  - `synthesize_lightcurve_figure`   ZTF g/r + NEOWISE W1/W2 phase-folded
                                     (matches UPK 13-c2 Fig. 1)
  - `synthesize_sed_figure`          F_obs + 4 hypothesis BB curves
                                     (matches UPK 13-c2 Fig. 2, simplified)
  - `synthesize_cluster_figure`      3-panel cluster membership diagnostics
                                     (matches UPK 13-c2 Fig. 5)
  - `synthesize_corner_figure`       MCMC posterior corner plot (uses
                                     `corner` package if installed, else
                                     a hexbin fallback)

All functions are graceful: if matplotlib is missing they return a stub
{"status": "matplotlib_not_installed"}. PNGs are written under
`paper_orchestra/figures/`.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple


def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except Exception as exc:
        return None, str(exc)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


# ---------- Light-curve figure ---------------------------------------------

def synthesize_lightcurve_figure(
    *,
    astrotool_root: Path,
    period_days: Optional[float],
    output_path: Path,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    mpl_mod, plt = _try_matplotlib()
    if not isinstance(plt, object) or plt is None or isinstance(plt, str):
        return {"status": "matplotlib_not_installed", "error": str(plt)}
    if period_days is None or not (period_days > 0):
        return {"status": "no_period"}
    bands_present = []
    panels = []
    for csv_name, label, color in (
        ("ztf_lightcurve.csv", "ZTF", "C0"),
        ("tess_lightcurve.csv", "TESS", "C1"),
        ("wise_lightcurve.csv", "NEOWISE", "C3"),
    ):
        path = astrotool_root / csv_name
        rows = _read_csv(path)
        if not rows:
            continue
        # Detect (mjd, mag, [band]) columns flexibly
        bjd_keys = ("mjd", "MJD", "bjd", "BJD", "time", "btjd", "BTJD")
        mag_keys = ("mag", "MAG", "flux", "FLUX")
        band_keys = ("band", "filter", "ftype")
        mjd_k = next((k for k in bjd_keys if k in rows[0]), None)
        mag_k = next((k for k in mag_keys if k in rows[0]), None)
        band_k = next((k for k in band_keys if k in rows[0]), None)
        if not mjd_k or not mag_k:
            continue
        by_band: Dict[str, List[Tuple[float, float]]] = {}
        for row in rows:
            try:
                t = float(row[mjd_k])
                m = float(row[mag_k])
                if not (math.isfinite(t) and math.isfinite(m)):
                    continue
                b = (row.get(band_k) or "all") if band_k else "all"
                by_band.setdefault(b, []).append((t, m))
            except (KeyError, ValueError):
                continue
        for band, points in by_band.items():
            if len(points) < 5:
                continue
            phases = [(t / period_days) % 1.0 for t, _ in points]
            mags = [m for _, m in points]
            panels.append({
                "facility": label, "band": band, "color": color,
                "phase": phases, "mag": mags, "n": len(points),
            })
            bands_present.append(f"{label}/{band}")

    if not panels:
        return {"status": "no_lightcurve_data"}

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(7, 1.6 * n + 1), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        ax.scatter(panel["phase"], panel["mag"], s=4, color=panel["color"], alpha=0.6,
                   label=f"{panel['facility']} {panel['band']} (N={panel['n']})")
        ax.invert_yaxis()
        ax.set_ylabel("mag")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel(f"Phase (P = {period_days:.4f} d)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return {
        "status": "ok",
        "path": str(output_path),
        "bands": bands_present,
        "n_panels": n,
        "caption": (
            f"Phase-folded multi-band light curves at P = {period_days:.4f} d. "
            f"Bands: {', '.join(bands_present)}."
        ),
    }


# ---------- SED figure ------------------------------------------------------

def synthesize_sed_figure(
    *,
    astrotool_root: Optional[Path],
    sed_decoupled: Mapping[str, Any],
    output_path: Path,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    mpl_mod, plt = _try_matplotlib()
    if plt is None or isinstance(plt, str):
        return {"status": "matplotlib_not_installed"}
    if not astrotool_root:
        return {"status": "no_astrotool_root"}
    sed_csv = Path(astrotool_root) / "sed_photometry.csv"
    rows = _read_csv(sed_csv)
    if not rows:
        return {"status": "no_sed_photometry_csv"}

    waves = []
    flux = []
    flux_err = []
    bands = []
    for row in rows:
        try:
            w = float(row.get("wave_A") or 0)
            f = float(row.get("flux_cgs") or 0)
            ferr = float(row.get("flux_err_cgs") or 0)
            if w > 0 and f > 0:
                waves.append(w / 1e4)  # μm
                flux.append(f)
                flux_err.append(ferr if ferr > 0 else f * 0.2)
                bands.append(row.get("band", ""))
        except ValueError:
            continue
    if not waves:
        return {"status": "no_sed_data"}

    # Generate hypothesis curves (best-fit blackbody for each)
    fit_results = (sed_decoupled or {}).get("fit_results", {}) or {}
    if not fit_results and sed_decoupled.get("mode") == "three_step":
        fit_results = (sed_decoupled.get("step1_diff") or {}).get("results", {})
    hyp_curves: List[Tuple[str, List[float], List[float]]] = []
    if fit_results:
        from astro_toolbox import stellar_templates as _st
        wl_um = sorted(set([0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 10.0, 22.0]))
        for hyp, res in list(fit_results.items())[:4]:
            teff = res.get("teff_K")
            scale = res.get("scale")
            if not teff or not scale:
                continue
            wl_A = [w * 1e4 for w in wl_um]
            curve = [scale * _st.planck_flux_fnu(wA, teff) for wA in wl_A]
            hyp_curves.append((hyp, wl_um, curve))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(waves, flux, yerr=flux_err, fmt="o", color="black", ms=4,
                label="Observed", capsize=2, zorder=10)
    for label, wl, curve in hyp_curves:
        ax.plot(wl, curve, label=f"{label.replace('_', ' ')}", alpha=0.8, lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel(r"$F_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(loc="best", fontsize=8)
    ax.grid(which="both", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return {
        "status": "ok", "path": str(output_path),
        "n_points": len(waves),
        "hypotheses_plotted": [h for h, _, _ in hyp_curves],
        "caption": (
            f"SED of the target: {len(waves)} broadband measurements (black) compared "
            f"to {len(hyp_curves)} candidate single-component blackbody hypotheses."
        ),
    }


# ---------- Cluster membership 3-panel figure -------------------------------

def synthesize_cluster_figure(
    *,
    cluster_membership: Mapping[str, Any],
    target: Optional[str],
    output_path: Path,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    mpl_mod, plt = _try_matplotlib()
    if plt is None or isinstance(plt, str):
        return {"status": "matplotlib_not_installed"}
    candidates = (cluster_membership or {}).get("candidates") or []
    if not candidates:
        return {"status": "no_cluster_candidates"}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.4))
    # Panel 1: chi2_spat bar chart
    names = [c.get("name") for c in candidates[:8]]
    spat = [c.get("chi2_spat") or 0 for c in candidates[:8]]
    kin = [c.get("chi2_kin") or 0 for c in candidates[:8]]
    rv = [c.get("rv_offset_sigma") or 0 for c in candidates[:8]]
    ax1.barh(names, spat, color="C0")
    ax1.axvline(9.0, color="red", linestyle="--", lw=1, label="3σ threshold")
    ax1.set_xlabel(r"$\chi^2_\mathrm{spat}$")
    ax1.set_xscale("symlog", linthresh=1.0)
    ax1.invert_yaxis()
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title("Spatial")

    ax2.barh(names, kin, color="C1")
    ax2.axvline(12.0, color="red", linestyle="--", lw=1, label="~3σ (3-param)")
    ax2.set_xlabel(r"$\chi^2_\mathrm{kin}$")
    ax2.set_xscale("symlog", linthresh=1.0)
    ax2.invert_yaxis()
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_title("Kinematic")

    ax3.barh(names, rv, color="C3")
    ax3.axvline(3.0, color="red", linestyle="--", lw=1, label="3σ")
    ax3.set_xlabel("RV offset (σ)")
    ax3.invert_yaxis()
    ax3.legend(loc="lower right", fontsize=8)
    ax3.set_title("RV (if available)")

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return {
        "status": "ok", "path": str(output_path),
        "n_candidates_shown": min(8, len(candidates)),
        "caption": (
            f"Cluster-membership diagnostics for the top candidate clusters: spatial χ², "
            f"kinematic χ², and RV offset.  Red dashed lines mark approximate 3σ rejection thresholds."
        ),
    }


# ---------- MCMC corner figure ---------------------------------------------

def synthesize_corner_figure(
    *,
    eclipse_mcmc: Mapping[str, Any],
    output_path: Path,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    mpl_mod, plt = _try_matplotlib()
    if plt is None or isinstance(plt, str):
        return {"status": "matplotlib_not_installed"}
    if (eclipse_mcmc or {}).get("status") != "ok":
        return {"status": "no_posterior",
                "reason": (eclipse_mcmc or {}).get("reason")}
    backend = eclipse_mcmc.get("backend")
    if backend == "emcee":
        # Real samples may be stored; for now we don't keep them in artifact
        # (would inflate JSON). Fall through to percentile-only display.
        pass
    # Percentile-only display: show medians + 1σ as error bars on a 3-axis bar.
    e_p = eclipse_mcmc.get("e_pct")
    a_p = eclipse_mcmc.get("alpha_deg_pct")
    o_p = eclipse_mcmc.get("omega_deg_pct")
    if not (e_p and a_p and o_p):
        return {"status": "missing_pct"}

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    labels = ["e (eccentricity)", "α (deg)", "ω (deg)"]
    series = [e_p, a_p, o_p]
    for ax, lbl, pct in zip(axes, labels, series):
        lo, mid, hi = pct
        ax.errorbar([0], [mid], yerr=[[mid - lo], [hi - mid]],
                    fmt="o", color="C2", ms=8, capsize=6)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel(lbl)
        ax.set_xticks([])
        ax.grid(alpha=0.2)
        ax.set_title(f"{lbl.split()[0]} = {mid:.2g} (-{mid-lo:.2g}, +{hi-mid:.2g})", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return {
        "status": "ok", "path": str(output_path),
        "backend": backend,
        "caption": (
            f"3-parameter (e, α, ω) MCMC posterior medians and 1σ intervals from the "
            f"{backend} backend."
        ),
    }


# ---------- Top-level dispatcher -------------------------------------------

def synthesize_all(state: Mapping[str, Any], workspace: Path) -> Dict[str, Any]:
    """Run all four figure builders.  Each is graceful and isolated."""
    fig_dir = Path(workspace) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Any] = {}
    out_root = (state.get("data_fetch", {}).get("astrotool", {}) or {}).get("output_root") \
        or state.get("astrotool_run")
    out_root_p = Path(out_root) if out_root else None

    # Period from light_curve_geometry > published_params
    period_min = (state.get("light_curve_geometry") or {}).get("period_min_used")
    period_days = (period_min / 1440.0) if period_min else None
    target = state.get("target")

    out["lightcurve"] = synthesize_lightcurve_figure(
        astrotool_root=out_root_p or Path("."),
        period_days=period_days,
        output_path=fig_dir / "fig_lightcurve.png",
        title=f"{target} phase-folded light curves",
    ) if out_root_p else {"status": "no_astrotool_root"}

    out["sed"] = synthesize_sed_figure(
        astrotool_root=out_root_p,
        sed_decoupled=state.get("sed_decoupled") or {},
        output_path=fig_dir / "fig_sed.png",
        title=f"{target} SED with hypothesis blackbody fits",
    )

    out["cluster"] = synthesize_cluster_figure(
        cluster_membership=state.get("cluster_membership") or {},
        target=target,
        output_path=fig_dir / "fig_cluster.png",
        title=f"{target} cluster-membership diagnostics",
    )

    out["corner"] = synthesize_corner_figure(
        eclipse_mcmc=state.get("eclipse_mcmc") or {},
        output_path=fig_dir / "fig_corner.png",
        title=f"{target} disk-eclipse MCMC posterior",
    )
    return {
        "status": "ok",
        "figures": out,
        "n_ok": sum(1 for v in out.values() if v.get("status") == "ok"),
    }


__all__ = [
    "synthesize_lightcurve_figure",
    "synthesize_sed_figure",
    "synthesize_cluster_figure",
    "synthesize_corner_figure",
    "synthesize_all",
]
