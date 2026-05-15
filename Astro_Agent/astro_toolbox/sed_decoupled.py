"""SED 3-step decoupled fit (UPK 13-c2 style).

Implements the Lin et al. 2025 (UPK 13-c2 ApJ) procedure:

  Step 1 (F_diff): F_high - F_low contains only the eclipsed component's flux.
        Fit a single-template hypothesis (e.g. cool blackbody / Koester DA WD /
        late-K dwarf) to the difference SED.  Returns χ²_diff per hypothesis.

  Step 2 (F_low): the un-eclipsed state.  Fit the survivor template + optional
        disk blackbody to F_low.  Returns χ²_low per hypothesis.

  Step 3 (F_high): synthesize F_synth = F_low_model + F_diff_model and compute
        χ² against the observed F_high.  Returns χ²_high per hypothesis.

The key insight is that χ²_diff is *extinction-independent* (the same A_V
column cancels in the F_high - F_low subtraction), so hypotheses that fit
F_diff badly cannot be rescued by re-tuning A_V — a robust discriminator
between e.g. WD + MS vs. MS + MS + disk.

The fitter is intentionally simple (grid search over Teff with R scaled to
match flux) so it runs in milliseconds.  Higher fidelity (TLUSTY/BSTAR
grids, dust radiative transfer) belongs in a Stage 2 module.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import stellar_templates as _st
from . import extinction as _ext


# ---------- helpers ---------------------------------------------------------

def _resolve_band_wavelength(band: str, observed_row: Mapping[str, Any]) -> Optional[float]:
    """Try to get a usable wavelength in Å for a band token."""
    wl = observed_row.get("wave_A") or observed_row.get("wavelength_A")
    if wl is not None:
        try:
            return float(wl)
        except (TypeError, ValueError):
            pass
    return _st.BAND_WAVELENGTHS_A.get(band)


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ---------- single-template χ² ---------------------------------------------

def _chi2_blackbody_grid(
    *,
    flux_pairs: List[Tuple[float, float, float]],  # (wave_A, F_nu_obs, sigma_F_nu)
    teff_grid: Iterable[float],
    a_v: float = 0.0,
    sys_floor_dex: float = 0.087,  # 20% systematic floor (log10)
) -> Dict[str, Any]:
    """Grid-search a blackbody Teff; analytic R²/d² normalisation."""
    best = {"chi2": float("inf"), "teff_K": None, "scale": None, "grid_boundary": False}
    teff_grid_list = list(teff_grid)
    if not teff_grid_list:
        return best
    teff_grid_min = min(teff_grid_list)
    teff_grid_max = max(teff_grid_list)
    n_dof = max(1, len(flux_pairs) - 2)
    for teff in teff_grid_list:
        # B_ν(T) at each band wavelength.  Subtract extinction in mag, then
        # multiply by 10^(0.4 A_λ) to deredden the observed flux.
        model = []
        observed = []
        sigma = []
        for wl_A, F_obs, sig in flux_pairs:
            if wl_A is None or F_obs is None or F_obs <= 0:
                continue
            wl_um = wl_A * 1e-4
            al = _ext.a_lambda(wl_um, a_v) if a_v > 0 else 0.0
            F_obs_dered = F_obs * 10 ** (0.4 * al)
            B = _st.planck_flux_fnu(wl_A, teff)
            if B <= 0:
                continue
            model.append(B)
            observed.append(F_obs_dered)
            # Combine statistical σ in quadrature with a 20% systematic floor.
            log_floor = sys_floor_dex
            sigma_sys = math.fabs(F_obs_dered) * (math.exp(log_floor * math.log(10)) - 1.0)
            sig_tot = math.hypot(sig if sig else 0.0, sigma_sys)
            sigma.append(sig_tot if sig_tot > 0 else 1.0)
        if not model:
            continue
        # Solve for the best multiplicative scale (R/d)²·π factor by χ²
        num = sum(o * m / (s * s) for o, m, s in zip(observed, model, sigma))
        den = sum((m / s) ** 2 for m, s in zip(model, sigma))
        if den <= 0:
            continue
        scale = num / den
        if scale <= 0:
            continue
        chi2 = sum(((o - scale * m) / s) ** 2 for o, m, s in zip(observed, model, sigma))
        if chi2 < best["chi2"]:
            best = {
                "chi2": chi2, "teff_K": teff, "scale": scale,
                "n_points": len(observed), "dof": max(1, len(observed) - 2),
                "grid_boundary": (teff == teff_grid_min or teff == teff_grid_max),
            }
    # Flag grid-boundary fits: a best-fit pegged at the edge of the Teff grid
    # means the true minimum is outside the explored range, so we penalise it
    # by inflating chi² by +9 (~3σ) so it doesn't crowd out properly-converged
    # competing hypotheses in the joint ranking.
    if best.get("grid_boundary"):
        best["chi2_raw"] = best["chi2"]
        best["chi2"] = best["chi2"] + 9.0
    return best


# ---------- canonical templates / hypotheses --------------------------------

# (label, Teff grid, comment)
HYPOTHESIS_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "WD_DA_blackbody": {
        "teff_grid": list(range(5000, 80001, 1000)),
        "description": "Pure blackbody approximation to a DA white dwarf; "
                       "Rayleigh-Jeans tail dominates at λ > 0.6 μm.  "
                       "Use as discriminating null for WD+MS vs MS+MS hypotheses.",
    },
    "hot_subdwarf_blackbody": {
        "teff_grid": list(range(20000, 50001, 2000)),
        "description": "sdOB blackbody; placeholder for TLUSTY/BSTAR grid.",
    },
    "late_K_dwarf": {
        "teff_grid": list(range(3500, 5001, 50)),
        "description": "Late-K dwarf BB approximation (Pecaut-Mamajek Teff range).",
    },
    "late_M_dwarf": {
        "teff_grid": list(range(2500, 4001, 50)),
        "description": "Mid-to-late M dwarf BB.",
    },
    "disk_blackbody_cool": {
        "teff_grid": [100, 150, 200, 250, 300, 400, 500, 700, 1000, 1500],
        "description": "Cool circumstellar/circumbinary dust disk BB; used in F_low.",
    },
}


# ---------- public API ------------------------------------------------------

def fit_diff_spectrum(
    *,
    flux_high: List[Mapping[str, Any]],
    flux_low: List[Mapping[str, Any]],
    hypotheses: Iterable[str] = ("late_K_dwarf", "late_M_dwarf", "WD_DA_blackbody", "hot_subdwarf_blackbody"),
    a_v: float = 0.0,
) -> Dict[str, Any]:
    """Step 1: fit F_high − F_low (the eclipsed component) with each hypothesis.

    Each entry in flux_high/flux_low must have keys:
        band, wave_A, F_nu_obs_cgs, sigma_F_nu
    The two lists are matched by `band`; missing matches are dropped.
    """
    low_by_band = {row["band"]: row for row in flux_low}
    diff_pairs: List[Tuple[float, float, float]] = []
    for hi in flux_high:
        band = hi.get("band")
        lo = low_by_band.get(band)
        if lo is None:
            continue
        wl = _safe_float(hi.get("wave_A")) or _safe_float(lo.get("wave_A")) or _st.BAND_WAVELENGTHS_A.get(band)
        F_hi = _safe_float(hi.get("F_nu_obs_cgs"))
        F_lo = _safe_float(lo.get("F_nu_obs_cgs"))
        sig_hi = _safe_float(hi.get("sigma_F_nu")) or 0.0
        sig_lo = _safe_float(lo.get("sigma_F_nu")) or 0.0
        if wl is None or F_hi is None or F_lo is None:
            continue
        F_diff = F_hi - F_lo
        if F_diff <= 0:
            continue
        sig = math.hypot(sig_hi, sig_lo)
        diff_pairs.append((wl, F_diff, sig))

    if not diff_pairs:
        return {"status": "no_paired_bands", "n_bands": 0, "results": {}}

    results: Dict[str, Any] = {}
    for hyp in hypotheses:
        cfg = HYPOTHESIS_TEMPLATES.get(hyp)
        if not cfg:
            continue
        best = _chi2_blackbody_grid(flux_pairs=diff_pairs, teff_grid=cfg["teff_grid"], a_v=a_v)
        results[hyp] = best
    # rank
    ranked = sorted(
        ((h, r) for h, r in results.items() if math.isfinite(r.get("chi2", math.inf))),
        key=lambda kv: kv[1]["chi2"],
    )
    return {
        "status": "ok",
        "n_bands": len(diff_pairs),
        "a_v": a_v,
        "results": results,
        "best_hypothesis": ranked[0][0] if ranked else None,
        "best_chi2": ranked[0][1]["chi2"] if ranked else None,
        "delta_chi2_vs_best": {
            h: r["chi2"] - ranked[0][1]["chi2"]
            for h, r in ranked
        } if ranked else {},
    }


def fit_low_state(
    *,
    flux_low: List[Mapping[str, Any]],
    survivor_teff_grid: Iterable[float],
    disk_teff_grid: Iterable[float] = (150, 200, 250, 300, 500, 700, 1000),
    a_v: float = 0.0,
) -> Dict[str, Any]:
    """Step 2: fit F_low with survivor stellar template (single Teff grid) plus
    optional cool blackbody disk component.

    Returns the joint χ²_low for the two-component fit.
    """
    pairs: List[Tuple[float, float, float]] = []
    for row in flux_low:
        wl = _safe_float(row.get("wave_A")) or _st.BAND_WAVELENGTHS_A.get(row.get("band", ""))
        F = _safe_float(row.get("F_nu_obs_cgs"))
        sig = _safe_float(row.get("sigma_F_nu")) or 0.0
        if wl is None or F is None or F <= 0:
            continue
        pairs.append((wl, F, sig))
    if not pairs:
        return {"status": "empty_flux_low", "results": []}

    best_combo = {"chi2": float("inf"), "survivor_teff_K": None, "disk_teff_K": None,
                  "survivor_scale": None, "disk_scale": None}
    for T_star in survivor_teff_grid:
        for T_disk in [*list(disk_teff_grid), 0.0]:  # 0 = no disk
            chi2 = 0.0
            n = 0
            for wl, F_obs, sig in pairs:
                wl_um = wl * 1e-4
                al = _ext.a_lambda(wl_um, a_v) if a_v > 0 else 0.0
                F_dered = F_obs * 10 ** (0.4 * al)
                B_star = _st.planck_flux_fnu(wl, T_star)
                B_disk = _st.planck_flux_fnu(wl, T_disk) if T_disk > 0 else 0.0
                # Solve for the two scales jointly per-row would be ill-posed;
                # use a simple two-step fit: first scale survivor to F_dered at
                # the optical band (assumed bluest available), then fit the
                # residual to the disk.
                # Practical shortcut: take the bluest wavelength as anchor.
                # For grid screening we use a single survivor scale equal to
                # F_dered(λmin)/B_star(λmin), and disk scale to absorb the
                # residual at the reddest wavelength.
                # ... but to keep the implementation honest and bounded, we
                # just fit each row residual independently using the two B's.
                model = B_star + 0.0 * B_disk  # we'll refine below
                if model <= 0:
                    continue
                # We'll patch this loop into a proper joint solve below.
                n += 1
            # joint solver
            # Build matrix A = [B_star(λ), B_disk(λ)] · row weights / σ.
            try:
                A11 = A12 = A22 = b1 = b2 = 0.0
                for wl, F_obs, sig in pairs:
                    wl_um = wl * 1e-4
                    al = _ext.a_lambda(wl_um, a_v) if a_v > 0 else 0.0
                    F_dered = F_obs * 10 ** (0.4 * al)
                    B_s = _st.planck_flux_fnu(wl, T_star)
                    B_d = _st.planck_flux_fnu(wl, T_disk) if T_disk > 0 else 0.0
                    sigma_sys = abs(F_dered) * 0.2  # 20% systematic
                    s = math.hypot(sig, sigma_sys) or 1.0
                    A11 += (B_s / s) ** 2
                    A12 += (B_s * B_d) / (s * s)
                    A22 += (B_d / s) ** 2
                    b1 += F_dered * B_s / (s * s)
                    b2 += F_dered * B_d / (s * s)
                det = A11 * A22 - A12 * A12
                if det <= 0 or A11 <= 0:
                    if A11 > 0:
                        s_star = max(0.0, b1 / A11)
                        s_disk = 0.0
                    else:
                        continue
                else:
                    s_star = (A22 * b1 - A12 * b2) / det
                    s_disk = (-A12 * b1 + A11 * b2) / det
                if s_star < 0 or s_disk < 0:
                    s_disk = max(0.0, s_disk)
                    s_star = max(0.0, s_star)
                chi2 = 0.0
                for wl, F_obs, sig in pairs:
                    wl_um = wl * 1e-4
                    al = _ext.a_lambda(wl_um, a_v) if a_v > 0 else 0.0
                    F_dered = F_obs * 10 ** (0.4 * al)
                    B_s = _st.planck_flux_fnu(wl, T_star)
                    B_d = _st.planck_flux_fnu(wl, T_disk) if T_disk > 0 else 0.0
                    model = s_star * B_s + s_disk * B_d
                    sigma_sys = abs(F_dered) * 0.2
                    s = math.hypot(sig, sigma_sys) or 1.0
                    chi2 += ((F_dered - model) / s) ** 2
                if chi2 < best_combo["chi2"]:
                    best_combo = {
                        "chi2": chi2,
                        "survivor_teff_K": T_star,
                        "disk_teff_K": T_disk if T_disk > 0 else None,
                        "survivor_scale": s_star,
                        "disk_scale": s_disk if T_disk > 0 else None,
                        "n_points": len(pairs),
                        "dof": max(1, len(pairs) - 3),
                    }
            except Exception:
                continue
    return {
        "status": "ok",
        "n_bands": len(pairs),
        "a_v": a_v,
        "best_combo": best_combo,
    }


def synthesize_high_state(
    *,
    flux_high_obs: List[Mapping[str, Any]],
    diff_fit: Mapping[str, Any],
    low_fit: Mapping[str, Any],
    diff_hypothesis: str,
    a_v: float = 0.0,
) -> Dict[str, Any]:
    """Step 3: F_synth = F_low_model + F_diff_model, compute χ² vs F_high."""
    diff_res = (diff_fit.get("results") or {}).get(diff_hypothesis)
    low_combo = low_fit.get("best_combo") or {}
    if not diff_res or not low_combo:
        return {"status": "missing_inputs"}
    T_diff = diff_res.get("teff_K")
    diff_scale = diff_res.get("scale")
    T_star_low = low_combo.get("survivor_teff_K")
    T_disk = low_combo.get("disk_teff_K")
    s_star = low_combo.get("survivor_scale") or 0.0
    s_disk = low_combo.get("disk_scale") or 0.0
    if T_diff is None or diff_scale is None or T_star_low is None:
        return {"status": "incomplete_fit_inputs"}

    chi2 = 0.0
    n = 0
    rows_used = []
    for row in flux_high_obs:
        wl = _safe_float(row.get("wave_A")) or _st.BAND_WAVELENGTHS_A.get(row.get("band", ""))
        F = _safe_float(row.get("F_nu_obs_cgs"))
        sig = _safe_float(row.get("sigma_F_nu")) or 0.0
        if wl is None or F is None or F <= 0:
            continue
        wl_um = wl * 1e-4
        al = _ext.a_lambda(wl_um, a_v) if a_v > 0 else 0.0
        F_dered = F * 10 ** (0.4 * al)
        B_diff = _st.planck_flux_fnu(wl, T_diff) * diff_scale
        B_star = _st.planck_flux_fnu(wl, T_star_low) * s_star
        B_disk = _st.planck_flux_fnu(wl, T_disk) * s_disk if T_disk and T_disk > 0 else 0.0
        F_synth = B_diff + B_star + B_disk
        sigma_sys = abs(F_dered) * 0.2
        s = math.hypot(sig, sigma_sys) or 1.0
        chi2 += ((F_dered - F_synth) / s) ** 2
        n += 1
        rows_used.append({"band": row.get("band"), "F_obs": F_dered, "F_synth": F_synth,
                          "B_diff": B_diff, "B_star": B_star, "B_disk": B_disk})
    return {
        "status": "ok",
        "diff_hypothesis": diff_hypothesis,
        "chi2_high": chi2,
        "dof": max(1, n - 3),
        "n_bands": n,
        "rows": rows_used,
    }


def run_three_step(
    *,
    flux_high: List[Mapping[str, Any]],
    flux_low: Optional[List[Mapping[str, Any]]] = None,
    a_v: float = 0.0,
    primary_hypotheses: Iterable[str] = ("late_K_dwarf", "late_M_dwarf", "WD_DA_blackbody", "hot_subdwarf_blackbody"),
    survivor_teff_grid: Iterable[float] = tuple(range(3500, 5001, 50)),
) -> Dict[str, Any]:
    """Run all three steps and return a single combined report.

    If flux_low is None or empty, we run a degenerate single-state SED fit
    (F_high only) using the same hypothesis grid.  This is the only mode
    available for sources without time-domain photometric state separation.
    """
    out: Dict[str, Any] = {"a_v": a_v}
    if not flux_low:
        # Single-state fallback: just fit F_high with each hypothesis.
        diff_pairs = []
        for row in flux_high:
            wl = _safe_float(row.get("wave_A")) or _st.BAND_WAVELENGTHS_A.get(row.get("band", ""))
            F = _safe_float(row.get("F_nu_obs_cgs"))
            sig = _safe_float(row.get("sigma_F_nu")) or 0.0
            if wl is None or F is None or F <= 0:
                continue
            diff_pairs.append((wl, F, sig))
        results = {}
        for hyp in primary_hypotheses:
            cfg = HYPOTHESIS_TEMPLATES.get(hyp)
            if not cfg:
                continue
            best = _chi2_blackbody_grid(flux_pairs=diff_pairs, teff_grid=cfg["teff_grid"], a_v=a_v)
            results[hyp] = best
        ranked = sorted(
            ((h, r) for h, r in results.items() if math.isfinite(r.get("chi2", math.inf))),
            key=lambda kv: kv[1]["chi2"],
        )
        out["mode"] = "single_state"
        out["n_bands"] = len(diff_pairs)
        out["fit_results"] = results
        out["best_hypothesis"] = ranked[0][0] if ranked else None
        out["best_chi2"] = ranked[0][1]["chi2"] if ranked else None
        return out

    diff = fit_diff_spectrum(flux_high=flux_high, flux_low=flux_low, hypotheses=primary_hypotheses, a_v=a_v)
    low = fit_low_state(flux_low=flux_low, survivor_teff_grid=survivor_teff_grid, a_v=a_v)
    out["mode"] = "three_step"
    out["step1_diff"] = diff
    out["step2_low"] = low
    out["step3_high"] = {}
    for hyp in (diff.get("results") or {}):
        out["step3_high"][hyp] = synthesize_high_state(
            flux_high_obs=flux_high, diff_fit=diff, low_fit=low,
            diff_hypothesis=hyp, a_v=a_v,
        )
    # Summary: pick the hypothesis with the joint (chi2_diff + chi2_high) min.
    summary = []
    for hyp in (diff.get("results") or {}):
        d = (diff.get("results") or {}).get(hyp, {}).get("chi2")
        h = (out["step3_high"].get(hyp) or {}).get("chi2_high")
        joint = (d or math.inf) + (h or math.inf)
        summary.append({"hypothesis": hyp, "chi2_diff": d, "chi2_high": h, "joint": joint})
    summary.sort(key=lambda r: r["joint"])
    out["ranking_joint_chi2"] = summary
    out["best_hypothesis_joint"] = summary[0]["hypothesis"] if summary else None
    return out


__all__ = [
    "HYPOTHESIS_TEMPLATES",
    "fit_diff_spectrum",
    "fit_low_state",
    "synthesize_high_state",
    "run_three_step",
]
