"""Measure ingress / egress / eclipse-fraction from a phase-folded light curve.

Fits a trapezoidal eclipse model

    F(φ) = F_high                                if φ outside eclipse window
         = F_high - depth * (φ - φ_in_start)/Δ_in   inside ingress
         = F_high - depth                           inside flat floor
         = F_high - depth * (φ_out_end - φ)/Δ_out  inside egress

via a fast grid-search (no scipy required).  Returns ingress duration in
days, eclipse fraction τ/P, depth, and a "morphology" verdict
(flat-bottomed vs U-shaped) used by physics_checks to decide whether the
disk-eclipse-binary hypothesis is admissible.

Inputs are flexible:
  * direct (phi, mag) arrays, OR
  * a phase-folded light-curve CSV (the `combined_fold_*.png` companion CSV
    produced by `astro_toolbox.period_analysis` — i.e. columns `phase`, `mag`),
    OR
  * the raw band light-curve CSV + a period + epoch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- low-level model -------------------------------------------------

def trapezoid_model(
    phi: float, *,
    F_high: float, depth: float,
    phi_in_start: float, phi_in_end: float,
    phi_out_start: float, phi_out_end: float,
) -> float:
    """Trapezoid in flux (NOT mag).  phi is folded to [0, 1).
    F_high is the out-of-eclipse flux; depth is the flat-bottom dip (>0)."""
    p = phi % 1.0
    if not (phi_in_start <= phi_in_end <= phi_out_start <= phi_out_end):
        return F_high
    if p < phi_in_start or p > phi_out_end:
        return F_high
    if phi_in_start <= p < phi_in_end:
        # ingress
        f = (p - phi_in_start) / max(1e-9, phi_in_end - phi_in_start)
        return F_high - depth * f
    if phi_in_end <= p <= phi_out_start:
        return F_high - depth
    # egress
    f = (phi_out_end - p) / max(1e-9, phi_out_end - phi_out_start)
    return F_high - depth * f


def _chi2_trapezoid(
    phis: List[float], fluxes: List[float], sigmas: List[float],
    F_high: float, depth: float,
    phi_in_start: float, phi_in_end: float, phi_out_start: float, phi_out_end: float,
) -> float:
    chi2 = 0.0
    for phi, F, sig in zip(phis, fluxes, sigmas):
        model = trapezoid_model(
            phi, F_high=F_high, depth=depth,
            phi_in_start=phi_in_start, phi_in_end=phi_in_end,
            phi_out_start=phi_out_start, phi_out_end=phi_out_end,
        )
        s = sig if sig > 0 else 1.0
        chi2 += ((F - model) / s) ** 2
    return chi2


def fit_trapezoid(
    *,
    phase: List[float],
    flux: List[float],
    flux_err: Optional[List[float]] = None,
    period_days: Optional[float] = None,
    n_phase_grid: int = 40,
) -> Dict[str, Any]:
    """Grid-search a symmetric trapezoid eclipse model.

    For speed we enforce symmetry (Δ_in = Δ_out) and centre the eclipse at
    phase=0.5, then scan over:
      half_width  ∈ [0.02, 0.45]  in n_phase_grid steps
      ingress_frac ∈ [0.0, 0.5]  fraction of half_width used by ingress
      depth         analytic   given F_high and the in-eclipse mean

    Returns:
      ingress_days, egress_days, eclipse_fraction tau/P,
      depth (in input flux units), F_high baseline,
      morphology ('flat_bottomed' if ingress_frac < 0.4, else 'U_shaped'),
      chi2, dof, BIC.
    """
    if len(phase) < 10 or len(phase) != len(flux):
        return {"status": "insufficient_points", "n_points": len(phase)}
    phi = [(p % 1.0) for p in phase]
    sig = list(flux_err) if flux_err else [1.0] * len(flux)
    # Estimate F_high as the upper-90% quantile of the data (assumes eclipse
    # depth < 50% so most points are in the out-of-eclipse baseline).
    sorted_flux = sorted(flux)
    F_high = sorted_flux[int(0.9 * len(sorted_flux))]
    # Centre eclipse at the phase of the deepest 10% of points
    centre_window = sorted(
        [(p, F) for p, F in zip(phi, flux)],
        key=lambda x: x[1],
    )[: max(3, len(flux) // 20)]
    centre = sum(p for p, _ in centre_window) / len(centre_window) if centre_window else 0.5
    # Search
    best = {"chi2": float("inf")}
    for half_width in [0.02 + 0.43 * i / (n_phase_grid - 1) for i in range(n_phase_grid)]:
        for ing_frac in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            d_in = half_width * ing_frac
            phi_in_start = (centre - half_width) % 1.0
            phi_in_end = (centre - half_width + d_in) % 1.0
            phi_out_start = (centre + half_width - d_in) % 1.0
            phi_out_end = (centre + half_width) % 1.0
            # If eclipse wraps around phase 1.0, skip — keep model simple.
            if (phi_in_start > phi_in_end or phi_in_end > phi_out_start or
                    phi_out_start > phi_out_end):
                continue
            # Best depth (analytical least-squares for the depth scalar)
            in_eclipse = [(p, F, s) for p, F, s in zip(phi, flux, sig)
                          if phi_in_start <= p <= phi_out_end]
            if len(in_eclipse) < 3:
                continue
            # weighted mean of (F_high - F) inside the flat region
            flat_phis = [(p, F, s) for p, F, s in in_eclipse
                         if phi_in_end <= p <= phi_out_start]
            if len(flat_phis) < 2:
                continue
            num = sum((F_high - F) / (s * s) for _, F, s in flat_phis)
            den = sum(1.0 / (s * s) for _, _, s in flat_phis)
            depth = num / den if den > 0 else 0.0
            if depth <= 0:
                continue
            chi2 = _chi2_trapezoid(
                phi, flux, sig,
                F_high=F_high, depth=depth,
                phi_in_start=phi_in_start, phi_in_end=phi_in_end,
                phi_out_start=phi_out_start, phi_out_end=phi_out_end,
            )
            if chi2 < best["chi2"]:
                best = {
                    "chi2": chi2,
                    "F_high": F_high,
                    "depth": depth,
                    "depth_over_F_high": depth / max(F_high, 1e-12),
                    "phi_centre": centre,
                    "half_width_phase": half_width,
                    "ingress_frac": ing_frac,
                    "phi_in_start": phi_in_start, "phi_in_end": phi_in_end,
                    "phi_out_start": phi_out_start, "phi_out_end": phi_out_end,
                    "tau_over_P": 2 * half_width,
                }
    if not math.isfinite(best.get("chi2", math.inf)):
        return {"status": "no_fit"}

    # Convert phase to days
    half_w = best["half_width_phase"]
    ing_d = best["ingress_frac"] * half_w
    if period_days:
        best["ingress_days"] = ing_d * period_days
        best["eclipse_duration_days"] = 2 * half_w * period_days
        best["egress_days"] = best["ingress_days"]  # symmetric
    # Morphology verdict
    if best["ingress_frac"] < 0.4:
        best["morphology"] = "flat_bottomed"
    else:
        best["morphology"] = "U_shaped_or_partial"
    best["status"] = "ok"
    best["n_points"] = len(phase)
    best["dof"] = max(1, len(phase) - 4)
    return best


# ---------- helpers to read astro_toolbox light-curve CSVs ------------------

def fold_light_curve(
    *,
    csv_path: Path,
    period_days: float,
    t0_mjd: Optional[float] = None,
    mag_col: str = "mag",
    err_col: Optional[str] = "mag_err",
    time_col: str = "mjd",
) -> Tuple[List[float], List[float], List[float]]:
    """Read a light-curve CSV, phase-fold by period and return
    (phase, flux, flux_err).  Magnitude is converted to relative flux via
    flux = 10^(-0.4 * (mag - mag_median))."""
    import csv as _csv
    phases: List[float] = []
    fluxes: List[float] = []
    flux_errs: List[float] = []
    mags = []
    raw_rows: List[Dict[str, str]] = []
    with Path(csv_path).open() as fh:
        rdr = _csv.DictReader(fh)
        for row in rdr:
            raw_rows.append(row)
            try:
                m = float(row.get(mag_col) or "nan")
                if math.isfinite(m):
                    mags.append(m)
            except ValueError:
                continue
    if not mags:
        return [], [], []
    mag_med = sorted(mags)[len(mags) // 2]
    for row in raw_rows:
        try:
            t = float(row.get(time_col) or "nan")
            m = float(row.get(mag_col) or "nan")
            if not (math.isfinite(t) and math.isfinite(m)):
                continue
            if t0_mjd is not None:
                phi = ((t - t0_mjd) / period_days) % 1.0
            else:
                phi = (t / period_days) % 1.0
            flux = 10 ** (-0.4 * (m - mag_med))
            flux_err = 0.05
            if err_col and row.get(err_col):
                try:
                    me = float(row[err_col])
                    if math.isfinite(me):
                        flux_err = abs(0.4 * math.log(10) * flux * me)
                except ValueError:
                    pass
            phases.append(phi)
            fluxes.append(flux)
            flux_errs.append(flux_err)
        except ValueError:
            continue
    return phases, fluxes, flux_errs


def measure_from_band_csv(
    *,
    csv_path: Path,
    period_days: float,
    t0_mjd: Optional[float] = None,
    **kw,
) -> Dict[str, Any]:
    """Convenience: fold a band CSV and run fit_trapezoid in one call."""
    phi, flux, ferr = fold_light_curve(
        csv_path=csv_path, period_days=period_days, t0_mjd=t0_mjd, **kw,
    )
    if not phi:
        return {"status": "no_data_in_csv", "csv_path": str(csv_path)}
    return fit_trapezoid(phase=phi, flux=flux, flux_err=ferr, period_days=period_days)


__all__ = [
    "trapezoid_model",
    "fit_trapezoid",
    "fold_light_curve",
    "measure_from_band_csv",
]
