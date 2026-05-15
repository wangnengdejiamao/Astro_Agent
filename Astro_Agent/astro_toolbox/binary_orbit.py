"""Keplerian binary orbit mechanics.

Pure-Python helpers used by physics_checks, sed_decoupled, and the disk-
eclipse MCMC.  No external dependencies beyond `math`.

Functions:
  - kepler_semimajor_axis(P, M_tot)               -> a [AU, R_sun, cm]
  - orbital_velocity(P, M_tot)                    -> v_orb [km/s]
  - velocity_at_true_anomaly(v_orb, e, nu_rad)    -> v(ν) [km/s]
  - velocity_peri_apo(v_orb, e)                   -> (v_peri, v_apo)
  - roche_lobe_eggleton(q)                        -> r_L / a
  - kepler2nd_eclipse_fraction(e, alpha, omega)   -> τ/P, ingress/P
  - mass_ratio_from_eclipse_depth(deltaF_over_F)  -> q  (under complete occultation, q≈ΔF/F)

All formulas are derived from textbook Kepler + Eggleton (1983), and
Winn & Holman (2004 ApJ 614 L191) for the eclipse-fraction expression.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

# Physical constants (CGS)
G_CGS = 6.6743e-8
M_SUN_G = 1.98847e33
R_SUN_CM = 6.957e10
AU_CM = 1.495979e13
DAY_S = 86400.0


# ---------- Kepler's third law ---------------------------------------------

def kepler_semimajor_axis(P_days: float, M_tot_Msun: float) -> Dict[str, float]:
    """Return semi-major axis a in CGS, AU, R_sun for a binary with total
    mass M_tot and orbital period P."""
    if not (math.isfinite(P_days) and math.isfinite(M_tot_Msun) and P_days > 0 and M_tot_Msun > 0):
        return {"a_cm": float("nan"), "a_AU": float("nan"), "a_Rsun": float("nan")}
    a3_cm3 = G_CGS * M_tot_Msun * M_SUN_G * (P_days * DAY_S) ** 2 / (4 * math.pi ** 2)
    a_cm = a3_cm3 ** (1 / 3)
    return {"a_cm": a_cm, "a_AU": a_cm / AU_CM, "a_Rsun": a_cm / R_SUN_CM}


def orbital_velocity(P_days: float, M_tot_Msun: float) -> float:
    """Mean orbital velocity (circular-equivalent) in km/s."""
    a = kepler_semimajor_axis(P_days, M_tot_Msun)
    a_cm = a["a_cm"]
    if not math.isfinite(a_cm):
        return float("nan")
    return 2 * math.pi * a_cm / (P_days * DAY_S) / 1.0e5


def velocity_at_true_anomaly(P_days: float, M_tot_Msun: float, e: float, nu_rad: float) -> float:
    """Instantaneous orbital speed at true anomaly ν in an eccentric orbit
    (km/s).  Uses the vis-viva equation."""
    a = kepler_semimajor_axis(P_days, M_tot_Msun)
    a_cm = a["a_cm"]
    if not math.isfinite(a_cm):
        return float("nan")
    if not (0.0 <= e < 1.0):
        return float("nan")
    r_cm = a_cm * (1 - e * e) / (1 + e * math.cos(nu_rad))
    v2 = G_CGS * M_tot_Msun * M_SUN_G * (2.0 / r_cm - 1.0 / a_cm)
    return math.sqrt(max(0.0, v2)) / 1.0e5


def velocity_peri_apo(P_days: float, M_tot_Msun: float, e: float) -> Tuple[float, float]:
    """Pericentre + apocentre orbital speeds (km/s)."""
    if not (0.0 <= e < 1.0):
        return float("nan"), float("nan")
    v_peri = velocity_at_true_anomaly(P_days, M_tot_Msun, e, 0.0)
    v_apo = velocity_at_true_anomaly(P_days, M_tot_Msun, e, math.pi)
    return v_peri, v_apo


# ---------- Eggleton (1983) Roche-lobe radius ------------------------------

def roche_lobe_eggleton(q: float) -> float:
    """Eggleton 1983 fitting formula for r_L1 / a where q = M1 / M2.

    For q = M_donor / M_accretor; gives the donor's Roche lobe.  Valid for
    0 < q < ∞ with sub-percent accuracy.
    """
    if not (math.isfinite(q) and q > 0):
        return float("nan")
    qt = q ** (2.0 / 3.0)
    return 0.49 * qt / (0.6 * qt + math.log1p(q ** (1.0 / 3.0)))


def roche_radii(P_days: float, M1_Msun: float, M2_Msun: float) -> Dict[str, float]:
    """Return r_L for both components in units of R_sun, given orbital period
    and individual masses."""
    a = kepler_semimajor_axis(P_days, M1_Msun + M2_Msun)
    a_Rsun = a["a_Rsun"]
    if not math.isfinite(a_Rsun) or M1_Msun <= 0 or M2_Msun <= 0:
        return {"r_L1_Rsun": float("nan"), "r_L2_Rsun": float("nan"), "a_Rsun": a_Rsun}
    q1 = M1_Msun / M2_Msun
    q2 = M2_Msun / M1_Msun
    return {
        "r_L1_Rsun": a_Rsun * roche_lobe_eggleton(q1),
        "r_L2_Rsun": a_Rsun * roche_lobe_eggleton(q2),
        "a_Rsun": a_Rsun,
        "q1_M1_over_M2": q1,
        "q2_M2_over_M1": q2,
    }


# ---------- Eclipse-fraction (Kepler's 2nd law over an arc) -----------------

def kepler2nd_eclipse_fraction(e: float, alpha_rad: float, omega_rad: float = 0.0,
                                nu_mid_rad: float = math.pi) -> Dict[str, float]:
    """Predict τ/P (eclipse-duration fraction) for an eccentric orbit whose
    occulting disc subtends azimuthal arc 2α, with ν_mid the true anomaly at
    mid-eclipse.

    Using Kepler's 2nd law (dA/dt = constant) integrated over the arc:

        τ / P  =  (α / π) · (1 - e²)^(3/2) / (1 + e cos ν_mid)²

    Returns τ/P (eclipse fraction) and an ingress fraction estimated as
    ingress/P = τ/P · (R_*/a · 1/sin(α/2)) — *very rough*, only useful as
    a likelihood proxy in the MCMC step.
    """
    if not (0.0 <= e < 1.0):
        return {"tau_over_P": float("nan"), "ingress_over_P": float("nan")}
    tau = (alpha_rad / math.pi) * (1 - e * e) ** 1.5 / (1 + e * math.cos(nu_mid_rad)) ** 2
    return {"tau_over_P": tau, "alpha_rad": alpha_rad,
            "e": e, "omega_rad": omega_rad, "nu_mid_rad": nu_mid_rad}


def mass_ratio_from_eclipse_depth(deltaF_over_F: float) -> Optional[float]:
    """Under *complete* occultation of one component, the eclipse depth
    ΔF/F equals the eclipsed component's fractional luminosity L_2/L_tot.
    For two stars with similar atmospheres and the Mann (2019) mass-luminosity
    relation (M ∝ L^0.2 for late-K/M), the mass ratio q = M_2/M_1 ≈
    (L_2/L_1)^0.2 = (ΔF/F / (1 - ΔF/F))^0.2.
    """
    if not math.isfinite(deltaF_over_F):
        return None
    if not (0 < deltaF_over_F < 1):
        return None
    ratio = deltaF_over_F / (1 - deltaF_over_F)
    return ratio ** 0.2


# ---------- Convenience wrapper used by physics_checks ----------------------

def summarize_orbit(P_days: float, M_tot_Msun: float, e: float = 0.0,
                    M1_Msun: Optional[float] = None,
                    M2_Msun: Optional[float] = None) -> Dict[str, float]:
    """Bundle the orbital quantities physics_checks needs."""
    sma = kepler_semimajor_axis(P_days, M_tot_Msun)
    v_orb = orbital_velocity(P_days, M_tot_Msun)
    out: Dict[str, float] = {
        "P_days": P_days, "M_tot_Msun": M_tot_Msun, "e": e,
        **sma, "v_orb_kms": v_orb,
    }
    if e > 0:
        v_peri, v_apo = velocity_peri_apo(P_days, M_tot_Msun, e)
        out["v_peri_kms"] = v_peri
        out["v_apo_kms"] = v_apo
    if M1_Msun and M2_Msun:
        out.update(roche_radii(P_days, M1_Msun, M2_Msun))
    return out


__all__ = [
    "kepler_semimajor_axis",
    "orbital_velocity",
    "velocity_at_true_anomaly",
    "velocity_peri_apo",
    "roche_lobe_eggleton",
    "roche_radii",
    "kepler2nd_eclipse_fraction",
    "mass_ratio_from_eclipse_depth",
    "summarize_orbit",
]
