"""Stellar template library + empirical mass-luminosity relations.

Implements:
  * Pecaut & Mamajek (2013, ApJS 208, 9 — `EEM_dwarf_UBVIJHK_colors_Teff.txt`)
    → mapping spectral_type ↔ Teff, M*, R*, L*, M_G  for dwarfs F0–M9.
    Hardcoded compact subset (mid-K to mid-M plus reference O/B/A/F/G) so
    pure-Python operation without HTTP fetch.
  * Mann et al. (2019, ApJ 871, 63 — Eq. 7) empirical mass-luminosity for
    low-mass dwarfs (0.075–0.7 M⊙) via absolute K_s magnitude.
  * Synthetic broadband photometry by integrating a Planck function (BB) or a
    grey body across band passes (treated here as top-hat filters at the
    effective wavelength for speed).

Designed for SED-decomposition use, not for high-precision parameter
inference.  Returns flux densities in cgs (erg/s/cm²/Hz at 1 pc) or in mags
(absolute).  The caller supplies the distance to convert.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


# --- Pecaut & Mamajek (2013) main-sequence dwarf table ----------------------
# Compact subset; effective_wavelength → A_V coupling done elsewhere.
# Columns: SpT, Teff [K], M [M_sun], R [R_sun], L [L_sun] (rough), MV, MJ, MKs
_PM_TABLE_RAW: List[Tuple[str, float, float, float, float, float, float, float]] = [
    # SpT,   Teff, M,    R,    L,        MV,    MJ,    MKs
    ("O5V",  41500, 60.0, 13.4, 5.0e5,  -5.7,  -5.0,  -4.7),
    ("B0V",  31400, 17.5, 7.40, 5.2e4,  -4.0,  -3.4,  -3.1),
    ("A0V",   9700,  2.51,2.19, 4.3e1,  +0.7,  +0.5,  +0.3),
    ("F0V",   7200,  1.61,1.46, 5.2e0,  +2.5,  +1.7,  +1.6),
    ("G0V",   5980,  1.07,1.05, 1.14,   +4.4,  +3.5,  +3.4),
    ("G5V",   5660,  0.98,0.98, 0.74,   +4.8,  +4.0,  +3.8),
    ("K0V",   5240,  0.90,0.85, 0.42,   +5.7,  +4.5,  +4.3),
    ("K2V",   4960,  0.83,0.79, 0.28,   +6.2,  +4.8,  +4.5),
    ("K4V",   4600,  0.74,0.74, 0.18,   +6.9,  +5.0,  +4.7),
    ("K5V",   4400,  0.70,0.72, 0.15,   +7.4,  +5.3,  +5.0),
    ("K6V",   4220,  0.68,0.71, 0.12,   +7.7,  +5.5,  +5.2),
    ("K7V",   4040,  0.67,0.69, 0.10,   +8.0,  +5.7,  +5.4),
    ("K8V",   3940,  0.66,0.67, 0.085,  +8.4,  +5.9,  +5.6),
    ("K9V",   3845,  0.64,0.65, 0.073,  +8.7,  +6.1,  +5.8),
    ("M0V",   3870,  0.61,0.62, 0.072,  +8.8,  +6.2,  +5.8),
    ("M1V",   3720,  0.55,0.55, 0.054,  +9.4,  +6.5,  +6.2),
    ("M2V",   3580,  0.49,0.50, 0.043,  +9.9,  +6.8,  +6.5),
    ("M3V",   3470,  0.41,0.43, 0.029, +10.7,  +7.4,  +7.0),
    ("M4V",   3260,  0.27,0.31, 0.013, +12.0,  +8.3,  +7.8),
    ("M5V",   3050,  0.18,0.22, 0.0061,+13.5,  +9.4,  +8.8),
    ("M6V",   2810,  0.114,0.15,0.0024,+14.9, +10.5,  +9.7),
    ("M7V",   2680,  0.090,0.13,0.0011,+16.1, +11.4, +10.4),
    ("M8V",   2570,  0.080,0.12,0.00066,+17.3,+12.3, +11.1),
    ("M9V",   2440,  0.075,0.11,0.00038,+18.3,+13.0, +11.7),
]

PM_TABLE: List[Dict[str, Any]] = [
    {"SpT": r[0], "Teff_K": r[1], "M_Msun": r[2], "R_Rsun": r[3],
     "L_Lsun": r[4], "MV": r[5], "MJ": r[6], "MKs": r[7]}
    for r in _PM_TABLE_RAW
]


def lookup_by_teff(teff_K: float) -> Dict[str, Any]:
    """Return the row whose Teff is closest to the requested value."""
    if not math.isfinite(teff_K):
        return PM_TABLE[len(PM_TABLE) // 2]
    return min(PM_TABLE, key=lambda r: abs(r["Teff_K"] - teff_K))


def lookup_by_spt(spt: str) -> Optional[Dict[str, Any]]:
    spt_norm = spt.upper().replace(" ", "")
    for row in PM_TABLE:
        if row["SpT"].upper().replace(" ", "") == spt_norm:
            return row
    return None


# --- Mann et al. (2019) M-L relation for low-mass dwarfs --------------------
# Eq. 7: log10(M / Msun) = a0 + a1*MKs + a2*MKs^2 + a3*MKs^3 + a4*MKs^4 + a5*MKs^5
# Valid for 4 < MKs < 11, equivalently 0.075 < M < 0.7 Msun.
_MANN_COEFFS = (-0.642, -0.208, -8.43e-4, 7.87e-3, 1.42e-4, -2.13e-4)


def mann2019_mass_from_MKs(MKs: float) -> Optional[float]:
    if not math.isfinite(MKs) or MKs < 3.5 or MKs > 11.5:
        return None
    poly = sum(c * MKs ** i for i, c in enumerate(_MANN_COEFFS))
    return float(10 ** poly)


# --- Mass from bolometric luminosity (rough, when MKs not available) --------

def mass_from_luminosity(L_Lsun: float) -> Optional[float]:
    """Empirical fit to PM table M(L).  For M < 1 Msun: M ≈ (L/L_sun)^0.20.
    For M ≥ 1 Msun: M ≈ (L/L_sun)^0.27."""
    if not math.isfinite(L_Lsun) or L_Lsun <= 0:
        return None
    if L_Lsun < 1.0:
        return max(0.075, L_Lsun ** 0.20)
    return min(2.5, L_Lsun ** 0.27)


# --- Blackbody flux (cgs) ---------------------------------------------------

_PLANCK_H  = 6.62607015e-27   # erg s
_PLANCK_C  = 2.99792458e10    # cm/s
_PLANCK_KB = 1.380649e-16     # erg/K


def planck_flux_fnu(wavelength_A: float, teff_K: float) -> float:
    """Return F_ν (erg/s/cm²/Hz/sr) for a blackbody at Teff."""
    if not (math.isfinite(wavelength_A) and math.isfinite(teff_K) and teff_K > 0):
        return 0.0
    wl_cm = wavelength_A * 1e-8
    nu = _PLANCK_C / wl_cm
    x = _PLANCK_H * nu / (_PLANCK_KB * teff_K)
    if x < 1e-4:
        # Rayleigh-Jeans limit
        return 2 * (nu ** 2) * _PLANCK_KB * teff_K / (_PLANCK_C ** 2)
    if x > 700:
        return 0.0
    return 2 * _PLANCK_H * (nu ** 3) / (_PLANCK_C ** 2) / (math.exp(x) - 1)


def planck_flux_flambda(wavelength_A: float, teff_K: float) -> float:
    """Return F_λ (erg/s/cm²/Å/sr) for a blackbody at Teff."""
    if not (math.isfinite(wavelength_A) and math.isfinite(teff_K) and teff_K > 0):
        return 0.0
    wl_cm = wavelength_A * 1e-8
    x = _PLANCK_H * _PLANCK_C / (wl_cm * _PLANCK_KB * teff_K)
    if x > 700:
        return 0.0
    flam = 2 * _PLANCK_H * (_PLANCK_C ** 2) / (wl_cm ** 5) / (math.exp(x) - 1)
    # convert from per cm to per Å
    return flam * 1e-8


# --- Synthetic broadband photometry at a given distance ---------------------

# Reference effective wavelengths (in Å) for common broadband filters.
#
# NOTE: SPHEREx is *not* listed here.  SPHEREx is a spectrograph (R~40-130
# across 0.75-5.0 um) — its "band labels" are channel bin centres that vary
# per epoch.  The real per-channel wavelengths are written by
# `astro_toolbox.spherex.query_spectrum()` into the SPHEREx spectrum CSV and
# carried through to `sed_photometry.csv` as the `wave_A` column.  The SED
# fitting code in `sed_decoupled.py` already prefers a row's `wave_A` over
# this fallback dict, so SPHEREx rows resolve correctly even without an entry
# here.  Adding fixed wavelengths would be misleading.
BAND_WAVELENGTHS_A: Dict[str, float] = {
    "FUV":      1528.0,
    "NUV":      2310.0,
    "u_PS1":    3500.0,
    "Gaia_BP":  5110.0,
    "g_PS1":    4860.0,
    "ZTF_g":    4720.0,
    "Gaia_G":   6230.0,
    "r_PS1":    6200.0,
    "ZTF_r":    6340.0,
    "i_PS1":    7520.0,
    "z_PS1":    8660.0,
    "y_PS1":    9620.0,
    "Gaia_RP":  7770.0,
    "2MASS_J":  12350.0,
    "2MASS_H":  16620.0,
    "2MASS_Ks": 21590.0,
    "WISE_W1":  33526.0,
    "WISE_W2":  46028.0,
    "WISE_W3":  115608.0,
    "WISE_W4":  220883.0,
}


def synthetic_flux_fnu_at_distance(
    *,
    teff_K: float,
    radius_Rsun: float,
    distance_pc: float,
    band: str,
) -> float:
    """Approximate F_ν (cgs, erg/s/cm²/Hz) at Earth assuming a blackbody
    photosphere with the given Teff and radius.

    F_ν = (R/d)² · B_ν(T)  with B_ν the Planck specific intensity integrated
    over solid angle (factor π for emergent flux from a uniformly emitting
    sphere).
    """
    wl = BAND_WAVELENGTHS_A.get(band)
    if wl is None:
        return 0.0
    R_cm = radius_Rsun * 6.957e10
    d_cm = distance_pc * 3.0857e18
    solid = (R_cm / d_cm) ** 2
    B_nu = planck_flux_fnu(wl, teff_K)
    return math.pi * B_nu * solid


def synthetic_flux_table(
    *,
    teff_K: float,
    radius_Rsun: float,
    distance_pc: float,
    bands: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for b in bands:
        wl = BAND_WAVELENGTHS_A.get(b)
        if wl is None:
            continue
        f_nu = synthetic_flux_fnu_at_distance(
            teff_K=teff_K, radius_Rsun=radius_Rsun, distance_pc=distance_pc, band=b
        )
        out[b] = {"wave_A": wl, "F_nu_cgs": f_nu}
    return out


__all__ = [
    "PM_TABLE",
    "lookup_by_teff",
    "lookup_by_spt",
    "mann2019_mass_from_MKs",
    "mass_from_luminosity",
    "planck_flux_fnu",
    "planck_flux_flambda",
    "BAND_WAVELENGTHS_A",
    "synthetic_flux_fnu_at_distance",
    "synthetic_flux_table",
]
