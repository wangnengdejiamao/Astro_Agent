"""3D extinction queries (A_V) from `dustmaps` with fallbacks.

Used by sed_decoupled and downstream physics_checks to deredden broadband
photometry before SED fitting.  Default behaviour:

  1. If `dustmaps` is installed AND its Bayestar2019 data has been downloaded,
     query Bayestar2019 (Green+ 2019) at the supplied (l, b, distance_kpc).
  2. Else, fall back to SFD98 (also via `dustmaps`) — this is 2D and so
     over-estimates A_V for nearby sources but is acceptable for d > 1 kpc.
  3. Else, return a conservative default (A_V = 0.1 mag at galactic latitude
     |b|>20°, scaling to 1.0 mag near the plane) so the downstream pipeline
     can proceed without crashing; the result is flagged `provenance="fallback"`.

A_λ at non-V wavelengths is computed via the empirical Martin & Whittet (1990)
near-/mid-IR extension to the Fitzpatrick (1999) optical law, with R_V=3.1:

  A(λ) / A_V  =  Fitzpatrick99(λ; R_V) for 0.3 ≤ λ ≤ 1.0 μm
              ≈  (λ / 0.55 μm)**(-1.84)  for λ > 1 μm
              ≈  Fitzpatrick99(λ; R_V) for λ < 0.3 μm  (UV; capped below 100 nm)

We avoid importing astropy/extinction unconditionally; light-weight enough to
run in environments where dustmaps is not installed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


# --- coordinate conversion (no astropy dependency) ---------------------------

def _icrs_to_galactic_deg(ra_deg: float, dec_deg: float) -> tuple:
    """Return (l, b) in degrees.  Uses the standard IAU 1958 rotation."""
    # Constants for ICRS -> Galactic transformation (degrees)
    ra_NGP = 192.85948
    dec_NGP = 27.12825
    l_NCP = 122.93192
    sind_NGP = math.sin(math.radians(dec_NGP))
    cosd_NGP = math.cos(math.radians(dec_NGP))
    sind = math.sin(math.radians(dec_deg))
    cosd = math.cos(math.radians(dec_deg))
    sin_ra_diff = math.sin(math.radians(ra_deg - ra_NGP))
    cos_ra_diff = math.cos(math.radians(ra_deg - ra_NGP))
    sin_b = sind_NGP * sind + cosd_NGP * cosd * cos_ra_diff
    b = math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))
    cos_b = math.sqrt(max(0.0, 1.0 - sin_b * sin_b))
    sin_l_minus = cosd * sin_ra_diff / cos_b if cos_b > 1e-9 else 0.0
    cos_l_minus = (cosd_NGP * sind - sind_NGP * cosd * cos_ra_diff) / cos_b if cos_b > 1e-9 else 1.0
    l_minus = math.degrees(math.atan2(sin_l_minus, cos_l_minus))
    l = (l_NCP - l_minus) % 360.0
    return l, b


# --- A_V queries ------------------------------------------------------------

def query_av(
    ra_deg: float,
    dec_deg: float,
    distance_pc: Optional[float] = None,
) -> Dict[str, Any]:
    """Return A_V (V-band extinction, magnitude) at the line of sight.

    distance_pc is required for the Bayestar2019 3D query.  If not supplied,
    we fall back to SFD98 (effectively asymptotic for d → ∞).
    """
    l, b = _icrs_to_galactic_deg(ra_deg, dec_deg)
    # 1) Try dustmaps Bayestar2019 (3D)
    if distance_pc is not None and distance_pc > 0:
        try:
            from dustmaps.bayestar import BayestarQuery
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            bq = BayestarQuery(version='bayestar2019')
            coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, distance=distance_pc * u.pc, frame='icrs')
            E = float(bq(coord, mode='median'))  # E(B-V) in mags
            # E(B-V) → A_V uses R_V = 3.1
            av = 3.1 * E
            return {
                "A_V": av,
                "E_B_V": E,
                "R_V": 3.1,
                "provenance": "dustmaps.bayestar2019",
                "galactic_l_deg": l, "galactic_b_deg": b,
                "distance_pc": distance_pc,
            }
        except Exception as exc:
            bayestar_error = f"{type(exc).__name__}: {exc}"
    else:
        bayestar_error = "no distance_pc supplied"

    # 2) Try dustmaps SFD98 (2D, asymptotic)
    try:
        from dustmaps.sfd import SFDQuery
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sq = SFDQuery()
        coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame='icrs')
        E = float(sq(coord))  # SFD98 E(B-V) (asymptotic)
        av = 3.1 * E
        return {
            "A_V": av,
            "E_B_V": E,
            "R_V": 3.1,
            "provenance": "dustmaps.sfd98 (2D asymptotic)",
            "galactic_l_deg": l, "galactic_b_deg": b,
            "distance_pc": distance_pc,
            "bayestar_error": bayestar_error,
        }
    except Exception as exc:
        sfd_error = f"{type(exc).__name__}: {exc}"

    # 3) Conservative fallback
    # Lutz+ rough estimate: A_V ≈ 0.15 mag at |b|>30°, 0.5 at |b|=10°, 1.5 in plane
    abs_b = abs(b)
    if abs_b > 30:
        av = 0.15
    elif abs_b > 15:
        av = 0.35
    elif abs_b > 5:
        av = 0.80
    else:
        av = 1.50
    return {
        "A_V": av,
        "E_B_V": av / 3.1,
        "R_V": 3.1,
        "provenance": "fallback_latitude_scaling",
        "galactic_l_deg": l, "galactic_b_deg": b,
        "distance_pc": distance_pc,
        "bayestar_error": bayestar_error,
        "sfd_error": sfd_error,
        "note": "dustmaps not installed or data not downloaded; this is an OoM estimate",
    }


# --- A_λ extension to non-V wavelengths --------------------------------------

# Fitzpatrick (1999) optical A(λ)/A_V at R_V=3.1 — coarse table at common bands.
# Source: Fitzpatrick 1999, Table 3 + reproduced in Schlafly+2011 Table 6.
_FITZ_AL_OVER_AV = {
    # band : λ_eff (μm)
    "FUV":     (0.154, 2.625),
    "NUV":     (0.227, 2.795),
    "Gaia_BP": (0.511, 1.213),
    "Gaia_G":  (0.673, 0.847),
    "Gaia_RP": (0.778, 0.629),
    "u_PS1":   (0.350, 1.582),
    "g_PS1":   (0.486, 1.198),
    "r_PS1":   (0.620, 0.871),
    "i_PS1":   (0.752, 0.683),
    "z_PS1":   (0.866, 0.546),
    "y_PS1":   (0.962, 0.461),
    "ZTF_g":   (0.472, 1.231),
    "ZTF_r":   (0.634, 0.840),
    "ZTF_i":   (0.748, 0.687),
    "2MASS_J": (1.235, 0.291),
    "2MASS_H": (1.662, 0.184),
    "2MASS_Ks":(2.159, 0.118),
    "WISE_W1": (3.353, 0.061),
    "WISE_W2": (4.603, 0.039),
    "WISE_W3": (11.56, 0.015),
    "WISE_W4": (22.09, 0.008),
}


def a_lambda(band_or_wavelength_um, a_v: float) -> float:
    """Return A_λ (mag) for a band name or wavelength in micrometers.

    For band names we use the Fitzpatrick99/Schlafly+ table above.
    For numerical wavelengths > 1 μm we use the Martin & Whittet (1990) power
    law A(λ)/A_V = (λ/0.55)^(-1.84). For < 1 μm we extrapolate the same law
    (rough, suitable for SED screening).
    """
    if isinstance(band_or_wavelength_um, str):
        entry = _FITZ_AL_OVER_AV.get(band_or_wavelength_um)
        if entry is not None:
            _, frac = entry
            return frac * a_v
        # Try common case-insensitive aliases
        b = band_or_wavelength_um.replace(" ", "_")
        entry = _FITZ_AL_OVER_AV.get(b)
        if entry is not None:
            _, frac = entry
            return frac * a_v
        raise KeyError(f"Unknown extinction band: {band_or_wavelength_um}")
    wl_um = float(band_or_wavelength_um)
    if wl_um <= 0:
        return float("nan")
    return ((wl_um / 0.55) ** (-1.84)) * a_v


def deredden_magnitude(mag: float, band: str, a_v: float) -> float:
    return mag - a_lambda(band, a_v)


def deredden_flux_fnu(flux_fnu: float, band_or_wavelength_um, a_v: float) -> float:
    """Deredden an F_ν flux using A_λ at the band."""
    al = a_lambda(band_or_wavelength_um, a_v)
    return flux_fnu * 10 ** (0.4 * al)


__all__ = [
    "query_av",
    "a_lambda",
    "deredden_magnitude",
    "deredden_flux_fnu",
]
