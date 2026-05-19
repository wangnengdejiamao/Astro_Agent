"""
White-dwarf age method comparisons for cluster studies.

The literature uses two related but different clocks:

1. Individual-WD clock:
   t_total = t_cool + t_MS(M_initial), with M_initial inferred from an IFMR.
   This is the right diagnostic for "can this WD form within the cluster age?"

2. Cluster-anchored IFMR clock:
   t_MS = t_cluster - t_cool, then invert the main-sequence lifetime to infer
   M_initial.  This is the common open-cluster IFMR method.

Full cluster WD cooling-sequence fitting is a population method and cannot be
reproduced from one WD alone; the helper below records it as not_applicable for
single-object runs.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def cummings2018_forward_ifmr(m_initial: float) -> float:
    """
    Cummings et al. 2018 MIST-based piecewise IFMR, forward direction.

    Parameters
    ----------
    m_initial : float
        Initial/progenitor mass in Msun.

    Returns
    -------
    float
        Predicted WD final mass in Msun.
    """
    mi = _finite(m_initial)
    if not np.isfinite(mi):
        return np.nan
    if mi < 0.80 or mi > 7.20:
        return np.nan
    if mi <= 2.85:
        return 0.0873 * mi + 0.476
    if mi <= 3.60:
        return 0.181 * mi + 0.210
    return 0.0835 * mi + 0.565


def cummings2018_inverse_ifmr(m_final: float) -> float:
    """Invert the Cummings et al. 2018 piecewise IFMR."""
    mf = _finite(m_final)
    if not np.isfinite(mf):
        return np.nan
    # Cummings et al. 2018, Table 1, MIST-based IFMR.
    # Segment final-mass ranges correspond to initial masses of
    # 0.80-2.85, 2.85-3.60, and 3.60-7.20 Msun.
    if mf <= 0.725:
        mi = (mf - 0.476) / 0.0873
    elif mf <= 0.862:
        mi = (mf - 0.210) / 0.181
    else:
        mi = (mf - 0.565) / 0.0835
    return mi if 0.50 <= mi <= 8.50 else np.nan


def mist_ms_lifetime_gyr(m_initial: float) -> float:
    """MIST-like main-sequence lifetime used by the toolbox, in Gyr."""
    mi = _finite(m_initial)
    if not np.isfinite(mi) or mi <= 0:
        return np.nan
    logm = math.log10(mi)
    log_t_yr = (
        9.921
        - 3.6648 * logm
        + 1.9697 * logm**2
        - 0.9369 * logm**3
    )
    return _finite(10**log_t_yr / 1e9)


def initial_mass_from_ms_lifetime(
    ms_lifetime_gyr: float,
    *,
    m_min: float = 0.80,
    m_max: float = 7.20,
) -> float:
    """
    Invert the MIST-like lifetime relation by bisection.

    Main-sequence lifetime is monotonic over the WD progenitor mass range used
    here.  Returns NaN when the requested lifetime falls outside the IFMR range.
    """
    target = _finite(ms_lifetime_gyr)
    if not np.isfinite(target) or target <= 0:
        return np.nan

    t_lo = mist_ms_lifetime_gyr(m_max)
    t_hi = mist_ms_lifetime_gyr(m_min)
    if not (np.isfinite(t_lo) and np.isfinite(t_hi)):
        return np.nan
    if target < t_lo or target > t_hi:
        return np.nan

    lo, hi = float(m_min), float(m_max)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        t_mid = mist_ms_lifetime_gyr(mid)
        if not np.isfinite(t_mid):
            return np.nan
        # lifetime decreases with mass
        if t_mid > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def single_star_ifmr_age(
    m_final: float,
    cooling_age_gyr: float,
) -> dict[str, Any]:
    """
    Individual WD age: t_cool + IFMR-derived progenitor main-sequence lifetime.
    """
    mf = _finite(m_final)
    tcool = _finite(cooling_age_gyr)
    mi = cummings2018_inverse_ifmr(mf)
    t_ms = mist_ms_lifetime_gyr(mi)
    total = tcool + t_ms if np.isfinite(tcool) and np.isfinite(t_ms) else np.nan
    return {
        'method': 'single_star_ifmr_cummings2018',
        'm_final_msun': mf,
        'm_initial_msun': mi,
        'cooling_age_gyr': tcool,
        'ms_lifetime_gyr': t_ms,
        'total_age_gyr': total,
        'status': 'ok' if np.isfinite(total) else 'not_available',
    }


def cluster_anchored_ifmr_age(
    m_final: float,
    cooling_age_gyr: float,
    cluster_age_gyr: float,
) -> dict[str, Any]:
    """
    Open-cluster IFMR method: t_MS = t_cluster - t_cool.

    This does not measure the cluster age; it uses the cluster age as an anchor
    to infer the progenitor mass and test whether the measured WD mass lies on
    the adopted IFMR.
    """
    mf = _finite(m_final)
    tcool = _finite(cooling_age_gyr)
    tcl = _finite(cluster_age_gyr)
    t_ms = tcl - tcool if np.isfinite(tcl) and np.isfinite(tcool) else np.nan
    mi = initial_mass_from_ms_lifetime(t_ms)
    mf_expected = cummings2018_forward_ifmr(mi)
    residual = mf - mf_expected if np.isfinite(mf + mf_expected) else np.nan

    if not np.isfinite(t_ms):
        status = 'not_available'
        note = 'missing cooling age or cluster age'
    elif t_ms <= 0:
        status = 'inconsistent'
        note = 'cooling age is older than the cluster age'
    elif not np.isfinite(mi):
        status = 'out_of_range'
        note = 'cluster-anchored progenitor lifetime is outside IFMR range'
    else:
        status = 'ok'
        note = 'cluster age can anchor progenitor lifetime'

    return {
        'method': 'cluster_anchored_ifmr',
        'm_final_msun': mf,
        'm_initial_msun': mi,
        'cooling_age_gyr': tcool,
        'cluster_age_gyr': tcl,
        'cluster_minus_cooling_gyr': t_ms,
        'expected_m_final_msun': mf_expected,
        'ifmr_mass_residual_msun': residual,
        'status': status,
        'note': note,
    }


def compare_wd_age_methods(
    *,
    m_final: float,
    cooling_age_gyr: float,
    cluster_age_gyr: float | None = None,
    method_label: str = '',
) -> dict[str, Any]:
    """
    Compare the toolbox individual-WD age with cluster-anchored literature logic.
    """
    tcl = _finite(cluster_age_gyr)
    single = single_star_ifmr_age(m_final, cooling_age_gyr)
    anchored = cluster_anchored_ifmr_age(m_final, cooling_age_gyr, tcl)
    tcool = _finite(cooling_age_gyr)

    total = single.get('total_age_gyr', np.nan)
    if np.isfinite(tcl):
        cooling_delta = tcool - tcl if np.isfinite(tcool) else np.nan
        total_delta = total - tcl if np.isfinite(total) else np.nan
    else:
        cooling_delta = np.nan
        total_delta = np.nan

    if not np.isfinite(tcl):
        verdict = 'no_cluster_age'
    elif np.isfinite(tcool) and tcool > tcl:
        verdict = 'cooling_age_exceeds_cluster_age'
    elif np.isfinite(total) and total > tcl:
        verdict = 'single_star_total_age_exceeds_cluster_age'
    elif anchored.get('status') == 'ok':
        verdict = 'consistent_with_single_star_cluster_membership'
    else:
        verdict = anchored.get('status', 'not_available')

    return {
        'method_label': method_label,
        'm_final_msun': _finite(m_final),
        'cooling_age_gyr': tcool,
        'cluster_age_gyr': tcl,
        'cooling_minus_cluster_gyr': cooling_delta,
        'single_star_total_age_gyr': total,
        'single_star_total_minus_cluster_gyr': total_delta,
        'single_star_m_initial_msun': single.get('m_initial_msun', np.nan),
        'single_star_ms_lifetime_gyr': single.get('ms_lifetime_gyr', np.nan),
        'cluster_anchored_m_initial_msun': anchored.get('m_initial_msun', np.nan),
        'cluster_anchored_ms_lifetime_gyr': anchored.get('cluster_minus_cooling_gyr', np.nan),
        'cluster_anchored_expected_m_final_msun': anchored.get('expected_m_final_msun', np.nan),
        'cluster_anchored_ifmr_residual_msun': anchored.get('ifmr_mass_residual_msun', np.nan),
        'cluster_anchored_status': anchored.get('status', ''),
        'cluster_anchored_note': anchored.get('note', ''),
        'cooling_sequence_population_method': 'not_applicable_to_single_wd',
        'verdict': verdict,
    }


def save_age_method_comparison(row: dict[str, Any], out_path: str | Path) -> str:
    """Save one comparison dictionary as CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return str(path)


def pretty_summary(row: dict[str, Any]) -> str:
    """Small human-readable summary for reports."""
    def fmt(v, nd=3):
        try:
            x = float(v)
        except Exception:
            return 'N/A'
        if not math.isfinite(x):
            return 'N/A'
        return f'{x:.{nd}f}'

    return (
        f"t_cool={fmt(row.get('cooling_age_gyr'))} Gyr, "
        f"t_cluster={fmt(row.get('cluster_age_gyr'))} Gyr, "
        f"t_total(IFMR)={fmt(row.get('single_star_total_age_gyr'))} Gyr, "
        f"cluster-anchored M_i={fmt(row.get('cluster_anchored_m_initial_msun'))} Msun, "
        f"verdict={row.get('verdict', '')}"
    )
