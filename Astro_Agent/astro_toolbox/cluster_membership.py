"""Cluster membership χ² computation against the Hunt+2023 OC catalog.

For a candidate source with Gaia DR3 (ra, dec, parallax, pmRA, pmDE, errors)
and an optional cluster name, this module computes:

    χ²_spat = Mahalanobis distance from the cluster center on the sky,
              weighted by the cluster angular extent (uses r_tidal mapped to
              an angular scale at the cluster distance as the spatial sigma).
    χ²_kin  = (Δμ_α / σ_μα)² + (Δμ_δ / σ_μδ)² + (Δϖ / σ_ϖ)²
              with σ taken in quadrature with the cluster-mean dispersion.
    σ_RV    = (RV_target - RV_cluster) / σ_RV_combined  (if RV available).

It also returns a "membership verdict" string consistent with the
UPK 13-c2-style discussion: spatially_consistent, kinematically_consistent,
parallax_consistent, rv_consistent.

This is intentionally a deterministic, classical χ²; it is NOT a Bayesian
membership probability.  Callers should compose this with the source-class
prior (e.g. Grondin+2024's WD+MS classifier P_class) for a full posterior.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import orbit_traceback as _ot


def _cluster_by_name(clusters: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    needle = name.strip().lower().replace(" ", "").replace("_", "")
    for cl in clusters:
        cl_name = str(cl.get("Name", "")).strip().lower().replace(" ", "").replace("_", "")
        if cl_name == needle:
            return cl
    return None


def _nearest_clusters(
    clusters: List[Dict[str, Any]],
    ra: float,
    dec: float,
    parallax: Optional[float],
    *,
    top_k: int = 10,
) -> List[Tuple[float, Dict[str, Any]]]:
    """Quick spatial pre-cut: return clusters sorted by 3D distance proxy."""
    out: List[Tuple[float, Dict[str, Any]]] = []
    for cl in clusters:
        try:
            d_ra = (cl["RA"] - ra) * math.cos(math.radians(dec))
            d_dec = cl["DEC"] - dec
            d_plx = 0.0
            if parallax is not None and not math.isnan(cl.get("Plx", float("nan"))):
                d_plx = (cl["Plx"] - parallax) * 50.0  # weight parallax mismatch
            score = math.sqrt(d_ra ** 2 + d_dec ** 2 + d_plx ** 2)
            out.append((score, cl))
        except (TypeError, KeyError):
            continue
    out.sort(key=lambda r: r[0])
    return out[:top_k]


def _chi2_spat(
    ra_target: float, dec_target: float,
    cluster: Dict[str, Any],
) -> float:
    """χ²_spat using cluster angular extent (mapped from tidal radius at cluster distance)."""
    d_ra = (ra_target - cluster["RA"]) * math.cos(math.radians(dec_target))
    d_dec = dec_target - cluster["DEC"]
    sep_deg = math.sqrt(d_ra * d_ra + d_dec * d_dec)
    rtpc = float(cluster.get("rtpc") or 0.0)
    dist_pc = float(cluster.get("dist50") or 0.0)
    if rtpc <= 0 or dist_pc <= 0:
        return float("nan")
    sigma_deg = math.degrees(rtpc / dist_pc)  # 1-σ extent
    if sigma_deg <= 0:
        return float("nan")
    return (sep_deg / sigma_deg) ** 2


def _chi2_kin(
    pmra: float, e_pmra: float,
    pmde: float, e_pmde: float,
    parallax: Optional[float], e_parallax: Optional[float],
    cluster: Dict[str, Any],
    *,
    cluster_pm_dispersion_mas_yr: float = 0.5,   # canonical OC PM dispersion
    cluster_plx_dispersion_mas: float = 0.05,    # canonical OC plx dispersion
) -> Tuple[float, Dict[str, float]]:
    parts: Dict[str, float] = {}
    chi2 = 0.0
    sigma_pmra = math.hypot(e_pmra, cluster_pm_dispersion_mas_yr)
    sigma_pmde = math.hypot(e_pmde, cluster_pm_dispersion_mas_yr)
    if not math.isnan(cluster.get("pmRA", float("nan"))) and sigma_pmra > 0:
        v = (pmra - cluster["pmRA"]) / sigma_pmra
        parts["pmRA"] = v * v
        chi2 += parts["pmRA"]
    if not math.isnan(cluster.get("pmDE", float("nan"))) and sigma_pmde > 0:
        v = (pmde - cluster["pmDE"]) / sigma_pmde
        parts["pmDE"] = v * v
        chi2 += parts["pmDE"]
    if (parallax is not None and e_parallax is not None
            and not math.isnan(cluster.get("Plx", float("nan")))):
        sigma_plx = math.hypot(e_parallax, cluster_plx_dispersion_mas)
        if sigma_plx > 0:
            v = (parallax - cluster["Plx"]) / sigma_plx
            parts["Plx"] = v * v
            chi2 += parts["Plx"]
    return chi2, parts


def membership(
    *,
    ra_deg: float,
    dec_deg: float,
    pmra: Optional[float] = None,
    e_pmra: Optional[float] = None,
    pmde: Optional[float] = None,
    e_pmde: Optional[float] = None,
    parallax: Optional[float] = None,
    e_parallax: Optional[float] = None,
    rv: Optional[float] = None,
    e_rv: Optional[float] = None,
    cluster_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute χ² membership scores against Hunt+2023 clusters.

    If cluster_name is given, evaluate that cluster only.  Otherwise return
    the top-5 nearest candidates with their χ² values.
    """
    clusters = _ot.load_hunt2023_clusters()
    if not clusters:
        return {"status": "no_cluster_catalog"}

    if cluster_name:
        cl = _cluster_by_name(clusters, cluster_name)
        if cl is None:
            return {"status": "cluster_not_found", "requested": cluster_name}
        candidates = [cl]
    else:
        candidates = [c for _, c in _nearest_clusters(clusters, ra_deg, dec_deg, parallax)]

    results: List[Dict[str, Any]] = []
    for cl in candidates:
        chi2_spat = _chi2_spat(ra_deg, dec_deg, cl)
        chi2_kin = float("nan")
        kin_parts: Dict[str, float] = {}
        if (pmra is not None and e_pmra is not None
                and pmde is not None and e_pmde is not None):
            chi2_kin, kin_parts = _chi2_kin(
                pmra, e_pmra, pmde, e_pmde,
                parallax, e_parallax,
                cl,
            )
        rv_sigma = float("nan")
        if (rv is not None and e_rv is not None
                and not math.isnan(cl.get("RV", float("nan")))
                and not math.isnan(cl.get("e_RV", float("nan")))):
            sigma = math.hypot(e_rv, cl["e_RV"])
            if sigma > 0:
                rv_sigma = abs(rv - cl["RV"]) / sigma
        verdict = []
        if not math.isnan(chi2_spat):
            verdict.append("spatial_ok" if chi2_spat < 9.0 else "spatial_off")
        if not math.isnan(chi2_kin):
            verdict.append("kin_ok" if chi2_kin < 12.0 else "kin_off")
        if not math.isnan(rv_sigma):
            verdict.append(f"rv_{rv_sigma:.1f}sigma")
        # Cluster age in Myr from log10(age)
        log_age = cl.get("logAge50")
        try:
            age_myr = 10 ** float(log_age) / 1.0e6 if log_age is not None else float("nan")
        except (TypeError, ValueError):
            age_myr = float("nan")
        results.append({
            "name": cl.get("Name"),
            "type": cl.get("Type"),
            "cluster_age_myr": age_myr,
            "cluster_dist_pc": cl.get("dist50"),
            "chi2_spat": chi2_spat,
            "chi2_kin": chi2_kin,
            "chi2_kin_parts": kin_parts,
            "rv_offset_sigma": rv_sigma,
            "verdict": verdict,
        })

    return {
        "status": "ok",
        "target": {
            "ra_deg": ra_deg, "dec_deg": dec_deg,
            "pmra": pmra, "pmde": pmde,
            "parallax": parallax, "rv": rv,
        },
        "candidates": results,
        "n_candidates": len(results),
        "requested_cluster": cluster_name,
    }


__all__ = ["membership"]
