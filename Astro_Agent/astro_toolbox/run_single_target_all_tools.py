#!/usr/bin/env python3
"""
Run the astro_toolbox science modules for one target and collect outputs.
"""

import argparse
import glob
import json
import os
import shutil
import signal
import sys
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

PYKOA_TMP = "/tmp/pykoa_deps"
if os.path.isdir(PYKOA_TMP) and PYKOA_TMP not in sys.path:
    sys.path.insert(0, PYKOA_TMP)

warnings.filterwarnings("ignore")

from astro_toolbox import (  # noqa: E402
    combined_plots,
    cooling_age,
    diagnostics,
    gaia_lc,
    galah,
    galex,
    hr_diagram,
    hst,
    jwst,
    kepler,
    koa,
    lamost,
    period_analysis,
    rv_correction,
    rv_fitting,
    sdss,
    sed,
    six_dim,
    spherex,
    tess,
    twomass,
    utils,
    wd_fitting,
    wise,
    xray,
    ztf,
)
from astro_toolbox.desi import DESITool  # noqa: E402


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(text):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in "._+-":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, pd.DataFrame):
        return {
            "type": "dataframe",
            "rows": int(len(value)),
            "columns": list(value.columns),
        }
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def summarize_result(result):
    if result is None:
        return {"status": "none"}
    if isinstance(result, pd.DataFrame):
        return {"type": "dataframe", "rows": int(len(result)),
                "columns": list(result.columns)}
    if isinstance(result, dict):
        summary = {"keys": sorted(result.keys())[:100]}
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                summary[f"{key}_rows"] = int(len(value))
            elif isinstance(value, np.ndarray):
                summary[f"{key}_shape"] = list(value.shape)
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
                summary[f"{key}_len"] = len(value)
            elif isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = to_jsonable(value)
        return summary
    return {"repr": str(result)}


def _compact_target_name(target):
    return "".join(ch for ch in str(target) if ch.isalnum() or ch in ".+-")


def _source_id_from_target(target):
    chunks = []
    current = []
    for ch in str(target):
        if ch.isdigit():
            current.append(ch)
        elif current:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    long_chunks = [c for c in chunks if len(c) >= 10]
    if long_chunks:
        return max(long_chunks, key=len)
    return max(chunks, key=len) if chunks else str(target)


def _load_cached_tess_lightcurve(target, output_root, tess_dir, ra, dec):
    """Load a previously downloaded TESS CSV when online MAST access fails."""
    compact = _compact_target_name(target)
    parent = os.path.dirname(output_root)
    candidates = [
        os.path.join(output_root, "tess", "tess_lightcurve.csv"),
        os.path.join(output_root, "tess_lightcurve.csv"),
    ]
    if parent and compact:
        candidates.extend(glob.glob(os.path.join(
            parent, f"{compact}*", "tess", "tess_lightcurve.csv")))
        candidates.extend(glob.glob(os.path.join(
            parent, f"{compact}*", "tess_lightcurve.csv")))

    seen = set()
    existing = []
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        existing.append(path)
    existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    for path in existing:
        try:
            df = pd.read_csv(path)
            time_col = None
            for col in ("time_BTJD", "time", "btjd", "BTJD"):
                if col in df.columns:
                    time_col = col
                    break
            if time_col is None or "flux" not in df.columns:
                continue
            time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
            flux = pd.to_numeric(df["flux"], errors="coerce").to_numpy(float)
            if "flux_err" in df.columns:
                flux_err = pd.to_numeric(
                    df["flux_err"], errors="coerce").to_numpy(float)
            else:
                flux_err = np.full_like(flux, np.nan, dtype=float)
            finite = np.isfinite(time) & np.isfinite(flux)
            if np.sum(finite) < 10:
                continue
            res = {
                "survey": "TESS",
                "ra": ra,
                "dec": dec,
                "time": time,
                "flux": flux,
                "flux_err": flux_err,
                "sectors": ["cached"],
                "author": "local_csv",
                "n_points": int(len(time)),
                "n_finite_points": int(np.sum(finite)),
                "obs_time_min": float(np.nanmin(time[np.isfinite(time)])),
                "obs_time_max": float(np.nanmax(time[np.isfinite(time)])),
                "time_system": "BTJD",
                "cache_source": path,
            }
            tess.plot_lightcurve(res, os.path.join(tess_dir, "tess_lightcurve.png"))
            tess.save_csv(res, tess_dir)
            save_text(os.path.join(tess_dir, "cache_source.txt"), path + "\n")
            return res
        except Exception:
            continue
    return None


def _load_cached_ztf_lightcurve(target, output_root, ztf_dir, ra, dec):
    """Load a previously downloaded ZTF CSV when online IRSA access fails."""
    compact = _compact_target_name(target)
    parent = os.path.dirname(output_root)
    candidates = [
        os.path.join(output_root, "ztf", "ztf_lightcurve.csv"),
        os.path.join(output_root, "ztf_lightcurve.csv"),
    ]
    if parent and compact:
        candidates.extend(glob.glob(os.path.join(
            parent, f"{compact}*", "ztf", "ztf_lightcurve.csv")))
        candidates.extend(glob.glob(os.path.join(
            parent, f"{compact}*", "ztf_lightcurve.csv")))

    seen = set()
    existing = []
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        existing.append(path)
    existing.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    band_map = {
        "zg": "g",
        "zr": "r",
        "zi": "i",
        "g": "g",
        "r": "r",
        "i": "i",
        "1": "g",
        "2": "r",
        "3": "i",
    }
    for path in existing:
        try:
            df = pd.read_csv(path)
            cols = {str(c).strip().lower(): c for c in df.columns}
            mjd_col = cols.get("mjd")
            hjd_col = cols.get("hjd")
            mag_col = cols.get("mag")
            err_col = cols.get("magerr") or cols.get("mag_err")
            band_col = cols.get("band") or cols.get("filtercode") or cols.get("filter")
            if mag_col is None or err_col is None or (mjd_col is None and hjd_col is None):
                continue
            if "catflags" in cols:
                df = df[pd.to_numeric(df[cols["catflags"]], errors="coerce").fillna(0).eq(0)].copy()
            if mjd_col is None and hjd_col is not None:
                df["mjd"] = pd.to_numeric(df[hjd_col], errors="coerce") - 2400000.5
                mjd_col = "mjd"
            keep = pd.DataFrame({
                "mjd": pd.to_numeric(df[mjd_col], errors="coerce"),
                "mag": pd.to_numeric(df[mag_col], errors="coerce"),
                "magerr": pd.to_numeric(df[err_col], errors="coerce"),
            })
            if hjd_col is not None:
                keep["hjd"] = pd.to_numeric(df[hjd_col], errors="coerce")
            if band_col is not None:
                keep["band"] = df[band_col].astype(str).str.strip().str.lower().map(
                    lambda x: band_map.get(x, x)
                )
            else:
                keep["band"] = "all"
            keep = keep.dropna(subset=["mjd", "mag", "magerr"]).sort_values("mjd")
            if len(keep) < 5:
                continue

            result = {
                "ra": ra,
                "dec": dec,
                "survey": "ZTF cached",
                "cache_source": path,
                "n_epochs": int(len(keep)),
                "obs_mjd_min": float(keep["mjd"].min()),
                "obs_mjd_max": float(keep["mjd"].max()),
                "web_url": ztf.get_web_url(ra, dec),
            }
            cols_out = ["mjd", "mag", "magerr"]
            if "hjd" in keep.columns:
                cols_out.insert(1, "hjd")
            for band in ("g", "r", "i", "all"):
                sub = keep[keep["band"].eq(band)][cols_out].copy()
                if len(sub) > 0:
                    result[band] = sub.reset_index(drop=True)
            if not any(b in result for b in ("g", "r", "i", "all")):
                continue
            ztf.plot_lightcurve(result, os.path.join(ztf_dir, "ztf_lightcurve.png"))
            ztf.save_csv(result, ztf_dir)
            save_text(os.path.join(ztf_dir, "cache_source.txt"), path + "\n")
            return result
        except Exception:
            continue
    return None


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=2, ensure_ascii=False)


def write_status_csv(status_rows, output_root):
    path = os.path.join(output_root, "module_status.csv")
    pd.DataFrame(status_rows).to_csv(path, index=False)
    return path


def add_status(status_rows, output_root, module, status, output_dir=None,
               note="", files=None):
    row = {
        "module": module,
        "status": status,
        "output_dir": output_dir or "",
        "note": note or "",
        "files": ";".join(files or []),
        "updated_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }
    status_rows.append(row)
    write_status_csv(status_rows, output_root)


def flatten_photometry(*phot_dicts):
    merged = {}
    for phot in phot_dicts:
        if isinstance(phot, dict):
            merged.update(phot)
    return merged


def _desi_spectrum_as_arrays(desi_res):
    if isinstance(desi_res, dict) and "spectrum" in desi_res:
        sp = desi_res["spectrum"]
        waves = []
        fluxes = []
        errors = []
        for band in ("B", "R", "Z"):
            if band not in sp:
                continue
            waves.append(np.asarray(sp[band]["wavelength"], dtype=float))
            fluxes.append(np.asarray(sp[band]["flux"], dtype=float))
            errors.append(np.asarray(sp[band]["error"], dtype=float))
        if waves:
            return {
                "wavelength": np.concatenate(waves),
                "flux": np.concatenate(fluxes),
                "error": np.concatenate(errors),
            }
    return None


def _normalize_wd_spectrum_candidate(name, spec):
    if not isinstance(spec, dict) or "wavelength" not in spec or "flux" not in spec:
        return None
    wave = np.asarray(spec["wavelength"], dtype=float)
    flux = np.asarray(spec["flux"], dtype=float)
    err = spec.get("error")
    err = np.asarray(err, dtype=float) if err is not None else None
    valid = np.isfinite(wave) & np.isfinite(flux)
    if err is not None and err.shape == wave.shape:
        valid &= np.isfinite(err)
    finite_wave = wave[valid]
    if finite_wave.size < 100:
        return None
    wmin = float(np.nanmin(finite_wave))
    wmax = float(np.nanmax(finite_wave))
    has_optical = wmax >= 3700 and wmin <= 9200
    balmer = [6562.8, 4861.3, 4340.5, 4101.7, 3970.1, 3889.1]
    n_balmer = sum(wmin <= line <= wmax for line in balmer)
    if not has_optical or n_balmer < 2:
        return None
    order = np.argsort(wave[valid])
    out = {
        "wavelength": wave[valid][order],
        "flux": flux[valid][order],
        "source_name": name,
        "wavelength_min_A": wmin,
        "wavelength_max_A": wmax,
        "n_balmer_lines_covered": int(n_balmer),
    }
    if err is not None and err.shape == wave.shape:
        out["error"] = err[valid][order]
    return out


def get_wd_spectrum_candidates(results):
    raw_candidates = [
        ("SDSS", results.get("SDSS_spectrum")),
        ("DESI", _desi_spectrum_as_arrays(results.get("DESI"))),
        ("LAMOST", results.get("LAMOST")),
        ("KOA", results.get("KOA_spectrum")),
        ("HST", results.get("HST_spectrum")),
        ("JWST", results.get("JWST_spectrum")),
        ("SPHEREx", results.get("SPHEREx")),
    ]
    candidates = []
    for name, spec in raw_candidates:
        norm = _normalize_wd_spectrum_candidate(name, spec)
        if norm is not None:
            candidates.append((name, norm))
    return candidates


def choose_wd_spectrum(results):
    candidates = get_wd_spectrum_candidates(results)
    if candidates:
        return candidates[0]
    return None, None


def _wd_fit_quality_summary(name, spec, pack, hr_params=None):
    balmer = (pack or {}).get("balmer_fit") or {}
    continuum = (pack or {}).get("continuum_fit") or {}
    phys = (pack or {}).get("physical_params") or {}
    cls = (pack or {}).get("classification") or {}

    def _f(value, default=np.nan):
        try:
            value = float(value)
            return value if np.isfinite(value) else default
        except Exception:
            return default

    teff = _f(balmer.get("teff"))
    logg = _f(balmer.get("logg"))
    chi2 = _f(balmer.get("chi2_red"), default=np.inf)
    cont_teff = _f(continuum.get("teff"))
    cont_logg = _f(continuum.get("logg"))
    n_lines = len(balmer.get("lines_used", []) or [])
    gaia_teff = _f((hr_params or {}).get("wd_teff_k"))
    gaia_logg = _f((hr_params or {}).get("wd_logg"))
    if not np.isfinite(gaia_teff):
        gaia_teff = _f(phys.get("gaia_hr_teff"))
    if not np.isfinite(gaia_logg):
        gaia_logg = _f(phys.get("gaia_hr_logg"))
    scale_ok = continuum.get("scale_radius_ok")
    if scale_ok is None:
        scale_ok = balmer.get("scale_radius_ok")

    score = 1e6
    if np.isfinite(chi2) and np.isfinite(teff) and np.isfinite(logg):
        score = chi2
        score += max(0, 4 - n_lines) * 0.30
        if np.isfinite(cont_teff):
            score += min(abs(teff - cont_teff) / 8000.0, 1.0) * 0.40
        if np.isfinite(cont_logg):
            score += min(abs(logg - cont_logg) / 0.8, 1.0) * 0.35
        if np.isfinite(gaia_teff):
            score += min(abs(teff - gaia_teff) / 6000.0, 1.0) * 0.35
        if np.isfinite(gaia_logg):
            score += min(abs(logg - gaia_logg) / 0.8, 1.0) * 0.50
        if scale_ok is False:
            score += 0.35
        if cls.get("spectral_type") != "DA":
            score += 0.25

    return {
        "source": name,
        "fit_quality_score": float(score),
        "spectral_type": cls.get("spectral_type", ""),
        "confidence": cls.get("confidence"),
        "teff": teff,
        "logg": logg,
        "balmer_chi2_red": chi2,
        "balmer_fit_score": balmer.get("fit_score"),
        "balmer_rv_kms": balmer.get("rv_kms"),
        "balmer_lines_used": ";".join(balmer.get("lines_used", []) or []),
        "n_balmer_lines_used": n_lines,
        "model_grid": balmer.get("model_grid") or continuum.get("model_grid"),
        "balmer_scale_radius_rsun": balmer.get("scale_radius_rsun"),
        "continuum_scale_radius_rsun": continuum.get("scale_radius_rsun"),
        "scale_radius_ok": scale_ok,
        "continuum_teff": cont_teff,
        "continuum_logg": cont_logg,
        "continuum_chi2_red": continuum.get("chi2_red"),
        "mass": phys.get("mass"),
        "radius_rsun": phys.get("radius_rsun"),
        "cooling_age_gyr": phys.get("cooling_age_gyr"),
        "gaia_hr_mass": phys.get("gaia_hr_mass") or (hr_params or {}).get("wd_mass_msun"),
        "gaia_hr_teff": phys.get("gaia_hr_teff") or (hr_params or {}).get("wd_teff_k"),
        "gaia_hr_logg": phys.get("gaia_hr_logg") or (hr_params or {}).get("wd_logg"),
        "gaia_hr_cooling_age_gyr": (
            phys.get("gaia_hr_cooling_age_gyr")
            or (hr_params or {}).get("wd_cooling_age_gyr")
        ),
        "wavelength_min_A": spec.get("wavelength_min_A"),
        "wavelength_max_A": spec.get("wavelength_max_A"),
        "n_balmer_lines_covered": spec.get("n_balmer_lines_covered"),
    }


def _run_wd_fit_for_candidate(name, spec, output_dir, hr_params, sed_fitter):
    fitter = wd_fitting.WDFitter(
        spec["wavelength"],
        spec["flux"],
        spec.get("error"),
    )
    bp_rp = hr_params.get("BP_RP") if hr_params else None
    abs_g = hr_params.get("M_G") if hr_params else None
    plx = hr_params.get("Plx") if hr_params else None
    phot = sed_fitter.photometry if sed_fitter is not None else None
    pack = fitter.run_all(
        photometry=phot,
        parallax_mas=plx,
        bp_rp=bp_rp,
        M_G=abs_g,
        output_dir=output_dir,
    )
    pack["spectrum_source"] = name
    if pack.get("physical_params") is not None:
        pack["physical_params"]["spectral_type"] = (
            (pack.get("classification") or {}).get("spectral_type", "DA")
        )
        pack["physical_params"]["spectrum_source"] = name
    return pack


def fit_best_wd_spectrum(results, wd_dir, hr_params=None, sed_fitter=None):
    candidates = get_wd_spectrum_candidates(results)
    if not candidates:
        return None, None, None

    rows = []
    packs = {}
    specs = {}
    for name, spec in candidates:
        subdir = ensure_dir(os.path.join(wd_dir, name.lower()))
        try:
            pack = _run_wd_fit_for_candidate(
                name, spec, subdir, hr_params, sed_fitter)
            summary = _wd_fit_quality_summary(name, spec, pack, hr_params)
            rows.append(summary)
            packs[name] = pack
            specs[name] = spec
        except Exception as exc:
            rows.append({
                "source": name,
                "fit_quality_score": np.inf,
                "error": f"{type(exc).__name__}: {exc}",
                "wavelength_min_A": spec.get("wavelength_min_A"),
                "wavelength_max_A": spec.get("wavelength_max_A"),
                "n_balmer_lines_covered": spec.get("n_balmer_lines_covered"),
            })

    valid_rows = [r for r in rows if np.isfinite(r.get("fit_quality_score", np.inf))]
    if not valid_rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(wd_dir, "wd_spectrum_comparison.csv"), index=False)
        return None, None, None

    selected_row = min(valid_rows, key=lambda r: r["fit_quality_score"])
    selected_name = selected_row["source"]
    for row in rows:
        row["selected"] = (row.get("source") == selected_name)

    pd.DataFrame(rows).to_csv(
        os.path.join(wd_dir, "wd_spectrum_comparison.csv"), index=False)

    # Refresh the legacy top-level WD products with the selected spectrum.
    selected_pack = _run_wd_fit_for_candidate(
        selected_name, specs[selected_name], wd_dir, hr_params, sed_fitter)
    selected_pack["comparison"] = rows
    selected_pack["fit_quality_score"] = selected_row["fit_quality_score"]
    return selected_name, specs[selected_name], selected_pack


def save_rv_physical_variants(rv_report, rv_corr, wd_out, wd_phys,
                              output_dir, selected_source=""):
    """Save RV_true variants using line-core, pipeline, CCF, and WD-fit RVs."""
    if output_dir is None:
        return None

    def _f(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except Exception:
            return np.nan

    rv_obs_rows = []
    if rv_corr:
        rv_obs_rows.append({
            "rv_obs_source": f"{selected_source}_line_core_correction",
            "rv_obs_kms": rv_corr.get("rv_obs"),
            "rv_obs_err_kms": rv_corr.get("rv_obs_err"),
            "rv_method": rv_corr.get("rv_obs_source", ""),
        })
    if rv_report:
        for p in rv_report.get("pipeline_rvs", []) or []:
            rv_obs_rows.append({
                "rv_obs_source": f"{p.get('survey')}_pipeline",
                "rv_obs_kms": p.get("rv"),
                "rv_obs_err_kms": p.get("rv_err"),
                "rv_method": p.get("source", "pipeline"),
            })
        for survey, ccf in (rv_report.get("ccf_results", {}) or {}).items():
            single = ccf.get("single") or {}
            rv_obs_rows.append({
                "rv_obs_source": f"{survey}_CCF",
                "rv_obs_kms": single.get("rv"),
                "rv_obs_err_kms": single.get("rv_err"),
                "rv_method": "CCF",
            })
        if rv_report.get("best_rv") is not None:
            rv_obs_rows.append({
                "rv_obs_source": rv_report.get("best_rv_source", "best_rv"),
                "rv_obs_kms": rv_report.get("best_rv"),
                "rv_obs_err_kms": rv_report.get("best_rv_err"),
                "rv_method": "adopted_by_rv_module",
            })

    balmer = (wd_out or {}).get("balmer_fit") or {}
    if balmer.get("rv_kms") is not None:
        rv_obs_rows.append({
            "rv_obs_source": f"{selected_source}_WD_Balmer_model_shift",
            "rv_obs_kms": balmer.get("rv_kms"),
            "rv_obs_err_kms": np.nan,
            "rv_method": "WD_atmosphere_balmer_grid",
        })

    phys_rows = []
    if wd_phys:
        phys_rows.append({
            "physical_source": "selected_spectroscopic_fit",
            "teff": wd_phys.get("teff"),
            "logg": wd_phys.get("logg"),
            "mass_msun": wd_phys.get("mass"),
            "radius_rsun": wd_phys.get("radius_rsun"),
            "cooling_age_gyr": wd_phys.get("cooling_age_gyr"),
        })
        g_mass = wd_phys.get("gaia_hr_mass")
        g_logg = wd_phys.get("gaia_hr_logg")
        if g_mass is not None and g_logg is not None:
            try:
                g_radius = wd_fitting.compute_wd_radius(float(g_mass), float(g_logg))
            except Exception:
                g_radius = wd_phys.get("gaia_hr_radius_rsun")
            phys_rows.append({
                "physical_source": "gaia_hr_distance_photometry_check",
                "teff": wd_phys.get("gaia_hr_teff"),
                "logg": g_logg,
                "mass_msun": g_mass,
                "radius_rsun": g_radius,
                "cooling_age_gyr": wd_phys.get("gaia_hr_cooling_age_gyr"),
            })

    rows = []
    for rv_row in rv_obs_rows:
        rv_obs = _f(rv_row.get("rv_obs_kms"))
        for phys in phys_rows:
            mass = _f(phys.get("mass_msun"))
            radius = _f(phys.get("radius_rsun"))
            if not np.isfinite(rv_obs + mass + radius) or radius <= 0:
                continue
            v_grav = rv_correction.gravitational_redshift(mass, radius)
            rows.append({
                **rv_row,
                **phys,
                "v_grav_kms": v_grav,
                "rv_true_kms": rv_obs - v_grav,
            })
    if not rows:
        return None

    path = os.path.join(output_dir, "rv_wd_physical_variants.csv")
    pd.DataFrame(rows).to_csv(path, index=False)

    best = None
    for row in rows:
        if row.get("rv_obs_source") == "SDSS_pipeline" and row.get(
                "physical_source") == "selected_spectroscopic_fit":
            best = row
            break
    if best is None:
        best = rows[0]

    txt = os.path.join(output_dir, "rv_wd_physical_variants.txt")
    lines = [
        "RV and WD physical-parameter variants",
        "=" * 44,
        "Rows combine each available observed RV with each WD mass/radius branch.",
        "Use this table to inspect whether the RV_true result is dominated by",
        "the observed RV choice or by gravitational-redshift/systematic log g.",
        "",
        f"Representative row: {best.get('rv_obs_source')} + "
        f"{best.get('physical_source')}",
        f"  RV_obs = {best.get('rv_obs_kms'):.2f} km/s",
        f"  V_grav = {best.get('v_grav_kms'):.2f} km/s",
        f"  RV_true = {best.get('rv_true_kms'):.2f} km/s",
    ]
    save_text(txt, "\n".join(lines) + "\n")
    return path


def _load_rv_physical_variants_rows(path):
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    return df.to_dict(orient="records")


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    text = str(value).strip().lower()
    return text in ("true", "1", "yes", "y", "t", "yes ***")


def _select_traceback_candidate(trace_result):
    """
    Prefer the first candidate inside the tidal radius, matching the
    backtrack_only logic used by the reference 6D plotting script.  If none are
    in-tidal, fall back to best_match / the nearest candidate.
    """
    if not isinstance(trace_result, dict):
        return None
    candidates = trace_result.get("candidates") or []
    for cand in candidates:
        if _as_bool(cand.get("within_tidal", False)):
            return cand
    best = trace_result.get("best_match")
    if isinstance(best, dict) and best:
        return best
    return candidates[0] if candidates else None


def _load_traceback_from_outputs(output_root, target):
    """Recover old orbit_traceback_candidates.csv files when replotting."""
    compact = _compact_target_name(target)
    parent = os.path.dirname(output_root)
    candidates = [
        os.path.join(output_root, "orbit_traceback_candidates.csv"),
        os.path.join(output_root, "orbit_traceback", "orbit_traceback_candidates.csv"),
    ]
    if parent and compact:
        candidates.extend(glob.glob(os.path.join(
            parent, f"{compact}*", "orbit_traceback_candidates.csv")))
        candidates.extend(glob.glob(os.path.join(
            parent, f"{compact}*", "orbit_traceback", "orbit_traceback_candidates.csv")))

    seen = set()
    rows = []
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        for _, r in df.iterrows():
            row = r.to_dict()
            if "rtpc" not in row and "tidal_radius_pc" in row:
                row["rtpc"] = row.get("tidal_radius_pc")
            row["_traceback_source_file"] = path
            rows.append(row)
        if rows:
            break
    if not rows:
        return None
    return {"candidates": rows, "best_match": rows[0]}


def _cluster_age_gyr(cluster_name):
    if not cluster_name:
        return np.nan
    try:
        import astro_toolbox.orbit_traceback as orbit_traceback
        target = str(cluster_name).replace("_", " ").strip().lower()
        for cl in orbit_traceback.load_hunt2023_clusters():
            name = str(cl.get("Name", "")).replace("_", " ").strip().lower()
            if name != target:
                continue
            log_age = cl.get("logAge50", np.nan)
            log_age = float(log_age)
            return 10 ** log_age / 1e9 if np.isfinite(log_age) else np.nan
    except Exception:
        return np.nan
    return np.nan


def _cluster_type(cluster_name):
    if not cluster_name:
        return ""
    try:
        import astro_toolbox.orbit_traceback as orbit_traceback
        target = str(cluster_name).replace("_", " ").strip().lower()
        for cl in orbit_traceback.load_hunt2023_clusters():
            name = str(cl.get("Name", "")).replace("_", " ").strip().lower()
            if name == target:
                return str(cl.get("Type", "") or "")
    except Exception:
        return ""
    return ""


def _split_csv_arg(value):
    if not value:
        return None
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out or None


def export_flat_outputs(output_root):
    """
    Copy nested module outputs into output_root with prefixed filenames.

    The pipeline keeps module subdirectories for reproducibility, but users often
    want one folder per source.  This flat export makes the root folder usable
    directly without opening each module directory.
    """
    rows = []
    output_root = os.path.abspath(output_root)
    for dirpath, dirnames, filenames in os.walk(output_root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__",)]
        rel_dir = os.path.relpath(dirpath, output_root)
        if rel_dir == ".":
            continue
        parts = rel_dir.split(os.sep)
        for filename in filenames:
            if filename.startswith("."):
                continue
            src = os.path.join(dirpath, filename)
            if not os.path.isfile(src):
                continue
            rel_parts = parts + [filename]
            flat_name = "__".join(safe_name(p) for p in rel_parts)
            dst = os.path.join(output_root, flat_name)
            if os.path.abspath(src) == os.path.abspath(dst):
                continue
            try:
                shutil.copy2(src, dst)
                rows.append({
                    "source_path": src,
                    "flat_file": dst,
                    "bytes": os.path.getsize(dst),
                })
            except OSError as exc:
                rows.append({
                    "source_path": src,
                    "flat_file": dst,
                    "error": f"{type(exc).__name__}: {exc}",
                })
    manifest = os.path.join(output_root, "flat_output_manifest.csv")
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return {"n_files": len(rows), "manifest": manifest}


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _load_local_sdss_pack(output_root, ra, dec):
    sdss_dir = os.path.join(output_root, "sdss")
    spec_path = os.path.join(sdss_dir, "sdss_spectrum.csv")
    phot_path = os.path.join(sdss_dir, "sdss_photometry.csv")
    prov_path = os.path.join(sdss_dir, "sdss_spectrum_provenance.json")

    spectrum = None
    photometry = {}
    if os.path.exists(spec_path):
        try:
            df = pd.read_csv(spec_path)
            wcol = "wavelength_A" if "wavelength_A" in df.columns else "wavelength"
            if wcol in df.columns and "flux" in df.columns:
                prov = {}
                if os.path.exists(prov_path):
                    try:
                        with open(prov_path, "r", encoding="utf-8") as fh:
                            prov = json.load(fh)
                    except Exception:
                        prov = {}
                spectrum = {
                    "survey": "SDSS",
                    "ra": ra,
                    "dec": dec,
                    "wavelength": pd.to_numeric(df[wcol], errors="coerce").to_numpy(float),
                    "flux": pd.to_numeric(df["flux"], errors="coerce").to_numpy(float),
                    "error": pd.to_numeric(df["error"], errors="coerce").to_numpy(float)
                    if "error" in df.columns else None,
                    "model": pd.to_numeric(df["model"], errors="coerce").to_numpy(float)
                    if "model" in df.columns else None,
                    "provenance": prov,
                    "z": prov.get("redshift", 0.0),
                    "class": prov.get("class", ""),
                    "subclass": prov.get("subclass", ""),
                    "plate": prov.get("plate", ""),
                    "mjd": prov.get("mjd", ""),
                    "fiberid": prov.get("fiberid", ""),
                    "specobjid": prov.get("specobjid", ""),
                    "run2d": prov.get("run2d", ""),
                    "programname": prov.get("programname", ""),
                    "data_release": prov.get("data_release", ""),
                }
        except Exception:
            spectrum = None

    if os.path.exists(phot_path):
        try:
            phot = pd.read_csv(phot_path)
            for _, row in phot.iterrows():
                band = str(row.get("band", "")).strip()
                if not band:
                    continue
                mag = float(row.get("mag"))
                mag_err = float(row.get("mag_err"))
                wave = float(row.get("wave_A"))
                if np.isfinite(mag) and np.isfinite(mag_err) and np.isfinite(wave):
                    photometry[band] = (mag, mag_err, wave)
        except Exception:
            photometry = {}

    if spectrum is None and not photometry:
        return None
    return {"spectrum": spectrum, "photometry": photometry, "source": "local_cache"}


def _load_local_desi_pack(output_root, ra, dec):
    desi_dir = os.path.join(output_root, "desi")
    paths = sorted(glob.glob(os.path.join(desi_dir, "spectrum_*.fits")))
    if not paths:
        return None
    try:
        from astropy.io import fits
    except Exception:
        return None

    for path in paths:
        spectrum = {}
        try:
            with fits.open(path, memmap=False) as hdul:
                names = {h.name for h in hdul}
                for band in ("B", "R", "Z"):
                    ext = f"{band}_BAND"
                    if ext not in names:
                        continue
                    data = np.asarray(hdul[ext].data, dtype=float)
                    if data.shape[0] < 3:
                        continue
                    spectrum[band] = {
                        "wavelength": data[0],
                        "flux": data[1],
                        "error": data[2],
                    }
        except Exception:
            continue
        if spectrum:
            return {
                "survey": "DESI",
                "ra": ra,
                "dec": dec,
                "spectrum": spectrum,
                "files": {"fits": path},
                "match": {},
                "source": "local_cache",
            }
    return None


def run_module(status_rows, output_root, module_name, func, output_dir):
    def _handle_timeout(signum, frame):
        raise TimeoutError(f"module timed out after {timeout_sec}s")

    timeout_sec = 180
    try:
        previous = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(timeout_sec)
        result = func()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        stale_error = os.path.join(output_dir, "error.txt")
        if os.path.exists(stale_error):
            try:
                os.remove(stale_error)
            except OSError:
                pass
        if result is None:
            add_status(status_rows, output_root, module_name, "empty",
                       output_dir=output_dir)
        else:
            add_status(status_rows, output_root, module_name, "ok",
                       output_dir=output_dir)
            save_json(os.path.join(output_dir, "summary.json"),
                      summarize_result(result))
        return result
    except Exception as exc:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        except Exception:
            pass
        ensure_dir(output_dir)
        save_text(os.path.join(output_dir, "error.txt"),
                  traceback.format_exc())
        add_status(status_rows, output_root, module_name, "error",
                   output_dir=output_dir, note=f"{type(exc).__name__}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run astro_toolbox science modules for one target."
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--ra", required=True, type=float)
    parser.add_argument("--dec", required=True, type=float)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--cluster",
        default="",
        help="Known input cluster to use for 5D/6D validation plots when traceback has no candidate.",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "6d"),
        default=os.environ.get("ASTRO_TOOLBOX_PROFILE", "full"),
        help="full runs every module; 6d keeps the modules needed for WD/RV/orbit 6D certification.",
    )
    parser.add_argument(
        "--skip-sdss",
        action="store_true",
        help="Skip online SDSS spectrum/photometry query. With --reuse-local-spectra, local sdss/*.csv is loaded instead.",
    )
    parser.add_argument(
        "--skip-desi",
        action="store_true",
        help="Skip online DESI spectrum query. With --reuse-local-spectra, local desi/spectrum_*.fits is loaded instead.",
    )
    parser.add_argument(
        "--reuse-local-spectra",
        action="store_true",
        help="When SDSS/DESI are skipped, read already-copied local spectra so downstream modules can still run.",
    )
    parser.add_argument(
        "--rv-mode",
        default=os.environ.get("ASTRO_TOOLBOX_RV_MODE", "absorption"),
        help="RV extraction mode passed to rv_correction, e.g. absorption, auto, line_core_only.",
    )
    parser.add_argument(
        "--preferred-rv-lines",
        default=os.environ.get("ASTRO_TOOLBOX_PREFERRED_RV_LINES", ""),
        help="Comma-separated line list for RV fitting, e.g. H-alpha,H-beta,H-gamma.",
    )
    parser.add_argument(
        "--rv-hahbhg",
        action="store_true",
        help="Convenience mode: use only H-alpha,H-beta,H-gamma line-core RV fitting.",
    )
    parser.add_argument(
        "--rv-error-mode",
        choices=("production", "rvopt_sandbox"),
        default=os.environ.get("ASTRO_TOOLBOX_RV_ERROR_MODE", "production"),
        help="production keeps legacy RV columns; rvopt_sandbox also writes optimized RV uncertainty columns.",
    )
    parser.add_argument(
        "--wd-param-mode",
        choices=("production", "wdopt_sandbox"),
        default=os.environ.get("ASTRO_TOOLBOX_WD_PARAM_MODE", "production"),
        help="production keeps legacy WD parameters; wdopt_sandbox also writes Gaia/parallax-prior posterior WD M/R products.",
    )
    parser.add_argument(
        "--wdopt-nwalkers",
        type=int,
        default=int(os.environ.get("ASTRO_TOOLBOX_WDOPT_NWALKERS", "32")),
        help="MCMC walkers for wdopt_sandbox.",
    )
    parser.add_argument(
        "--wdopt-nsteps",
        type=int,
        default=int(os.environ.get("ASTRO_TOOLBOX_WDOPT_NSTEPS", "1600")),
        help="MCMC steps for wdopt_sandbox.",
    )
    parser.add_argument(
        "--wdopt-burn",
        type=int,
        default=int(os.environ.get("ASTRO_TOOLBOX_WDOPT_BURN", "450")),
        help="MCMC burn-in steps for wdopt_sandbox.",
    )
    parser.add_argument(
        "--wdopt-thin",
        type=int,
        default=int(os.environ.get("ASTRO_TOOLBOX_WDOPT_THIN", "5")),
        help="MCMC thinning for wdopt_sandbox.",
    )
    parser.add_argument(
        "--wdopt-max-pixels",
        type=int,
        default=int(os.environ.get("ASTRO_TOOLBOX_WDOPT_MAX_PIXELS", "900")),
        help="Maximum spectral pixels sampled by wdopt_sandbox.",
    )
    parser.add_argument(
        "--wdopt-seed",
        type=int,
        default=int(os.environ.get("ASTRO_TOOLBOX_WDOPT_SEED", "42")),
        help="Random seed for wdopt_sandbox.",
    )
    parser.add_argument(
        "--cache-first-lightcurves",
        action="store_true",
        default=os.environ.get("ASTRO_TOOLBOX_CACHE_FIRST_LIGHTCURVES", "").strip().lower()
        in {"1", "true", "yes", "y"},
        help="Use local ZTF/TESS CSV caches before attempting online light-curve queries.",
    )
    parser.add_argument(
        "--cache-only-lightcurves",
        action="store_true",
        default=os.environ.get("ASTRO_TOOLBOX_CACHE_ONLY_LIGHTCURVES", "").strip().lower()
        in {"1", "true", "yes", "y"},
        help="For ZTF/TESS, use local CSV caches only and skip online light-curve queries when no cache is available.",
    )
    args = parser.parse_args()

    target = args.target.strip()
    source_id_text = _source_id_from_target(target)
    ra = float(args.ra)
    dec = float(args.dec)
    output_root = os.path.abspath(args.output_root)
    ensure_dir(output_root)
    input_cluster = args.cluster.strip()
    rv_mode = str(args.rv_mode or "absorption").strip()
    preferred_rv_lines = _split_csv_arg(args.preferred_rv_lines)
    if args.rv_hahbhg:
        rv_mode = "line_core_only"
        preferred_rv_lines = ["H-alpha", "H-beta", "H-gamma"]
    profile_6d = args.profile == "6d"
    if profile_6d:
        # Fast certification profile: keep spectra, photometry, SED/HRD,
        # WD fitting, RV correction, orbit traceback and six_dim.  Expensive
        # all-sky archival/light-curve searches are stubbed out.
        galah.query_spectrum = lambda *a, **k: None
        lamost.query_spectrum = lambda *a, **k: None
        hst.query_spectrum = lambda *a, **k: None
        hst.query_lightcurve = lambda *a, **k: None
        jwst.query_spectrum = lambda *a, **k: None
        jwst.query_lightcurve = lambda *a, **k: None
        spherex.query_spectrum = lambda *a, **k: None
        koa.download_and_extract_spectrum = lambda *a, **k: None
        ztf.query_lightcurve = lambda *a, **k: None
        wise.query_lightcurve = lambda *a, **k: None
        gaia_lc.query_lightcurve = lambda *a, **k: None
        tess.query_lightcurve = lambda *a, **k: None
        kepler.query_lightcurve = lambda *a, **k: None
        xray.query_xray = lambda *a, **k: None
        xray.query_heasarc_browse = lambda *a, **k: None
        period_analysis.run_period_analysis = lambda *a, **k: None
        combined_plots.plot_combined_spectra = lambda *a, **k: False
        combined_plots.plot_spectra_with_photometry = lambda *a, **k: False
        combined_plots.plot_combined_fold = lambda *a, **k: None

    target_info = {
        "target": target,
        "ra_deg": ra,
        "dec_deg": dec,
        "output_root": output_root,
        "script": os.path.abspath(__file__),
        "profile": args.profile,
        "input_cluster": input_cluster,
        "rv_mode": rv_mode,
        "preferred_rv_lines": preferred_rv_lines or [],
        "rv_error_mode": args.rv_error_mode,
        "wd_param_mode": args.wd_param_mode,
        "cache_first_lightcurves": bool(args.cache_first_lightcurves),
        "cache_only_lightcurves": bool(args.cache_only_lightcurves),
    }
    save_json(os.path.join(output_root, "target_info.json"), target_info)

    status_rows = []
    add_status(status_rows, output_root, "config", "skipped",
               note="library module")
    add_status(status_rows, output_root, "utils", "skipped",
               note="library module")
    add_status(status_rows, output_root, "gui", "skipped",
               note="GUI entry point")
    add_status(status_rows, output_root, "koa_batch", "skipped",
               note="batch workflow, not single-target science query")
    add_status(status_rows, output_root, "koa_metadata_cache", "skipped",
               note="metadata cache utility")

    results = {}

    # Spectra and photometry
    sdss_dir = ensure_dir(os.path.join(output_root, "sdss"))
    if args.skip_sdss:
        sdss_pack = _load_local_sdss_pack(output_root, ra, dec) if args.reuse_local_spectra else None
        add_status(
            status_rows,
            output_root,
            "sdss",
            "ok" if sdss_pack else "empty",
            output_dir=sdss_dir,
            note="online SDSS query skipped; reused local cache" if sdss_pack else "online SDSS query skipped; no local cache",
        )
    else:
        def _run_sdss():
            res = sdss.query_spectrum(ra, dec)
            phot = sdss.get_photometry(ra, dec)
            if res is not None:
                sdss.plot_spectrum(res, os.path.join(sdss_dir, "sdss_spectrum.png"))
                sdss.save_spectrum_csv(res, sdss_dir)
            if phot:
                sdss.save_photometry_csv(phot, sdss_dir)
            return {"spectrum": res, "photometry": phot}
        sdss_pack = run_module(status_rows, output_root, "sdss", _run_sdss, sdss_dir)
    results["SDSS_spectrum"] = sdss_pack.get("spectrum") if sdss_pack else None
    sdss_phot = sdss_pack.get("photometry") if sdss_pack else {}

    desi_dir = ensure_dir(os.path.join(output_root, "desi"))
    if args.skip_desi:
        desi_res = _load_local_desi_pack(output_root, ra, dec) if args.reuse_local_spectra else None
        add_status(
            status_rows,
            output_root,
            "desi",
            "ok" if desi_res else "empty",
            output_dir=desi_dir,
            note="online DESI query skipped; reused local cache" if desi_res else "online DESI query skipped; no local cache",
        )
    else:
        def _run_desi():
            cache_dir = os.environ.get("ASTRO_TOOLBOX_DESI_CACHE_DIR")
            if cache_dir:
                tool = DESITool(output_dir=desi_dir, cache_dir=cache_dir, log_func=print)
            else:
                tool = DESITool(output_dir=desi_dir, log_func=print)
            if profile_6d and os.environ.get("ASTRO_TOOLBOX_DESI_CACHE_ONLY"):
                match = tool.index.query(ra, dec)
                if match is None:
                    return None
                _, filename = tool.downloader.get_coadd_url(
                    match["survey"], match["program"], match["healpix"]
                )
                coadd_path = os.path.join(tool.downloader.cache_dir, filename)
                if not (os.path.exists(coadd_path) and os.path.getsize(coadd_path) > 0):
                    return None
            return tool.process_single(ra, dec, show_plot=False, save_fits=True, save_png=True)
        desi_res = run_module(status_rows, output_root, "desi", _run_desi, desi_dir)
    results["DESI"] = desi_res

    galah_dir = ensure_dir(os.path.join(output_root, "galah"))
    def _run_galah():
        res = galah.query_spectrum(ra, dec)
        if res is not None:
            galah.save_csv(res, galah_dir)
        return res
    results["GALAH"] = run_module(status_rows, output_root, "galah", _run_galah, galah_dir)

    lamost_dir = ensure_dir(os.path.join(output_root, "lamost"))
    def _run_lamost():
        res = lamost.query_spectrum(ra, dec)
        if res is not None:
            lamost.save_csv(res, lamost_dir)
            if "wavelength" in res:
                lamost.plot_spectrum(res, os.path.join(lamost_dir, "lamost_spectrum.png"))
        return res
    results["LAMOST"] = run_module(status_rows, output_root, "lamost", _run_lamost, lamost_dir)

    hst_dir = ensure_dir(os.path.join(output_root, "hst"))
    def _run_hst():
        res = {}
        spec = hst.query_spectrum(ra, dec)
        if spec is not None:
            hst.plot_spectrum(spec, os.path.join(hst_dir, "hst_spectrum.png"))
            hst.save_spectrum_csv(spec, hst_dir)
            res["spectrum"] = spec
        lc = hst.query_lightcurve(ra, dec)
        if lc is not None:
            hst.plot_lightcurve(lc, os.path.join(hst_dir, "hst_lightcurve.png"))
            hst.save_lightcurve_csv(lc, hst_dir)
            res["lightcurve"] = lc
        return res or None
    hst_pack = run_module(status_rows, output_root, "hst", _run_hst, hst_dir)
    results["HST_spectrum"] = hst_pack.get("spectrum") if hst_pack else None
    results["HST_lightcurve"] = hst_pack.get("lightcurve") if hst_pack else None

    jwst_dir = ensure_dir(os.path.join(output_root, "jwst"))
    def _run_jwst():
        res = {}
        spec = jwst.query_spectrum(ra, dec)
        if spec is not None:
            jwst.plot_spectrum(spec, os.path.join(jwst_dir, "jwst_spectrum.png"))
            jwst.save_spectrum_csv(spec, jwst_dir)
            res["spectrum"] = spec
        lc = jwst.query_lightcurve(ra, dec)
        if lc is not None:
            jwst.plot_lightcurve(lc, os.path.join(jwst_dir, "jwst_lightcurve.png"))
            jwst.save_lightcurve_csv(lc, jwst_dir)
            res["lightcurve"] = lc
        return res or None
    jwst_pack = run_module(status_rows, output_root, "jwst", _run_jwst, jwst_dir)
    results["JWST_spectrum"] = jwst_pack.get("spectrum") if jwst_pack else None
    results["JWST_lightcurve"] = jwst_pack.get("lightcurve") if jwst_pack else None

    spherex_dir = ensure_dir(os.path.join(output_root, "spherex"))
    def _run_spherex():
        res = {}
        spec = spherex.query_spectrum(
            ra,
            dec,
            timeout=(10, 30),
            allow_cutout_fallback=False,
        )
        if spec is not None:
            spherex.plot_spectrum(spec, os.path.join(spherex_dir, "spherex_spectrum.png"))
            spherex.save_spectrum_csv(spec, spherex_dir)
            res["spectrum"] = spec
        return res or None
    spherex_pack = run_module(status_rows, output_root, "spherex", _run_spherex, spherex_dir)
    results["SPHEREx"] = spherex_pack.get("spectrum") if spherex_pack else None

    koa_dir = ensure_dir(os.path.join(output_root, "koa"))
    koa_work = ensure_dir(os.path.join(koa_dir, "work"))
    def _run_koa():
        res = koa.download_and_extract_spectrum(
            ra=ra,
            dec=dec,
            target=target.replace(" ", ""),
            instruments=("lris",),
            work_dir=koa_work,
            output_dir=koa_dir,
            download=True,
            calibfile=False,
            lev0file=True,
            lev1file=False,
            row_limit=2,
            auto_pypeit=True,
            pypeit_setup_only=True,
        )
        save_json(os.path.join(koa_dir, "koa_result.json"), summarize_result(res))
        return res
    koa_res = run_module(status_rows, output_root, "koa", _run_koa, koa_dir)
    if isinstance(koa_res, dict) and "wavelength" in koa_res:
        results["KOA_spectrum"] = koa_res
    else:
        results["KOA_spectrum"] = None

    # Light curves
    ztf_dir = ensure_dir(os.path.join(output_root, "ztf"))
    cached_ztf_first = (
        _load_cached_ztf_lightcurve(target, output_root, ztf_dir, ra, dec)
        if args.cache_first_lightcurves else None
    )
    if cached_ztf_first is not None:
        results["ZTF_lightcurve"] = cached_ztf_first
        add_status(status_rows, output_root, "ztf", "ok",
                   output_dir=ztf_dir,
                   note=f"cache-first local CSV: {cached_ztf_first.get('cache_source', '')}",
                   files=[os.path.join(ztf_dir, "ztf_lightcurve.csv"),
                          os.path.join(ztf_dir, "ztf_lightcurve.png")])
    elif args.cache_only_lightcurves:
        results["ZTF_lightcurve"] = None
        add_status(status_rows, output_root, "ztf", "empty",
                   output_dir=ztf_dir,
                   note="cache-only light-curve mode; no local ZTF cache")
    else:
        def _run_ztf():
            res = ztf.query_lightcurve(ra, dec)
            if res is not None:
                ztf.plot_lightcurve(res, os.path.join(ztf_dir, "ztf_lightcurve.png"))
                ztf.save_csv(res, ztf_dir)
                save_text(os.path.join(ztf_dir, "ztf_web_url.txt"),
                          res.get("web_url", "") + "\n")
            return res
        results["ZTF_lightcurve"] = run_module(status_rows, output_root, "ztf", _run_ztf, ztf_dir)
    if results["ZTF_lightcurve"] is None:
        cached_ztf = _load_cached_ztf_lightcurve(target, output_root, ztf_dir, ra, dec)
        if cached_ztf is not None:
            results["ZTF_lightcurve"] = cached_ztf
            add_status(status_rows, output_root, "ztf_cache", "ok",
                       output_dir=ztf_dir,
                       note=f"loaded cached CSV: {cached_ztf.get('cache_source', '')}",
                       files=[os.path.join(ztf_dir, "ztf_lightcurve.csv"),
                              os.path.join(ztf_dir, "ztf_lightcurve.png")])

    wise_dir = ensure_dir(os.path.join(output_root, "wise"))
    def _run_wise():
        res = {}
        phot = wise.get_photometry(ra, dec)
        if phot:
            wise.save_photometry_csv(phot, wise_dir)
            res["photometry"] = phot
        lc = wise.query_lightcurve(ra, dec)
        if lc is not None:
            wise.plot_lightcurve(lc, os.path.join(wise_dir, "wise_lightcurve.png"))
            wise.save_lightcurve_csv(lc, wise_dir)
            res["lightcurve"] = lc
        return res or None
    wise_pack = run_module(status_rows, output_root, "wise", _run_wise, wise_dir)
    wise_phot = wise_pack.get("photometry") if wise_pack else {}
    results["WISE_lightcurve"] = wise_pack.get("lightcurve") if wise_pack else None

    gaia_lc_dir = ensure_dir(os.path.join(output_root, "gaia_lc"))
    def _run_gaia_lc():
        res = gaia_lc.query_lightcurve(ra, dec)
        if res is not None:
            gaia_lc.plot_lightcurve(res, os.path.join(gaia_lc_dir, "gaia_lightcurve.png"))
            gaia_lc.save_csv(res, gaia_lc_dir)
        return res
    results["Gaia_lightcurve"] = run_module(status_rows, output_root, "gaia_lc", _run_gaia_lc, gaia_lc_dir)

    tess_dir = ensure_dir(os.path.join(output_root, "tess"))
    cached_tess_first = (
        _load_cached_tess_lightcurve(target, output_root, tess_dir, ra, dec)
        if args.cache_first_lightcurves else None
    )
    if cached_tess_first is not None:
        results["TESS"] = cached_tess_first
        add_status(status_rows, output_root, "tess", "ok",
                   output_dir=tess_dir,
                   note=f"cache-first local CSV: {cached_tess_first.get('cache_source', '')}",
                   files=[os.path.join(tess_dir, "tess_lightcurve.csv"),
                          os.path.join(tess_dir, "tess_lightcurve.png")])
    elif args.cache_only_lightcurves:
        results["TESS"] = None
        add_status(status_rows, output_root, "tess", "empty",
                   output_dir=tess_dir,
                   note="cache-only light-curve mode; no local TESS cache")
    else:
        def _run_tess():
            res = tess.query_lightcurve(ra, dec)
            if res is not None:
                tess.plot_lightcurve(res, os.path.join(tess_dir, "tess_lightcurve.png"))
                tess.save_csv(res, tess_dir)
                lk_period = tess.analyze_period_lightkurve(res, tess_dir)
                if lk_period is not None:
                    res["lightkurve_period_analysis"] = lk_period
            return res
        results["TESS"] = run_module(status_rows, output_root, "tess", _run_tess, tess_dir)
    if results["TESS"] is None:
        cached_tess = _load_cached_tess_lightcurve(target, output_root, tess_dir, ra, dec)
        if cached_tess is not None:
            results["TESS"] = cached_tess
            add_status(status_rows, output_root, "tess_cache", "ok",
                       output_dir=tess_dir,
                       note=f"loaded cached CSV: {cached_tess.get('cache_source', '')}",
                       files=[os.path.join(tess_dir, "tess_lightcurve.csv"),
                              os.path.join(tess_dir, "tess_lightcurve.png")])

    kepler_dir = ensure_dir(os.path.join(output_root, "kepler"))
    def _run_kepler():
        pack = {}
        kep = kepler.query_lightcurve(ra, dec, mission="Kepler")
        if kep is not None:
            kepler.plot_lightcurve(kep, os.path.join(kepler_dir, "kepler_lightcurve.png"))
            kepler.save_csv(kep, kepler_dir)
            pack["Kepler"] = kep
        k2 = kepler.query_lightcurve(ra, dec, mission="K2")
        if k2 is not None:
            kepler.plot_lightcurve(k2, os.path.join(kepler_dir, "k2_lightcurve.png"))
            save_json(os.path.join(kepler_dir, "k2_summary.json"), summarize_result(k2))
            pack["K2"] = k2
        return pack or None
    kepler_pack = run_module(status_rows, output_root, "kepler", _run_kepler, kepler_dir)
    results["Kepler/K2"] = None
    if kepler_pack:
        results["Kepler/K2"] = kepler_pack.get("Kepler") or kepler_pack.get("K2")

    # Photometry
    galex_dir = ensure_dir(os.path.join(output_root, "galex"))
    def _run_galex():
        res = galex.get_photometry(ra, dec)
        if res:
            galex.save_csv(res, galex_dir)
        return res or None
    galex_phot = run_module(status_rows, output_root, "galex", _run_galex, galex_dir) or {}

    twomass_dir = ensure_dir(os.path.join(output_root, "twomass"))
    def _run_twomass():
        res = twomass.get_photometry(ra, dec)
        if res:
            twomass.save_csv(res, twomass_dir)
        return res or None
    twomass_phot = run_module(status_rows, output_root, "twomass", _run_twomass, twomass_dir) or {}

    xray_dir = ensure_dir(os.path.join(output_root, "xray"))
    def _run_xray():
        xr = xray.query_xray(ra, dec)
        hx = xray.query_heasarc_browse(ra, dec)
        if xr:
            xray.save_csv(xr, xray_dir)
        if hx:
            xray.save_heasarc_csv(hx, xray_dir)
        analysis = xray.analyze_xray(
            xray_result=xr,
            heasarc_result=hx,
            results=results,
            ra=ra,
            dec=dec,
        )
        xray.save_analysis(analysis, xray_dir)
        return {"catalogs": xr, "heasarc": hx, "analysis": analysis}
    xray_pack = run_module(status_rows, output_root, "xray", _run_xray, xray_dir)

    # SED and HRD
    sed_dir = ensure_dir(os.path.join(output_root, "sed"))
    def _run_sed():
        fitter = sed.SEDFitter(ra, dec)
        fitter.load_photometry(galex_phot, sdss_phot, twomass_phot, wise_phot)
        fitter.collect_photometry(
            include_galex=not bool(galex_phot),
            include_sdss=not bool(sdss_phot),
            include_gaia=True,
            include_2mass=not bool(twomass_phot),
            include_wise=not bool(wise_phot),
            include_spherex=False,
        )
        fitter.apply_extinction()
        fitter.plot(os.path.join(sed_dir, "sed.png"))
        fitter.save_csv(sed_dir)
        fitter.save_diagnostics(sed_dir)
        return fitter
    sed_fitter = run_module(status_rows, output_root, "sed", _run_sed, sed_dir)
    results["SED"] = sed_fitter

    hrd_dir = ensure_dir(os.path.join(output_root, "hr_diagram"))
    hr_params = None
    def _run_hrd():
        nonlocal_hr = hr_diagram.HRDiagram()
        nonlocal_hr.plot_single(ra, dec, save_path=os.path.join(hrd_dir, "hr_diagram.png"))
        params = nonlocal_hr._query_gaia_params(ra, dec)
        if params is not None:
            hr_diagram.save_csv(params, hrd_dir)
            hr_diagram.save_analysis_report(params, hrd_dir)
        return params
    hr_params = run_module(status_rows, output_root, "hr_diagram", _run_hrd, hrd_dir)

    # Diagnostics and combined plots
    diag_dir = ensure_dir(os.path.join(output_root, "diagnostics"))
    def _run_diagnostics():
        spec_diag = diagnostics.analyze_all_spectra(results)
        diagnostics.save_spectral_diagnostics(spec_diag, diag_dir)
        return spec_diag
    spectral_diag = run_module(status_rows, output_root, "diagnostics", _run_diagnostics, diag_dir)

    combined_dir = ensure_dir(os.path.join(output_root, "combined_plots"))
    def _run_combined():
        made = []
        p1 = os.path.join(combined_dir, "combined_spectra.png")
        if combined_plots.plot_combined_spectra(results, save_path=p1, ra=ra, dec=dec):
            made.append(p1)
        p2 = os.path.join(combined_dir, "spectra_with_photometry.png")
        if combined_plots.plot_spectra_with_photometry(results, save_path=p2, ra=ra, dec=dec):
            made.append(p2)
        return {"files": made}
    run_module(status_rows, output_root, "combined_plots_pre", _run_combined, combined_dir)

    # Period analysis
    pa_dir = ensure_dir(os.path.join(output_root, "period_analysis"))
    def _run_period():
        pa = period_analysis.run_period_analysis(results, pa_dir, ra=ra, dec=dec, title_prefix=target)
        if pa is not None:
            period_analysis.save_csv(pa, pa_dir)
        return pa
    pa_res = run_module(status_rows, output_root, "period_analysis", _run_period, pa_dir)
    results["period_analysis"] = pa_res
    if results.get("ZTF_lightcurve") is not None:
        def _run_ztf_aov_reference():
            path = period_analysis.plot_ztf_aov_reference(
                results["ZTF_lightcurve"],
                pa_dir,
                ra=ra,
                dec=dec,
                target_name=target,
                filename="ztf_aov_reference.png",
            )
            return {"file": path} if path else None
        run_module(status_rows, output_root, "ztf_aov_reference",
                   _run_ztf_aov_reference, pa_dir)

    def _run_combined_fold():
        path = os.path.join(combined_dir, "combined_fold.png")
        fig = combined_plots.plot_combined_fold(results, save_path=path, ra=ra, dec=dec)
        return {"file": path} if fig else None
    run_module(status_rows, output_root, "combined_plots_fold", _run_combined_fold, combined_dir)

    # WD fitting
    wd_dir = ensure_dir(os.path.join(output_root, "wd_fitting"))
    wd_out = None
    wd_phys = None
    chosen_name, chosen_spec = None, None
    if chosen_spec is None:
        def _run_wd():
            selection = fit_best_wd_spectrum(
                results, wd_dir, hr_params=hr_params, sed_fitter=sed_fitter)
            if not selection or selection[2] is None:
                return None
            return selection

        wd_out = run_module(status_rows, output_root, "wd_fitting", _run_wd, wd_dir)
        if wd_out and isinstance(wd_out, tuple) and len(wd_out) == 3:
            chosen_name, chosen_spec, selected_pack = wd_out
            wd_out = selected_pack
            wd_phys = (wd_out or {}).get("physical_params")
    if chosen_spec is None:
        add_status(status_rows, output_root, "wd_fitting_selection", "skipped",
                   output_dir=wd_dir, note="no usable optical WD spectrum")
    elif wd_out is not None:
        add_status(status_rows, output_root, "wd_fitting_selection", "ok",
                   output_dir=wd_dir,
                   note=f"selected {chosen_name} for WD params/RV correction",
                   files=[os.path.join(wd_dir, "wd_spectrum_comparison.csv")])

    wdopt_result = None
    wdopt_dir = ensure_dir(os.path.join(output_root, "wdopt_sandbox"))
    if args.wd_param_mode == "wdopt_sandbox" and chosen_spec is not None:
        def _run_wdopt():
            cls = ((wd_out or {}).get("classification") or {}).get("spectral_type", "DA")
            specclass = "DB" if str(cls).upper().startswith("DB") else "DA"
            gaia_mass = None
            gaia_radius = None
            gaia_logg = None
            gaia_teff = None
            if isinstance(wd_phys, dict):
                gaia_mass = wd_phys.get("gaia_hr_mass")
                gaia_radius = wd_phys.get("gaia_hr_radius_rsun")
                gaia_logg = wd_phys.get("gaia_hr_logg")
                gaia_teff = wd_phys.get("gaia_hr_teff")
            if not gaia_mass and hr_params:
                gaia_mass = hr_params.get("wd_mass_msun")
            if not gaia_logg and hr_params:
                gaia_logg = hr_params.get("wd_logg")
            if not gaia_teff and hr_params:
                gaia_teff = hr_params.get("wd_teff_k")
            if gaia_radius is None and gaia_mass is not None and gaia_logg is not None:
                try:
                    gaia_radius = wd_fitting.compute_wd_radius(float(gaia_mass), float(gaia_logg))
                except Exception:
                    gaia_radius = None
            initial = ((wd_out or {}).get("balmer_fit") or
                       (wd_out or {}).get("continuum_fit") or {})
            res = wd_fitting.fit_wd_mcmc_nn(
                chosen_spec["wavelength"],
                chosen_spec["flux"],
                chosen_spec.get("error"),
                specclass=specclass,
                initial=initial,
                parallax_mas=(hr_params or {}).get("Plx"),
                parallax_err_mas=(hr_params or {}).get("e_Plx"),
                teff_prior=initial.get("teff") if initial else None,
                teff_prior_sigma=3500.0,
                gaia_teff_prior=gaia_teff,
                gaia_teff_prior_sigma=2500.0,
                gaia_logg_prior=gaia_logg,
                gaia_logg_prior_sigma=0.20,
                gaia_mass_prior=gaia_mass,
                gaia_mass_prior_sigma=0.12,
                gaia_radius_prior=gaia_radius,
                gaia_radius_prior_sigma=0.0025,
                nwalkers=args.wdopt_nwalkers,
                nsteps=args.wdopt_nsteps,
                burn=args.wdopt_burn,
                thin=args.wdopt_thin,
                random_seed=args.wdopt_seed,
                max_pixels=args.wdopt_max_pixels,
                model_grid="auto",
                sampler="auto",
                output_dir=wdopt_dir,
            )
            if not isinstance(res, dict) or res.get("status") != "ok":
                return res
            phys_row = {
                "source_id": target,
                "cluster": input_cluster,
                "spectrum_source": chosen_name,
                "wd_param_mode": args.wd_param_mode,
                "sampler_backend": res.get("sampler_backend"),
                "model_grid": res.get("model_grid"),
                "teff": res.get("teff"),
                "teff_err": res.get("teff_err"),
                "logg": res.get("logg"),
                "logg_err": res.get("logg_err"),
                "mass_msun_preferred": res.get("mass_msun_preferred"),
                "mass_msun_preferred_err": res.get("mass_msun_preferred_err"),
                "radius_rsun_preferred": res.get("radius_rsun_preferred"),
                "radius_rsun_preferred_err": res.get("radius_rsun_preferred_err"),
                "v_grav_preferred_kms": res.get("v_grav_preferred_kms"),
                "v_grav_preferred_err": res.get("v_grav_preferred_err"),
                "preferred_physical_source": res.get("preferred_physical_source"),
                "mass_msun_mr": res.get("mass_msun_mr"),
                "mass_msun_mr_err": res.get("mass_msun_mr_err"),
                "radius_rsun_mr": res.get("radius_rsun_mr"),
                "radius_rsun_mr_err": res.get("radius_rsun_mr_err"),
                "v_grav_mr_kms": res.get("v_grav_mr_kms"),
                "v_grav_mr_err": res.get("v_grav_mr_err"),
                "scale_radius_rsun": res.get("scale_radius_rsun"),
                "scale_radius_rsun_err": res.get("scale_radius_rsun_err"),
                "mass_msun_from_scale_logg": res.get("mass_msun_from_scale_logg"),
                "mass_msun_from_scale_logg_err": res.get("mass_msun_from_scale_logg_err"),
                "v_grav_scale_logg_kms": res.get("v_grav_scale_logg_kms"),
                "v_grav_scale_logg_err": res.get("v_grav_scale_logg_err"),
                "cooling_age_gyr": res.get("cooling_age_gyr"),
                "cooling_age_gyr_err": res.get("cooling_age_gyr_err"),
                "total_age_gyr": res.get("total_age_gyr"),
                "total_age_gyr_err": res.get("total_age_gyr_err"),
                "total_age_with_ms_gyr": res.get("total_age_with_ms_gyr"),
                "total_age_with_ms_gyr_err": res.get("total_age_with_ms_gyr_err"),
                "m_progenitor_msun": res.get("m_progenitor_msun"),
                "m_progenitor_msun_err": res.get("m_progenitor_msun_err"),
                "ms_lifetime_gyr": res.get("ms_lifetime_gyr"),
                "ms_lifetime_gyr_err": res.get("ms_lifetime_gyr_err"),
                "age_source": res.get("age_source"),
                "gaia_prior_mass": gaia_mass,
                "gaia_prior_radius": gaia_radius,
                "gaia_prior_logg": gaia_logg,
                "gaia_prior_teff": gaia_teff,
            }
            phys_path = os.path.join(wdopt_dir, "wdopt_sandbox_physical_params.csv")
            pd.DataFrame([phys_row]).to_csv(phys_path, index=False)
            res["physical_params_file"] = phys_path
            return res
        wdopt_result = run_module(status_rows, output_root, "wdopt_sandbox",
                                  _run_wdopt, wdopt_dir)
    elif args.wd_param_mode == "wdopt_sandbox":
        add_status(status_rows, output_root, "wdopt_sandbox", "skipped",
                   output_dir=wdopt_dir, note="need selected WD spectrum")

    # RV fitting and correction
    rv_dir = ensure_dir(os.path.join(output_root, "rv"))
    def _run_rv():
        rv_report = rv_fitting.run_rv_analysis(results, output_dir=rv_dir, ra=ra, dec=dec)
        return rv_report
    rv_report = run_module(status_rows, output_root, "rv_fitting", _run_rv, rv_dir)

    rvc_dir = ensure_dir(os.path.join(output_root, "rv_correction"))
    if chosen_spec is None or wd_phys is None:
        add_status(status_rows, output_root, "rv_correction", "skipped",
                   output_dir=rvc_dir, note="need usable spectrum and WD params")
        rv_corr = None
    else:
        def _run_rvc():
            return rv_correction.run_rv_correction(
                chosen_spec["wavelength"],
                chosen_spec["flux"],
                chosen_spec.get("error"),
                physical_params=wd_phys,
                survey_name=chosen_name,
                output_dir=rvc_dir,
                ra=ra,
                dec=dec,
                rv_mode=rv_mode,
                preferred_lines=preferred_rv_lines,
            )
        rv_corr = run_module(status_rows, output_root, "rv_correction", _run_rvc, rvc_dir)
    if wd_out is not None and wd_phys is not None:
        rv_variants_path = None
        def _run_rv_variants():
            return save_rv_physical_variants(
                rv_report, rv_corr, wd_out, wd_phys, rvc_dir,
                selected_source=chosen_name or "")
        rv_variants_path = run_module(status_rows, output_root, "rv_wd_variants",
                                      _run_rv_variants, rvc_dir)
    else:
        rv_variants_path = None
    wdopt_rv_result = None
    if (args.wd_param_mode == "wdopt_sandbox"
            and isinstance(wdopt_result, dict)
            and wdopt_result.get("status") == "ok"
            and rv_corr is not None):
        def _run_wdopt_rv():
            rv_obs = float(rv_corr.get("rv_obs", np.nan))
            rv_obs_err = float(rv_corr.get("rv_obs_err", np.nan))
            vgrav = float(wdopt_result.get("v_grav_preferred_kms", np.nan))
            vgrav_err = float(wdopt_result.get("v_grav_preferred_err", np.nan))
            rv_true = rv_obs - vgrav if np.isfinite(rv_obs + vgrav) else np.nan
            rv_true_err = (
                float(np.hypot(rv_obs_err, vgrav_err))
                if np.isfinite(rv_obs_err + vgrav_err) else np.nan
            )
            row = {
                "source_id": target,
                "cluster": input_cluster,
                "rv_obs_kms": rv_obs,
                "rv_obs_err_kms": rv_obs_err,
                "production_v_grav_kms": rv_corr.get("v_grav"),
                "production_v_grav_err_kms": rv_corr.get("v_grav_err"),
                "production_rv_true_kms": rv_corr.get("rv_true"),
                "production_rv_true_err_kms": rv_corr.get("rv_true_err"),
                "wdopt_v_grav_kms": vgrav,
                "wdopt_v_grav_err_kms": vgrav_err,
                "wdopt_rv_true_kms": rv_true,
                "wdopt_rv_true_err_kms": rv_true_err,
                "wdopt_mass_msun": wdopt_result.get("mass_msun_preferred"),
                "wdopt_mass_msun_err": wdopt_result.get("mass_msun_preferred_err"),
                "wdopt_radius_rsun": wdopt_result.get("radius_rsun_preferred"),
                "wdopt_radius_rsun_err": wdopt_result.get("radius_rsun_preferred_err"),
                "wdopt_teff": wdopt_result.get("teff"),
                "wdopt_teff_err": wdopt_result.get("teff_err"),
                "wdopt_logg": wdopt_result.get("logg"),
                "wdopt_logg_err": wdopt_result.get("logg_err"),
                "wdopt_cooling_age_gyr": wdopt_result.get("cooling_age_gyr"),
                "wdopt_cooling_age_gyr_err": wdopt_result.get("cooling_age_gyr_err"),
                "wdopt_total_age_gyr": wdopt_result.get("total_age_gyr"),
                "wdopt_total_age_gyr_err": wdopt_result.get("total_age_gyr_err"),
                "wdopt_total_age_with_ms_gyr": wdopt_result.get("total_age_with_ms_gyr"),
                "wdopt_total_age_with_ms_gyr_err": wdopt_result.get("total_age_with_ms_gyr_err"),
                "wdopt_m_progenitor_msun": wdopt_result.get("m_progenitor_msun"),
                "wdopt_m_progenitor_msun_err": wdopt_result.get("m_progenitor_msun_err"),
                "wdopt_ms_lifetime_gyr": wdopt_result.get("ms_lifetime_gyr"),
                "wdopt_ms_lifetime_gyr_err": wdopt_result.get("ms_lifetime_gyr_err"),
                "preferred_physical_source": wdopt_result.get("preferred_physical_source"),
                "sampler_backend": wdopt_result.get("sampler_backend"),
            }
            path = os.path.join(wdopt_dir, "wdopt_rv_recomputed.csv")
            pd.DataFrame([row]).to_csv(path, index=False)
            save_json(os.path.join(wdopt_dir, "wdopt_rv_recomputed.json"), row)
            return row
        wdopt_rv_result = run_module(status_rows, output_root, "wdopt_rv_recomputed",
                                     _run_wdopt_rv, wdopt_dir)
    rvopt_result = None
    if args.rv_error_mode == "rvopt_sandbox" and rv_corr is not None:
        def _run_rvopt():
            cluster_rv, cluster_rv_err, cluster_rv_source, cluster_rv_n = (
                six_dim._resolve_cluster_rv(input_cluster)
            )
            opt = rv_correction.rvopt_uncertainty_layer(
                rv_corr,
                cluster_rv=cluster_rv,
                cluster_rv_err=cluster_rv_err,
                cluster_intrinsic_rv_dispersion=six_dim.DEFAULT_CLUSTER_RV_DISPERSION_KMS,
                wd_physical_params=wd_phys,
                rv_physical_variants=_load_rv_physical_variants_rows(rv_variants_path),
            )
            opt.update({
                "source_id": target,
                "cluster": input_cluster,
                "cluster_rv": cluster_rv,
                "cluster_rv_err": cluster_rv_err,
                "cluster_rv_source": cluster_rv_source,
                "cluster_rv_n": cluster_rv_n,
                "rv_error_mode": args.rv_error_mode,
                "production_rv_obs": rv_corr.get("rv_obs"),
                "production_v_grav": rv_corr.get("v_grav"),
                "production_rv_true": rv_corr.get("rv_true"),
            })
            path = os.path.join(rvc_dir, "rvopt_sandbox_uncertainty.csv")
            pd.DataFrame([opt]).to_csv(path, index=False)
            save_json(os.path.join(rvc_dir, "rvopt_sandbox_uncertainty.json"), opt)
            opt["file"] = path
            return opt
        rvopt_result = run_module(status_rows, output_root, "rvopt_sandbox",
                                  _run_rvopt, rvc_dir)

    # Cooling age
    cool_dir = ensure_dir(os.path.join(output_root, "cooling_age"))
    def _run_cooling():
        gaia_phot = cooling_age.get_gaia_photometry(ra, dec)
        return cooling_age.run_cooling_age_analysis(
            ra,
            dec,
            cluster_name=input_cluster,
            output_dir=cool_dir,
            gaia_phot=gaia_phot,
        )
    cool_res = run_module(status_rows, output_root, "cooling_age", _run_cooling, cool_dir)

    # Orbit traceback
    trace_dir = ensure_dir(os.path.join(output_root, "orbit_traceback"))
    if rv_report is None:
        add_status(status_rows, output_root, "orbit_traceback", "skipped",
                   output_dir=trace_dir, note="no RV report")
        trace_res = None
    else:
        def _run_trace():
            return cooling_age.orbit_traceback.run_traceback_analysis(results, rv_report, output_dir=trace_dir, ra=ra, dec=dec)  # type: ignore[attr-defined]
        try:
            import astro_toolbox.orbit_traceback as orbit_traceback
            trace_res = run_module(
                status_rows,
                output_root,
                "orbit_traceback",
                lambda: orbit_traceback.run_traceback_analysis(results, rv_report, output_dir=trace_dir, ra=ra, dec=dec),
                trace_dir,
            )
        except Exception as exc:
            save_text(os.path.join(trace_dir, "error.txt"), traceback.format_exc())
            add_status(status_rows, output_root, "orbit_traceback", "error",
                       output_dir=trace_dir, note=f"{type(exc).__name__}: {exc}")
            trace_res = None

    # six_dim summary plots
    sixd_dir = ensure_dir(os.path.join(output_root, "six_dim"))
    def _run_sixd():
        made = []
        gaia_astrom = None
        try:
            import astro_toolbox.orbit_traceback as orbit_traceback
            gaia_astrom = orbit_traceback.get_gaia_astrometry(ra, dec)
        except Exception:
            gaia_astrom = None
        trace_source = trace_res if isinstance(trace_res, dict) else None
        if trace_source is None or not (trace_source.get("candidates") or trace_source.get("best_match")):
            trace_source = _load_traceback_from_outputs(output_root, target)
        trace_best = _select_traceback_candidate(trace_source)
        trace_cluster = str((trace_best or {}).get("cluster_name", "") or "")
        trace_within_tidal = bool((trace_best or {}).get("within_tidal", False))
        trace_match_path = "backtrack" if trace_within_tidal else (
            "traceback_candidate" if trace_cluster else ""
        )
        trace_membership = (
            "backtrack_only" if trace_within_tidal else
            ("5D traceback candidate" if trace_cluster else "")
        )
        def _trace_num(key):
            try:
                value = float((trace_best or {}).get(key, np.nan))
                return value if np.isfinite(value) else np.nan
            except Exception:
                return np.nan

        def _finite_num(value):
            try:
                value = float(value)
                return value if np.isfinite(value) else np.nan
            except Exception:
                return np.nan

        trace_age_myr = _trace_num("cluster_age_myr")
        trace_age_gyr = trace_age_myr / 1000.0 if np.isfinite(trace_age_myr) else np.nan
        plot_cluster = trace_cluster or input_cluster
        plot_cluster_age_gyr = (
            trace_age_gyr if np.isfinite(trace_age_gyr)
            else _cluster_age_gyr(plot_cluster)
        )
        plot_cluster_type = _cluster_type(plot_cluster)
        plot_match_path = trace_match_path or (
            "input_cluster_5d" if input_cluster else ""
        )
        plot_membership = trace_membership or (
            "input_cluster_5d_candidate" if input_cluster else ""
        )
        trace_note = ""
        if trace_cluster:
            trace_note = (
                f"traceback best={trace_cluster}, "
                f"min_sep={_trace_num('min_sep_pc'):.1f} pc, "
                f"rt={_trace_num('rtpc'):.1f} pc"
            )
        elif input_cluster:
            trace_note = (
                f"input cluster fallback={input_cluster}; "
                "traceback did not provide a cluster candidate for plotting"
            )
        row = {
            "ra": ra,
            "dec": dec,
            "source_id": (
                source_id_text
                or (gaia_astrom or {}).get("source_id")
                or (hr_params or {}).get("source_id", "")
            ),
            "cluster": plot_cluster,
            "input_cluster": input_cluster,
            "traceback_best_cluster": trace_cluster,
            "sixdim_plot_cluster": plot_cluster,
            "sixdim_cluster_choice_basis": (
                "traceback" if trace_cluster
                else ("input_cluster_fallback" if input_cluster else "")
            ),
            "membership": plot_membership,
            "match_path": plot_match_path,
            "match_note": trace_note,
            "orbit_within_tidal": trace_within_tidal,
            "phot_g_mean_mag": hr_params.get("Gmag") if hr_params else np.nan,
            "bp_rp": hr_params.get("BP_RP") if hr_params else np.nan,
            "M_G": hr_params.get("M_G") if hr_params else np.nan,
            "parallax": (
                (gaia_astrom or {}).get("Plx")
                if gaia_astrom else (hr_params.get("Plx") if hr_params else np.nan)
            ),
            "e_parallax": (
                (gaia_astrom or {}).get("e_Plx")
                if gaia_astrom else (hr_params.get("e_Plx") if hr_params else np.nan)
            ),
            "pmRA": (gaia_astrom or {}).get("pmRA", np.nan),
            "e_pmRA": (gaia_astrom or {}).get("e_pmRA", np.nan),
            "pmDE": (gaia_astrom or {}).get("pmDE", np.nan),
            "e_pmDE": (gaia_astrom or {}).get("e_pmDE", np.nan),
            "RUWE": (
                (gaia_astrom or {}).get("RUWE")
                if gaia_astrom else (hr_params.get("RUWE") if hr_params else np.nan)
            ),
            "teff": (wd_phys or {}).get("teff", np.nan),
            "logg": (wd_phys or {}).get("logg", np.nan),
            "mass": (wd_phys or {}).get("mass", np.nan),
            "radius_rsun": (wd_phys or {}).get("radius_rsun", np.nan),
            "cooling_age_gyr": (wd_phys or {}).get("cooling_age_gyr", np.nan),
            "cluster_age_gyr": plot_cluster_age_gyr,
            "cluster_age_myr": (
                plot_cluster_age_gyr * 1000.0
                if np.isfinite(plot_cluster_age_gyr) else np.nan
            ),
            "cluster_type": plot_cluster_type,
            "sixdim_cluster_type": plot_cluster_type,
            "rv_true": (rv_corr or {}).get("rv_true", np.nan),
            "rv_true_err": (rv_corr or {}).get("rv_true_err", np.nan),
            "rv_true_err_with_grav_floor": (
                (rv_corr or {}).get("rv_true_err_conservative_6d", np.nan)
            ),
            "rv_true_random_err": (rv_corr or {}).get("rv_true_random_err", np.nan),
            "rv_true_grav_err": (rv_corr or {}).get("rv_true_grav_err", np.nan),
            "rv_true_grav_err_conservative_6d": (
                (rv_corr or {}).get("rv_true_grav_err_conservative_6d", np.nan)
            ),
            "rv_obs": (rv_corr or {}).get("rv_obs", np.nan),
            "rv_obs_err": (rv_corr or {}).get("rv_obs_err", np.nan),
            "rv_obs_source": (rv_corr or {}).get("rv_obs_source", ""),
            "rv_mode": (rv_corr or {}).get("rv_mode", rv_mode),
            "preferred_lines": (rv_corr or {}).get(
                "preferred_lines", ";".join(preferred_rv_lines or [])
            ),
            "rv_error_mode": args.rv_error_mode,
            "v_grav": (rv_corr or {}).get("v_grav", np.nan),
            "v_grav_err": (rv_corr or {}).get("v_grav_err", np.nan),
            "v_grav_err_source": (rv_corr or {}).get("v_grav_err_source", ""),
            "gravitational_redshift_applied": (
                (rv_corr or {}).get("gravitational_redshift_applied", np.nan)
            ),
            "rv_true_source": chosen_name if rv_corr else "",
            "cluster_rv": _trace_num("cluster_rv"),
            "cluster_rv_err": _trace_num("cluster_rv_err"),
            "cluster_rv_dispersion": six_dim.DEFAULT_CLUSTER_RV_DISPERSION_KMS,
            "spectral_type": ((wd_out or {}).get("classification") or {}).get("spectral_type", ""),
            "is_dwd": ((wd_out or {}).get("dwd_fit") or {}).get("is_dwd", False),
            "has_DESI": results.get("DESI") is not None,
            "has_SDSS": results.get("SDSS_spectrum") is not None,
            "has_LAMOST": results.get("LAMOST") is not None,
        }
        if args.rv_error_mode == "rvopt_sandbox" and isinstance(rvopt_result, dict):
            row.update({
                k: v for k, v in rvopt_result.items()
                if k != "file"
            })
        if args.wd_param_mode == "wdopt_sandbox" and isinstance(wdopt_rv_result, dict):
            row.update(wdopt_rv_result)
            rv_obs_err_adopted = row.get("rv_obs_err_kms", row.get("rv_obs_err", np.nan))
            if args.rv_error_mode == "rvopt_sandbox" and isinstance(rvopt_result, dict):
                rv_obs_err_adopted = rvopt_result.get("rv_obs_err_opt", rv_obs_err_adopted)
            wdopt_vgrav_err = row.get("wdopt_v_grav_err_kms", np.nan)
            wdopt_rv_err = row.get("wdopt_rv_true_err_kms", np.nan)
            rv_obs_err_num = _finite_num(rv_obs_err_adopted)
            wdopt_vgrav_err_num = _finite_num(wdopt_vgrav_err)
            if np.isfinite(rv_obs_err_num + wdopt_vgrav_err_num):
                wdopt_rv_err = float(np.hypot(rv_obs_err_num, wdopt_vgrav_err_num))
            row.update({
                "wd_param_mode": args.wd_param_mode,
                "rv_obs_adopted": row.get("rv_obs_kms", row.get("rv_obs", np.nan)),
                "rv_obs_err_adopted": rv_obs_err_adopted,
                "v_grav_adopted": row.get("wdopt_v_grav_kms", np.nan),
                "v_grav_err_adopted": row.get("wdopt_v_grav_err_kms", np.nan),
                "rv_true_adopted": row.get("wdopt_rv_true_kms", np.nan),
                "rv_true_err_adopted": wdopt_rv_err,
                "rv_true_adopted_source": "wdopt_sandbox_gaia_parallax_mcmc",
            })
        rv_true_for_6d = row.get("rv_true_adopted", row.get("rv_true"))
        rv_err_for_6d = row.get("rv_true_err_adopted", row["rv_true_err_with_grav_floor"])
        if args.rv_error_mode == "rvopt_sandbox" and isinstance(rvopt_result, dict):
            if not (args.wd_param_mode == "wdopt_sandbox" and isinstance(wdopt_rv_result, dict)):
                rv_err_for_6d = rvopt_result.get("rv_true_err_opt", rv_err_for_6d)
        rv_true_for_6d = _finite_num(rv_true_for_6d)
        rv_err_for_6d = _finite_num(rv_err_for_6d)
        if not np.isfinite(rv_err_for_6d):
            rv_err_for_6d = row["rv_true_err"]
        match = six_dim.check_6d_match(
            row,
            rv_true=rv_true_for_6d,
            rv_err=rv_err_for_6d,
            max_rv_err=200.0,
            max_rv_diff=50.0,
            chi2_kin_limit=10.0,
            rv_nsigma=3.0,
        )
        row.update(match)
        if not np.isfinite(row.get("cluster_rv", np.nan)):
            row["cluster_rv"] = match.get("cluster_rv", np.nan)
        if not np.isfinite(row.get("cluster_rv_err", np.nan)):
            row["cluster_rv_err"] = match.get("cluster_rv_err", np.nan)
        row["cluster_rv_source"] = match.get("cluster_rv_source", "")
        row["cluster_rv_n"] = match.get("cluster_rv_n", np.nan)
        row["cluster_rv_dispersion"] = match.get(
            "cluster_rv_dispersion", row["cluster_rv_dispersion"]
        )
        if "rv_true_adopted" in row:
            adopted_rv = _finite_num(row.get("rv_true_adopted"))
            adopted_err = _finite_num(row.get("rv_true_err_adopted"))
            cl_rv = _finite_num(row.get("cluster_rv"))
            cl_err = _finite_num(row.get("cluster_rv_err"))
            cl_disp = _finite_num(row.get("cluster_rv_dispersion"))
            if np.isfinite(adopted_rv + cl_rv):
                row["delta_rv_adopted"] = adopted_rv - cl_rv
            if np.isfinite(adopted_err + cl_err + cl_disp):
                denom = float(np.sqrt(adopted_err**2 + cl_err**2 + cl_disp**2))
                if denom > 0 and np.isfinite(row.get("delta_rv_adopted", np.nan)):
                    row["rv_sigma_adopted"] = abs(row["delta_rv_adopted"]) / denom
        p = os.path.join(sixd_dir, "sixdim_5d.png")
        six_dim.plot_5d_astrometry(row, p)
        made.append(p)
        if results.get("ZTF_lightcurve") is not None:
            p = os.path.join(sixd_dir, "sixdim_ztf.png")
            six_dim.plot_ztf(row, results["ZTF_lightcurve"], p)
            made.append(p)
        p = os.path.join(sixd_dir, "sixdim_sed.png")
        if six_dim.plot_sed(ra, dec, p):
            made.append(p)
        p = os.path.join(sixd_dir, "sixdim_rv_info.png")
        six_dim.plot_rv_info(row, p)
        made.append(p)
        final_files = six_dim.plot_final_6d_validation_set(row, sixd_dir)
        made.extend(final_files)
        save_json(os.path.join(sixd_dir, "sixdim_validation_row.json"), row)
        return {"files": made}
    run_module(status_rows, output_root, "six_dim", _run_sixd, sixd_dir)

    # Final combined plot refresh after period analysis and RV products.
    run_module(status_rows, output_root, "combined_plots_final", _run_combined, combined_dir)
    run_module(status_rows, output_root, "flat_outputs",
               lambda: export_flat_outputs(output_root), output_root)

    final_summary = {
        "target": target,
        "ra_deg": ra,
        "dec_deg": dec,
        "results_keys": sorted(results.keys()),
        "wd_spectrum_source": chosen_name,
        "wd_fitting_available": wd_out is not None,
        "rv_report_available": rv_report is not None,
        "rv_correction_available": rv_corr is not None,
        "cooling_age_available": cool_res is not None,
        "xray_detected": bool((xray_pack or {}).get("analysis", {}).get("n_detections")),
        "status_file": os.path.join(output_root, "module_status.csv"),
    }
    save_json(os.path.join(output_root, "run_summary.json"), final_summary)


if __name__ == "__main__":
    main()
