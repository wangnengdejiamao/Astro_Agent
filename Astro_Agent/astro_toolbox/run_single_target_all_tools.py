#!/usr/bin/env python3
"""
Run the astro_toolbox science modules for one target and collect outputs.
"""

import argparse
import json
import os
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


def choose_wd_spectrum(results):
    candidates = [
        ("SDSS", results.get("SDSS_spectrum")),
        ("LAMOST", results.get("LAMOST")),
        ("KOA", results.get("KOA_spectrum")),
        ("HST", results.get("HST_spectrum")),
        ("JWST", results.get("JWST_spectrum")),
        ("SPHEREx", results.get("SPHEREx")),
    ]
    for name, spec in candidates:
        if isinstance(spec, dict) and "wavelength" in spec and "flux" in spec:
            wave = np.asarray(spec["wavelength"])
            if wave.size >= 100:
                return name, spec

    desi_res = results.get("DESI")
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
            return "DESI", {
                "wavelength": np.concatenate(waves),
                "flux": np.concatenate(fluxes),
                "error": np.concatenate(errors),
            }
    return None, None


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def run_module(status_rows, output_root, module_name, func, output_dir,
               timeout_sec=180):
    def _handle_timeout(signum, frame):
        raise TimeoutError(f"module timed out after {timeout_sec}s")

    try:
        previous = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(timeout_sec)
        result = func()
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
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
    args = parser.parse_args()

    target = args.target.strip()
    ra = float(args.ra)
    dec = float(args.dec)
    output_root = os.path.abspath(args.output_root)
    ensure_dir(output_root)

    target_info = {
        "target": target,
        "ra_deg": ra,
        "dec_deg": dec,
        "output_root": output_root,
        "script": os.path.abspath(__file__),
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
    desi_catalog = os.path.join(PARENT_DIR, "data", "mws_gaia.csv")
    if not os.path.exists(desi_catalog):
        add_status(status_rows, output_root, "desi", "skipped",
                   output_dir=desi_dir,
                   note=f"local DESI MWS catalog missing: {desi_catalog}")
        desi_res = None
    else:
        def _run_desi():
            tool = DESITool(output_dir=desi_dir, log_func=print)
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
        spec = spherex.query_spectrum(ra, dec)
        if spec is not None:
            spherex.plot_spectrum(spec, os.path.join(spherex_dir, "spherex_spectrum.png"))
            spherex.save_spectrum_csv(spec, spherex_dir)
            res["spectrum"] = spec
        phot = spherex.get_photometry(ra, dec)
        if phot:
            spherex.save_photometry_csv(phot, spherex_dir)
            res["photometry"] = phot
        return res or None
    if os.getenv("ASTRO_TOOLBOX_SKIP_SPHEREX", "").lower() in {"1", "true", "yes"}:
        add_status(status_rows, output_root, "spherex", "skipped",
                   output_dir=spherex_dir, note="ASTRO_TOOLBOX_SKIP_SPHEREX is set")
        spherex_pack = None
    else:
        spherex_pack = run_module(status_rows, output_root, "spherex", _run_spherex, spherex_dir)
    results["SPHEREx"] = spherex_pack.get("spectrum") if spherex_pack else None
    spherex_phot = spherex_pack.get("photometry") if spherex_pack else {}

    koa_dir = ensure_dir(os.path.join(output_root, "koa"))
    koa_work = ensure_dir(os.path.join(koa_dir, "work"))
    if os.getenv("ASTRO_TOOLBOX_ENABLE_KOA", "").lower() not in {"1", "true", "yes"}:
        add_status(status_rows, output_root, "koa", "skipped",
                   output_dir=koa_dir,
                   note="online KOA disabled; set ASTRO_TOOLBOX_ENABLE_KOA=1 to query/download Keck raw products")
        koa_res = None
    else:
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
    def _run_ztf():
        res = ztf.query_lightcurve(ra, dec)
        if res is not None:
            ztf.plot_lightcurve(res, os.path.join(ztf_dir, "ztf_lightcurve.png"))
            ztf.save_csv(res, ztf_dir)
            save_text(os.path.join(ztf_dir, "ztf_web_url.txt"),
                      res.get("web_url", "") + "\n")
        return res
    results["ZTF_lightcurve"] = run_module(status_rows, output_root, "ztf", _run_ztf, ztf_dir)

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
    def _run_tess():
        res = tess.query_lightcurve(ra, dec)
        if res is not None:
            tess.plot_lightcurve(res, os.path.join(tess_dir, "tess_lightcurve.png"))
            tess.save_csv(res, tess_dir)
        return res
    results["TESS"] = run_module(status_rows, output_root, "tess", _run_tess, tess_dir, timeout_sec=900)

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
        fitter.load_photometry(galex_phot, sdss_phot, twomass_phot, wise_phot, spherex_phot)
        fitter.collect_photometry(
            include_galex=not bool(galex_phot),
            include_sdss=not bool(sdss_phot),
            include_gaia=True,
            include_2mass=not bool(twomass_phot),
            include_wise=not bool(wise_phot),
            include_spherex=not bool(spherex_phot),
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

    def _run_combined_fold():
        path = os.path.join(combined_dir, "combined_fold.png")
        fig = combined_plots.plot_combined_fold(results, save_path=path, ra=ra, dec=dec)
        return {"file": path} if fig else None
    run_module(status_rows, output_root, "combined_plots_fold", _run_combined_fold, combined_dir)

    # WD fitting
    wd_dir = ensure_dir(os.path.join(output_root, "wd_fitting"))
    wd_out = None
    wd_phys = None
    chosen_name, chosen_spec = choose_wd_spectrum(results)
    if chosen_spec is None:
        add_status(status_rows, output_root, "wd_fitting", "skipped",
                   output_dir=wd_dir, note="no usable spectrum")
    else:
        def _run_wd():
            fitter = wd_fitting.WDFitter(
                chosen_spec["wavelength"],
                chosen_spec["flux"],
                chosen_spec.get("error"),
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
                output_dir=wd_dir,
            )
            pack["spectrum_source"] = chosen_name
            return pack
        wd_out = run_module(status_rows, output_root, "wd_fitting", _run_wd, wd_dir)
        wd_phys = (wd_out or {}).get("physical_params")

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
            )
        rv_corr = run_module(status_rows, output_root, "rv_correction", _run_rvc, rvc_dir)

    # Cooling age
    cool_dir = ensure_dir(os.path.join(output_root, "cooling_age"))
    def _run_cooling():
        gaia_phot = cooling_age.get_gaia_photometry(ra, dec)
        return cooling_age.run_cooling_age_analysis(
            ra,
            dec,
            cluster_name="",
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
        row = {
            "ra": ra,
            "dec": dec,
            "cluster": "",
            "membership": "",
            "phot_g_mean_mag": hr_params.get("Gmag") if hr_params else np.nan,
            "bp_rp": hr_params.get("BP_RP") if hr_params else np.nan,
            "parallax": hr_params.get("Plx") if hr_params else np.nan,
            "teff": (wd_phys or {}).get("teff", np.nan),
            "logg": (wd_phys or {}).get("logg", np.nan),
            "mass": (wd_phys or {}).get("mass", np.nan),
            "radius_rsun": (wd_phys or {}).get("radius_rsun", np.nan),
            "cooling_age_gyr": (wd_phys or {}).get("cooling_age_gyr", np.nan),
            "cluster_age_gyr": np.nan,
            "rv_true": (rv_corr or {}).get("rv_true", np.nan),
            "rv_true_err": (rv_corr or {}).get("rv_true_err", np.nan),
            "v_grav": (rv_corr or {}).get("v_grav", np.nan),
            "rv_true_source": chosen_name if rv_corr else "",
            "spectral_type": ((wd_out or {}).get("classification") or {}).get("spectral_type", ""),
            "is_dwd": ((wd_out or {}).get("dwd_fit") or {}).get("is_dwd", False),
            "has_DESI": results.get("DESI") is not None,
            "has_SDSS": results.get("SDSS_spectrum") is not None,
            "has_LAMOST": results.get("LAMOST") is not None,
        }
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
        return {"files": made}
    run_module(status_rows, output_root, "six_dim", _run_sixd, sixd_dir)

    # Final combined plot refresh after period analysis and RV products.
    run_module(status_rows, output_root, "combined_plots_final", _run_combined, combined_dir)

    report_dir = ensure_dir(os.path.join(output_root, "compact_binary_report"))
    def _run_compact_binary_report():
        from astro_toolbox.compact_binary_report import build_report, write_report

        report = build_report(output_root, target=target, ra=ra, dec=dec)
        files = write_report(report, report_dir)
        return {"status": "written", "files": files}
    compact_report = run_module(
        status_rows,
        output_root,
        "compact_binary_report",
        _run_compact_binary_report,
        report_dir,
    )

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
        "compact_binary_report": compact_report,
        "status_file": os.path.join(output_root, "module_status.csv"),
    }
    save_json(os.path.join(output_root, "run_summary.json"), final_summary)


if __name__ == "__main__":
    main()
