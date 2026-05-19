"""
SDSS-V DR19 multi-exposure spectra and RV variability tools.

This module targets the SDSS-V/BOSS DR19 ``spectra/full`` products.  The
``full`` FITS files contain the coadd spectrum plus one table HDU per
sub-exposure (``MJD_EXP_*``), which makes them suitable for DWD RV variability
checks similar to Adamane Pallathadka et al. (2026).

Example
-------
python -m astro_toolbox.sdssv \\
    --target J133725.22+395238.8 \\
    --ra 204.3550833 --dec 39.8774583 \\
    --output-dir /tmp/sdssv_j1337 \\
    --reference-period-hr 1.6508
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from scipy.stats import chi2

from . import config
from .rv_fitting import _fit_ccf_peak, compute_ccf, measure_rv


SKYSERVER_SQL_URL = "https://skyserver.sdss.org/dr19/SkyServerWS/SearchTools/SqlSearch"
SAS_BOSS_REDUX = "https://data.sdss.org/sas/dr19/spectro/boss/redux/v6_1_3"


@dataclass
class RVVariabilitySummary:
    n_epoch: int
    rv_mean_kms: float
    rv_chi2: float
    rv_chi2_dof: int
    rv_chi2_pvalue: float
    rv_eta: float
    delta_rv_max_kms: float
    delta_rv_max_sig: float
    baseline_days: float
    rv_variable_candidate: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _read_skyserver_csv(text: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join(lines)))


def _urlopen_with_retries(url: str, timeout: int = 60, retries: int | None = None):
    if retries is None:
        retries = int(getattr(config, "MAX_RETRIES", 3))
    last_exc = None
    for attempt in range(max(1, retries)):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= max(1, retries):
                break
            time.sleep(min(2 ** attempt, 8))
    raise last_exc


def _skyserver_sql(query: str, timeout: int = 60) -> pd.DataFrame:
    url = SKYSERVER_SQL_URL + "?" + urllib.parse.urlencode({
        "cmd": query,
        "format": "csv",
    })
    with _urlopen_with_retries(url, timeout=timeout) as response:
        text = response.read().decode("utf-8", "replace")
    return _read_skyserver_csv(text)


def _angular_sep_arcsec(ra1, dec1, ra2, dec2):
    ra1 = np.deg2rad(np.asarray(ra1, dtype=float))
    dec1 = np.deg2rad(np.asarray(dec1, dtype=float))
    ra2 = math.radians(float(ra2))
    dec2 = math.radians(float(dec2))
    cos_sep = (
        np.sin(dec1) * math.sin(dec2)
        + np.cos(dec1) * math.cos(dec2) * np.cos(ra1 - ra2)
    )
    return np.rad2deg(np.arccos(np.clip(cos_sep, -1.0, 1.0))) * 3600.0


def query_spall_matches(
    ra: float,
    dec: float,
    radius_arcsec: float = config.SEARCH_RADIUS_ARCSEC,
    run2d: str = "v6_1_3",
    max_rows: int = 200,
) -> pd.DataFrame:
    """Query SDSS DR19 ``spAll`` matches near a coordinate."""
    radius_deg = float(radius_arcsec) / 3600.0
    query = f"""
SELECT TOP {int(max_rows)}
    field, mjd, catalogid, sdss_id, gaia_id, fiber_ra, fiber_dec,
    racat, deccat, programname, survey, cadence, firstcarton,
    nexp, exptime, spec_file, run2d
FROM spAll
WHERE fiber_ra BETWEEN {ra - radius_deg:.10f} AND {ra + radius_deg:.10f}
  AND fiber_dec BETWEEN {dec - radius_deg:.10f} AND {dec + radius_deg:.10f}
  AND run2d = '{run2d}'
ORDER BY mjd
"""
    df = _skyserver_sql(query)
    if df.empty:
        return df
    df["sep_arcsec"] = _angular_sep_arcsec(df["fiber_ra"], df["fiber_dec"], ra, dec)
    df = df[df["sep_arcsec"] <= radius_arcsec].copy()
    df.sort_values(["sep_arcsec", "mjd"], inplace=True)
    return df.reset_index(drop=True)


def sdssv_full_spectrum_url(field: int, mjd: int, catalogid: int) -> str:
    field_s = f"{int(field):06d}"
    name = f"spec-{field_s}-{int(mjd)}-{int(catalogid)}.fits"
    return f"{SAS_BOSS_REDUX}/spectra/full/{field_s}/{int(mjd)}/{name}"


def download_full_spectrum(
    field: int,
    mjd: int,
    catalogid: int,
    cache_dir: str,
    overwrite: bool = False,
) -> str:
    """Download a DR19 full spectrum and return the local path."""
    os.makedirs(cache_dir, exist_ok=True)
    field_s = f"{int(field):06d}"
    name = f"spec-{field_s}-{int(mjd)}-{int(catalogid)}.fits"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path) and not overwrite:
        return path
    url = sdssv_full_spectrum_url(field, mjd, catalogid)
    last_exc = None
    for attempt in range(int(getattr(config, "MAX_RETRIES", 3))):
        try:
            with _urlopen_with_retries(url, timeout=int(getattr(config, "TIMEOUT", 600))) as response:
                with open(path, "wb") as fh:
                    while True:
                        chunk = response.read(int(getattr(config, "CHUNK_SIZE", 1024 * 1024)))
                        if not chunk:
                            break
                        fh.write(chunk)
            return path
        except Exception as exc:
            last_exc = exc
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            if attempt + 1 < int(getattr(config, "MAX_RETRIES", 3)):
                time.sleep(min(2 ** attempt, 8))
    raise last_exc
    return path


def _mid_mjd_from_header(header) -> float:
    date_obs = str(header.get("DATE-OBS", "")).strip()
    exptime = float(header.get("EXPTIME", 0.0) or 0.0)
    if date_obs:
        try:
            return float((Time(date_obs, format="isot", scale="utc") + exptime / 2.0 / 86400.0).mjd)
        except Exception:
            pass
    if header.get("TAI-BEG") is not None and header.get("TAI-END") is not None:
        try:
            tai_mid = 0.5 * (float(header["TAI-BEG"]) + float(header["TAI-END"]))
            # SDSS TAI values are seconds from MJD 0.
            return float(tai_mid / 86400.0)
        except Exception:
            pass
    return float(header.get("MJD", np.nan))


def extract_subexposures(fits_path: str) -> list[dict]:
    """Extract all ``MJD_EXP_*`` sub-exposures from a DR19 full spectrum."""
    rows = []
    with fits.open(fits_path, memmap=True) as hdul:
        primary = hdul[0].header
        spall = {}
        if "SPALL" in hdul and len(hdul["SPALL"].data):
            rec = hdul["SPALL"].data[0]
            for key in ("FIELD", "MJD", "CATALOGID", "SDSS_ID", "GAIA_ID", "NEXP"):
                if key in hdul["SPALL"].data.names:
                    value = rec[key]
                    try:
                        value = int(value)
                    except Exception:
                        pass
                    spall[key.lower()] = value
        for hdu in hdul[1:]:
            if not hdu.name.startswith("MJD_EXP"):
                continue
            data = hdu.data
            wave = 10.0 ** np.asarray(data["LOGLAM"], dtype=float)
            flux = np.asarray(data["FLUX"], dtype=float)
            ivar = np.asarray(data["IVAR"], dtype=float)
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
            header = hdu.header
            row = {
                "fits_path": fits_path,
                "extname": hdu.name,
                "field": spall.get("field", primary.get("FIELD")),
                "mjd": int(header.get("MJD", spall.get("mjd", -1))),
                "mjd_mid": _mid_mjd_from_header(header),
                "date_obs": str(header.get("DATE-OBS", "")).strip(),
                "exposure": int(header.get("EXPOSURE", -1)),
                "exptime_s": float(header.get("EXPTIME", np.nan)),
                "helio_rv_kms": float(header.get("HELIO_RV", np.nan)),
                "catalogid": spall.get("catalogid"),
                "sdss_id": spall.get("sdss_id"),
                "gaia_id": spall.get("gaia_id"),
                "wavelength": wave,
                "flux": flux,
                "error": err,
            }
            rows.append(row)
    return rows


def extract_coadd_spectrum(fits_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract the coadd spectrum from a DR19 full spectrum."""
    with fits.open(fits_path, memmap=True) as hdul:
        data = hdul["COADD"].data if "COADD" in hdul else hdul[1].data
        wave = 10.0 ** np.asarray(data["LOGLAM"], dtype=float)
        flux = np.asarray(data["FLUX"], dtype=float)
        err = None
        if "IVAR" in data.names:
            ivar = np.asarray(data["IVAR"], dtype=float)
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
    return wave, flux, err


def measure_rv_against_template(
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray | None,
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    v_min: float = -2500.0,
    v_max: float = 2500.0,
    dv: float = 30.0,
) -> dict | None:
    """Measure relative RV against a supplied template spectrum."""
    velocities, ccf = compute_ccf(
        wave, flux, err, template_wave, template_flux,
        v_min=v_min, v_max=v_max, dv=dv)
    if velocities is None:
        return None
    fit = _fit_ccf_peak(velocities, ccf)
    if fit is None:
        return None
    fit["best_template"] = ("coadd", np.nan)
    fit["velocities"] = velocities
    fit["ccf"] = ccf
    fit["method"] = "CCF_coadd_template"
    return fit


def compute_rv_variability(
    rv_kms: Iterable[float],
    rv_err_kms: Iterable[float],
    mjd: Iterable[float] | None = None,
    eta_threshold: float = 3.0,
) -> RVVariabilitySummary:
    """Compute SDSS-V style RV variability summary statistics."""
    rv = np.asarray(list(rv_kms), dtype=float)
    err = np.asarray(list(rv_err_kms), dtype=float)
    mask = np.isfinite(rv) & np.isfinite(err) & (err > 0)
    if mjd is not None:
        t = np.asarray(list(mjd), dtype=float)
        mask &= np.isfinite(t)
    else:
        t = None
    rv = rv[mask]
    err = err[mask]
    if t is not None:
        t = t[mask]

    if rv.size < 2:
        return RVVariabilitySummary(
            n_epoch=int(rv.size),
            rv_mean_kms=np.nan,
            rv_chi2=np.nan,
            rv_chi2_dof=0,
            rv_chi2_pvalue=np.nan,
            rv_eta=np.nan,
            delta_rv_max_kms=np.nan,
            delta_rv_max_sig=np.nan,
            baseline_days=np.nan,
            rv_variable_candidate=False,
        )

    weights = 1.0 / err**2
    mean = float(np.sum(weights * rv) / np.sum(weights))
    chi2_value = float(np.sum(((rv - mean) / err) ** 2))
    dof = int(rv.size - 1)
    pvalue = float(chi2.sf(chi2_value, dof))
    eta = float(-np.log10(max(pvalue, 1e-300)))

    max_dv = 0.0
    max_sig = 0.0
    for i in range(rv.size):
        for j in range(i + 1, rv.size):
            dv = abs(float(rv[i] - rv[j]))
            sig = dv / math.sqrt(float(err[i] ** 2 + err[j] ** 2))
            max_dv = max(max_dv, dv)
            max_sig = max(max_sig, sig)

    baseline = float(np.nanmax(t) - np.nanmin(t)) if t is not None else np.nan
    return RVVariabilitySummary(
        n_epoch=int(rv.size),
        rv_mean_kms=mean,
        rv_chi2=chi2_value,
        rv_chi2_dof=dof,
        rv_chi2_pvalue=pvalue,
        rv_eta=eta,
        delta_rv_max_kms=max_dv,
        delta_rv_max_sig=max_sig,
        baseline_days=baseline,
        rv_variable_candidate=bool(eta >= eta_threshold),
    )


def save_rv_timeseries_plot(rv_df: pd.DataFrame, summary: RVVariabilitySummary, path: str) -> str:
    good = rv_df[rv_df["used_for_variability"].astype(bool)].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not good.empty:
        t0 = float(good["mjd_mid"].min())
        ax.errorbar(
            good["mjd_mid"] - t0,
            good["rv_kms"],
            yerr=good["rv_err_eff_kms"],
            fmt="o",
            color="tab:blue",
            ecolor="0.55",
            capsize=2,
            label="used sub-exposures",
        )
        ax.axhline(summary.rv_mean_kms, color="tab:red", lw=1.5, alpha=0.8, label="weighted mean")
        ax.set_xlabel(f"MJD - {t0:.5f} (days)")
    else:
        ax.text(0.5, 0.5, "No usable sub-exposure RVs", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_xlabel("MJD")
    bad = rv_df[~rv_df["used_for_variability"].astype(bool)].copy()
    if not bad.empty and "mjd_mid" in bad:
        t0 = float(good["mjd_mid"].min()) if not good.empty else float(bad["mjd_mid"].min())
        valid_bad = bad[np.isfinite(bad["rv_kms"])]
        if not valid_bad.empty:
            ax.scatter(valid_bad["mjd_mid"] - t0, valid_bad["rv_kms"],
                       marker="x", color="0.5", label="rejected")
    ax.set_ylabel("RV (km/s)")
    ax.grid(True, alpha=0.3)
    title = (
        f"SDSS-V DR19 sub-exposure RVs: "
        f"eta={summary.rv_eta:.2f}, max dRV={summary.delta_rv_max_kms:.0f} km/s"
    )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_rv_periodogram_plot(
    rv_df: pd.DataFrame,
    path: str,
    min_period_hr: float = 0.5,
    max_period_hr: float = 100.0,
    reference_period_hr: float | None = None,
) -> dict:
    good = rv_df[rv_df["used_for_variability"].astype(bool)].copy()
    result = {"best_period_hr": np.nan, "periodogram_path": path}
    if len(good) < 5:
        return result
    try:
        from astropy.timeseries import LombScargle
    except Exception:
        return result

    t = good["mjd_mid"].to_numpy(float)
    y = good["rv_kms"].to_numpy(float)
    dy = good["rv_err_eff_kms"].to_numpy(float)
    t = t - np.nanmin(t)
    y = y - np.nanmedian(y)
    min_freq = 24.0 / max_period_hr
    max_freq = 24.0 / min_period_hr
    freq = np.linspace(min_freq, max_freq, 5000)
    ls = LombScargle(t, y, dy=dy)
    power = ls.power(freq)
    period_hr = 24.0 / freq
    best = int(np.nanargmax(power))
    best_period = float(period_hr[best])
    result["best_period_hr"] = best_period
    result["best_power"] = float(power[best])

    phase = (t / (best_period / 24.0)) % 1.0
    order = np.argsort(phase)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(period_hr, power, color="k", lw=1.0)
    ax1.axvline(best_period, color="tab:red", lw=1.5, alpha=0.8,
                label=f"best {best_period:.2f} hr")
    if reference_period_hr:
        ax1.axvline(reference_period_hr, color="tab:blue", lw=1.3, ls="--",
                    alpha=0.8, label=f"reference {reference_period_hr:.2f} hr")
    ax1.set_xscale("log")
    ax1.set_xlabel("Period (hr)")
    ax1.set_ylabel("Lomb-Scargle power")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.errorbar(phase[order], y[order], yerr=dy[order], fmt="o",
                 color="tab:blue", ecolor="0.6", capsize=2)
    ax2.errorbar(phase[order] + 1.0, y[order], yerr=dy[order], fmt="o",
                 color="tab:blue", ecolor="0.6", capsize=2, alpha=0.8)
    ax2.set_xlabel(f"Phase at P={best_period:.2f} hr")
    ax2.set_ylabel("RV - median (km/s)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return result


def run_sdssv_rv_analysis(
    ra: float,
    dec: float,
    output_dir: str,
    target_name: str = "",
    radius_arcsec: float = config.SEARCH_RADIUS_ARCSEC,
    rv_error_floor_kms: float = 10.0,
    min_ccf_height: float = 0.08,
    max_rv_err_kms: float = 90.0,
    v_min: float = -2500.0,
    v_max: float = 2500.0,
    dv: float = 30.0,
    template_mode: str = "coadd",
    max_spectra: int | None = None,
    reference_period_hr: float | None = None,
    overwrite_downloads: bool = False,
) -> dict:
    """Run a complete SDSS-V DR19 sub-exposure RV variability analysis."""
    os.makedirs(output_dir, exist_ok=True)
    spectra_dir = os.path.join(output_dir, "sdssv_full_spectra")
    os.makedirs(spectra_dir, exist_ok=True)

    matches_path = os.path.join(output_dir, "sdssv_dr19_spall_matches.csv")
    try:
        matches = query_spall_matches(ra, dec, radius_arcsec=radius_arcsec)
        matches.to_csv(matches_path, index=False)
    except Exception:
        if os.path.exists(matches_path):
            matches = pd.read_csv(matches_path)
        else:
            raise

    report = {
        "target": target_name,
        "ra_deg": float(ra),
        "dec_deg": float(dec),
        "matches_path": matches_path,
        "n_spall_matches": int(len(matches)),
        "rv_timeseries_path": "",
        "rv_summary_path": "",
        "rv_plot_path": "",
        "rv_periodogram_path": "",
        "summary": None,
    }
    if matches.empty:
        return report

    if max_spectra is not None:
        matches = matches.head(int(max_spectra)).copy()

    subexp_meta = []
    rv_rows = []
    coadd_template = None
    for _, match in matches.iterrows():
        try:
            path = download_full_spectrum(
                int(match["field"]), int(match["mjd"]), int(match["catalogid"]),
                spectra_dir, overwrite=overwrite_downloads)
        except Exception as exc:
            rv_rows.append({
                "field": match.get("field"),
                "mjd": match.get("mjd"),
                "catalogid": match.get("catalogid"),
                "rv_kms": np.nan,
                "rv_err_kms": np.nan,
                "rv_err_eff_kms": np.nan,
                "ccf_height": np.nan,
                "used_for_variability": False,
                "reject_reason": f"download_failed:{type(exc).__name__}",
            })
            continue

        if template_mode == "coadd" and coadd_template is None:
            coadd_template = extract_coadd_spectrum(path)

        for sub in extract_subexposures(path):
            meta = {k: v for k, v in sub.items()
                    if k not in ("wavelength", "flux", "error")}
            subexp_meta.append(meta)
            try:
                if template_mode == "coadd" and coadd_template is not None:
                    rv = measure_rv_against_template(
                        sub["wavelength"], sub["flux"], sub["error"],
                        coadd_template[0], coadd_template[1],
                        v_min=v_min, v_max=v_max, dv=dv)
                else:
                    rv = measure_rv(
                        sub["wavelength"], sub["flux"], sub["error"],
                        v_min=v_min, v_max=v_max, dv=dv)
            except Exception:
                rv = None
            row = meta.copy()
            if rv is None:
                row.update({
                    "rv_kms": np.nan,
                    "rv_err_kms": np.nan,
                    "rv_err_eff_kms": np.nan,
                    "ccf_height": np.nan,
                    "template_teff": np.nan,
                    "template_logg": np.nan,
                    "used_for_variability": False,
                    "reject_reason": "ccf_failed",
                })
            else:
                err = float(rv.get("rv_err", np.nan))
                eff_err = max(err, float(rv_error_floor_kms)) if np.isfinite(err) else np.nan
                ccf = float(rv.get("ccf_height", np.nan))
                reject = []
                if not np.isfinite(err) or err > max_rv_err_kms:
                    reject.append("large_rv_error")
                if not np.isfinite(ccf) or ccf < min_ccf_height:
                    reject.append("weak_ccf")
                tmpl = rv.get("best_template") or (np.nan, np.nan)
                row.update({
                    "rv_kms": float(rv.get("rv", np.nan)),
                    "rv_err_kms": err,
                    "rv_err_eff_kms": eff_err,
                    "ccf_height": ccf,
                    "template_teff": tmpl[0],
                    "template_logg": tmpl[1],
                    "rv_reference_mode": template_mode,
                    "used_for_variability": len(reject) == 0,
                    "reject_reason": ";".join(reject),
                })
            rv_rows.append(row)

    subexp_path = os.path.join(output_dir, "sdssv_dr19_subexposures.csv")
    pd.DataFrame(subexp_meta).to_csv(subexp_path, index=False)
    rv_df = pd.DataFrame(rv_rows)
    rv_path = os.path.join(output_dir, "sdssv_dr19_rv_timeseries.csv")
    rv_df.to_csv(rv_path, index=False)

    if "used_for_variability" in rv_df.columns:
        good = rv_df[rv_df["used_for_variability"].astype(bool)].copy()
    else:
        good = rv_df.iloc[0:0].copy()
    summary = compute_rv_variability(
        good["rv_kms"].to_numpy(float) if not good.empty else [],
        good["rv_err_eff_kms"].to_numpy(float) if not good.empty else [],
        good["mjd_mid"].to_numpy(float) if not good.empty else [],
    )
    summary_path = os.path.join(output_dir, "sdssv_dr19_rv_summary.csv")
    pd.DataFrame([summary.as_dict()]).to_csv(summary_path, index=False)
    plot_path = os.path.join(output_dir, "sdssv_dr19_rv_timeseries.png")
    save_rv_timeseries_plot(rv_df, summary, plot_path)
    period_path = os.path.join(output_dir, "sdssv_dr19_rv_periodogram.png")
    period_result = save_rv_periodogram_plot(
        rv_df, period_path, reference_period_hr=reference_period_hr)

    notes_path = os.path.join(output_dir, "sdssv_dr19_rv_notes.txt")
    with open(notes_path, "w", encoding="utf-8") as fh:
        fh.write("SDSS-V DR19 sub-exposure RV analysis\n")
        if target_name:
            fh.write(f"Target: {target_name}\n")
        fh.write(f"Coordinates: RA={ra:.8f}, Dec={dec:.8f}\n")
        fh.write(f"spAll matches: {len(matches)}\n")
        fh.write(f"RV template mode: {template_mode}\n")
        fh.write(f"usable RV sub-exposures: {summary.n_epoch}\n")
        fh.write(f"eta=-log10(p_chi2) = {summary.rv_eta:.4g}\n")
        fh.write(f"max delta RV = {summary.delta_rv_max_kms:.3f} km/s\n")
        if np.isfinite(period_result.get("best_period_hr", np.nan)):
            fh.write(f"best RV LS period = {period_result['best_period_hr']:.4f} hr\n")
        fh.write("\nData products:\n")
        for key in (matches_path, subexp_path, rv_path, summary_path, plot_path, period_path):
            fh.write(f"  {key}\n")

    report.update({
        "subexposures_path": subexp_path,
        "rv_timeseries_path": rv_path,
        "rv_summary_path": summary_path,
        "rv_plot_path": plot_path,
        "rv_periodogram_path": period_path,
        "notes_path": notes_path,
        "summary": summary.as_dict(),
        "period_result": period_result,
    })
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run SDSS-V DR19 sub-exposure RV analysis.")
    parser.add_argument("--ra", required=True, type=float)
    parser.add_argument("--dec", required=True, type=float)
    parser.add_argument("--target", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius-arcsec", type=float, default=config.SEARCH_RADIUS_ARCSEC)
    parser.add_argument("--reference-period-hr", type=float, default=None)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--min-ccf-height", type=float, default=0.08)
    parser.add_argument("--rv-error-floor-kms", type=float, default=10.0)
    parser.add_argument("--template-mode", choices=("coadd", "wd_grid"), default="coadd")
    parser.add_argument("--v-min", type=float, default=-2500.0)
    parser.add_argument("--v-max", type=float, default=2500.0)
    args = parser.parse_args()

    report = run_sdssv_rv_analysis(
        args.ra,
        args.dec,
        args.output_dir,
        target_name=args.target,
        radius_arcsec=args.radius_arcsec,
        reference_period_hr=args.reference_period_hr,
        max_spectra=args.max_spectra,
        min_ccf_height=args.min_ccf_height,
        rv_error_floor_kms=args.rv_error_floor_kms,
        template_mode=args.template_mode,
        v_min=args.v_min,
        v_max=args.v_max,
    )
    summary = report.get("summary") or {}
    if summary:
        print(
            "SDSS-V RV: "
            f"n={summary.get('n_epoch')}, "
            f"eta={summary.get('rv_eta'):.3g}, "
            f"max_dRV={summary.get('delta_rv_max_kms'):.1f} km/s"
        )
    print(report)


if __name__ == "__main__":
    _main()
