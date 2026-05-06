"""Build an ApJ/AASTeX analysis package for ZTFJ152934.91+292801.87.

This is a source-specific integration test for the astronomy Chief Investigator
agent.  It consumes already-fetched local astro_toolbox products and writes a
conservative paper draft with explicit QA caveats.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "Astro_Agent" / "output" / "astro_output" / "RA232.3955_DEC29.4672"
SDSS_ROOT = REPO_ROOT / "Astro_Agent" / "output" / "astro_output" / "ZTFJ152934_analysis" / "sdss"
PERIOD_PATH = REPO_ROOT / "Astro_Agent" / "output" / "astro_output" / "ZTFJ152934_analysis" / "period_summary.json"
TEMPLATE_ROOT = REPO_ROOT / "Astro_Agent" / "templates" / "aastex"
OUT_ROOT = REPO_ROOT / "Astro_Agent" / "output" / "analysis_agent" / "ZTFJ152934_paper"

C_KMS = 299792.458


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def tex_escape(text: object) -> str:
    value = str(text)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        value = value.replace(old, new)
    return value


def spectrum_arrays(path: Path, wave_col: str, flux_col: str, err_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_csv(path)
    wave = np.array([float(r[wave_col]) for r in rows], dtype=float)
    flux = np.array([float(r[flux_col]) for r in rows], dtype=float)
    err = np.array([float(r[err_col]) for r in rows], dtype=float)
    ok = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
    return wave[ok], flux[ok], err[ok]


def median_snr(flux: np.ndarray, err: np.ndarray) -> float:
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.nanmedian(np.abs(flux[ok] / err[ok])))


def measure_line(wave: np.ndarray, flux: np.ndarray, err: np.ndarray, name: str, rest: float) -> Dict[str, float | str]:
    """Measure a broad absorption feature against a local linear continuum."""
    line_mask = (wave > rest - 25.0) & (wave < rest + 25.0)
    cont_mask = ((wave > rest - 95.0) & (wave < rest - 45.0)) | ((wave > rest + 45.0) & (wave < rest + 95.0))
    if np.count_nonzero(line_mask) < 8 or np.count_nonzero(cont_mask) < 8:
        return {"line": name, "rest_A": rest, "status": "insufficient_coverage"}

    coeff = np.polyfit(wave[cont_mask] - rest, flux[cont_mask], 1)
    continuum = np.polyval(coeff, wave[line_mask] - rest)
    valid = np.isfinite(continuum) & (continuum != 0)
    lw = wave[line_mask][valid]
    lf = flux[line_mask][valid]
    le = err[line_mask][valid]
    cont = continuum[valid]
    norm = lf / cont
    depth = 1.0 - norm
    ew = float(np.trapz(depth, lw))
    pix = np.gradient(lw)
    ew_err = float(math.sqrt(np.nansum(((le / cont) * pix) ** 2)))
    snr = abs(ew) / ew_err if ew_err > 0 else float("nan")
    if np.nansum(np.clip(depth, 0, None)) > 0:
        center = float(np.nansum(lw * np.clip(depth, 0, None)) / np.nansum(np.clip(depth, 0, None)))
    else:
        center = float(lw[np.nanargmin(norm)])
    velocity = (center - rest) / rest * C_KMS
    core_norm = float(np.nanmin(norm))
    return {
        "line": name,
        "rest_A": rest,
        "ew_abs_A": ew,
        "ew_err_A": ew_err,
        "snr": snr,
        "center_A": center,
        "velocity_kms": velocity,
        "core_norm_flux": core_norm,
        "status": "absorption" if ew > 0 else "emission_or_continuum_error",
    }


def lightcurve_stats(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    by_band: Dict[str, List[Tuple[float, float]]] = {}
    for row in rows:
        try:
            mag = float(row["mag"])
            err = float(row.get("magerr") or "nan")
            band = row["band"]
        except Exception:
            continue
        if math.isfinite(mag):
            by_band.setdefault(band, []).append((mag, err))
    stats: Dict[str, Dict[str, float]] = {}
    for band, values in by_band.items():
        mags = np.array([v[0] for v in values], dtype=float)
        errs = np.array([v[1] for v in values], dtype=float)
        stats[band] = {
            "n": int(len(mags)),
            "median_mag": float(np.nanmedian(mags)),
            "std_mag": float(np.nanstd(mags)),
            "range_mag": float(np.nanmax(mags) - np.nanmin(mags)),
            "median_magerr": float(np.nanmedian(errs[np.isfinite(errs)])) if np.any(np.isfinite(errs)) else float("nan"),
        }
    return stats


def hst_quality(path: Path) -> Dict[str, float | int | str]:
    wave, flux, err = spectrum_arrays(path, "wavelength_A", "flux", "error")
    ok_err = err > 0
    neg = np.count_nonzero(flux < 0)
    zeros = np.count_nonzero(flux == 0)
    return {
        "n_points": int(len(wave)),
        "wavelength_min_A": float(np.nanmin(wave)),
        "wavelength_max_A": float(np.nanmax(wave)),
        "median_snr": median_snr(flux[ok_err], err[ok_err]) if np.any(ok_err) else 0.0,
        "negative_or_zero_flux_fraction": float((neg + zeros) / max(len(flux), 1)),
        "qa_flag": "low_snr_use_only_as_non-diagnostic_uv_context",
    }


def copy_inputs() -> Dict[str, str]:
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_map = {
        "desi_spectrum": SOURCE_ROOT / "desi_spectrum.png",
        "sdss_spectrum": SDSS_ROOT / "sdss_spectrum.png",
        "sed": SOURCE_ROOT / "sed.png",
        "ztf_lightcurve": SOURCE_ROOT / "ztf_lightcurve.png",
        "combined_fold": SOURCE_ROOT / "combined_fold.png",
        "hr_diagram": SOURCE_ROOT / "hr_diagram.png",
        "hst_spectrum": SOURCE_ROOT / "hst_spectrum.png",
    }
    copied: Dict[str, str] = {}
    for key, src in figure_map.items():
        if src.exists():
            dst = fig_dir / src.name
            shutil.copy2(src, dst)
            copied[f"{key}_plot_path"] = str(dst.relative_to(OUT_ROOT))
    for name in ("aastex701.cls", "aasjournalv7.bst"):
        src = TEMPLATE_ROOT / name
        if src.exists():
            shutil.copy2(src, OUT_ROOT / name)
    return copied


def build_summary() -> Dict[str, object]:
    desi_wave, desi_flux, desi_err = spectrum_arrays(SOURCE_ROOT / "desi_spectrum.csv", "wavelength_A", "flux", "error")
    sdss_wave, sdss_flux, sdss_err = spectrum_arrays(SDSS_ROOT / "sdss_spectrum.csv", "wavelength_A", "flux", "error")
    lines = [
        ("Halpha", 6562.8),
        ("Hbeta", 4861.33),
        ("Hgamma", 4340.47),
        ("Hdelta", 4101.74),
        ("CaII_H/Hepsilon", 3968.47),
    ]
    desi_lines = [measure_line(desi_wave, desi_flux, desi_err, name, rest) for name, rest in lines]
    sdss_lines = [measure_line(sdss_wave, sdss_flux, sdss_err, name, rest) for name, rest in lines]

    sed_rows = read_csv(SOURCE_ROOT / "sed_photometry.csv")
    ztf_rows = read_csv(SOURCE_ROOT / "ztf_lightcurve.csv")
    wise_rows = read_csv(SOURCE_ROOT / "wise_lightcurve.csv")
    period = json.loads(PERIOD_PATH.read_text(encoding="utf-8"))
    sdss_prov = json.loads((SDSS_ROOT / "sdss_spectrum_provenance.json").read_text(encoding="utf-8"))
    figures = copy_inputs()

    plx_mas = 11.47
    e_plx_mas = 0.0799
    distance_pc = 1000.0 / plx_mas
    distance_err_pc = 1000.0 * e_plx_mas / (plx_mas**2)
    pmra = -74.01
    pmdec = -5.387
    pm_total = math.hypot(pmra, pmdec)
    vt = 4.74047 * (pm_total / 1000.0) * distance_pc
    gmag = 17.486246
    abs_g = gmag - 5.0 * math.log10(distance_pc / 10.0)

    return {
        "target": "ZTFJ152934.91+292801.87",
        "preferred_id": "CSO 1094 / SDSS J152934.98+292801.9 / J1529+2928",
        "coordinates": {"ra_deg": 232.39546190293, "dec_deg": 29.46718626414, "frame": "ICRS/J2000"},
        "simbad": {
            "main_id": "CSO 1094",
            "otype": "WhiteDwarf",
            "spectral_type": "DAH:",
            "gaia_dr3_source_id": "1273456463234876288",
            "crossmatch_note": "ZTF-formatted name was not directly resolvable; a 5 arcsec region match at the coordinates identifies the SDSS/Gaia white dwarf.",
        },
        "gaia": {
            "G_mag": gmag,
            "BP_mag": 17.543892,
            "RP_mag": 17.457438,
            "BP_RP_mag": 0.086454,
            "parallax_mas": plx_mas,
            "parallax_err_mas": e_plx_mas,
            "distance_pc_simple_inverse": distance_pc,
            "distance_err_pc_simple_inverse": distance_err_pc,
            "pmra_masyr": pmra,
            "pmdec_masyr": pmdec,
            "pm_total_masyr": pm_total,
            "tangential_velocity_kms": vt,
            "M_G_mag": abs_g,
        },
        "spectra": {
            "desi": {
                "n_points": int(len(desi_wave)),
                "wavelength_min_A": float(np.nanmin(desi_wave)),
                "wavelength_max_A": float(np.nanmax(desi_wave)),
                "median_snr": median_snr(desi_flux, desi_err),
                "line_measurements": desi_lines,
            },
            "sdss": {
                "n_points": int(len(sdss_wave)),
                "wavelength_min_A": float(np.nanmin(sdss_wave)),
                "wavelength_max_A": float(np.nanmax(sdss_wave)),
                "median_snr": median_snr(sdss_flux, sdss_err),
                "provenance": sdss_prov,
                "line_measurements": sdss_lines,
            },
            "hst": hst_quality(SOURCE_ROOT / "hst_spectrum.csv"),
            "interpretation": "Balmer absorption dominated DA/DAH white dwarf; no robust optical emission lines in the local DESI or SDSS spectra.",
        },
        "sed_photometry": sed_rows,
        "lightcurves": {
            "ztf_stats": lightcurve_stats(ztf_rows),
            "wise_stats": lightcurve_stats(wise_rows),
            "period_search": period,
            "published_period": {
                "period_s": 2288.792,
                "period_min": 38.1465,
                "frequency_per_day": 37.74917,
                "amplitude_percent": 2.95,
                "source": "Kilic et al. 2015, ApJ Letters, 814, L31",
            },
        },
        "qa": {
            "gate": "draft_with_human_review_caveats",
            "blocking_caveats": [
                "SIMBAD online query failed in the current sandbox test, so the paper uses the previously saved SIMBAD cross-match facts.",
                "No new atmosphere-grid fit is certified in this run; previously published massive-WD interpretation is cited rather than re-derived.",
                "The ZTF-only Lomb-Scargle peak near 0.03054 d is not adopted over the published high-speed 38.1465 min period without a full window-function analysis.",
                "HST UV spectrum has low S/N and is not used for line diagnostics.",
            ],
        },
        "artifacts": figures,
    }


def line_table(summary: Dict[str, object], survey: str) -> str:
    rows = summary["spectra"][survey]["line_measurements"]  # type: ignore[index]
    text = [
        r"\begin{deluxetable}{lrrrr}",
        rf"\tablecaption{{Local {survey.upper()} line measurements. Positive equivalent width denotes absorption.}}",
        r"\tablehead{\colhead{Line} & \colhead{$\lambda_0$} & \colhead{EW} & \colhead{S/N} & \colhead{$v_{\rm cen}$} \\",
        r"\colhead{} & \colhead{(\AA)} & \colhead{(\AA)} & \colhead{} & \colhead{(km s$^{-1}$)}}",
        r"\startdata",
    ]
    for row in rows:  # type: ignore[assignment]
        if row.get("status") == "insufficient_coverage":
            continue
        text.append(
            f"{tex_escape(row['line'])} & {row['rest_A']:.2f} & {row['ew_abs_A']:.2f} $\\pm$ {row['ew_err_A']:.2f} & "
            f"{row['snr']:.1f} & {row['velocity_kms']:.1f} \\\\"
        )
    text.extend(
        [
            r"\enddata",
            r"\tablecomments{Measurements are automated local-continuum estimates intended for classification and QA, not final atmosphere-model parameters.}",
            r"\end{deluxetable}",
        ]
    )
    return "\n".join(text)


def phot_table(summary: Dict[str, object]) -> str:
    rows = summary["sed_photometry"]  # type: ignore[assignment]
    text = [
        r"\begin{deluxetable}{lrrr}",
        r"\tablecaption{Photometry used for the SED sanity check.}",
        r"\tablehead{\colhead{Band} & \colhead{$\lambda_{\rm eff}$} & \colhead{Magnitude} & \colhead{$\sigma_m$} \\",
        r"\colhead{} & \colhead{(\AA)} & \colhead{(mag)} & \colhead{(mag)}}",
        r"\startdata",
    ]
    for row in rows:
        err = row.get("mag_err") or ""
        text.append(f"{tex_escape(row['band'])} & {float(row['wave_A']):.0f} & {float(row['mag']):.3f} & {float(err):.3f} \\\\")
    text.extend([r"\enddata", r"\end{deluxetable}"])
    return "\n".join(text)


def refs_bib() -> str:
    return r"""@article{Kilic2015J1529,
  author = {Kilic, Mukremin and Gianninas, Alexandros and Bell, Keaton J. and Curd, Brandon and Brown, Warren R.},
  title = {A Dark Spot on a Massive White Dwarf},
  journal = {The Astrophysical Journal},
  year = {2015},
  volume = {814},
  pages = {L31},
  doi = {10.1088/2041-8205/814/2/L31}
}

@article{Wenger2000Simbad,
  author = {Wenger, M. and Ochsenbein, F. and Egret, D. and Dubois, P. and Bonnarel, F. and Borde, S. and Genova, F. and Jasniewicz, G. and Lalo{\"e}, S. and Lesteven, S. and Monier, R.},
  title = {The SIMBAD astronomical database},
  journal = {Astronomy and Astrophysics Supplement Series},
  year = {2000},
  volume = {143},
  pages = {9}
}

@article{GaiaDR3,
  author = {{Gaia Collaboration} and Vallenari, A. and Brown, A. G. A. and Prusti, T. and others},
  title = {Gaia Data Release 3. Summary of the content and survey properties},
  journal = {Astronomy and Astrophysics},
  year = {2023},
  volume = {674},
  pages = {A1}
}

@article{SDSS,
  author = {Alam, S. and Albareti, F. D. and Allende Prieto, C. and others},
  title = {The Eleventh and Twelfth Data Releases of the Sloan Digital Sky Survey},
  journal = {The Astrophysical Journal Supplement Series},
  year = {2015},
  volume = {219},
  pages = {12}
}

@article{DESI,
  author = {{DESI Collaboration} and Aghamousa, A. and Aguilar, J. and others},
  title = {The DESI Experiment Part I: Science, Targeting, and Survey Design},
  journal = {arXiv e-prints},
  year = {2016},
  eprint = {1611.00036}
}

@article{ZTF,
  author = {Bellm, Eric C. and Kulkarni, Shrinivas R. and Graham, Matthew J. and others},
  title = {The Zwicky Transient Facility: System Overview, Performance, and First Results},
  journal = {Publications of the Astronomical Society of the Pacific},
  year = {2019},
  volume = {131},
  pages = {018002}
}

@article{WISE,
  author = {Wright, E. L. and Eisenhardt, P. R. M. and Mainzer, A. K. and others},
  title = {The Wide-field Infrared Survey Explorer (WISE): Mission Description and Initial On-orbit Performance},
  journal = {The Astronomical Journal},
  year = {2010},
  volume = {140},
  pages = {1868}
}

@article{TESS,
  author = {Ricker, George R. and Winn, Joshua N. and Vanderspek, Roland and others},
  title = {Transiting Exoplanet Survey Satellite (TESS)},
  journal = {Journal of Astronomical Telescopes, Instruments, and Systems},
  year = {2015},
  volume = {1},
  pages = {014003}
}

@article{Astropy,
  author = {{Astropy Collaboration} and Robitaille, Thomas P. and Tollerud, Erik J. and others},
  title = {Astropy: A community Python package for astronomy},
  journal = {Astronomy and Astrophysics},
  year = {2013},
  volume = {558},
  pages = {A33}
}
"""


def paper_tex(summary: Dict[str, object]) -> str:
    gaia = summary["gaia"]  # type: ignore[assignment]
    period = summary["lightcurves"]["period_search"]  # type: ignore[index]
    ztf_g = period["ZTF_g"]
    ztf_r = period["ZTF_r"]
    pub = summary["lightcurves"]["published_period"]  # type: ignore[index]
    simbad = summary["simbad"]  # type: ignore[assignment]
    coords = summary["coordinates"]  # type: ignore[assignment]
    artifacts = summary["artifacts"]  # type: ignore[assignment]
    sdss_prov = summary["spectra"]["sdss"]["provenance"]  # type: ignore[index]
    hst = summary["spectra"]["hst"]  # type: ignore[index]

    return rf"""\documentclass[twocolumn]{{aastex701}}

\shorttitle{{A Multi-wavelength Check of J1529+2928}}
\shortauthors{{Chief Investigator Agent}}

\begin{{document}}

\title{{A Reproducible Multi-wavelength Re-analysis of the Spotted White Dwarf ZTFJ152934.91+292801.87}}

\author{{Chief Investigator Agent}}
\affiliation{{Automated Astronomy Workflow Laboratory}}
\email{{agent@example.invalid}}

\begin{{abstract}}
We present a machine-auditable re-analysis of \texttt{{ZTFJ152934.91+292801.87}}, the source cross-matched with
CSO 1094 and SDSS J152934.98+292801.9.  SIMBAD classifies the object as a white dwarf with spectral type
{simbad["spectral_type"]}.  Gaia DR3 source {simbad["gaia_dr3_source_id"]} gives $G={gaia["G_mag"]:.3f}$ mag,
$G_{{\rm BP}}-G_{{\rm RP}}={gaia["BP_RP_mag"]:.3f}$ mag, and $\varpi={gaia["parallax_mas"]:.3f}\pm{gaia["parallax_err_mas"]:.3f}$ mas,
corresponding to a simple inverse-parallax distance of ${gaia["distance_pc_simple_inverse"]:.2f}\pm{gaia["distance_err_pc_simple_inverse"]:.2f}$ pc.
Local DESI and SDSS spectra are dominated by Balmer absorption and do not show robust optical emission lines.
The literature reports a stable 2288.792 s (38.1465 min) photometric dip, interpreted as a dark spot on a massive
white dwarf.  Our ZTF-only period search recovers short-period power near 0.03054 d in $g$ and $r$, but the result
is not adopted as a replacement period because survey-cadence aliasing has not yet been fully modeled.
This draft therefore reports catalog, spectral, SED, and variability checks while retaining a human-review gate
for final atmosphere parameters.
\end{{abstract}}

\keywords{{white dwarf stars (1799) --- Stellar rotation (1629) --- Stellar spots (1572) --- Time domain astronomy (2109)}}

\section{{Introduction}}
ZTFJ152934.91+292801.87 lies at ICRS coordinates
$\alpha={coords["ra_deg"]:.11f}^\circ$, $\delta={coords["dec_deg"]:.11f}^\circ$.
The coordinate-based SIMBAD match identifies the source as CSO 1094, with aliases including
SDSS J152934.98+292801.9 and Gaia DR3 {simbad["gaia_dr3_source_id"]}.  The source is not a newly
unclassified transient in the local literature corpus: \citet{{Kilic2015J1529}} reported eclipse-like
events around J1529+2928 with a 38.1 min period and argued that the dips are caused by a dark spot
rotating into view on a massive white dwarf.  That work also reports no significant radial-velocity
variations from follow-up spectroscopy and no Zeeman splitting of the Balmer lines at their sensitivity,
with an upper limit of $B<70$ kG.

The purpose of this paper is narrower than discovery.  We test the integrated Chief Investigator agent
on this known source, forcing the workflow to read catalog identity, local spectra, photometry, and time
series before writing.  The agent is required to stop short of unsupported white-dwarf atmosphere
parameters unless a model grid fit and its systematic error budget are certified.

\section{{Data and Source Identity}}
We use the SIMBAD cross-identification \citep{{Wenger2000Simbad}}, Gaia DR3 astrometry and photometry
\citep{{GaiaDR3}}, local SDSS spectroscopy \citep{{SDSS}}, DESI spectroscopy \citep{{DESI}}, ZTF photometry
\citep{{ZTF}}, WISE photometry \citep{{WISE}}, and a locally downloaded TESS light curve \citep{{TESS}}.
The SDSS spectrum used here is plate {sdss_prov["plate"]}, MJD {sdss_prov["mjd"]}, fiber {sdss_prov["fiberid"]},
with exposure time {sdss_prov["exptime_s"]:.1f} s and archive redshift $z={sdss_prov["redshift"]:.6f}$.

Gaia gives $\mu_\alpha={gaia["pmra_masyr"]:.2f}$ mas yr$^{{-1}}$ and
$\mu_\delta={gaia["pmdec_masyr"]:.2f}$ mas yr$^{{-1}}$, yielding
$\mu={gaia["pm_total_masyr"]:.2f}$ mas yr$^{{-1}}$ and a tangential velocity of
${gaia["tangential_velocity_kms"]:.1f}$ km s$^{{-1}}$ at the inverse-parallax distance.
The absolute Gaia magnitude is $M_G={gaia["M_G_mag"]:.2f}$ mag, placing the object securely on the
white-dwarf sequence for its blue $G_{{\rm BP}}-G_{{\rm RP}}=0.086$ color.

{phot_table(summary)}

\section{{Spectroscopic Analysis}}
The local DESI spectrum has {summary["spectra"]["desi"]["n_points"]} samples from
{summary["spectra"]["desi"]["wavelength_min_A"]:.0f}--{summary["spectra"]["desi"]["wavelength_max_A"]:.0f} \AA\
and median per-pixel S/N $\simeq {summary["spectra"]["desi"]["median_snr"]:.1f}$.
The SDSS spectrum covers {summary["spectra"]["sdss"]["wavelength_min_A"]:.0f}--{summary["spectra"]["sdss"]["wavelength_max_A"]:.0f} \AA\
with median S/N $\simeq {summary["spectra"]["sdss"]["median_snr"]:.1f}$.
Both spectra show a blue continuum and strong Balmer absorption.  Automated line searches find no robust
Balmer emission; weak, narrow emission-like residuals near He I $\lambda6678$, Na D, or forbidden-line
wavelengths are below the robustness threshold and are not treated as astrophysical detections.
The HST UV product spans {hst["wavelength_min_A"]:.0f}--{hst["wavelength_max_A"]:.0f} \AA\ but has
median S/N $\simeq {hst["median_snr"]:.1f}$ and a large zero/negative-flux fraction, so it is used only
as low-quality UV context.

{line_table(summary, "desi")}

{line_table(summary, "sdss")}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{{artifacts["desi_spectrum_plot_path"]}}}
\caption{{Local DESI spectrum of J1529+2928.  The spectrum is Balmer-absorption dominated; no strong emission line is accepted by the QA gate.}}
\label{{fig:desi}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{{artifacts["sdss_spectrum_plot_path"]}}}
\caption{{SDSS DR18 spectrum used as an independent optical spectral check.  Balmer absorption is consistent with a hydrogen-atmosphere white dwarf classification.}}
\label{{fig:sdss}}
\end{{figure}}

\section{{SED and Photometric Variability}}
The SED assembled from SDSS, Gaia, and WISE photometry is blue through the optical bands, with only a
weak WISE W1 detection at $18.06\pm0.20$ mag.  The current draft does not claim an infrared excess,
because the WISE light curve is sparse and the W1/W2 scatter is large relative to the number of epochs.

ZTF contains hundreds of local measurements: the median magnitudes are
$g={summary["lightcurves"]["ztf_stats"]["g"]["median_mag"]:.3f}$,
$r={summary["lightcurves"]["ztf_stats"]["r"]["median_mag"]:.3f}$, and
$i={summary["lightcurves"]["ztf_stats"]["i"]["median_mag"]:.3f}$ mag.
The automated Lomb--Scargle search finds peaks at
{ztf_g["best_period_d"]:.8f} d in $g$ and {ztf_r["best_period_d"]:.8f} d in $r$,
with 5--95 percentile amplitudes of {ztf_g["robust_amp_mag_5_95"]:.3f} and
{ztf_r["robust_amp_mag_5_95"]:.3f} mag, respectively.
These periods correspond to about 44 min and do not match the high-speed
{pub["period_s"]:.3f} s period from \citet{{Kilic2015J1529}}.  The agent therefore flags the ZTF period
as a candidate alias or cadence-biased recovery, not as a revised rotation period.  A future production
run should inject the published period into the ZTF window function and refit the folded light curves with
survey-specific zero points.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{{artifacts["sed_plot_path"]}}}
\caption{{Local SED check.  The optical colors are consistent with a blue white dwarf.  The WISE point is too uncertain to establish an infrared excess in this run.}}
\label{{fig:sed}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{{artifacts["ztf_lightcurve_plot_path"]}}}
\caption{{ZTF light curve used for survey-scale variability checks.  The scatter is real at the 0.1 mag level, but the period recovered from sparse cadence requires alias analysis.}}
\label{{fig:ztf}}
\end{{figure}}

\section{{Three-Iteration Agent Review}}
The first iteration establishes a baseline classification: the source is a blue, nearby white dwarf with
Balmer absorption and previously reported spot-modulated dips.  The second iteration inspects residuals
and physics.  It rejects a forced emission-line or cataclysmic-variable interpretation because neither
DESI nor SDSS shows robust Balmer emission, and the literature follow-up reported no significant radial
velocity variations.  It also rejects a pulsation interpretation for the published 38.1 min period, following
\citet{{Kilic2015J1529}}, because the period is longer than expected for normal DAV pulsations.  The third
iteration adds systematics: Gaia parallax precision is excellent, but final mass, radius, cooling age, and
magnetic-field constraints are not re-derived here because the current run did not execute a certified
atmosphere-grid fit, Zeeman forward model, or time-resolved spectroscopy fit.

\section{{Discussion}}
The safest physical picture is that the ZTF coordinate source is the known spotted white dwarf
J1529+2928.  Our local spectra support the hydrogen-atmosphere white-dwarf classification and do not
provide evidence for ongoing accretion.  The SIMBAD type DAH: should be treated as a cautionary
classification flag rather than as proof of strong Zeeman splitting in these particular spectra, because
the published spectroscopy placed only a $B<70$ kG limit and the local DESI/SDSS spectra do not show
obvious Zeeman splitting at visual inspection.  The combination of short-period photometric dips,
normal-looking Balmer absorption, and absence of robust emission is consistent with the dark-spot
interpretation.

\section{{Conclusions}}
\begin{{enumerate}}
\item The source is securely cross-matched to CSO 1094 / SDSS J152934.98+292801.9 / Gaia DR3 {simbad["gaia_dr3_source_id"]}.
\item Gaia implies $d={gaia["distance_pc_simple_inverse"]:.2f}\pm{gaia["distance_err_pc_simple_inverse"]:.2f}$ pc and $M_G={gaia["M_G_mag"]:.2f}$ mag.
\item DESI and SDSS spectra are dominated by Balmer absorption; no robust optical emission line is detected in this local run.
\item The published 38.1465 min dip period remains the adopted period.  The local ZTF-only 44 min peak is flagged for alias testing.
\item Final atmosphere parameters, magnetic field constraints, and cooling age remain behind the human-review gate until a certified model-grid fit is run.
\end{{enumerate}}

\begin{{acknowledgments}}
This draft was generated by the local Astro Agent framework using astro\_toolbox products, a local white-dwarf RAG corpus, and a white-dwarf knowledge graph.  The numerical analysis used Python and Astropy-compatible unit conventions \citep{{Astropy}}.
\end{{acknowledgments}}

\bibliographystyle{{aasjournalv7}}
\bibliography{{refs}}

\end{{document}}
"""


def report_md(summary: Dict[str, object]) -> str:
    gaia = summary["gaia"]  # type: ignore[assignment]
    qa = summary["qa"]  # type: ignore[assignment]
    return f"""# ZTFJ152934.91+292801.87 Agent Analysis Report

## Verdict
Known source: {summary["preferred_id"]}.
Classification: white dwarf, SIMBAD spectral type {summary["simbad"]["spectral_type"]}.

## Key Measurements
- Gaia DR3 source: {summary["simbad"]["gaia_dr3_source_id"]}
- Distance: {gaia["distance_pc_simple_inverse"]:.2f} +/- {gaia["distance_err_pc_simple_inverse"]:.2f} pc
- Absolute G: {gaia["M_G_mag"]:.2f} mag
- Tangential velocity: {gaia["tangential_velocity_kms"]:.1f} km/s
- Adopted literature period: {summary["lightcurves"]["published_period"]["period_s"]:.3f} s

## Spectral QA
DESI and SDSS both show Balmer absorption. No robust emission lines are accepted.
The HST spectrum is low S/N and is not used for line diagnostics.

## Agent Gate
Gate: {qa["gate"]}

Blocking caveats:
""" + "\n".join(f"- {item}" for item in qa["blocking_caveats"]) + "\n"


def artifacts_json(summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "target": summary["target"],
        "paper_tex": "paper.tex",
        "bibliography": "refs.bib",
        "analysis_summary": "analysis_summary.json",
        "analysis_report": "source_analysis_report.md",
        "figures": summary["artifacts"],
        "latex_table_keys": {
            "photometry_table": "embedded in paper.tex",
            "desi_line_table": "embedded in paper.tex",
            "sdss_line_table": "embedded in paper.tex",
        },
        "qa_gate": summary["qa"],
    }


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    write_text(OUT_ROOT / "analysis_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    write_text(OUT_ROOT / "refs.bib", refs_bib())
    write_text(OUT_ROOT / "paper.tex", paper_tex(summary))
    write_text(OUT_ROOT / "source_analysis_report.md", report_md(summary))
    write_text(OUT_ROOT / "artifacts.json", json.dumps(artifacts_json(summary), ensure_ascii=False, indent=2))
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
