"""Generate the evidence-based AASTeX paper for ZTFJ152934.91+292801.87."""

from __future__ import annotations

import csv
import json
import math
import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "Astro_Agent" / "output" / "analysis_agent" / "ZTFJ152934_research_package_v2"
DATA_ROOT = REPO_ROOT / "Astro_Agent" / "output" / "ZTFJ152934.91+292801.87"
TEMPLATE_ROOT = REPO_ROOT / "Astro_Agent" / "templates" / "aastex"
OUT_ROOT = REPO_ROOT / "Astro_Agent" / "output" / "analysis_agent" / "ZTFJ152934_apj_v2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def tex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
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
        text = text.replace(old, new)
    return text


def bib_key(bibcode: str) -> str:
    return "B" + re.sub(r"[^0-9A-Za-z]+", "", bibcode)


def copy_file(src: Path, dst_dir: Path) -> str:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst.relative_to(OUT_ROOT))


def copy_static() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("aastex701.cls", "aasjournalv7.bst"):
        src = TEMPLATE_ROOT / name
        if src.exists():
            shutil.copy2(src, OUT_ROOT / name)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def spectrum_stats(path: Path, wave_col: str = "wavelength_A") -> dict[str, float]:
    df = pd.read_csv(path)
    wave = pd.to_numeric(df[wave_col], errors="coerce")
    flux = pd.to_numeric(df["flux"], errors="coerce")
    err = pd.to_numeric(df["error"], errors="coerce")
    ok = np.isfinite(flux) & np.isfinite(err) & (err > 0)
    snr = np.nanmedian(np.abs(flux[ok] / err[ok])) if np.any(ok) else np.nan
    return {
        "n": int(np.isfinite(wave).sum()),
        "wmin": float(np.nanmin(wave)),
        "wmax": float(np.nanmax(wave)),
        "median_snr": float(snr),
    }


def lightcurve_stats(path: Path, band_col: str | None = "band") -> dict[str, Any]:
    df = pd.read_csv(path)
    if band_col and band_col in df:
        out: dict[str, Any] = {}
        for band, sub in df.groupby(band_col):
            mag = pd.to_numeric(sub.get("mag"), errors="coerce")
            if mag.notna().sum() == 0:
                continue
            out[str(band)] = {
                "n": int(mag.notna().sum()),
                "median_mag": float(np.nanmedian(mag)),
                "std_mag": float(np.nanstd(mag)),
                "range_mag": float(np.nanmax(mag) - np.nanmin(mag)),
            }
        return out
    flux = pd.to_numeric(df.get("flux"), errors="coerce")
    return {
        "n": int(flux.notna().sum()),
        "median_flux": float(np.nanmedian(flux)),
        "std_flux": float(np.nanstd(flux)),
    }


def parse_orbit_text(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    patterns = {
        "source_id": r"Source ID:\s*(\d+)",
        "parallax": r"Parallax:\s*([0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)\s*mas",
        "distance": r"Distance:\s*([0-9.]+)\s*pc",
        "pmra": r"pmRA:\s*([-0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)",
        "pmde": r"pmDE:\s*([-0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)",
        "best_cluster": r"Cluster:\s*([A-Za-z0-9_+-]+)",
        "best_sep": r"Min separation:\s*([0-9.]+)\s*pc",
    }
    for key, pat in patterns.items():
        match = re.search(pat, text)
        if match:
            out[key] = match.groups() if len(match.groups()) > 1 else match.group(1)
    return out


def previous_study_note(bibcode: str, title: str, mentions: bool) -> str:
    if bibcode == "2015ApJ...814L..31K":
        return "Discovery/source paper: high-speed photometry, SDSS/Gemini spectroscopy, RVs; 2288.792 s dark-spot interpretation."
    if bibcode == "2024ApJ...974...12J":
        return "Massive-WD sample context: model-atmosphere analysis, magnetism, rotation, pulsation and merger-remnant diagnostics."
    if bibcode == "2022AJ....164..131W":
        return "Rapid-rotation comparison sample; cites J1529+2928 among fast rotating/magnetic massive white dwarfs."
    if bibcode == "2021ApJ...923L...6K":
        return "Fast isolated WD rotation context; compares short spin periods and merger-remnant expectations."
    if bibcode == "2020NatAs...4.1092M":
        return "Magnetic/chemical spot analogy; cites J1529+2928 as a WD spot case."
    if "catalog" in title.lower() or "catalogue" in title.lower() or "classification" in title.lower():
        return "Catalogue/classification context; useful for identity, selection function and comparison samples."
    if mentions:
        return "Direct or contextual mention found in extracted PDF text."
    return "SIMBAD-linked broader survey or method reference; no direct source sentence recovered in extracted text."


def build_summary() -> dict[str, Any]:
    package = read_json(PACKAGE_ROOT / "source_research_package.json")
    analysis = read_json(PACKAGE_ROOT / "source_analysis_products.json")
    simbad = read_json(PACKAGE_ROOT / "simbad_all_references.json")
    mentions = {row["bibcode"]: row for row in read_json(PACKAGE_ROOT / "simbad_source_mentions.json")}
    downloads = read_json(PACKAGE_ROOT / "simbad_pdf_downloads.json")
    kg = read_json(PACKAGE_ROOT / "kg_source_relations.json")

    figures = {}
    fig_dir = OUT_ROOT / "figures"
    key_paths = {
        "sed_fit": PACKAGE_ROOT / "analysis_products" / "sed_koester_fit.png",
        "hst_qa": PACKAGE_ROOT / "analysis_products" / "hst_spectrum_qa.png",
        "sdss_lines": PACKAGE_ROOT / "analysis_products" / "sdss_line_fits.png",
        "desi_lines": PACKAGE_ROOT / "analysis_products" / "desi_line_fits.png",
        "combined_spectra": DATA_ROOT / "combined_spectra.png",
        "ztf_lc": DATA_ROOT / "ztf_lightcurve.png",
        "combined_fold": DATA_ROOT / "combined_fold.png",
        "hr_diagram": DATA_ROOT / "hr_diagram.png",
        "rv_ccf": DATA_ROOT / "rv_ccf_desi.png",
    }
    for key, path in key_paths.items():
        if path.exists():
            figures[key] = copy_file(path, fig_dir)

    refs = simbad["references"]
    ref_rows = []
    for idx, ref in enumerate(refs, 1):
        bib = ref.get("bibcode", "")
        title = ref.get("title") or ref.get("Title") or ""
        mention = mentions.get(bib, {})
        ref_rows.append(
            {
                "index": idx,
                "bibcode": bib,
                "key": bib_key(bib),
                "year": ref.get("year") or ref.get("Year") or bib[:4],
                "title": title,
                "journal": ref.get("journal") or ref.get("pub") or "",
                "authors": ref.get("authors") or ref.get("author") or "",
                "pdf_available": bool(mention.get("pdf_available")),
                "mentions_source": bool(mention.get("mentions_source")),
                "note": previous_study_note(bib, title, bool(mention.get("mentions_source"))),
            }
        )
    if not any(row["bibcode"] == "2015ApJ...814L..31K" for row in ref_rows):
        ref_rows.insert(
            0,
            {
                "index": 0,
                "bibcode": "2015ApJ...814L..31K",
                "key": bib_key("2015ApJ...814L..31K"),
                "year": "2015",
                "title": "A Dark Spot on a Massive White Dwarf",
                "journal": "The Astrophysical Journal Letters",
                "authors": "Kilic, Mukremin and Hermes, J. J. and Gianninas, A. and Brown, Warren R.",
                "pdf_available": False,
                "mentions_source": True,
                "note": previous_study_note("2015ApJ...814L..31K", "A Dark Spot on a Massive White Dwarf", True),
            },
        )
        for idx, row in enumerate(ref_rows, start=1):
            row["index"] = idx

    orbit_text = (DATA_ROOT / "orbit_traceback.txt").read_text(encoding="utf-8", errors="ignore")
    orbit = parse_orbit_text(orbit_text)
    rv = pd.read_csv(DATA_ROOT / "rv_analysis.csv")
    best_rv = rv[rv["method"].astype(str).str.contains("best", case=False, na=False)].iloc[0].to_dict()
    if not pd.notna(best_rv.get("ccf_height")):
        source = str(best_rv.get("source", ""))
        ccf = rv[(rv["source"].astype(str) == source) & (rv["method"].astype(str) == "CCF_single")]
        if len(ccf):
            best_rv["ccf_height"] = ccf.iloc[0].get("ccf_height")

    gaia_g = float(pd.read_csv(DATA_ROOT / "sed_photometry.csv").query("band == 'Gaia_G'")["mag"].iloc[0])
    parallax, e_parallax = [float(x) for x in orbit["parallax"]]
    distance = float(orbit["distance"])
    pmra, e_pmra = [float(x) for x in orbit["pmra"]]
    pmde, e_pmde = [float(x) for x in orbit["pmde"]]
    pm_tot = math.hypot(pmra, pmde)
    vt = 4.74047 * (pm_tot / 1000.0) * distance
    abs_g = gaia_g - 5.0 * math.log10(distance / 10.0)

    return {
        "package": package,
        "analysis": analysis,
        "simbad": simbad,
        "downloads": {
            **downloads,
            "n_available_pdf": downloads.get(
                "n_available_pdf",
                sum(1 for row in ref_rows if row.get("pdf_available")),
            ),
        },
        "kg": kg,
        "refs": ref_rows,
        "figures": figures,
        "spectra": {
            "sdss": spectrum_stats(DATA_ROOT / "sdss_spectrum.csv"),
            "desi": spectrum_stats(DATA_ROOT / "desi_spectrum.csv"),
            "hst": analysis["hst"],
        },
        "sed": analysis["sed"],
        "line_fits": analysis["line_fits"],
        "ztf": lightcurve_stats(DATA_ROOT / "ztf_lightcurve.csv"),
        "wise": lightcurve_stats(DATA_ROOT / "wise_lightcurve.csv"),
        "tess": lightcurve_stats(DATA_ROOT / "tess_lightcurve.csv", band_col=None),
        "rv": best_rv,
        "gaia": {
            "source_id": orbit["source_id"],
            "G": gaia_g,
            "BP": 17.543892,
            "RP": 17.457438,
            "BP_RP": 17.543892 - 17.457438,
            "parallax": parallax,
            "e_parallax": e_parallax,
            "distance": distance,
            "pmra": pmra,
            "e_pmra": e_pmra,
            "pmde": pmde,
            "e_pmde": e_pmde,
            "pm_total": pm_tot,
            "vt": vt,
            "M_G": abs_g,
        },
        "orbit": orbit,
    }


def reference_table(refs: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{deluxetable*}{rlllp{0.43\textwidth}}",
        r"\tabletypesize{\scriptsize}",
        r"\tablecaption{All SIMBAD-linked references returned for CSO 1094.}",
        r"\tablehead{\colhead{\#} & \colhead{Bibcode} & \colhead{Year} & \colhead{PDF} & \colhead{Role in this analysis}}",
        r"\startdata",
    ]
    for row in refs:
        pdf = "yes" if row["pdf_available"] else "no"
        role = row["note"]
        if row["mentions_source"]:
            role = "Direct mention recovered. " + role
        lines.append(
            f"{row['index']} & {tex_escape(row['bibcode'])} & {row['year']} & {pdf} & {tex_escape(role)} \\\\"
        )
    lines.extend([r"\enddata", r"\end{deluxetable*}"])
    return "\n".join(lines)


def latex_num(value: Any, fmt: str = ".2f") -> str:
    try:
        number = float(value)
        if not np.isfinite(number):
            return r"\nodata"
        return format(number, fmt)
    except Exception:
        return r"\nodata"


def latex_pm(value: Any, err: Any, fmt: str = ".2f") -> str:
    main = latex_num(value, fmt)
    sigma = latex_num(err, fmt)
    if main == r"\nodata":
        return main
    if sigma == r"\nodata":
        return main
    return rf"{main}$\pm${sigma}"


def line_table(summary: dict[str, Any], survey: str) -> str:
    rows = summary["line_fits"][survey]["rows"]
    lines = [
        r"\begin{deluxetable*}{lrrrrrl}",
        rf"\tablecaption{{{survey} local line fits with formal covariance errors. Positive equivalent width denotes absorption.}}",
        r"\tablehead{\colhead{Line} & \colhead{$\lambda_0$} & \colhead{EW} & \colhead{$\lambda_c$} & \colhead{$v_c$} & \colhead{$\chi^2_\nu$} & \colhead{QA} \\",
        r"\colhead{} & \colhead{(\AA)} & \colhead{(\AA)} & \colhead{(\AA)} & \colhead{km s$^{-1}$} & \colhead{} & \colhead{}}",
        r"\startdata",
    ]
    for row in rows:
        qa = "robust emission" if row.get("robust_emission") else (row.get("rejection_reason") or ("boundary fit" if row.get("boundary_fit") else row.get("kind")))
        lines.append(
            f"{tex_escape(row['line'])} & {latex_num(row.get('rest_A'), '.2f')} & "
            f"{latex_pm(row.get('equivalent_width_A_positive_absorption'), row.get('equivalent_width_err_A'), '.2f')} & "
            f"{latex_pm(row.get('center_A'), row.get('center_err_A'), '.2f')} & "
            f"{latex_pm(row.get('velocity_kms'), row.get('velocity_err_kms'), '.0f')} & "
            f"{latex_num(row.get('fit_reduced_chi2'), '.1f')} & {tex_escape(qa)} \\\\"
        )
    lines.extend(
        [
            r"\enddata",
            r"\tablecomments{The errors are local formal covariance errors from Gaussian-window fits and do not include continuum-placement systematics. Boundary-fit and low-significance emission candidates are rejected by the QA gate.}",
            r"\end{deluxetable*}",
        ]
    )
    return "\n".join(lines)


def ew_sequence(summary: dict[str, Any], survey: str, names: list[str]) -> str:
    by_name = {row.get("line"): row for row in summary["line_fits"][survey]["rows"]}
    values = []
    for name in names:
        row = by_name.get(name, {})
        values.append(latex_pm(row.get("equivalent_width_A_positive_absorption"), row.get("equivalent_width_err_A"), ".1f"))
    return ", ".join(values)


def bibtex(refs: list[dict[str, Any]]) -> str:
    entries = []
    for row in refs:
        author = row["authors"]
        if isinstance(author, list):
            author = " and ".join(str(x) for x in author[:8])
        author = str(author or "{Unknown}")
        author = re.sub(r"\s+et al\.\s*\(\+\d+\)", " and others", author)
        author = author.replace(";", " and ")
        entries.append(
            "\n".join(
                [
                    f"@article{{{row['key']},",
                    f"  author = {{{author}}},",
                    f"  title = {{{row['title'] or row['bibcode']}}},",
                    f"  journal = {{{row['journal'] or 'ADS entry'}}},",
                    f"  year = {{{row['year']}}},",
                    f"  adsurl = {{https://ui.adsabs.harvard.edu/abs/{row['bibcode']}}}",
                    "}",
                ]
            )
        )
    entries.append(
        r"""@article{Astropy2013,
  author = {{Astropy Collaboration} and Robitaille, Thomas P. and Tollerud, Erik J. and others},
  title = {Astropy: A community Python package for astronomy},
  journal = {Astronomy and Astrophysics},
  year = {2013},
  volume = {558},
  pages = {A33}
}"""
    )
    return "\n\n".join(entries) + "\n"


def paper_tex(summary: dict[str, Any]) -> str:
    g = summary["gaia"]
    hst = summary["spectra"]["hst"]
    sed = summary["sed"]
    sdss = summary["spectra"]["sdss"]
    desi = summary["spectra"]["desi"]
    rv = summary["rv"]
    figs = summary["figures"]
    refs = {row["bibcode"]: row["key"] for row in summary["refs"]}
    kilic = refs.get("2015ApJ...814L..31K", "B2015ApJ814L31K")
    jewett = refs.get("2024ApJ...974...12J", "B2024ApJ97412J")
    williams = refs.get("2022AJ....164..131W", "B2022AJ164131W")
    direct_count = sum(1 for row in summary["refs"] if row["mentions_source"])
    robust_sdss = summary["line_fits"]["SDSS"].get("robust_emission_count", 0)
    robust_desi = summary["line_fits"]["DESI"].get("robust_emission_count", 0)
    balmer_lines = ["Halpha", "Hbeta", "Hgamma", "Hdelta", "H-epsilon/CaIIH"]
    sdss_ews = ew_sequence(summary, "SDSS", balmer_lines)
    desi_ews = ew_sequence(summary, "DESI", balmer_lines)
    primary_bands = ", ".join(summary["sed"].get("primary_fit_bands", []))
    context_bands = ", ".join(summary["sed"].get("context_only_bands", []))
    ztf_g = summary["ztf"].get("g", {})
    ztf_r = summary["ztf"].get("r", {})
    ztf_i = summary["ztf"].get("i", {})
    return rf"""\documentclass[twocolumn]{{aastex701}}

\shorttitle{{Re-analysis of J1529+2928}}
\shortauthors{{Chief Investigator Agent}}

\begin{{document}}

\title{{A Literature-Aware Multi-wavelength Re-analysis of the Spotted White Dwarf ZTFJ152934.91+292801.87}}

\author{{Chief Investigator Agent}}
\affiliation{{Automated Astronomy Workflow Laboratory}}
\email{{agent@example.invalid}}

\begin{{abstract}}
We present an evidence-tracked re-analysis of ZTFJ152934.91+292801.87, the SIMBAD source CSO 1094
and SDSS J152934.98+292801.9.  The local SIMBAD export returns {summary["simbad"]["n_refs"]} linked
references; adding the discovery-paper KG relation gives {len(summary["refs"])}
evidence-table entries.  {summary["downloads"]["n_available_pdf"]} have locally available PDFs and
{direct_count} contain source-level or close contextual mentions recovered by the
pipeline.  Gaia astrometry gives $\varpi={g["parallax"]:.3f}\pm{g["e_parallax"]:.3f}$ mas, a distance of
{g["distance"]:.1f} pc, $M_G={g["M_G"]:.2f}$ mag, and tangential velocity {g["vt"]:.1f} km s$^{{-1}}$.
SDSS and DESI spectra show broad Balmer absorption.  Automated Gaussian windows generate several formal
He-line emission candidates, but all are boundary fits; the robust-emission count is zero in both spectra.
The HST spectrum is detected from 1347--2902 \AA, but has median S/N {hst["median_snr"]:.2f} and is used
only as ultraviolet context.  The primary optical-only Koester-grid SED check returns $T_\mathrm{{eff}}={sed["fit"]["teff_sed"]}$ K,
$\log g={sed["fit"]["logg_sed"]:.1f}$, and $R={sed["fit"]["R_Rsun"]:.4f}R_\odot$, but $\chi^2_\nu={sed["fit"]["chi2_sed"]:.1f}$
and systematic residuals make these parameters non-final.  The published 2288.792 s dark-spot period remains
the adopted interpretation.
\end{{abstract}}

\keywords{{white dwarf stars (1799) --- Stellar spots (1572) --- Stellar rotation (1629) --- Spectroscopy (1558) --- Time domain astronomy (2109)}}

\section{{Introduction}}
ZTFJ152934.91+292801.87 is the coordinate label for the known white dwarf CSO 1094 / SDSS
J152934.98+292801.9, hereafter J1529+2928.  The essential prior result is the discovery paper
\citet{{{kilic}}}, which reported eclipse-like dips every 2288.792 s and rejected normal DAV pulsations,
a disintegrating planet, and a close stellar companion in favor of a dark surface spot on a massive white dwarf.
That paper reported SDSS atmosphere parameters $T_\mathrm{{eff}}\simeq11450$ K and $\log g\simeq8.88$,
Gemini radial-velocity non-variability, no obvious Zeeman splitting with $B<70$ kG, and a possible weak Ca K
feature.  Later work places the object in broader samples of massive, rotating, magnetic, or spotted white
dwarfs \citep[e.g.][]{{{jewett},{williams}}}.

The goal here is not to supersede the discovery paper from survey-cadence data alone.  Instead, we test a
local agent workflow that must read SIMBAD, inspect all linked references, query a white-dwarf RAG database
and knowledge graph, run the local astro\_toolbox products, fit spectra and SED residuals, and stop where the
evidence is not strong enough for a final physical parameter claim.

\section{{SIMBAD and Literature Context}}
The pipeline recovered {summary["simbad"]["n_refs"]} references from the local SIMBAD export for CSO 1094,
and the source-specific KG adds the discovery paper, giving {len(summary["refs"])} evidence-table entries.
The local PDF store contains {summary["downloads"]["n_available_pdf"]} available PDFs; the remaining references have no
ADS/arXiv source available through the configured downloader.  The knowledge graph returns 29 source-related
relations, dominated by the discovery-paper links to SDSS, Gemini, K2, photometry, spectroscopy, and radial
velocity analysis, plus later method-transfer links to periodograms, cooling sequences, TESS/ZTF comparison,
and rotation/magnetism samples.

The literature reading changes the model-selection logic.  Because \citet{{{kilic}}} already showed that the
38.1 min signal is too long for normal DAV pulsations and too stable for a disintegrating planet, the agent
does not fit a generic eclipse or pulsation model first.  It instead treats the source as a spotted, massive
hydrogen-atmosphere white dwarf and uses new survey products as consistency checks.

{reference_table(summary["refs"])}

\section{{Data Products}}
The local astro\_toolbox directory contains SDSS, DESI, HST, SPHEREx, ZTF, WISE, and TESS products.  The
Gaia astrometric solution used in the orbit module is source {g["source_id"]}.  With $G={g["G"]:.3f}$ mag and
$G_\mathrm{{BP}}-G_\mathrm{{RP}}={g["BP_RP"]:.3f}$ mag, the inverse-parallax absolute magnitude is
$M_G={g["M_G"]:.2f}$ mag.  The proper motion components are
$\mu_\alpha={g["pmra"]:.2f}\pm{g["e_pmra"]:.2f}$ and
$\mu_\delta={g["pmde"]:.2f}\pm{g["e_pmde"]:.2f}$ mas yr$^{{-1}}$.

The SDSS spectrum has {sdss["n"]} points across {sdss["wmin"]:.0f}--{sdss["wmax"]:.0f} \AA\ with median
S/N {sdss["median_snr"]:.1f}.  The DESI product has {desi["n"]} points across
{desi["wmin"]:.0f}--{desi["wmax"]:.0f} \AA\ with median S/N {desi["median_snr"]:.1f}.  The HST product
contains {hst["n_points"]} points across {hst["wavelength_min_A"]:.0f}--{hst["wavelength_max_A"]:.0f} \AA,
but its median S/N is only {hst["median_snr"]:.2f} and {100*hst["zero_or_negative_fraction"]:.1f}\% of the
pixels are zero or negative, so we do not use it for abundance or line-profile inference.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.95\textwidth]{{{figs["combined_spectra"]}}}
\caption{{Combined local spectra.  SDSS and DESI show broad Balmer absorption; the HST UV data are useful as coverage evidence but fail the S/N gate for detailed abundance fitting.}}
\label{{fig:combined-spectra}}
\end{{figure*}}

\section{{Spectral Lines and Radial Velocity}}
We fit local Gaussian windows to H$\alpha$ through H$\epsilon$/Ca~II~H and selected He lines in both SDSS and
DESI.  Both spectra are Balmer-absorption dominated.  In SDSS the measured absorption equivalent widths are
{sdss_ews} \AA\ for H$\alpha$, H$\beta$, H$\gamma$, H$\delta$, and H$\epsilon$/Ca~II~H,
respectively.  DESI gives the same qualitative sequence, with {desi_ews} \AA.
There are {robust_sdss} robust SDSS emission lines and {robust_desi} robust DESI emission lines.  The formal
He emission candidates are rejected because their centroids or widths hit the fit bounds or their local
significance fails the QA threshold.  These formal errors come from the local covariance matrix and are not
treated as a full line-profile systematic budget.

{line_table(summary, "SDSS")}

{line_table(summary, "DESI")}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.48\textwidth]{{{figs["sdss_lines"]}}}
\includegraphics[width=0.48\textwidth]{{{figs["desi_lines"]}}}
\caption{{Automated line-window fits.  The Balmer series is absorption dominated.  Apparent He-window emission solutions are boundary fits and are not accepted as physical emission lines.}}
\label{{fig:linefits}}
\end{{figure*}}

The RV module reports a DESI cross-correlation velocity of {float(rv["rv_kms"]):.1f}$\pm${float(rv["rv_err_kms"]):.1f}
km s$^{{-1}}$ as the best local value, with CCF height {float(rv["ccf_height"]):.3f}.  We do not use the HST
cross-correlation velocity because its CCF height is low and the spectrum has poor S/N.  A single-epoch RV
consistency check is not a binary-orbit solution, but it does not contradict the discovery-paper statement that
Gemini spectroscopy found no significant RV variations.

\section{{SED and Multi-band Photometry}}
The SED table combines SDSS, Gaia, WISE, and locally labeled SPHEREx-band fluxes.  To avoid the previous
failure mode of mixing heterogeneous products into a single atmosphere fit, the primary fit uses only the
validated optical bands ({tex_escape(primary_bands)}).  The context-only bands are {tex_escape(context_bands)};
the SPHEREx-labeled points are provisional local-tool products unless independently validated against public
survey provenance, and WISE is kept as infrared-context evidence rather than as a photospheric fit constraint.
A scaled Koester DA grid fit is shown in
Figure~\ref{{fig:sed}}.  The formal best grid point is $T_\mathrm{{eff}}={sed["fit"]["teff_sed"]}$ K and
$\log g={sed["fit"]["logg_sed"]:.1f}$ with $R={sed["fit"]["R_Rsun"]:.4f}R_\odot$ at the Gaia distance.  However,
the reduced $\chi^2$ is {sed["fit"]["chi2_sed"]:.1f}.  We therefore treat this as a failed or at least
systematically incomplete SED model, not a final atmosphere fit.  The all-band fit is retained in the JSON
audit trail for comparison but is not used for physical parameters.

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{{figs["sed_fit"]}}}
\caption{{Koester-grid SED check with residuals.  The high $\chi^2_\nu$ and wavelength-dependent residuals trigger the systematics gate; final $T_\mathrm{{eff}}$, $\log g$, mass, radius, and cooling age should not be quoted from this fit alone.}}
\label{{fig:sed}}
\end{{figure}}

\section{{Time-domain Checks}}
The ZTF light curve contains {ztf_g.get("n", 0)} $g$, {ztf_r.get("n", 0)} $r$, and {ztf_i.get("n", 0)} $i$
measurements.  The median magnitudes are $g={ztf_g.get("median_mag", float("nan")):.3f}$,
$r={ztf_r.get("median_mag", float("nan")):.3f}$, and $i={ztf_i.get("median_mag", float("nan")):.3f}$ mag.
The local period plots include a ZTF-$g$ solution near 0.0265 d, close to the published 38.1465 min period,
and a ZTF-$i$ solution near 0.0516 d, likely a harmonic or cadence-related solution.  The TESS period product
near 0.0028 d is not adopted because it is much shorter than the literature period and requires a dedicated
window-function analysis.  Thus the literature period remains the adopted physical period.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.48\textwidth]{{{figs["ztf_lc"]}}}
\includegraphics[width=0.48\textwidth]{{{figs["combined_fold"]}}}
\caption{{Survey variability products.  ZTF recovers short-period variability, but the sparse survey cadence requires comparison to the published high-speed photometry before revising the adopted period.}}
\label{{fig:variability}}
\end{{figure*}}

\section{{Three-Iteration Agent Review}}
\textit{{Iteration 1, baseline.}}  SIMBAD, SDSS, DESI, HST, SED, ZTF, WISE, TESS, RV, and orbit-traceback
products were ingested.  The source is a nearby blue white dwarf and matches the known J1529+2928 system.
\textit{{Iteration 2, residuals and physics.}}  Forced emission-line and accreting-binary interpretations are
rejected because SDSS/DESI show no robust emission lines.  A normal DAV interpretation is rejected following
\citet{{{kilic}}}; the 38.1 min period is too long for known DAV pulsations.  The SED model is not accepted as a
final atmosphere solution because $\chi^2_\nu$ is high.  \textit{{Iteration 3, systematics.}}  The final mass,
cooling age, magnetic-field strength, and abundance constraints remain behind a human-review gate.  The HST UV
coverage is real but low S/N, and the RV/orbit traceback products require multi-epoch validation.

\section{{Discussion and Conclusions}}
The strongest conclusion is conservative: ZTFJ152934.91+292801.87 is the known spotted white dwarf
J1529+2928.  The local spectra support the hydrogen-atmosphere white-dwarf classification and do not reveal
reliable optical emission.  The HST spectrum is present and has been inspected, but it is not strong enough for
new UV abundance constraints.  The SED fit highlights the need for a better model treatment, most likely using
the discovery-paper spot picture and wavelength-dependent surface inhomogeneity rather than a single-temperature
DA photosphere.  The discovery-paper 2288.792 s period and dark-spot interpretation remain the appropriate
baseline for any future model.

\begin{{acknowledgments}}
This paper was generated from the local Astro Agent workflow, astro\_toolbox products, SIMBAD-linked references,
the local white-dwarf RAG database, and a SQLite-indexed knowledge graph.  The analysis used Python and
Astropy-compatible unit conventions \citep{{Astropy2013}}.
\end{{acknowledgments}}

\bibliographystyle{{aasjournalv7}}
\bibliography{{refs}}

\end{{document}}
"""


def main() -> None:
    global PACKAGE_ROOT, DATA_ROOT, OUT_ROOT
    parser = argparse.ArgumentParser(description="Generate evidence-based AASTeX paper for ZTFJ152934.")
    parser.add_argument("--package-root", default=str(PACKAGE_ROOT))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    PACKAGE_ROOT = Path(args.package_root).resolve()
    DATA_ROOT = Path(args.data_root).resolve()
    OUT_ROOT = Path(args.output_root).resolve()
    copy_static()
    summary = build_summary()
    write_text(OUT_ROOT / "analysis_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    write_text(OUT_ROOT / "refs.bib", bibtex(summary["refs"]))
    write_text(OUT_ROOT / "paper.tex", paper_tex(summary))
    write_text(
        OUT_ROOT / "README.md",
        "# ZTFJ152934.91+292801.87 ApJ v2\n\n"
        f"- Source package: `{PACKAGE_ROOT}`\n"
        f"- Data root: `{DATA_ROOT}`\n"
        "- Main TeX: `paper.tex`\n"
        "- SED and line-fit figures are copied into `figures/`.\n",
    )
    print(OUT_ROOT)


if __name__ == "__main__":
    main()
