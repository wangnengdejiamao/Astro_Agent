#!/usr/bin/env python3
"""Integrate DWD candidate products into astro_output tables and paper figures."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OPTICAL_SPECTRUM_FILES = {
    "sdss": "sdss_spectrum.csv",
    "desi": "desi_spectrum.csv",
    "hst": "hst_spectrum.csv",
    "jwst": "jwst_spectrum.csv",
    "lamost": "lamost_spectrum.csv",
    "koa": "koa_spectrum.csv",
}

LC_FILES = {
    "ztf": "ztf_lightcurve.csv",
    "wise": "wise_lightcurve.csv",
    "tess": "tess_lightcurve.csv",
    "kepler": "kepler_lightcurve.csv",
    "gaia": "gaia_lightcurve.csv",
    "hst": "hst_lightcurve.csv",
    "jwst": "jwst_lightcurve.csv",
}

NEW_SOURCE_TARGETS = [
    "ZTFJ171532.47-194407.03",
    "ZTFJ184551.87-255127.65",
    "ZTFJ075052.56-082854.12",
    "ZTFJ235115.39+630527.72",
    "ZTFJ150822.14+432245.78",
    "ZTFJ150514.54+070102.09",
    "ZTFJ170033.08+645154.41",
    "ZTFJ132153.54+254309.27",
]

NEW_SOURCE_NOTES = {
    "ZTFJ171532.47-194407.03": (
        "主推。P=23.04 min, EW, high CNN probability, SIMBAD no match; "
        "a strong ultracompact unstudied candidate, but spectroscopy is still missing."
    ),
    "ZTFJ184551.87-255127.65": (
        "主推。P=23.04 min, EA-like, SIMBAD no match; eclipse-like morphology makes "
        "the photometric period more likely to be close to Porb, but needs confirmation."
    ),
    "ZTFJ075052.56-082854.12": (
        "短周期补充。P=27.36 min and unstudied, but CNN probability is low; "
        "use as a cautious candidate requiring visual light-curve vetting."
    ),
    "ZTFJ235115.39+630527.72": (
        "主推。P=43.20 min, EW, high CNN probability, SIMBAD no match; "
        "good clean follow-up target despite no current spectrum."
    ),
    "ZTFJ150822.14+432245.78": (
        "可马上写。Unstudied with SDSS spectrum and high CNN probability; "
        "use as a spectroscopic example of a new candidate."
    ),
    "ZTFJ150514.54+070102.09": (
        "可马上写。Unstudied with SDSS spectrum and high CNN probability; "
        "good companion example to ZTFJ150822."
    ),
    "ZTFJ170033.08+645154.41": (
        "次优先。Unstudied with SDSS spectrum; lower CNN probability but useful "
        "as a spectroscopic follow-up-ready candidate."
    ),
    "ZTFJ132153.54+254309.27": (
        "次优先。Unstudied with SDSS spectrum; lower CNN probability, useful "
        "for a small comparison set rather than headline claim."
    ),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_maybe(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def batch_dir_to_target(name: str) -> str:
    """Convert batch_output/spectra directory names back to catalog target names."""
    if "p" not in name:
        return name
    return name.replace("p", "+", 1)


def valid_spectrum(path: Path, min_rows: int = 20) -> bool:
    df = read_csv_maybe(path)
    if df is None or len(df) < min_rows:
        return False
    cols = {c.lower() for c in df.columns}
    return (
        ("wavelength" in cols or "wavelength_a" in cols or "wave" in cols)
        and any(c in cols for c in ("flux", "flux_cgs", "flam"))
    )


def spectrum_row_count(path: Path) -> int:
    df = read_csv_maybe(path)
    return 0 if df is None else int(len(df))


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
        table[col] = table[col].str.replace("|", r"\|", regex=False)
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()]
    return "\n".join([header, sep, *rows])


def copy_batch_spectra(batch_spectra: Path, astro_output: Path, catalog_targets: set[str]) -> list[dict]:
    copied = []
    if not batch_spectra.exists():
        return copied
    for src_dir in sorted(p for p in batch_spectra.iterdir() if p.is_dir()):
        target = batch_dir_to_target(src_dir.name)
        if target not in catalog_targets:
            continue
        dst_dir = ensure_dir(astro_output / target)
        for src in sorted(src_dir.glob("*spectrum.csv")):
            dst = dst_dir / src.name
            action = "exists"
            if not dst.exists():
                shutil.copy2(src, dst)
                action = "copied"
            copied.append({
                "target": target,
                "source_path": str(src),
                "dest_path": str(dst),
                "action": action,
            })
    return copied


def copy_selected_all_tools(all_tools_dir: Path, astro_output: Path) -> list[dict]:
    """Copy compact products from a single-target all_tools run into astro_output."""
    copied = []
    if not all_tools_dir.exists():
        return copied
    info_path = all_tools_dir / "target_info.json"
    target = None
    if info_path.exists():
        try:
            import json
            target = json.loads(info_path.read_text(encoding="utf-8")).get("target", "")
            target = str(target).replace(" ", "")
        except Exception:
            target = None
    if not target:
        m = re.search(r"all_tools_(ZTFJ[0-9.+-]+)", all_tools_dir.name)
        target = m.group(1) if m else None
    if not target:
        return copied
    dst_dir = ensure_dir(astro_output / target)
    patterns = [
        ("ztf/ztf_lightcurve.csv", "ztf_lightcurve.csv"),
        ("ztf/ztf_lightcurve.png", "ztf_lightcurve.png"),
        ("wise/wise_lightcurve.csv", "wise_lightcurve.csv"),
        ("wise/wise_lightcurve.png", "wise_lightcurve.png"),
        ("sed/sed_photometry.csv", "sed_photometry.csv"),
        ("sed/sed.png", "sed.png"),
        ("sed/sed_diagnostics.txt", "sed_diagnostics.txt"),
        ("hr_diagram/hr_diagram_params.csv", "hr_diagram_params.csv"),
        ("hr_diagram/hr_diagram.png", "hr_diagram.png"),
        ("period_analysis/period_analysis.csv", "period_analysis.csv"),
        ("period_analysis/ZTF_g_period.png", "ZTF_g_period.png"),
        ("period_analysis/ZTF_r_period.png", "ZTF_r_period.png"),
        ("combined_plots/combined_fold.png", "combined_fold.png"),
        ("combined_plots/spectra_with_photometry.png", "spectra_with_photometry.png"),
        ("cooling_age/cooling_age_analysis.txt", "cooling_age_analysis.txt"),
        ("rv/rv_analysis.csv", "rv_analysis.csv"),
        ("rv/rv_analysis.txt", "rv_analysis.txt"),
        ("koa/koa_result.json", "koa_result.json"),
        ("run_summary.json", "all_tools_run_summary.json"),
    ]
    for rel_src, rel_dst in patterns:
        src = all_tools_dir / rel_src
        if not src.exists():
            continue
        dst = dst_dir / rel_dst
        action = "exists"
        if not dst.exists():
            shutil.copy2(src, dst)
            action = "copied"
        copied.append({
            "target": target,
            "source_path": str(src),
            "dest_path": str(dst),
            "action": action,
        })
    raw_paths = sorted((all_tools_dir / "koa" / "work" / "download" / "lris" / "lev0").glob("*.fits"))
    if raw_paths:
        link_path = dst_dir / "koa_raw_files.txt"
        link_path.write_text("\n".join(str(p) for p in raw_paths) + "\n", encoding="utf-8")
        copied.append({
            "target": target,
            "source_path": str(all_tools_dir / "koa" / "work"),
            "dest_path": str(link_path),
            "action": "listed",
        })
    return copied


def read_koa_inventory(koa_batch: Path) -> dict[str, dict]:
    inv: dict[str, dict] = {}
    summary_path = koa_batch / "koa_batch_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        for _, row in df.iterrows():
            target = str(row.get("target", "")).strip()
            if target:
                inv.setdefault(target, {})["koa_selected_rows"] = row.get("n_selected_rows")
                inv[target]["koa_metadata_rows"] = row.get("n_metadata_rows")
                inv[target]["koa_status"] = row.get("status")
                inv[target]["koa_downloaded_files"] = row.get("n_downloaded_files")
    reduction_path = koa_batch / "koa_reduction_summary.csv"
    if reduction_path.exists():
        df = pd.read_csv(reduction_path)
        for _, row in df.iterrows():
            target = str(row.get("target", row.get("safe_target", ""))).strip()
            if target:
                inv.setdefault(target, {})["koa_reduction_status"] = row.get("status")
                inv[target]["koa_setup_status"] = row.get("setup_status", "")
                inv[target]["koa_spectrum_status"] = row.get("spectrum_status", "")
                inv[target]["koa_reduction_message"] = row.get("message", "")
                inv[target]["koa_spectrum_path"] = row.get("spectrum_csv", row.get("saved_csv", ""))
                inv[target]["koa_spectrum_png"] = row.get("spectrum_png", "")
                inv[target]["koa_spectrum_report"] = row.get("spectrum_report", "")
                inv[target]["koa_spectrum_exposures"] = row.get("spectrum_exposures", "")
                inv[target]["koa_spectrum_points"] = row.get("n_spectrum_points", np.nan)
    observed_path = koa_batch / "koa_lris_observed_targets.csv"
    if observed_path.exists():
        df = pd.read_csv(observed_path)
        for _, row in df.iterrows():
            target = str(row.get("input_target", "")).strip()
            if target:
                inv.setdefault(target, {})["koa_lris_exposures"] = row.get("n_koa_lris_exposures")
                inv[target]["koa_date_obs_min"] = row.get("date_obs_min")
                inv[target]["koa_date_obs_max"] = row.get("date_obs_max")
                inv[target]["koa_date_obs_list"] = row.get("date_obs_list")
                inv[target]["koa_program_ids"] = row.get("program_ids")
                inv[target]["koa_program_pis"] = row.get("program_pis")
    manifest_path = koa_batch / "koa_file_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        if "target" in df.columns:
            for target, sub in df.groupby("target"):
                inv.setdefault(str(target), {})["koa_downloaded_files"] = len(sub)
    return inv


def copy_koa_spectra(koa_batch: Path, astro_output: Path,
                     catalog_targets: set[str], koa_inv: dict[str, dict]) -> tuple[list[dict], pd.DataFrame]:
    copied = []
    rows = []
    all_targets = set(koa_inv)
    if koa_batch.exists():
        all_targets.update(p.parent.parent.name for p in koa_batch.glob("*/spectrum/koa_spectrum.csv"))
    for target in sorted(t for t in all_targets if t in catalog_targets):
        info = koa_inv.get(target, {})
        target_dir = ensure_dir(astro_output / target)
        spectrum_csv = info.get("koa_spectrum_path", "")
        if not spectrum_csv or str(spectrum_csv).lower() == "nan":
            candidate = koa_batch / target / "spectrum" / "koa_spectrum.csv"
            spectrum_csv = str(candidate) if candidate.exists() else ""
        src_csv = Path(str(spectrum_csv)) if spectrum_csv else None
        has_file = bool(src_csv and src_csv.exists())
        n_rows = spectrum_row_count(src_csv) if has_file else 0
        usable = bool(has_file and valid_spectrum(src_csv))
        copied_paths = []
        if has_file:
            copy_plan = [
                (src_csv, target_dir / "koa_spectrum.csv"),
                (src_csv.with_suffix(".png"), target_dir / "koa_spectrum.png"),
                (src_csv.with_name("koa_spectrum_report.txt"), target_dir / "koa_spectrum_report.txt"),
                (src_csv.with_name("koa_exposures.csv"), target_dir / "koa_exposures.csv"),
            ]
            for src, dst in copy_plan:
                if not src.exists():
                    continue
                action = "exists"
                if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(src, dst)
                    action = "copied"
                copied.append({
                    "target": target,
                    "source_path": str(src),
                    "dest_path": str(dst),
                    "action": action,
                })
                copied_paths.append(str(dst))
        rows.append({
            "target": target,
            "has_koa_lris_metadata": pd.notna(info.get("koa_metadata_rows")) and pd.to_numeric(info.get("koa_metadata_rows"), errors="coerce") > 0,
            "koa_metadata_rows": info.get("koa_metadata_rows", np.nan),
            "koa_selected_rows": info.get("koa_selected_rows", np.nan),
            "koa_downloaded_files": info.get("koa_downloaded_files", np.nan),
            "koa_lris_exposures": info.get("koa_lris_exposures", np.nan),
            "koa_date_obs_min": info.get("koa_date_obs_min", ""),
            "koa_date_obs_max": info.get("koa_date_obs_max", ""),
            "koa_program_ids": info.get("koa_program_ids", ""),
            "koa_program_pis": info.get("koa_program_pis", ""),
            "koa_reduction_status": info.get("koa_reduction_status", ""),
            "koa_setup_status": info.get("koa_setup_status", ""),
            "koa_spectrum_status": info.get("koa_spectrum_status", ""),
            "koa_spectrum_file_exists": has_file,
            "koa_spectrum_rows": n_rows,
            "koa_spectrum_usable_1d": usable,
            "koa_spectrum_source_path": str(src_csv) if has_file else "",
            "koa_copied_paths": ";".join(copied_paths),
            "koa_reduction_message": info.get("koa_reduction_message", ""),
        })
    inventory = pd.DataFrame(rows)
    inventory.to_csv(astro_output / "koa_spectral_inventory.csv", index=False)
    return copied, inventory


def read_paper_auxiliary(paper_pdf: str) -> dict[str, dict[str, dict]]:
    """Load useful tables already produced in the paper directory, if present."""
    aux: dict[str, dict[str, dict]] = {"dwd": {}, "desi": {}, "sed": {}, "gw": {}}
    if not paper_pdf:
        return aux
    paper_dir = Path(paper_pdf).parent

    dwd_path = paper_dir / "desi和双白矮星样本" / "DWD.csv"
    df = read_csv_maybe(dwd_path)
    if df is not None and "FirstColumn_23chars" in df.columns:
        for _, row in df.iterrows():
            target = str(row.get("FirstColumn_23chars", "")).strip()
            if target:
                aux["dwd"][target] = row.to_dict()

    desi_path = paper_dir / "desi和双白矮星样本" / "matched_dwd_with_desi.csv"
    df = read_csv_maybe(desi_path)
    if df is not None:
        for _, row in df.iterrows():
            target = ""
            img = str(row.get("img_path", ""))
            m = re.search(r"(ZTF\s?J[0-9.+-]+)", img)
            if m:
                target = m.group(1).replace(" ", "")
            if target:
                aux["desi"][target] = row.to_dict()

    sed_path = paper_dir / "sed_results.csv"
    df = read_csv_maybe(sed_path)
    if df is not None and "name" in df.columns:
        for _, row in df.iterrows():
            target = str(row.get("name", "")).strip()
            if target:
                aux["sed"][target] = row.to_dict()

    gw_path = paper_dir / "gw_results.csv"
    df = read_csv_maybe(gw_path)
    if df is not None and "name" in df.columns:
        for _, row in df.iterrows():
            target = str(row.get("name", "")).strip()
            if target:
                aux["gw"][target] = row.to_dict()
    return aux


def auxiliary_metrics(target: str, aux: dict[str, dict[str, dict]], row: dict) -> dict:
    out = {
        "distance_source": "",
        "desi_teff_k": np.nan,
        "desi_logg": np.nan,
        "desi_sn_r": np.nan,
        "gw_p_orb_min_assumed": np.nan,
        "gw_f_mHz_assumed": np.nan,
        "gw_A_assumed_mc03": np.nan,
        "gw_h_assumed_mc03": np.nan,
        "gw_hc_lisa_assumed_mc03": np.nan,
        "gw_snr_tianqin_assumed_mc03": np.nan,
        "gw_snr_lisa_assumed_mc03": np.nan,
        "gw_snr_taiji_assumed_mc03": np.nan,
        "gw_snr_decigo_assumed_mc03": np.nan,
        "gw_snr_net_assumed_mc03": np.nan,
    }
    desi = aux.get("desi", {}).get(target, {})
    sed = aux.get("sed", {}).get(target, {})
    dwd = aux.get("dwd", {}).get(target, {})
    gw = aux.get("gw", {}).get(target, {})

    for src, dst in [
        ("DESI_TEFF", "desi_teff_k"),
        ("DESI_LOGG", "desi_logg"),
        ("DESI_SN_R", "desi_sn_r"),
    ]:
        value = pd.to_numeric(desi.get(src), errors="coerce")
        if pd.notna(value):
            out[dst] = float(value)

    if pd.isna(row.get("parallax_mas")):
        for source in (desi.get("DESI_Plx"), dwd.get("plx")):
            value = pd.to_numeric(source, errors="coerce")
            if pd.notna(value) and value > 0:
                out["parallax_mas"] = float(value)
                break

    if pd.isna(row.get("distance_pc")):
        distance_candidates = [
            ("paper_sed_d_pc", sed.get("d_pc")),
            ("paper_desi_rgeo", desi.get("DESI_rgeo")),
        ]
        for label, source in distance_candidates:
            value = pd.to_numeric(source, errors="coerce")
            if pd.notna(value) and value > 0:
                out["distance_pc"] = float(value)
                out["distance_source"] = label
                break
        if "distance_pc" not in out:
            plx = pd.to_numeric(out.get("parallax_mas", row.get("parallax_mas")), errors="coerce")
            if pd.notna(plx) and plx > 0:
                out["distance_pc"] = float(1000.0 / plx)
                out["distance_source"] = "paper_dwd_1_over_parallax"
    else:
        out["distance_source"] = "hr_diagram"

    gw_map = [
        ("p_orb_min", "gw_p_orb_min_assumed"),
        ("f_GW_mHz", "gw_f_mHz_assumed"),
        ("A", "gw_A_assumed_mc03"),
        ("h", "gw_h_assumed_mc03"),
        ("hc_LISA", "gw_hc_lisa_assumed_mc03"),
        ("snr_TQ", "gw_snr_tianqin_assumed_mc03"),
        ("snr_LISA", "gw_snr_lisa_assumed_mc03"),
        ("snr_Taiji", "gw_snr_taiji_assumed_mc03"),
        ("snr_DECIGO", "gw_snr_decigo_assumed_mc03"),
        ("snr_net", "gw_snr_net_assumed_mc03"),
    ]
    for src, dst in gw_map:
        value = pd.to_numeric(gw.get(src), errors="coerce")
        if pd.notna(value):
            out[dst] = float(value)
    return out


def parse_summary_flags(summary_path: Path) -> dict[str, bool]:
    flags = {}
    if not summary_path.exists():
        return flags
    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = re.match(r"([^:]+):\s+有数据", line)
        if m:
            flags[m.group(1).strip()] = True
        m = re.match(r"([^:]+):\s+无数据", line)
        if m:
            flags[m.group(1).strip()] = False
    return flags


def parse_period_from_files(target_dir: Path, default_period: float) -> tuple[float, str]:
    default = float(default_period) if pd.notna(default_period) else np.nan
    candidates: list[tuple[float, str]] = []
    pa_path = target_dir / "period_analysis.csv"
    df = read_csv_maybe(pa_path)
    if df is not None and "best_period_day" in df.columns:
        det = df[df.get("detected", False).astype(str).str.lower().isin(["true", "1"])]
        for _, row in det.iterrows():
            p = pd.to_numeric(row.get("best_period_day"), errors="coerce")
            if pd.notna(p) and p > 0:
                candidates.append((float(p), str(row.get("curve", "period_analysis"))))
        if candidates and np.isfinite(default) and default > 0:
            p, label = min(candidates, key=lambda item: abs(math.log(item[0] / default)))
            if 0.5 <= p / default <= 2.0:
                return p, label
        elif candidates:
            row = det.iloc[0]
            return float(row["best_period_day"]), str(row.get("curve", "period_analysis"))
    for path in sorted(target_dir.glob("combined_fold_P*d_*.png")):
        m = re.search(r"combined_fold_P([0-9.]+)d_([^/]+)\.png", path.name)
        if not m:
            continue
        p = float(m.group(1))
        label = m.group(2)
        candidates.append((p, label))
    if candidates and np.isfinite(default) and default > 0:
        p, label = min(candidates, key=lambda item: abs(math.log(item[0] / default)))
        if 0.5 <= p / default <= 2.0:
            return p, label
    elif candidates:
        return candidates[0]
    return default, "catalog"


def sed_metrics(sed_path: Path) -> dict:
    out = {
        "sed_bands": "",
        "n_sed_bands": 0,
        "has_gaia_phot": False,
        "has_galex_phot": False,
        "has_wise_phot": False,
        "has_sdss_phot": False,
        "has_spherex_phot": False,
        "max_uv_excess_dex": np.nan,
        "max_ir_excess_dex": np.nan,
        "sed_excess_flag": "",
    }
    df = read_csv_maybe(sed_path)
    if df is None or "band" not in df.columns:
        return out
    out["sed_bands"] = ";".join(map(str, df["band"].tolist()))
    out["n_sed_bands"] = int(len(df))
    bands = df["band"].astype(str)
    out["has_gaia_phot"] = bool(bands.str.startswith("Gaia").any())
    out["has_galex_phot"] = bool(bands.str.startswith("GALEX").any())
    out["has_wise_phot"] = bool(bands.str.startswith("WISE").any())
    out["has_sdss_phot"] = bool(bands.str.startswith("SDSS").any())
    out["has_spherex_phot"] = bool(bands.str.startswith("SPHEREx").any())
    if not {"wave_A", "flux_cgs"}.issubset(df.columns):
        return out
    data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["wave_A", "flux_cgs"])
    data = data[(data["wave_A"] > 0) & (data["flux_cgs"] > 0)].copy()
    if len(data) < 3:
        return out
    x = np.log10(data["wave_A"].astype(float).values)
    y = np.log10((data["wave_A"] * data["flux_cgs"]).astype(float).values)
    wave = data["wave_A"].astype(float).values
    optical = (wave >= 3000) & (wave <= 10000)
    try:
        coeff = np.polyfit(x[optical] if optical.sum() >= 2 else x,
                           y[optical] if optical.sum() >= 2 else y, 1)
        residual = y - np.polyval(coeff, x)
        uv = wave < 3000
        ir = wave > 10000
        if uv.any():
            out["max_uv_excess_dex"] = float(np.nanmax(residual[uv]))
        if ir.any():
            out["max_ir_excess_dex"] = float(np.nanmax(residual[ir]))
        flags = []
        if np.isfinite(out["max_uv_excess_dex"]) and out["max_uv_excess_dex"] > 0.30:
            flags.append("UV_EXCESS")
        if np.isfinite(out["max_ir_excess_dex"]) and out["max_ir_excess_dex"] > 0.30:
            flags.append("IR_EXCESS")
        out["sed_excess_flag"] = ";".join(flags)
    except Exception:
        pass
    return out


def rv_metrics(rv_path: Path) -> dict:
    out = {"best_rv_kms": np.nan, "best_rv_err_kms": np.nan,
           "best_rv_method": "", "rv_valid_for_dynamics": False}
    df = read_csv_maybe(rv_path)
    if df is None:
        return out
    row = None
    if "method" in df.columns:
        best = df[df["method"].astype(str).str.lower() == "best"]
        if len(best):
            row = best.iloc[0]
    if row is None and len(df):
        row = df.iloc[-1]
    if row is None:
        return out
    rv = pd.to_numeric(row.get("rv_kms"), errors="coerce")
    err = pd.to_numeric(row.get("rv_err_kms"), errors="coerce")
    out["best_rv_kms"] = float(rv) if pd.notna(rv) else np.nan
    out["best_rv_err_kms"] = float(err) if pd.notna(err) else np.nan
    out["best_rv_method"] = str(row.get("source", row.get("method", "")))
    out["rv_valid_for_dynamics"] = bool(np.isfinite(out["best_rv_kms"]) and abs(out["best_rv_kms"]) < 1500)
    return out


def hr_metrics(target_dir: Path) -> dict:
    out = {"gaia_source_id": "", "parallax_mas": np.nan, "distance_pc": np.nan,
           "Gmag": np.nan, "BP_RP": np.nan, "M_G": np.nan,
           "wd_mass_msun": np.nan, "wd_teff_k": np.nan,
           "wd_logg": np.nan, "wd_cooling_age_gyr": np.nan}
    for path in [target_dir / "hr_diagram_params.csv"]:
        df = read_csv_maybe(path)
        if df is None or not len(df):
            continue
        row = df.iloc[0]
        for src, dst in [
            ("source_id", "gaia_source_id"),
            ("Plx", "parallax_mas"),
            ("dist_pc", "distance_pc"),
            ("Gmag", "Gmag"),
            ("BP_RP", "BP_RP"),
            ("M_G", "M_G"),
            ("wd_mass_msun", "wd_mass_msun"),
            ("wd_teff_k", "wd_teff_k"),
            ("wd_logg", "wd_logg"),
            ("wd_cooling_age_gyr", "wd_cooling_age_gyr"),
        ]:
            if src in row:
                out[dst] = row[src]
        return out
    ca_path = target_dir / "cooling_age_analysis.txt"
    if ca_path.exists():
        text = ca_path.read_text(encoding="utf-8", errors="ignore")
        patterns = {
            "parallax_mas": r"Plx\s*=\s*([0-9.+-]+)",
            "distance_pc": r"距离\s*=\s*([0-9.+-]+)\s*pc",
            "M_G": r"M_G\s*=\s*([0-9.+-]+)",
            "wd_mass_msun": r"M_WD\s*=\s*([0-9.+-]+)",
            "wd_teff_k": r"T_eff\s*=\s*([0-9.+-]+)",
            "wd_logg": r"log g\s*=\s*([0-9.+-]+)",
            "wd_cooling_age_gyr": r"t_cool\s*=\s*([0-9.+-]+)\s*Gyr",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text)
            if m:
                out[key] = float(m.group(1))
        m = re.search(r"Source ID:\s*([0-9.eE+-]+)", text)
        if m:
            out["gaia_source_id"] = m.group(1)
    return out


def research_status(row: pd.Series) -> str:
    if normalize_bool(row.get("is_DWD_catalog")):
        return "known_dwd_catalog"
    if normalize_bool(row.get("is_individually_studied")):
        return "individually_studied"
    refs = pd.to_numeric(row.get("simbad_n_refs"), errors="coerce")
    if pd.notna(refs) and refs > 0:
        return "literature_refs_no_dwd"
    return "apparently_unstudied"


def compute_priority(row: dict) -> tuple[float, str]:
    score = 0.0
    reasons = []
    period = row.get("period_day_for_gw")
    if pd.notna(period) and period > 0:
        pmin = period * 1440.0
        if pmin < 30:
            score += 4
            reasons.append("P<30min")
        elif pmin < 60:
            score += 3
            reasons.append("P<60min")
        elif pmin < 120:
            score += 1
            reasons.append("P<2hr")
    if row.get("research_status") == "apparently_unstudied":
        score += 3
        reasons.append("unstudied")
    if row.get("has_optical_spectrum"):
        score += 2
        reasons.append("has_optical_spectrum")
    if row.get("rv_valid_for_dynamics"):
        score += 1
        reasons.append("has_plausible_RV")
    if row.get("has_koa_raw"):
        score += 2
        reasons.append("KOA_LRIS_raw")
    if "IR_EXCESS" in str(row.get("sed_excess_flag", "")):
        score += 2
        reasons.append("IR_excess")
    if "UV_EXCESS" in str(row.get("sed_excess_flag", "")):
        score += 1
        reasons.append("UV_excess")
    if row.get("has_xray"):
        score += 2
        reasons.append("Xray_counterpart")
    prob = pd.to_numeric(row.get("probability"), errors="coerce")
    if pd.notna(prob) and prob > 0.9:
        score += 1
        reasons.append("high_CNN_score")
    if pd.notna(row.get("distance_pc")):
        score += 1
        reasons.append("distance_available")
    return score, ";".join(reasons)


def gw_formula_fields(period_day: float) -> dict:
    if pd.isna(period_day) or period_day <= 0:
        return {"period_min_for_gw": np.nan, "f_gw_mHz": np.nan}
    p_sec = period_day * 86400.0
    return {
        "period_min_for_gw": period_day * 1440.0,
        "f_gw_mHz": 2.0 / p_sec * 1000.0,
    }


def build_tables(args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    catalog = pd.read_csv(args.catalog)
    astro_output = Path(args.astro_output)
    ensure_dir(astro_output)
    targets = set(catalog["FirstColumn_23chars"].astype(str))

    copied = []
    copied.extend(copy_batch_spectra(Path(args.batch_spectra), astro_output, targets))
    if args.all_tools_dir:
        copied.extend(copy_selected_all_tools(Path(args.all_tools_dir), astro_output))

    koa_inv = read_koa_inventory(Path(args.koa_batch))
    koa_copied, koa_spectral_inventory = copy_koa_spectra(
        Path(args.koa_batch), astro_output, targets, koa_inv
    )
    copied.extend(koa_copied)
    if copied:
        pd.DataFrame(copied).to_csv(astro_output / "integration_file_manifest.csv", index=False)
    paper_aux = read_paper_auxiliary(args.paper_pdf)

    rows = []
    for _, cat_row in catalog.iterrows():
        target = str(cat_row["FirstColumn_23chars"])
        target_dir = ensure_dir(astro_output / target)
        summary_flags = parse_summary_flags(target_dir / "summary.txt")
        optical_sources = []
        spectrum_paths = []
        for source, fname in OPTICAL_SPECTRUM_FILES.items():
            p = target_dir / fname
            if valid_spectrum(p):
                optical_sources.append(source.upper())
                spectrum_paths.append(str(p))
        has_spherex = valid_spectrum(target_dir / "spherex_spectrum.csv")
        if has_spherex:
            spectrum_paths.append(str(target_dir / "spherex_spectrum.csv"))
        koi = koa_inv.get(target, {})
        koa_spec_row = koa_spectral_inventory[koa_spectral_inventory["target"] == target]
        has_koa_raw = bool(pd.to_numeric(koi.get("koa_downloaded_files"), errors="coerce") > 0)
        if not has_koa_raw and (target_dir / "koa_raw_files.txt").exists():
            has_koa_raw = True

        period_day, period_source = parse_period_from_files(target_dir, cat_row.get("Period"))
        row = {
            "target": target,
            "ra": cat_row.get("RA_Decimal"),
            "dec": cat_row.get("Dec_Decimal"),
            "catalog_period_day": cat_row.get("Period"),
            "period_day_for_gw": period_day,
            "period_source": period_source,
            "pred_class": cat_row.get("pred_class"),
            "probability": cat_row.get("probability"),
            "research_status": research_status(cat_row),
            "is_DWD_catalog": normalize_bool(cat_row.get("is_DWD_catalog")),
            "is_individually_studied": normalize_bool(cat_row.get("is_individually_studied")),
            "simbad_main_id": cat_row.get("simbad_main_id"),
            "simbad_n_refs": cat_row.get("simbad_n_refs"),
            "study_notes": cat_row.get("study_notes"),
            "has_astro_output_dir": target_dir.exists(),
            "has_optical_spectrum": bool(optical_sources),
            "optical_spectrum_sources": ";".join(sorted(set(optical_sources))),
            "has_spherex_spectrum": has_spherex,
            "has_koa_raw": has_koa_raw,
            "has_any_spectrum_or_raw": bool(optical_sources or has_spherex or has_koa_raw),
            "spectrum_paths": ";".join(spectrum_paths),
            "has_ztf_lc": (target_dir / "ztf_lightcurve.csv").exists() or summary_flags.get("ZTF_lightcurve", False),
            "has_wise_lc": (target_dir / "wise_lightcurve.csv").exists() or summary_flags.get("WISE_lightcurve", False),
            "has_tess_lc": (target_dir / "tess_lightcurve.csv").exists() or summary_flags.get("TESS", False),
            "has_hst_lc": (target_dir / "hst_lightcurve.csv").exists() or summary_flags.get("HST_lightcurve", False),
            "has_xray": bool(summary_flags.get("X-ray", False) or (target_dir / "xray_analysis.csv").exists()),
            "koa_selected_rows": koi.get("koa_selected_rows", np.nan),
            "koa_downloaded_files": koi.get("koa_downloaded_files", np.nan),
            "koa_lris_exposures": koi.get("koa_lris_exposures", np.nan),
            "koa_reduction_status": koi.get("koa_reduction_status", ""),
            "koa_setup_status": koi.get("koa_setup_status", ""),
            "koa_spectrum_status": koi.get("koa_spectrum_status", ""),
            "koa_spectrum_file_exists": bool(len(koa_spec_row) and koa_spec_row.iloc[0].get("koa_spectrum_file_exists", False)),
            "koa_spectrum_rows": int(koa_spec_row.iloc[0].get("koa_spectrum_rows", 0)) if len(koa_spec_row) else 0,
            "koa_spectrum_usable_1d": bool(len(koa_spec_row) and koa_spec_row.iloc[0].get("koa_spectrum_usable_1d", False)),
            "koa_reduction_message": koi.get("koa_reduction_message", ""),
        }
        row.update(gw_formula_fields(row["period_day_for_gw"]))
        row.update(sed_metrics(target_dir / "sed_photometry.csv"))
        row.update(rv_metrics(target_dir / "rv_analysis.csv"))
        row.update(hr_metrics(target_dir))
        row.update(auxiliary_metrics(target, paper_aux, row))
        score, reasons = compute_priority(row)
        row["special_priority_score"] = score
        row["special_reasons"] = reasons
        row["needs_spectroscopy"] = not row["has_optical_spectrum"]
        row["needs_rv_for_gw"] = not row["rv_valid_for_dynamics"]
        row["needs_distance_for_gw"] = pd.isna(row.get("distance_pc"))
        row["period_alias_note"] = "verify whether photometric period equals orbital period; ellipsoidal curves can appear at Porb/2"
        rows.append(row)

    integrated = pd.DataFrame(rows)
    spectroscopy = integrated[integrated["has_any_spectrum_or_raw"]].copy()
    special = integrated[
        (integrated["research_status"] == "apparently_unstudied")
        & (integrated["special_priority_score"] >= 4)
    ].sort_values(["special_priority_score", "period_min_for_gw", "probability"],
                  ascending=[False, True, False])
    gw_cols = [
        "target", "ra", "dec", "period_day_for_gw", "period_min_for_gw",
        "f_gw_mHz", "gw_p_orb_min_assumed", "gw_f_mHz_assumed",
        "pred_class", "probability",
        "distance_pc", "distance_source", "parallax_mas", "wd_mass_msun",
        "wd_teff_k", "wd_logg", "desi_teff_k", "desi_logg", "desi_sn_r",
        "has_optical_spectrum",
        "optical_spectrum_sources", "rv_valid_for_dynamics", "best_rv_kms",
        "best_rv_err_kms", "research_status", "special_priority_score",
        "gw_A_assumed_mc03", "gw_h_assumed_mc03", "gw_hc_lisa_assumed_mc03",
        "gw_snr_tianqin_assumed_mc03", "gw_snr_lisa_assumed_mc03",
        "gw_snr_taiji_assumed_mc03", "gw_snr_decigo_assumed_mc03",
        "gw_snr_net_assumed_mc03",
        "period_alias_note",
    ]
    gw_input = integrated[gw_cols].copy()
    gw_input["mass_1_msun"] = np.nan
    gw_input["mass_2_msun"] = np.nan
    gw_input["chirp_mass_msun"] = np.nan
    gw_input["inclination_deg"] = np.nan
    gw_input["epoch_t0_bjd_or_mjd"] = np.nan
    gw_input["notes_for_gw_colleague"] = (
        "Need distance and component masses/chirp mass; use period with alias caution."
    )
    return integrated, spectroscopy, special, gw_input, copied


def make_apj_plots(integrated: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(outdir)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    df = integrated.copy()
    period_min = pd.to_numeric(df["period_min_for_gw"], errors="coerce")
    prob = pd.to_numeric(df["probability"], errors="coerce")
    score = pd.to_numeric(df["special_priority_score"], errors="coerce")

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    studied = df["research_status"] != "apparently_unstudied"
    spec = df["has_optical_spectrum"].astype(bool)
    ax.scatter(period_min[studied], prob[studied], s=14, c="0.65",
               marker="o", label="literature-known", linewidths=0)
    ax.scatter(period_min[~studied], prob[~studied], s=18, c="#0072B2",
               marker="o", label="apparently unstudied", linewidths=0)
    ax.scatter(period_min[spec], prob[spec], s=36, facecolors="none",
               edgecolors="#D55E00", linewidths=0.8, label="1D spectrum")
    ax.set_xscale("log")
    ax.set_xlabel(r"$P_{\rm phot}$ or $P_{\rm orb}$ (min)")
    ax.set_ylabel("CNN probability")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.text(0.03, 0.95, f"N={len(df)}", transform=ax.transAxes, va="top")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_period_probability_spectra.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.scatter(period_min, score, c=np.where(df["has_optical_spectrum"], "#D55E00", "#009E73"),
               s=18, linewidths=0, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(r"$P_{\rm phot}$ or $P_{\rm orb}$ (min)")
    ax.set_ylabel("follow-up priority score")
    ax.grid(alpha=0.18, linewidth=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_priority_vs_period.{ext}", dpi=300)
    plt.close(fig)

    ir = pd.to_numeric(df["max_ir_excess_dex"], errors="coerce")
    uv = pd.to_numeric(df["max_uv_excess_dex"], errors="coerce")
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    mask_ir = np.isfinite(ir)
    ax.scatter(period_min[mask_ir], ir[mask_ir], s=18, c="#CC79A7",
               linewidths=0, label="IR residual")
    mask_uv = np.isfinite(uv)
    ax.scatter(period_min[mask_uv], uv[mask_uv], s=18, c="#56B4E9",
               marker="s", linewidths=0, label="UV residual")
    ax.axhline(0.30, color="0.25", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$P_{\rm phot}$ or $P_{\rm orb}$ (min)")
    ax.set_ylabel(r"SED residual (dex)")
    ax.legend(fontsize=6, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_sed_residuals.{ext}", dpi=300)
    plt.close(fig)

    fgw = pd.to_numeric(df["f_gw_mHz"], errors="coerce")
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.hist(fgw[np.isfinite(fgw)], bins=np.logspace(np.log10(0.05), np.log10(max(10, np.nanmax(fgw) * 1.1)), 28),
            color="0.35", histtype="stepfilled", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel(r"$f_{\rm GW}=2/P_{\rm orb}$ (mHz)")
    ax.set_ylabel("number")
    ax.axvspan(0.1, 100, color="#0072B2", alpha=0.08)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_gw_frequency_distribution.{ext}", dpi=300)
    plt.close(fig)


def write_orphan_output_dirs(astro_output: Path, catalog_targets: set[str]) -> pd.DataFrame:
    rows = []
    for target_dir in sorted(p for p in astro_output.iterdir() if p.is_dir() and p.name.startswith("ZTFJ")):
        if target_dir.name in catalog_targets:
            continue
        optical_sources = []
        spectrum_paths = []
        for source, fname in OPTICAL_SPECTRUM_FILES.items():
            p = target_dir / fname
            if valid_spectrum(p):
                optical_sources.append(source.upper())
                spectrum_paths.append(str(p))
        rows.append({
            "target": target_dir.name,
            "path": str(target_dir),
            "n_files": len([p for p in target_dir.iterdir() if p.is_file()]),
            "has_optical_spectrum": bool(optical_sources),
            "optical_spectrum_sources": ";".join(sorted(set(optical_sources))),
            "spectrum_paths": ";".join(spectrum_paths),
            "has_ztf_lc": (target_dir / "ztf_lightcurve.csv").exists(),
            "has_wise_lc": (target_dir / "wise_lightcurve.csv").exists(),
            "has_summary": (target_dir / "summary.txt").exists(),
            "note": "Existing astro_output directory not present in current DWD_combined_clean.csv",
        })
    orphan = pd.DataFrame(rows)
    orphan.to_csv(astro_output / "orphan_astro_output_dirs_not_in_current_catalog.csv", index=False)
    return orphan


def plot_spectrum_preview(spectrum_csv: Path, out_png: Path, title: str) -> bool:
    df = read_csv_maybe(spectrum_csv)
    if df is None or len(df) < 2:
        return False
    cols = {c.lower(): c for c in df.columns}
    wave_col = cols.get("wavelength_a") or cols.get("wavelength") or cols.get("wave")
    flux_col = cols.get("flux") or cols.get("flux_cgs") or cols.get("flam")
    if not wave_col or not flux_col:
        return False
    data = df[[wave_col, flux_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        return False
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.plot(data[wave_col], data[flux_col], lw=0.5, color="0.15")
    for line, label in [(4101.7, "Hd"), (4340.5, "Hg"), (4861.3, "Hb"), (6562.8, "Ha")]:
        ax.axvline(line, color="#D55E00", lw=0.45, alpha=0.45)
        ax.text(line, 0.96, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Flux")
    ax.tick_params(direction="in", top=True, right=True, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def write_new_source_folder(astro_output: Path, integrated: pd.DataFrame) -> pd.DataFrame:
    outdir = ensure_dir(astro_output / "new_source")
    rows = integrated[integrated["target"].isin(NEW_SOURCE_TARGETS)].copy()
    order = {target: i for i, target in enumerate(NEW_SOURCE_TARGETS)}
    rows["new_source_order"] = rows["target"].map(order)
    rows["paper_angle"] = rows["target"].map(NEW_SOURCE_NOTES)
    rows = rows.sort_values("new_source_order")

    keep_cols = [
        "target", "ra", "dec", "pred_class", "probability", "period_min_for_gw",
        "f_gw_mHz", "distance_pc", "research_status", "has_optical_spectrum",
        "optical_spectrum_sources", "koa_spectrum_usable_1d", "has_koa_raw",
        "special_priority_score", "special_reasons", "paper_angle",
    ]
    rows[keep_cols].to_csv(outdir / "selected_new_sources.csv", index=False)

    copy_names = [
        "sdss_spectrum.csv", "desi_spectrum.csv", "hst_spectrum.csv", "koa_spectrum.csv",
        "koa_spectrum.png", "koa_spectrum_report.txt", "koa_exposures.csv",
        "ztf_lightcurve.csv", "ztf_lightcurve.png", "wise_lightcurve.csv", "wise_lightcurve.png",
        "sed_photometry.csv", "sed.png", "sed_diagnostics.txt", "period_analysis.csv",
        "combined_fold.png", "ZTF_g_period.png", "ZTF_r_period.png", "hr_diagram_params.csv",
        "hr_diagram.png", "rv_analysis.csv", "rv_analysis.txt", "summary.txt",
        "simbad_references.txt", "simbad_references.csv",
    ]

    for _, row in rows.iterrows():
        target = row["target"]
        src_dir = astro_output / target
        dst_dir = ensure_dir(outdir / target)
        copied = []
        if src_dir.exists():
            for name in copy_names:
                src = src_dir / name
                if src.exists() and src.is_file():
                    dst = dst_dir / name
                    shutil.copy2(src, dst)
                    copied.append(name)
                    if name.endswith("_spectrum.csv"):
                        plot_spectrum_preview(src, dst_dir / f"{src.stem}_preview.png", f"{target} {src.stem}")
            for src in sorted(src_dir.glob("combined_fold_P*.png")):
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
                copied.append(src.name)

        source_lines = [
            f"# {target}",
            "",
            f"- RA, Dec: {row.get('ra')}, {row.get('dec')}",
            f"- Morphology / probability: {row.get('pred_class')} / {row.get('probability')}",
            f"- Period: {row.get('period_min_for_gw'):.3f} min; f_GW from photometric period: {row.get('f_gw_mHz'):.3f} mHz",
            f"- Research status: {row.get('research_status')}",
            f"- Optical spectrum: {row.get('has_optical_spectrum')} ({row.get('optical_spectrum_sources')})",
            f"- Distance: {row.get('distance_pc')}",
            f"- Priority score/reasons: {row.get('special_priority_score')} / {row.get('special_reasons')}",
            "",
            "## Why It Is Interesting",
            "",
            str(row.get("paper_angle", "")),
            "",
            "## Files Copied Here",
            "",
        ]
        if copied:
            source_lines.extend(f"- `{name}`" for name in sorted(set(copied)))
        else:
            source_lines.append("- No per-source products were available yet beyond the integrated catalog row.")
        (dst_dir / "README.md").write_text("\n".join(source_lines) + "\n", encoding="utf-8")

    readme = [
        "# New Special Sources",
        "",
        "This folder collects the eight apparently unstudied sources that are most useful for a new paper discussion.",
        "",
        "## Headline Candidates",
        "",
        "- `ZTFJ171532.47-194407.03`: 23.04 min, high probability, no SIMBAD match, no spectrum yet.",
        "- `ZTFJ184551.87-255127.65`: 23.04 min, EA-like, no SIMBAD match, no spectrum yet.",
        "- `ZTFJ235115.39+630527.72`: 43.20 min, high probability, clean no-SIMBAD follow-up target.",
        "",
        "## Spectroscopy-Ready Candidates",
        "",
        "- `ZTFJ150822.14+432245.78` and `ZTFJ150514.54+070102.09`: unstudied, high probability, SDSS spectra already available.",
        "- `ZTFJ170033.08+645154.41` and `ZTFJ132153.54+254309.27`: unstudied with SDSS spectra, lower probability; useful as secondary comparison cases.",
        "",
        "## Caution",
        "",
        "`ZTFJ075052.56-082854.12` is short-period and unstudied, but its CNN probability is low. Treat it as a candidate needing visual light-curve and spectroscopy checks before making a strong claim.",
        "",
        "## Summary Table",
        "",
        markdown_table(rows[keep_cols]),
        "",
    ]
    (outdir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    return rows


def write_markdown_reports(astro_output: Path, integrated: pd.DataFrame,
                           spectroscopy: pd.DataFrame, special: pd.DataFrame,
                           gw_input: pd.DataFrame, paper_pdf: str = "") -> None:
    def simple_markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return ""
        table = df.copy()
        for col in table.columns:
            if pd.api.types.is_float_dtype(table[col]):
                table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
            else:
                table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
            table[col] = table[col].str.replace("|", r"\|", regex=False)
        header = "| " + " | ".join(table.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(table.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.to_numpy()]
        return "\n".join([header, sep, *rows])

    def simple_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
        def latex_escape(value) -> str:
            if pd.isna(value):
                text = ""
            elif isinstance(value, float):
                text = f"{value:.4g}"
            else:
                text = str(value)
            replacements = {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
            }
            for old, new in replacements.items():
                text = text.replace(old, new)
            return text

        cols = list(df.columns)
        align = "l" * len(cols)
        lines = [
            r"\begin{table*}",
            r"\centering",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            rf"\begin{{tabular}}{{{align}}}",
            r"\hline",
            " & ".join(latex_escape(c) for c in cols) + r" \\",
            r"\hline",
        ]
        for _, row in df.iterrows():
            lines.append(" & ".join(latex_escape(row[c]) for c in cols) + r" \\")
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ])
        return "\n".join(lines)

    total = len(integrated)
    n_opt = int(integrated["has_optical_spectrum"].sum())
    n_any = int(integrated["has_any_spectrum_or_raw"].sum())
    n_unstudied = int((integrated["research_status"] == "apparently_unstudied").sum())
    n_special = len(special)
    n_distance = int(integrated["distance_pc"].notna().sum()) if "distance_pc" in integrated else 0
    n_gw_ref = int(integrated["gw_snr_lisa_assumed_mc03"].notna().sum()) if "gw_snr_lisa_assumed_mc03" in integrated else 0
    orphan_path = astro_output / "orphan_astro_output_dirs_not_in_current_catalog.csv"
    n_orphan = len(pd.read_csv(orphan_path)) if orphan_path.exists() else 0
    lines = [
        "# Integrated DWD Candidate Summary",
        "",
        f"- Catalog candidates: {total}",
        f"- Optical 1D spectra available: {n_opt}",
        f"- Any spectrum/raw spectral data available: {n_any}",
        f"- Distance/parallax estimates available: {n_distance}",
        f"- Existing paper GW reference rows merged: {n_gw_ref}",
        f"- Apparently unstudied candidates: {n_unstudied}",
        f"- Unstudied high-priority special candidates: {n_special}",
        f"- Existing astro_output directories not in current catalog: {n_orphan}",
        "",
        "## Most useful GW inputs",
        "",
        "Provide the GW colleague with `gw_calculator_input.csv`. The minimum physical inputs are sky position, orbital period, distance, and either component masses or chirp mass. The current table includes placeholders for unknown component masses and inclinations.",
        "",
        "For a circular detached DWD, useful formulae are:",
        "",
        r"- \(f_{\rm GW}=2/P_{\rm orb}\)",
        r"- \(\mathcal{M}_c=(M_1M_2)^{3/5}/(M_1+M_2)^{1/5}\)",
        r"- \(h_0=2(G\mathcal{M}_c)^{5/3}(\pi f_{\rm GW})^{2/3}/(c^4 d)\), up to inclination/polarization convention",
        r"- \(\dot f=(96/5)\pi^{8/3}(G\mathcal{M}_c/c^3)^{5/3}f^{11/3}\) for GR-driven detached evolution",
        "",
        "Important caution: for ellipsoidal variables the photometric period can be half the orbital period, while eclipsing systems usually give the orbital period directly. The `period_alias_note` column marks this.",
        "",
        "## Paper update suggestions",
        "",
        "- Add a compact table of unstudied high-priority systems, ordered by period, spectrum availability, SED excess flags, and KOA/HST/DESI follow-up readiness.",
        "- Add an explicit GW-input table for collaborators: RA, Dec, period, frequency, distance/parallax, available mass proxy, spectrum/RV status, and alias caveat.",
        "- Discuss sources with IR/UV residuals separately from ordinary single-component SED candidates; these are the natural analogs of the W1/W2/W3/W4 excess discussion.",
        "- Keep DESI/SDSS pipeline RVs as morphology or rough flags only unless the value is physically plausible and verified from WD lines.",
        "- For systems without spectra, state that Gaia CMD masses are unresolved-binary proxies and can be biased; prioritize spectroscopy or multi-band eclipse modeling.",
        "",
    ]
    if paper_pdf:
        lines.append(f"Paper inspected: `{paper_pdf}`")
    (astro_output / "integrated_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_koa_file = int(integrated.get("koa_spectrum_file_exists", pd.Series(dtype=bool)).fillna(False).sum())
    n_koa_usable = int(integrated.get("koa_spectrum_usable_1d", pd.Series(dtype=bool)).fillna(False).sum())
    readme_outputs = [
        "# astro_output File Guide",
        "",
        "This directory now contains the integrated products for the current cleaned DWD catalog.",
        "",
        "## Quick Counts",
        "",
        f"- Current catalog rows: {total}",
        f"- Usable optical 1D spectra: {n_opt}",
        f"- No usable optical 1D spectrum: {total - n_opt}",
        f"- Any spectrum/raw spectral material: {n_any}",
        f"- KOA spectrum CSV files copied: {n_koa_file}",
        f"- KOA spectrum CSV files usable as 1D spectra: {n_koa_usable}",
        f"- Apparently unstudied sources: {n_unstudied}",
        f"- Selected new-source folder: `new_source/`",
        "",
        "## Main CSV Files",
        "",
        "- `integrated_catalog.csv`: master table for all current catalog sources.",
        "- `targets_with_optical_spectra.csv`: sources with usable SDSS/DESI/HST/JWST/LAMOST/KOA 1D spectra.",
        "- `targets_without_optical_spectra.csv`: sources still needing optical spectroscopy.",
        "- `spectroscopy_inventory.csv`: all sources with usable spectra, KOA raw data, or other spectral material.",
        "- `koa_spectral_inventory.csv`: KOA/LRIS metadata, downloaded raw counts, reduction status, and copied KOA spectrum files.",
        "- `gw_calculator_input.csv`: compact table to send to GW collaborators.",
        "- `unstudied_special_candidates.csv`: apparently unstudied high-priority candidates.",
        "- `apparently_unstudied_targets.csv`: all apparently unstudied sources, regardless of priority.",
        "- `orphan_astro_output_dirs_not_in_current_catalog.csv`: old output directories not present in the current cleaned catalog.",
        "- `integration_file_manifest.csv`: files copied or refreshed by the integration script.",
        "",
        "## Human-Readable Notes",
        "",
        "- `integrated_summary.md`: short scientific summary and paper suggestions.",
        "- `gw_collaborator_readme.md`: what to give the GW colleague and what is still missing.",
        "- `paper_discussion_candidates.md`: compact Markdown table for source discussion.",
        "- `new_source/README.md`: focused notes on the eight new special sources.",
        "",
        "## Figures And Tables",
        "",
        "- `apj_figures/`: ApJ-style PDF/PNG plots.",
        "- `latex_table_spectroscopic_inventory.tex`: LaTeX spectroscopic inventory.",
        "- `latex_table_unstudied_special_candidates.tex`: LaTeX table for the new/unstudied candidates.",
        "",
        "## KOA Caveat",
        "",
        "KOA raw LRIS observations are not the same thing as a usable 1D spectrum. Use `koa_spectrum_usable_1d=True` or the `KOA` entry in `optical_spectrum_sources` for a real 1D spectrum. Rows with raw data but `koa_spectrum_status=not_found` still need extraction/reduction.",
        "",
    ]
    (astro_output / "README_OUTPUTS.md").write_text("\n".join(readme_outputs), encoding="utf-8")

    gw_readme = [
        "# GW Collaborator Packet",
        "",
        "Send `gw_calculator_input.csv` as the main machine-readable table.",
        "",
        "Recommended columns to use first:",
        "",
        "- `target`, `ra`, `dec`: source identifier and sky position.",
        "- `period_min_for_gw`: measured photometric period from the current cleaned catalog/pipeline.",
        "- `gw_p_orb_min_assumed`, `gw_f_mHz_assumed`: orbital period and GW frequency already adopted in the paper GW calculation where available.",
        "- `pred_class`: light-curve morphology; EW/ellipsoidal systems may have `P_orb = 2 P_phot`, while EA/eclipsing systems usually have `P_orb = P_phot`.",
        "- `distance_pc`, `parallax_mas`, `distance_source`: current distance information.",
        "- `mass_1_msun`, `mass_2_msun`, `chirp_mass_msun`, `inclination_deg`: intentionally blank placeholders for the collaborator or later modeling.",
        "- `has_optical_spectrum`, `optical_spectrum_sources`, `rv_valid_for_dynamics`: whether spectroscopy/RV constraints exist.",
        "- `gw_snr_lisa_assumed_mc03` and related columns: existing paper estimates using an assumed chirp mass of 0.3 Msun; use as reference only.",
        "",
        "Minimum missing physics before a robust GW claim:",
        "",
        "1. Confirm the orbital period and alias choice with radial velocities or eclipse geometry.",
        "2. Estimate component masses or chirp mass from WD atmosphere fits, RV mass functions, and/or eclipse modeling.",
        "3. Use Gaia/Bailer-Jones distance where available; avoid claiming precise strain for rows with blank `distance_pc`.",
        "4. Treat DESI RVSpecFit Teff/logg as morphology flags, not WD atmospheric parameters.",
        "",
    ]
    (astro_output / "gw_collaborator_readme.md").write_text("\n".join(gw_readme), encoding="utf-8")

    top = special.head(25).copy()
    cols = [
        "target", "period_min_for_gw", "f_gw_mHz", "probability",
        "has_optical_spectrum", "optical_spectrum_sources",
        "sed_excess_flag", "special_priority_score", "special_reasons",
    ]
    md = ["# Unstudied Special Candidates", ""]
    if len(top):
        md.append(simple_markdown_table(top[cols]))
    else:
        md.append("No candidates met the current high-priority unstudied threshold.")
    (astro_output / "paper_discussion_candidates.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    spec_cols = ["target", "optical_spectrum_sources", "has_koa_raw", "has_spherex_spectrum",
                 "best_rv_kms", "rv_valid_for_dynamics", "research_status"]
    (astro_output / "latex_table_spectroscopic_inventory.tex").write_text(
        simple_latex_table(
            spectroscopy[spec_cols],
            "Spectroscopic inventory of candidate short-period double white dwarf binaries.",
            "tab:spectroscopic_inventory",
        ),
        encoding="utf-8",
    )
    (astro_output / "latex_table_unstudied_special_candidates.tex").write_text(
        simple_latex_table(
            special[cols].head(20),
            "High-priority apparently unstudied candidates for discussion and follow-up.",
            "tab:unstudied_special",
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="/Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv")
    parser.add_argument("--astro-output", default="/Users/ljm/Desktop/csst/desi匹配/DWD_new/astro_output")
    parser.add_argument("--batch-spectra", default="/Users/ljm/Desktop/csst/desi匹配/DWD_new/batch_output/spectra")
    parser.add_argument("--koa-batch", default="/Users/ljm/Desktop/csst/desi匹配/DWD_new/koa_batch")
    parser.add_argument("--all-tools-dir", default="/Users/ljm/Desktop/csst/desi匹配/DWD_new/all_tools_ZTFJ035352.96+431525.16_20260426")
    parser.add_argument("--paper-pdf", default="/Users/ljm/Desktop/双白矮星搜寻论文/dwd_paper_v2.pdf")
    args = parser.parse_args()

    astro_output = Path(args.astro_output)
    integrated, spectroscopy, special, gw_input, copied = build_tables(args)
    integrated.to_csv(astro_output / "integrated_catalog.csv", index=False)
    spectroscopy.to_csv(astro_output / "spectroscopy_inventory.csv", index=False)
    integrated[integrated["has_optical_spectrum"]].to_csv(
        astro_output / "targets_with_optical_spectra.csv", index=False
    )
    integrated[~integrated["has_optical_spectrum"]].to_csv(
        astro_output / "targets_without_optical_spectra.csv", index=False
    )
    integrated[integrated["research_status"] == "apparently_unstudied"].to_csv(
        astro_output / "apparently_unstudied_targets.csv", index=False
    )
    special.to_csv(astro_output / "unstudied_special_candidates.csv", index=False)
    gw_input.to_csv(astro_output / "gw_calculator_input.csv", index=False)
    orphan = write_orphan_output_dirs(astro_output, set(integrated["target"].astype(str)))
    make_apj_plots(integrated, astro_output / "apj_figures")
    new_sources = write_new_source_folder(astro_output, integrated)
    write_markdown_reports(astro_output, integrated, spectroscopy, special, gw_input, args.paper_pdf)
    print(f"Integrated catalog rows: {len(integrated)}")
    print(f"Optical spectra: {int(integrated['has_optical_spectrum'].sum())}")
    print(f"Any spectra/raw: {int(integrated['has_any_spectrum_or_raw'].sum())}")
    print(f"Unstudied high-priority candidates: {len(special)}")
    print(f"Selected new-source folder rows: {len(new_sources)}")
    print(f"Existing output dirs not in current catalog: {len(orphan)}")
    print(f"Outputs written under: {astro_output}")


if __name__ == "__main__":
    main()
