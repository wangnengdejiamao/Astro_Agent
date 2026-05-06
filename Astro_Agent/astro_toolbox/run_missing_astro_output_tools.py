#!/usr/bin/env python3
"""Run the full toolbox workflow for astro_output targets missing key products."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_CATALOG = Path("/Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv")
DEFAULT_ASTRO_OUTPUT = Path("/Users/ljm/Desktop/csst/desi匹配/DWD_new/astro_output")
SINGLE_RUNNER = Path(__file__).with_name("run_single_target_all_tools.py")


KEY_PRODUCTS = [
    ("sdss/sdss_spectrum.csv", "sdss_spectrum.csv"),
    ("sdss/sdss_spectrum.png", "sdss_spectrum.png"),
    ("desi/desi_spectrum.csv", "desi_spectrum.csv"),
    ("desi/desi_spectrum.png", "desi_spectrum.png"),
    ("hst/hst_spectrum.csv", "hst_spectrum.csv"),
    ("hst/hst_spectrum.png", "hst_spectrum.png"),
    ("koa/koa_spectrum.csv", "koa_spectrum.csv"),
    ("koa/koa_spectrum.png", "koa_spectrum.png"),
    ("ztf/ztf_lightcurve.csv", "ztf_lightcurve.csv"),
    ("ztf/ztf_lightcurve.png", "ztf_lightcurve.png"),
    ("wise/wise_lightcurve.csv", "wise_lightcurve.csv"),
    ("wise/wise_lightcurve.png", "wise_lightcurve.png"),
    ("sed/sed_photometry.csv", "sed_photometry.csv"),
    ("sed/sed.png", "sed.png"),
    ("sed/sed_diagnostics.txt", "sed_diagnostics.txt"),
    ("hr_diagram/hr_diagram_params.csv", "hr_diagram_params.csv"),
    ("hr_diagram/hr_diagram_analysis.txt", "hr_diagram_analysis.txt"),
    ("hr_diagram/hr_diagram.png", "hr_diagram.png"),
    ("period_analysis/period_analysis.csv", "period_analysis.csv"),
    ("period_analysis/ZTF_g_period.png", "ZTF_g_period.png"),
    ("period_analysis/ZTF_r_period.png", "ZTF_r_period.png"),
    ("period_analysis/ZTF_i_period.png", "ZTF_i_period.png"),
    ("combined_plots/combined_fold.png", "combined_fold.png"),
    ("combined_plots/combined_spectra.png", "combined_spectra.png"),
    ("combined_plots/spectra_with_photometry.png", "spectra_with_photometry.png"),
    ("rv/rv_analysis.csv", "rv_analysis.csv"),
    ("rv/rv_analysis.txt", "rv_analysis.txt"),
    ("rv/rv_ccf_sdss.png", "rv_ccf_sdss.png"),
    ("rv/rv_ccf_desi.png", "rv_ccf_desi.png"),
    ("rv_correction/rv_correction_sdss.csv", "rv_correction_sdss.csv"),
    ("rv_correction/rv_true_sdss.txt", "rv_true_sdss.txt"),
    ("rv_correction/rv_line_fit_sdss.png", "rv_line_fit_sdss.png"),
    ("wd_fitting/wd_fitting_results.csv", "wd_fitting_results.csv"),
    ("wd_fitting/wd_fitting_report.txt", "wd_fitting_report.txt"),
    ("wd_fitting/wd_spectral_fit.png", "wd_spectral_fit.png"),
    ("wd_fitting/wd_chi2_map.png", "wd_chi2_map.png"),
    ("cooling_age/cooling_age_analysis.txt", "cooling_age_analysis.txt"),
    ("orbit_traceback/orbit_traceback_candidates.csv", "orbit_traceback_candidates.csv"),
    ("orbit_traceback/orbit_traceback.txt", "orbit_traceback.txt"),
    ("orbit_traceback/orbit_traceback.png", "orbit_traceback.png"),
    ("six_dim/sixdim_rv_info.png", "sixdim_rv_info.png"),
    ("six_dim/sixdim_sed.png", "sixdim_sed.png"),
    ("six_dim/sixdim_ztf.png", "sixdim_ztf.png"),
    ("xray/xray_analysis.csv", "xray_analysis.csv"),
    ("xray/xray_analysis.txt", "xray_analysis.txt"),
    ("run_summary.json", "all_tools_run_summary.json"),
    ("module_status.csv", "toolbox_module_status.csv"),
]


CORE_PRODUCTS = [
    "ztf_lightcurve.csv",
    "sed_photometry.csv",
    "sed.png",
    "hr_diagram.png",
    "combined_spectra.png",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def product_status(target_dir: Path) -> dict[str, object]:
    row = {
        "n_files": len([p for p in target_dir.glob("*") if p.is_file()]) if target_dir.exists() else 0,
    }
    for name in CORE_PRODUCTS + ["combined_fold.png", "summary.txt", "period_analysis.csv"]:
        row[name] = (target_dir / name).exists()
    row["missing_core"] = not all(bool(row[name]) for name in CORE_PRODUCTS)
    row["missing_period_fold"] = not (target_dir / "combined_fold.png").exists()
    return row


def build_missing_queue(catalog: Path, astro_output: Path, include_fold_only: bool) -> pd.DataFrame:
    cat = pd.read_csv(catalog)
    rows = []
    for _, source in cat.iterrows():
        target = str(source.get("target") or source.get("FirstColumn_23chars"))
        target_dir = astro_output / target
        status = product_status(target_dir)
        should_run = bool(status["missing_core"]) or (include_fold_only and bool(status["missing_period_fold"]))
        if not should_run:
            continue
        period = source.get("Period_minutes")
        if pd.isna(period):
            period = pd.to_numeric(source.get("Period"), errors="coerce") * 1440.0
        rows.append({
            "target": target,
            "ra": source.get("RA_Decimal"),
            "dec": source.get("Dec_Decimal"),
            "period_min": period,
            "probability": source.get("probability"),
            "pred_class": source.get("pred_class"),
            "from_ren2023": source.get("from_ren2023"),
            "is_individually_studied": source.get("is_individually_studied"),
            **status,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["period_min", "probability"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(
        ["missing_core", "period_min", "probability"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return df


def copy_key_products(run_dir: Path, target_dir: Path, overwrite: bool = False) -> list[dict[str, str]]:
    copied = []
    for rel_src, rel_dst in KEY_PRODUCTS:
        src = run_dir / rel_src
        if not src.exists() or not src.is_file():
            continue
        dst = target_dir / rel_dst
        action = "exists"
        if overwrite or not dst.exists():
            shutil.copy2(src, dst)
            action = "copied"
        copied.append({"source": str(src), "dest": str(dst), "action": action})
    return copied


def read_status_counts(run_dir: Path) -> dict[str, int]:
    status_path = run_dir / "module_status.csv"
    if not status_path.exists():
        return {}
    try:
        status = pd.read_csv(status_path)
    except Exception:
        return {}
    if "status" not in status.columns:
        return {}
    return {str(k): int(v) for k, v in status["status"].value_counts().items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--astro-output", type=Path, default=DEFAULT_ASTRO_OUTPUT)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite-products", action="store_true")
    parser.add_argument("--include-fold-only", action="store_true")
    args = parser.parse_args()

    astro_output = ensure_dir(args.astro_output)
    queue = build_missing_queue(args.catalog, astro_output, args.include_fold_only)
    if args.target:
        wanted = set(args.target)
        queue = queue[queue["target"].isin(wanted)].reset_index(drop=True)
    if args.limit:
        queue = queue.head(args.limit).copy()

    queue_path = astro_output / "toolbox_missing_targets_queue.csv"
    queue.to_csv(queue_path, index=False)
    if queue.empty:
        print("No missing targets selected.")
        print(f"Queue: {queue_path}")
        return

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp")
    env["XDG_CACHE_HOME"] = env.get("XDG_CACHE_HOME", "/tmp")
    pykoa = "/tmp/pykoa_deps"
    parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in [parent, pykoa, env.get("PYTHONPATH", "")] if p]
    )

    summary_path = astro_output / "toolbox_missing_run_summary.csv"
    copy_manifest_path = astro_output / "toolbox_missing_copy_manifest.csv"
    summary_rows = []
    copy_rows = []

    for _, row in queue.iterrows():
        target = str(row["target"])
        target_dir = ensure_dir(astro_output / target)
        run_dir = ensure_dir(target_dir / "toolbox_run")
        done = run_dir / "run_summary.json"
        log_path = target_dir / "toolbox_run.log"

        if done.exists() and not args.force:
            status = "existing_ok"
            return_code = 0
        else:
            cmd = [
                sys.executable,
                str(SINGLE_RUNNER),
                "--target", target,
                "--ra", str(row["ra"]),
                "--dec", str(row["dec"]),
                "--output-root", str(run_dir),
            ]
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                log.write(" ".join(cmd) + "\n\n")
                log.write(f"started={datetime.now().isoformat(timespec='seconds')}\n")
                log.write(f"timeout_sec={args.timeout_sec}\n\n")
                log.flush()
                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(Path(__file__).parent),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=args.timeout_sec,
                    )
                    return_code = int(proc.returncode)
                    status = "ok" if return_code == 0 else "error"
                except subprocess.TimeoutExpired as exc:
                    elapsed = time.monotonic() - started
                    log.write(f"\nTIMEOUT after {elapsed:.1f} seconds; partial outputs kept.\n")
                    if exc.stdout:
                        log.write(str(exc.stdout))
                    if exc.stderr:
                        log.write(str(exc.stderr))
                    return_code = 124
                    status = "timeout"

        copied = copy_key_products(run_dir, target_dir, overwrite=args.overwrite_products)
        for item in copied:
            copy_rows.append({"target": target, **item})

        status_counts = read_status_counts(run_dir)
        after = product_status(target_dir)
        summary_rows.append({
            "target": target,
            "status": status,
            "return_code": return_code,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "n_key_products_copied_or_existing": len(copied),
            "status_counts_json": json.dumps(status_counts, ensure_ascii=False, sort_keys=True),
            "missing_core_after": after["missing_core"],
            "missing_period_fold_after": after["missing_period_fold"],
            "updated": datetime.now().isoformat(timespec="seconds"),
        })
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.DataFrame(copy_rows).to_csv(copy_manifest_path, index=False)

    readme = astro_output / "README_TOOLBOX_MISSING_RUN.md"
    lines = [
        "# Missing astro_output toolbox run",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Queue file: `{queue_path}`",
        f"- Summary file: `{summary_path}`",
        f"- Copy manifest: `{copy_manifest_path}`",
        f"- Targets selected this run: {len(queue)}",
        "",
        "Each target's raw module outputs are under `<target>/toolbox_run/`; selected products are copied back to `<target>/` using the standard astro_output filenames.",
    ]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Processed {len(summary_rows)} targets")
    print(f"Queue: {queue_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
