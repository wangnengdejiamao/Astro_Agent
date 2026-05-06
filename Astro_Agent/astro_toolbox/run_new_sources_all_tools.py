#!/usr/bin/env python3
"""Run the full single-target toolbox workflow for selected new sources."""

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


DEFAULT_ASTRO_OUTPUT = Path("/Users/ljm/Desktop/csst/desi匹配/DWD_new/astro_output")
DEFAULT_SELECTED = DEFAULT_ASTRO_OUTPUT / "new_source" / "selected_new_sources.csv"
SINGLE_RUNNER = Path(__file__).with_name("run_single_target_all_tools.py")


KEY_PRODUCTS = [
    ("sdss/sdss_spectrum.csv", "toolbox_sdss_spectrum.csv"),
    ("sdss/sdss_spectrum.png", "toolbox_sdss_spectrum.png"),
    ("desi/desi_spectrum.csv", "toolbox_desi_spectrum.csv"),
    ("desi/desi_spectrum.png", "toolbox_desi_spectrum.png"),
    ("hst/hst_spectrum.csv", "toolbox_hst_spectrum.csv"),
    ("hst/hst_spectrum.png", "toolbox_hst_spectrum.png"),
    ("koa/koa_spectrum.csv", "toolbox_koa_spectrum.csv"),
    ("koa/koa_spectrum.png", "toolbox_koa_spectrum.png"),
    ("ztf/ztf_lightcurve.csv", "toolbox_ztf_lightcurve.csv"),
    ("ztf/ztf_lightcurve.png", "toolbox_ztf_lightcurve.png"),
    ("wise/wise_lightcurve.csv", "toolbox_wise_lightcurve.csv"),
    ("wise/wise_lightcurve.png", "toolbox_wise_lightcurve.png"),
    ("sed/sed_photometry.csv", "toolbox_sed_photometry.csv"),
    ("sed/sed.png", "toolbox_sed.png"),
    ("hr_diagram/hr_diagram_params.csv", "toolbox_hr_diagram_params.csv"),
    ("hr_diagram/hr_diagram.png", "toolbox_hr_diagram.png"),
    ("period_analysis/period_analysis.csv", "toolbox_period_analysis.csv"),
    ("period_analysis/ZTF_g_period.png", "toolbox_ZTF_g_period.png"),
    ("period_analysis/ZTF_r_period.png", "toolbox_ZTF_r_period.png"),
    ("combined_plots/combined_fold.png", "toolbox_combined_fold.png"),
    ("combined_plots/combined_spectra.png", "toolbox_combined_spectra.png"),
    ("combined_plots/spectra_with_photometry.png", "toolbox_spectra_with_photometry.png"),
    ("rv/rv_analysis.csv", "toolbox_rv_analysis.csv"),
    ("rv/rv_analysis.txt", "toolbox_rv_analysis.txt"),
    ("rv/rv_ccf_sdss.png", "toolbox_rv_ccf_sdss.png"),
    ("rv_correction/rv_correction_sdss.csv", "toolbox_rv_correction_sdss.csv"),
    ("rv_correction/rv_true_sdss.txt", "toolbox_rv_true_sdss.txt"),
    ("rv_correction/rv_line_fit_sdss.png", "toolbox_rv_line_fit_sdss.png"),
    ("wd_fitting/wd_fitting_results.csv", "toolbox_wd_fitting_results.csv"),
    ("wd_fitting/wd_fitting_report.txt", "toolbox_wd_fitting_report.txt"),
    ("wd_fitting/wd_spectral_fit.png", "toolbox_wd_spectral_fit.png"),
    ("wd_fitting/wd_chi2_map.png", "toolbox_wd_chi2_map.png"),
    ("cooling_age/cooling_age_analysis.txt", "toolbox_cooling_age_analysis.txt"),
    ("orbit_traceback/orbit_traceback_candidates.csv", "toolbox_orbit_traceback_candidates.csv"),
    ("orbit_traceback/orbit_traceback.txt", "toolbox_orbit_traceback.txt"),
    ("orbit_traceback/orbit_traceback.png", "toolbox_orbit_traceback.png"),
    ("six_dim/sixdim_rv_info.png", "toolbox_sixdim_rv_info.png"),
    ("six_dim/sixdim_sed.png", "toolbox_sixdim_sed.png"),
    ("six_dim/sixdim_ztf.png", "toolbox_sixdim_ztf.png"),
    ("xray/xray_analysis.csv", "toolbox_xray_analysis.csv"),
    ("xray/xray_analysis.txt", "toolbox_xray_analysis.txt"),
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_status_counts(status_path: Path) -> dict[str, int]:
    if not status_path.exists():
        return {}
    try:
        df = pd.read_csv(status_path)
    except Exception:
        return {}
    if "status" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["status"].value_counts().items()}


def copy_key_products(run_dir: Path, target_dir: Path) -> list[str]:
    copied = []
    for rel_src, name in KEY_PRODUCTS:
        src = run_dir / rel_src
        if not src.exists() or not src.is_file():
            continue
        dst = target_dir / name
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


def write_target_readme(target_dir: Path, target: str, row: pd.Series,
                        run_dir: Path, copied: list[str],
                        status_counts: dict[str, int]) -> None:
    lines = [
        f"# {target} Toolbox Rerun",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- RA, Dec: {row.get('ra')}, {row.get('dec')}",
        f"- Class/probability: {row.get('pred_class')} / {row.get('probability')}",
        f"- Period: {row.get('period_min_for_gw')} min",
        f"- Existing optical spectrum before rerun: {row.get('has_optical_spectrum')} ({row.get('optical_spectrum_sources')})",
        "",
        "## Module Status Counts",
        "",
    ]
    if status_counts:
        lines.extend(f"- {status}: {count}" for status, count in sorted(status_counts.items()))
    else:
        lines.append("- No module_status.csv was produced.")
    lines.extend(["", "## Key Products Copied To This Folder", ""])
    if copied:
        lines.extend(f"- `{name}`" for name in sorted(copied))
    else:
        lines.append("- No key products matched the copy list yet.")
    lines.extend([
        "",
        "Full raw module outputs are under `toolbox_run/`.",
        "",
    ])
    (target_dir / "TOOLBOX_RERUN_README.md").write_text("\n".join(lines), encoding="utf-8")


def yes_no(path: Path) -> str:
    return "yes" if path.exists() else "no"


def write_batch_readme(new_source_root: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Toolbox Rerun Summary",
        "",
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Each target has a `toolbox_run/` directory with full raw module outputs. "
        "Frequently used products are also copied to the target folder with a `toolbox_` prefix.",
        "",
        "## Quick Table",
        "",
        "| target | status | copied | SDSS spectrum | WD fit | RV correction | period analysis |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        target = str(row["target"])
        target_dir = new_source_root / target
        lines.append(
            "| {target} | {status} | {copied} | {sdss} | {wd} | {rv_corr} | {period} |".format(
                target=target,
                status=row["status"],
                copied=row["n_key_products_copied"],
                sdss=yes_no(target_dir / "toolbox_sdss_spectrum.csv"),
                wd=yes_no(target_dir / "toolbox_wd_fitting_results.csv"),
                rv_corr=yes_no(target_dir / "toolbox_rv_correction_sdss.csv"),
                period=yes_no(target_dir / "toolbox_period_analysis.csv"),
            )
        )

    lines.extend([
        "",
        "## Common Files",
        "",
        "- `toolbox_batch_summary.csv`: batch-level status and module status counts.",
        "- `<target>/TOOLBOX_RERUN_README.md`: per-target copied products and module status counts.",
        "- `<target>/toolbox_run/module_status.csv`: detailed module-by-module result.",
        "- `<target>/toolbox_sdss_spectrum.csv/png`: SDSS optical spectrum when available.",
        "- `<target>/toolbox_wd_fitting_results.csv`: preliminary WD spectral-fit parameters when fitting succeeded.",
        "- `<target>/toolbox_rv_correction_sdss.csv`: SDSS radial velocity corrected for gravitational redshift when fitting succeeded.",
        "- `<target>/toolbox_period_analysis.csv`: period-analysis output when the module finished before timeout.",
        "",
        "KOA single-target queries returned metadata-table errors in this rerun, so KOA results should be taken from the separate KOA batch outputs if needed.",
        "",
    ])
    (new_source_root / "TOOLBOX_RERUN_README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", default=str(DEFAULT_SELECTED))
    parser.add_argument("--new-source-root", default=str(DEFAULT_ASTRO_OUTPUT / "new_source"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=1800,
        help="Maximum wall time per target before keeping partial outputs and moving on.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected_csv = Path(args.selected_csv)
    new_source_root = ensure_dir(Path(args.new_source_root))
    df = pd.read_csv(selected_csv)
    if args.limit:
        df = df.head(args.limit)

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp")
    pykoa = "/tmp/pykoa_deps"
    env["PYTHONPATH"] = pykoa + os.pathsep + env.get("PYTHONPATH", "")

    summary_rows = []
    for _, row in df.iterrows():
        target = str(row["target"])
        target_dir = ensure_dir(new_source_root / target)
        run_dir = target_dir / "toolbox_run"
        done_marker = run_dir / "run_summary.json"
        log_path = target_dir / "toolbox_run.log"

        if done_marker.exists() and not args.force:
            status = "existing_ok"
            return_code = 0
        else:
            ensure_dir(run_dir)
            started = time.monotonic()
            cmd = [
                sys.executable,
                str(SINGLE_RUNNER),
                "--target", target,
                "--ra", str(row["ra"]),
                "--dec", str(row["dec"]),
                "--output-root", str(run_dir),
            ]
            with log_path.open("w", encoding="utf-8") as log:
                log.write(" ".join(cmd) + "\n\n")
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
                    log.write(
                        f"\nTIMEOUT after {elapsed:.1f} seconds. "
                        "Partial outputs were kept and the batch continued.\n"
                    )
                    if exc.stdout:
                        log.write(str(exc.stdout))
                    if exc.stderr:
                        log.write(str(exc.stderr))
                    return_code = 124
                    status = "timeout"

        status_counts = load_status_counts(run_dir / "module_status.csv")
        copied = copy_key_products(run_dir, target_dir)
        write_target_readme(target_dir, target, row, run_dir, copied, status_counts)
        summary_rows.append({
            "target": target,
            "status": status,
            "return_code": return_code,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "n_key_products_copied": len(copied),
            "status_counts_json": json.dumps(status_counts, ensure_ascii=False, sort_keys=True),
            "updated": datetime.now().isoformat(timespec="seconds"),
        })
        pd.DataFrame(summary_rows).to_csv(new_source_root / "toolbox_batch_summary.csv", index=False)
        write_batch_readme(new_source_root, summary_rows)

    pd.DataFrame(summary_rows).to_csv(new_source_root / "toolbox_batch_summary.csv", index=False)
    write_batch_readme(new_source_root, summary_rows)
    print(f"Processed {len(summary_rows)} targets")
    print(f"Summary: {new_source_root / 'toolbox_batch_summary.csv'}")


if __name__ == "__main__":
    main()
