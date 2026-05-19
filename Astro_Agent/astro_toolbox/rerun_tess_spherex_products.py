#!/usr/bin/env python3
"""Rerun lightweight TESS products and one SPHEREx check sample.

This script intentionally calls the existing toolbox modules instead of
duplicating science logic.  TESS is run target-by-target in subprocesses so one
slow MAST download cannot stall the whole batch.  SPHEREx is limited to one
sample by default, matching the usual "run one and inspect" workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
PARENT_DIR = ROOT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from astro_toolbox import spherex, tess  # noqa: E402
from astro_toolbox.run_existing_astro_output_analysis import (  # noqa: E402
    _parse_target_coord,
    iter_target_dirs,
)


DEFAULT_ASTRO_OUTPUT = Path("/Users/ljm/Desktop/csst/desi匹配/DWD_new/astro_output")
TESS_AUTHORS = ("SPOC", "TESS-SPOC", "QLP")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _target_rows(root: Path, recursive: bool = False) -> list[Path]:
    return list(iter_target_dirs(root, recursive=recursive, include_archival=False))


def _has_tess_outputs(target_dir: Path) -> bool:
    return (
        (target_dir / "tess_lightcurve.csv").exists()
        and (target_dir / "tess_lightcurve.png").exists()
    )


def _run_tess_worker(target_dir: Path) -> dict[str, object]:
    target_dir = target_dir.resolve()
    ra, dec = _parse_target_coord(target_dir.name)
    if not np.isfinite(ra + dec):
        return {
            "target": target_dir.name,
            "status": "skipped",
            "reason": "cannot_parse_coordinates",
        }

    last_error = ""
    for author in TESS_AUTHORS:
        try:
            result = tess.query_lightcurve(ra, dec, author=author)
        except Exception as exc:
            last_error = f"{author}: {type(exc).__name__}: {exc}"
            continue
        if result is None or int(result.get("n_points", 0) or 0) <= 0:
            continue

        result["author"] = author
        csv_path = tess.save_csv(result, str(target_dir))
        png_path = str(target_dir / "tess_lightcurve.png")
        tess.plot_lightcurve(result, png_path)

        period_result = None
        try:
            period_result = tess.analyze_period_lightkurve(result, str(target_dir))
            if period_result is not None:
                result["lightkurve_period_analysis"] = period_result
        except Exception as exc:
            last_error = f"period_analysis: {type(exc).__name__}: {exc}"

        meta = {
            "target": target_dir.name,
            "ra": ra,
            "dec": dec,
            "status": "available",
            "author": author,
            "n_points": int(result.get("n_points", 0) or 0),
            "sectors": result.get("sectors", []),
            "obs_time_min": result.get("obs_time_min"),
            "obs_time_max": result.get("obs_time_max"),
            "csv_path": csv_path,
            "png_path": png_path,
            "period_csv": str(target_dir / "tess_lightkurve_period.csv")
            if (target_dir / "tess_lightkurve_period.csv").exists()
            else "",
            "period_days": (period_result or {}).get("best_period_day")
            if isinstance(period_result, dict)
            else np.nan,
            "note": last_error,
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        (target_dir / "tess_rerun_metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        return meta

    return {
        "target": target_dir.name,
        "ra": ra,
        "dec": dec,
        "status": "no_data",
        "reason": last_error,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }


def _worker_tess_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args(argv)
    row = _run_tess_worker(args.target_dir)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(row, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)
    return 0 if row.get("status") in {"available", "no_data", "skipped"} else 1


def run_tess_batch(args: argparse.Namespace) -> Path:
    root = args.astro_output.expanduser().resolve()
    targets = _target_rows(root, recursive=args.recursive)
    if args.target:
        wanted = set(args.target)
        targets = [p for p in targets if p.name in wanted]
    if args.resume_summary and args.resume_summary.exists():
        try:
            done = pd.read_csv(args.resume_summary)
            if "target" in done.columns and "status" in done.columns:
                finished = set(
                    done.loc[
                        done["status"].isin(["available", "no_data", "skipped", "timeout"]),
                        "target",
                    ].astype(str)
                )
                targets = [p for p in targets if p.name not in finished]
        except Exception as exc:
            print(f"Could not read resume summary {args.resume_summary}: {exc}", flush=True)
    if not args.force_existing:
        targets = [p for p in targets if not _has_tess_outputs(p)]
    if args.limit:
        targets = targets[: args.limit]

    log_dir = root / "tess_rerun_logs"
    result_dir = log_dir / "worker_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = root / args.tess_summary_name

    rows: list[dict[str, object]] = []
    print(
        f"TESS rerun: {len(targets)} targets, force_existing={args.force_existing}, "
        f"timeout={args.tess_timeout_sec}s",
        flush=True,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(PARENT_DIR), env.get("PYTHONPATH", "")]
    )
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp")
    env["XDG_CACHE_HOME"] = env.get("XDG_CACHE_HOME", "/tmp")

    for idx, target_dir in enumerate(targets, 1):
        t0 = time.monotonic()
        result_json = result_dir / f"{target_dir.name}.json"
        log_path = log_dir / f"{target_dir.name}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-tess",
            "--target-dir",
            str(target_dir),
            "--result-json",
            str(result_json),
        ]
        print(f"[{idx}/{len(targets)}] {target_dir.name}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n")
            log.write(f"started={datetime.now().isoformat(timespec='seconds')}\n\n")
            log.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT_DIR),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.tess_timeout_sec,
                )
                return_code = int(proc.returncode)
                if result_json.exists():
                    row = json.loads(result_json.read_text(encoding="utf-8"))
                else:
                    row = {
                        "target": target_dir.name,
                        "status": "error",
                        "reason": "worker_result_missing",
                    }
            except subprocess.TimeoutExpired as exc:
                log.write(f"\nTIMEOUT after {args.tess_timeout_sec} seconds\n")
                if exc.stdout:
                    log.write(str(exc.stdout))
                if exc.stderr:
                    log.write(str(exc.stderr))
                return_code = 124
                row = {
                    "target": target_dir.name,
                    "status": "timeout",
                    "reason": f"timeout after {args.tess_timeout_sec}s",
                }

        row.update(
            {
                "return_code": return_code,
                "elapsed_sec": round(time.monotonic() - t0, 2),
                "target_dir": str(target_dir),
                "log_path": str(log_path),
            }
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    counts = (
        pd.DataFrame(rows)
        .groupby(["status"], dropna=False)
        .size()
        .reset_index(name="n_targets")
    )
    counts.to_csv(root / args.tess_counts_name, index=False)
    print(f"TESS summary: {summary_path}", flush=True)
    return summary_path


def run_spherex_sample(args: argparse.Namespace) -> dict[str, object]:
    root = args.astro_output.expanduser().resolve()
    targets = _target_rows(root, recursive=args.recursive)
    if args.spherex_target:
        targets = [p for p in targets if p.name == args.spherex_target]
    elif args.prefer_missing_spherex:
        missing = [p for p in targets if not (p / "spherex_spectrum.csv").exists()]
        targets = missing or targets
    else:
        with_existing = [p for p in targets if (p / "spherex_spectrum.csv").exists()]
        targets = with_existing or targets
    if not targets:
        raise RuntimeError("No target selected for SPHEREx sample")

    check_root = root / args.spherex_output_name
    check_root.mkdir(parents=True, exist_ok=True)
    attempted: list[dict[str, object]] = []

    for target_dir in targets[: max(1, args.spherex_attempts)]:
        ra, dec = _parse_target_coord(target_dir.name)
        if not np.isfinite(ra + dec):
            attempted.append({"target": target_dir.name, "status": "skipped"})
            continue
        sample_dir = check_root / target_dir.name
        sample_dir.mkdir(parents=True, exist_ok=True)
        log_path = sample_dir / "spherex_check.log"
        t0 = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(
                f"target={target_dir.name}\nra={ra}\ndec={dec}\n"
                f"allow_cutout_fallback={args.allow_spherex_cutout_fallback}\n\n"
            )
            try:
                spec = spherex.query_spectrum(
                    ra,
                    dec,
                    timeout=(args.spherex_connect_timeout_sec, args.spherex_read_timeout_sec),
                    allow_cutout_fallback=args.allow_spherex_cutout_fallback,
                )
            except Exception as exc:
                spec = None
                log.write(f"ERROR {type(exc).__name__}: {exc}\n")

        if spec is None:
            row = {
                "target": target_dir.name,
                "ra": ra,
                "dec": dec,
                "status": "no_data",
                "elapsed_sec": round(time.monotonic() - t0, 2),
                "log_path": str(log_path),
            }
            attempted.append(row)
            continue

        spherex.plot_spectrum(spec, str(sample_dir / "spherex_spectrum.png"))
        saved = spherex.save_spectrum_csv(spec, str(sample_dir))
        row = {
            "target": target_dir.name,
            "ra": ra,
            "dec": dec,
            "status": "available",
            "method": spec.get("method", ""),
            "n_channels": int(spec.get("n_channels", 0) or 0),
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "sample_dir": str(sample_dir),
            "spectrum_csv": str(sample_dir / "spherex_spectrum.csv"),
            "spectrum_png": str(sample_dir / "spherex_spectrum.png"),
            "full_table_csv": (saved or {}).get("full_table_csv")
            if isinstance(saved, dict)
            else "",
            "log_path": str(log_path),
        }
        attempted.append(row)
        break

    summary_path = check_root / "spherex_check_summary.csv"
    pd.DataFrame(attempted).to_csv(summary_path, index=False)
    print(f"SPHEREx check summary: {summary_path}", flush=True)
    print(pd.DataFrame(attempted).tail(1).to_string(index=False), flush=True)
    return attempted[-1] if attempted else {}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--worker-tess":
        return _worker_tess_main(argv[1:])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astro-output", type=Path, default=DEFAULT_ASTRO_OUTPUT)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--target", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tess", action="store_true")
    parser.add_argument("--force-existing", action="store_true")
    parser.add_argument("--resume-summary", type=Path)
    parser.add_argument("--tess-timeout-sec", type=int, default=600)
    parser.add_argument("--tess-summary-name", default="tess_rerun_summary.csv")
    parser.add_argument("--tess-counts-name", default="tess_rerun_counts.csv")
    parser.add_argument("--spherex-sample", action="store_true")
    parser.add_argument("--spherex-target")
    parser.add_argument("--prefer-missing-spherex", action="store_true")
    parser.add_argument("--spherex-attempts", type=int, default=1)
    parser.add_argument("--spherex-output-name", default="spherex_check_sample")
    parser.add_argument("--spherex-connect-timeout-sec", type=int, default=10)
    parser.add_argument("--spherex-read-timeout-sec", type=int, default=120)
    parser.add_argument("--allow-spherex-cutout-fallback", action="store_true")
    args = parser.parse_args(argv)

    if not args.tess and not args.spherex_sample:
        parser.error("select at least one of --tess or --spherex-sample")

    if args.spherex_sample:
        run_spherex_sample(args)
    if args.tess:
        run_tess_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
