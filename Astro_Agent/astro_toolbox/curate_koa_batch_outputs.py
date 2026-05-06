#!/usr/bin/env python3
"""Curate KOA batch target folders and gather spectrum review products."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TargetInfo:
    target: str
    selected_rows: int
    downloaded_files: int
    reduction_status: str = ""
    spectrum_status: str = ""
    reduction_message: str = ""
    target_dir: Path | None = None
    spec1d_files: list[Path] | None = None
    spec2d_files: list[Path] | None = None
    spectrum_png: Path | None = None
    preview_png: Path | None = None
    spectrum_csv: Path | None = None
    raw_download_fits: list[Path] | None = None
    setup_raw_fits: list[Path] | None = None
    csv_points: int = 0
    category: str = ""
    reason: str = ""


def read_summary(path: Path) -> dict[str, TargetInfo]:
    out: dict[str, TargetInfo] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target = row["target"]
            out[target] = TargetInfo(
                target=target,
                selected_rows=int(float(row.get("n_selected_rows") or 0)),
                downloaded_files=int(float(row.get("n_downloaded_files") or 0)),
                target_dir=Path(row["target_dir"]),
            )
    return out


def read_reduction(path: Path, info: dict[str, TargetInfo]) -> None:
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            target = row["target"]
            if target not in info:
                info[target] = TargetInfo(
                    target=target,
                    selected_rows=int(float(row.get("n_selected_rows") or 0)),
                    downloaded_files=int(float(row.get("n_downloaded_fits_after") or 0)),
                    target_dir=Path(row["target_dir"]),
                )
            info[target].reduction_status = row.get("status", "")
            info[target].spectrum_status = row.get("spectrum_status", "")
            info[target].reduction_message = row.get("message", "")


def count_csv_points(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        # subtract header
        return max(sum(1 for _ in handle) - 1, 0)


def first_match(paths: list[Path]) -> Path | None:
    return paths[0] if paths else None


def analyze_target(info: TargetInfo) -> None:
    target_dir = info.target_dir
    if not target_dir or not target_dir.exists():
        info.spec1d_files = []
        info.spec2d_files = []
        info.category = "missing_dir"
        info.reason = "Target directory is missing."
        return

    spectrum_dir = target_dir / "spectrum"
    info.spec1d_files = sorted(spectrum_dir.glob("spec1d*.fits")) if spectrum_dir.exists() else []
    info.spec2d_files = sorted(spectrum_dir.glob("spec2d*.fits")) if spectrum_dir.exists() else []
    info.spectrum_png = spectrum_dir / "koa_spectrum.png" if (spectrum_dir / "koa_spectrum.png").exists() else None
    info.spectrum_csv = spectrum_dir / "koa_spectrum.csv" if (spectrum_dir / "koa_spectrum.csv").exists() else None
    lev0_pngs = sorted((target_dir / "png" / "lev0").glob("*.png"))
    info.raw_download_fits = sorted(target_dir.glob("download/*/lev0/*.fits"))
    info.setup_raw_fits = sorted((target_dir / "pypeit_setup" / "pypeit_raw" / "lev0").glob("*.fits"))
    info.preview_png = first_match(lev0_pngs)
    info.csv_points = count_csv_points(info.spectrum_csv)

    if info.spec1d_files and info.csv_points > 10:
        info.category = "usable_1d"
        info.reason = "Has extracted spec1d FITS and a non-trivial KOA spectrum CSV."
        return
    if info.spec1d_files and info.csv_points <= 10:
        info.category = "tiny_1d_not_usable"
        info.reason = (
            f"spec1d FITS exists, but koa_spectrum.csv has only {info.csv_points} point(s); "
            "likely an incomplete extraction."
        )
        return
    if info.spec2d_files and info.csv_points > 10:
        info.category = "fallback_from_2d"
        info.reason = (
            "No spec1d FITS, but koa_spectrum.csv was generated from a spec2d fallback and looks non-trivial."
        )
        return
    if info.spec2d_files:
        info.category = "only_2d_no_1d"
        info.reason = info.reduction_message or "Only spec2d FITS found; no readable extracted 1D spectrum."
        return
    if info.raw_download_fits or info.setup_raw_fits or info.preview_png:
        info.category = "raw_download_no_extraction"
        info.reason = (
            info.reduction_message
            or "Raw KOA lev0 spectroscopic frames exist, but no extracted spec1d/spec2d products were found yet."
        )
        return
    if info.downloaded_files > 0:
        info.category = "downloaded_but_no_spectrum_products"
        info.reason = info.reduction_message or "Downloaded files are recorded, but no spectrum products are present."
        return
    if info.selected_rows > 0:
        info.category = "metadata_only"
        info.reason = "KOA metadata matched this target, but no downloaded spectrum products are present."
        return

    info.category = "no_koa_data"
    info.reason = "No KOA-selected rows and no downloaded files."


def should_keep(info: TargetInfo) -> bool:
    return info.category in {
        "usable_1d",
        "tiny_1d_not_usable",
        "fallback_from_2d",
        "only_2d_no_1d",
        "raw_download_no_extraction",
        "downloaded_but_no_spectrum_products",
    }


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_review_products(info: TargetInfo, review_dir: Path) -> list[str]:
    copied: list[str] = []

    def _copy_all(paths: list[Path], suffix: str) -> None:
        for idx, src in enumerate(paths, start=1):
            label = suffix if len(paths) == 1 else f"{suffix}_{idx}"
            dst = review_dir / f"{info.target}_{label}{src.suffix}"
            safe_copy(src, dst)
            copied.append(dst.name)

    _copy_all(info.spec1d_files or [], "spec1d")
    _copy_all(info.spec2d_files or [], "spec2d")
    if not copied:
        preview_stem = info.preview_png.stem if info.preview_png else None
        raw_candidates = (info.raw_download_fits or []) + (info.setup_raw_fits or [])
        raw_match = None
        if preview_stem:
            for candidate in raw_candidates:
                if candidate.stem == preview_stem:
                    raw_match = candidate
                    break
        if not raw_match and raw_candidates:
            raw_match = raw_candidates[0]
        if raw_match:
            dst = review_dir / f"{info.target}_lev0.fits"
            safe_copy(raw_match, dst)
            copied.append(dst.name)
    if info.spectrum_png:
        dst = review_dir / f"{info.target}_koa_spectrum.png"
        safe_copy(info.spectrum_png, dst)
        copied.append(dst.name)
    if info.preview_png:
        dst = review_dir / f"{info.target}_lev0_preview.png"
        safe_copy(info.preview_png, dst)
        copied.append(dst.name)
    return copied


def write_report(review_dir: Path, kept: list[TargetInfo], deleted: list[str]) -> None:
    report_csv = review_dir / "koa_review_summary.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target", "selected_rows", "downloaded_files", "reduction_status",
                "spectrum_status", "csv_points", "category", "reason"
            ],
        )
        writer.writeheader()
        for info in kept:
            writer.writerow({
                "target": info.target,
                "selected_rows": info.selected_rows,
                "downloaded_files": info.downloaded_files,
                "reduction_status": info.reduction_status,
                "spectrum_status": info.spectrum_status,
                "csv_points": info.csv_points,
                "category": info.category,
                "reason": info.reason,
            })

    readme = review_dir / "README.txt"
    lines = [
        "KOA batch curation summary",
        "",
        f"Review folder: {review_dir}",
        f"Targets kept (with KOA rows/files): {len(kept)}",
        f"Target directories deleted (no KOA rows/files): {len(deleted)}",
        "",
        "Category guide:",
        "  usable_1d                 spec1d FITS exists and KOA CSV looks usable.",
        "  tiny_1d_not_usable        spec1d exists but the extracted CSV is basically empty.",
        "  fallback_from_2d          no spec1d, but a non-trivial fallback spectrum was made from spec2d.",
        "  only_2d_no_1d             only 2D products exist; no readable 1D extraction.",
        "  raw_download_no_extraction  raw lev0/downloaded spectroscopic frames exist, but no extracted products yet.",
        "  downloaded_but_no_spectrum_products  downloads recorded but no spectrum products found.",
        "  metadata_only             KOA matched metadata only; no downloads present.",
        "",
        "Deleted target directories:",
    ]
    lines.extend(f"  {name}" for name in deleted)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def delete_target_dirs(root: Path, deleted: list[str]) -> None:
    for name in deleted:
        target_dir = root / name
        if target_dir.exists():
            shutil.rmtree(target_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--koa-root", required=True)
    parser.add_argument("--review-dir", required=True)
    parser.add_argument("--delete-empty", action="store_true")
    args = parser.parse_args()

    koa_root = Path(args.koa_root)
    review_dir = Path(args.review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)

    summary_path = koa_root / "koa_batch_summary.csv"
    reduction_path = koa_root / "koa_reduction_summary.csv"

    info = read_summary(summary_path)
    read_reduction(reduction_path, info)

    for row in info.values():
        analyze_target(row)

    kept = sorted((row for row in info.values() if should_keep(row)), key=lambda x: x.target)
    deleted = sorted(row.target for row in info.values() if not should_keep(row))

    for row in kept:
        copy_review_products(row, review_dir)

    write_report(review_dir, kept, deleted)

    if args.delete_empty:
        delete_target_dirs(koa_root, deleted)

    print(f"keep_targets={len(kept)}")
    print(f"delete_targets={len(deleted)}")
    print(f"review_dir={review_dir}")


if __name__ == "__main__":
    main()
