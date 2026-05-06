#!/usr/bin/env python3
"""
Update the DWD candidate catalogue from visually screened DWD_NEW PNGs and
organize spectra / paper plotting products.

This script is intentionally conservative:
  * the original catalogue is backed up before it is overwritten;
  * rows from the Ren et al. catalogue are always retained;
  * raw products are copied into organized review folders, not deleted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


DEFAULT_DWD_NEW = Path("/Users/ljm/Desktop/DWD/DWD_NEW")
DEFAULT_CATALOG = Path(
    "/Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv"
)
DEFAULT_DWD_ROOT = Path("/Users/ljm/Desktop/csst/desi匹配/DWD_new")
DEFAULT_PAPER_DIR = Path("/Users/ljm/Desktop/双白矮星搜寻论文")


TARGET_RE = re.compile(r"ZTF\s*J(\d{6}\.\d{2}[+-]\d{6}\.\d{2})")


def norm_target(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    text = text.replace("ZTF J", "ZTFJ").replace(" ", "")
    m = re.search(r"ZTFJ\d{6}\.\d{2}[+-]\d{6}\.\d{2}", text)
    return m.group(0) if m else text


def target_to_batch_dir(target: str) -> str:
    return target.replace("+", "p").replace("-", "m")


def parse_png_targets(png_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(png_dir.glob("*.png")):
        m = TARGET_RE.search(path.name)
        if not m:
            rows.append(
                {
                    "target": "",
                    "dwd_new_png_name": path.name,
                    "dwd_new_png_path": str(path),
                    "parse_ok": False,
                }
            )
            continue
        rows.append(
            {
                "target": f"ZTFJ{m.group(1)}",
                "dwd_new_png_name": path.name,
                "dwd_new_png_path": str(path),
                "parse_ok": True,
            }
        )
    parsed = pd.DataFrame(rows)
    if parsed.empty:
        return pd.DataFrame(
            columns=["target", "dwd_new_png_name", "dwd_new_png_path", "parse_ok"]
        )
    return parsed.drop_duplicates("target", keep="first").reset_index(drop=True)


def truthy(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y", "t"]) | text.str.startswith("true")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dest_dir: Path, prefix: str = "") -> Path | None:
    if not src.exists() or not src.is_file():
        return None
    ensure_dir(dest_dir)
    name = f"{prefix}{src.name}" if prefix else src.name
    dest = dest_dir / name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        i = 2
        while dest.exists():
            dest = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
    shutil.copy2(src, dest)
    return dest


def split_paths(value: object) -> list[Path]:
    if pd.isna(value):
        return []
    parts = []
    for chunk in str(value).split(";"):
        chunk = chunk.strip()
        if chunk:
            parts.append(Path(chunk))
    return parts


def is_spectrum_file(path: Path) -> bool:
    name = path.name.lower()
    if name in {"combined_spectra.png", "spectra_with_photometry.png"}:
        return True
    if name.endswith(("_spec1d.fits", "_spec2d.fits", "_lev0.fits")):
        return True
    if name.endswith("_lev0_preview.png"):
        return True
    if name in {"koa_exposures.csv", "exposures.csv", "koa_spectrum_report.txt"}:
        return True
    if "spectrum" in name or "spectra" in name:
        return path.suffix.lower() in {
            ".csv",
            ".png",
            ".pdf",
            ".fits",
            ".fit",
            ".gz",
            ".txt",
        }
    return False


def candidate_spectrum_files(
    target: str,
    integrated_row: pd.Series | None,
    astro_output: Path,
    batch_spectra: Path,
    koa_review: Path,
    koa_batch: Path,
) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []

    def add_many(paths: Iterable[Path], label: str) -> None:
        for path in paths:
            if path.exists() and path.is_file() and is_spectrum_file(path):
                files.append((path, label))

    if integrated_row is not None:
        for path in split_paths(integrated_row.get("spectrum_paths", np.nan)):
            add_many([path], "catalog")

    for subdir, label in [
        (astro_output / target, "astro"),
        (astro_output / "new_source" / target, "newsource"),
        (batch_spectra / target_to_batch_dir(target), "batch"),
        (koa_batch / target / "spectrum", "koa_batch"),
    ]:
        if subdir.exists():
            add_many(subdir.glob("*"), label)

    if koa_review.exists():
        add_many(koa_review.glob(f"{target}_*"), "koa_review")

    seen = set()
    unique: list[tuple[Path, str]] = []
    for path, label in files:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append((path, label))
    return unique


def preferred_spectrum_image(target: str, astro_output: Path) -> Path | None:
    candidates = [
        astro_output / target / "spectra_with_photometry.png",
        astro_output / target / "combined_spectra.png",
        astro_output / target / "sdss_spectrum.png",
        astro_output / target / "desi_spectrum.png",
        astro_output / target / "koa_spectrum.png",
        astro_output / target / "hst_spectrum.png",
        astro_output / "new_source" / target / "toolbox_spectra_with_photometry.png",
        astro_output / "new_source" / target / "toolbox_combined_spectra.png",
        astro_output / "new_source" / target / "toolbox_sdss_spectrum.png",
        astro_output / "new_source" / target / "sdss_spectrum_preview.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def save_source_panel(
    row: pd.Series,
    png_path: Path | None,
    spectrum_path: Path | None,
    out_base: Path,
) -> None:
    target = str(row.get("target", ""))
    p_min = row.get("period_min_for_gw", np.nan)
    if pd.isna(p_min):
        p_min = row.get("Period_minutes", np.nan)
    probability = row.get("probability", np.nan)
    spec_sources = row.get("optical_spectrum_sources", "")
    reasons = row.get("special_reasons", "")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), dpi=160)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

    if png_path and png_path.exists():
        try:
            axes[0].imshow(mpimg.imread(png_path))
            axes[0].set_title("Screened ZTF light curve", fontsize=8)
        except Exception as exc:
            axes[0].text(0.5, 0.5, f"Could not load LC PNG\n{exc}", ha="center")
    else:
        axes[0].text(0.5, 0.5, "No DWD_NEW PNG", ha="center", va="center")

    if spectrum_path and spectrum_path.exists():
        try:
            axes[1].imshow(mpimg.imread(spectrum_path))
            axes[1].set_title("Available spectrum product", fontsize=8)
        except Exception as exc:
            axes[1].text(
                0.5, 0.5, f"Could not load spectrum PNG\n{exc}", ha="center"
            )
    else:
        axes[1].text(
            0.5,
            0.5,
            "No extracted optical spectrum\nin current products",
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
        )

    detail = []
    if pd.notna(p_min):
        detail.append(f"P={float(p_min):.2f} min")
    if pd.notna(probability):
        detail.append(f"CNN={float(probability):.3f}")
    if isinstance(spec_sources, str) and spec_sources.strip():
        detail.append(f"spec: {spec_sources}")
    title = target
    if detail:
        title += "   " + "   ".join(detail)
    fig.suptitle(title, fontsize=9)
    if isinstance(reasons, str) and reasons.strip():
        fig.text(0.02, 0.01, reasons[:180], fontsize=7, color="0.25")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    ensure_dir(out_base.parent)
    png_path = Path(f"{out_base}.png")
    pdf_path = Path(f"{out_base}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_wrapper_scripts(paper_dir: Path, figure_output_dir: Path) -> list[Path]:
    scripts_dir = ensure_dir(paper_dir / "figure_scripts")
    common = f"""#!/usr/bin/env python3
import argparse
import os
import sys

PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PAPER_DIR)

import generate_all_figures as gf


def set_output_dir(out_dir):
    gf.FIGURES = out_dir
    os.makedirs(gf.FIGURES, exist_ok=True)
    os.makedirs(os.path.join(gf.FIGURES, "source_panels"), exist_ok=True)
    os.makedirs(os.path.join(gf.FIGURES, "appendix_panels"), exist_ok=True)
"""
    wrappers = {
        "generate_sed_figures.py": common
        + f"""
parser = argparse.ArgumentParser(description="Generate SED + spectrum figures only.")
parser.add_argument("--output-dir", default={str(figure_output_dir / "sed")!r})
args = parser.parse_args()
set_output_dir(args.output_dir)
gf.fig_spec_sed_combined()
print("Saved SED figures to:", gf.FIGURES)
""",
        "generate_spectrum_lightcurve_figures.py": common
        + f"""
parser = argparse.ArgumentParser(description="Generate spectrum and light-curve summary figures.")
parser.add_argument("--output-dir", default={str(figure_output_dir / "spectra_lightcurves")!r})
parser.add_argument("--skip-ztf", action="store_true", help="Use placeholders instead of downloading ZTF data.")
args = parser.parse_args()
set_output_dir(args.output_dir)
gf.fig_spectra_panel()
gf.fig_lightcurves_panel(skip_ztf=args.skip_ztf)
print("Saved spectrum/light-curve figures to:", gf.FIGURES)
""",
        "generate_source_panels.py": common
        + f"""
parser = argparse.ArgumentParser(description="Generate per-source panels from generate_all_figures.py.")
parser.add_argument("--output-dir", default={str(figure_output_dir / "source_panels")!r})
parser.add_argument("--skip-ztf", action="store_true", help="Use placeholders instead of downloading ZTF data.")
args = parser.parse_args()
set_output_dir(args.output_dir)
gf.fig_all_source_panels(skip_ztf=args.skip_ztf)
print("Saved source panels to:", gf.FIGURES)
""",
        "generate_publication_figures.py": common
        + f"""
parser = argparse.ArgumentParser(description="Generate the main publication figures.")
parser.add_argument("--output-dir", default={str(figure_output_dir / "publication")!r})
parser.add_argument("--skip-ztf", action="store_true", help="Use placeholders instead of downloading ZTF data.")
args = parser.parse_args()
set_output_dir(args.output_dir)
for key in ["hr", "spectra", "period_teff", "sky", "cmd", "pipeline", "retention", "rpm", "gw", "cnn", "quality", "sed_spec"]:
    func, kwargs = gf.ALL_FIGS[key]
    func(**kwargs)
if not args.skip_ztf:
    gf.fig_lightcurves_panel(skip_ztf=False)
print("Saved publication figures to:", gf.FIGURES)
""",
    }

    written = []
    for name, text in wrappers.items():
        path = scripts_dir / name
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
        written.append(path)
    return written


def organize_paper_folder(paper_dir: Path, products_dir: Path) -> dict[str, object]:
    paper_build = ensure_dir(paper_dir / "paper_build")
    figure_outputs = ensure_dir(paper_dir / "figure_outputs")
    source_panels_unstudied = ensure_dir(figure_outputs / "source_panels_unstudied")

    paper_patterns = [
        "dwd_paper.tex",
        "dwd_paper.bib",
        "aastex7.cls",
        "dwd_paper.pdf",
        "dwd_paper_v2.pdf",
        "dwd_paper.bbl",
        "dwd_paper.log",
        "dwd_paper.aux",
        "dwd_paper.out",
    ]
    copied_paper = []
    for name in paper_patterns:
        src = paper_dir / name
        if src.exists() and src.is_file():
            dest = paper_build / src.name
            shutil.copy2(src, dest)
            copied_paper.append(dest)

    figures_dir = paper_dir / "figures"
    copied_figures = []
    if figures_dir.exists():
        for src in figures_dir.rglob("*"):
            if src.is_file() and src.suffix.lower() in {
                ".png",
                ".pdf",
                ".md",
                ".txt",
                ".tex",
            }:
                rel = src.relative_to(figures_dir)
                dest = figure_outputs / "existing_figures" / rel
                ensure_dir(dest.parent)
                shutil.copy2(src, dest)
                copied_figures.append(dest)

    panel_src = products_dir / "source_panels_unstudied"
    copied_panels = []
    if panel_src.exists():
        for src in panel_src.glob("*"):
            if src.is_file() and src.suffix.lower() in {".png", ".pdf"}:
                dest = source_panels_unstudied / src.name
                shutil.copy2(src, dest)
                copied_panels.append(dest)

    wrappers = write_wrapper_scripts(paper_dir, figure_outputs)

    readme = paper_dir / "README_PAPER_ORGANIZATION.md"
    readme.write_text(
        "\n".join(
            [
                "# Paper folder organization",
                "",
                f"Generated on {dt.datetime.now().isoformat(timespec='seconds')}.",
                "",
                "## Folders",
                f"- `paper_build/`: copied TeX, bibliography, class file, and latest PDFs/logs ({len(copied_paper)} files).",
                f"- `figure_outputs/existing_figures/`: copied existing publication figures ({len(copied_figures)} files).",
                f"- `figure_outputs/source_panels_unstudied/`: generated PNG/PDF panels for apparently unstudied sources ({len(copied_panels)} files).",
                "- `figure_scripts/`: split entry points for SED, spectra/light curves, source panels, and publication figures.",
                "",
                "## Split plotting commands",
                "- `python figure_scripts/generate_sed_figures.py`",
                "- `python figure_scripts/generate_spectrum_lightcurve_figures.py --skip-ztf`",
                "- `python figure_scripts/generate_source_panels.py --skip-ztf`",
                "- `python figure_scripts/generate_publication_figures.py --skip-ztf`",
                "",
                "Use commands without `--skip-ztf` only when network downloads from IRSA are desired.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "paper_build": str(paper_build),
        "figure_outputs": str(figure_outputs),
        "wrappers": [str(p) for p in wrappers],
        "copied_paper_files": len(copied_paper),
        "copied_figure_files": len(copied_figures),
        "copied_unstudied_panel_files": len(copied_panels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dwd-new-dir", type=Path, default=DEFAULT_DWD_NEW)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--source-catalog",
        type=Path,
        default=None,
        help="Read candidates from this catalogue but write/update --catalog.",
    )
    parser.add_argument("--dwd-root", type=Path, default=DEFAULT_DWD_ROOT)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=DEFAULT_DWD_ROOT / "DWD_NEW_selected_products",
    )
    parser.add_argument(
        "--no-overwrite-catalog",
        action="store_true",
        help="Write the filtered catalogue but do not replace the input catalogue.",
    )
    args = parser.parse_args()

    now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    astro_output = args.dwd_root / "astro_output"
    batch_spectra = args.dwd_root / "batch_output" / "spectra"
    koa_review = args.dwd_root / "koa_spectrum_review"
    koa_batch = args.dwd_root / "koa_batch"

    products = ensure_dir(args.products_dir)
    catalogs_dir = ensure_dir(products / "catalogs")
    spectra_all_dir = ensure_dir(products / "spectra_all")
    spectra_unstudied_dir = ensure_dir(products / "spectra_unstudied")
    lightcurves_dir = ensure_dir(products / "lightcurves")
    panels_unstudied_dir = ensure_dir(products / "source_panels_unstudied")

    pngs = parse_png_targets(args.dwd_new_dir)
    if pngs.empty:
        raise RuntimeError(f"No parseable PNGs found in {args.dwd_new_dir}")
    png_target_set = set(pngs.loc[pngs["parse_ok"], "target"])

    source_catalog = args.source_catalog or args.catalog
    catalog = pd.read_csv(source_catalog)
    catalog["target"] = catalog["FirstColumn_23chars"].map(norm_target)
    ren_mask = truthy(catalog["from_ren2023"]) if "from_ren2023" in catalog else False
    keep_mask = catalog["target"].isin(png_target_set) | ren_mask
    updated = catalog.loc[keep_mask].copy()
    removed = catalog.loc[~keep_mask].copy()

    updated = updated.merge(
        pngs[["target", "dwd_new_png_name", "dwd_new_png_path", "parse_ok"]],
        on="target",
        how="left",
    )
    updated["in_dwd_new_png"] = updated["target"].isin(png_target_set)
    if "Period" in updated.columns:
        updated["Period_minutes"] = pd.to_numeric(updated["Period"], errors="coerce") * 1440.0

    backup_path = args.catalog.with_name(
        f"{args.catalog.stem}_before_DWD_NEW_filter_{now}{args.catalog.suffix}"
    )
    shutil.copy2(args.catalog if args.catalog.exists() else source_catalog, backup_path)

    filtered_path = catalogs_dir / "DWD_combined_clean_updated_from_DWD_NEW.csv"
    removed_path = catalogs_dir / "DWD_combined_clean_removed_not_in_DWD_NEW.csv"
    png_list_path = catalogs_dir / "DWD_NEW_png_targets.csv"
    updated.to_csv(filtered_path, index=False)
    removed.to_csv(removed_path, index=False)
    pngs.to_csv(png_list_path, index=False)

    if not args.no_overwrite_catalog:
        updated.to_csv(args.catalog, index=False)

    integrated_path = astro_output / "integrated_catalog.csv"
    if integrated_path.exists():
        integrated = pd.read_csv(integrated_path)
        integrated["target"] = integrated["target"].map(norm_target)
    else:
        integrated = pd.DataFrame()

    if not integrated.empty:
        filtered_integrated = integrated[integrated["target"].isin(updated["target"])].copy()
        updated_targets = set(updated["target"])
        missing_integrated = updated.loc[
            ~updated["target"].isin(set(integrated["target"])),
            ["target", "RA_Decimal", "Dec_Decimal", "Period", "probability"],
        ].copy()
    else:
        filtered_integrated = pd.DataFrame()
        missing_integrated = updated[["target"]].copy()

    filtered_integrated_path = catalogs_dir / "integrated_catalog_DWD_NEW_filtered.csv"
    missing_integrated_path = catalogs_dir / "targets_missing_from_integrated_catalog.csv"
    filtered_integrated.to_csv(filtered_integrated_path, index=False)
    missing_integrated.to_csv(missing_integrated_path, index=False)

    bool_false = ["false", "0", "no", "nan", ""]
    if not filtered_integrated.empty:
        studied = filtered_integrated["is_individually_studied"].astype(str).str.lower()
        unstudied_mask = studied.isin(bool_false)
        has_optical = filtered_integrated["has_optical_spectrum"].fillna(False).astype(bool)
        has_any = filtered_integrated["has_any_spectrum_or_raw"].fillna(False).astype(bool)

        spectra_inventory = filtered_integrated[has_optical].copy()
        spectra_or_raw_inventory = filtered_integrated[has_any | has_optical].copy()
        unstudied = filtered_integrated[unstudied_mask].copy()
        unstudied_spectra = filtered_integrated[unstudied_mask & has_optical].copy()
        priority = unstudied.sort_values(
            ["special_priority_score", "period_min_for_gw", "probability"],
            ascending=[False, True, False],
        ).copy()
    else:
        spectra_inventory = pd.DataFrame()
        spectra_or_raw_inventory = pd.DataFrame()
        unstudied = pd.DataFrame()
        unstudied_spectra = pd.DataFrame()
        priority = pd.DataFrame()

    spectra_inventory.to_csv(catalogs_dir / "DWD_NEW_spectroscopy_inventory.csv", index=False)
    spectra_or_raw_inventory.to_csv(
        catalogs_dir / "DWD_NEW_spectroscopy_or_raw_inventory.csv", index=False
    )
    unstudied.to_csv(catalogs_dir / "DWD_NEW_apparently_unstudied.csv", index=False)
    unstudied_spectra.to_csv(
        catalogs_dir / "DWD_NEW_unstudied_spectroscopy_inventory.csv", index=False
    )
    priority.to_csv(catalogs_dir / "DWD_NEW_unstudied_priority.csv", index=False)

    integrated_lookup = {
        row["target"]: row for _, row in filtered_integrated.iterrows()
    } if not filtered_integrated.empty else {}
    png_lookup = {
        row["target"]: Path(row["dwd_new_png_path"])
        for _, row in pngs.iterrows()
        if row.get("parse_ok")
    }

    copied_rows = []
    targets_to_collect = list(spectra_or_raw_inventory["target"]) if not spectra_or_raw_inventory.empty else []
    for target in targets_to_collect:
        row = integrated_lookup.get(target)
        files = candidate_spectrum_files(
            target, row, astro_output, batch_spectra, koa_review, koa_batch
        )
        dest_target = ensure_dir(spectra_all_dir / target)
        for src, label in files:
            dest = copy_file(src, dest_target, prefix=f"{label}__")
            if dest:
                copied_rows.append(
                    {
                        "target": target,
                        "group": "all",
                        "source_label": label,
                        "source_path": str(src),
                        "copied_path": str(dest),
                    }
                )
                if row is not None and not bool(row.get("is_individually_studied", True)):
                    dest_u = copy_file(src, ensure_dir(spectra_unstudied_dir / target), prefix=f"{label}__")
                    if dest_u:
                        copied_rows.append(
                            {
                                "target": target,
                                "group": "unstudied",
                                "source_label": label,
                                "source_path": str(src),
                                "copied_path": str(dest_u),
                            }
                        )

    copied_manifest = pd.DataFrame(copied_rows)
    copied_manifest.to_csv(catalogs_dir / "copied_spectrum_files_manifest.csv", index=False)

    # Copy all screened PNG light curves for the updated target list.
    copied_lc = []
    for target in updated["target"]:
        src = png_lookup.get(target)
        if src and src.exists():
            dest = copy_file(src, lightcurves_dir, prefix=f"{target}__")
            if dest:
                copied_lc.append({"target": target, "source_path": str(src), "copied_path": str(dest)})
    pd.DataFrame(copied_lc).to_csv(catalogs_dir / "copied_lightcurve_png_manifest.csv", index=False)

    # Make quick-look panels for all apparently unstudied sources.
    panel_rows = []
    if not unstudied.empty:
        for _, row in unstudied.iterrows():
            target = row["target"]
            png_path = png_lookup.get(target)
            spec_img = preferred_spectrum_image(target, astro_output)
            out_base = panels_unstudied_dir / f"{target}_lc_spectrum_panel"
            save_source_panel(row, png_path, spec_img, out_base)
            panel_png = Path(f"{out_base}.png")
            panel_pdf = Path(f"{out_base}.pdf")
            panel_rows.append(
                {
                    "target": target,
                    "panel_png": str(panel_png),
                    "panel_pdf": str(panel_pdf),
                    "dwd_new_png": str(png_path) if png_path else "",
                    "spectrum_png": str(spec_img) if spec_img else "",
                }
            )
    pd.DataFrame(panel_rows).to_csv(catalogs_dir / "unstudied_source_panels_manifest.csv", index=False)

    paper_info = organize_paper_folder(args.paper_dir, products)

    special_cols = [
        "target",
        "period_min_for_gw",
        "probability",
        "research_status",
        "is_individually_studied",
        "has_optical_spectrum",
        "optical_spectrum_sources",
        "has_koa_raw",
        "special_priority_score",
        "special_reasons",
        "needs_spectroscopy",
    ]
    special_table = priority[[c for c in special_cols if c in priority.columns]].head(30)
    special_table.to_csv(catalogs_dir / "DWD_NEW_top30_unstudied_for_paper.csv", index=False)

    readme_lines = [
        "# DWD_NEW catalogue update",
        "",
        f"Generated on {dt.datetime.now().isoformat(timespec='seconds')}.",
        "",
        "## Filtering rule",
        f"- Parsed `{len(pngs)}` PNG files from `{args.dwd_new_dir}`.",
        "- Kept rows whose `FirstColumn_23chars` appears in DWD_NEW PNG names.",
        "- Also kept every row with `from_ren2023 == True`, even if absent from DWD_NEW.",
        f"- Original catalogue rows: `{len(catalog)}`.",
        f"- Updated catalogue rows: `{len(updated)}`.",
        f"- Removed non-Ren rows absent from DWD_NEW: `{len(removed)}`.",
        "",
        "## Main files",
        f"- Backup of original catalogue: `{backup_path}`.",
        f"- Updated catalogue copy: `{filtered_path}`.",
        f"- Current catalogue overwritten: `{args.catalog}`." if not args.no_overwrite_catalog else "- Current catalogue was not overwritten.",
        f"- Removed-row list: `{removed_path}`.",
        f"- Filtered integrated catalogue: `{filtered_integrated_path}`.",
        "",
        "## Spectroscopy",
        f"- Sources with extracted optical spectra in filtered catalogue: `{len(spectra_inventory)}`.",
        f"- Sources with spectra or KOA raw products: `{len(spectra_or_raw_inventory)}`.",
        f"- Apparently unstudied sources with extracted spectra: `{len(unstudied_spectra)}`.",
        f"- Copied spectrum files: `{len(copied_manifest)}`.",
        f"- All copied spectra folder: `{spectra_all_dir}`.",
        f"- Unstudied copied spectra folder: `{spectra_unstudied_dir}`.",
        "",
        "## Apparently unstudied sources",
        f"- Apparently unstudied sources in filtered catalogue: `{len(unstudied)}`.",
        f"- Quick-look LC+spectrum panels generated: `{len(panel_rows)}` sources.",
        f"- Panel folder: `{panels_unstudied_dir}`.",
        "- Priority tables:",
        "  - `DWD_NEW_unstudied_priority.csv`",
        "  - `DWD_NEW_top30_unstudied_for_paper.csv`",
        "",
        "## Paper folder",
        f"- Paper build folder: `{paper_info['paper_build']}`.",
        f"- Figure output folder: `{paper_info['figure_outputs']}`.",
        "- Split plotting scripts are in `figure_scripts/` under the paper folder.",
        "",
        "## Notes for paper discussion",
        "- The most important unstudied systems are short-period objects, especially P < 30 min, and unstudied systems that already have SDSS/DESI/KOA spectra.",
        "- Period aliases should be checked before gravitational-wave calculations: ellipsoidal modulation can appear at Porb/2.",
        "- For GW follow-up, provide P_orb, sky position, distance/parallax, masses or mass assumptions, RV constraints if available, and whether the listed period is photometric or orbital.",
    ]
    (products / "README_DWD_NEW_UPDATE.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )

    print("DWD_NEW update complete")
    print(f"  PNG targets parsed: {len(pngs)}")
    print(f"  Original catalogue rows: {len(catalog)}")
    print(f"  Updated catalogue rows: {len(updated)}")
    print(f"  Removed rows: {len(removed)}")
    print(f"  Spectra sources: {len(spectra_inventory)}")
    print(f"  Spectra/raw sources: {len(spectra_or_raw_inventory)}")
    print(f"  Unstudied sources: {len(unstudied)}")
    print(f"  Unstudied spectra sources: {len(unstudied_spectra)}")
    print(f"  Products: {products}")
    print(f"  Paper figures: {paper_info['figure_outputs']}")


if __name__ == "__main__":
    main()
