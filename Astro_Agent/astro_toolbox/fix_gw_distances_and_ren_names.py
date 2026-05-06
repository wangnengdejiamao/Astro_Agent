#!/usr/bin/env python3
"""Fix GW input distances and normalize short Ren target names.

The script updates machine-readable tables in DWD_new/astro_output while keeping
timestamped backups.  It converts short Ren names such as ``ZTFJ0112+5827`` to
the same coordinate-based style used by the rest of the catalogue, then fills
missing Gaia parallax distances for the GW calculator table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


TOOLBOX = Path(__file__).resolve().parent
PARENT = TOOLBOX.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from astro_toolbox.hr_diagram import (  # noqa: E402
    HRDiagram,
    classify_hr_position,
    save_analysis_report,
    save_csv,
)


DEFAULT_ASTRO_OUTPUT = Path("/Users/ljm/Desktop/csst/desi匹配/DWD_new/astro_output")
DEFAULT_CATALOG = Path("/Users/ljm/Desktop/csst/desi匹配/DWD_new/DWD_combined_clean.csv")
FULL_ZTF_RE = re.compile(r"^ZTFJ\d{6}\.\d{2}[+-]\d{6}\.\d{2}$")


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_file(path: Path, stamp: str) -> Path | None:
    if not path.exists() or not path.is_file():
        return None
    backup = path.with_name(f"{path.stem}.before_gw_fix_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def as_float(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def clean_cell(value: object) -> object:
    try:
        if np.ma.is_masked(value):
            return np.nan
    except Exception:
        pass
    return value


def full_target_from_radec(ra: float, dec: float) -> str:
    coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
    ra_s = coord.ra.to_string(unit=u.hourangle, sep="", precision=2, pad=True)
    dec_s = coord.dec.to_string(unit=u.deg, sep="", precision=2, alwayssign=True, pad=True)
    return f"ZTFJ{ra_s}{dec_s}"


def canonical_target(target: object, ra: object, dec: object) -> str:
    text = str(target).strip().replace("ZTF J", "ZTFJ").replace(" ", "")
    if FULL_ZTF_RE.match(text):
        return text
    ra_f = as_float(ra)
    dec_f = as_float(dec)
    if np.isfinite(ra_f) and np.isfinite(dec_f):
        return full_target_from_radec(ra_f, dec_f)
    return text


def merge_dirs(src: Path, dst: Path) -> str:
    if not src.exists() or src == dst:
        return "missing_or_same"
    if not dst.exists():
        src.rename(dst)
        return "renamed"
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
    try:
        src.rmdir()
    except OSError:
        pass
    return "merged"


def read_local_gaia(target_dir: Path) -> dict[str, object] | None:
    candidates = [
        target_dir / "hr_diagram_params.csv",
        target_dir / "toolbox_run" / "hr_diagram" / "hr_diagram_params.csv",
        target_dir / "hr_diagram" / "hr_diagram_params.csv",
    ]
    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        plx = as_float(row.get("Plx", row.get("parallax_mas", row.get("parallax"))))
        dist = as_float(row.get("dist_pc", row.get("distance_pc")))
        if not np.isfinite(dist) and np.isfinite(plx) and plx > 0:
            dist = 1000.0 / plx
        if np.isfinite(dist) or np.isfinite(plx):
            return {
                "source": f"local:{path.name}",
                "gaia_source_id": str(row.get("source_id", row.get("Source", ""))).strip(),
                "parallax_mas": plx,
                "distance_pc": dist,
                "Gmag": as_float(row.get("Gmag")),
                "BP_RP": as_float(row.get("BP_RP")),
                "M_G": as_float(row.get("M_G")),
                "raw": row,
            }
    return None


def gaia_query(hr: HRDiagram, ra: float, dec: float) -> dict[str, object] | None:
    try:
        from astroquery.vizier import Vizier

        Vizier.ROW_LIMIT = 20
        center = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        columns = [
            "Source", "RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag",
            "Plx", "e_Plx", "RUWE", "Teff", "logg",
        ]
        candidates = []
        for radius_arcsec in (2.0, 5.0, 10.0):
            tables = Vizier(columns=columns).query_region(
                center,
                radius=radius_arcsec * u.arcsec,
                catalog="I/355/gaiadr3",
            )
            if not tables:
                continue
            table = tables[0]
            for row in table:
                plx = as_float(clean_cell(row["Plx"]))
                if not np.isfinite(plx) or plx <= 0:
                    continue
                row_coord = SkyCoord(
                    as_float(clean_cell(row["RA_ICRS"])) * u.deg,
                    as_float(clean_cell(row["DE_ICRS"])) * u.deg,
                )
                sep = row_coord.separation(center).arcsec
                candidates.append((sep, row))
            if candidates:
                break
        if candidates:
            sep, row = sorted(candidates, key=lambda item: item[0])[0]
            plx = as_float(clean_cell(row["Plx"]))
            gmag = as_float(clean_cell(row["Gmag"]))
            bpmag = as_float(clean_cell(row["BPmag"]))
            rpmag = as_float(clean_cell(row["RPmag"]))
            bp_rp = bpmag - rpmag if np.isfinite(bpmag) and np.isfinite(rpmag) else np.nan
            dist = 1000.0 / plx
            m_g = gmag + 5 * np.log10(plx / 1000.0) + 5 if np.isfinite(gmag) else np.nan
            params = {
                "ra": float(ra),
                "dec": float(dec),
                "source_id": str(clean_cell(row["Source"])).strip(),
                "gaia_ra": as_float(clean_cell(row["RA_ICRS"])),
                "gaia_dec": as_float(clean_cell(row["DE_ICRS"])),
                "Gmag": gmag,
                "BPmag": bpmag,
                "RPmag": rpmag,
                "BP_RP": bp_rp,
                "M_G": m_g,
                "Plx": plx,
                "e_Plx": as_float(clean_cell(row["e_Plx"])),
                "RUWE": as_float(clean_cell(row["RUWE"])),
                "Teff": as_float(clean_cell(row["Teff"])),
                "logg": as_float(clean_cell(row["logg"])),
                "dist_pc": dist,
                "gaia_match_sep_arcsec": sep,
            }
            if np.isfinite(bp_rp) and np.isfinite(m_g):
                params["hr_analysis"] = classify_hr_position(bp_rp, m_g, gaia_params=params)
            wd = ((params.get("hr_analysis") or {}).get("wd_model") or {})
            return {
                "source": "gaia_dr3_vizier_closest_positive_parallax",
                "gaia_source_id": params["source_id"],
                "parallax_mas": plx,
                "parallax_error_mas": params["e_Plx"],
                "distance_pc": dist,
                "Gmag": gmag,
                "BP_RP": bp_rp,
                "M_G": m_g,
                "wd_mass_msun": as_float(wd.get("mass_msun")),
                "wd_teff_k": as_float(wd.get("teff_k")),
                "wd_logg": as_float(wd.get("logg")),
                "wd_cooling_age_gyr": as_float(wd.get("cooling_age_gyr")),
                "gaia_match_sep_arcsec": sep,
                "gaia_ruwe": params["RUWE"],
                "raw": params,
            }
    except Exception:
        pass

    params = hr._query_gaia_params(float(ra), float(dec))
    if not params:
        return None
    plx = as_float(params.get("Plx"))
    dist = as_float(params.get("dist_pc"))
    if not np.isfinite(dist) and np.isfinite(plx) and plx > 0:
        dist = 1000.0 / plx
    if not np.isfinite(dist) and not np.isfinite(plx):
        return None
    wd = ((params.get("hr_analysis") or {}).get("wd_model") or {})
    return {
        "source": "gaia_dr3_vizier_1_over_parallax",
        "gaia_source_id": str(params.get("source_id", "")).strip(),
        "parallax_mas": plx,
        "parallax_error_mas": as_float(params.get("e_Plx")),
        "distance_pc": dist,
        "Gmag": as_float(params.get("Gmag")),
        "BP_RP": as_float(params.get("BP_RP")),
        "M_G": as_float(params.get("M_G")),
        "wd_mass_msun": as_float(wd.get("mass_msun")),
        "wd_teff_k": as_float(wd.get("teff_k")),
        "wd_logg": as_float(wd.get("logg")),
        "wd_cooling_age_gyr": as_float(wd.get("cooling_age_gyr")),
        "raw": params,
    }


def apply_gaia_to_row(df: pd.DataFrame, idx: int, info: dict[str, object]) -> None:
    mapping = {
        "distance_pc": "distance_pc",
        "distance_source": "source",
        "parallax_mas": "parallax_mas",
        "gaia_source_id": "gaia_source_id",
        "Gmag": "Gmag",
        "BP_RP": "BP_RP",
        "M_G": "M_G",
        "wd_mass_msun": "wd_mass_msun",
        "wd_teff_k": "wd_teff_k",
        "wd_logg": "wd_logg",
        "wd_cooling_age_gyr": "wd_cooling_age_gyr",
    }
    for col, key in mapping.items():
        if col not in df.columns:
            continue
        value = info.get(key)
        if col == "distance_source":
            if is_missing(df.at[idx, col]) or str(df.at[idx, col]).strip() == "":
                df.at[idx, col] = value
            continue
        if is_missing(df.at[idx, col]) or not np.isfinite(as_float(df.at[idx, col])):
            if value is not None and not (isinstance(value, float) and not np.isfinite(value)):
                df.at[idx, col] = value
    if "notes_for_gw_colleague" in df.columns:
        note = str(df.at[idx, "notes_for_gw_colleague"]) if not is_missing(df.at[idx, "notes_for_gw_colleague"]) else ""
        if "Gaia DR3 distance filled" not in note:
            suffix = f" Gaia DR3 distance filled from {info.get('source')}."
            df.at[idx, "notes_for_gw_colleague"] = (note.rstrip() + suffix).strip()


def update_target_column(path: Path, name_map: dict[str, str], stamp: str) -> int:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return 0
    if ".before_gw_fix_" in path.name:
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    changed = 0
    for col in ["target", "FirstColumn_23chars"]:
        if col in df.columns:
            new = df[col].astype(str).map(lambda x: name_map.get(x, x))
            changed += int((new != df[col].astype(str)).sum())
            df[col] = new
    if changed:
        backup_file(path, stamp)
        df.to_csv(path, index=False)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--astro-output", type=Path, default=DEFAULT_ASTRO_OUTPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--no-gaia-query", action="store_true")
    parser.add_argument("--radius-arcsec", type=float, default=None)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

    astro_output = args.astro_output
    gw_path = astro_output / "gw_calculator_input.csv"
    if not gw_path.exists():
        raise FileNotFoundError(gw_path)

    stamp = now_stamp()
    backups = []
    for path in [gw_path, astro_output / "integrated_catalog.csv", args.catalog]:
        b = backup_file(path, stamp)
        if b:
            backups.append(str(b))

    gw = pd.read_csv(gw_path)
    old_targets = gw["target"].astype(str).tolist()
    gw["target"] = [
        canonical_target(row["target"], row["ra"], row["dec"])
        for _, row in gw.iterrows()
    ]
    name_map = {
        old: new for old, new in zip(old_targets, gw["target"].astype(str)) if old != new
    }

    rename_rows = []
    for old, new in sorted(name_map.items()):
        action = merge_dirs(astro_output / old, astro_output / new)
        rename_rows.append({"old_target": old, "new_target": new, "directory_action": action})

    # Synchronize target names in top-level astro_output CSV tables and the main catalogue.
    csv_name_changes = []
    for path in sorted(astro_output.glob("*.csv")) + [args.catalog]:
        if ".before_gw_fix_" in path.name:
            continue
        if path == gw_path:
            continue
        n = update_target_column(path, name_map, stamp)
        if n:
            csv_name_changes.append({"path": str(path), "n_changed_cells": n})

    hr = HRDiagram()
    fill_rows = []
    n_missing_before = int(gw["distance_pc"].isna().sum()) if "distance_pc" in gw else 0
    for idx, row in gw.iterrows():
        target = str(row["target"])
        needs_distance = (
            "distance_pc" not in gw.columns
            or not np.isfinite(as_float(row.get("distance_pc")))
        )
        needs_parallax = (
            "parallax_mas" in gw.columns
            and not np.isfinite(as_float(row.get("parallax_mas")))
        )
        if not needs_distance and not needs_parallax:
            continue

        target_dir = astro_output / target
        info = read_local_gaia(target_dir)
        source_used = "local"
        if info is None and not args.no_gaia_query:
            try:
                info = gaia_query(hr, row["ra"], row["dec"])
                source_used = "gaia_query"
            except Exception as exc:
                fill_rows.append({
                    "target": target,
                    "status": "error",
                    "source": "gaia_query",
                    "message": f"{type(exc).__name__}: {exc}",
                })
                continue
        if info is None:
            fill_rows.append({
                "target": target,
                "status": "not_found",
                "source": "",
                "message": "No local Gaia params and Gaia query returned no usable positive parallax.",
            })
            continue

        apply_gaia_to_row(gw, idx, info)
        raw = info.get("raw")
        if isinstance(raw, dict):
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
                save_csv(raw, target_dir)
                save_analysis_report(raw, target_dir)
            except Exception:
                pass
        fill_rows.append({
            "target": target,
            "status": "filled",
            "source": info.get("source", source_used),
            "gaia_source_id": info.get("gaia_source_id", ""),
            "parallax_mas": info.get("parallax_mas", np.nan),
            "distance_pc": info.get("distance_pc", np.nan),
        })

    if "distance_pc" in gw.columns:
        gw["distance_pc"] = pd.to_numeric(gw["distance_pc"], errors="coerce")
    if "parallax_mas" in gw.columns:
        gw["parallax_mas"] = pd.to_numeric(gw["parallax_mas"], errors="coerce")
    gw.to_csv(gw_path, index=False)

    # Propagate filled distances and target names into integrated_catalog.csv.
    integrated_path = astro_output / "integrated_catalog.csv"
    if integrated_path.exists():
        integrated = pd.read_csv(integrated_path)
        if "target" in integrated.columns:
            integrated["target"] = integrated["target"].astype(str).map(lambda x: name_map.get(x, x))
            gw_lookup = gw.set_index("target")
            for idx, row in integrated.iterrows():
                target = str(row["target"])
                if target not in gw_lookup.index:
                    continue
                grows = gw_lookup.loc[target]
                if isinstance(grows, pd.DataFrame):
                    grows = grows.iloc[0]
                for col in [
                    "distance_pc", "distance_source", "parallax_mas",
                    "gaia_source_id", "Gmag", "BP_RP", "M_G",
                    "wd_mass_msun", "wd_teff_k", "wd_logg",
                    "wd_cooling_age_gyr",
                ]:
                    if col in integrated.columns and col in gw.columns:
                        value = grows[col]
                        if not is_missing(value):
                            integrated.at[idx, col] = value
                if "needs_distance_for_gw" in integrated.columns:
                    integrated.at[idx, "needs_distance_for_gw"] = not np.isfinite(as_float(grows.get("distance_pc")))
            integrated.to_csv(integrated_path, index=False)

    # Update main catalogue target and FirstColumn_23chars with the name map.
    if args.catalog.exists():
        cat = pd.read_csv(args.catalog)
        for col in ["target", "FirstColumn_23chars"]:
            if col in cat.columns:
                cat[col] = cat[col].astype(str).map(lambda x: name_map.get(x, x))
        cat.to_csv(args.catalog, index=False)

    fill_df = pd.DataFrame(fill_rows)
    rename_df = pd.DataFrame(rename_rows)
    csv_changes_df = pd.DataFrame(csv_name_changes)
    fill_df.to_csv(astro_output / "gw_distance_completion_log.csv", index=False)
    rename_df.to_csv(astro_output / "ren_target_rename_log.csv", index=False)
    csv_changes_df.to_csv(astro_output / "ren_target_rename_csv_changes.csv", index=False)

    n_missing_after = int(gw["distance_pc"].isna().sum()) if "distance_pc" in gw else 0
    summary = [
        "# GW distance and Ren-name fix",
        "",
        f"Updated: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Backups: {len(backups)}",
        f"- Ren/short target names converted: {len(name_map)}",
        f"- Missing distance before: {n_missing_before}",
        f"- Missing distance after: {n_missing_after}",
        f"- Filled rows: {int((fill_df.get('status') == 'filled').sum()) if not fill_df.empty else 0}",
        "",
        "## Ren name mapping",
    ]
    if rename_rows:
        for row in rename_rows:
            summary.append(f"- `{row['old_target']}` -> `{row['new_target']}` ({row['directory_action']})")
    else:
        summary.append("- No short Ren names needed conversion.")
    summary.extend([
        "",
        "## Output files",
        "- `gw_calculator_input.csv`: updated GW collaborator table.",
        "- `gw_distance_completion_log.csv`: per-target distance fill status.",
        "- `ren_target_rename_log.csv`: target-name and directory rename log.",
        "- `ren_target_rename_csv_changes.csv`: CSV tables whose target names were synchronized.",
    ])
    (astro_output / "README_GW_DISTANCE_REN_FIX.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("GW distance/name fix complete")
    print(f"  Ren short names converted: {len(name_map)}")
    print(f"  Missing distance before: {n_missing_before}")
    print(f"  Missing distance after: {n_missing_after}")
    print(f"  Filled rows: {int((fill_df.get('status') == 'filled').sum()) if not fill_df.empty else 0}")
    print(f"  Logs: {astro_output / 'gw_distance_completion_log.csv'}")


if __name__ == "__main__":
    main()
