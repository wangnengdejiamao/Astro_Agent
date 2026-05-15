"""Compact binary report builder for astro_toolbox outputs.

This module turns an astro_toolbox single-target output directory into a
human-readable science QA report.  It is intentionally conservative: it reports
which products exist, flags harmonic/alias risks, and avoids certifying final
stellar parameters when the required fit products are missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _first_existing(base: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        path = base / name
        if path.exists():
            return path
    return None


def _find_product(base: Path, filename: str) -> Optional[Path]:
    direct = base / filename
    if direct.exists():
        return direct
    hits = list(base.rglob(filename))
    return hits[0] if hits else None


def _summarize_periods(base: Path) -> Dict[str, Any]:
    path = _find_product(base, "period_analysis.csv")
    df = _read_csv(path) if path else None
    if df is None or df.empty:
        return {"status": "missing"}

    detected = df[df.get("detected", False).astype(str).str.lower().isin(["true", "1"])] if "detected" in df else df.iloc[0:0]
    candidates: List[Dict[str, Any]] = []
    for _, row in detected.iterrows():
        candidates.append(
            {
                "curve": row.get("curve"),
                "period_min": row.get("best_period_min"),
                "period_hour": row.get("best_period_hour"),
                "fap": row.get("fap"),
                "quality": row.get("quality"),
                "agreement": row.get("agreement"),
                "cross_survey_support": row.get("cross_survey_support"),
            }
        )

    orbital_candidate = None
    if "method" in df:
        harmonic = df[df["method"].astype(str).str.contains("2x_harmonic_candidate", case=False, na=False)]
        if not harmonic.empty:
            row = harmonic.iloc[0]
            orbital_candidate = {
                "period_min": row.get("best_period_min"),
                "period_hour": row.get("best_period_hour"),
                "source": row.get("curve"),
                "reason": "explicit 2x harmonic candidate from automated period analysis",
            }

    if orbital_candidate is None and candidates:
        short = [c for c in candidates if c.get("period_min") and 15 <= float(c["period_min"]) <= 25]
        if short:
            p = float(short[0]["period_min"]) * 2.0
            orbital_candidate = {
                "period_min": p,
                "period_hour": p / 60.0,
                "source": "2x of strongest short-period signal",
                "reason": "ellipsoidal/eclipsing morphology can put the strongest periodogram peak at half the orbital period",
            }

    return {
        "status": "ok",
        "path": str(path),
        "detected_candidates": candidates,
        "preferred_orbital_candidate": orbital_candidate,
        "alias_warning": bool(orbital_candidate is not None and candidates),
    }


def _summarize_sed(base: Path) -> Dict[str, Any]:
    diag_path = _find_product(base, "sed_diagnostics.csv")
    phot_path = _find_product(base, "sed_photometry.csv")
    diag = _read_csv(diag_path) if diag_path else None
    phot = _read_csv(phot_path) if phot_path else None
    out: Dict[str, Any] = {"status": "missing"}
    if diag is not None and not diag.empty:
        row = diag.iloc[0].to_dict()
        out.update({"status": "ok", "diagnostics": row, "diagnostics_path": str(diag_path)})
    if phot is not None:
        out["photometry_path"] = str(phot_path)
        out["photometry_bands"] = [str(x) for x in phot.get("band", [])]
        out["n_photometry_points"] = int(len(phot))
    return out


def _summarize_spectra(base: Path) -> Dict[str, Any]:
    diag_path = _find_product(base, "spectral_diagnostics.csv")
    line_path = _find_product(base, "spectral_line_measurements.csv")
    diag = _read_csv(diag_path) if diag_path else None
    lines = _read_csv(line_path) if line_path else None
    out: Dict[str, Any] = {"status": "missing"}
    if diag is not None:
        out = {
            "status": "ok",
            "diagnostics_path": str(diag_path),
            "surveys": diag.to_dict(orient="records"),
        }
    if lines is not None:
        strong_em = lines[lines.get("is_strong_emission", False).astype(str).str.lower().isin(["true", "1"])] if "is_strong_emission" in lines else lines.iloc[0:0]
        strong_abs = lines[lines.get("is_strong_absorption", False).astype(str).str.lower().isin(["true", "1"])] if "is_strong_absorption" in lines else lines.iloc[0:0]
        out["line_measurements_path"] = str(line_path)
        out["n_lines"] = int(len(lines))
        out["n_strong_emission"] = int(len(strong_em))
        out["n_strong_absorption"] = int(len(strong_abs))
    return out


def _summarize_rv(base: Path) -> Dict[str, Any]:
    path = _find_product(base, "rv_analysis.csv")
    df = _read_csv(path) if path else None
    if df is None or df.empty:
        return {"status": "missing"}
    best = df[df.get("method", "").astype(str).str.lower().eq("best")] if "method" in df else df
    row = (best.iloc[0] if not best.empty else df.iloc[0]).to_dict()
    return {"status": "ok", "path": str(path), "best": row}


def _summarize_refs(base: Path) -> Dict[str, Any]:
    refs_path = _find_product(base, "simbad_references.csv")
    if refs_path is None:
        refs_path = _find_product(base, "simbad_all_references.json")
    if refs_path is None:
        return {"status": "missing"}
    if refs_path.suffix.lower() == ".csv":
        df = _read_csv(refs_path)
        rows = [] if df is None else df.to_dict(orient="records")
    else:
        data = json.loads(refs_path.read_text(encoding="utf-8"))
        rows = data.get("references", data if isinstance(data, list) else [])
    key_refs = []
    for row in rows:
        title = str(row.get("title", ""))
        haystack = " ".join(str(row.get(k, "")) for k in ("title", "bibcode", "abstract", "summary"))
        if any(token.lower() in haystack.lower() for token in ["ZTF J213056", "Roche", "subdwarf", "ultracompact", "X-Ray", "X-ray"]):
            key_refs.append(row)
    return {
        "status": "ok",
        "path": str(refs_path),
        "n_refs": len(rows),
        "key_refs": key_refs[:12],
    }


def build_report(output_root: str | Path, target: str = "", ra: float | None = None, dec: float | None = None) -> Dict[str, Any]:
    base = Path(output_root).expanduser().resolve()
    products = {
        "summary": str(_first_existing(base, ["summary.txt", "run_summary.json"]) or ""),
        "hst_spectrum": str(_find_product(base, "hst_spectrum.csv") or ""),
        "spherex_spectrum": str(_find_product(base, "spherex_spectrum.csv") or ""),
        "ztf_lightcurve": str(_find_product(base, "ztf_lightcurve.csv") or ""),
        "tess_lightcurve": str(_find_product(base, "tess_lightcurve.csv") or ""),
        "wise_lightcurve": str(_find_product(base, "wise_lightcurve.csv") or ""),
        "hr_diagram": str(_find_product(base, "hr_diagram.png") or ""),
    }
    sections = {
        "periods": _summarize_periods(base),
        "sed": _summarize_sed(base),
        "spectra": _summarize_spectra(base),
        "rv": _summarize_rv(base),
        "references": _summarize_refs(base),
    }
    cautions = [
        "Do not treat the HST/FUV spectrum as an optical Balmer WD fit; it lacks optical Balmer coverage.",
        "The strongest photometric period can be half the orbital period for ellipsoidal/eclipsing light curves.",
    ]
    if sections["sed"].get("diagnostics", {}).get("status") in {"poor_fit", "no_fit"}:
        cautions.append("SED fitting is not cleared; final Teff/logg/mass/cooling-age claims need human/model review.")
    if sections["spectra"].get("n_strong_emission", 0) == 0:
        cautions.append("Automated line measurements found no robust strong emission lines in the available HST/FUV windows.")

    return {
        "target": target or base.name,
        "ra_deg": ra,
        "dec_deg": dec,
        "output_root": str(base),
        "products": products,
        "sections": sections,
        "cautions": cautions,
    }


def write_report(report: Dict[str, Any], output_dir: str | Path) -> Dict[str, str]:
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "compact_binary_report.json"
    md_path = out / "compact_binary_report.md"
    json_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")

    periods = report["sections"]["periods"]
    sed = report["sections"]["sed"]
    spectra = report["sections"]["spectra"]
    rv = report["sections"]["rv"]
    refs = report["sections"]["references"]

    lines = [
        f"# Compact Binary Report: {report['target']}",
        "",
        f"- Coordinates: RA={report.get('ra_deg')}, Dec={report.get('dec_deg')} deg",
        f"- Output root: `{report['output_root']}`",
        "",
        "## Period / Variability",
    ]
    preferred = periods.get("preferred_orbital_candidate")
    if preferred:
        lines.append(f"- Preferred orbital-period candidate: {preferred.get('period_min'):.6g} min ({preferred.get('period_hour'):.6g} h).")
        lines.append(f"- Reason: {preferred.get('reason')}.")
    for item in periods.get("detected_candidates", [])[:8]:
        lines.append(f"- {item.get('curve')}: {item.get('period_min')} min, quality={item.get('quality')}, agreement={item.get('agreement')}.")

    lines.extend(["", "## SED / Photometry"])
    lines.append(f"- SED status: {sed.get('diagnostics', {}).get('status', sed.get('status'))}.")
    lines.append(f"- Bands: {', '.join(sed.get('photometry_bands', [])) or 'missing'}.")

    lines.extend(["", "## Spectroscopy"])
    for row in spectra.get("surveys", []):
        lines.append(
            f"- {row.get('survey')}: status={row.get('status')}, region={row.get('spectral_region')}, "
            f"S/N={row.get('median_snr')}, interpretation={row.get('likely_interpretation')}."
        )
    lines.append(f"- Strong emission lines: {spectra.get('n_strong_emission', 0)}; strong absorption lines: {spectra.get('n_strong_absorption', 0)}.")

    lines.extend(["", "## Radial Velocity"])
    if rv.get("status") == "ok":
        best = rv.get("best", {})
        lines.append(f"- Best RV: {best.get('rv_kms')} +/- {best.get('rv_err_kms')} km/s from {best.get('source')} ({best.get('method')}).")
    else:
        lines.append("- RV product missing.")

    lines.extend(["", "## Literature / References"])
    lines.append(f"- Reference table: {refs.get('path', 'missing')} ({refs.get('n_refs', 0)} refs).")
    for ref in refs.get("key_refs", [])[:8]:
        lines.append(f"- {ref.get('year', '')} {ref.get('bibcode', '')}: {ref.get('title', '')} {ref.get('url', '')}")

    lines.extend(["", "## QA Cautions"])
    for caution in report["cautions"]:
        lines.append(f"- {caution}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact-binary science QA report from an astro_toolbox output directory.")
    parser.add_argument("output_root")
    parser.add_argument("--target", default="")
    parser.add_argument("--ra", type=float)
    parser.add_argument("--dec", type=float)
    parser.add_argument("--report-dir", default="")
    args = parser.parse_args()
    report = build_report(args.output_root, target=args.target, ra=args.ra, dec=args.dec)
    out = write_report(report, args.report_dir or args.output_root)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
