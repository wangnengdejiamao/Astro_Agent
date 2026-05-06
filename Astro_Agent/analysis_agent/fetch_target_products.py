"""Fetch selected astro_toolbox products for a single target.

This is a focused recovery runner for cases where the full all-tools workflow
hangs or times out on unrelated survey modules. It writes module-by-module
status and uses the same astro_toolbox adapters as the production workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ASTRO_AGENT = REPO_ROOT / "Astro_Agent"
sys.path.insert(0, str(ASTRO_AGENT))

from astro_toolbox import hst, sed, wise, ztf  # noqa: E402
from astro_toolbox.desi import DESITool  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["module", "status", "output_dir", "note"])
        writer.writeheader()
        writer.writerows(rows)


def run_module(rows: list[dict[str, str]], module: str, out_dir: Path, func, timeout_sec: int = 180) -> Any:
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = signal.getsignal(signal.SIGALRM)

    def handle_timeout(signum, frame):
        raise TimeoutError(f"{module} timed out after {timeout_sec}s")

    try:
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(timeout_sec)
        result = func(out_dir)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        status = "ok" if result else "empty"
        rows.append({"module": module, "status": status, "output_dir": str(out_dir), "note": ""})
        write_json(out_dir / "summary.json", {"status": status, "result_type": type(result).__name__, "truthy": bool(result)})
        return result
    except Exception as exc:
        try:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        except Exception:
            pass
        note = f"{type(exc).__name__}: {exc}"
        rows.append({"module": module, "status": "error", "output_dir": str(out_dir), "note": note})
        (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch focused astro_toolbox products.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    def fetch_hst(out_dir: Path) -> dict[str, Any]:
        spec = hst.query_spectrum(args.ra, args.dec)
        lc = hst.query_lightcurve(args.ra, args.dec)
        if spec:
            hst.plot_spectrum(spec, str(out_dir / "hst_spectrum.png"))
            hst.save_spectrum_csv(spec, str(out_dir))
        if lc:
            hst.plot_lightcurve(lc, str(out_dir / "hst_lightcurve.png"))
            hst.save_lightcurve_csv(lc, str(out_dir))
        return {"spectrum": bool(spec), "lightcurve": bool(lc)}

    def fetch_desi(out_dir: Path) -> Any:
        tool = DESITool(output_dir=str(out_dir), log_func=print)
        return tool.process_single(args.ra, args.dec)

    def fetch_ztf(out_dir: Path) -> Any:
        result = ztf.query_lightcurve(args.ra, args.dec)
        if result:
            ztf.plot_lightcurve(result, str(out_dir / "ztf_lightcurve.png"))
            ztf.save_csv(result, str(out_dir))
        return result

    def fetch_wise(out_dir: Path) -> dict[str, Any]:
        phot = wise.get_photometry(args.ra, args.dec)
        if phot:
            wise.save_photometry_csv(phot, str(out_dir))
        lc = wise.query_lightcurve(args.ra, args.dec)
        if lc:
            wise.plot_lightcurve(lc, str(out_dir / "wise_lightcurve.png"))
            wise.save_lightcurve_csv(lc, str(out_dir))
        return {"photometry": bool(phot), "lightcurve": bool(lc)}

    def build_sed(out_dir: Path) -> Any:
        fitter = sed.SEDFitter(args.ra, args.dec)
        fitter.collect_photometry()
        fitter.apply_extinction()
        fitter.save_csv(str(out_dir))
        fitter.plot(str(out_dir / "sed.png"))
        return fitter.photometry

    for name, func in (
        ("hst", fetch_hst),
        ("desi", fetch_desi),
        ("ztf", fetch_ztf),
        ("wise", fetch_wise),
        ("sed", build_sed),
    ):
        write_status(root / "module_status.csv", rows)
        run_module(rows, name, root / name, func)
        write_status(root / "module_status.csv", rows)

    write_json(root / "run_summary.json", {"target": args.target, "ra_deg": args.ra, "dec_deg": args.dec, "modules": rows})
    print(json.dumps({"output_root": str(root), "modules": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
