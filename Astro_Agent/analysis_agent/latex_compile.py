"""Compile the drafted paper.tex into a PDF (graceful fallback).

Tries `latexmk -pdf` first (preferred — handles \\bibtex auto-runs), then
falls back to a manual `pdflatex; bibtex; pdflatex; pdflatex` loop.  If
neither is installed, returns status `"no_latex_compiler"` and does not
fail the workflow.

Used at the end of the drafter chain.  Also extracts the LAST 80 lines of
compiler output on failure so the user gets actionable error context
without having to dig through 2 MB of .log files.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def _which(*names: str) -> Optional[str]:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def compile_paper(
    *,
    workspace: Path,
    tex_filename: str = "paper.tex",
    bib_filename: str = "refs.bib",
    output_subdir: str = "build",
    timeout_sec: int = 120,
) -> Dict[str, Any]:
    workspace = Path(workspace)
    tex_path = workspace / "final" / tex_filename
    if not tex_path.exists():
        tex_path_alt = workspace / "drafts" / tex_filename
        if tex_path_alt.exists():
            tex_path = tex_path_alt
        else:
            return {"status": "no_tex_file",
                    "looked_at": [str(workspace / "final" / tex_filename),
                                  str(tex_path_alt)]}
    bib_path = workspace / bib_filename
    build_dir = workspace / output_subdir
    build_dir.mkdir(parents=True, exist_ok=True)

    latexmk = _which("latexmk")
    pdflatex = _which("pdflatex")
    bibtex = _which("bibtex")

    if not latexmk and not pdflatex:
        return {
            "status": "no_latex_compiler",
            "tex_path": str(tex_path),
            "note": "Install MacTeX / TeX Live (or `brew install mactex-no-gui`) "
                    "to enable PDF compilation.",
        }

    # Stage refs.bib into the build dir so bibtex finds it.
    if bib_path.exists():
        shutil.copy(bib_path, build_dir / bib_filename)
    # Stage figures alongside the tex
    fig_dir = workspace / "figures"
    if fig_dir.exists():
        for fpath in fig_dir.iterdir():
            if fpath.is_file():
                shutil.copy(fpath, build_dir / fpath.name)
    # Copy tex into build dir
    shutil.copy(tex_path, build_dir / tex_filename)

    if latexmk:
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
               tex_filename]
        try:
            proc = subprocess.run(cmd, cwd=str(build_dir),
                                  capture_output=True, text=True,
                                  timeout=timeout_sec)
            ok = proc.returncode == 0
            return {
                "status": "ok" if ok else "compile_failed",
                "compiler": "latexmk",
                "returncode": proc.returncode,
                "pdf_path": str(build_dir / tex_filename.replace(".tex", ".pdf"))
                            if ok else None,
                "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
                "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
                "errors": _extract_errors(proc.stdout + "\n" + proc.stderr),
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "compiler": "latexmk",
                    "timeout_sec": timeout_sec}
        except Exception as exc:
            return {"status": "error", "compiler": "latexmk",
                    "error": f"{type(exc).__name__}: {exc}"}

    # Fallback: manual pdflatex sequence
    if not pdflatex:
        return {"status": "no_latex_compiler"}
    try:
        env = dict()
        # First pass
        r1 = subprocess.run([pdflatex, "-interaction=nonstopmode", tex_filename],
                            cwd=str(build_dir), capture_output=True, text=True,
                            timeout=timeout_sec)
        # bibtex if available + refs.bib present
        if bibtex and bib_path.exists():
            base = tex_filename.replace(".tex", "")
            subprocess.run([bibtex, base], cwd=str(build_dir),
                           capture_output=True, text=True, timeout=timeout_sec)
            subprocess.run([pdflatex, "-interaction=nonstopmode", tex_filename],
                           cwd=str(build_dir), capture_output=True, text=True,
                           timeout=timeout_sec)
        r2 = subprocess.run([pdflatex, "-interaction=nonstopmode", tex_filename],
                            cwd=str(build_dir), capture_output=True, text=True,
                            timeout=timeout_sec)
        pdf_path = build_dir / tex_filename.replace(".tex", ".pdf")
        ok = r2.returncode == 0 and pdf_path.exists()
        return {
            "status": "ok" if ok else "compile_failed",
            "compiler": "pdflatex (manual sequence)",
            "returncode": r2.returncode,
            "pdf_path": str(pdf_path) if ok else None,
            "stdout_tail": (r2.stdout or "")[-2000:],
            "stderr_tail": (r2.stderr or "")[-2000:],
            "errors": _extract_errors((r1.stdout or "") + (r2.stdout or "")),
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "compiler": "pdflatex",
                "timeout_sec": timeout_sec}
    except Exception as exc:
        return {"status": "error", "compiler": "pdflatex",
                "error": f"{type(exc).__name__}: {exc}"}


# ---------- LaTeX error extraction -----------------------------------------

_ERROR_REGEX = re.compile(r"^!\s.+|^l\.\d+\s.+|Undefined control sequence", re.MULTILINE)


def _extract_errors(blob: str) -> list:
    """Pluck the most likely error lines from a pdflatex log/stdout dump."""
    if not blob:
        return []
    matches = _ERROR_REGEX.findall(blob)
    # De-dupe while preserving order
    seen: set = set()
    out = []
    for m in matches:
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
        if len(out) >= 20:
            break
    return out


__all__ = ["compile_paper"]
