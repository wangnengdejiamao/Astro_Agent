"""Lightweight reward model for ranking paper sections during Best-of-N.

Trains a simple feature-engineering + LogisticRegression / Ridge model on
(text -> paper_qc score) pairs extracted from
`_prompt_experiments.sqlite`. We do NOT use deep embeddings to keep the
script dependency-light; the goal is a *re-ranking* signal, not absolute
ground truth.

Features (deterministic, fast):
  * char length, word count, sentence count
  * number of \\pm symbols
  * number of \\citep keys
  * number of chi^2 expressions
  * presence of forbidden hype words (-1 per occurrence)
  * abstract length (within 120-350 band)
  * uncertainty density (\\pm per number)
  * fraction of sentences ending in citation

Target: paper_qc_pass - 2*paper_qc_fail (clamped to [-4, +16]).

Outputs a small JSON model + a `score(text)` callable that
paper_orchestra.write_section can import when best_of_n > 1.

Usage:
    python scripts/prompt_tuning/reward_model.py --train
    python scripts/prompt_tuning/reward_model.py --rank file1.tex file2.tex
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PKG = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PKG))


MODEL_PATH = Path(__file__).resolve().parent / "reward_model.json"
DB_PATH = PKG / "output" / "analysis_agent" / "_prompt_experiments.sqlite"


_FORBIDDEN = (
    "obviously", "remarkable", "remarkably", "groundbreaking",
    "we believe", "novel result", "clearly demonstrates",
    "unprecedented", "incredibly", "amazing", "astonishing",
)


def featurise(text: str) -> Dict[str, float]:
    text = text or ""
    lower = text.lower()
    sentences = re.split(r"(?<=[.?!])\s+", text)
    n_pm = len(re.findall(r"\\pm|±", text))
    n_cite = len(re.findall(r"\\cite[pt]?\{", text))
    n_chi = len(re.findall(r"\\chi\^|chi\^2|\$\\chi", text))
    n_num = len(re.findall(r"(?<![A-Za-z_\\])-?\d+(?:\.\d+)?", text))
    n_hype = sum(lower.count(h) for h in _FORBIDDEN)
    n_words = len(re.findall(r"\b[A-Za-z][A-Za-z\-']+", text))
    n_chars = len(text)
    n_sent = max(1, len([s for s in sentences if len(s) > 10]))
    cite_per_sent = n_cite / n_sent
    pm_per_num = n_pm / max(n_num, 1)
    abstract_bonus = 0.0
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, flags=re.DOTALL)
    if m:
        abs_words = len(re.findall(r"\b[A-Za-z][A-Za-z\-']+", m.group(1)))
        if 120 <= abs_words <= 350:
            abstract_bonus = 1.0
        elif 80 <= abs_words < 120 or 350 < abs_words <= 450:
            abstract_bonus = 0.5
    return {
        "len_chars": float(n_chars),
        "len_words": float(n_words),
        "n_sentences": float(n_sent),
        "n_pm": float(n_pm),
        "n_cite": float(n_cite),
        "n_chi2": float(n_chi),
        "n_numbers": float(n_num),
        "n_hype": float(n_hype),
        "cite_per_sent": float(cite_per_sent),
        "pm_per_num": float(pm_per_num),
        "abstract_bonus": float(abstract_bonus),
    }


def _normalise(features: List[Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-feature mean / std on a list of dicts."""
    if not features:
        return {}, {}
    keys = features[0].keys()
    mean = {k: sum(f[k] for f in features) / len(features) for k in keys}
    var = {
        k: sum((f[k] - mean[k]) ** 2 for f in features) / max(len(features) - 1, 1)
        for k in keys
    }
    std = {k: max(var[k] ** 0.5, 1e-6) for k in keys}
    return mean, std


def _z(f: Dict[str, float], mean: Dict[str, float], std: Dict[str, float]) -> Dict[str, float]:
    return {k: (f[k] - mean.get(k, 0.0)) / std.get(k, 1.0) for k in f}


def fit_linear(X: List[Dict[str, float]], y: List[float]) -> Dict[str, float]:
    """Closed-form OLS on the normalised feature dicts."""
    if not X:
        return {}
    keys = list(X[0].keys())
    n = len(X)
    p = len(keys)
    # Build matrix manually to avoid numpy dependency if not installed.
    try:
        import numpy as np
    except Exception:
        # Pure-python fallback: per-feature univariate slope (crude).
        out: Dict[str, float] = {"_intercept": sum(y) / max(n, 1)}
        for k in keys:
            xs = [row[k] for row in X]
            mu_x = sum(xs) / n
            mu_y = sum(y) / n
            num = sum((xs[i] - mu_x) * (y[i] - mu_y) for i in range(n))
            den = sum((xs[i] - mu_x) ** 2 for i in range(n)) or 1.0
            out[k] = num / den
        return out
    Xm = np.array([[row[k] for k in keys] for row in X], dtype=float)
    Xm = np.hstack([Xm, np.ones((n, 1))])  # intercept col
    ym = np.array(y, dtype=float)
    # Ridge with small lambda for stability with tiny n
    lam = 1.0
    A = Xm.T @ Xm + lam * np.eye(Xm.shape[1])
    b = Xm.T @ ym
    w = np.linalg.solve(A, b)
    coefs = {k: float(w[i]) for i, k in enumerate(keys)}
    coefs["_intercept"] = float(w[-1])
    return coefs


def predict(text: str, model: Dict[str, Any]) -> float:
    feats = featurise(text)
    mean = model.get("feature_mean", {})
    std = model.get("feature_std", {})
    coefs = model.get("coefficients", {})
    z = _z(feats, mean, std) if mean else feats
    pred = coefs.get("_intercept", 0.0) + sum(coefs.get(k, 0.0) * z[k] for k in z)
    return float(pred)


def load_trainset(db_path: Path = DB_PATH) -> List[Tuple[str, float]]:
    """Pull (output, target) pairs from _prompt_experiments.sqlite. Output
    text is NOT stored in the db (only hashes), so we approximate from
    the metadata: target = n_pass - 2*n_fail. Without the actual text we
    cannot featurise. As a stopgap, we walk recent runs' final/paper.tex
    files and read their paper_qc to construct (text, score) pairs."""
    pairs: List[Tuple[str, float]] = []
    base = PKG / "output" / "analysis_agent"
    if not base.exists():
        return pairs
    for run_dir in sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        qc = run_dir / "09_paper_qc.json"
        tex = run_dir / "paper_orchestra" / "final" / "paper.tex"
        if not (qc.exists() and tex.exists()):
            tex = run_dir / "paper_orchestra" / "drafts" / "paper.tex"
        if not (qc.exists() and tex.exists()):
            continue
        try:
            qd = json.loads(qc.read_text(encoding="utf-8"))
            text = tex.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        score = float((qd.get("n_pass") or 0) - 2 * (qd.get("n_fail") or 0))
        pairs.append((text, score))
    return pairs


def train(db_path: Path = DB_PATH, save_to: Path = MODEL_PATH) -> Dict[str, Any]:
    pairs = load_trainset(db_path)
    if len(pairs) < 3:
        return {"status": "insufficient_data", "n": len(pairs)}
    X = [featurise(t) for t, _ in pairs]
    y = [s for _, s in pairs]
    mean, std = _normalise(X)
    Xn = [_z(f, mean, std) for f in X]
    coefs = fit_linear(Xn, y)
    # Self-consistency: training R^2
    preds = [coefs.get("_intercept", 0.0) + sum(coefs.get(k, 0.0) * Xn[i][k] for k in Xn[i])
             for i in range(len(Xn))]
    ss_res = sum((y[i] - preds[i]) ** 2 for i in range(len(y)))
    ss_tot = sum((y[i] - sum(y) / len(y)) ** 2 for i in range(len(y))) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    model = {
        "status": "ok",
        "n_train": len(pairs),
        "feature_mean": mean,
        "feature_std": std,
        "coefficients": coefs,
        "train_r2": round(r2, 3),
    }
    save_to.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return model


def rank(text_paths: List[str], model_path: Path = MODEL_PATH) -> List[Dict[str, Any]]:
    if not model_path.exists():
        raise RuntimeError("Model not trained; run --train first")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for p in text_paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace") if Path(p).exists() else p
        out.append({"path": p, "score": predict(text, model)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--rank", nargs="*", default=[])
    ap.add_argument("--evaluate", action="store_true",
                    help="train, then compare predicted ranking against actual paper_qc on a hold-out set")
    args = ap.parse_args()
    if args.train:
        m = train()
        print(json.dumps({"status": m.get("status"),
                          "n_train": m.get("n_train"),
                          "train_r2": m.get("train_r2"),
                          "saved_to": str(MODEL_PATH)}, indent=2, ensure_ascii=False))
        return 0
    if args.rank:
        ranked = rank(args.rank)
        for r in ranked:
            print(f"  {r['score']:7.3f}  {r['path']}")
        return 0
    if args.evaluate:
        pairs = load_trainset()
        if len(pairs) < 4:
            print(json.dumps({"status": "insufficient_data", "n": len(pairs)}))
            return 1
        split = max(1, len(pairs) // 5)
        train_pairs, hold = pairs[split:], pairs[:split]
        # train on majority
        X = [featurise(t) for t, _ in train_pairs]
        y = [s for _, s in train_pairs]
        mean, std = _normalise(X)
        Xn = [_z(f, mean, std) for f in X]
        coefs = fit_linear(Xn, y)
        model = {"feature_mean": mean, "feature_std": std, "coefficients": coefs}
        preds = [predict(t, model) for t, _ in hold]
        truths = [s for _, s in hold]
        # Spearman rho (rank correlation)
        def _rank(xs):
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            rk = [0]*len(xs)
            for r, i in enumerate(order):
                rk[i] = r
            return rk
        rp, rt = _rank(preds), _rank(truths)
        n = len(rp)
        d2 = sum((rp[i] - rt[i])**2 for i in range(n))
        rho = 1 - 6*d2/(n*(n**2 - 1)) if n >= 2 else 0.0
        print(json.dumps({"holdout_n": n, "spearman_rho": round(rho, 3),
                          "preds": preds, "truths": truths}, indent=2))
        return 0
    print("nothing to do; pass --train, --rank, or --evaluate")
    return 1


if __name__ == "__main__":
    sys.exit(main())
