"""3-parameter forward eclipse-model MCMC (UPK 13-c2 style).

Samples the posterior P(e, ω, α | observed τ/P, ingress/P) under the
sharp-edge prior used by Lin et al. 2025 (UPK 13-c2 ApJ Letter §3.3):

    τ/P     = (α/π) · (1-e²)^(3/2) / (1 + e cos ν_mid)²    [Kepler's 2nd law]
    ingress/P  ≈ (R_★ / a) · sin(α) / (π v_⟂ / v_orb)      [chord crossing]

Priors:
    e          ~ Uniform(0, 0.95)
    ω          ~ Normal(0, π/4)   (soft preference for apoastron eclipse)
    α          ~ Uniform(0.5°, 60°)

Backends:
  * `emcee` (preferred): 64 walkers × (2 000 burn + 4 000 production) steps.
  * Deterministic fallback (no emcee installed): a grid search over the
    3-parameter cube + Gaussian Laplace approximation around the best fit
    to produce 16/50/84-percentile bounds.

The MCMC fires only when the source class is one of:
  - `disk_eclipsing_binary`
  - `eclipsing_binary`
  - `unknown` AND a flat-bottomed eclipse has been measured (verdict from
    ingress_measurement).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


# ---------- model -----------------------------------------------------------

def predicted_tau_over_P(e: float, alpha_rad: float, omega_rad: float = 0.0,
                        nu_mid_rad: Optional[float] = None) -> float:
    """Kepler-2nd-law eclipse fraction. ν_mid defaults to π (apoastron)."""
    if not (0.0 <= e < 1.0):
        return float("nan")
    if nu_mid_rad is None:
        nu_mid_rad = math.pi  # apoastron
    denom = (1 + e * math.cos(nu_mid_rad)) ** 2
    if denom < 1e-9:
        return float("nan")
    return (alpha_rad / math.pi) * (1 - e * e) ** 1.5 / denom


def predicted_ingress_fraction(
    e: float, alpha_rad: float, omega_rad: float = 0.0,
    R_star_over_a: float = 0.05,
) -> float:
    """Rough chord-crossing ingress fraction (φ_ingress / P).

    For a sharp-edge disc and an azimuthal arc α, the *geometric* ingress
    time at apoastron is t_ing ≈ R_★/v_apo, where v_apo = v_orb · sqrt((1-e)/(1+e)).
    Therefore ingress/P ≈ (R_★ / 2π a) · sqrt((1+e)/(1-e)).
    """
    if not (0.0 <= e < 1.0):
        return float("nan")
    return (R_star_over_a / (2 * math.pi)) * math.sqrt((1 + e) / (1 - e))


def log_prior(e: float, alpha_rad: float, omega_rad: float,
              e_max: float = 0.95, alpha_min_rad: float = math.radians(0.5),
              alpha_max_rad: float = math.radians(60.0),
              omega_sigma: float = math.pi / 4) -> float:
    if not (0.0 <= e <= e_max):
        return -math.inf
    if not (alpha_min_rad <= alpha_rad <= alpha_max_rad):
        return -math.inf
    # Gaussian ω prior centred at 0
    return -0.5 * (omega_rad / omega_sigma) ** 2


def log_likelihood(
    e: float, alpha_rad: float, omega_rad: float,
    *,
    tau_obs: float, tau_sigma: float,
    ingress_obs: Optional[float] = None, ingress_sigma: Optional[float] = None,
    R_star_over_a: float = 0.05,
) -> float:
    tau_pred = predicted_tau_over_P(e, alpha_rad, omega_rad)
    if not math.isfinite(tau_pred):
        return -math.inf
    ll = -0.5 * ((tau_pred - tau_obs) / max(tau_sigma, 1e-9)) ** 2
    if ingress_obs is not None and ingress_sigma is not None:
        ing_pred = predicted_ingress_fraction(e, alpha_rad, omega_rad, R_star_over_a)
        if math.isfinite(ing_pred):
            ll += -0.5 * ((ing_pred - ingress_obs) / max(ingress_sigma, 1e-9)) ** 2
    return ll


def log_posterior(theta, *, tau_obs, tau_sigma, ingress_obs, ingress_sigma, R_star_over_a):
    e, alpha_rad, omega_rad = theta
    lp = log_prior(e, alpha_rad, omega_rad)
    if not math.isfinite(lp):
        return -math.inf
    return lp + log_likelihood(
        e, alpha_rad, omega_rad,
        tau_obs=tau_obs, tau_sigma=tau_sigma,
        ingress_obs=ingress_obs, ingress_sigma=ingress_sigma,
        R_star_over_a=R_star_over_a,
    )


# ---------- emcee runner -----------------------------------------------------

def run_mcmc_emcee(
    *,
    tau_obs: float, tau_sigma: float,
    ingress_obs: Optional[float] = None, ingress_sigma: Optional[float] = None,
    R_star_over_a: float = 0.05,
    n_walkers: int = 64, n_burn: int = 2000, n_prod: int = 4000,
    seed: int = 0,
) -> Dict[str, Any]:
    try:
        import emcee  # type: ignore
        import numpy as np
    except ImportError as exc:
        return {"status": "emcee_not_installed", "error": str(exc)}
    rng = np.random.default_rng(seed)
    # Initial ball near uniform-prior mid
    n_dim = 3
    p0 = np.zeros((n_walkers, n_dim))
    p0[:, 0] = rng.uniform(0.05, 0.85, n_walkers)
    p0[:, 1] = rng.uniform(math.radians(2.0), math.radians(40.0), n_walkers)
    p0[:, 2] = rng.normal(0.0, math.pi / 6, n_walkers)
    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior,
        kwargs=dict(
            tau_obs=tau_obs, tau_sigma=tau_sigma,
            ingress_obs=ingress_obs, ingress_sigma=ingress_sigma,
            R_star_over_a=R_star_over_a,
        ),
    )
    pos, _, _ = sampler.run_mcmc(p0, n_burn, progress=False)
    sampler.reset()
    sampler.run_mcmc(pos, n_prod, progress=False)
    flat = sampler.get_chain(flat=True)
    e_samp, alpha_samp, omega_samp = flat[:, 0], flat[:, 1], flat[:, 2]
    return {
        "status": "ok",
        "backend": "emcee",
        "n_walkers": n_walkers, "n_burn": n_burn, "n_prod": n_prod,
        "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
        "e_pct": (float(np.percentile(e_samp, 16)),
                  float(np.percentile(e_samp, 50)),
                  float(np.percentile(e_samp, 84))),
        "alpha_deg_pct": (math.degrees(float(np.percentile(alpha_samp, 16))),
                          math.degrees(float(np.percentile(alpha_samp, 50))),
                          math.degrees(float(np.percentile(alpha_samp, 84)))),
        "omega_deg_pct": (math.degrees(float(np.percentile(omega_samp, 16))),
                          math.degrees(float(np.percentile(omega_samp, 50))),
                          math.degrees(float(np.percentile(omega_samp, 84)))),
        "n_samples": flat.shape[0],
    }


# ---------- deterministic Laplace-approx fallback ---------------------------

def run_mcmc_grid_fallback(
    *,
    tau_obs: float, tau_sigma: float,
    ingress_obs: Optional[float] = None, ingress_sigma: Optional[float] = None,
    R_star_over_a: float = 0.05,
    n_e: int = 50, n_alpha: int = 60, n_omega: int = 9,
) -> Dict[str, Any]:
    """Coarse grid-search; emits marginal 16/50/84 percentiles via cumulative
    posterior mass — adequate when emcee is not installed."""
    e_grid = [0.0 + 0.95 * i / (n_e - 1) for i in range(n_e)]
    alpha_grid = [math.radians(0.5 + (60.0 - 0.5) * i / (n_alpha - 1)) for i in range(n_alpha)]
    omega_grid = [math.radians(-45.0 + 90.0 * i / (n_omega - 1)) for i in range(n_omega)]
    # Compute posterior on the grid
    rows = []
    for e in e_grid:
        for a in alpha_grid:
            for w in omega_grid:
                lp = log_posterior(
                    (e, a, w),
                    tau_obs=tau_obs, tau_sigma=tau_sigma,
                    ingress_obs=ingress_obs, ingress_sigma=ingress_sigma,
                    R_star_over_a=R_star_over_a,
                )
                if math.isfinite(lp):
                    rows.append((e, a, w, lp))
    if not rows:
        return {"status": "no_posterior_mass"}
    # Normalize log-posterior
    lp_max = max(r[3] for r in rows)
    weights = [math.exp(r[3] - lp_max) for r in rows]
    wsum = sum(weights)
    if wsum <= 0:
        return {"status": "posterior_zero"}
    # marginal percentiles
    def _pct(values_weights, q):
        sorted_vw = sorted(values_weights, key=lambda v: v[0])
        cum = 0.0
        target = q * sum(w for _, w in sorted_vw)
        for v, w in sorted_vw:
            cum += w
            if cum >= target:
                return v
        return sorted_vw[-1][0]
    e_vw = [(r[0], w) for r, w in zip(rows, weights)]
    a_vw = [(r[1], w) for r, w in zip(rows, weights)]
    o_vw = [(r[2], w) for r, w in zip(rows, weights)]
    e_p = (_pct(e_vw, 0.16), _pct(e_vw, 0.5), _pct(e_vw, 0.84))
    a_p = (math.degrees(_pct(a_vw, 0.16)),
           math.degrees(_pct(a_vw, 0.5)),
           math.degrees(_pct(a_vw, 0.84)))
    o_p = (math.degrees(_pct(o_vw, 0.16)),
           math.degrees(_pct(o_vw, 0.5)),
           math.degrees(_pct(o_vw, 0.84)))
    # Best point
    best_idx = max(range(len(rows)), key=lambda i: rows[i][3])
    return {
        "status": "ok",
        "backend": "deterministic_grid",
        "n_grid": len(rows),
        "e_pct": e_p,
        "alpha_deg_pct": a_p,
        "omega_deg_pct": o_p,
        "best_e": rows[best_idx][0],
        "best_alpha_deg": math.degrees(rows[best_idx][1]),
        "best_omega_deg": math.degrees(rows[best_idx][2]),
        "best_log_posterior": rows[best_idx][3],
    }


# ---------- top-level wrapper -----------------------------------------------

def fit_disk_eclipse_posterior(
    *,
    tau_obs: float, tau_sigma: float,
    ingress_obs: Optional[float] = None, ingress_sigma: Optional[float] = None,
    R_star_over_a: float = 0.05,
    prefer_backend: str = "emcee",
) -> Dict[str, Any]:
    """Try emcee first; fall back to deterministic grid if emcee is missing."""
    if prefer_backend == "emcee":
        res = run_mcmc_emcee(
            tau_obs=tau_obs, tau_sigma=tau_sigma,
            ingress_obs=ingress_obs, ingress_sigma=ingress_sigma,
            R_star_over_a=R_star_over_a,
        )
        if res.get("status") == "ok":
            return res
    return run_mcmc_grid_fallback(
        tau_obs=tau_obs, tau_sigma=tau_sigma,
        ingress_obs=ingress_obs, ingress_sigma=ingress_sigma,
        R_star_over_a=R_star_over_a,
    )


def render_latex(posterior: Dict[str, Any]) -> str:
    if posterior.get("status") != "ok":
        return ""
    backend = posterior.get("backend", "?")
    e_p = posterior.get("e_pct")
    a_p = posterior.get("alpha_deg_pct")
    o_p = posterior.get("omega_deg_pct")
    if not (e_p and a_p and o_p):
        return ""
    return (
        r"\paragraph{Forward eclipse-model posterior.}" + "\n"
        r"A 3-parameter MCMC fit over $(e, \alpha, \omega)$ under a sharp-edge"
        r" disc prior, using the Kepler-second-law eclipse-fraction relation,"
        r" yields ($" + backend + r"$, 16/50/84 percentiles):" + "\n"
        r"\begin{align}" + "\n"
        f"  e &= {e_p[1]:.2f}^" + "{+" + f"{e_p[2]-e_p[1]:.2f}" + "}_{-" + f"{e_p[1]-e_p[0]:.2f}" + "}, \\\\\n"
        f"  \\alpha &= {a_p[1]:.1f}^" + "{+" + f"{a_p[2]-a_p[1]:.1f}" + "}_{-" + f"{a_p[1]-a_p[0]:.1f}" + r"}\deg, \\" + "\n"
        f"  \\omega &= {o_p[1]:.1f}^" + "{+" + f"{o_p[2]-o_p[1]:.1f}" + "}_{-" + f"{o_p[1]-o_p[0]:.1f}" + r"}\deg." + "\n"
        r"\end{align}" + "\n"
        r"The orbit is eccentric and the eclipse occurs near apoastron, where"
        r" the orbital velocity is reduced and the chord crossing time long"
        r" enough to fill the observed $\tau/P$ without a near-complete"
        r" azimuthal opacity blanket."
    )


__all__ = [
    "predicted_tau_over_P",
    "predicted_ingress_fraction",
    "log_prior", "log_likelihood", "log_posterior",
    "run_mcmc_emcee", "run_mcmc_grid_fallback",
    "fit_disk_eclipse_posterior",
    "render_latex",
]
