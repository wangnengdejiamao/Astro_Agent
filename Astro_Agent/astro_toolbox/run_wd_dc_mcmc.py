#!/usr/bin/env python3
"""
DC white-dwarf MCMC fitter.

Featureless DC spectra cannot be fit with Balmer-line profiles; the standard
literature recipe (Bergeron, Leggett & Ruiz 2001 ApJS 133, 413; Bergeron+2019;
Blouin+2019; Gentile Fusillo+2021; Kilic+2020) is to fit the broadband
continuum shape against pure-H or pure-He model atmospheres, with the radius
constrained by the Gaia parallax and a canonical logg = 8.0 +/- 0.25 prior.
This driver runs both compositions (DA and DB neural-network grids), masks
Balmer/He I line cores in both data and model, jointly fits an optional
photometric SED, and selects the composition with the lower BIC.

Outputs (in --output-dir):
  wd_dc_summary.json
  wd_dc_results.csv
  wd_dc_quality.csv
  wd_dc_spectral_fit.png
  wd_dc_corner.png   (only if `corner` is importable)
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astro_toolbox import wd_fitting, config, utils  # noqa: E402

C_KMS = 2.99792458e5

# Wavelengths to mask in DC fit: Balmer + strong He I.
LINE_MASKS_A = [
    (6564.61, 35.0),  # H-alpha
    (4862.68, 30.0),  # H-beta
    (4341.68, 25.0),  # H-gamma
    (4102.89, 22.0),  # H-delta
    (3971.20, 20.0),  # H-epsilon
    (3890.16, 20.0),  # H8
    (4471.5,  15.0),  # He I
    (5875.6,  15.0),  # He I
    (6678.2,  15.0),  # He I
    (5015.7,  12.0),  # He I 5015
    (4922.0,  12.0),  # He I 4922
]


# ---- inputs ------------------------------------------------------------

def _first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _load_spectrum_csv(path):
    df = pd.read_csv(path)
    wave_col = _first_existing(
        df.columns, ['wavelength_A', 'wavelength', 'wave', 'lambda', 'lam'])
    flux_col = _first_existing(df.columns, ['flux', 'flam', 'f_lambda'])
    err_col = _first_existing(
        df.columns, ['error', 'flux_err', 'ivar_error', 'sigma', 'err'])
    if wave_col is None or flux_col is None:
        raise ValueError(
            'spectrum CSV must contain wavelength_A/wavelength and flux columns')
    wave = pd.to_numeric(df[wave_col], errors='coerce').to_numpy(float)
    flux = pd.to_numeric(df[flux_col], errors='coerce').to_numpy(float)
    err = None
    if err_col is not None:
        err = pd.to_numeric(df[err_col], errors='coerce').to_numpy(float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if err is not None and err.shape == wave.shape:
        good &= np.isfinite(err) & (err > 0)
    order = np.argsort(wave[good])
    out = {
        'wave': wave[good][order],
        'flux': flux[good][order],
        'err': err[good][order] if err is not None and err.shape == wave.shape else None,
    }
    if len(out['wave']) < 100:
        raise ValueError('not enough finite spectral pixels')
    return out


def _load_hr_params(path):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _float(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _build_photometry(hr):
    """Return list of {band, wave, flux_cgs, err_cgs} from the hr_diagram csv."""
    bands = []
    candidates = [
        ('Gaia_G',  ['Gmag'],  ['e_Gmag']),
        ('Gaia_BP', ['BPmag'], ['e_BPmag']),
        ('Gaia_RP', ['RPmag'], ['e_RPmag']),
        ('SDSS_u',  ['umag', 'SDSS_u'], ['e_umag']),
        ('SDSS_g',  ['gmag', 'SDSS_g'], ['e_gmag']),
        ('SDSS_r',  ['rmag', 'SDSS_r'], ['e_rmag']),
        ('SDSS_i',  ['imag', 'SDSS_i'], ['e_imag']),
        ('SDSS_z',  ['zmag', 'SDSS_z'], ['e_zmag']),
    ]
    for band, mag_keys, err_keys in candidates:
        mag = next((_float(hr.get(k)) for k in mag_keys if hr.get(k) is not None), None)
        err = next((_float(hr.get(k)) for k in err_keys if hr.get(k) is not None), None)
        if mag is None:
            continue
        info = config.BAND_INFO.get(band, {})
        wave_A = info.get('wave_A')
        zero_jy = info.get('zero_Jy', 3631.0)
        if wave_A is None:
            continue
        if err is None or err <= 0:
            err = 0.05  # 5% mag floor when no Gaia mag error provided
        f_obs, f_err = utils.mag_to_flux_cgs(mag, wave_A, err, zero_jy)
        if f_obs is None or f_err is None or f_obs <= 0 or f_err <= 0:
            continue
        bands.append({'band': band, 'wave': float(wave_A),
                      'flux': float(f_obs), 'err': float(f_err)})
    return bands


# ---- continuum prep ---------------------------------------------------

def _line_mask(wavelength_A):
    w = np.asarray(wavelength_A, dtype=float)
    keep = np.ones_like(w, dtype=bool)
    for centre, half in LINE_MASKS_A:
        keep &= ~((w >= centre - half) & (w <= centre + half))
    return keep


def _smooth(flux, win=25):
    flux = np.asarray(flux, dtype=float)
    if flux.size < 5:
        return flux.copy()
    win = max(int(win) | 1, 5)
    pad = win // 2
    padded = np.pad(flux, pad, mode='edge')
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(padded, kernel, mode='valid')


def _prepare_dc_spectrum(wave, flux, err, max_pixels=900):
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    e = None if err is None else np.asarray(err, dtype=float)
    mask = (w >= 3700.0) & (w <= 9200.0) & np.isfinite(w) & np.isfinite(f)
    if e is not None:
        mask &= np.isfinite(e) & (e > 0)
    w, f = w[mask], f[mask]
    e = e[mask] if e is not None else None
    if len(w) < 200:
        raise ValueError('not enough optical pixels for DC continuum fit')

    f_smooth = _smooth(f, win=25)
    line_keep = _line_mask(w)

    med_flux = float(np.nanmedian(np.abs(f_smooth)))
    if not np.isfinite(med_flux) or med_flux <= 0:
        med_flux = 1.0
    if e is None:
        e = np.full_like(f_smooth, 0.03 * med_flux)
    e_floor = np.maximum(0.03 * np.abs(f_smooth), 0.02 * med_flux)
    # The original per-pixel error is dominated by photon shot noise;
    # smoothing lowers the effective noise by sqrt(win) but template + flux
    # calibration systematics floor it at a few percent.
    e_smoothed = np.maximum(e / np.sqrt(25.0), e_floor)

    if len(w) > max_pixels:
        idx = np.linspace(0, len(w) - 1, int(max_pixels)).astype(int)
        w = w[idx]
        f_smooth = f_smooth[idx]
        e_smoothed = e_smoothed[idx]
        line_keep = line_keep[idx]
    return w, f_smooth, e_smoothed, line_keep


# ---- model evaluation -------------------------------------------------

def _model_at_observed(model_wave, model_flux_surface, w_obs, rv_kms):
    rest = w_obs / (1.0 + rv_kms / C_KMS)
    return np.interp(rest, model_wave, model_flux_surface, left=0.0, right=0.0)


def _model_band_flux(model_wave, model_flux_surface, scale, band_wave):
    val = np.interp(band_wave, model_wave, model_flux_surface,
                    left=np.nan, right=np.nan)
    return val * scale


# ---- single-composition MCMC -------------------------------------------

def _run_one_composition(specclass, spec_pack, photometry, priors, mcmc_cfg):
    arrays = wd_fitting._template_sampler_arrays(specclass, model_grid='auto')
    if arrays is None:
        return {'status': 'skipped',
                'error': f'no NN templates for {specclass}',
                'specclass': specclass}
    model_wave, labels, flux_grid, model_source = arrays
    t_min, t_max = float(np.min(labels[:, 0])), float(np.max(labels[:, 0]))
    g_min, g_max = float(np.min(labels[:, 1])), float(np.max(labels[:, 1]))

    w, f, e, line_keep = spec_pack
    flux_unit, flux_unit_label = wd_fitting._infer_observed_flux_unit(f)

    # Initial scale guess from a centre-of-grid model.
    t0 = float(priors.get('teff_init') or np.nanmedian(labels[:, 0]))
    g0 = float(priors.get('logg_init') or 8.0)
    t0 = float(np.clip(t0, t_min + 1.0, t_max - 1.0))
    g0 = float(np.clip(g0, g_min + 0.01, g_max - 0.01))
    rv0 = 0.0
    surface0 = wd_fitting._weighted_grid_spectrum(labels, flux_grid, t0, g0)
    model0 = _model_at_observed(model_wave, surface0, w, rv0)
    denom = float(np.sum((model0[line_keep] / e[line_keep]) ** 2))
    scale0 = (float(np.sum(f[line_keep] * model0[line_keep]
                           / e[line_keep] ** 2)) / denom
              if denom > 0 else 1e-20)
    scale0 = max(scale0, 1.0e-40)
    ln_scale0 = float(np.log(scale0))

    parallax_mas = priors.get('parallax_mas')
    parallax_err_mas = priors.get('parallax_err_mas')
    radius_prior = priors.get('gaia_radius_rsun')
    radius_prior_sigma = priors.get('gaia_radius_rsun_sigma')
    teff_prior = priors.get('gaia_teff')
    teff_prior_sigma = priors.get('gaia_teff_sigma')
    logg_prior = priors.get('logg_prior_mu', 8.0)
    logg_prior_sigma = priors.get('logg_prior_sigma', 0.25)

    band_waves = np.array([b['wave'] for b in photometry], dtype=float) \
        if photometry else np.empty(0)
    band_flux = np.array([b['flux'] for b in photometry], dtype=float) \
        if photometry else np.empty(0)
    band_err = np.array([b['err'] for b in photometry], dtype=float) \
        if photometry else np.empty(0)

    def _log_prob(theta):
        teff, logg, rv, ln_scale = theta
        if not (t_min <= teff <= t_max and g_min <= logg <= g_max
                and -650.0 <= rv <= 650.0 and -120.0 <= ln_scale <= 60.0):
            return -np.inf
        surface = wd_fitting._weighted_grid_spectrum(
            labels, flux_grid, teff, logg)
        model = _model_at_observed(model_wave, surface, w, rv)
        scale = np.exp(ln_scale)
        model = model * scale
        good = np.isfinite(model) & (model > 0) & line_keep
        if np.sum(good) < max(50, line_keep.sum() // 3):
            return -np.inf
        resid = f[good] - model[good]
        logp = -0.5 * np.sum(
            (resid / e[good]) ** 2 + np.log(2.0 * np.pi * e[good] ** 2))

        if band_waves.size:
            bm = np.interp(band_waves, model_wave, surface,
                           left=np.nan, right=np.nan) * (scale * flux_unit)
            ok = np.isfinite(bm) & (bm > 0)
            if np.any(ok):
                logp += -0.5 * np.sum(
                    ((band_flux[ok] - bm[ok]) / band_err[ok]) ** 2
                    + np.log(2.0 * np.pi * band_err[ok] ** 2))

        # logg canonical prior
        logp += -0.5 * ((logg - logg_prior) / max(logg_prior_sigma, 0.05)) ** 2
        # weak RV prior (DC has no lines to anchor RV)
        logp += -0.5 * (rv / 200.0) ** 2

        if teff_prior is not None:
            sig = max(float(teff_prior_sigma or 3000.0), 500.0)
            logp += -0.5 * ((teff - float(teff_prior)) / sig) ** 2

        # mass-radius radius prior
        mass_theta = wd_fitting._logg_to_mass(logg)
        radius_theta = wd_fitting.compute_wd_radius(mass_theta, logg)
        if (radius_prior is not None
                and radius_prior_sigma is not None
                and radius_prior_sigma > 0):
            logp += -0.5 * ((radius_theta - radius_prior)
                            / radius_prior_sigma) ** 2

        # parallax-driven scale->radius consistency
        if (parallax_mas is not None and np.isfinite(parallax_mas)
                and parallax_mas > 0):
            r_info = wd_fitting._distance_scale_radius(
                scale, parallax_mas, observed_flux_unit=flux_unit)
            if r_info is not None:
                par_frac = (
                    abs(parallax_err_mas / parallax_mas)
                    if (parallax_err_mas is not None
                        and np.isfinite(parallax_err_mas)
                        and parallax_err_mas > 0) else 0.05
                )
                sigma_r = np.hypot(max(0.18 * radius_theta, 0.0012),
                                   par_frac * r_info['radius_rsun'])
                logp += -0.5 * ((r_info['radius_rsun'] - radius_theta)
                                / sigma_r) ** 2
        return float(logp)

    try:
        import emcee
    except ImportError:
        emcee = None
    if emcee is None:
        return {'status': 'failed', 'error': 'emcee not installed',
                'specclass': specclass}

    nwalkers = max(int(mcmc_cfg['nwalkers']), 12)
    nsteps = int(mcmc_cfg['nsteps'])
    burn = int(mcmc_cfg['burn'])
    thin = max(int(mcmc_cfg['thin']), 1)
    rng = np.random.default_rng(int(mcmc_cfg['seed']))
    p0_centre = np.array([t0, g0, rv0, ln_scale0], dtype=float)
    p0 = np.repeat(p0_centre[None, :], nwalkers, axis=0)
    p0[:, 0] += rng.normal(0.0, 800.0, nwalkers)
    p0[:, 1] += rng.normal(0.0, 0.10, nwalkers)
    p0[:, 2] += rng.normal(0.0, 50.0, nwalkers)
    p0[:, 3] += rng.normal(0.0, 0.25, nwalkers)
    p0[:, 0] = np.clip(p0[:, 0], t_min + 1.0, t_max - 1.0)
    p0[:, 1] = np.clip(p0[:, 1], g_min + 0.01, g_max - 0.01)

    sampler = emcee.EnsembleSampler(nwalkers, 4, _log_prob)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sampler.run_mcmc(p0, nsteps, progress=False)
    flat = sampler.get_chain(discard=min(burn, nsteps - 1), thin=thin, flat=True)
    logp_chain = sampler.get_log_prob(
        discard=min(burn, nsteps - 1), thin=thin, flat=True)
    if flat.size == 0:
        return {'status': 'failed', 'error': 'empty MCMC chain',
                'specclass': specclass}

    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_idx = int(np.nanargmax(logp_chain))
    best = flat[best_idx]
    teff_med, logg_med, rv_med, ln_scale_med = q50

    # chi2 / BIC on continuum mask using the median-parameter model.
    surface_med = wd_fitting._weighted_grid_spectrum(
        labels, flux_grid, teff_med, logg_med)
    model_med = _model_at_observed(model_wave, surface_med, w, rv_med) \
        * np.exp(ln_scale_med)
    good_cont = np.isfinite(model_med) & (model_med > 0) & line_keep
    chi2_cont = float(np.sum(((f[good_cont] - model_med[good_cont])
                              / e[good_cont]) ** 2))
    n_cont = int(np.sum(good_cont))
    n_phot = int(band_waves.size)
    chi2_phot = 0.0
    if n_phot:
        bm = np.interp(band_waves, model_wave, surface_med,
                       left=np.nan, right=np.nan) * (
            np.exp(ln_scale_med) * flux_unit)
        ok = np.isfinite(bm) & (bm > 0)
        chi2_phot = float(np.sum(
            ((band_flux[ok] - bm[ok]) / band_err[ok]) ** 2))
    n_tot = n_cont + n_phot
    k_params = 4
    bic = chi2_cont + chi2_phot + k_params * np.log(max(n_tot, 1))

    # Residual diagnostics inside vs outside line masks.
    full_model = _model_at_observed(model_wave, surface_med, w, rv_med) \
        * np.exp(ln_scale_med)
    in_mask = ~line_keep
    out_mask = line_keep & np.isfinite(full_model) & (full_model > 0)
    in_rms = float(np.sqrt(np.nanmean(
        ((f[in_mask] - full_model[in_mask]) / e[in_mask]) ** 2))) \
        if np.sum(in_mask) else float('nan')
    out_rms = float(np.sqrt(np.nanmean(
        ((f[out_mask] - full_model[out_mask]) / e[out_mask]) ** 2))) \
        if np.sum(out_mask) else float('nan')

    scale_med = float(np.exp(ln_scale_med))
    r_info = wd_fitting._distance_scale_radius(
        scale_med, parallax_mas, observed_flux_unit=flux_unit) \
        if (parallax_mas is not None) else None

    mass_samples = np.array(
        [wd_fitting._logg_to_mass(float(g)) for g in flat[:, 1]],
        dtype=float)
    radius_mr_samples = wd_fitting.compute_wd_radius(mass_samples, flat[:, 1])
    mass_q = np.percentile(mass_samples, [16, 50, 84])
    radius_mr_q = np.percentile(radius_mr_samples, [16, 50, 84])

    return {
        'status': 'ok',
        'specclass': specclass,
        'model_grid': model_source,
        'flat_chain': flat,
        'log_prob_chain': logp_chain,
        'teff': float(teff_med),
        'teff_err_minus': float(teff_med - q16[0]),
        'teff_err_plus': float(q84[0] - teff_med),
        'teff_err': float(0.5 * ((teff_med - q16[0]) + (q84[0] - teff_med))),
        'logg': float(logg_med),
        'logg_err_minus': float(logg_med - q16[1]),
        'logg_err_plus': float(q84[1] - logg_med),
        'logg_err': float(0.5 * ((logg_med - q16[1]) + (q84[1] - logg_med))),
        'rv_kms': float(rv_med),
        'rv_err': float(0.5 * ((rv_med - q16[2]) + (q84[2] - rv_med))),
        'scale': scale_med,
        'best_teff': float(best[0]),
        'best_logg': float(best[1]),
        'best_rv_kms': float(best[2]),
        'best_scale': float(np.exp(best[3])),
        'mass_msun_mr': float(mass_q[1]),
        'mass_msun_mr_err_minus': float(mass_q[1] - mass_q[0]),
        'mass_msun_mr_err_plus': float(mass_q[2] - mass_q[1]),
        'radius_rsun_mr': float(radius_mr_q[1]),
        'radius_rsun_mr_err_minus': float(radius_mr_q[1] - radius_mr_q[0]),
        'radius_rsun_mr_err_plus': float(radius_mr_q[2] - radius_mr_q[1]),
        'radius_rsun_scale': float(r_info['radius_rsun']) if r_info else None,
        'flux_unit': flux_unit,
        'flux_unit_label': flux_unit_label,
        'chi2_continuum': chi2_cont,
        'chi2_phot': chi2_phot,
        'n_continuum_pixels': n_cont,
        'n_photometric_bands': n_phot,
        'chi2_red_continuum': float(chi2_cont / max(n_cont - k_params, 1)),
        'BIC': float(bic),
        'rms_inside_line_masks': in_rms,
        'rms_outside_line_masks': out_rms,
        'model_wave_obs': w,
        'model_flux_obs': full_model,
        'data_wave': w,
        'data_flux_smoothed': f,
        'data_err_smoothed': e,
        'line_keep': line_keep,
        'photometry': photometry,
        'photometric_model_flux': (
            np.interp(band_waves, model_wave, surface_med,
                      left=np.nan, right=np.nan)
            * (np.exp(ln_scale_med) * flux_unit)
        ).tolist() if band_waves.size else [],
    }


# ---- plotting ---------------------------------------------------------

def _plot_spectral_fit(best, output_path, title=''):
    w = best['data_wave']
    f = best['data_flux_smoothed']
    m = best['model_flux_obs']
    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(11, 6),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05},
        sharex=True)
    ax_main.plot(w, f, color='k', lw=0.7, label='data (boxcar-smoothed)')
    ax_main.plot(w, m, color='C3', lw=1.2,
                 label=(f"DC fit: {best['specclass']}, "
                        f"Teff={best['teff']:.0f} K, "
                        f"logg={best['logg']:.2f}"))
    for centre, half in LINE_MASKS_A:
        ax_main.axvspan(centre - half, centre + half,
                        color='gray', alpha=0.10)
    ax_main.set_ylabel('flux (input units)')
    ax_main.set_title(title)
    ax_main.legend(loc='best', fontsize=9)

    # photometry panel as bullets
    for band in best.get('photometry') or []:
        ax_main.errorbar(band['wave'], band['flux'] / best['flux_unit'],
                         yerr=band['err'] / best['flux_unit'],
                         fmt='o', color='C0', mec='k', ms=5,
                         label=None)

    resid = (f - m) / best['data_err_smoothed']
    ax_resid.axhline(0, color='gray', lw=0.5)
    ax_resid.plot(w, resid, color='k', lw=0.5)
    for centre, half in LINE_MASKS_A:
        ax_resid.axvspan(centre - half, centre + half,
                         color='gray', alpha=0.10)
    ax_resid.set_xlabel('wavelength (A)')
    ax_resid.set_ylabel('(data - model) / sigma')
    ax_resid.set_ylim(-6, 6)
    fig.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def _plot_corner(flat, output_path, labels=('Teff', 'logg', 'RV', 'lnScale')):
    try:
        import corner  # type: ignore
    except ImportError:
        return False
    fig = corner.corner(flat, labels=list(labels),
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True, title_fmt='.2f')
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return True


# ---- driver -----------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description='Fit a DC white-dwarf spectrum (featureless continuum) '
                    'using H and He NN atmospheres + photometry, MCMC.')
    p.add_argument('--spectrum-csv', required=True)
    p.add_argument('--hr-csv', default='')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--compositions', default='DA,DB',
                   help='comma-separated NN grids to try (DA = pure-H, DB = pure-He)')
    p.add_argument('--nwalkers', type=int, default=32)
    p.add_argument('--nsteps', type=int, default=1500)
    p.add_argument('--burn', type=int, default=400)
    p.add_argument('--thin', type=int, default=5)
    p.add_argument('--max-pixels', type=int, default=900)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--logg-prior-mu', type=float, default=8.0)
    p.add_argument('--logg-prior-sigma', type=float, default=0.25)
    p.add_argument('--gaia-teff-prior', type=float, default=None)
    p.add_argument('--gaia-teff-prior-sigma', type=float, default=2500.0)
    p.add_argument('--gaia-radius-prior', type=float, default=None)
    p.add_argument('--gaia-radius-prior-sigma', type=float, default=0.0025)
    p.add_argument('--no-photometry', action='store_true',
                   help='skip the photometric chi2 term even if hr-csv has mags')
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    spec = _load_spectrum_csv(args.spectrum_csv)
    hr = _load_hr_params(args.hr_csv)

    parallax_mas = _float(hr.get('Plx') or hr.get('parallax'))
    parallax_err_mas = _float(hr.get('e_Plx') or hr.get('parallax_error'))
    gaia_teff = args.gaia_teff_prior
    if gaia_teff is None:
        gaia_teff = _float(hr.get('wd_teff_k') or hr.get('gaia_hr_teff'))
    gaia_radius = args.gaia_radius_prior
    if gaia_radius is None:
        gaia_radius = _float(
            hr.get('wd_radius_rsun') or hr.get('gaia_hr_radius_rsun'))
    if gaia_radius is None:
        gaia_mass = _float(hr.get('wd_mass_msun') or hr.get('gaia_hr_mass'))
        gaia_logg = _float(hr.get('wd_logg') or hr.get('gaia_hr_logg'))
        if gaia_mass is not None and gaia_logg is not None:
            gaia_radius = float(wd_fitting.compute_wd_radius(gaia_mass, gaia_logg))

    photometry = [] if args.no_photometry else _build_photometry(hr)

    spec_pack = _prepare_dc_spectrum(
        spec['wave'], spec['flux'], spec['err'],
        max_pixels=args.max_pixels)

    priors = {
        'parallax_mas': parallax_mas,
        'parallax_err_mas': parallax_err_mas,
        'gaia_teff': gaia_teff,
        'gaia_teff_sigma': args.gaia_teff_prior_sigma,
        'gaia_radius_rsun': gaia_radius,
        'gaia_radius_rsun_sigma': args.gaia_radius_prior_sigma,
        'logg_prior_mu': args.logg_prior_mu,
        'logg_prior_sigma': args.logg_prior_sigma,
        'teff_init': gaia_teff,
        'logg_init': args.logg_prior_mu,
    }
    mcmc_cfg = {
        'nwalkers': args.nwalkers,
        'nsteps': args.nsteps,
        'burn': args.burn,
        'thin': args.thin,
        'seed': args.seed,
    }

    runs = {}
    for kind in [s.strip().upper() for s in args.compositions.split(',') if s.strip()]:
        runs[kind] = _run_one_composition(kind, spec_pack, photometry,
                                          priors, mcmc_cfg)

    ok_runs = {k: v for k, v in runs.items() if v.get('status') == 'ok'}
    if not ok_runs:
        msgs = '; '.join(f"{k}:{v.get('error', 'unknown')}" for k, v in runs.items())
        print(f'DC fit failed for all compositions: {msgs}')
        return 1
    best_key = min(ok_runs, key=lambda k: ok_runs[k]['BIC'])
    best = ok_runs[best_key]

    # ---- write outputs ----------------------------------------------------
    summary = {
        'status': 'ok',
        'spectral_class': 'DC',
        'best_composition': best_key,
        'compositions_tried': list(runs.keys()),
        'BIC_by_composition': {k: float(v['BIC'])
                               for k, v in ok_runs.items()},
        'teff_K': best['teff'],
        'teff_K_err': best['teff_err'],
        'logg': best['logg'],
        'logg_err': best['logg_err'],
        'rv_kms': best['rv_kms'],
        'rv_kms_err': best['rv_err'],
        'mass_msun_mr': best['mass_msun_mr'],
        'mass_msun_mr_err_minus': best['mass_msun_mr_err_minus'],
        'mass_msun_mr_err_plus': best['mass_msun_mr_err_plus'],
        'radius_rsun_mr': best['radius_rsun_mr'],
        'radius_rsun_mr_err_minus': best['radius_rsun_mr_err_minus'],
        'radius_rsun_mr_err_plus': best['radius_rsun_mr_err_plus'],
        'radius_rsun_from_scale': best.get('radius_rsun_scale'),
        'gaia_hr_teff_K': gaia_teff,
        'gaia_hr_radius_rsun': gaia_radius,
        'parallax_mas': parallax_mas,
        'chi2_red_continuum': best['chi2_red_continuum'],
        'chi2_continuum': best['chi2_continuum'],
        'chi2_phot': best['chi2_phot'],
        'n_continuum_pixels': best['n_continuum_pixels'],
        'n_photometric_bands': best['n_photometric_bands'],
        'BIC': best['BIC'],
        'rms_inside_line_masks': best['rms_inside_line_masks'],
        'rms_outside_line_masks': best['rms_outside_line_masks'],
        'model_grid': best['model_grid'],
        'flux_unit_label': best['flux_unit_label'],
        'photometric_bands': [b['band']
                              for b in (best.get('photometry') or [])],
    }
    summary_path = os.path.join(args.output_dir, 'wd_dc_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(summary, fh, indent=2, default=float)

    pd.DataFrame([{
        'composition': k,
        'status': v.get('status'),
        'teff_K': v.get('teff'),
        'teff_err_K': v.get('teff_err'),
        'logg': v.get('logg'),
        'logg_err': v.get('logg_err'),
        'mass_msun_mr': v.get('mass_msun_mr'),
        'radius_rsun_mr': v.get('radius_rsun_mr'),
        'radius_rsun_from_scale': v.get('radius_rsun_scale'),
        'chi2_continuum': v.get('chi2_continuum'),
        'chi2_phot': v.get('chi2_phot'),
        'n_continuum_pixels': v.get('n_continuum_pixels'),
        'n_photometric_bands': v.get('n_photometric_bands'),
        'chi2_red_continuum': v.get('chi2_red_continuum'),
        'BIC': v.get('BIC'),
    } for k, v in runs.items()]).to_csv(
        os.path.join(args.output_dir, 'wd_dc_results.csv'), index=False)

    # quality table (absolute pass/fail evidence)
    radius_frac_diff = (
        abs(best['radius_rsun_mr'] - gaia_radius) / gaia_radius
        if gaia_radius and gaia_radius > 0 else None
    )
    teff_sigma_offset = (
        (best['teff'] - gaia_teff)
        / max(args.gaia_teff_prior_sigma, 1.0)
        if gaia_teff is not None else None
    )
    quality = {
        'chi2_red_continuum': best['chi2_red_continuum'],
        'chi2_red_lt_3': bool(best['chi2_red_continuum'] < 3.0),
        'teff_err_K': best['teff_err'],
        'teff_err_lt_2000': bool(best['teff_err'] < 2000.0),
        'radius_frac_diff_vs_gaia': radius_frac_diff,
        'radius_within_25pct': (
            bool(radius_frac_diff < 0.25) if radius_frac_diff is not None else None
        ),
        'teff_sigma_offset_vs_gaia': teff_sigma_offset,
        'teff_within_3sigma_of_gaia': (
            bool(abs(teff_sigma_offset) < 3.0)
            if teff_sigma_offset is not None else None
        ),
        'rms_inside_line_masks': best['rms_inside_line_masks'],
        'rms_outside_line_masks': best['rms_outside_line_masks'],
        'rms_inside_le_outside_x2': bool(
            np.isfinite(best['rms_inside_line_masks'])
            and np.isfinite(best['rms_outside_line_masks'])
            and best['rms_inside_line_masks']
            <= 2.0 * best['rms_outside_line_masks']
        ),
        'BIC': best['BIC'],
        'best_composition': best_key,
    }
    pd.DataFrame([quality]).to_csv(
        os.path.join(args.output_dir, 'wd_dc_quality.csv'), index=False)

    plot_path = os.path.join(args.output_dir, 'wd_dc_spectral_fit.png')
    _plot_spectral_fit(best, plot_path,
                       title=(f"DC fit, best={best_key}, "
                              f"chi2_red={best['chi2_red_continuum']:.2f}"))
    corner_path = os.path.join(args.output_dir, 'wd_dc_corner.png')
    _plot_corner(best['flat_chain'], corner_path)

    print('DC fit OK')
    print(f"  best composition = {best_key}")
    for k, v in ok_runs.items():
        print(f"  {k}: BIC={v['BIC']:.1f} chi2_red_cont={v['chi2_red_continuum']:.2f} "
              f"Teff={v['teff']:.0f}+/-{v['teff_err']:.0f}K")
    print(f"  Teff = {best['teff']:.0f} +/- {best['teff_err']:.0f} K")
    print(f"  logg = {best['logg']:.2f} +/- {best['logg_err']:.2f}")
    print(f"  M(MR) = {best['mass_msun_mr']:.3f} M_sun")
    print(f"  R(MR) = {best['radius_rsun_mr']:.5f} R_sun")
    if gaia_teff is not None:
        print(f"  Gaia HR Teff = {gaia_teff:.0f} K, "
              f"sigma offset = {teff_sigma_offset:+.2f}")
    if gaia_radius is not None:
        print(f"  Gaia HR R = {gaia_radius:.5f} R_sun, "
              f"frac diff = {radius_frac_diff:+.3f}")
    print(f"  summary  -> {summary_path}")
    print(f"  fit_plot -> {plot_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
