#!/usr/bin/env python3
"""Run local NN-grid MCMC white-dwarf fitting for one spectrum CSV."""

import argparse
import os
import sys

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from astro_toolbox import wd_fitting  # noqa: E402


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
    if not path:
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return row


def _float_or_none(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Fit a WD spectrum with local DA/DB NN templates and MCMC.')
    parser.add_argument('--spectrum-csv', required=True)
    parser.add_argument('--hr-csv', default='')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--specclass', default='DA', choices=['DA', 'DB'])
    parser.add_argument('--teff-prior', type=float, default=None)
    parser.add_argument('--teff-prior-sigma', type=float, default=None)
    parser.add_argument('--nwalkers', type=int, default=32)
    parser.add_argument('--nsteps', type=int, default=1600)
    parser.add_argument('--burn', type=int, default=450)
    parser.add_argument('--thin', type=int, default=5)
    parser.add_argument('--max-pixels', type=int, default=900)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-grid', default='auto')
    parser.add_argument('--sampler', default='auto',
                        choices=['auto', 'emcee', 'importance', 'importance_sampling', 'fallback'])
    parser.add_argument('--gaia-teff-prior', type=float, default=None)
    parser.add_argument('--gaia-logg-prior', type=float, default=None)
    parser.add_argument('--gaia-mass-prior', type=float, default=None)
    parser.add_argument('--gaia-radius-prior', type=float, default=None)
    parser.add_argument('--gaia-teff-prior-sigma', type=float, default=2500.0)
    parser.add_argument('--gaia-logg-prior-sigma', type=float, default=0.20)
    parser.add_argument('--gaia-mass-prior-sigma', type=float, default=0.12)
    parser.add_argument('--gaia-radius-prior-sigma', type=float, default=0.0025)
    args = parser.parse_args(argv)

    spec = _load_spectrum_csv(args.spectrum_csv)
    hr = _load_hr_params(args.hr_csv)
    parallax_mas = _float_or_none(hr.get('Plx') or hr.get('parallax'))
    parallax_err_mas = _float_or_none(hr.get('e_Plx') or hr.get('parallax_error'))
    gaia_teff = args.gaia_teff_prior
    gaia_logg = args.gaia_logg_prior
    gaia_mass = args.gaia_mass_prior
    gaia_radius = args.gaia_radius_prior
    if gaia_teff is None:
        gaia_teff = _float_or_none(hr.get('wd_teff_k') or hr.get('gaia_hr_teff'))
    if gaia_logg is None:
        gaia_logg = _float_or_none(hr.get('wd_logg') or hr.get('gaia_hr_logg'))
    if gaia_mass is None:
        gaia_mass = _float_or_none(hr.get('wd_mass_msun') or hr.get('gaia_hr_mass'))
    if gaia_radius is None:
        gaia_radius = _float_or_none(hr.get('wd_radius_rsun') or hr.get('gaia_hr_radius_rsun'))
    if gaia_radius is None and gaia_mass is not None and gaia_logg is not None:
        gaia_radius = wd_fitting.compute_wd_radius(float(gaia_mass), float(gaia_logg))

    result = wd_fitting.fit_wd_mcmc_nn(
        spec['wave'], spec['flux'], spec['err'],
        specclass=args.specclass,
        parallax_mas=parallax_mas,
        parallax_err_mas=parallax_err_mas,
        teff_prior=args.teff_prior,
        teff_prior_sigma=args.teff_prior_sigma,
        gaia_teff_prior=gaia_teff,
        gaia_teff_prior_sigma=args.gaia_teff_prior_sigma,
        gaia_logg_prior=gaia_logg,
        gaia_logg_prior_sigma=args.gaia_logg_prior_sigma,
        gaia_mass_prior=gaia_mass,
        gaia_mass_prior_sigma=args.gaia_mass_prior_sigma,
        gaia_radius_prior=gaia_radius,
        gaia_radius_prior_sigma=args.gaia_radius_prior_sigma,
        nwalkers=args.nwalkers,
        nsteps=args.nsteps,
        burn=args.burn,
        thin=args.thin,
        random_seed=args.seed,
        output_dir=args.output_dir,
        max_pixels=args.max_pixels,
        model_grid=args.model_grid,
        sampler=args.sampler,
    )

    if result.get('status') != 'ok':
        print(f"WD NN MCMC failed: {result.get('error', 'unknown error')}")
        return 1

    print('WD NN MCMC OK')
    print(f"  grid = {result.get('model_grid')}")
    print(f"  Teff = {result['teff']:.0f} -{result['teff_err_minus']:.0f} +{result['teff_err_plus']:.0f} K")
    print(f"  logg = {result['logg']:.3f} -{result['logg_err_minus']:.3f} +{result['logg_err_plus']:.3f}")
    print(f"  RV = {result['rv_kms']:.1f} -{result['rv_err_minus']:.1f} +{result['rv_err_plus']:.1f} km/s")
    print(f"  M(MR) = {result['mass_msun_mr']:.3f} -{result['mass_msun_mr_err_minus']:.3f} +{result['mass_msun_mr_err_plus']:.3f} M_sun")
    print(f"  R(MR) = {result['radius_rsun_mr']:.5f} -{result['radius_rsun_mr_err_minus']:.5f} +{result['radius_rsun_mr_err_plus']:.5f} R_sun")
    if result.get('cooling_age_gyr') is not None:
        print(f"  cooling age = {result['cooling_age_gyr']:.3f} -{result.get('cooling_age_gyr_err_minus', 0):.3f} +{result.get('cooling_age_gyr_err_plus', 0):.3f} Gyr")
    if result.get('total_age_with_ms_gyr') is not None:
        print(f"  total age incl. MS = {result['total_age_with_ms_gyr']:.3f} -{result.get('total_age_with_ms_gyr_err_minus', 0):.3f} +{result.get('total_age_with_ms_gyr_err_plus', 0):.3f} Gyr")
    if result.get('scale_radius_rsun') is not None:
        print(f"  R(scale, Gaia) = {result['scale_radius_rsun']:.5f} -{result.get('scale_radius_rsun_err_minus', 0):.5f} +{result.get('scale_radius_rsun_err_plus', 0):.5f} R_sun")
    if result.get('mass_msun_from_scale_logg') is not None:
        print(f"  M(scale+logg) = {result['mass_msun_from_scale_logg']:.3f} -{result.get('mass_msun_from_scale_logg_err_minus', 0):.3f} +{result.get('mass_msun_from_scale_logg_err_plus', 0):.3f} M_sun")
    if result.get('v_grav_preferred_kms') is not None:
        print(f"  V_grav = {result['v_grav_preferred_kms']:.1f} +/- {result.get('v_grav_preferred_err', 0):.1f} km/s")
    if result.get('mcmc_warning'):
        print(f"  MCMC warning = {result.get('mcmc_warning')}")
    print(f"  summary = {result.get('summary_path')}")
    print(f"  fit_plot = {result.get('fit_plot_path')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
