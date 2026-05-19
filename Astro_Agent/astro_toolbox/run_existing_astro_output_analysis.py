#!/usr/bin/env python3
"""Offline rerun of analysis/plotting for existing astro_output folders.

This script intentionally separates analysis/plotting from data acquisition:
it never calls survey query or download functions.  It only reads files that
already exist inside each target folder, then writes refreshed science products
back into that same folder.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from astro_toolbox import (  # noqa: E402
    combined_plots,
    diagnostics,
    magnetic_field,
    period_analysis,
    rv_correction,
    rv_fitting,
    six_dim,
    wd_age_methods,
    wd_fitting,
)


WAVE_COLS = ('wavelength_A', 'wavelength', 'wave', 'lambda', 'lam')
FLUX_COLS = ('flux', 'flam', 'f_lambda')
ERR_COLS = ('error', 'flux_err', 'ivar_error', 'sigma', 'err')
MAG_COLS = ('mag', 'magnitude')
MAGERR_COLS = ('magerr', 'mag_err', 'e_mag', 'mag_error')
OPTICAL_SURVEYS = ('SDSS', 'DESI', 'LAMOST', 'KOA/LRIS', 'HST')


class LocalSed:
    """Small adapter matching combined_plots/diagnostics SED expectations."""

    def __init__(self, flux_data: dict[str, tuple[float, float, float]]):
        self.flux_data = flux_data
        self.ebv = None


def _first_existing(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if str(name).lower() in lower:
            return lower[str(name).lower()]
    return None


def _to_float_array(values):
    return pd.to_numeric(values, errors='coerce').to_numpy(dtype=float)


def _parse_target_coord(name):
    compact = name.split('_full_rerun')[0]
    if not compact.startswith('ZTFJ') or len(compact) < 20:
        return np.nan, np.nan
    body = compact[4:]
    sign_pos = max(body.find('+'), body.find('-'))
    if sign_pos <= 0:
        return np.nan, np.nan
    ra_s = body[:sign_pos]
    dec_s = body[sign_pos:]
    try:
        ra_h = float(ra_s[0:2])
        ra_m = float(ra_s[2:4])
        ra_sec = float(ra_s[4:])
        sign = -1.0 if dec_s[0] == '-' else 1.0
        dec_body = dec_s[1:]
        dec_d = float(dec_body[0:2])
        dec_m = float(dec_body[2:4])
        dec_sec = float(dec_body[4:])
        ra = 15.0 * (ra_h + ra_m / 60.0 + ra_sec / 3600.0)
        dec = sign * (dec_d + dec_m / 60.0 + dec_sec / 3600.0)
        return ra, dec
    except Exception:
        return np.nan, np.nan


def _is_target_dir(path, include_archival=False):
    if not path.is_dir() or not path.name.startswith('ZTFJ'):
        return False
    bad_parts = {
        '__pycache__',
        'toolbox_run',
        'toolbox_rerun',
        'wd_validation_20260428',
        'wd_validation_20260428_v2',
        'wd_validation_20260428_v3',
        'magnetic_field_batch',
        'six_dim',
    }
    if any(part in bad_parts for part in path.parts):
        return False
    if not include_archival and '_full_rerun' in path.name:
        return False
    return True


def iter_target_dirs(root, recursive=False, include_archival=False):
    root = Path(root).expanduser().resolve()
    candidates = root.rglob('ZTFJ*') if recursive else root.iterdir()
    for path in sorted(candidates):
        if _is_target_dir(path, include_archival=include_archival):
            yield path


def _read_spectrum_csv(path):
    df = pd.read_csv(path)
    wave_col = _first_existing(df.columns, WAVE_COLS)
    flux_col = _first_existing(df.columns, FLUX_COLS)
    err_col = _first_existing(df.columns, ERR_COLS)
    if wave_col is None or flux_col is None:
        raise ValueError('missing wavelength/flux column')
    wave = _to_float_array(df[wave_col])
    flux = _to_float_array(df[flux_col])
    err = _to_float_array(df[err_col]) if err_col is not None else None
    good = np.isfinite(wave) & np.isfinite(flux)
    if err is not None and err.shape == wave.shape:
        good &= np.isfinite(err) & (err > 0)
    order = np.argsort(wave[good])
    spec = {
        'wavelength': wave[good][order],
        'flux': flux[good][order],
        'error': err[good][order] if err is not None and err.shape == wave.shape else None,
        'file': str(path),
    }
    return spec, df


def _spectral_coverage(spec):
    wave = np.asarray(spec.get('wavelength', []), dtype=float)
    wave = wave[np.isfinite(wave)]
    if wave.size == 0:
        return {'n_points': 0, 'wmin': np.nan, 'wmax': np.nan,
                'n_balmer': 0, 'optical': False}
    balmer = (6562.8, 4861.3, 4340.5, 4101.7)
    wmin = float(np.nanmin(wave))
    wmax = float(np.nanmax(wave))
    return {
        'n_points': int(wave.size),
        'wmin': wmin,
        'wmax': wmax,
        'n_balmer': int(sum(wmin <= line <= wmax for line in balmer)),
        'optical': bool(wmax >= 3700.0 and wmin <= 9200.0),
    }


def _load_desi(path):
    spec, df = _read_spectrum_csv(path)
    if 'band' not in df.columns:
        return {'spectrum': {'B': spec}, 'source_file': str(path)}
    out = {}
    for band, grp in df.groupby('band'):
        wave_col = _first_existing(grp.columns, WAVE_COLS)
        flux_col = _first_existing(grp.columns, FLUX_COLS)
        err_col = _first_existing(grp.columns, ERR_COLS)
        if wave_col is None or flux_col is None:
            continue
        wave = _to_float_array(grp[wave_col])
        flux = _to_float_array(grp[flux_col])
        err = _to_float_array(grp[err_col]) if err_col is not None else np.zeros_like(flux)
        good = np.isfinite(wave) & np.isfinite(flux)
        if err.shape == wave.shape:
            good &= np.isfinite(err)
        order = np.argsort(wave[good])
        key = str(band).strip().upper()[:1] or 'B'
        out[key] = {
            'wavelength': wave[good][order],
            'flux': flux[good][order],
            'error': err[good][order] if err.shape == wave.shape else np.zeros(np.sum(good)),
        }
    return {'spectrum': out, 'source_file': str(path)}


def _load_spectra(source_dir):
    results = {}
    used = {}
    for path in sorted(source_dir.glob('*spectrum.csv')):
        name = path.name.lower()
        if 'magnetic_field' in name:
            continue
        try:
            if 'spherex' in name:
                spec, _ = _read_spectrum_csv(path)
                results['SPHEREx'] = spec
                used['SPHEREx'] = path.name
            elif 'desi' in name:
                results['DESI'] = _load_desi(path)
                used['DESI'] = path.name
            elif 'sdss' in name:
                spec, _ = _read_spectrum_csv(path)
                results['SDSS_spectrum'] = spec
                used['SDSS'] = path.name
            elif 'lamost' in name:
                spec, _ = _read_spectrum_csv(path)
                results['LAMOST'] = spec
                used['LAMOST'] = path.name
            elif 'koa' in name or 'lris' in name:
                spec, _ = _read_spectrum_csv(path)
                results['KOA_spectrum'] = spec
                used['KOA'] = path.name
            elif 'hst' in name:
                spec, _ = _read_spectrum_csv(path)
                cov = _spectral_coverage(spec)
                spec['usable_for_optical_rv'] = cov['optical'] and cov['n_balmer'] >= 2
                results['HST_spectrum'] = spec
                used['HST'] = path.name
            elif 'jwst' in name:
                spec, _ = _read_spectrum_csv(path)
                results['JWST_spectrum'] = spec
                used['JWST'] = path.name
        except Exception as exc:
            used[f'failed:{path.name}'] = str(exc)
    return results, used


def _load_ztf(path):
    df = pd.read_csv(path)
    if 'band' not in df.columns:
        return None
    need = {'mjd', 'mag', 'magerr'}
    if not need.issubset(set(df.columns)):
        return None
    out = {}
    for band, grp in df.groupby('band'):
        band = str(band).strip()
        if not band:
            continue
        sub = grp.copy()
        sub['mjd'] = pd.to_numeric(sub['mjd'], errors='coerce')
        if 'hjd' in sub.columns:
            sub['hjd'] = pd.to_numeric(sub['hjd'], errors='coerce')
        sub['mag'] = pd.to_numeric(sub['mag'], errors='coerce')
        sub['magerr'] = pd.to_numeric(sub['magerr'], errors='coerce')
        sub = sub[np.isfinite(sub['mjd']) & np.isfinite(sub['mag'])
                  & np.isfinite(sub['magerr']) & (sub['magerr'] > 0)]
        if len(sub):
            out[band] = sub
    return out or None


def _load_wise(path):
    df = pd.read_csv(path)
    if 'band' not in df.columns or 'mjd' not in df.columns:
        return None
    mag_col = _first_existing(df.columns, MAG_COLS)
    err_col = _first_existing(df.columns, MAGERR_COLS)
    if mag_col is None or err_col is None:
        return None
    out = {}
    for band, grp in df.groupby('band'):
        band = str(band).strip()
        sub = grp.copy()
        sub['mjd'] = pd.to_numeric(sub['mjd'], errors='coerce')
        sub['mag'] = pd.to_numeric(sub[mag_col], errors='coerce')
        sub['magerr'] = pd.to_numeric(sub[err_col], errors='coerce')
        sub = sub[np.isfinite(sub['mjd']) & np.isfinite(sub['mag'])
                  & np.isfinite(sub['magerr']) & (sub['magerr'] > 0)]
        if len(sub):
            out[band] = sub[['mjd', 'mag', 'magerr']].copy()
    return out or None


def _load_space_lc(path, default_system):
    df = pd.read_csv(path)
    time_col = _first_existing(
        df.columns,
        ('time_BTJD', 'time_BKJD', 'time', 'btjd', 'bkjd', 'mjd', 'jd'))
    flux_col = _first_existing(df.columns, ('flux', 'pdcsap_flux', 'sap_flux', 'rel_flux'))
    err_col = _first_existing(df.columns, ('flux_err', 'flux_error', 'pdcsap_flux_err', 'sap_flux_err', 'err'))
    if time_col is None or flux_col is None:
        return None
    time = _to_float_array(df[time_col])
    flux = _to_float_array(df[flux_col])
    ferr = _to_float_array(df[err_col]) if err_col is not None else None
    good = np.isfinite(time) & np.isfinite(flux)
    if ferr is not None and ferr.shape == time.shape:
        good &= np.isfinite(ferr)
    system = default_system
    low = str(time_col).lower()
    if 'bkjd' in low:
        system = 'BKJD'
    elif 'btjd' in low:
        system = 'BTJD'
    elif 'mjd' in low:
        system = 'MJD'
    return {
        'time': time[good],
        'flux': flux[good],
        'flux_err': ferr[good] if ferr is not None and ferr.shape == time.shape else None,
        'n_points': int(np.sum(good)),
        'time_system': system,
        'source_file': str(path),
    }


def _load_hst_like_lc(path):
    df = pd.read_csv(path)
    if 'mjd' not in df.columns:
        return None
    filt_col = _first_existing(df.columns, ('filter', 'band', 'filt'))
    mag_col = _first_existing(df.columns, MAG_COLS)
    err_col = _first_existing(df.columns, MAGERR_COLS)
    if filt_col is None or mag_col is None or err_col is None:
        return None
    out = {}
    for filt, grp in df.groupby(filt_col):
        sub = grp.copy()
        sub['mjd'] = pd.to_numeric(sub['mjd'], errors='coerce')
        sub['mag'] = pd.to_numeric(sub[mag_col], errors='coerce')
        sub['magerr'] = pd.to_numeric(sub[err_col], errors='coerce')
        sub = sub[np.isfinite(sub['mjd']) & np.isfinite(sub['mag'])
                  & np.isfinite(sub['magerr']) & (sub['magerr'] > 0)]
        if len(sub):
            out[str(filt)] = sub[['mjd', 'mag', 'magerr']].copy()
    return {'filters': out} if out else None


def _load_lightcurves(source_dir):
    results = {}
    used = {}
    for path in sorted(source_dir.glob('*lightcurve.csv')):
        name = path.name.lower()
        try:
            if 'ztf' in name:
                lc = _load_ztf(path)
                if lc:
                    results['ZTF_lightcurve'] = lc
                    used['ZTF'] = path.name
            elif 'wise' in name:
                lc = _load_wise(path)
                if lc:
                    results['WISE_lightcurve'] = lc
                    used['WISE'] = path.name
            elif 'tess' in name:
                lc = _load_space_lc(path, 'BTJD')
                if lc and lc['n_points'] > 0:
                    results['TESS'] = lc
                    used['TESS'] = path.name
            elif 'kepler' in name or 'k2' in name:
                lc = _load_space_lc(path, 'BKJD')
                if lc and lc['n_points'] > 0:
                    lc['survey'] = 'K2' if 'k2' in name else 'Kepler'
                    results['Kepler/K2'] = lc
                    used['Kepler/K2'] = path.name
            elif 'hst' in name:
                lc = _load_hst_like_lc(path)
                if lc:
                    results['HST_lightcurve'] = lc
                    used['HST_lightcurve'] = path.name
            elif 'jwst' in name:
                lc = _load_hst_like_lc(path)
                if lc:
                    results['JWST_lightcurve'] = lc
                    used['JWST_lightcurve'] = path.name
        except Exception as exc:
            used[f'failed:{path.name}'] = str(exc)
    return results, used


def _load_sed(source_dir):
    path = source_dir / 'sed_photometry.csv'
    if not path.exists():
        return None, None
    df = pd.read_csv(path)
    required = {'band', 'wave_A', 'flux_cgs'}
    if not required.issubset(df.columns):
        return None, None
    flux_data = {}
    photometry = {}
    for _, row in df.iterrows():
        band = str(row.get('band', '')).strip()
        if not band:
            continue
        wave = pd.to_numeric(pd.Series([row.get('wave_A')]), errors='coerce').iloc[0]
        flux = pd.to_numeric(pd.Series([row.get('flux_cgs')]), errors='coerce').iloc[0]
        ferr = pd.to_numeric(pd.Series([row.get('flux_err_cgs')]), errors='coerce').iloc[0]
        if np.isfinite(wave) and np.isfinite(flux) and flux > 0:
            flux_data[band] = (float(flux), float(ferr) if np.isfinite(ferr) else 0.0, float(wave))
        mag = pd.to_numeric(pd.Series([row.get('mag')]), errors='coerce').iloc[0]
        merr = pd.to_numeric(pd.Series([row.get('mag_err')]), errors='coerce').iloc[0]
        if np.isfinite(wave) and np.isfinite(mag):
            photometry[band] = (float(mag), float(merr) if np.isfinite(merr) else 0.05, float(wave))
    return (LocalSed(flux_data) if flux_data else None), (photometry or None)


def _load_selected_row(source_dir):
    candidates = [
        source_dir / 'sixdim_selected_row.csv',
        source_dir / 'six_dim' / 'sixdim_selected_row.csv',
    ]
    candidates.extend(sorted(source_dir.glob('*sixdim_selected_row.csv')))
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                if not df.empty:
                    return df.iloc[0].to_dict(), path
            except Exception:
                continue
    return None, None


def _best_optical_spectrum(results):
    candidates = _candidate_optical_spectra(results)
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, label, spec, cov = candidates[0]
    return label, spec, cov


def _candidate_optical_spectra(results):
    """Return all usable optical spectra with a score for WD fitting."""
    choices = []
    sdss = results.get('SDSS_spectrum')
    if sdss:
        choices.append(('SDSS', sdss))
    desi = results.get('DESI')
    if desi and isinstance(desi, dict) and 'spectrum' in desi:
        waves, fluxes, errs = [], [], []
        for band in ('B', 'R', 'Z'):
            if band in desi['spectrum']:
                sp = desi['spectrum'][band]
                waves.append(np.asarray(sp['wavelength'], dtype=float))
                fluxes.append(np.asarray(sp['flux'], dtype=float))
                errs.append(np.asarray(sp.get('error', np.zeros_like(waves[-1])), dtype=float))
        if waves:
            choices.append(('DESI', {
                'wavelength': np.concatenate(waves),
                'flux': np.concatenate(fluxes),
                'error': np.concatenate(errs),
            }))
    for key, label in (
            ('LAMOST', 'LAMOST'),
            ('KOA_spectrum', 'KOA/LRIS'),
            ('HST_spectrum', 'HST')):
        sp = results.get(key)
        if sp:
            choices.append((label, sp))
    candidates = []
    for label, spec in choices:
        cov = _spectral_coverage(spec)
        if not cov['optical'] or cov['n_points'] < 100:
            continue
        score = cov['n_balmer'] * 100000 + cov['n_points']
        candidates.append((score, label, spec, cov))
    return candidates


def _run_magnetic(results, output_dir, args):
    waves, fluxes, errs = [], [], []
    has_err = False
    for survey in OPTICAL_SURVEYS:
        if survey == 'SDSS':
            specs = [results.get('SDSS_spectrum')]
        elif survey == 'DESI':
            specs = []
            desi = results.get('DESI')
            if desi and isinstance(desi, dict):
                specs = list((desi.get('spectrum') or {}).values())
        elif survey == 'LAMOST':
            specs = [results.get('LAMOST')]
        elif survey == 'KOA/LRIS':
            specs = [results.get('KOA_spectrum')]
        else:
            specs = [results.get('HST_spectrum')]
        for spec in specs:
            if not spec:
                continue
            cov = _spectral_coverage(spec)
            if not cov['optical'] or cov['n_points'] < 80:
                continue
            wave = np.asarray(spec['wavelength'], dtype=float)
            flux = np.asarray(spec['flux'], dtype=float)
            err = spec.get('error')
            err = np.asarray(err, dtype=float) if err is not None else None
            good = np.isfinite(wave) & np.isfinite(flux) & (wave >= 3200) & (wave <= 10000)
            if err is not None and err.shape == wave.shape:
                good &= np.isfinite(err) & (err > 0)
                has_err = True
                errs.append(err[good])
            else:
                errs.append(np.full(np.sum(good), np.nan))
            waves.append(wave[good])
            fluxes.append(flux[good])
    if not waves:
        return {'status': 'skipped', 'reason': 'no_usable_optical_spectrum'}
    wave = np.concatenate(waves)
    flux = np.concatenate(fluxes)
    err = np.concatenate(errs) if has_err else None
    good = np.isfinite(wave) & np.isfinite(flux)
    if err is not None:
        good &= np.isfinite(err) & (err > 0)
    order = np.argsort(wave[good])
    rv_grid = np.arange(args.magnetic_rv_min,
                        args.magnetic_rv_max + 0.5 * args.magnetic_rv_step,
                        args.magnetic_rv_step)
    result = magnetic_field.measure_magnetic_field(
        wave[good][order], flux[good][order],
        err=err[good][order] if err is not None else None,
        series=args.magnetic_series,
        b_min_mg=args.magnetic_b_min_mg,
        b_max_mg=args.magnetic_b_max_mg,
        n_b_grid=args.magnetic_n_b_grid,
        rv_grid_kms=rv_grid,
        search_half_width_A=args.magnetic_search_half_width_A,
        min_depth=args.magnetic_min_depth,
        min_snr=args.magnetic_min_snr,
        min_trough_width_A=args.magnetic_min_trough_width_A,
        emission_avoid_A=args.magnetic_emission_avoid_A,
        absorption_core_avoid_A=args.magnetic_absorption_core_avoid_A,
        baseline_mode=args.magnetic_baseline_mode,
        wd_model_grid=args.magnetic_wd_model_grid,
        spectral_type='auto',
        field_mode=args.magnetic_field_mode,
        low_high_boundary_mg=args.magnetic_low_high_boundary_mg)
    if result is None:
        return {'status': 'failed', 'reason': 'fit_returned_none'}
    files = magnetic_field.save_magnetic_field_outputs(result, str(output_dir))
    n_det = int(result.get('n_detected_components', 0) or 0)
    review = str(result.get('magnetic_review_status', ''))
    if review == 'secure_zeeman':
        batch_class = 'strong_candidate'
    elif review == 'candidate_zeeman':
        batch_class = 'candidate'
    elif review == 'low_confidence_zeeman':
        batch_class = 'low_confidence_candidate'
    elif n_det < 3:
        batch_class = 'no_detection'
    else:
        batch_class = 'candidate'
    return {
        'status': 'ok',
        'batch_class': batch_class,
        'B_MG': result.get('B_MG'),
        'B_err_minus_MG': result.get('B_err_minus_MG'),
        'B_err_plus_MG': result.get('B_err_plus_MG'),
        'quality': result.get('quality'),
        'magnetic_claim': result.get('magnetic_claim'),
        'magnetic_review_status': result.get('magnetic_review_status'),
        'magnetic_review_reasons': result.get('magnetic_review_reasons'),
        'n_detected_components': n_det,
        'n_minor_absorption_components': result.get(
            'n_minor_absorption_components'),
        'weighted_detected_components': result.get(
            'weighted_detected_components'),
        'analysis_regime': result.get('analysis_regime'),
        'field_mode': result.get('field_mode'),
        'series_detected': result.get('series_detected'),
        'summary_file': files.get('summary', ''),
        'plot_file': files.get('plot', ''),
        'blue_region_plot': files.get('blue_region_plot_png', ''),
        'red_region_plot': files.get('red_region_plot_png', ''),
    }


def _physical_params_from_wd(wd_result):
    if not wd_result:
        return None
    params = dict(wd_result.get('physical_params') or {})
    if wd_result.get('classification') and 'spectral_type' not in params:
        params['spectral_type'] = wd_result['classification'].get('spectral_type')
    preferred = wd_result.get('balmer_fit') or wd_result.get('single_fit') or {}
    for key in ('teff_err', 'logg_err'):
        if key not in params and preferred.get(key) is not None:
            params[key] = preferred.get(key)
    if ('mass_err' not in params or 'radius_rsun_err' not in params):
        logg = _float_or_none(params.get('logg'))
        logg_err = _float_or_none(params.get('logg_err'))
        if logg is not None and logg_err is not None and logg_err > 0:
            try:
                lo = max(logg - logg_err, 6.0)
                hi = min(logg + logg_err, 10.0)
                m_lo = wd_fitting._logg_to_mass(lo)
                m_hi = wd_fitting._logg_to_mass(hi)
                r_lo = wd_fitting.compute_wd_radius(m_lo, lo)
                r_hi = wd_fitting.compute_wd_radius(m_hi, hi)
                params.setdefault('mass_err', abs(m_hi - m_lo) / 2.0)
                params.setdefault('radius_rsun_err', abs(r_hi - r_lo) / 2.0)
            except Exception:
                pass
    return params or None


def _float_or_none(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def analyze_target(source_dir, args):
    t0 = time.time()
    target = source_dir.name
    ra, dec = _parse_target_coord(target)
    results, spec_used = _load_spectra(source_dir)
    lc_results, lc_used = _load_lightcurves(source_dir)
    results.update(lc_results)
    sed_obj, photometry = _load_sed(source_dir)
    if sed_obj is not None:
        results['SED'] = sed_obj

    selected_row, selected_row_path = _load_selected_row(source_dir)
    if selected_row:
        ra = _float_or_none(selected_row.get('ra')) or ra
        dec = _float_or_none(selected_row.get('dec')) or dec

    row = {
        'target': target,
        'source_dir': str(source_dir),
        'ra': ra,
        'dec': dec,
        'spectra_files': ';'.join(f'{k}:{v}' for k, v in spec_used.items()),
        'lightcurve_files': ';'.join(f'{k}:{v}' for k, v in lc_used.items()),
        'has_sed_photometry': sed_obj is not None,
        'status': 'ok',
    }

    def step(name, fn):
        try:
            out = fn()
            row[f'{name}_status'] = 'ok'
            return out
        except Exception as exc:
            row[f'{name}_status'] = 'failed'
            row[f'{name}_error'] = str(exc)
            if args.verbose_errors:
                traceback.print_exc()
            return None

    if results and not args.skip_combined:
        out = step('combined_spectra', lambda: combined_plots.plot_combined_spectra(
            results, save_path=str(source_dir / 'combined_spectra.png'), ra=ra, dec=dec))
        row['combined_spectra_plot'] = str(source_dir / 'combined_spectra.png') if out else ''
        if sed_obj is not None:
            out = step('spectra_photometry', lambda: combined_plots.plot_spectra_with_photometry(
                results, save_path=str(source_dir / 'spectra_with_photometry.png'), ra=ra, dec=dec))
            row['spectra_with_photometry_plot'] = str(source_dir / 'spectra_with_photometry.png') if out else ''

    spec_diag = None
    if results and not args.skip_diagnostics:
        spec_diag = step('spectral_diagnostics', lambda: diagnostics.analyze_all_spectra(results))
        if spec_diag:
            saved = step('spectral_diagnostics_save',
                         lambda: diagnostics.save_spectral_diagnostics(spec_diag, str(source_dir)))
            if saved:
                row['spectral_diagnostics_csv'] = saved[0] or ''
                row['spectral_line_measurements_csv'] = saved[2] or ''

    if sed_obj is not None and not args.skip_diagnostics:
        sed_diag = step('sed_diagnostics', lambda: diagnostics.analyze_sed(sed_obj.flux_data))
        if sed_diag:
            saved = step('sed_diagnostics_save',
                         lambda: diagnostics.save_sed_diagnostics(sed_diag, str(source_dir)))
            if saved:
                row['sed_diagnostics_csv'] = saved[0] or ''

    pa = None
    if (not args.skip_period
            and any(k in results for k in ('ZTF_lightcurve', 'WISE_lightcurve', 'TESS',
                                           'Kepler/K2', 'Gaia_lightcurve',
                                           'HST_lightcurve', 'JWST_lightcurve'))):
        pa = step('period_analysis', lambda: period_analysis.run_period_analysis(
            results, str(source_dir), ra=ra, dec=dec, title_prefix=f'{target} '))
        if pa:
            results['period_analysis'] = pa
            row['reference_period_day'] = pa.get('reference_period')
            row['reference_period_hour'] = (
                pa.get('reference_period') * 24.0
                if pa.get('reference_period') else np.nan
            )
            row['reference_period_source'] = pa.get('reference_source')
            row['period_figures'] = ';'.join(pa.get('figures', []))
            p_csv = step('period_csv', lambda: period_analysis.save_csv(pa, str(source_dir)))
            row['period_analysis_csv'] = p_csv or ''
            out = step('combined_fold', lambda: combined_plots.plot_combined_fold(
                results, save_path=str(source_dir / 'combined_fold.png'), ra=ra, dec=dec))
            row['combined_fold_plot'] = str(source_dir / 'combined_fold.png') if out else ''

    rv_report = None
    if (not args.skip_rv
            and any(k in results for k in ('SDSS_spectrum', 'DESI', 'LAMOST',
                                           'KOA_spectrum', 'HST_spectrum'))):
        rv_report = step('rv_analysis', lambda: rv_fitting.run_rv_analysis(
            results, output_dir=str(source_dir), ra=ra, dec=dec))
        if rv_report:
            row['best_rv_kms'] = rv_report.get('best_rv')
            row['best_rv_err_kms'] = rv_report.get('best_rv_err')
            row['best_rv_source'] = rv_report.get('best_rv_source')

    wd_result = None
    if not args.skip_wd:
        wd_candidates = _candidate_optical_spectra(results)
        wd_candidates.sort(key=lambda item: item[0], reverse=True)
        if wd_candidates and wd_candidates[0][3].get('n_balmer', 0) >= 2:
            _, survey, spec, cov = wd_candidates[0]

            def _fit_one_wd(spec, output_dir):
                parallax = _float_or_none(
                    (selected_row or {}).get('parallax')
                    or (selected_row or {}).get('Plx')
                    or (selected_row or {}).get('plx'))
                bp_rp = _float_or_none(
                    (selected_row or {}).get('bp_rp')
                    or (selected_row or {}).get('BP_RP'))
                m_g = _float_or_none(
                    (selected_row or {}).get('M_G')
                    or (selected_row or {}).get('MG'))
                fitter = wd_fitting.WDFitter(
                    spec['wavelength'], spec['flux'], spec.get('error'),
                    model_grid=args.wd_model_grid)
                return fitter.run_all(
                    photometry=photometry,
                    parallax_mas=parallax,
                    bp_rp=bp_rp,
                    M_G=m_g,
                    output_dir=str(output_dir))

            def _wd():
                return _fit_one_wd(spec, source_dir)
            wd_result = step('wd_fitting', _wd)
            if wd_result:
                phys = wd_result.get('physical_params') or {}
                row['wd_spectrum_source'] = survey
                row['wd_spectral_type'] = (wd_result.get('classification') or {}).get('spectral_type')
                row['wd_teff'] = phys.get('teff')
                row['wd_logg'] = phys.get('logg')
                row['wd_mass_msun'] = phys.get('mass')
                row['wd_radius_rsun'] = phys.get('radius_rsun')
                row['wd_cooling_age_gyr'] = phys.get('cooling_age_gyr')
                sed_fit = wd_result.get('sed_fit') or {}
                if sed_fit:
                    row['wd_sed_chi2_photospheric'] = sed_fit.get('chi2_sed')
                    row['wd_sed_chi2_all'] = sed_fit.get('chi2_sed_all')
                    row['wd_sed_ir_excess_flag'] = sed_fit.get('ir_excess_flag')
                    row['wd_sed_ir_excess_bands'] = ';'.join(
                        sed_fit.get('ir_excess_bands', []))
                    row['wd_sed_max_ir_excess_dex'] = sed_fit.get(
                        'max_ir_excess_dex')
                    row['wd_sed_max_ir_excess_sigma'] = sed_fit.get(
                        'max_ir_excess_sigma')

            if len(wd_candidates) > 1:
                per_dir = source_dir / 'wd_fits_by_spectrum'
                per_dir.mkdir(exist_ok=True)
                per_rows = []
                for _, label, cand_spec, cand_cov in wd_candidates:
                    if cand_cov.get('n_balmer', 0) < 2:
                        continue
                    safe_label = label.lower().replace('/', '_').replace(' ', '_')
                    out_dir = per_dir / safe_label
                    out_dir.mkdir(exist_ok=True)

                    def _fit_cand(cand_spec=cand_spec, out_dir=out_dir):
                        return _fit_one_wd(cand_spec, out_dir)

                    cand_result = step(f'wd_fitting_{safe_label}', _fit_cand)
                    if not cand_result:
                        continue
                    phys = cand_result.get('physical_params') or {}
                    cl = cand_result.get('classification') or {}
                    per_rows.append({
                        'survey': label,
                        'n_points': cand_cov.get('n_points'),
                        'n_balmer': cand_cov.get('n_balmer'),
                        'spectral_type': cl.get('spectral_type'),
                        'teff': phys.get('teff'),
                        'logg': phys.get('logg'),
                        'mass_msun': phys.get('mass'),
                        'cooling_age_gyr': phys.get('cooling_age_gyr'),
                        'fit_dir': str(out_dir),
                        'wd_spectral_fit': str(out_dir / 'wd_spectral_fit.png'),
                    })
                if per_rows:
                    per_df = pd.DataFrame(per_rows)
                    per_csv = per_dir / 'wd_per_spectrum_fits.csv'
                    per_df.to_csv(per_csv, index=False)
                    row['wd_per_spectrum_fits'] = str(per_csv)
        else:
            row['wd_fitting_status'] = 'skipped'
            row['wd_fitting_error'] = 'no_optical_spectrum_with_balmer_coverage'

    if wd_result:
        phys = wd_result.get('physical_params') or {}
        cluster_age = _float_or_none((selected_row or {}).get('cluster_age_gyr'))
        if cluster_age is None:
            cluster_age_myr = _float_or_none((selected_row or {}).get('cluster_age_myr'))
            cluster_age = cluster_age_myr / 1000.0 if cluster_age_myr is not None else None
        mass = _float_or_none(phys.get('mass'))
        cooling = _float_or_none(phys.get('cooling_age_gyr'))
        if mass is not None and cooling is not None:
            age_cmp = step('wd_age_method_comparison',
                           lambda: wd_age_methods.compare_wd_age_methods(
                               m_final=mass,
                               cooling_age_gyr=cooling,
                               cluster_age_gyr=cluster_age,
                               method_label=target))
            if age_cmp:
                age_csv = source_dir / 'age_method_comparison.csv'
                wd_age_methods.save_age_method_comparison(age_cmp, age_csv)
                row['age_method_comparison_csv'] = str(age_csv)
                for key, value in age_cmp.items():
                    if key in {'method_label'}:
                        continue
                    row[f'agecmp_{key}'] = value

    physical_params = _physical_params_from_wd(wd_result)
    if physical_params and not args.skip_rv_correction:
        for label, spec in [('SDSS', results.get('SDSS_spectrum')),
                            ('DESI', _best_optical_spectrum({'DESI': results.get('DESI')})[1]
                             if results.get('DESI') else None),
                            ('LAMOST', results.get('LAMOST'))]:
            if not spec:
                continue
            cov = _spectral_coverage(spec)
            if cov['optical'] and cov['n_balmer'] >= 2:
                rv_true = step(f'rv_correction_{label.lower()}',
                               lambda spec=spec, label=label: rv_correction.run_rv_correction(
                                   spec['wavelength'], spec['flux'], spec.get('error'),
                                   physical_params=physical_params,
                                   survey_name=label,
                                   output_dir=str(source_dir), ra=ra, dec=dec))
                if rv_true:
                    row[f'{label.lower()}_rv_true_kms'] = rv_true.get('rv_true')
                    row[f'{label.lower()}_rv_true_err_kms'] = rv_true.get('rv_true_err')

    if selected_row is not None and not args.skip_sixdim:
        p5 = source_dir / 'sixdim_5d.png'
        step('sixdim_5d', lambda: six_dim.plot_5d_astrometry(selected_row, str(p5)))
        row['sixdim_5d_plot'] = str(p5) if p5.exists() else ''
        prv = source_dir / 'sixdim_rv_info.png'
        if all(k in selected_row for k in ('ra', 'dec')):
            step('sixdim_rv_info', lambda: six_dim.plot_rv_info(selected_row, str(prv)))
            row['sixdim_rv_info_plot'] = str(prv) if prv.exists() else ''
        row['sixdim_selected_row'] = str(selected_row_path)
    elif not args.skip_sixdim:
        row['sixdim_5d_status'] = 'skipped'
        row['sixdim_5d_error'] = 'no_local_sixdim_selected_row'

    if not args.skip_magnetic:
        mag = step('magnetic_field', lambda: _run_magnetic(results, source_dir, args))
        if mag:
            for k, v in mag.items():
                row[f'magnetic_{k}'] = v

    row['elapsed_sec'] = round(time.time() - t0, 2)
    return row


def write_summary(rows, root, args):
    summary = pd.DataFrame(rows)
    summary_name = Path(args.summary_name)
    summary_path = summary_name if summary_name.is_absolute() else Path(root) / summary_name
    summary.to_csv(summary_path, index=False)
    counts_rows = []
    for col in ('status', 'magnetic_batch_class', 'wd_fitting_status',
                'period_analysis_status', 'rv_analysis_status'):
        if col in summary.columns:
            counts = summary[col].fillna('').value_counts(dropna=False)
            for value, n in counts.items():
                counts_rows.append({'field': col, 'value': value, 'n_targets': int(n)})
    counts_name = Path(args.counts_name)
    counts_path = counts_name if counts_name.is_absolute() else Path(root) / counts_name
    pd.DataFrame(counts_rows).to_csv(counts_path, index=False)
    return summary_path, counts_path, summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Offline rerun of analysis/plotting for existing astro_output target folders.')
    parser.add_argument('root')
    parser.add_argument('--recursive', action='store_true',
                        help='Also scan nested ZTFJ* folders. Default is top-level folders only.')
    parser.add_argument('--include-archival-reruns', action='store_true',
                        help='Also process *_full_rerun_* folders.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--summary-name', default='offline_existing_analysis_summary.csv')
    parser.add_argument('--counts-name', default='offline_existing_analysis_counts.csv')
    parser.add_argument('--skip-combined', action='store_true',
                        help='Skip combined spectra/SED plotting.')
    parser.add_argument('--skip-diagnostics', action='store_true',
                        help='Skip spectral and SED diagnostics.')
    parser.add_argument('--skip-period', action='store_true',
                        help='Skip local light-curve period search and folded plots.')
    parser.add_argument('--skip-rv', action='store_true',
                        help='Skip CCF RV analysis.')
    parser.add_argument('--skip-rv-correction', action='store_true',
                        help='Skip gravitational-redshift RV_true correction.')
    parser.add_argument('--skip-sixdim', action='store_true',
                        help='Skip local 5D/6D summary plots.')
    parser.add_argument('--skip-wd', action='store_true',
                        help='Skip fast grid WD fitting.')
    parser.add_argument('--wd-model-grid', default='auto',
                        help='WD template grid for fast fitting; auto prefers local NN DA/DB grids.')
    parser.add_argument('--skip-magnetic', action='store_true')
    parser.add_argument('--magnetic-series', default='Halpha,Hbeta',
                        help='Default Halpha,Hbeta; Hgamma intentionally excluded.')
    parser.add_argument('--magnetic-b-min-mg', type=float, default=5.0)
    parser.add_argument('--magnetic-b-max-mg', type=float, default=950.0)
    parser.add_argument('--magnetic-n-b-grid', type=int, default=320)
    parser.add_argument('--magnetic-rv-min', type=float, default=-250.0)
    parser.add_argument('--magnetic-rv-max', type=float, default=250.0)
    parser.add_argument('--magnetic-rv-step', type=float, default=25.0)
    parser.add_argument('--magnetic-search-half-width-A', type=float, default=8.0)
    parser.add_argument('--magnetic-min-depth', type=float, default=0.04)
    parser.add_argument('--magnetic-min-snr', type=float, default=3.0)
    parser.add_argument('--magnetic-min-trough-width-A', type=float, default=6.0,
                        help='Reject narrow line-like features; low-field mode internally uses at least 8 A.')
    parser.add_argument('--magnetic-emission-avoid-A', type=float, default=10.0)
    parser.add_argument('--magnetic-absorption-core-avoid-A', type=float, default=25.0)
    parser.add_argument('--magnetic-baseline-mode', default='continuum',
                        choices=['template', 'continuum', 'auto'])
    parser.add_argument('--magnetic-wd-model-grid', default='auto')
    parser.add_argument('--magnetic-field-mode', default='auto',
                        choices=['auto', 'split', 'single', 'legacy', 'low', 'high'])
    parser.add_argument('--magnetic-low-high-boundary-mg', type=float, default=35.0)
    parser.add_argument('--verbose-errors', action='store_true')
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    targets = list(iter_target_dirs(
        root,
        recursive=args.recursive,
        include_archival=args.include_archival_reruns))
    if args.limit:
        targets = targets[:args.limit]
    print(f'Offline existing-data rerun: {len(targets)} target folders')
    rows = []
    for idx, source_dir in enumerate(targets, 1):
        print(f'[{idx}/{len(targets)}] {source_dir.name}', flush=True)
        try:
            rows.append(analyze_target(source_dir, args))
        except Exception as exc:
            if args.verbose_errors:
                traceback.print_exc()
            rows.append({
                'target': source_dir.name,
                'source_dir': str(source_dir),
                'status': 'failed',
                'error': str(exc),
            })
    summary_path, counts_path, summary = write_summary(rows, root, args)
    print(f'Wrote summary: {summary_path}')
    print(f'Wrote counts: {counts_path}')
    if 'magnetic_batch_class' in summary.columns:
        print(summary['magnetic_batch_class'].fillna('unknown').value_counts().to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
