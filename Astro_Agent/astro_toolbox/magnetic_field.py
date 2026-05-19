"""
Magnetic white-dwarf Zeeman-field measurement.

This module matches weak absorption dips against precomputed Balmer Zeeman
component curves.  It is deliberately conservative around emission features:
known nebular/He emission wavelengths and data-driven emission peaks are masked
before a component can contribute to the magnetic-field score.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config, utils


C_KMS = 2.99792458e5

DEFAULT_TEMPLATE_DIR = os.path.join(
    config.SCRIPT_DIR, 'data', 'magnetic_zeeman', 'balmer_series_data')

BALMER_REST = {
    'Halpha': 6564.61,
    'Hbeta': 4862.68,
    'Hgamma': 4341.68,
}

CORE_SIDE_MAX_A = {
    'Halpha': 650.0,
    'Hbeta': 320.0,
    'Hgamma': 260.0,
}

WAVE_COLS = ('wavelength_A', 'wavelength', 'wave', 'lambda', 'lam')
FLUX_COLS = ('flux', 'flam', 'f_lambda')
ERR_COLS = ('error', 'flux_err', 'ivar_error', 'sigma', 'err')

STANDARD_ABSORPTION_CORES = {
    'H-alpha': 6564.61,
    'H-beta': 4862.68,
    'H-gamma': 4341.68,
    'H-delta': 4102.89,
    'H-epsilon': 3971.20,
    'H-zeta': 3890.16,
    'Ca II K': 3933.7,
    'Ca II H': 3968.5,
}

SERIES_LABELS = {
    'Halpha': 'H-alpha',
    'Hbeta': 'H-beta',
    'Hgamma': 'H-gamma',
}

GUIDED_ZEEMAN_PRESETS = {
    # Reference nodes from cechichang.ipynb.  These are deliberately line-ID
    # guided: weak high-field Zeeman troughs are broad and the blind dip search
    # can otherwise lock onto continuum ripples.
    'cechichang_220': [
        {'series': 'Hbeta', 'component': 5, 'observed_wavelength_A': 3420.0,
         'sigma_A': 35.0, 'region': 'blue'},
        {'series': 'Hbeta', 'component': 3, 'observed_wavelength_A': 3680.0,
         'sigma_A': 30.0, 'region': 'blue'},
        {'series': 'Hbeta', 'component': 1, 'observed_wavelength_A': 3994.0,
         'sigma_A': 60.0, 'region': 'blue'},
        {'series': 'Hbeta', 'component': 6, 'observed_wavelength_A': 4250.0,
         'sigma_A': 35.0, 'region': 'blue'},
        {'series': 'Hbeta', 'component': 2, 'observed_wavelength_A': 4680.0,
         'sigma_A': 35.0, 'region': 'blue'},
        {'series': 'Halpha', 'component': 2, 'observed_wavelength_A': 5824.7,
         'sigma_A': 30.0, 'region': 'red'},
        {'series': 'Halpha', 'component': 1, 'observed_wavelength_A': 6774.1,
         'sigma_A': 30.0, 'region': 'red'},
        {'series': 'Halpha', 'component': 3, 'observed_wavelength_A': 7245.2,
         'sigma_A': 50.0, 'region': 'red'},
        {'series': 'Halpha', 'component': 4, 'observed_wavelength_A': 8306.5,
         'sigma_A': 40.0, 'region': 'red'},
    ],
}

# Lines that are often emission contaminants in optical WD spectra.  Balmer
# cores are handled by the data-driven emission mask, not masked blindly.
KNOWN_EMISSION_LINES = {
    'He I 4026': 4026.2,
    'He I 4471': 4471.5,
    'He II 4686': 4685.7,
    'He I 4922': 4921.9,
    '[O III] 4959': 4958.9,
    '[O III] 5007': 5006.8,
    'He I 5876': 5875.6,
    '[N II] 6548': 6548.1,
    '[N II] 6583': 6583.5,
    'He I 6678': 6678.2,
    '[S II] 6716': 6716.4,
    '[S II] 6731': 6730.8,
    'He I 7065': 7065.2,
    '[O I] sky 5577': 5577.34,
    'Na I D2/sky': 5889.95,
    'Na I D1/sky': 5895.92,
    '[O I] sky 6300': 6300.30,
    '[O I] sky 6364': 6363.78,
}

KNOWN_ARTIFACT_BANDS = {
    'telluric/OH A': (7470.0, 7725.0),
}

GREEK_SERIES = {
    '\u03b1': 'Halpha',
    '\u03b2': 'Hbeta',
    '\u03b3': 'Hgamma',
}


def _first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def coerce_wavelength_to_angstrom(wave):
    """
    Return wavelengths in Angstrom using common optical spectrum units.

    The magnetic-field templates are tabulated in Angstrom.  Local spectra are
    normally already in Angstrom, but some externally processed files use
    meters, microns, or nanometers.  The conversion is deliberately based on the
    wavelength scale only, so existing Angstrom CSV products pass through.
    """
    wave = np.asarray(wave, dtype=float)
    finite = wave[np.isfinite(wave)]
    if finite.size == 0:
        return wave
    scale = np.nanmedian(np.abs(finite))
    if not np.isfinite(scale) or scale <= 0:
        return wave
    if scale < 1e-3:
        return wave * 1e10  # meters
    if scale < 100.0:
        return wave * 1e4  # microns
    if scale < 2000.0:
        return wave * 10.0  # nanometers
    return wave


def _odd_window_from_angstrom(wave, width_A=140.0, minimum=51, maximum=401):
    wave = np.asarray(wave, dtype=float)
    finite = wave[np.isfinite(wave)]
    if finite.size < 3:
        window = minimum
    else:
        diffs = np.diff(np.sort(finite))
        dw = np.nanmedian(diffs[np.isfinite(diffs) & (diffs > 0)])
        if not np.isfinite(dw) or dw <= 0:
            window = minimum
        else:
            window = int(round(float(width_A) / dw))
    window = int(max(minimum, min(maximum, window)))
    if window % 2 == 0:
        window += 1
    return window


def read_spectrum_file(path, wave_col=None, flux_col=None, err_col=None):
    """Read a CSV/TSV or whitespace two/three-column spectrum file."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in ('.csv', '.tsv'):
        sep = '\t' if ext == '.tsv' else ','
        df = pd.read_csv(path, sep=sep)
        wave_col = wave_col or _first_existing(df.columns, WAVE_COLS)
        flux_col = flux_col or _first_existing(df.columns, FLUX_COLS)
        err_col = err_col or _first_existing(df.columns, ERR_COLS)
        if wave_col is None or flux_col is None:
            raise ValueError(f'{path} needs wavelength and flux columns')
        wave = pd.to_numeric(df[wave_col], errors='coerce').to_numpy(float)
        flux = pd.to_numeric(df[flux_col], errors='coerce').to_numpy(float)
        err = None
        if err_col is not None and err_col in df.columns:
            err = pd.to_numeric(df[err_col], errors='coerce').to_numpy(float)
    else:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f'{path} needs at least two numeric columns')
        wave = data[:, 0].astype(float)
        flux = data[:, 1].astype(float)
        err = data[:, 2].astype(float) if data.shape[1] >= 3 else None
    wave = coerce_wavelength_to_angstrom(wave)
    return wave, flux, err


def _parse_component_name(path):
    base = os.path.basename(path)
    m = re.match(r'H(.)(\d+)', base)
    if not m:
        return None
    series = GREEK_SERIES.get(m.group(1))
    if series is None:
        return None
    return series, int(m.group(2))


def load_balmer_zeeman_templates(template_dir=None, interpolated=True,
                                 b_max_mg=1000.0,
                                 wavelength_min_A=2500.0,
                                 wavelength_max_A=12000.0):
    """
    Load Balmer Zeeman component curves.

    Returns a dataframe with columns:
    ``series, component, B_MG, wavelength_A, source_file``.
    """
    template_dir = template_dir or DEFAULT_TEMPLATE_DIR
    npz_path = os.path.join(template_dir, 'balmer_zeeman_templates.npz')
    if interpolated and os.path.exists(npz_path):
        for allow_pickle in (False, True):
            try:
                db = np.load(npz_path, allow_pickle=allow_pickle)
                out = pd.DataFrame({
                    'series': db['series'].astype(str),
                    'component': db['component'].astype(int),
                    'B_MG': db['B_MG'].astype(float),
                    'wavelength_A': db['wavelength_A'].astype(float),
                    'source_file': db['source_file'].astype(str),
                })
                good = ((out['B_MG'] > 0) & (out['B_MG'] <= b_max_mg)
                        & (out['wavelength_A'] >= wavelength_min_A)
                        & (out['wavelength_A'] <= wavelength_max_A))
                return out.loc[good].sort_values(
                    ['series', 'component', 'B_MG']).reset_index(drop=True)
            except Exception:
                continue

    if interpolated:
        pattern = os.path.join(template_dir, 'interpolated',
                               '*_cubic_interpolated.csv')
    else:
        pattern = os.path.join(template_dir, 'H*.csv')

    rows = []
    for path in sorted(glob.glob(pattern)):
        parsed = _parse_component_name(path)
        if parsed is None:
            continue
        series, component = parsed
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if interpolated:
            if not {'Magnetic_Field', 'Wavelength_Cubic'} <= set(df.columns):
                continue
            b = pd.to_numeric(df['Magnetic_Field'], errors='coerce').to_numpy(float)
            wl = pd.to_numeric(df['Wavelength_Cubic'], errors='coerce').to_numpy(float)
        else:
            if not {'MagneticField(MG)', 'Wavelength'} <= set(df.columns):
                continue
            b = pd.to_numeric(df['MagneticField(MG)'], errors='coerce').to_numpy(float)
            wl = pd.to_numeric(df['Wavelength'], errors='coerce').to_numpy(float)
        good = (np.isfinite(b) & np.isfinite(wl)
                & (b > 0) & (b <= b_max_mg)
                & (wl >= wavelength_min_A) & (wl <= wavelength_max_A))
        for bi, wi in zip(b[good], wl[good]):
            rows.append({
                'series': series,
                'component': int(component),
                'B_MG': float(bi),
                'wavelength_A': float(wi),
                'source_file': os.path.basename(path),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(['series', 'component', 'B_MG']).reset_index(drop=True)


def _prepare_spectrum(wave, flux, err=None, continuum_window=None):
    wave = coerce_wavelength_to_angstrom(wave)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float) if err is not None else None
    good = np.isfinite(wave) & np.isfinite(flux) & (flux != 0)
    if err is not None and err.shape == wave.shape:
        good &= np.isfinite(err) & (err > 0)
    order = np.argsort(wave[good])
    wave = wave[good][order]
    flux = flux[good][order]
    if err is not None and err.shape == good.shape:
        err = err[good][order]
    else:
        err = None
    if len(wave) < 50:
        return None

    n = len(wave)
    if continuum_window is None:
        continuum_window = 101
    continuum_window = int(max(31, continuum_window))
    if continuum_window % 2 == 0:
        continuum_window += 1
    continuum = median_filter(flux, size=continuum_window, mode='nearest')
    positive = flux[np.isfinite(flux) & (flux > 0)]
    floor = np.nanpercentile(positive, 5) * 0.1 if positive.size else 1.0
    continuum = np.where(np.isfinite(continuum) & (np.abs(continuum) > 0),
                         continuum, floor)
    continuum = np.where(np.abs(continuum) < abs(floor), floor, continuum)
    norm = flux / continuum
    if err is not None:
        norm_err = np.abs(err / continuum)
    else:
        resid = norm - median_filter(norm, size=max(11, continuum_window // 7),
                                     mode='nearest')
        sigma = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.02
        norm_err = np.full_like(norm, sigma)
    return wave, flux, err, continuum, norm, norm_err


def _template_baseline_normalize(wave, flux, err=None, baseline_mode='template',
                                 wd_model_grid='auto', spectral_type='auto',
                                 continuum_window=None):
    """
    Fit an ordinary WD template first, then search Zeeman dips in data/model.

    This suppresses broad DA/DB/DC continuum and normal Balmer cores before the
    magnetic-field scoring step.  If no usable template fit is available, the
    median-filter continuum from ``_prepare_spectrum`` is used as a fallback.
    """
    prepared = _prepare_spectrum(wave, flux, err, continuum_window=continuum_window)
    if prepared is None:
        return None
    wave, flux, err, continuum, norm, norm_err = prepared
    info = {
        'baseline_method': 'median_continuum',
        'wd_spectral_type': '',
        'wd_template_grid': '',
        'wd_template_teff': np.nan,
        'wd_template_logg': np.nan,
        'wd_template_chi2_red': np.nan,
    }
    model_flux = None

    if str(baseline_mode or 'template').lower() not in ('template', 'auto'):
        return wave, flux, err, continuum, norm, norm_err, model_flux, info

    try:
        from .wd_fitting import classify_wd_type, fit_single_wd
        cls = classify_wd_type(wave, flux, err)
        cls_type = str(cls.get('spectral_type', 'DA') or 'DA').upper()
        info['wd_spectral_type'] = cls_type
        if spectral_type and str(spectral_type).lower() != 'auto':
            cls_type = str(spectral_type).upper()

        if cls_type.startswith('DB'):
            candidates = ['DB', 'DA']
        elif cls_type.startswith('DC'):
            candidates = ['DA', 'DB']
        else:
            candidates = ['DA', 'DB']

        best_fit = None
        best_kind = ''
        for kind in candidates:
            fit = fit_single_wd(
                wave, flux, err, line_only=False,
                model_grid=wd_model_grid, spectral_type=kind)
            if fit is None:
                continue
            chi2 = fit.get('chi2_red', np.inf)
            if best_fit is None or chi2 < best_fit.get('chi2_red', np.inf):
                best_fit = fit
                best_kind = kind

        if best_fit is None:
            return wave, flux, err, continuum, norm, norm_err, model_flux, info

        model_flux = np.interp(
            wave,
            np.asarray(best_fit['best_model_wave'], dtype=float),
            np.asarray(best_fit['best_model_flux'], dtype=float),
            left=np.nan, right=np.nan)
        good_model = np.isfinite(model_flux) & (np.abs(model_flux) > 0)
        if np.sum(good_model) < max(50, len(wave) // 3):
            return wave, flux, err, continuum, norm, norm_err, None, info

        ratio = np.full_like(flux, np.nan, dtype=float)
        ratio[good_model] = flux[good_model] / model_flux[good_model]
        ratio_cont_window = max(31, min(1001, len(ratio) // 45))
        if ratio_cont_window % 2 == 0:
            ratio_cont_window += 1
        ratio_fill = ratio.copy()
        fill = np.nanmedian(ratio_fill[np.isfinite(ratio_fill)])
        if not np.isfinite(fill) or fill == 0:
            fill = 1.0
        ratio_fill[~np.isfinite(ratio_fill)] = fill
        ratio_cont = median_filter(ratio_fill, size=ratio_cont_window, mode='nearest')
        ratio_cont = np.where(np.isfinite(ratio_cont) & (np.abs(ratio_cont) > 0),
                              ratio_cont, fill)
        template_norm = ratio_fill / ratio_cont
        if err is not None:
            template_err = np.full_like(template_norm, np.nan, dtype=float)
            template_err[good_model] = np.abs(
                err[good_model] / model_flux[good_model] / ratio_cont[good_model])
            noise_floor = _robust_noise(template_norm - 1.0, fallback=0.01)
            template_err = np.where(np.isfinite(template_err) & (template_err > 0),
                                    np.maximum(template_err, noise_floor * 0.5),
                                    noise_floor)
        else:
            noise = _robust_noise(template_norm - 1.0, fallback=0.015)
            template_err = np.full_like(template_norm, noise)

        info.update({
            'baseline_method': 'wd_template_ratio',
            'wd_spectral_type': best_kind,
            'wd_template_grid': best_fit.get('model_grid', ''),
            'wd_template_teff': best_fit.get('teff', np.nan),
            'wd_template_logg': best_fit.get('logg', np.nan),
            'wd_template_chi2_red': best_fit.get('chi2_red', np.nan),
        })
        return wave, flux, err, model_flux, template_norm, template_err, model_flux, info
    except Exception as exc:
        info['baseline_error'] = str(exc)
        return wave, flux, err, continuum, norm, norm_err, model_flux, info


def _robust_noise(values, fallback=0.02):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return fallback
    med = np.nanmedian(values)
    sigma = 1.4826 * np.nanmedian(np.abs(values - med))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(values)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = fallback
    return float(max(sigma, fallback * 0.25))


def _make_emission_mask(wave, norm, norm_err, emission_sigma=3.0,
                        emission_min_height=0.025):
    err_med = np.nanmedian(norm_err[np.isfinite(norm_err)])
    if not np.isfinite(err_med) or err_med <= 0:
        err_med = _robust_noise(norm - 1.0, fallback=0.02)
    thresh = max(emission_min_height, emission_sigma * err_med)
    return np.isfinite(norm) & ((norm - 1.0) > thresh)


def _near_masked_wavelength(wave, mask, center, half_width):
    lo = np.searchsorted(wave, center - half_width)
    hi = np.searchsorted(wave, center + half_width)
    return bool(hi > lo and np.any(mask[lo:hi]))


def _known_emission_name(center, avoid_A):
    for name, wl in KNOWN_EMISSION_LINES.items():
        if abs(center - wl) <= avoid_A:
            return name
    return ''


def _known_artifact_name(center, half_width_A=0.0):
    lo = center - max(float(half_width_A), 0.0)
    hi = center + max(float(half_width_A), 0.0)
    for name, (band_lo, band_hi) in KNOWN_ARTIFACT_BANDS.items():
        if hi >= band_lo and lo <= band_hi:
            return name
    return ''


def _known_absorption_core_name(center, avoid_A, rv_kms=0.0):
    for name, wl in STANDARD_ABSORPTION_CORES.items():
        shifted = wl * (1.0 + rv_kms / C_KMS)
        if abs(center - shifted) <= avoid_A:
            return name
    return ''


def _component_wavelengths(templates, B_mg, series=None):
    if isinstance(templates, list):
        rows = []
        series_set = set(series or [])
        for curve in templates:
            if series and curve['series'] not in series_set:
                continue
            b = curve['B_MG']
            if B_mg < b[0] or B_mg > b[-1]:
                continue
            rows.append({
                'series': curve['series'],
                'component': int(curve['component']),
                'B_MG': float(B_mg),
                'wavelength_A': float(np.interp(B_mg, b, curve['wavelength_A'])),
            })
        return rows

    if templates.empty:
        return []
    rows = []
    for (ser, comp), df in templates.groupby(['series', 'component']):
        if series and ser not in series:
            continue
        b = df['B_MG'].to_numpy(float)
        wl = df['wavelength_A'].to_numpy(float)
        order = np.argsort(b)
        b = b[order]
        wl = wl[order]
        if len(b) < 2 or B_mg < b.min() or B_mg > b.max():
            continue
        pred = float(np.interp(B_mg, b, wl))
        rows.append({
            'series': ser,
            'component': int(comp),
            'B_MG': float(B_mg),
            'wavelength_A': pred,
        })
    return rows


def _core_side_pit_summary(details, rv_kms=0.0, core_guard_A=25.0):
    """
    Mark detected pits that sit next to a normal Balmer absorption core.

    Low-field Zeeman splitting can appear as extra troughs close to the broad
    Balmer absorption line.  The central ordinary core is still guarded
    elsewhere; this helper records nearby side troughs so they can support a
    low-field solution without turning a single normal DA core into a detection.
    """
    by_series = {}
    rv_factor = 1.0 + float(rv_kms) / C_KMS
    if not np.isfinite(rv_factor) or rv_factor <= 0:
        rv_factor = 1.0
    for detail in details:
        ser = detail.get('series')
        rest = BALMER_REST.get(ser)
        if rest is None:
            detail['balmer_core_offset_A'] = np.nan
            detail['core_side_pit'] = False
            continue
        obs = detail.get('detected_wavelength_A', np.nan)
        if not np.isfinite(obs):
            obs = detail.get('predicted_wavelength_A', np.nan)
        if not np.isfinite(obs):
            detail['balmer_core_offset_A'] = np.nan
            detail['core_side_pit'] = False
            continue
        offset = float(obs) / rv_factor - rest
        side_limit = CORE_SIDE_MAX_A.get(ser, 300.0)
        minor = bool(detail.get('minor_absorption_support'))
        embedded_minor = (
            minor
            and str(detail.get('evidence_class', '')) == 'embedded_absorption_notch')
        side_evidence = bool(detail.get('detected')) or (
            minor and not embedded_minor)
        is_side = (
            side_evidence
            and abs(offset) > float(core_guard_A)
            and abs(offset) <= side_limit)
        detail['balmer_core_offset_A'] = float(offset)
        detail['core_side_pit'] = bool(is_side)
        if not is_side:
            continue
        bucket = by_series.setdefault(ser, {'left': 0, 'right': 0, 'n': 0})
        bucket['n'] += 1
        if offset < 0:
            bucket['left'] += 1
        else:
            bucket['right'] += 1

    support_series = []
    paired_series = []
    n_side = 0
    for ser, bucket in by_series.items():
        n_side += int(bucket['n'])
        if bucket['n'] >= 2 or (bucket['left'] > 0 and bucket['right'] > 0):
            support_series.append(ser)
        if bucket['left'] > 0 and bucket['right'] > 0:
            paired_series.append(ser)
    support_series = sorted(support_series)
    paired_series = sorted(paired_series)
    return {
        'n_core_side_pits': int(n_side),
        'n_core_side_pit_series': int(len(support_series)),
        'core_side_pit_series': ';'.join(support_series),
        'n_core_side_pair_series': int(len(paired_series)),
        'core_side_pair_series': ';'.join(paired_series),
    }


def _compile_template_curves(templates, series=None):
    """Precompile component curves for fast repeated B-grid scoring."""
    curves = []
    series_set = set(series or [])
    for (ser, comp), df in templates.groupby(['series', 'component']):
        if series and ser not in series_set:
            continue
        b = df['B_MG'].to_numpy(float)
        wl = df['wavelength_A'].to_numpy(float)
        good = np.isfinite(b) & np.isfinite(wl)
        if np.sum(good) < 2:
            continue
        order = np.argsort(b[good])
        curves.append({
            'series': ser,
            'component': int(comp),
            'B_MG': b[good][order],
            'wavelength_A': wl[good][order],
        })
    return curves


def _score_one_field(wave, norm, norm_err, emission_mask, templates,
                     B_mg, rv_kms=0.0, series=None,
                     search_half_width_A=8.0,
                     min_depth=0.008, min_snr=1.5,
                     min_trough_width_A=6.0,
                     max_component_offset_A=None,
                     emission_avoid_A=10.0,
                     absorption_core_avoid_A=25.0,
                     minor_absorption_weight=0.28):
    components = _component_wavelengths(templates, B_mg, series=series)
    details = []
    score = 0.0
    n_detected = 0
    n_usable = 0
    n_minor_support = 0
    minor_weight_sum = 0.0

    for comp in components:
        pred = comp['wavelength_A'] * (1.0 + rv_kms / C_KMS)
        detail = dict(comp)
        detail['rv_kms'] = float(rv_kms)
        detail['predicted_wavelength_A'] = float(pred)
        detail['usable'] = False
        detail['detected'] = False
        detail['minor_absorption_support'] = False
        detail['evidence_class'] = 'none'
        detail['evidence_weight'] = 0.0
        detail['skip_reason'] = ''

        if pred < wave[0] + search_half_width_A or pred > wave[-1] - search_half_width_A:
            detail['skip_reason'] = 'outside_spectrum'
            details.append(detail)
            continue
        line_avoid_A = max(
            float(emission_avoid_A),
            min(0.6 * float(search_half_width_A), 20.0))
        known = _known_emission_name(pred, line_avoid_A)
        contamination_reasons = []
        if known:
            contamination_reasons.append(f'known_emission:{known}')
        core = _known_absorption_core_name(pred, absorption_core_avoid_A,
                                           rv_kms=rv_kms)
        if core:
            contamination_reasons.append(f'standard_absorption_core:{core}')
        if _near_masked_wavelength(wave, emission_mask, pred, emission_avoid_A):
            contamination_reasons.append('observed_emission_peak')
        artifact = _known_artifact_name(
            pred, half_width_A=min(search_half_width_A, 25.0))
        if artifact:
            detail['skip_reason'] = f'observed_artifact_band:{artifact}'
            details.append(detail)
            continue

        lo = np.searchsorted(wave, pred - search_half_width_A)
        hi = np.searchsorted(wave, pred + search_half_width_A)
        if hi - lo < 3:
            detail['skip_reason'] = 'too_few_pixels'
            details.append(detail)
            continue

        local_norm = norm[lo:hi]
        local_err = norm_err[lo:hi]
        local_wave = wave[lo:hi]
        finite = np.isfinite(local_norm) & np.isfinite(local_wave)
        if np.sum(finite) < 3:
            detail['skip_reason'] = 'nonfinite_window'
            details.append(detail)
            continue

        local_norm = local_norm[finite]
        local_err = local_err[finite]
        local_wave = local_wave[finite]
        local_step = np.nanmedian(np.diff(local_wave))
        if not np.isfinite(local_step) or local_step <= 0:
            local_step = 1.0
        smooth_size = int(np.ceil(4.0 / max(local_step, 1e-3)))
        smooth_size = max(3, min(11, smooth_size))
        if smooth_size % 2 == 0:
            smooth_size += 1
        if smooth_size >= len(local_norm):
            smooth_size = len(local_norm) - 1 if len(local_norm) % 2 == 0 else len(local_norm)
        if smooth_size >= 3:
            smooth_norm = median_filter(local_norm, size=smooth_size, mode='nearest')
        else:
            smooth_norm = local_norm
        noise = np.nanmedian(local_err[np.isfinite(local_err) & (local_err > 0)])
        if not np.isfinite(noise) or noise <= 0:
            noise = _robust_noise(local_norm - np.nanmedian(local_norm), fallback=0.02)
        edge_n = max(2, min(len(local_norm) // 4, len(local_norm) - 1))
        edge_level = np.nanmedian(
            np.r_[smooth_norm[:edge_n], smooth_norm[-edge_n:]])
        if not np.isfinite(edge_level):
            edge_level = 1.0
        imin = int(np.nanargmin(smooth_norm))
        depth = float(max(0.0, edge_level - smooth_norm[imin]))
        snr = depth / max(noise, 1e-6)
        local_baseline = float(edge_level)
        trough_level = edge_level - 0.45 * depth
        left = imin
        right = imin
        while left > 0 and smooth_norm[left - 1] <= trough_level:
            left -= 1
        while right < len(local_norm) - 1 and smooth_norm[right + 1] <= trough_level:
            right += 1
        trough_width_A = float(local_wave[right] - local_wave[left])
        base_level = edge_level - 0.20 * depth
        base_left = imin
        base_right = imin
        while base_left > 0 and smooth_norm[base_left - 1] <= base_level:
            base_left -= 1
        while (base_right < len(local_norm) - 1
               and smooth_norm[base_right + 1] <= base_level):
            base_right += 1
        trough_base_width_A = float(local_wave[base_right] - local_wave[base_left])
        embedded_baseline = float(max(
            edge_level,
            np.nanpercentile(smooth_norm[np.isfinite(smooth_norm)], 72)
            if np.any(np.isfinite(smooth_norm)) else edge_level))
        embedded_depth = float(max(0.0, embedded_baseline - smooth_norm[imin]))
        embedded_snr = embedded_depth / max(noise, 1e-6)
        embedded_level = embedded_baseline - 0.45 * embedded_depth
        emb_left = imin
        emb_right = imin
        while emb_left > 0 and smooth_norm[emb_left - 1] <= embedded_level:
            emb_left -= 1
        while (emb_right < len(local_norm) - 1
               and smooth_norm[emb_right + 1] <= embedded_level):
            emb_right += 1
        embedded_width_A = float(local_wave[emb_right] - local_wave[emb_left])
        pred_idx = int(np.nanargmin(np.abs(local_wave - pred)))
        anchored_half_A = max(1.0, min(3.0, 1.5 * float(local_step)))
        anchored_mask = np.abs(local_wave - pred) <= anchored_half_A
        if not np.any(anchored_mask):
            anchored_mask = np.zeros(len(local_wave), dtype=bool)
            anchored_mask[pred_idx] = True
        anchored_level = float(np.nanmedian(smooth_norm[anchored_mask]))
        if not np.isfinite(anchored_level):
            anchored_level = float(smooth_norm[pred_idx])
        anchored_depth = float(max(0.0, embedded_baseline - anchored_level))
        anchored_snr = anchored_depth / max(noise, 1e-6)
        anchored_left_A = float(max(pred - anchored_half_A, local_wave[0]))
        anchored_right_A = float(min(pred + anchored_half_A, local_wave[-1]))
        anchored_width_A = float(max(0.0, anchored_right_A - anchored_left_A))
        broad_width_min_A = max(float(min_trough_width_A), 2.5 * float(local_step))
        minor_width_min_A = max(
            1.8 * float(local_step),
            min(4.0, 0.38 * broad_width_min_A))
        embedded_notch_min_width_A = max(
            0.50 * float(local_step),
            min(1.5, 0.18 * broad_width_min_A))
        if max_component_offset_A is None:
            offset_limit_A = max(4.0, 0.75 * float(search_half_width_A))
        else:
            offset_limit_A = float(max_component_offset_A)
        minor_offset_limit_A = min(
            float(search_half_width_A),
            offset_limit_A + max(3.0, 0.45 * broad_width_min_A))
        embedded_notch_offset_limit_A = min(
            float(search_half_width_A),
            max(minor_offset_limit_A, 0.75 * float(search_half_width_A)))
        left_recovery = float(edge_level - smooth_norm[imin])
        right_recovery = left_recovery
        if edge_n > 0:
            left_recovery = float(np.nanmedian(smooth_norm[:edge_n])
                                  - smooth_norm[imin])
            right_recovery = float(np.nanmedian(smooth_norm[-edge_n:])
                                   - smooth_norm[imin])
        recovery = max(left_recovery, right_recovery)
        edge_min = imin <= max(1, int(0.08 * len(local_norm)))
        edge_min = edge_min or imin >= len(local_norm) - max(
            2, int(0.08 * len(local_norm))) - 1
        anchored_edge = pred_idx <= max(1, int(0.08 * len(local_norm)))
        anchored_edge = anchored_edge or pred_idx >= len(local_norm) - max(
            2, int(0.08 * len(local_norm))) - 1
        base_margin_A = max(2.0 * float(local_step),
                            min(12.0, 0.20 * max(trough_base_width_A, 0.0)))
        pred_in_broad_base = (
            local_wave[base_left] - base_margin_A
            <= pred <=
            local_wave[base_right] + base_margin_A)
        support_width_A = max(trough_width_A, trough_base_width_A)

        detail.update({
            'usable': True,
            'detected_wavelength_A': float(local_wave[imin]),
            'depth': depth,
            'snr': float(snr),
            'local_noise': float(noise),
            'local_baseline': float(local_baseline),
            'smoothing_window_pixels': int(smooth_size),
            'trough_width_A': float(trough_width_A),
            'trough_base_left_A': float(local_wave[base_left]),
            'trough_base_right_A': float(local_wave[base_right]),
            'trough_base_width_A': float(trough_base_width_A),
            'support_width_A': float(support_width_A),
            'embedded_notch_baseline': float(embedded_baseline),
            'embedded_notch_depth': float(embedded_depth),
            'embedded_notch_snr': float(embedded_snr),
            'embedded_notch_width_A': float(embedded_width_A),
            'embedded_notch_left_A': float(local_wave[emb_left]),
            'embedded_notch_right_A': float(local_wave[emb_right]),
            'anchored_notch_level': float(anchored_level),
            'anchored_notch_depth': float(anchored_depth),
            'anchored_notch_snr': float(anchored_snr),
            'anchored_notch_half_width_A': float(anchored_half_A),
            'anchored_notch_width_A': float(anchored_width_A),
            'anchored_notch_left_A': float(anchored_left_A),
            'anchored_notch_right_A': float(anchored_right_A),
            'anchored_notch_at_window_edge': bool(anchored_edge),
            'evidence_depth': depth,
            'evidence_snr': float(snr),
            'predicted_inside_broad_trough': bool(pred_in_broad_base),
            'min_trough_width_A': float(broad_width_min_A),
            'minor_absorption_min_width_A': float(minor_width_min_A),
            'embedded_notch_min_width_A': float(embedded_notch_min_width_A),
            'max_component_offset_A': float(offset_limit_A),
            'minor_absorption_max_offset_A': float(minor_offset_limit_A),
            'embedded_notch_max_offset_A': float(embedded_notch_offset_limit_A),
            'broad_trough_ok': bool(support_width_A >= broad_width_min_A),
            'minor_width_ok': bool(support_width_A >= minor_width_min_A),
            'contamination_reasons': ';'.join(contamination_reasons),
            'left_recovery': float(left_recovery),
            'right_recovery': float(right_recovery),
            'delta_detected_minus_predicted_A': float(local_wave[imin] - pred),
        })
        abs_offset_A = abs(float(local_wave[imin] - pred))
        recovery_ok = recovery >= 0.20 * max(depth, 1e-6)
        morphology_ok = (
            not edge_min
            and support_width_A >= broad_width_min_A
            and recovery >= 0.25 * max(depth, 1e-6)
            and (abs_offset_A <= offset_limit_A or pred_in_broad_base)
        )
        minor_absorption_ok = (
            not contamination_reasons
            and not anchored_edge
            and anchored_depth >= min_depth
            and anchored_snr >= min_snr
            and anchored_width_A >= embedded_notch_min_width_A
        )
        embedded_notch_ok = (
            bool(contamination_reasons)
            and not anchored_edge
            and anchored_depth >= min_depth
            and anchored_snr >= min_snr
            and anchored_width_A >= embedded_notch_min_width_A
        )
        detail['pit_shape_ok'] = bool(morphology_ok)
        detail['minor_absorption_shape_ok'] = bool(minor_absorption_ok)
        detail['embedded_notch_shape_ok'] = bool(embedded_notch_ok)
        if not contamination_reasons and depth >= min_depth and snr >= min_snr and morphology_ok:
            detail['detected'] = True
            detail['evidence_class'] = 'broad_trough'
            detail['evidence_weight'] = 1.0
        elif embedded_notch_ok:
            detail['minor_absorption_support'] = True
            detail['evidence_class'] = 'embedded_absorption_notch'
            detail['evidence_weight'] = float(0.35 * minor_absorption_weight)
            detail['evidence_depth'] = float(anchored_depth)
            detail['evidence_snr'] = float(anchored_snr)
            detail['detected_wavelength_A'] = float(pred)
            detail['delta_detected_minus_predicted_A'] = 0.0
            detail['skip_reason'] = (
                'embedded_absorption_notch_low_weight:'
                + ';'.join(contamination_reasons))
        elif minor_absorption_ok:
            detail['minor_absorption_support'] = True
            detail['evidence_class'] = 'minor_absorption'
            detail['evidence_weight'] = float(minor_absorption_weight)
            detail['evidence_depth'] = float(anchored_depth)
            detail['evidence_snr'] = float(anchored_snr)
            detail['detected_wavelength_A'] = float(pred)
            detail['delta_detected_minus_predicted_A'] = 0.0
            detail['skip_reason'] = 'minor_absorption_low_weight'
        elif contamination_reasons:
            detail['skip_reason'] = ';'.join(contamination_reasons)
        elif depth >= min_depth and snr >= min_snr and support_width_A < broad_width_min_A:
            detail['skip_reason'] = 'narrow_line_not_zeeman_band'
        elif (depth >= min_depth and snr >= min_snr
              and abs_offset_A > offset_limit_A and not pred_in_broad_base):
            detail['skip_reason'] = 'wavelength_offset_too_large'
        n_usable += 1
        details.append(detail)

    detected_idx = [
        i for i, detail in enumerate(details)
        if (detail.get('detected') or detail.get('minor_absorption_support'))
        and np.isfinite(detail.get('detected_wavelength_A', np.nan))
    ]
    if detected_idx:
        duplicate_tol_A = max(0.5, min(3.0, 0.5 * float(search_half_width_A)))
        detected_idx = sorted(
            detected_idx, key=lambda i: details[i]['detected_wavelength_A'])
        groups = []
        for idx in detected_idx:
            wl = float(details[idx]['detected_wavelength_A'])
            if not groups or abs(wl - groups[-1]['last_wavelength_A']) > duplicate_tol_A:
                groups.append({'indices': [idx], 'last_wavelength_A': wl})
            else:
                groups[-1]['indices'].append(idx)
                groups[-1]['last_wavelength_A'] = wl
        for group in groups:
            indices = group['indices']
            if len(indices) <= 1:
                continue
            keep = max(
                indices,
                key=lambda i: (
                    1 if details[i].get('detected') else 0,
                    float(details[i].get('evidence_weight', 0.0) or 0.0),
                    float(details[i].get('snr', 0.0) or 0.0),
                    -abs(float(details[i].get('detected_wavelength_A', np.nan))
                         - float(details[i].get('predicted_wavelength_A', np.nan)))))
            keep_label = _series_component_label(
                details[keep].get('series'), details[keep].get('component'))
            for idx in indices:
                if idx == keep:
                    continue
                details[idx]['detected'] = False
                details[idx]['minor_absorption_support'] = False
                details[idx]['evidence_class'] = 'none'
                details[idx]['evidence_weight'] = 0.0
                details[idx]['duplicate_of'] = keep_label
                details[idx]['skip_reason'] = (
                    f'duplicate_detected_trough:{keep_label}')

    core_side = _core_side_pit_summary(
        details, rv_kms=rv_kms, core_guard_A=absorption_core_avoid_A)

    for detail in details:
        if not detail.get('usable'):
            continue
        depth = float(detail.get('depth', 0.0) or 0.0)
        snr = float(detail.get('snr', 0.0) or 0.0)
        if detail.get('detected'):
            n_detected += 1
            component_score = min(snr, 8.0) + 0.75
            if detail.get('pit_shape_ok'):
                component_score += 0.35
            else:
                component_score *= 0.62
            offset = abs(float(
                detail.get('delta_detected_minus_predicted_A', np.nan)))
            if np.isfinite(offset):
                component_score -= 0.06 * min(
                    offset, float(search_half_width_A))
            if float(B_mg) >= 70.0 and detail.get('core_side_pit'):
                component_score *= 0.82
            score += max(component_score, 0.05)
        elif detail.get('minor_absorption_support'):
            n_minor_support += 1
            evidence_snr = float(
                detail.get('evidence_snr',
                           detail.get('anchored_notch_snr',
                                      detail.get('snr', 0.0))) or 0.0)
            offset = abs(float(
                detail.get('delta_detected_minus_predicted_A', np.nan)))
            offset_penalty = 0.0
            if np.isfinite(offset):
                offset_penalty = 0.035 * min(
                    offset, float(search_half_width_A))
            evidence_weight = float(
                detail.get('evidence_weight', minor_absorption_weight)
                or minor_absorption_weight)
            minor_weight_sum += evidence_weight
            component_score = (
                evidence_weight * (min(evidence_snr, 5.0) + 0.35)
                - offset_penalty)
            if detail.get('predicted_inside_broad_trough'):
                component_score += 0.20
            if detail.get('core_side_pit'):
                component_score += 0.12
            score += max(component_score, 0.03)
        elif depth > 0 and 'duplicate_detected_trough' not in str(detail.get('skip_reason', '')):
            score += 0.10 * min(snr, 2.0)

    score -= 0.05 * max(n_usable - n_detected, 0)
    if float(B_mg) <= 35.0:
        score += 0.70 * core_side['n_core_side_pits']
        score += 0.90 * core_side['n_core_side_pit_series']
        score += 2.20 * core_side['n_core_side_pair_series']
        if n_detected >= 3 and core_side['n_core_side_pair_series'] == 0:
            score -= 3.0
    elif n_detected:
        core_fraction = core_side['n_core_side_pits'] / max(n_detected, 1)
        if core_fraction > 0.55:
            score -= 1.5 * (core_fraction - 0.55) * n_detected
    if np.isfinite(rv_kms):
        score -= 0.0035 * abs(float(rv_kms))
    return {
        'B_MG': float(B_mg),
        'rv_kms': float(rv_kms),
        'score': float(score),
        'n_detected': int(n_detected),
        'n_minor_support': int(n_minor_support),
        'weighted_detected_components': float(
            n_detected + minor_weight_sum),
        'n_usable': int(n_usable),
        **core_side,
        'details': details,
    }


def _weighted_quantile(values, weights, qs):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.full(len(qs), np.nan)
    values = values[good]
    weights = weights[good]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return np.interp(qs, cdf, values)


def _field_solution_peaks(score_df, min_separation_mg=8.0, max_peaks=8):
    """Return separated local maxima from a B-score grid for review."""
    if score_df is None or score_df.empty:
        return pd.DataFrame()
    required = {'B_MG', 'best_score'}
    if not required <= set(score_df.columns):
        return pd.DataFrame()
    df = score_df.copy()
    df['B_MG'] = pd.to_numeric(df['B_MG'], errors='coerce')
    df['best_score'] = pd.to_numeric(df['best_score'], errors='coerce')
    df = df[np.isfinite(df['B_MG']) & np.isfinite(df['best_score'])].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('B_MG').reset_index(drop=True)
    scores = df['best_score'].to_numpy(float)
    is_peak = np.ones(len(df), dtype=bool)
    if len(df) > 2:
        is_peak[:] = False
        is_peak[0] = scores[0] >= scores[1]
        is_peak[-1] = scores[-1] >= scores[-2]
        is_peak[1:-1] = (
            (scores[1:-1] >= scores[:-2])
            & (scores[1:-1] >= scores[2:])
        )
    candidates = df.loc[is_peak].copy()
    if candidates.empty:
        candidates = df.nlargest(min(max_peaks, len(df)), 'best_score').copy()
    candidates = candidates.sort_values('best_score', ascending=False)
    selected = []
    for _, row in candidates.iterrows():
        b = float(row['B_MG'])
        if any(abs(b - float(prev['B_MG'])) < float(min_separation_mg)
               for prev in selected):
            continue
        selected.append(row.to_dict())
        if len(selected) >= int(max_peaks):
            break
    if not selected:
        return pd.DataFrame()
    out = pd.DataFrame(selected)
    best_score = float(out['best_score'].max())
    out['delta_score_from_best'] = best_score - pd.to_numeric(
        out['best_score'], errors='coerce')
    out['solution_rank'] = np.arange(1, len(out) + 1)
    cols = ['solution_rank', 'B_MG', 'best_score',
            'delta_score_from_best', 'best_rv_kms',
            'n_detected', 'n_minor_support',
            'weighted_detected_components', 'n_usable']
    cols = [c for c in cols if c in out.columns]
    return out[cols].reset_index(drop=True)


def _field_regime(b_mg):
    if not np.isfinite(b_mg) or b_mg <= 0:
        return 'unknown'
    if b_mg < 20.0:
        return 'weak_field_split'
    if b_mg < 100.0:
        return 'intermediate_field_split'
    return 'strong_field_split'


def _interval_errors_around_best(best_b, q16, q84):
    if not (np.isfinite(best_b) and np.isfinite(q16) and np.isfinite(q84)):
        return np.nan, np.nan, np.nan, np.nan, False
    lo = float(min(q16, q84))
    hi = float(max(q16, q84))
    err_minus = max(0.0, float(best_b) - lo)
    err_plus = max(0.0, hi - float(best_b))
    contains = bool(lo <= float(best_b) <= hi)
    return lo, hi, err_minus, err_plus, contains


def _component_summary(component_df):
    if component_df is None or component_df.empty:
        return 0, ''
    detected = component_df[component_df.get('detected', False) == True]
    if detected.empty:
        return 0, ''
    series = sorted(set(str(x) for x in detected['series'] if str(x)))
    labels = []
    for _, row in detected.iterrows():
        labels.append(_series_component_label(row.get('series'), row.get('component')))
    return len(series), ';'.join(labels)


def _review_magnetic_detection(component_df, quality, rel_err, rv_edge=False,
                               b_mg=np.nan, search_half_width_A=8.0,
                               interval_contains_best=True):
    """Reviewer-facing sanity check for automated Zeeman candidates."""
    if component_df is None or component_df.empty:
        return {
            'magnetic_review_status': 'no_detection',
            'magnetic_review_reasons': 'no usable component table',
            'magnetic_high_snr_components': 0,
            'magnetic_max_component_snr': np.nan,
            'magnetic_median_abs_component_offset_A': np.nan,
            'magnetic_core_side_fraction': np.nan,
            'magnetic_pit_shape_fraction': np.nan,
            'magnetic_broad_trough_fraction': np.nan,
            'magnetic_broad_components': 0,
            'magnetic_minor_absorption_components': 0,
            'magnetic_weighted_components': 0.0,
        }

    detected_mask = component_df.get('detected', False) == True
    detected = component_df[detected_mask].copy()
    minor_mask = component_df.get('minor_absorption_support', False) == True
    minor = component_df[minor_mask].copy()
    minor_count = int(len(minor))
    if minor_count and 'evidence_weight' in minor.columns:
        minor_weight_sum = float(np.nansum(
            pd.to_numeric(minor.get('evidence_weight'), errors='coerce')))
    else:
        minor_weight_sum = 0.0
    weighted_components = float(
        len(detected) + minor_weight_sum)
    if detected.empty:
        reason = 'no detected broad components'
        if minor_count:
            reason = (
                f'no detected broad components; {minor_count} low-weight '
                'minor absorption supports')
        return {
            'magnetic_review_status': 'no_detection',
            'magnetic_review_reasons': reason,
            'magnetic_high_snr_components': 0,
            'magnetic_max_component_snr': np.nan,
            'magnetic_median_abs_component_offset_A': np.nan,
            'magnetic_core_side_fraction': np.nan,
            'magnetic_pit_shape_fraction': np.nan,
            'magnetic_broad_trough_fraction': np.nan,
            'magnetic_broad_components': 0,
            'magnetic_minor_absorption_components': minor_count,
            'magnetic_weighted_components': weighted_components,
        }

    snr = pd.to_numeric(detected.get('snr'), errors='coerce')
    offsets = pd.to_numeric(
        detected.get('delta_detected_minus_predicted_A'), errors='coerce')
    high_snr = int(np.sum(np.isfinite(snr) & (snr >= 4.5)))
    max_snr = float(np.nanmax(snr)) if np.any(np.isfinite(snr)) else np.nan
    med_abs_offset = (
        float(np.nanmedian(np.abs(offsets)))
        if np.any(np.isfinite(offsets)) else np.nan)
    pit_ok = detected.get('pit_shape_ok', False) == True
    core_side = detected.get('core_side_pit', False) == True
    broad_ok = detected.get('broad_trough_ok', False) == True
    pit_frac = float(np.nanmean(pit_ok)) if len(detected) else np.nan
    core_frac = float(np.nanmean(core_side)) if len(detected) else np.nan
    broad_frac = float(np.nanmean(broad_ok)) if len(detected) else np.nan
    series_detected = set(str(x) for x in detected.get('series', []) if str(x))

    reasons = []
    if len(detected) < 3:
        reasons.append('fewer than 3 detected components')
    if len(series_detected) < 2:
        reasons.append('only one Balmer series detected')
    if not {'Halpha', 'Hbeta'}.issubset(series_detected):
        reasons.append('missing Halpha/Hbeta joint support')
    if high_snr < 2:
        reasons.append('fewer than two components with SNR >= 4.5')
    offset_limit = max(8.0, float(search_half_width_A))
    if np.isfinite(med_abs_offset) and med_abs_offset > offset_limit:
        reasons.append('component wavelength offsets are large')
    if np.isfinite(pit_frac) and pit_frac < 0.50:
        reasons.append('pit shapes are not consistently recovered')
    if np.isfinite(broad_frac) and broad_frac < 0.75:
        reasons.append('most features are too narrow for Zeeman bands')
    if np.isfinite(core_frac) and core_frac > 0.60 and (
            not np.isfinite(b_mg) or float(b_mg) < 70.0):
        reasons.append('mostly broad Balmer-core/wing side pits')
    if rv_edge:
        reasons.append('best RV lies at search-grid edge')
    if interval_contains_best is False:
        reasons.append('best B lies outside the 68 percent score interval')
    if np.isfinite(rel_err) and rel_err > 0.45:
        reasons.append('magnetic-field interval is broad')
    if minor_count and len(detected) < 2:
        reasons.append('solution relies partly on low-weight minor absorptions')

    if not reasons and str(quality) in ('good', 'guided_good'):
        status = 'secure_zeeman'
    elif (len(detected) >= 4 and len(series_detected) >= 2
          and np.isfinite(max_snr) and max_snr >= 4.0
          and (not np.isfinite(med_abs_offset) or med_abs_offset <= 1.5 * offset_limit)
          and not rv_edge):
        status = 'candidate_zeeman'
    else:
        status = 'low_confidence_zeeman'

    return {
        'magnetic_review_status': status,
        'magnetic_review_reasons': '; '.join(reasons) if reasons else 'passes strict automated checks',
        'magnetic_high_snr_components': high_snr,
        'magnetic_max_component_snr': max_snr,
        'magnetic_median_abs_component_offset_A': med_abs_offset,
        'magnetic_core_side_fraction': core_frac,
        'magnetic_pit_shape_fraction': pit_frac,
        'magnetic_broad_trough_fraction': broad_frac,
        'magnetic_broad_components': int(len(detected)),
        'magnetic_minor_absorption_components': minor_count,
        'magnetic_weighted_components': weighted_components,
    }


def _magnetic_claim(quality, n_detected, n_series, rel_err, rv_edge=False,
                    review_status=''):
    quality = str(quality or '')
    if quality == 'no_detection' or n_detected < 3:
        return 'no_zeeman_detection'
    if review_status == 'secure_zeeman':
        return 'probable_zeeman'
    if review_status == 'candidate_zeeman':
        return 'zeeman_candidate'
    if review_status == 'low_confidence_zeeman':
        return 'low_confidence_zeeman_candidate'
    rel_ok = (not np.isfinite(rel_err)) or rel_err <= 0.50
    if quality in ('good', 'guided_good') and rel_ok and not rv_edge and (
            n_series >= 2 or n_detected >= 4):
        return 'probable_zeeman'
    if quality in ('good', 'candidate', 'guided_good', 'guided_candidate'):
        return 'zeeman_candidate'
    return 'few_pit_candidate'


def _measure_magnetic_field_single(wave, flux, err=None, template_dir=None,
                                   series='auto', b_min_mg=20.0, b_max_mg=950.0,
                                   n_b_grid=700, rv_grid_kms=None,
                                   search_half_width_A=8.0,
                                   min_depth=0.008, min_snr=1.5,
                                   min_trough_width_A=6.0,
                                   emission_avoid_A=10.0,
                                   absorption_core_avoid_A=25.0,
                                   baseline_mode='continuum',
                                   wd_model_grid='auto',
                                   spectral_type='auto',
                                   continuum_window=None):
    """
    Estimate a WD magnetic field from weak Balmer Zeeman absorption dips.

    Parameters are intentionally permissive because some spectra only show one
    Balmer region or a few shallow split components.
    """
    prepared = _template_baseline_normalize(
        wave, flux, err, baseline_mode=baseline_mode,
        wd_model_grid=wd_model_grid, spectral_type=spectral_type,
        continuum_window=continuum_window)
    if prepared is None:
        return None
    wave, flux, err, continuum, norm, norm_err, baseline_model_flux, baseline_info = prepared

    templates = load_balmer_zeeman_templates(
        template_dir=template_dir, interpolated=True, b_max_mg=b_max_mg)
    if templates.empty:
        return None

    if series == 'auto' or series is None:
        chosen_series = []
        for ser, rest in BALMER_REST.items():
            if wave[0] - 300 <= rest <= wave[-1] + 300:
                chosen_series.append(ser)
        if not chosen_series:
            chosen_series = sorted(templates['series'].unique())
    elif isinstance(series, str):
        chosen_series = [s.strip() for s in series.split(',') if s.strip()]
    else:
        chosen_series = list(series)
    chosen_series = [
        {'ha': 'Halpha', 'halpha': 'Halpha',
         'hb': 'Hbeta', 'hbeta': 'Hbeta',
         'hg': 'Hgamma', 'hgamma': 'Hgamma'}.get(s.lower(), s)
        for s in chosen_series
    ]
    template_curves = _compile_template_curves(templates, series=chosen_series)
    if not template_curves:
        return None

    emission_mask = _make_emission_mask(wave, norm, norm_err)
    b_values = templates['B_MG'].to_numpy(float)
    b_values = b_values[np.isfinite(b_values)]
    if b_min_mg is None:
        b_min_mg = max(float(np.nanmin(b_values)), 1.0)
    b_max = min(float(b_max_mg), float(np.nanmax(b_values)))
    b_grid = np.linspace(float(b_min_mg), b_max, int(n_b_grid))
    if rv_grid_kms is None:
        rv_grid_kms = np.arange(-250.0, 251.0, 25.0)
    else:
        rv_grid_kms = np.asarray(rv_grid_kms, dtype=float)

    best = None
    score_rows = []
    for B in b_grid:
        best_for_b = None
        for rv in rv_grid_kms:
            scored = _score_one_field(
                wave, norm, norm_err, emission_mask, template_curves, B,
                rv_kms=rv, series=chosen_series,
                search_half_width_A=search_half_width_A,
                min_depth=min_depth, min_snr=min_snr,
                min_trough_width_A=min_trough_width_A,
                emission_avoid_A=emission_avoid_A,
                absorption_core_avoid_A=absorption_core_avoid_A)
            if best_for_b is None or scored['score'] > best_for_b['score']:
                best_for_b = scored
            if best is None or scored['score'] > best['score']:
                best = scored
        score_rows.append({
            'B_MG': float(B),
            'best_score': float(best_for_b['score']),
            'best_rv_kms': float(best_for_b['rv_kms']),
            'n_detected': int(best_for_b['n_detected']),
            'n_minor_support': int(best_for_b.get('n_minor_support', 0)),
            'weighted_detected_components': float(
                best_for_b.get('weighted_detected_components',
                               best_for_b['n_detected'])),
            'n_usable': int(best_for_b['n_usable']),
        })

    score_df = pd.DataFrame(score_rows)
    if best is None:
        return None
    solution_table = _field_solution_peaks(
        score_df, min_separation_mg=max(6.0, 0.012 * float(b_max)))
    ambiguous_solutions = False
    if not solution_table.empty and len(solution_table) >= 2:
        top = solution_table.iloc[0]
        second = solution_table.iloc[1]
        delta_score = float(second.get('delta_score_from_best', np.inf))
        b_gap = abs(float(second.get('B_MG', np.nan))
                    - float(top.get('B_MG', np.nan)))
        ambiguous_solutions = (
            np.isfinite(delta_score) and np.isfinite(b_gap)
            and delta_score <= 4.0 and b_gap >= 8.0)
    score = score_df['best_score'].to_numpy(float)
    likelihood = np.exp((score - np.nanmax(score)) / 2.0)
    q16, q50, q84 = _weighted_quantile(
        score_df['B_MG'].to_numpy(float), likelihood, [0.16, 0.5, 0.84])

    component_df = pd.DataFrame(best['details'])
    detected = component_df[component_df.get('detected', False) == True] if not component_df.empty else component_df
    series_detected = sorted(set(detected['series'])) if not detected.empty else []
    rv_edge = bool(np.isclose(best['rv_kms'], np.nanmin(rv_grid_kms))
                   or np.isclose(best['rv_kms'], np.nanmax(rv_grid_kms)))
    rel_err = np.nan
    ci_lo, ci_hi, err_minus, err_plus, ci_contains_best = (
        _interval_errors_around_best(best['B_MG'], q16, q84))
    if np.isfinite(ci_lo) and np.isfinite(ci_hi) and best['B_MG'] > 0:
        rel_err = 0.5 * (ci_hi - ci_lo) / best['B_MG']
    if best['n_detected'] >= 3 and len(series_detected) >= 1 and not rv_edge and (
            not np.isfinite(rel_err) or rel_err < 0.45):
        quality = 'good'
    elif best['n_detected'] >= 3:
        quality = 'candidate'
    else:
        quality = 'no_detection'
    n_series_detected, detected_labels = _component_summary(component_df)
    review = _review_magnetic_detection(
        component_df, quality, rel_err, rv_edge=rv_edge,
        b_mg=float(best['B_MG']), search_half_width_A=search_half_width_A,
        interval_contains_best=ci_contains_best)
    claim = _magnetic_claim(
        quality, best['n_detected'], n_series_detected, rel_err,
        rv_edge=rv_edge,
        review_status=review.get('magnetic_review_status', ''))

    return {
        'B_MG': float(best['B_MG']),
        'B_err_minus_MG': float(err_minus) if np.isfinite(err_minus) else np.nan,
        'B_err_plus_MG': float(err_plus) if np.isfinite(err_plus) else np.nan,
        'B_interval_lower_MG': float(ci_lo) if np.isfinite(ci_lo) else np.nan,
        'B_interval_upper_MG': float(ci_hi) if np.isfinite(ci_hi) else np.nan,
        'B_interval_contains_best': ci_contains_best,
        'B_posterior_median_MG': float(q50) if np.isfinite(q50) else np.nan,
        'rv_kms': float(best['rv_kms']),
        'score': float(best['score']),
        'quality': quality,
        'field_regime': _field_regime(float(best['B_MG'])),
        'magnetic_claim': claim,
        **review,
        'n_detected_components': int(best['n_detected']),
        'n_minor_absorption_components': int(best.get('n_minor_support', 0)),
        'weighted_detected_components': float(
            best.get('weighted_detected_components', best['n_detected'])),
        'n_usable_components': int(best['n_usable']),
        'n_core_side_pits': int(best.get('n_core_side_pits', 0)),
        'n_core_side_pit_series': int(best.get('n_core_side_pit_series', 0)),
        'core_side_pit_series': best.get('core_side_pit_series', ''),
        'n_core_side_pair_series': int(best.get('n_core_side_pair_series', 0)),
        'core_side_pair_series': best.get('core_side_pair_series', ''),
        'n_detected_series': int(n_series_detected),
        'detected_component_labels': detected_labels,
        'series_used': ';'.join(chosen_series),
        'series_detected': ';'.join(series_detected),
        'rv_at_search_edge': rv_edge,
        'relative_B_uncertainty': float(rel_err) if np.isfinite(rel_err) else np.nan,
        'component_table': component_df,
        'score_grid': score_df,
        'field_solution_table': solution_table,
        'ambiguous_field_solutions': bool(ambiguous_solutions),
        'wave': wave,
        'flux': flux,
        'continuum': continuum,
        'norm_flux': norm,
        'norm_err': norm_err,
        'emission_mask': emission_mask,
        'baseline_model_flux': baseline_model_flux,
        **baseline_info,
        'template_dir': template_dir or DEFAULT_TEMPLATE_DIR,
        'notes': 'B is from dominant Zeeman-template absorption dips; emission-contaminated components are skipped.',
    }


def evaluate_fixed_magnetic_field(wave, flux, reference_b_mg, err=None,
                                  template_dir=None,
                                  series='Halpha,Hbeta',
                                  rv_grid_kms=None,
                                  search_half_width_A=30.0,
                                  min_depth=0.008,
                                  min_snr=1.5,
                                  min_trough_width_A=10.0,
                                  max_component_offset_A=12.0,
                                  emission_avoid_A=10.0,
                                  absorption_core_avoid_A=25.0,
                                  baseline_mode='continuum',
                                  wd_model_grid='auto',
                                  spectral_type='auto',
                                  continuum_window=None):
    """
    Evaluate whether a fixed literature/reference field is supported.

    This does not search for the best B.  It projects the Balmer Zeeman
    component wavelengths for ``reference_b_mg`` and asks whether broad
    absorption bands are present at those expected wavelengths.
    """
    if not np.isfinite(reference_b_mg) or float(reference_b_mg) <= 0:
        return None
    prepared = _template_baseline_normalize(
        wave, flux, err, baseline_mode=baseline_mode,
        wd_model_grid=wd_model_grid, spectral_type=spectral_type,
        continuum_window=continuum_window)
    if prepared is None:
        return None
    wave, flux, err, continuum, norm, norm_err, baseline_model_flux, baseline_info = prepared

    templates = load_balmer_zeeman_templates(
        template_dir=template_dir, interpolated=True,
        b_max_mg=max(float(reference_b_mg) + 50.0, 950.0))
    if templates.empty:
        return None
    if isinstance(series, str):
        chosen_series = [s.strip() for s in series.split(',') if s.strip()]
    else:
        chosen_series = list(series or ['Halpha', 'Hbeta'])
    chosen_series = [
        {'ha': 'Halpha', 'halpha': 'Halpha',
         'hb': 'Hbeta', 'hbeta': 'Hbeta',
         'hg': 'Hgamma', 'hgamma': 'Hgamma'}.get(str(s).lower(), str(s))
        for s in chosen_series
    ]
    template_curves = _compile_template_curves(templates, series=chosen_series)
    if not template_curves:
        return None
    if rv_grid_kms is None:
        rv_grid_kms = np.arange(-250.0, 251.0, 25.0)
    else:
        rv_grid_kms = np.asarray(rv_grid_kms, dtype=float)
    emission_mask = _make_emission_mask(wave, norm, norm_err)
    best = None
    score_rows = []
    for rv in rv_grid_kms:
        scored = _score_one_field(
            wave, norm, norm_err, emission_mask, template_curves,
            float(reference_b_mg), rv_kms=float(rv), series=chosen_series,
            search_half_width_A=search_half_width_A,
            min_depth=min_depth, min_snr=min_snr,
            min_trough_width_A=min_trough_width_A,
            max_component_offset_A=max_component_offset_A,
            emission_avoid_A=emission_avoid_A,
            absorption_core_avoid_A=absorption_core_avoid_A)
        score_rows.append({
            'B_MG': float(reference_b_mg),
            'best_score': float(scored['score']),
            'best_rv_kms': float(scored['rv_kms']),
            'n_detected': int(scored['n_detected']),
            'n_minor_support': int(scored.get('n_minor_support', 0)),
            'weighted_detected_components': float(
                scored.get('weighted_detected_components',
                           scored['n_detected'])),
            'n_usable': int(scored['n_usable']),
        })
        if best is None or scored['score'] > best['score']:
            best = scored
    if best is None:
        return None

    component_df = pd.DataFrame(best['details'])
    detected = component_df[component_df.get('detected', False) == True] if not component_df.empty else component_df
    series_detected = sorted(set(detected['series'])) if not detected.empty else []
    rv_edge = bool(np.isclose(best['rv_kms'], np.nanmin(rv_grid_kms))
                   or np.isclose(best['rv_kms'], np.nanmax(rv_grid_kms)))
    if best['n_detected'] >= 3 and len(series_detected) >= 2 and not rv_edge:
        quality = 'reference_supported'
    else:
        quality = 'reference_not_supported'
    n_series_detected, detected_labels = _component_summary(component_df)
    review = _review_magnetic_detection(
        component_df, quality, np.nan, rv_edge=rv_edge,
        b_mg=float(reference_b_mg),
        search_half_width_A=search_half_width_A,
        interval_contains_best=True)
    claim = _magnetic_claim(
        quality, best['n_detected'], n_series_detected, np.nan,
        rv_edge=rv_edge,
        review_status=review.get('magnetic_review_status', ''))
    score_df = pd.DataFrame(score_rows)
    return {
        'B_MG': float(reference_b_mg),
        'B_err_minus_MG': np.nan,
        'B_err_plus_MG': np.nan,
        'B_interval_lower_MG': np.nan,
        'B_interval_upper_MG': np.nan,
        'B_interval_contains_best': True,
        'B_posterior_median_MG': float(reference_b_mg),
        'rv_kms': float(best['rv_kms']),
        'score': float(best['score']),
        'quality': quality,
        'field_regime': _field_regime(float(reference_b_mg)),
        'magnetic_claim': claim,
        **review,
        'n_detected_components': int(best['n_detected']),
        'n_minor_absorption_components': int(best.get('n_minor_support', 0)),
        'weighted_detected_components': float(
            best.get('weighted_detected_components', best['n_detected'])),
        'n_usable_components': int(best['n_usable']),
        'n_core_side_pits': int(best.get('n_core_side_pits', 0)),
        'n_core_side_pit_series': int(best.get('n_core_side_pit_series', 0)),
        'core_side_pit_series': best.get('core_side_pit_series', ''),
        'n_core_side_pair_series': int(best.get('n_core_side_pair_series', 0)),
        'core_side_pair_series': best.get('core_side_pair_series', ''),
        'n_detected_series': int(n_series_detected),
        'detected_component_labels': detected_labels,
        'series_used': ';'.join(chosen_series),
        'series_detected': ';'.join(series_detected),
        'rv_at_search_edge': rv_edge,
        'relative_B_uncertainty': np.nan,
        'component_table': component_df,
        'score_grid': score_df,
        'field_solution_table': pd.DataFrame([{
            'solution_rank': 1,
            'B_MG': float(reference_b_mg),
            'best_score': float(best['score']),
            'delta_score_from_best': 0.0,
            'best_rv_kms': float(best['rv_kms']),
            'n_detected': int(best['n_detected']),
            'n_minor_support': int(best.get('n_minor_support', 0)),
            'weighted_detected_components': float(
                best.get('weighted_detected_components',
                         best['n_detected'])),
            'n_usable': int(best['n_usable']),
        }]),
        'ambiguous_field_solutions': False,
        'wave': wave,
        'flux': flux,
        'continuum': continuum,
        'norm_flux': norm,
        'norm_err': norm_err,
        'emission_mask': emission_mask,
        'baseline_model_flux': baseline_model_flux,
        **baseline_info,
        'template_dir': template_dir or DEFAULT_TEMPLATE_DIR,
        'fit_method': 'fixed_reference_field',
        'notes': 'Fixed literature/reference B; checks broad absorption support without searching B.',
    }


def _regime_metrics(prefix, result):
    if result is None:
        return {
            f'{prefix}_B_MG': np.nan,
            f'{prefix}_B_err_minus_MG': np.nan,
            f'{prefix}_B_err_plus_MG': np.nan,
            f'{prefix}_rv_kms': np.nan,
            f'{prefix}_quality': '',
            f'{prefix}_magnetic_claim': '',
            f'{prefix}_magnetic_review_status': '',
            f'{prefix}_magnetic_review_reasons': '',
            f'{prefix}_n_detected_components': 0,
            f'{prefix}_n_minor_absorption_components': 0,
            f'{prefix}_weighted_detected_components': 0.0,
            f'{prefix}_n_usable_components': 0,
            f'{prefix}_n_core_side_pits': 0,
            f'{prefix}_n_core_side_pit_series': 0,
            f'{prefix}_core_side_pit_series': '',
            f'{prefix}_n_core_side_pair_series': 0,
            f'{prefix}_core_side_pair_series': '',
            f'{prefix}_rv_at_search_edge': False,
            f'{prefix}_score': np.nan,
            f'{prefix}_ambiguous_field_solutions': False,
        }
    return {
        f'{prefix}_B_MG': result.get('B_MG', np.nan),
        f'{prefix}_B_err_minus_MG': result.get('B_err_minus_MG', np.nan),
        f'{prefix}_B_err_plus_MG': result.get('B_err_plus_MG', np.nan),
        f'{prefix}_rv_kms': result.get('rv_kms', np.nan),
        f'{prefix}_quality': result.get('quality', ''),
        f'{prefix}_magnetic_claim': result.get('magnetic_claim', ''),
        f'{prefix}_magnetic_review_status': result.get('magnetic_review_status', ''),
        f'{prefix}_magnetic_review_reasons': result.get('magnetic_review_reasons', ''),
        f'{prefix}_n_detected_components': result.get('n_detected_components', 0),
        f'{prefix}_n_minor_absorption_components': result.get(
            'n_minor_absorption_components', 0),
        f'{prefix}_weighted_detected_components': result.get(
            'weighted_detected_components', 0.0),
        f'{prefix}_n_usable_components': result.get('n_usable_components', 0),
        f'{prefix}_n_core_side_pits': result.get('n_core_side_pits', 0),
        f'{prefix}_n_core_side_pit_series': result.get('n_core_side_pit_series', 0),
        f'{prefix}_core_side_pit_series': result.get('core_side_pit_series', ''),
        f'{prefix}_n_core_side_pair_series': result.get('n_core_side_pair_series', 0),
        f'{prefix}_core_side_pair_series': result.get('core_side_pair_series', ''),
        f'{prefix}_rv_at_search_edge': result.get('rv_at_search_edge', False),
        f'{prefix}_score': result.get('score', np.nan),
        f'{prefix}_ambiguous_field_solutions': result.get('ambiguous_field_solutions', False),
    }


def _relative_interval_width(result):
    if result is None:
        return np.inf
    b = float(result.get('B_MG', np.nan))
    em = float(result.get('B_err_minus_MG', np.nan))
    ep = float(result.get('B_err_plus_MG', np.nan))
    if not (np.isfinite(b) and b > 0 and np.isfinite(em) and np.isfinite(ep)):
        return np.inf
    return 0.5 * (em + ep) / b


def _choose_split_regime(low_result, high_result, boundary_mg):
    if low_result is None:
        return high_result, 'high'
    if high_result is None:
        return low_result, 'low'

    boundary = float(boundary_mg)
    low_b = float(low_result.get('B_MG', np.nan))
    high_b = float(high_result.get('B_MG', np.nan))
    low_score = float(low_result.get('score', np.nan))
    high_score = float(high_result.get('score', np.nan))
    if not np.isfinite(low_score):
        low_score = -np.inf
    if not np.isfinite(high_score):
        high_score = -np.inf
    low_det = int(low_result.get('n_detected_components', 0) or 0)
    high_det = int(high_result.get('n_detected_components', 0) or 0)
    low_support = float(low_result.get(
        'weighted_detected_components', low_det) or 0.0)
    high_support = float(high_result.get(
        'weighted_detected_components', high_det) or 0.0)
    low_pair = int(low_result.get('n_core_side_pair_series', 0) or 0)
    high_core = int(high_result.get('n_core_side_pits', 0) or 0)
    high_rel = _relative_interval_width(high_result)
    low_rel = _relative_interval_width(low_result)
    low_quality = str(low_result.get('quality', ''))
    high_quality = str(high_result.get('quality', ''))
    high_review = str(high_result.get('magnetic_review_status', ''))
    high_core_frac = high_core / max(high_det, 1)
    low_ok = (
        low_quality in ('good', 'candidate', 'guided_good', 'guided_candidate')
        and low_det >= 4
        and low_pair >= 1
        and np.isfinite(low_b)
        and low_b <= 0.75 * boundary
        and (not np.isfinite(low_rel) or low_rel <= 0.80))
    low_notch_prefer = (
        np.isfinite(low_b)
        and low_b <= 0.80 * boundary
        and low_det >= 3
        and low_support >= 3.0
        and low_pair >= 1
        and low_score >= 1.15 * high_score)
    strong_high = (
        np.isfinite(high_b)
        and high_b >= max(180.0, 2.3 * boundary)
        and (high_det >= 6 or (high_det >= 4 and high_support >= 5.0))
        and high_rel <= 0.90
        and high_score >= 0.55 * low_score)
    moderate_high = (
        np.isfinite(high_b)
        and high_b >= max(85.0, 1.20 * boundary)
        and high_quality in ('good', 'candidate', 'guided_good', 'guided_candidate')
        and (high_det >= 8 or (high_det >= 5 and high_support >= 6.0))
        and high_rel <= 0.90
        and high_score >= 0.74 * low_score)
    boundary_high = (
        np.isfinite(high_b)
        and boundary <= high_b <= boundary + 12.0
        and high_quality in ('good', 'candidate', 'guided_good', 'guided_candidate')
        and (high_det >= 6 or (high_det >= 4 and high_support >= 5.0))
        and high_score >= 0.65 * low_score)
    high_score_win = (
        np.isfinite(high_b)
        and high_b >= boundary
        and high_quality in ('good', 'candidate', 'guided_good', 'guided_candidate')
        and (high_det >= 6 or (high_det >= 4 and high_support >= 5.0))
        and high_rel <= 0.90
        and high_score >= 1.15 * low_score)

    if low_notch_prefer:
        return low_result, 'low'
    if strong_high or moderate_high or boundary_high or high_score_win:
        return high_result, 'high'
    if low_ok and (
            low_score >= 0.80 * high_score
            or (high_core_frac >= 0.55 and low_score >= 0.62 * high_score)
            or (high_review == 'low_confidence_zeeman'
                and low_score >= 0.62 * high_score)):
        return low_result, 'low'
    return high_result, 'high'


def measure_magnetic_field(wave, flux, err=None, template_dir=None,
                           series='auto', b_min_mg=5.0, b_max_mg=950.0,
                           n_b_grid=700, rv_grid_kms=None,
                           search_half_width_A=8.0,
                           min_depth=0.008, min_snr=1.5,
                           min_trough_width_A=6.0,
                           emission_avoid_A=10.0,
                           absorption_core_avoid_A=25.0,
                           baseline_mode='continuum',
                           wd_model_grid='auto',
                           spectral_type='auto',
                           continuum_window=None,
                           field_mode='auto',
                           low_high_boundary_mg=35.0):
    """
    Estimate a WD magnetic field with separate low/high-field searches.

    ``field_mode='auto'`` runs a low-field Balmer-core split search below
    ``low_high_boundary_mg`` and a high-field Zeeman-component search above it.
    The returned result is the selected regime, with both regime summaries
    copied into the output dictionary for review.
    """
    mode = str(field_mode or 'auto').lower()
    if mode in ('single', 'legacy'):
        result = _measure_magnetic_field_single(
            wave, flux, err=err, template_dir=template_dir,
            series=series, b_min_mg=b_min_mg, b_max_mg=b_max_mg,
            n_b_grid=n_b_grid, rv_grid_kms=rv_grid_kms,
            search_half_width_A=search_half_width_A,
            min_depth=min_depth, min_snr=min_snr,
            min_trough_width_A=min_trough_width_A,
            emission_avoid_A=emission_avoid_A,
            absorption_core_avoid_A=absorption_core_avoid_A,
            baseline_mode=baseline_mode,
            wd_model_grid=wd_model_grid,
            spectral_type=spectral_type,
            continuum_window=continuum_window)
        if result is not None:
            result['analysis_regime'] = 'single'
            result['field_mode'] = 'single'
        return result

    boundary = float(low_high_boundary_mg)
    low_result = None
    high_result = None
    if mode in ('auto', 'split', 'low') and float(b_min_mg) < boundary:
        low_max = min(float(b_max_mg), boundary)
        if low_max > float(b_min_mg):
            low_n_grid = max(180, min(int(n_b_grid), 360))
            low_result = _measure_magnetic_field_single(
                wave, flux, err=err, template_dir=template_dir,
                series=series, b_min_mg=float(b_min_mg), b_max_mg=low_max,
                n_b_grid=low_n_grid, rv_grid_kms=rv_grid_kms,
                search_half_width_A=max(float(search_half_width_A), 20.0),
                min_depth=min_depth, min_snr=min_snr,
                min_trough_width_A=max(float(min_trough_width_A), 8.0),
                emission_avoid_A=emission_avoid_A,
                absorption_core_avoid_A=absorption_core_avoid_A,
                baseline_mode=baseline_mode,
                wd_model_grid=wd_model_grid,
                spectral_type=spectral_type,
                continuum_window=continuum_window)
            if low_result is not None:
                low_result['analysis_regime'] = 'low'
                low_result['field_mode'] = mode
    if mode == 'low':
        result = low_result
        selected_regime = 'low'
    else:
        high_min = max(float(b_min_mg), boundary)
        if float(b_max_mg) > high_min:
            high_result = _measure_magnetic_field_single(
                wave, flux, err=err, template_dir=template_dir,
                series=series, b_min_mg=high_min, b_max_mg=float(b_max_mg),
                n_b_grid=int(n_b_grid), rv_grid_kms=rv_grid_kms,
                search_half_width_A=search_half_width_A,
                min_depth=min_depth, min_snr=min_snr,
                min_trough_width_A=min_trough_width_A,
                emission_avoid_A=emission_avoid_A,
                absorption_core_avoid_A=absorption_core_avoid_A,
                baseline_mode=baseline_mode,
                wd_model_grid=wd_model_grid,
                spectral_type=spectral_type,
                continuum_window=continuum_window)
            if high_result is not None:
                high_result['analysis_regime'] = 'high'
                high_result['field_mode'] = mode
        if mode == 'high':
            result = high_result
            selected_regime = 'high'
        else:
            result, selected_regime = _choose_split_regime(
                low_result, high_result, boundary)

    if result is None:
        return None
    result = dict(result)
    result.update(_regime_metrics('low_field', low_result))
    result.update(_regime_metrics('high_field', high_result))
    result['analysis_regime'] = selected_regime
    result['field_mode'] = mode
    result['low_high_boundary_MG'] = boundary
    result['notes'] = (
        'Auto split: low-field search covers Balmer-core side pits; '
        'high-field search covers wider Zeeman component curves.')
    return result


def plot_magnetic_field_result(result, save_path=None, title=None):
    if result is None:
        return None
    wave = result['wave']
    norm = result['norm_flux']
    norm_err = result['norm_err']
    comp = result['component_table']
    score = result['score_grid']
    emission_mask = result['emission_mask']

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=False,
        gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    ax.fill_between(wave, norm - norm_err, norm + norm_err,
                    color='0.75', alpha=0.22, lw=0, label='1 sigma')
    ax.plot(wave, norm, color='black', lw=0.55, alpha=0.75,
            label=f"Spectrum residual ({result.get('baseline_method', 'baseline')})")
    if np.any(emission_mask):
        ax.scatter(wave[emission_mask], norm[emission_mask],
                   s=4, color='tab:red', alpha=0.25,
                   label='masked emission peaks')

    if comp is not None and not comp.empty:
        ymax = np.nanpercentile(norm[np.isfinite(norm)], 98)
        for _, row in comp.iterrows():
            pred = row.get('predicted_wavelength_A', np.nan)
            if not np.isfinite(pred):
                continue
            detected = bool(row.get('detected', False))
            minor = bool(row.get('minor_absorption_support', False))
            usable = bool(row.get('usable', False))
            if detected:
                color = 'tab:green'
                alpha = 0.90
            elif minor:
                color = 'tab:blue'
                alpha = 0.55
            elif usable:
                color = 'tab:orange'
                alpha = 0.45
            else:
                color = '0.55'
                alpha = 0.25
            ax.axvline(pred, color=color, lw=0.9, ls='--', alpha=alpha)
            if detected:
                label = f"{SERIES_LABELS.get(row['series'], row['series'])}{int(row['component'])}"
                ax.text(pred, ymax, label, rotation=90, va='top', ha='right',
                        fontsize=7, color=color)

    ax.axhline(1.0, color='0.4', lw=0.8, ls=':')
    ax.set_ylabel('Normalized flux')
    title = title or 'Magnetic WD Zeeman-field fit'
    ax.set_title(
        f"{title}\nB={result['B_MG']:.1f} -{result['B_err_minus_MG']:.1f} "
        f"+{result['B_err_plus_MG']:.1f} MG, "
        f"RV={result['rv_kms']:.0f} km/s, quality={result['quality']}\n"
        f"review={result.get('magnetic_review_status', 'not_reviewed')}, "
        f"baseline={result.get('baseline_method', '')}, "
        f"WD={result.get('wd_spectral_type', '')} "
        f"Teff={result.get('wd_template_teff', np.nan):.0f}K "
        f"logg={result.get('wd_template_logg', np.nan):.2f}")
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.25)
    utils.set_spectrum_axes(ax, wave, norm)

    ax2 = axes[1]
    ax2.plot(score['B_MG'], score['best_score'], color='black', lw=1.0)
    ax2.axvline(result['B_MG'], color='tab:green', lw=1.2,
                label='best B')
    lo = result['B_MG'] - result['B_err_minus_MG']
    hi = result['B_MG'] + result['B_err_plus_MG']
    if np.isfinite(lo) and np.isfinite(hi):
        ax2.axvspan(lo, hi, color='tab:green', alpha=0.18,
                    label='68% score interval')
    ax2.set_xlabel('Magnetic field B (MG)')
    ax2.set_ylabel('Zeeman score')
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8, loc='best')

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def _robust_axis_limits(values, percentiles=(1.0, 99.0), pad_fraction=0.10):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    lo, hi = np.nanpercentile(values, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if np.isclose(lo, hi):
        delta = max(abs(hi) * 0.05, 1.0)
        return lo - delta, hi + delta
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def _series_component_label(series, component):
    base = SERIES_LABELS.get(series, series)
    return f"{base}{int(component)}"


def _is_adopted_zeeman_result(result):
    if result is None:
        return False
    claim = str(result.get('magnetic_claim', ''))
    review = str(result.get('magnetic_review_status', ''))
    n_det = int(result.get('n_detected_components', 0) or 0)
    return (
        claim != 'no_zeeman_detection'
        and n_det >= 3
        and review in ('secure_zeeman', 'candidate_zeeman'))


def is_adopted_zeeman_result(result):
    """Return True only for reviewer-facing adopted Zeeman detections."""
    return _is_adopted_zeeman_result(result)


def _field_label(result, b_mg=None, reference=False):
    if result is None:
        return ''
    if b_mg is None:
        b_mg = result.get('B_MG', np.nan)
    if not np.isfinite(b_mg):
        return 'No adopted B'
    if reference:
        return f'B reference = {float(b_mg):.0f} MG'
    prefix = 'B' if _is_adopted_zeeman_result(result) else 'trial B (not adopted)'
    return (
        f"{prefix} = {float(b_mg):.1f} "
        f"-{float(result.get('B_err_minus_MG', np.nan)):.1f} "
        f"+{float(result.get('B_err_plus_MG', np.nan)):.1f} MG")


def get_guided_zeeman_lines(preset='cechichang_220'):
    """Return a copy of a named guided Zeeman line-identification preset."""
    if not preset or str(preset).lower() in ('none', 'off', 'false'):
        return []
    key = str(preset)
    if key not in GUIDED_ZEEMAN_PRESETS:
        raise ValueError(
            f"Unknown guided Zeeman preset {preset!r}; available: "
            f"{', '.join(sorted(GUIDED_ZEEMAN_PRESETS))}")
    return [dict(row) for row in GUIDED_ZEEMAN_PRESETS[key]]


def _huber_loss(z, delta=2.0):
    z = np.asarray(z, dtype=float)
    az = np.abs(z)
    return np.where(az <= delta, z * z, 2.0 * delta * az - delta * delta)


def fit_magnetic_field_from_guided_lines(lines=None, preset='cechichang_220',
                                         template_dir=None,
                                         b_min_mg=120.0, b_max_mg=300.0,
                                         n_b_grid=2500,
                                         rv_grid_kms=None,
                                         robust=True):
    """
    Fit B from explicitly identified Zeeman component wavelengths.

    This is the safer mode for broad, hard-to-recognize magnetic WDs where a
    blind local-minimum search can be dominated by continuum ripples.
    """
    if lines is None:
        lines = get_guided_zeeman_lines(preset)
    lines = [dict(row) for row in lines or []]
    if not lines:
        return None

    templates = load_balmer_zeeman_templates(
        template_dir=template_dir, interpolated=True,
        b_max_mg=max(float(b_max_mg) + 50.0, 1000.0))
    if templates.empty:
        return None

    curves = {}
    usable_lines = []
    for row in lines:
        series = row.get('series')
        component = int(row.get('component'))
        observed = float(row.get('observed_wavelength_A'))
        sigma = float(row.get('sigma_A', 35.0) or 35.0)
        df = templates[
            (templates['series'] == series)
            & (templates['component'] == component)
        ].sort_values('B_MG')
        if df.empty or not np.isfinite(observed):
            continue
        b = df['B_MG'].to_numpy(float)
        wl = df['wavelength_A'].to_numpy(float)
        good = np.isfinite(b) & np.isfinite(wl)
        if np.sum(good) < 2:
            continue
        order = np.argsort(b[good])
        curves[(series, component)] = (b[good][order], wl[good][order])
        row['observed_wavelength_A'] = observed
        row['sigma_A'] = max(sigma, 1.0)
        row['label'] = row.get('label') or _series_component_label(series, component)
        usable_lines.append(row)
    if len(usable_lines) < 2:
        return None

    b_grid = np.linspace(float(b_min_mg), float(b_max_mg), int(n_b_grid))
    if rv_grid_kms is None:
        rv_grid_kms = np.array([0.0])
    else:
        rv_grid_kms = np.asarray(rv_grid_kms, dtype=float)

    score_rows = []
    best = None
    best_details = None
    for B in b_grid:
        best_for_b = None
        for rv in rv_grid_kms:
            details = []
            residuals = []
            sigmas = []
            scale = 1.0 + float(rv) / C_KMS
            invalid = False
            for row in usable_lines:
                key = (row['series'], int(row['component']))
                b, wl = curves[key]
                if B < b.min() or B > b.max():
                    invalid = True
                    break
                pred_rest = float(np.interp(B, b, wl))
                pred = pred_rest * scale
                obs = float(row['observed_wavelength_A'])
                sigma = float(row['sigma_A'])
                residual = obs - pred
                residuals.append(residual)
                sigmas.append(sigma)
                details.append({
                    'series': row['series'],
                    'component': int(row['component']),
                    'label': row['label'],
                    'region': row.get('region', ''),
                    'observed_wavelength_A': obs,
                    'detected_wavelength_A': obs,
                    'predicted_rest_wavelength_A': pred_rest,
                    'predicted_wavelength_A': pred,
                    'delta_detected_minus_model_A': residual,
                    'sigma_A': sigma,
                    'B_MG': float(B),
                    'rv_kms': float(rv),
                    'detected': True,
                    'usable': True,
                    'skip_reason': '',
                })
            if invalid:
                continue
            z = np.asarray(residuals) / np.asarray(sigmas)
            if robust:
                chi2 = float(np.sum(_huber_loss(z, delta=2.0)))
            else:
                chi2 = float(np.sum(z * z))
            scored = {
                'B_MG': float(B),
                'rv_kms': float(rv),
                'chi2': chi2,
                'chi2_red': chi2 / max(len(usable_lines) - 1, 1),
                'details': details,
            }
            if best_for_b is None or chi2 < best_for_b['chi2']:
                best_for_b = scored
            if best is None or chi2 < best['chi2']:
                best = scored
                best_details = details
        if best_for_b is not None:
            score_rows.append({
                'B_MG': float(B),
                'best_score': float(-0.5 * best_for_b['chi2']),
                'best_chi2': float(best_for_b['chi2']),
                'best_rv_kms': float(best_for_b['rv_kms']),
                'n_detected': int(len(usable_lines)),
                'n_usable': int(len(usable_lines)),
            })

    if best is None or not score_rows:
        return None

    score_df = pd.DataFrame(score_rows)
    solution_table = _field_solution_peaks(score_df, min_separation_mg=8.0)
    likelihood = np.exp(-0.5 * (
        score_df['best_chi2'].to_numpy(float)
        - float(score_df['best_chi2'].min())))
    q16, q50, q84 = _weighted_quantile(
        score_df['B_MG'].to_numpy(float), likelihood, [0.16, 0.5, 0.84])
    line_table = pd.DataFrame(best_details)
    line_table['residual_over_sigma'] = (
        line_table['delta_detected_minus_model_A'] / line_table['sigma_A'])
    ci_lo, ci_hi, err_minus, err_plus, ci_contains_best = (
        _interval_errors_around_best(best['B_MG'], q16, q84))
    if np.isfinite(ci_lo) and np.isfinite(ci_hi) and best['B_MG'] > 0:
        rel_err = 0.5 * (ci_hi - ci_lo) / best['B_MG']
    else:
        rel_err = np.nan
    n_series_detected, detected_labels = _component_summary(line_table)
    quality = 'guided_good' if best['chi2_red'] < 1.5 else 'guided_candidate'
    return {
        'B_MG': float(best['B_MG']),
        'B_err_minus_MG': float(err_minus) if np.isfinite(err_minus) else np.nan,
        'B_err_plus_MG': float(err_plus) if np.isfinite(err_plus) else np.nan,
        'B_interval_lower_MG': float(ci_lo) if np.isfinite(ci_lo) else np.nan,
        'B_interval_upper_MG': float(ci_hi) if np.isfinite(ci_hi) else np.nan,
        'B_interval_contains_best': ci_contains_best,
        'B_posterior_median_MG': float(q50) if np.isfinite(q50) else np.nan,
        'rv_kms': float(best['rv_kms']),
        'chi2_red': float(best['chi2_red']),
        'quality': quality,
        'field_regime': _field_regime(float(best['B_MG'])),
        'magnetic_claim': _magnetic_claim(
            quality, len(line_table), n_series_detected, rel_err,
            rv_edge=False),
        'component_table': line_table,
        'score_grid': score_df,
        'field_solution_table': solution_table,
        'ambiguous_field_solutions': bool(
            len(solution_table) >= 2
            and float(solution_table.iloc[1].get('delta_score_from_best', np.inf)) <= 4.0),
        'series_used': ';'.join(sorted(set(line_table['series']))),
        'series_detected': ';'.join(sorted(set(line_table['series']))),
        'n_detected_components': int(len(line_table)),
        'n_usable_components': int(len(line_table)),
        'n_detected_series': int(n_series_detected),
        'detected_component_labels': detected_labels,
        'relative_B_uncertainty': float(rel_err) if np.isfinite(rel_err) else np.nan,
        'fit_method': 'guided_zeeman_lines',
        'guided_preset': preset,
        'template_dir': template_dir or DEFAULT_TEMPLATE_DIR,
        'notes': 'B is fitted from line-identified Zeeman component wavelengths.',
    }


def apply_guided_zeeman_fit(result, preset='cechichang_220', template_dir=None,
                            b_min_mg=120.0, b_max_mg=300.0,
                            n_b_grid=2500, rv_grid_kms=None):
    """Replace the blind-search B estimate with a guided line-ID fit."""
    guided = fit_magnetic_field_from_guided_lines(
        preset=preset, template_dir=template_dir or (
            result.get('template_dir') if result else None),
        b_min_mg=b_min_mg, b_max_mg=b_max_mg, n_b_grid=n_b_grid,
        rv_grid_kms=rv_grid_kms)
    if guided is None:
        return result, None
    if result is None:
        return guided, guided
    merged = dict(result)
    old = {
        'blind_B_MG': result.get('B_MG', np.nan),
        'blind_B_err_minus_MG': result.get('B_err_minus_MG', np.nan),
        'blind_B_err_plus_MG': result.get('B_err_plus_MG', np.nan),
        'blind_rv_kms': result.get('rv_kms', np.nan),
        'blind_quality': result.get('quality', ''),
        'blind_score': result.get('score', np.nan),
    }
    merged.update(old)
    merged.update(guided)
    merged['guided_fit_applied'] = True
    return merged, guided


def balmer_blue_region_node_table(result, template_dir=None,
                                  reference_b_mg=None,
                                  xlim=(3250.0, 5500.0)):
    """Return the Zeeman component nodes plotted in the blue-region figure."""
    if result is None:
        return pd.DataFrame()
    xlo, xhi = float(xlim[0]), float(xlim[1])
    b_mg = result.get('B_MG', np.nan) if reference_b_mg is None else reference_b_mg
    if not np.isfinite(b_mg):
        return pd.DataFrame()
    rv_kms = float(result.get('rv_kms', 0.0) or 0.0)
    comp = result.get('component_table')
    result_b = result.get('B_MG', np.nan)
    use_component_table = (
        isinstance(comp, pd.DataFrame) and not comp.empty
        and (reference_b_mg is None
             or (np.isfinite(result_b)
                 and abs(float(result_b) - float(reference_b_mg)) <= 1e-6)))
    if use_component_table:
        rows = comp.copy()
        if 'predicted_wavelength_A' not in rows.columns:
            return pd.DataFrame()
        rows['node_wavelength_A'] = pd.to_numeric(
            rows['predicted_wavelength_A'], errors='coerce')
        rows['B_node_MG'] = float(b_mg)
        for col in ('detected', 'usable', 'minor_absorption_support'):
            if col not in rows.columns:
                rows[col] = False
        if 'detected_wavelength_A' not in rows.columns:
            rows['detected_wavelength_A'] = np.nan
        if 'skip_reason' not in rows.columns:
            rows['skip_reason'] = ''
        rows['delta_detected_minus_model_A'] = (
            pd.to_numeric(rows['detected_wavelength_A'], errors='coerce')
            - rows['node_wavelength_A'])
        rows['label'] = [
            _series_component_label(ser, component)
            for ser, component in zip(rows['series'], rows['component'])
        ]
    else:
        templates = load_balmer_zeeman_templates(
            template_dir=template_dir or result.get('template_dir'),
            interpolated=True, b_max_mg=max(float(b_mg) + 50.0, 1000.0),
            wavelength_min_A=2500.0, wavelength_max_A=12000.0)
        used = str(result.get('series_used', '') or '').split(';')
        used = [u for u in used if u]
        if not used:
            used = ['Halpha', 'Hbeta']
        nodes = _component_wavelengths(templates, float(b_mg), series=used)
        rows = pd.DataFrame(nodes)
        if rows.empty:
            return rows
        rows['node_wavelength_A'] = (
            rows['wavelength_A'] * (1.0 + rv_kms / C_KMS))
        rows['B_node_MG'] = float(b_mg)
        rows['rv_kms'] = rv_kms
        rows['detected'] = False
        rows['minor_absorption_support'] = False
        rows['usable'] = True
        rows['detected_wavelength_A'] = np.nan
        rows['delta_detected_minus_model_A'] = np.nan
        rows['skip_reason'] = ''
        rows['label'] = [
            _series_component_label(ser, component)
            for ser, component in zip(rows['series'], rows['component'])
        ]

    good = (
        np.isfinite(rows['node_wavelength_A'])
        & (rows['node_wavelength_A'] >= xlo)
        & (rows['node_wavelength_A'] <= xhi)
    )
    keep_cols = [
        'series', 'component', 'label', 'B_node_MG', 'node_wavelength_A',
        'detected_wavelength_A', 'delta_detected_minus_model_A',
        'detected', 'minor_absorption_support', 'usable',
        'depth', 'snr', 'sigma_A', 'skip_reason',
        'evidence_class', 'evidence_weight',
        'evidence_depth', 'evidence_snr',
        'embedded_notch_depth', 'embedded_notch_snr',
        'embedded_notch_width_A', 'contamination_reasons',
        'anchored_notch_depth', 'anchored_notch_snr',
        'anchored_notch_width_A', 'anchored_notch_left_A',
        'anchored_notch_right_A',
    ]
    rows = rows.loc[good].copy()
    for col in keep_cols:
        if col not in rows.columns:
            rows[col] = (
                False if col in ('detected', 'usable', 'minor_absorption_support')
                else '' if col == 'skip_reason'
                else np.nan)
    return rows[keep_cols].sort_values(
        ['series', 'component']).reset_index(drop=True)


def plot_balmer_blue_region_interpolation(result, save_path=None, template_dir=None,
                                          reference_wavelengths=None,
                                          reference_b_mg=None,
                                          xlim=(3250.0, 5500.0),
                                          flux_scale=1e16,
                                          title=None,
                                          dpi=300):
    """
    Reproduce the Balmer blue-region Zeeman interpolation diagnostic.

    The layout follows the reference notebook figure
    ``Balmer_series_blue_region_interpolation.pdf``: observed blue-arm spectrum
    on top, H-beta Zeeman component curves below, green wavelength markers, and
    a red horizontal magnetic-field reference line.
    """
    if result is None:
        return None
    wave = np.asarray(result.get('wave'), dtype=float)
    flux = np.asarray(result.get('flux'), dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if not np.any(good):
        return None
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    xlo, xhi = float(xlim[0]), float(xlim[1])
    in_blue = (wave >= xlo) & (wave <= xhi)
    if np.sum(in_blue) < 10:
        return None

    is_fit_field = reference_b_mg is None
    if reference_b_mg is None:
        reference_b_mg = result.get('B_MG', np.nan)
    if not np.isfinite(reference_b_mg):
        reference_b_mg = 220.0
    reference_y = float(reference_b_mg) / 100.0

    templates = load_balmer_zeeman_templates(
        template_dir=template_dir or result.get('template_dir'),
        interpolated=True, b_max_mg=1000.0,
        wavelength_min_A=2500.0, wavelength_max_A=12000.0)
    nodes = balmer_blue_region_node_table(
        result, template_dir=template_dir,
        reference_b_mg=None if is_fit_field else reference_b_mg,
        xlim=xlim)
    if reference_wavelengths is not None:
        manual_nodes = pd.DataFrame({
            'series': 'manual',
            'component': np.arange(1, len(reference_wavelengths) + 1),
            'label': [f'manual {i}' for i in range(1, len(reference_wavelengths) + 1)],
            'B_node_MG': float(reference_b_mg),
            'node_wavelength_A': reference_wavelengths,
            'detected_wavelength_A': np.nan,
            'delta_detected_minus_model_A': np.nan,
            'detected': True,
            'usable': True,
            'depth': np.nan,
            'snr': np.nan,
            'sigma_A': 12.0,
            'skip_reason': '',
        })
        nodes = pd.concat([nodes, manual_nodes], ignore_index=True)
    node_series = set(nodes['series']) if not nodes.empty else set()
    plot_series = [
        ser for ser in ['Hbeta', 'Hgamma', 'Halpha']
        if ser in node_series
    ]
    if not plot_series:
        plot_series = ['Halpha'] if xlo >= 5400 else ['Hbeta']
    curve_styles = {
        'Hbeta': {'color': 'black', 'ls': '-', 'alpha': 0.74},
        'Hgamma': {'color': '0.35', 'ls': '--', 'alpha': 0.80},
        'Halpha': {'color': '0.55', 'ls': ':', 'alpha': 0.65},
    }
    rv_shift = 1.0 + float(result.get('rv_kms', 0.0) or 0.0) / C_KMS

    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    }):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 10), sharex=True,
            gridspec_kw={'hspace': 0, 'height_ratios': [1.05, 1.0]})

        flux_plot = flux * flux_scale
        ax1.plot(wave, flux_plot, c='k', lw=0.7)
        ax1.set_ylabel(
            r"$F_{\lambda}$ (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)",
            fontsize=20)
        ylims = _robust_axis_limits(flux_plot[in_blue], (0.5, 99.5), 0.10)
        if ylims is not None:
            ax1.set_ylim(*ylims)

        if templates is not None and not templates.empty:
            for ser in plot_series:
                series_df = templates[templates['series'] == ser]
                style = curve_styles.get(
                    ser, {'color': '0.45', 'ls': '-', 'alpha': 0.65})
                labelled = False
                for component, df in series_df.groupby('component'):
                    shifted_wave = df['wavelength_A'].to_numpy(float) * rv_shift
                    if not np.any((shifted_wave >= xlo) & (shifted_wave <= xhi)):
                        continue
                    label_curve = SERIES_LABELS.get(ser, ser) if not labelled else None
                    labelled = True
                    order = np.argsort(df['B_MG'].to_numpy(float))
                    ax2.plot(shifted_wave[order],
                             df['B_MG'].to_numpy(float)[order] / 100.0,
                             color=style['color'], linestyle=style['ls'],
                             alpha=style['alpha'], lw=1.0, label=label_curve)

        if not nodes.empty:
            for i, (_, row) in enumerate(nodes.iterrows()):
                wl_line = float(row['node_wavelength_A'])
                detected = bool(row.get('detected', False))
                minor = bool(row.get('minor_absorption_support', False))
                usable = bool(row.get('usable', False))
                if detected:
                    line_color = 'g'
                    marker = dict(marker='o', s=72, facecolor='r',
                                  edgecolor='r', linewidth=0.8, zorder=6)
                    ls = '--'
                    alpha = 0.78
                elif minor:
                    line_color = 'tab:blue'
                    marker = dict(marker='o', s=52, facecolor='tab:blue',
                                  edgecolor='white', linewidth=0.8, zorder=5)
                    ls = ':'
                    alpha = 0.55
                elif usable:
                    line_color = 'tab:orange'
                    marker = dict(marker='o', s=58, facecolor='white',
                                  edgecolor='tab:orange', linewidth=1.3, zorder=5)
                    ls = ':'
                    alpha = 0.55
                else:
                    line_color = '0.55'
                    marker = dict(marker='x', s=54, color='0.45',
                                  linewidth=1.2, zorder=4)
                    ls = ':'
                    alpha = 0.35
                detected_wl = row.get('detected_wavelength_A', np.nan)
                top_center = wl_line if minor else (
                    detected_wl if np.isfinite(detected_wl) else wl_line)
                if minor:
                    anchored_width = row.get('anchored_notch_width_A', np.nan)
                    half_range = (
                        max(0.5 * float(anchored_width), 1.2)
                        if np.isfinite(anchored_width) and anchored_width > 0
                        else 2.0)
                    half_range = min(half_range, 3.5)
                else:
                    sigma_A = row.get('sigma_A', np.nan)
                    if not np.isfinite(sigma_A) or sigma_A <= 0:
                        sigma_A = 14.0 if detected else 20.0
                    half_range = min(max(0.35 * float(sigma_A), 5.0), 22.0)
                if detected:
                    ax1.axvspan(top_center - half_range, top_center + half_range,
                                color='tab:green', alpha=0.08, lw=0)
                    ax1.vlines(top_center, 0.90, 0.995,
                               transform=ax1.get_xaxis_transform(),
                               color='tab:green', linestyle='-', lw=1.1,
                               alpha=0.85)
                elif minor:
                    ax1.axvspan(top_center - half_range, top_center + half_range,
                                color='tab:blue', alpha=0.055, lw=0)
                    ax1.vlines(top_center, 0.93, 0.995,
                               transform=ax1.get_xaxis_transform(),
                               color='tab:blue', linestyle='-', lw=0.9,
                               alpha=0.62)
                else:
                    ax1.axvspan(top_center - half_range, top_center + half_range,
                                color=line_color, alpha=0.045, lw=0)
                    ax1.vlines(top_center, 0.93, 0.995,
                               transform=ax1.get_xaxis_transform(),
                               color=line_color, linestyle='-', lw=0.9,
                               alpha=0.55)
                ax2.axvline(wl_line, color=line_color, linestyle=ls,
                            lw=0.8, alpha=alpha)
                ax2.scatter([wl_line], [reference_y], **marker)
                label_y = reference_y + 0.12 + 0.12 * (i % 3)
                ax2.text(wl_line, label_y, str(row.get('label', '')),
                         rotation=90, fontsize=8,
                         color='r' if detected else 'tab:blue' if minor else '0.35',
                         ha='center', va='bottom')

        elif reference_wavelengths:
            for wl_line in reference_wavelengths:
                ax1.axvline(wl_line, color='g', linestyle='--', lw=0.8, alpha=0.7)
                ax2.axvline(wl_line, color='g', linestyle='--', lw=0.8, alpha=0.7)
                ax2.plot(wl_line, reference_y, 'o', color='r', markersize=7.5)

        if is_fit_field:
            label = _field_label(result, reference_b_mg)
        else:
            label = _field_label(result, reference_b_mg, reference=True)
        ax2.axhline(y=reference_y, color='r', linestyle='--',
                    lw=2, alpha=0.8, label=label)
        ax2.set_ylabel('B (100 MG)', fontsize=20)
        ax2.set_xlabel(r"$\lambda\,(\AA)$", fontsize=20)
        ax2.set_ylim(0, 9)
        ax2.set_xlim(xlo, xhi)
        ax2.legend(loc='upper right', frameon=False, fontsize=13)

        if title:
            ax1.set_title(title, fontsize=15, pad=8)
        for ax in (ax1, ax2):
            ax.tick_params(labelsize=14, top=True, right=True)

        fig.tight_layout()
        if save_path:
            utils.ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        return fig


def plot_balmer_red_region_interpolation(result, save_path=None, template_dir=None,
                                         reference_b_mg=None, dpi=300):
    """Plot the H-alpha red-region Zeeman diagnostic."""
    return plot_balmer_blue_region_interpolation(
        result, save_path=save_path, template_dir=template_dir,
        reference_b_mg=reference_b_mg, xlim=(5500.0, 9000.0),
        title='H-alpha Zeeman red-region diagnostic', dpi=dpi)


def plot_balmer_full_field_overlay(result, save_path=None, template_dir=None,
                                   reference_b_mg=None,
                                   series_focus=('Halpha', 'Hbeta'),
                                   xlim=None, b_ylim=None,
                                   feature_half_width_A=40.0,
                                   dpi=300):
    """Plot the full spectrum directly on the Balmer Zeeman B-lambda plane."""
    if result is None:
        return None
    wave = np.asarray(result.get('wave'), dtype=float)
    flux = np.asarray(result.get('flux'), dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if np.sum(good) < 10:
        return None
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    b_mg = result.get('B_MG', np.nan) if reference_b_mg is None else reference_b_mg
    if not np.isfinite(b_mg):
        return None
    if b_ylim is None:
        margin = 5.0 if float(b_mg) < 50.0 else 15.0
        b_hi = max(30.0, np.ceil((float(b_mg) + margin) / 5.0) * 5.0)
        b_ylim = (0.0, float(b_hi))
    used = [
        {'ha': 'Halpha', 'halpha': 'Halpha',
         'hb': 'Hbeta', 'hbeta': 'Hbeta',
         'hg': 'Hgamma', 'hgamma': 'Hgamma'}.get(str(s).lower(), str(s))
        for s in series_focus
    ]

    templates = load_balmer_zeeman_templates(
        template_dir=template_dir or result.get('template_dir'),
        interpolated=True, b_max_mg=max(float(b_ylim[1]), float(b_mg) + 50.0),
        wavelength_min_A=2500.0, wavelength_max_A=12000.0)
    if templates.empty:
        return None
    curve_x = []
    for ser in used:
        sdf = templates[
            (templates['series'] == ser)
            & (templates['B_MG'] >= float(b_ylim[0]))
            & (templates['B_MG'] <= float(b_ylim[1]))
        ]
        curve_x.extend(sdf['wavelength_A'].to_numpy(float))
    curve_x = np.asarray(curve_x, dtype=float)
    curve_x = curve_x[np.isfinite(curve_x)]
    if xlim is None:
        wave_min = float(np.nanmin(wave))
        wave_max = float(np.nanmax(wave))
        curve_min = float(np.nanmin(curve_x)) if curve_x.size else wave_min
        curve_max = float(np.nanmax(curve_x)) if curve_x.size else wave_max
        xlo = min(wave_min - 300.0, curve_min - 120.0)
        xhi = max(wave_max + 300.0, curve_max + 120.0)
        xlim = (max(3000.0, xlo), min(10000.0, xhi))
    xlo, xhi = float(xlim[0]), float(xlim[1])

    features = balmer_full_field_overlay_feature_table(
        result, template_dir=template_dir, reference_b_mg=reference_b_mg,
        series_focus=used, xlim=xlim,
        feature_half_width_A=feature_half_width_A)

    styles = {
        'Halpha': {'color': '#f05a78', 'lw': 4.2, 'alpha': 0.22,
                   'label': 'H-alpha'},
        'Hbeta': {'color': '#6ec6df', 'lw': 1.9, 'alpha': 0.34,
                  'label': 'H-beta'},
        'Hgamma': {'color': '#5ca96b', 'lw': 1.8, 'alpha': 0.30,
                   'label': 'H-gamma'},
    }
    with plt.rc_context({
        'font.family': 'Times New Roman',
        'font.size': 15,
        'axes.labelsize': 17,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    }):
        fig, ax = plt.subplots(figsize=(8.5, 6.2))
        for ser in used:
            sdf = templates[
                (templates['series'] == ser)
                & (templates['B_MG'] >= float(b_ylim[0]))
                & (templates['B_MG'] <= float(b_ylim[1]))
            ]
            style = styles.get(
                ser, {'color': '0.4', 'lw': 2.0, 'alpha': 0.45, 'label': ser})
            labelled = False
            for _, df in sdf.groupby('component'):
                df = df.sort_values('B_MG')
                wl = df['wavelength_A'].to_numpy(float)
                b = df['B_MG'].to_numpy(float)
                keep = (
                    np.isfinite(wl) & np.isfinite(b)
                    & (wl >= xlo - 300.0) & (wl <= xhi + 300.0)
                )
                if np.sum(keep) < 2:
                    continue
                label = style['label'] if not labelled else None
                labelled = True
                ax.plot(wl[keep], b[keep], color=style['color'],
                        lw=style['lw'], alpha=style['alpha'],
                        solid_capstyle='round', label=label, zorder=0.5)

        ax.axhline(float(b_mg), color='black', linestyle='--',
                   lw=1.3, alpha=0.9, zorder=2)
        in_plot = (wave >= xlo) & (wave <= xhi)
        flux_y = _scale_flux_to_b_axis(flux[in_plot], b_ylim)
        ax.plot(wave[in_plot], flux_y, color='0.45', lw=1.25,
                alpha=0.92, zorder=3, label='spectrum')

        if features is not None and not features.empty:
            for _, row in features.iterrows():
                pred = float(row.get('predicted_wavelength_A', np.nan))
                has_feature = bool(row.get('has_flux_feature', False))
                if has_feature:
                    wl_mark = float(row.get('feature_wavelength_A'))
                    left_A, right_A, center_A = _feature_span_from_row(
                        row, fallback_half_width_A=4.0)
                    _draw_feature_span(
                        ax, left_A, right_A, center_A, color='0.20',
                        band_alpha=0.07, line_alpha=0.18,
                        line_lw=1.45, zorder=0.75)
                    ax.plot(wl_mark, float(b_mg), 'o', color='red',
                            markersize=6.0, markeredgecolor='white',
                            markeredgewidth=0.6, zorder=5)

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(float(b_ylim[0]), float(b_ylim[1]))
        ax.set_xlabel(r'$\lambda\,(\AA)$')
        ax.set_ylabel(r'$B,\,\mathrm{MG}$')
        ax.tick_params(top=True, right=True)
        ax.legend(loc='upper right', frameon=False, fontsize=11)
        fig.tight_layout()
        if save_path:
            utils.ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        return fig


def _paper_flux_scale(flux):
    finite = np.asarray(flux, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0, 'Flux'
    scale = np.nanmedian(np.abs(finite))
    if np.isfinite(scale) and 0 < scale < 1e-10:
        return 1e16, r"$F_{\lambda}$ (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"
    return 1.0, 'Flux'


def _scale_flux_to_b_axis(flux, b_ylim, percentiles=(1.0, 99.0),
                          occupied_fraction=(0.50, 0.78)):
    flux = np.asarray(flux, dtype=float)
    out = np.full_like(flux, np.nan, dtype=float)
    finite = np.isfinite(flux)
    if np.sum(finite) < 2:
        return out
    lo, hi = np.nanpercentile(flux[finite], percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo, hi = np.nanmin(flux[finite]), np.nanmax(flux[finite])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return out
    clipped = np.clip(flux, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    y0 = float(b_ylim[0]) + occupied_fraction[0] * (float(b_ylim[1]) - float(b_ylim[0]))
    y1 = float(b_ylim[0]) + occupied_fraction[1] * (float(b_ylim[1]) - float(b_ylim[0]))
    out[finite] = y0 + scaled[finite] * (y1 - y0)
    return out


def _feature_signal_arrays(result):
    wave = np.asarray(result.get('wave'), dtype=float)
    flux = np.asarray(result.get('flux'), dtype=float)
    signal = result.get('norm_flux')
    if signal is None:
        signal = flux
        method = 'raw_flux'
    else:
        signal = np.asarray(signal, dtype=float)
        if signal.shape != wave.shape:
            signal = flux
            method = 'raw_flux'
        else:
            method = 'wd_template_residual'
    good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(signal)
    if np.sum(good) < 10:
        return np.array([]), np.array([]), method
    order = np.argsort(wave[good])
    return wave[good][order], signal[good][order], method


def _standard_absorption_avoid_wavelengths(rv_kms=0.0):
    scale = 1.0 + float(rv_kms or 0.0) / C_KMS
    return [
        (name, float(wl) * scale)
        for name, wl in STANDARD_ABSORPTION_CORES.items()
    ]


def _feature_span_edges(local_wave, mask, center_idx, half_width_A,
                        min_width_A=8.0):
    local_wave = np.asarray(local_wave, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = len(local_wave)
    if n == 0 or center_idx < 0 or center_idx >= n:
        return np.nan, np.nan, np.nan
    diffs = np.diff(local_wave)
    step = np.nanmedian(diffs[np.isfinite(diffs) & (diffs > 0)])
    if not np.isfinite(step) or step <= 0:
        step = 1.0
    gap_tol = max(6.0, 4.0 * step)

    left_i = int(center_idx)
    right_i = int(center_idx)
    while left_i > 0 and mask[left_i - 1]:
        if local_wave[left_i] - local_wave[left_i - 1] > gap_tol:
            break
        left_i -= 1
    while right_i < n - 1 and mask[right_i + 1]:
        if local_wave[right_i + 1] - local_wave[right_i] > gap_tol:
            break
        right_i += 1

    left = float(local_wave[left_i] - 0.5 * step)
    right = float(local_wave[right_i] + 0.5 * step)
    center = float(local_wave[center_idx])
    min_width_A = min(float(min_width_A), 1.6 * float(half_width_A))
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        left = center - 0.5 * min_width_A
        right = center + 0.5 * min_width_A
    if right - left < min_width_A:
        left = center - 0.5 * min_width_A
        right = center + 0.5 * min_width_A

    lo = float(np.nanmin(local_wave))
    hi = float(np.nanmax(local_wave))
    left = max(lo, center - float(half_width_A), left)
    right = min(hi, center + float(half_width_A), right)
    if right <= left:
        left = max(lo, center - 0.5 * min_width_A)
        right = min(hi, center + 0.5 * min_width_A)
    return float(left), float(right), float(max(0.0, right - left))


def _strong_flux_feature_near(wave, flux, center, half_width_A=18.0,
                              min_points=5, avoid_wavelengths=None,
                              avoid_half_width_A=14.0,
                              min_feature_width_A=8.0):
    """Find the clearest local flux edge/change near a predicted Zeeman line."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    center = float(center)
    good = (
        np.isfinite(wave) & np.isfinite(flux)
        & (wave >= center - float(half_width_A))
        & (wave <= center + float(half_width_A))
    )
    avoided = []
    for item in avoid_wavelengths or []:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            name, wl = item[0], float(item[1])
        else:
            name, wl = '', float(item)
        near = np.abs(wave - wl) <= float(avoid_half_width_A)
        if np.any(good & near):
            avoided.append(str(name))
        good &= ~near
    if np.sum(good) < min_points:
        return None
    local_wave = wave[good]
    local_flux = flux[good]
    order = np.argsort(local_wave)
    local_wave = local_wave[order]
    local_flux = local_flux[order]
    n = len(local_wave)
    size = min(9, max(3, n // 5 * 2 + 1))
    if size >= n:
        size = n - 1 if n % 2 == 0 else n
    if size < 3:
        smooth = local_flux
    else:
        if size % 2 == 0:
            size += 1
        smooth = median_filter(local_flux, size=size, mode='nearest')
    grad = np.gradient(smooth, local_wave)
    finite = np.isfinite(grad)
    if not np.any(finite):
        return None
    abs_grad = np.abs(grad)

    baseline = float(np.nanmedian(smooth[np.isfinite(smooth)]))
    resid = smooth - baseline
    resid_noise = _robust_noise(resid, fallback=0.01)
    abs_resid = np.abs(resid)
    if n > 2:
        abs_resid[[0, -1]] = -np.inf
    central = np.abs(local_wave - center) <= max(8.0, 0.75 * float(half_width_A))
    central &= np.isfinite(abs_grad) & np.isfinite(abs_resid)
    if np.sum(central) < min_points:
        return None
    abs_grad_pick = abs_grad.copy()
    abs_resid_pick = abs_resid.copy()
    abs_grad_pick[~central] = -np.inf
    abs_resid_pick[~central] = -np.inf
    if not np.isfinite(np.nanmax(abs_grad_pick)):
        return None
    idx_grad = int(np.nanargmax(abs_grad_pick))
    idx_anom = int(np.nanargmax(abs_resid_pick))
    anom_amp = float(abs(resid[idx_anom])) if np.isfinite(resid[idx_anom]) else 0.0
    if anom_amp >= max(2.0 * resid_noise, 0.008):
        idx = idx_anom
        direction = -1.0 if resid[idx] < 0 else 1.0
        signed_resid = direction * resid
        threshold = max(0.35 * abs(float(resid[idx])),
                        1.5 * resid_noise, 0.004)
        span_mask = np.isfinite(signed_resid) & (signed_resid >= threshold)
        feature_kind = 'trough' if direction < 0 else 'hump'
    else:
        idx = idx_grad
        grad_noise = _robust_noise(abs_grad, fallback=0.003)
        threshold = max(0.35 * float(abs_grad[idx]), 2.0 * grad_noise)
        span_mask = np.isfinite(abs_grad) & (abs_grad >= threshold)
        feature_kind = 'edge'

    if not span_mask[idx]:
        span_mask[idx] = True
    min_feature_width_A = max(float(min_feature_width_A),
                              0.50 * float(half_width_A))
    left_A, right_A, width_A = _feature_span_edges(
        local_wave, span_mask, idx, half_width_A,
        min_width_A=min_feature_width_A)

    noise = _robust_noise(np.diff(smooth), fallback=0.01)
    jump_strength = float(abs(grad[idx]) * np.nanmedian(np.diff(local_wave)))
    jump_snr = jump_strength / max(noise, 1e-8)
    return {
        'feature_wavelength_A': float(local_wave[idx]),
        'feature_left_A': left_A,
        'feature_right_A': right_A,
        'feature_width_A': width_A,
        'feature_kind': feature_kind,
        'feature_flux': float(local_flux[idx]),
        'feature_smooth_flux': float(smooth[idx]),
        'feature_local_baseline': baseline,
        'feature_residual': float(resid[idx]) if np.isfinite(resid[idx]) else np.nan,
        'feature_jump_strength': jump_strength,
        'feature_jump_snr': float(jump_snr),
        'feature_delta_from_pred_A': float(local_wave[idx] - center),
        'feature_avoided_absorption': ';'.join(sorted(set(avoided))),
    }


def _feature_span_from_row(row, fallback_half_width_A=4.0):
    minor = bool(row.get('minor_absorption_support', False))
    if minor:
        center = float(row.get(
            'predicted_wavelength_A',
            row.get('wavelength_A', row.get('feature_wavelength_A', np.nan))))
        left = float(row.get('anchored_notch_left_A', np.nan))
        right = float(row.get('anchored_notch_right_A', np.nan))
        if not (np.isfinite(left) and np.isfinite(right) and right > left):
            half = max(1.2, min(3.0, 0.5 * float(fallback_half_width_A)))
            left = center - half
            right = center + half
        return float(left), float(right), float(center)
    center = float(row.get('feature_wavelength_A', row.get('wavelength_A', np.nan)))
    left = float(row.get('feature_left_A', np.nan))
    right = float(row.get('feature_right_A', np.nan))
    if not np.isfinite(center):
        return np.nan, np.nan, np.nan
    if not (np.isfinite(left) and np.isfinite(right) and right > left):
        half = float(fallback_half_width_A)
        left = center - half
        right = center + half
    return float(left), float(right), float(center)


def _draw_feature_span(ax, left, right, center, color='green',
                       band_alpha=0.10, line_alpha=0.28,
                       line_lw=1.8, zorder=1.0):
    if not (np.isfinite(left) and np.isfinite(right)
            and np.isfinite(center) and right > left):
        return
    ax.axvspan(left, right, facecolor=color, edgecolor='none',
               alpha=band_alpha, zorder=zorder)
    edge_alpha = max(0.06, 0.65 * float(line_alpha))
    for bound in (left, right):
        ax.axvline(bound, color=color, linestyle='-', lw=max(0.8, 0.65 * line_lw),
                   alpha=edge_alpha, zorder=zorder + 0.02)
    ax.axvline(center, color=color, linestyle='--', lw=float(line_lw),
               alpha=float(line_alpha), zorder=zorder + 0.04)


def balmer_full_field_overlay_feature_table(result, template_dir=None,
                                            reference_b_mg=None,
                                            series_focus=('Halpha', 'Hbeta'),
                                            xlim=None,
                                            feature_half_width_A=40.0):
    if result is None:
        return pd.DataFrame()
    wave, signal, signal_method = _feature_signal_arrays(result)
    if wave.size < 10:
        return pd.DataFrame()
    b_mg = result.get('B_MG', np.nan) if reference_b_mg is None else reference_b_mg
    if not np.isfinite(b_mg):
        return pd.DataFrame()
    used = [
        {'ha': 'Halpha', 'halpha': 'Halpha',
         'hb': 'Hbeta', 'hbeta': 'Hbeta'}.get(str(s).lower(), str(s))
        for s in series_focus
    ]
    templates = load_balmer_zeeman_templates(
        template_dir=template_dir or result.get('template_dir'),
        interpolated=True, b_max_mg=max(float(b_mg) + 50.0, 1000.0))
    nodes = _component_wavelengths(templates, float(b_mg), series=used)
    rv_shift = 1.0 + float(result.get('rv_kms', 0.0) or 0.0) / C_KMS
    avoid_wavelengths = _standard_absorption_avoid_wavelengths(
        result.get('rv_kms', 0.0) or 0.0)
    rows = []
    for row in nodes:
        pred = float(row['wavelength_A']) * rv_shift
        if xlim is not None and not (float(xlim[0]) <= pred <= float(xlim[1])):
            continue
        feature = _strong_flux_feature_near(
            wave, signal, pred, half_width_A=feature_half_width_A,
            avoid_wavelengths=avoid_wavelengths)
        out = {
            'series': row['series'],
            'component': int(row['component']),
            'label': _series_component_label(row['series'], row['component']),
            'B_MG': float(b_mg),
            'predicted_wavelength_A': pred,
            'feature_signal_method': signal_method,
            'has_flux_feature': feature is not None,
        }
        if feature is not None:
            out.update(feature)
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ['series', 'component']).reset_index(drop=True)


def _region_reference_rows(result, series_focus, xlim, reference_b_mg=None,
                           template_dir=None, feature_half_width_A=40.0):
    xlo, xhi = float(xlim[0]), float(xlim[1])
    signal_wave, signal, signal_method = _feature_signal_arrays(result)
    avoid_wavelengths = _standard_absorption_avoid_wavelengths(
        result.get('rv_kms', 0.0) or 0.0)
    comp = result.get('component_table') if result is not None else None
    rows = []
    if isinstance(comp, pd.DataFrame) and not comp.empty:
        subset = comp[comp['series'] == series_focus].copy()
        for _, row in subset.iterrows():
            pred = row.get('predicted_wavelength_A', np.nan)
            if not np.isfinite(pred) or pred < xlo or pred > xhi:
                continue
            skip_reason = str(row.get('skip_reason', '') or '')
            if 'standard_absorption_core' in skip_reason:
                continue
            detected = bool(row.get('detected', False))
            minor = bool(row.get('minor_absorption_support', False))
            usable = bool(row.get('usable', False))
            detected_wl = row.get('detected_wavelength_A', np.nan)
            if minor:
                width_A = row.get('anchored_notch_width_A', np.nan)
                center = float(pred)
            else:
                width_A = row.get('trough_width_A', np.nan)
                center = (
                    float(detected_wl) if np.isfinite(detected_wl)
                    else float(pred))
            if minor:
                left_A = row.get('anchored_notch_left_A', np.nan)
                right_A = row.get('anchored_notch_right_A', np.nan)
                if (np.isfinite(left_A) and np.isfinite(right_A)
                        and float(right_A) > float(left_A)):
                    feature_left_A = float(left_A)
                    feature_right_A = float(right_A)
                    feature_width_A = float(right_A) - float(left_A)
                else:
                    half_width = (
                        max(0.5 * float(width_A), 1.2)
                        if np.isfinite(width_A) and float(width_A) > 0
                        else 2.0)
                    feature_left_A = center - half_width
                    feature_right_A = center + half_width
                    feature_width_A = 2.0 * half_width
            elif np.isfinite(width_A) and float(width_A) > 0:
                half_width = max(0.5 * float(width_A), 1.5)
                feature_left_A = center - half_width
                feature_right_A = center + half_width
                feature_width_A = 2.0 * half_width
            else:
                half_width = 4.0 if detected else 2.5
                feature_left_A = center - half_width
                feature_right_A = center + half_width
                feature_width_A = 2.0 * half_width
            evidence_depth = (
                row.get('anchored_notch_depth', np.nan) if minor
                else row.get('depth', np.nan))
            evidence_snr = (
                row.get('anchored_notch_snr', np.nan) if minor
                else row.get('snr', np.nan))
            feature = {
                'feature_wavelength_A': center,
                'feature_left_A': feature_left_A,
                'feature_right_A': feature_right_A,
                'feature_width_A': feature_width_A,
                'feature_kind': (
                    'accepted_trough' if detected else
                    'minor_absorption_support' if minor else
                    'rejected_node'),
                'feature_flux': np.nan,
                'feature_smooth_flux': np.nan,
                'feature_local_baseline': np.nan,
                'feature_residual': -float(evidence_depth)
                if np.isfinite(evidence_depth) else np.nan,
                'feature_jump_strength': np.nan,
                'feature_jump_snr': evidence_snr,
                'feature_delta_from_pred_A': (
                    float(center - pred) if np.isfinite(center) else np.nan),
                'feature_avoided_absorption': '',
            }
            wl = feature['feature_wavelength_A']
            rows.append({
                'series': series_focus,
                'component': int(row.get('component')),
                'label': _series_component_label(series_focus, row.get('component')),
                'wavelength_A': float(pred),
                'predicted_wavelength_A': float(pred) if np.isfinite(pred) else np.nan,
                'detected': detected,
                'minor_absorption_support': minor,
                'evidence_class': row.get('evidence_class', ''),
                'evidence_weight': row.get('evidence_weight', np.nan),
                'evidence_depth': row.get('evidence_depth', np.nan),
                'evidence_snr': row.get('evidence_snr', np.nan),
                'anchored_notch_left_A': row.get('anchored_notch_left_A', np.nan),
                'anchored_notch_right_A': row.get('anchored_notch_right_A', np.nan),
                'anchored_notch_width_A': row.get('anchored_notch_width_A', np.nan),
                'usable': usable,
                'skip_reason': skip_reason,
                'feature_signal_method': signal_method,
                **feature,
            })
    if rows:
        return pd.DataFrame(rows)

    b_mg = result.get('B_MG', np.nan) if reference_b_mg is None else reference_b_mg
    if not np.isfinite(b_mg):
        return pd.DataFrame()
    templates = load_balmer_zeeman_templates(
        template_dir=template_dir or result.get('template_dir'),
        interpolated=True, b_max_mg=max(float(b_mg) + 50.0, 1000.0))
    nodes = _component_wavelengths(templates, float(b_mg), series=[series_focus])
    scale = 1.0 + float(result.get('rv_kms', 0.0) or 0.0) / C_KMS
    for row in nodes:
        pred = float(row['wavelength_A']) * scale
        if xlo <= pred <= xhi:
            feature = None
            if signal_wave.size >= 10:
                feature = _strong_flux_feature_near(
                    signal_wave, signal, pred,
                    half_width_A=feature_half_width_A,
                    avoid_wavelengths=avoid_wavelengths)
            if feature is None:
                continue
            rows.append({
                'series': series_focus,
                'component': int(row['component']),
                'label': _series_component_label(series_focus, row['component']),
                'wavelength_A': feature['feature_wavelength_A'],
                'predicted_wavelength_A': pred,
                'detected': True,
                'usable': True,
                'skip_reason': '',
                'feature_signal_method': signal_method,
                **feature,
            })
    return pd.DataFrame(rows)


def plot_balmer_region_interpolation_paper(result, save_path=None, template_dir=None,
                                           series_focus='Hbeta',
                                           reference_b_mg=None,
                                           xlim=None, flux_scale=None,
                                           b_max_mg=None, title=None,
                                           dpi=300):
    """
    Notebook-style Balmer Zeeman interpolation figure.

    The figure follows the cechichang.ipynb layout: observed spectrum on top,
    Zeeman component wavelength-vs-field curves below, vertical line markers,
    and red points at the adopted magnetic field.
    """
    if result is None:
        return None
    wave = np.asarray(result.get('wave'), dtype=float)
    flux = np.asarray(result.get('flux'), dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if np.sum(good) < 10:
        return None
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    series_focus = {
        'ha': 'Halpha', 'halpha': 'Halpha', 'h-alpha': 'Halpha',
        'hb': 'Hbeta', 'hbeta': 'Hbeta', 'h-beta': 'Hbeta',
    }.get(str(series_focus).lower(), series_focus)
    if xlim is None:
        xlim = (5500.0, 9000.0) if series_focus == 'Halpha' else (3250.0, 5500.0)
    xlo = float(xlim[0])
    xhi = float(xlim[1])
    if xhi - xlo < 80.0:
        return None
    in_region = (wave >= xlo) & (wave <= xhi)
    if np.sum(in_region) < 10:
        return None

    b_mg = result.get('B_MG', np.nan) if reference_b_mg is None else reference_b_mg
    if not np.isfinite(b_mg):
        return None
    if b_max_mg is None:
        if float(b_mg) < 100.0:
            margin = 5.0 if float(b_mg) < 50.0 else 15.0
            b_max_mg = max(30.0, np.ceil((float(b_mg) + margin) / 5.0) * 5.0)
        else:
            b_max_mg = min(
                950.0, max(300.0, np.ceil((float(b_mg) + 50.0) / 50.0) * 50.0))

    if flux_scale is None:
        flux_scale, flux_label = _paper_flux_scale(flux[in_region])
    else:
        flux_label = (
            r"$F_{\lambda}$ (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"
            if np.isclose(float(flux_scale), 1e16) else 'Flux')

    templates = load_balmer_zeeman_templates(
        template_dir=template_dir or result.get('template_dir'),
        interpolated=True, b_max_mg=max(float(b_max_mg), float(b_mg) + 20.0),
        wavelength_min_A=2500.0, wavelength_max_A=12000.0)
    if templates.empty:
        return None
    nodes = _region_reference_rows(
        result, series_focus, (xlo, xhi), reference_b_mg=reference_b_mg,
        template_dir=template_dir)

    color = {'Halpha': 'black', 'Hbeta': 'black', 'Hgamma': '0.35'}.get(
        series_focus, 'black')
    line_label = SERIES_LABELS.get(series_focus, series_focus)
    with plt.rc_context({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.3,
        'lines.markersize': 6,
    }):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8.5, 8.8), sharex=True,
            gridspec_kw={'height_ratios': [1.0, 1.35], 'hspace': 0.05})

        ax1.plot(wave[in_region], flux[in_region] * float(flux_scale),
                 c='k', lw=0.9)
        ax1.set_ylabel(flux_label, fontweight='bold')
        ylims = _robust_axis_limits(flux[in_region] * float(flux_scale),
                                    (0.5, 99.5), 0.08)
        if ylims is not None:
            ax1.set_ylim(*ylims)
        ax1.grid(True, alpha=0.25, linestyle='--')

        plotted_label = False
        series_df = templates[templates['series'] == series_focus]
        for component, df in series_df.groupby('component'):
            df = df.sort_values('B_MG')
            b = df['B_MG'].to_numpy(float)
            wl = df['wavelength_A'].to_numpy(float)
            keep = (
                np.isfinite(b) & np.isfinite(wl)
                & (b <= float(b_max_mg))
                & (wl >= xlo - 250.0) & (wl <= xhi + 250.0)
            )
            if np.sum(keep) < 2:
                continue
            label = line_label if not plotted_label else None
            plotted_label = True
            ax2.plot(wl[keep], b[keep], color=color, alpha=0.74, lw=1.2,
                     label=label)

        has_detected_nodes = False
        has_minor_nodes = False
        has_rejected_nodes = False
        if not nodes.empty:
            for _, row in nodes.iterrows():
                wl_line = float(row['wavelength_A'])
                detected = bool(row.get('detected', False))
                minor = bool(row.get('minor_absorption_support', False))
                usable = bool(row.get('usable', False))
                if detected:
                    line_color = 'green'
                    alpha = 0.75
                    has_detected_nodes = True
                elif minor:
                    line_color = 'tab:blue'
                    alpha = 0.48
                    has_minor_nodes = True
                elif usable:
                    line_color = 'tab:orange'
                    alpha = 0.45
                    has_rejected_nodes = True
                else:
                    line_color = '0.55'
                    alpha = 0.30
                left_A, right_A, center_A = _feature_span_from_row(
                    row, fallback_half_width_A=4.0)
                for ax in (ax1, ax2):
                    _draw_feature_span(
                        ax, left_A, right_A, center_A, color=line_color,
                        band_alpha=0.10 if detected else 0.07 if minor else 0.04,
                        line_alpha=min(0.40, alpha * 0.45),
                        line_lw=1.65 if detected else 1.0,
                        zorder=1.0)
                if detected:
                    ax2.plot(wl_line, float(b_mg), 'o', color='red',
                             markersize=5.8, markeredgecolor='white',
                             markeredgewidth=0.5, zorder=5)
                    ax2.text(wl_line, float(b_mg) + 0.04 * float(b_max_mg),
                             str(row.get('label', '')), fontsize=8,
                             rotation=90, color='red', ha='center',
                             va='bottom')
                elif minor:
                    ax2.plot(wl_line, float(b_mg), 'o', color='tab:blue',
                             markersize=4.3, markeredgecolor='white',
                             markeredgewidth=0.45, zorder=4)
        if has_detected_nodes:
            ax2.plot([], [], color='green', linestyle='--',
                     lw=1.4, label='accepted broad trough')
        if has_minor_nodes:
            ax2.plot([], [], color='tab:blue', linestyle='--',
                     lw=1.2, label='low-weight minor absorption')
        if has_rejected_nodes:
            ax2.plot([], [], color='tab:orange', linestyle='--',
                     lw=1.2, label='usable/rejected node')

        b_label = _field_label(result, b_mg, reference=reference_b_mg is not None)
        ax2.axhline(y=float(b_mg), color='red', linestyle='--',
                    lw=1.6, alpha=0.85, label=b_label)
        ax2.set_ylabel('Magnetic field B (MG)', fontweight='bold')
        ax2.set_xlabel(r"Wavelength ($\AA$)", fontweight='bold')
        ax2.set_xlim(xlo, xhi)
        ax2.set_ylim(0.0, float(b_max_mg))
        ax2.grid(True, alpha=0.25, linestyle='--')
        ax2.legend(loc='upper right', frameon=False)
        if title:
            ax1.set_title(title, pad=6)

        fig.tight_layout()
        if save_path:
            utils.ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        return fig


def plot_balmer_blue_red_region_interpolation(result, save_path=None,
                                              template_dir=None,
                                              reference_b_mg=None,
                                              blue_xlim=(3250.0, 5500.0),
                                              red_xlim=(5500.0, 9000.0),
                                              flux_scale=None,
                                              b_max_mg=None,
                                              vline_lw=1.7,
                                              vline_alpha=0.34,
                                              dpi=300):
    """Combined notebook-style H-beta blue and H-alpha red interpolation plot."""
    if result is None:
        return None
    wave = np.asarray(result.get('wave'), dtype=float)
    flux = np.asarray(result.get('flux'), dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    if np.sum(good) < 10:
        return None
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    b_mg = result.get('B_MG', np.nan) if reference_b_mg is None else reference_b_mg
    if not np.isfinite(b_mg):
        return None
    if b_max_mg is None:
        if float(b_mg) < 100.0:
            margin = 5.0 if float(b_mg) < 50.0 else 15.0
            b_max_mg = max(30.0, np.ceil((float(b_mg) + margin) / 5.0) * 5.0)
        else:
            b_max_mg = min(
                950.0, max(300.0, np.ceil((float(b_mg) + 50.0) / 50.0) * 50.0))

    templates = load_balmer_zeeman_templates(
        template_dir=template_dir or result.get('template_dir'),
        interpolated=True, b_max_mg=max(float(b_max_mg), float(b_mg) + 20.0),
        wavelength_min_A=2500.0, wavelength_max_A=12000.0)
    if templates.empty:
        return None

    def _draw_region(ax_flux, ax_field, series_focus, xlim, title):
        xlo, xhi = float(xlim[0]), float(xlim[1])
        in_region = (wave >= xlo) & (wave <= xhi)
        if np.sum(in_region) >= 2:
            if flux_scale is None:
                scale, flux_label = _paper_flux_scale(flux[in_region])
            else:
                scale = float(flux_scale)
                flux_label = (
                    r"$F_{\lambda}$ (10$^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)"
                    if np.isclose(scale, 1e16) else 'Flux')
            y = flux[in_region] * scale
            ax_flux.plot(wave[in_region], y, c='k', lw=0.85)
            ylims = _robust_axis_limits(y, (0.5, 99.5), 0.08)
            if ylims is not None:
                ax_flux.set_ylim(*ylims)
        else:
            flux_label = 'Flux'

        nodes = _region_reference_rows(
            result, series_focus, (xlo, xhi), reference_b_mg=reference_b_mg,
            template_dir=template_dir)
        has_detected_nodes = False
        has_minor_nodes = False
        has_rejected_nodes = False
        if not nodes.empty:
            for _, row in nodes.iterrows():
                wl_line = float(row['wavelength_A'])
                detected = bool(row.get('detected', False))
                minor = bool(row.get('minor_absorption_support', False))
                usable = bool(row.get('usable', False))
                if detected:
                    color = 'green'
                    alpha = float(vline_alpha)
                    has_detected_nodes = True
                elif minor:
                    color = 'tab:blue'
                    alpha = 0.26
                    has_minor_nodes = True
                elif usable:
                    color = 'tab:orange'
                    alpha = 0.22
                    has_rejected_nodes = True
                else:
                    color = '0.50'
                    alpha = 0.16
                left_A, right_A, center_A = _feature_span_from_row(
                    row, fallback_half_width_A=4.0)
                for ax in (ax_flux, ax_field):
                    _draw_feature_span(
                        ax, left_A, right_A, center_A, color=color,
                        band_alpha=0.11 if detected else 0.07 if minor else 0.04,
                        line_alpha=alpha,
                        line_lw=float(vline_lw), zorder=1)
                if detected:
                    ax_field.plot(wl_line, float(b_mg), 'o', color='red',
                                  markersize=5.6, markeredgecolor='white',
                                  markeredgewidth=0.5, zorder=5)
                elif minor:
                    ax_field.plot(wl_line, float(b_mg), 'o', color='tab:blue',
                                  markersize=4.1, markeredgecolor='white',
                                  markeredgewidth=0.45, zorder=4)
        if has_detected_nodes:
            ax_field.plot([], [], color='green', linestyle='--',
                          lw=1.4, label='accepted broad trough')
        if has_minor_nodes:
            ax_field.plot([], [], color='tab:blue', linestyle='--',
                          lw=1.2, label='low-weight minor absorption')
        if has_rejected_nodes:
            ax_field.plot([], [], color='tab:orange', linestyle='--',
                          lw=1.2, label='usable/rejected node')

        series_df = templates[templates['series'] == series_focus]
        label = SERIES_LABELS.get(series_focus, series_focus)
        labelled = False
        for _, df in series_df.groupby('component'):
            df = df.sort_values('B_MG')
            b = df['B_MG'].to_numpy(float)
            wl = df['wavelength_A'].to_numpy(float)
            keep = (
                np.isfinite(b) & np.isfinite(wl)
                & (b <= float(b_max_mg))
                & (wl >= xlo - 300.0) & (wl <= xhi + 300.0)
            )
            if np.sum(keep) < 2:
                continue
            line_label = label if not labelled else None
            labelled = True
            ax_field.plot(wl[keep], b[keep], color='0.18', alpha=0.72,
                          lw=1.25, label=line_label, zorder=0.5)

        b_label = _field_label(result, b_mg, reference=reference_b_mg is not None)
        ax_field.axhline(float(b_mg), color='red', linestyle='--',
                         lw=1.7, alpha=0.82, label=b_label, zorder=2)
        ax_flux.set_title(title, fontsize=13, pad=5)
        ax_flux.set_xlim(xlo, xhi)
        ax_field.set_xlim(xlo, xhi)
        ax_field.set_ylim(0.0, float(b_max_mg))
        ax_flux.grid(True, alpha=0.22, linestyle='--')
        ax_field.grid(True, alpha=0.22, linestyle='--')
        return flux_label

    with plt.rc_context({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.3,
        'lines.markersize': 6,
    }):
        fig, axes = plt.subplots(
            2, 2, figsize=(14.0, 8.5), sharex='col',
            gridspec_kw={'height_ratios': [1.0, 1.25],
                         'hspace': 0.05, 'wspace': 0.08})
        flux_label_left = _draw_region(
            axes[0, 0], axes[1, 0], 'Hbeta', blue_xlim, 'H-beta blue region')
        _draw_region(
            axes[0, 1], axes[1, 1], 'Halpha', red_xlim, 'H-alpha red region')

        axes[0, 0].set_ylabel(flux_label_left, fontweight='bold')
        axes[1, 0].set_ylabel(
            'Magnetic field B (MG)', fontweight='bold', labelpad=8)
        axes[1, 0].set_xlabel(r"Wavelength ($\AA$)", fontweight='bold')
        axes[1, 1].set_xlabel(r"Wavelength ($\AA$)", fontweight='bold')
        axes[0, 1].tick_params(labelleft=False)
        axes[1, 1].tick_params(labelleft=False)
        for ax in axes.ravel():
            ax.tick_params(top=True, right=True)
        axes[1, 0].legend(loc='upper right', frameon=False)
        axes[1, 1].legend(loc='upper right', frameon=False)

        fig.tight_layout()
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.92)
        if save_path:
            utils.ensure_dir(os.path.dirname(save_path))
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
        return fig


def save_magnetic_field_outputs(result, output_dir, prefix='magnetic_field',
                                blue_reference_b_mg=None):
    if result is None:
        return {}
    os.makedirs(output_dir, exist_ok=True)
    files = {}

    summary = {
        key: value for key, value in result.items()
        if key not in ('component_table', 'score_grid', 'field_solution_table',
                       'wave', 'flux',
                       'continuum', 'norm_flux', 'norm_err', 'emission_mask',
                       'baseline_model_flux')
    }
    summary_path = os.path.join(output_dir, f'{prefix}_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    files['summary'] = summary_path

    comp_path = os.path.join(output_dir, f'{prefix}_components.csv')
    result['component_table'].to_csv(comp_path, index=False)
    files['components'] = comp_path

    score_path = os.path.join(output_dir, f'{prefix}_score_grid.csv')
    result['score_grid'].to_csv(score_path, index=False)
    files['score_grid'] = score_path

    solution_table = result.get('field_solution_table')
    if isinstance(solution_table, pd.DataFrame) and not solution_table.empty:
        solutions_path = os.path.join(
            output_dir, f'{prefix}_field_solutions.csv')
        solution_table.to_csv(solutions_path, index=False)
        files['field_solutions'] = solutions_path

    blue_nodes = balmer_blue_region_node_table(
        result, reference_b_mg=blue_reference_b_mg)
    if not blue_nodes.empty:
        nodes_path = os.path.join(output_dir, f'{prefix}_blue_region_nodes.csv')
        blue_nodes.to_csv(nodes_path, index=False)
        files['blue_region_nodes'] = nodes_path

    red_nodes = balmer_blue_region_node_table(
        result, reference_b_mg=blue_reference_b_mg, xlim=(5500.0, 9000.0))
    if not red_nodes.empty:
        red_nodes_path = os.path.join(output_dir, f'{prefix}_red_region_nodes.csv')
        red_nodes.to_csv(red_nodes_path, index=False)
        files['red_region_nodes'] = red_nodes_path

    overlay_features = balmer_full_field_overlay_feature_table(
        result, reference_b_mg=blue_reference_b_mg,
        series_focus=('Halpha', 'Hbeta'))
    if not overlay_features.empty:
        overlay_features_path = os.path.join(
            output_dir, f'{prefix}_full_field_overlay_features.csv')
        overlay_features.to_csv(overlay_features_path, index=False)
        files['full_field_overlay_features'] = overlay_features_path

    plot_path = os.path.join(output_dir, f'{prefix}_zeeman_fit.png')
    plot_magnetic_field_result(result, save_path=plot_path)
    files['plot'] = plot_path

    full_overlay_pdf = os.path.join(
        output_dir, f'{prefix}_balmer_full_field_overlay.pdf')
    if plot_balmer_full_field_overlay(
            result, save_path=full_overlay_pdf,
            reference_b_mg=blue_reference_b_mg,
            series_focus=('Halpha', 'Hbeta')) is not None:
        files['full_field_overlay_pdf'] = full_overlay_pdf
        full_overlay_png = os.path.join(
            output_dir, f'{prefix}_balmer_full_field_overlay.png')
        plot_balmer_full_field_overlay(
            result, save_path=full_overlay_png,
            reference_b_mg=blue_reference_b_mg,
            series_focus=('Halpha', 'Hbeta'))
        files['full_field_overlay_png'] = full_overlay_png

    combined_region_pdf = os.path.join(
        output_dir, f'{prefix}_balmer_blue_red_region_interpolation.pdf')
    if plot_balmer_blue_red_region_interpolation(
            result, save_path=combined_region_pdf,
            reference_b_mg=blue_reference_b_mg) is not None:
        files['combined_region_plot_pdf'] = combined_region_pdf
        combined_region_png = os.path.join(
            output_dir, f'{prefix}_balmer_blue_red_region_interpolation.png')
        plot_balmer_blue_red_region_interpolation(
            result, save_path=combined_region_png,
            reference_b_mg=blue_reference_b_mg)
        files['combined_region_plot_png'] = combined_region_png

    blue_pdf = os.path.join(
        output_dir, f'{prefix}_balmer_blue_region_interpolation.pdf')
    if plot_balmer_region_interpolation_paper(
            result, save_path=blue_pdf,
            reference_b_mg=blue_reference_b_mg,
            series_focus='Hbeta') is not None:
        files['blue_region_plot_pdf'] = blue_pdf
        blue_png = os.path.join(
            output_dir, f'{prefix}_balmer_blue_region_interpolation.png')
        plot_balmer_region_interpolation_paper(
            result, save_path=blue_png,
            reference_b_mg=blue_reference_b_mg,
            series_focus='Hbeta')
        files['blue_region_plot_png'] = blue_png

    red_pdf = os.path.join(
        output_dir, f'{prefix}_balmer_red_region_interpolation.pdf')
    if plot_balmer_region_interpolation_paper(
            result, save_path=red_pdf,
            reference_b_mg=blue_reference_b_mg,
            series_focus='Halpha') is not None:
        files['red_region_plot_pdf'] = red_pdf
        red_png = os.path.join(
            output_dir, f'{prefix}_balmer_red_region_interpolation.png')
        plot_balmer_region_interpolation_paper(
            result, save_path=red_png,
            reference_b_mg=blue_reference_b_mg,
            series_focus='Halpha')
        files['red_region_plot_png'] = red_png
    return files
