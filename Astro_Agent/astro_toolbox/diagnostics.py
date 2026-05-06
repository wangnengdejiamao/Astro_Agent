"""Lightweight diagnostic helpers for spectra, SEDs, variability and RVs."""
import os
import numpy as np

from . import utils


OPTICAL_LINES = [
    ('H-alpha', 6562.8, 'Balmer'),
    ('H-beta', 4861.3, 'Balmer'),
    ('H-gamma', 4340.5, 'Balmer'),
    ('H-delta', 4101.7, 'Balmer'),
    ('He I 4471', 4471.5, 'He'),
    ('He II 4686', 4685.7, 'He'),
    ('He I 5876', 5875.6, 'He'),
    ('He I 6678', 6678.2, 'He'),
    ('[O III] 4959', 4958.9, 'Forbidden'),
    ('[O III] 5007', 5006.8, 'Forbidden'),
    ('[N II] 6548', 6548.1, 'Forbidden'),
    ('[N II] 6583', 6583.5, 'Forbidden'),
    ('[S II] 6716', 6716.4, 'Forbidden'),
    ('[S II] 6731', 6730.8, 'Forbidden'),
    ('Ca II K', 3933.7, 'Metal'),
    ('Ca II H', 3968.5, 'Metal'),
    ('Na D', 5892.0, 'Metal'),
]


def _as_array(values):
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5:
        return np.nan
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    if mad > 0:
        return 1.4826 * mad
    return np.nanstd(values)


def _local_line_measurement(wave, flux, err, center, half_width=10.0):
    core = np.abs(wave - center) <= half_width
    side = ((np.abs(wave - center) >= half_width * 1.8)
            & (np.abs(wave - center) <= half_width * 4.0))
    if np.sum(core) < 3 or np.sum(side) < 6:
        return None

    cont = np.nanmedian(flux[side])
    if not np.isfinite(cont) or cont == 0:
        return None

    norm_core = flux[core] / cont
    norm_side = flux[side] / cont
    dw = np.nanmedian(np.diff(wave[core]))
    if not np.isfinite(dw) or dw <= 0:
        dw = half_width * 2.0 / max(np.sum(core), 1)

    noise = _robust_sigma(norm_side - 1.0)
    if not np.isfinite(noise) or noise <= 0:
        if err is not None and len(err) == len(wave):
            e_side = np.nanmedian(np.abs(err[side] / cont))
            noise = e_side if np.isfinite(e_side) and e_side > 0 else 0.05
        else:
            noise = 0.05

    excess = norm_core - 1.0
    deficit = 1.0 - norm_core
    emission_ew = float(np.nansum(np.clip(excess, 0, None)) * dw)
    absorption_ew = float(np.nansum(np.clip(deficit, 0, None)) * dw)
    emission_peak = float(np.nanmax(excess)) if np.any(np.isfinite(excess)) else 0.0
    absorption_depth = float(np.nanmax(deficit)) if np.any(np.isfinite(deficit)) else 0.0
    c_kms = 299792.458

    def _weighted_center(weights):
        weights = np.asarray(weights, dtype=float)
        good = np.isfinite(weights) & (weights > 0)
        if np.sum(good) < 2:
            return np.nan, np.nan
        lam = wave[core][good]
        w = weights[good]
        cen = float(np.nansum(lam * w) / np.nansum(w))
        width = float(np.sqrt(np.nansum(w * (lam - cen) ** 2) / np.nansum(w)))
        return cen, width

    em_center, em_sigma = _weighted_center(np.clip(excess, 0, None))
    abs_center, abs_sigma = _weighted_center(np.clip(deficit, 0, None))
    em_vel = c_kms * (em_center / center - 1.0) if np.isfinite(em_center) else np.nan
    abs_vel = c_kms * (abs_center / center - 1.0) if np.isfinite(abs_center) else np.nan

    return {
        'center_A': center,
        'emission_ew_A': max(emission_ew, 0.0),
        'absorption_ew_A': max(absorption_ew, 0.0),
        'emission_snr': emission_peak / noise,
        'absorption_snr': absorption_depth / noise,
        'emission_center_A': em_center,
        'absorption_center_A': abs_center,
        'emission_velocity_kms': em_vel,
        'absorption_velocity_kms': abs_vel,
        'emission_sigma_A': em_sigma,
        'absorption_sigma_A': abs_sigma,
        'continuum': float(cont),
    }


def _continuum_by_bins(wave, flux, n_bins=60):
    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    if len(wave) < 20:
        return None, None, None

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    n_bins = int(np.clip(n_bins, 8, max(8, len(wave) // 20)))
    edges = np.linspace(wave.min(), wave.max(), n_bins + 1)
    x_med, y_med = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (wave >= lo) & (wave < hi)
        if np.sum(mask) >= 3:
            x_med.append(np.nanmedian(wave[mask]))
            y_med.append(np.nanmedian(flux[mask]))
    if len(x_med) < 4:
        return wave, flux, np.full_like(flux, np.nanmedian(flux))
    cont = np.interp(wave, x_med, y_med)
    return wave, flux, cont


def analyze_spectrum(wave, flux, err=None, survey='', metadata=None):
    """Detect emission/anomalies and check for non-stellar spectral clues."""
    metadata = metadata or {}
    wave = _as_array(wave)
    flux = _as_array(flux)
    err = _as_array(err)
    result = {
        'survey': survey or metadata.get('survey', ''),
        'status': 'no_data',
        'n_points': 0,
        'wavelength_min_A': np.nan,
        'wavelength_max_A': np.nan,
        'median_snr': np.nan,
        'flags': [],
        'emission_flag': False,
        'anomaly_flag': False,
        'misclassification_flag': False,
        'nonstellar_score': 0.0,
        'likely_interpretation': 'insufficient spectrum',
        'line_measurements': {},
    }
    if wave is None or flux is None or len(wave) != len(flux):
        return result

    good = np.isfinite(wave) & np.isfinite(flux)
    if err is not None and len(err) == len(wave):
        good &= np.isfinite(err) | (err == 0)
    wave = wave[good]
    flux = flux[good]
    err = err[good] if err is not None and len(err) == len(good) else None
    if len(wave) < 20:
        result['n_points'] = len(wave)
        return result

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    if err is not None and len(err) == len(wave):
        err = err[order]

    result.update({
        'status': 'ok',
        'n_points': int(len(wave)),
        'wavelength_min_A': float(np.nanmin(wave)),
        'wavelength_max_A': float(np.nanmax(wave)),
    })

    flags = []
    if err is not None and len(err) == len(wave):
        sn = np.abs(flux) / np.where(err > 0, err, np.nan)
        result['median_snr'] = float(np.nanmedian(sn))
        if np.isfinite(result['median_snr']) and result['median_snr'] < 5:
            flags.append('LOW_SNR')

    neg_frac = float(np.mean(flux < 0))
    if neg_frac > 0.15:
        flags.append('MANY_NEGATIVE_PIXELS')

    cw, cf, cont = _continuum_by_bins(wave, flux)
    if cw is not None and cont is not None:
        norm = cf / np.where(cont != 0, cont, np.nan)
        scatter = _robust_sigma(norm - 1.0)
        outlier_frac = float(np.mean(np.abs(norm - 1.0) > max(0.5, 5 * scatter)))
        result['continuum_scatter'] = float(scatter) if np.isfinite(scatter) else np.nan
        result['outlier_fraction'] = outlier_frac
        if outlier_frac > 0.08:
            flags.append('MANY_SPECTRAL_OUTLIERS')

    z = 0.0
    try:
        z = float(metadata.get('z', metadata.get('redshift', 0.0)) or 0.0)
    except (TypeError, ValueError):
        z = 0.0

    line_results = {}
    strong_emission = []
    strong_absorption = []
    for name, rest, family in OPTICAL_LINES:
        center = rest * (1.0 + z)
        if center < wave.min() + 20 or center > wave.max() - 20:
            continue
        meas = _local_line_measurement(wave, flux, err, center)
        if not meas:
            continue
        meas['rest_A'] = rest
        meas['family'] = family
        line_results[name] = meas
        if meas['emission_snr'] >= 4.0 and meas['emission_ew_A'] >= 1.0:
            strong_emission.append(name)
        if meas['absorption_snr'] >= 4.0 and meas['absorption_ew_A'] >= 1.5:
            strong_absorption.append(name)

    result['line_measurements'] = line_results
    result['strong_emission_lines'] = strong_emission
    result['strong_absorption_lines'] = strong_absorption
    result['emission_flag'] = bool(strong_emission)
    if strong_emission:
        flags.append('EMISSION_LINES')

    forbidden = [n for n in strong_emission if n.startswith('[')]
    balmer_em = [n for n in strong_emission if n.startswith('H-')]
    he_em = [n for n in strong_emission if n.startswith('He')]
    balmer_abs_ew = sum(line_results.get(n, {}).get('absorption_ew_A', 0.0)
                        for n in ('H-alpha', 'H-beta', 'H-gamma', 'H-delta'))

    meta_class = str(metadata.get('class', metadata.get('Class', ''))).lower()
    subclass = str(metadata.get('subclass', metadata.get('SubClass', ''))).lower()
    nonstellar_score = 0.0
    if any(k in meta_class for k in ('galaxy', 'qso', 'quasar', 'agn')):
        flags.append('CATALOG_NONSTELLAR_CLASS')
        nonstellar_score += 0.75
    if np.isfinite(z) and abs(z) > 0.01:
        flags.append('NONZERO_REDSHIFT')
        nonstellar_score += 0.45
    if forbidden:
        flags.append('FORBIDDEN_EMISSION')
        nonstellar_score += 0.25
        if 'star' in meta_class or not meta_class:
            flags.append('MISCLASSIFICATION_CHECK')
            nonstellar_score += 0.20

    if nonstellar_score >= 0.6:
        likely = 'non-stellar contaminant / galaxy / QSO candidate'
        result['misclassification_flag'] = True
    elif forbidden and balmer_em:
        likely = 'nebular/CV/accretion emission-line source'
    elif balmer_em or he_em:
        likely = 'emission-line star or interacting binary candidate'
    elif balmer_abs_ew > 15:
        likely = 'hydrogen-atmosphere white dwarf / A-type spectrum'
    elif 'wd' in subclass or 'white' in subclass:
        likely = 'white dwarf candidate'
    elif strong_absorption:
        likely = 'stellar absorption spectrum'
    else:
        likely = 'feature-poor or low-S/N stellar spectrum'

    result['nonstellar_score'] = float(min(nonstellar_score, 1.0))
    result['likely_interpretation'] = likely
    result['flags'] = sorted(set(flags))
    if 'MISCLASSIFICATION_CHECK' in result['flags']:
        result['misclassification_flag'] = True
    result['anomaly_flag'] = any(f in result['flags'] for f in (
        'LOW_SNR', 'MANY_NEGATIVE_PIXELS', 'MANY_SPECTRAL_OUTLIERS'))
    return result


def analyze_spectrum_metadata(metadata, survey=''):
    """Fallback diagnostic when only catalog spectral metadata exist."""
    metadata = metadata or {}
    flags = []
    score = 0.0
    meta_class = str(metadata.get('class', metadata.get('Class', ''))).lower()
    if any(k in meta_class for k in ('galaxy', 'qso', 'quasar', 'agn')):
        flags.append('CATALOG_NONSTELLAR_CLASS')
        score += 0.75
    for key in ('flag_sp', 'flag_fe_h'):
        val = metadata.get(key)
        try:
            if val is not None and float(val) != 0:
                flags.append(f'{key.upper()}_NONZERO')
        except (TypeError, ValueError):
            pass
    return {
        'survey': survey or metadata.get('survey', ''),
        'status': 'metadata_only',
        'n_points': 0,
        'flags': flags,
        'emission_flag': False,
        'anomaly_flag': bool(flags),
        'misclassification_flag': score >= 0.6,
        'nonstellar_score': score,
        'likely_interpretation': (
            'catalog non-stellar object' if score >= 0.6
            else 'catalog metadata only; no line diagnostics'),
        'line_measurements': {},
        'strong_emission_lines': [],
        'strong_absorption_lines': [],
    }


def extract_spectra_from_results(results):
    spectra = []
    sdss = results.get('SDSS_spectrum')
    if sdss and 'wavelength' in sdss:
        spectra.append(('SDSS', sdss['wavelength'], sdss['flux'],
                        sdss.get('error'), sdss))

    desi = results.get('DESI')
    if desi and isinstance(desi, dict) and 'spectrum' in desi:
        sp = desi['spectrum']
        waves, fluxes, errors = [], [], []
        for band in ('B', 'R', 'Z'):
            if band in sp:
                waves.append(np.asarray(sp[band]['wavelength'], dtype=float))
                fluxes.append(np.asarray(sp[band]['flux'], dtype=float))
                errors.append(np.asarray(sp[band].get('error',
                                                       np.zeros_like(waves[-1])),
                                         dtype=float))
        if waves:
            meta = {}
            meta.update(desi.get('match', {}) if isinstance(desi.get('match'), dict) else {})
            meta.update({k: v for k, v in sp.items() if isinstance(v, (int, float, str))})
            spectra.append(('DESI', np.concatenate(waves), np.concatenate(fluxes),
                            np.concatenate(errors), meta))

    for key, name in (
            ('KOA_spectrum', 'KOA/LRIS'),
            ('LAMOST', 'LAMOST'),
            ('HST_spectrum', 'HST'),
            ('JWST_spectrum', 'JWST'),
            ('SPHEREx', 'SPHEREx')):
        r = results.get(key)
        if r and isinstance(r, dict) and 'wavelength' in r and 'flux' in r:
            spectra.append((name, r['wavelength'], r['flux'], r.get('error'), r))

    return spectra


def analyze_all_spectra(results):
    diagnostics = {}
    for survey, wave, flux, err, meta in extract_spectra_from_results(results):
        diagnostics[survey] = analyze_spectrum(
            wave, flux, err, survey=survey, metadata=meta)
    galah = results.get('GALAH')
    if galah and 'GALAH' not in diagnostics:
        diagnostics['GALAH'] = analyze_spectrum_metadata(galah, survey='GALAH')
    return diagnostics


def save_spectral_diagnostics(diagnostics, output_dir):
    import pandas as pd
    if not diagnostics or output_dir is None:
        return None, None
    rows = []
    line_rows = []
    lines = ['# Spectral Diagnostics', '']
    for survey, diag in diagnostics.items():
        rows.append({
            'survey': survey,
            'status': diag.get('status'),
            'n_points': diag.get('n_points'),
            'wavelength_min_A': diag.get('wavelength_min_A'),
            'wavelength_max_A': diag.get('wavelength_max_A'),
            'median_snr': diag.get('median_snr'),
            'emission_flag': diag.get('emission_flag'),
            'anomaly_flag': diag.get('anomaly_flag'),
            'misclassification_flag': diag.get('misclassification_flag'),
            'nonstellar_score': diag.get('nonstellar_score'),
            'likely_interpretation': diag.get('likely_interpretation'),
            'flags': ';'.join(diag.get('flags', [])),
            'strong_emission_lines': ';'.join(diag.get('strong_emission_lines', [])),
            'strong_absorption_lines': ';'.join(diag.get('strong_absorption_lines', [])),
        })
        lines.append(f'## {survey}')
        lines.append(f"  interpretation: {diag.get('likely_interpretation', '')}")
        lines.append(f"  flags: {', '.join(diag.get('flags', [])) or 'none'}")
        if diag.get('strong_emission_lines'):
            lines.append("  emission: " + ', '.join(diag['strong_emission_lines']))
        if diag.get('strong_absorption_lines'):
            lines.append("  absorption: " + ', '.join(diag['strong_absorption_lines']))
        for line_name, meas in diag.get('line_measurements', {}).items():
            line_rows.append({
                'survey': survey,
                'line': line_name,
                'family': meas.get('family'),
                'rest_A': meas.get('rest_A'),
                'expected_center_A': meas.get('center_A'),
                'emission_ew_A': meas.get('emission_ew_A'),
                'emission_snr': meas.get('emission_snr'),
                'emission_center_A': meas.get('emission_center_A'),
                'emission_velocity_kms': meas.get('emission_velocity_kms'),
                'emission_sigma_A': meas.get('emission_sigma_A'),
                'absorption_ew_A': meas.get('absorption_ew_A'),
                'absorption_snr': meas.get('absorption_snr'),
                'absorption_center_A': meas.get('absorption_center_A'),
                'absorption_velocity_kms': meas.get('absorption_velocity_kms'),
                'absorption_sigma_A': meas.get('absorption_sigma_A'),
                'continuum': meas.get('continuum'),
                'is_strong_emission': line_name in diag.get('strong_emission_lines', []),
                'is_strong_absorption': line_name in diag.get('strong_absorption_lines', []),
            })
            is_interesting = (
                line_name in diag.get('strong_emission_lines', [])
                or line_name in diag.get('strong_absorption_lines', []))
            if is_interesting:
                parts = [f"  {line_name}:"]
                if line_name in diag.get('strong_emission_lines', []):
                    parts.append(
                        f"em EW={meas.get('emission_ew_A', np.nan):.2f} A, "
                        f"S/N={meas.get('emission_snr', np.nan):.1f}, "
                        f"v={meas.get('emission_velocity_kms', np.nan):+.1f} km/s")
                if line_name in diag.get('strong_absorption_lines', []):
                    parts.append(
                        f"abs EW={meas.get('absorption_ew_A', np.nan):.2f} A, "
                        f"S/N={meas.get('absorption_snr', np.nan):.1f}, "
                        f"v={meas.get('absorption_velocity_kms', np.nan):+.1f} km/s")
                lines.append(" ".join(parts))
        lines.append("")

    csv_path = utils.write_csv(pd.DataFrame(rows), output_dir,
                               'spectral_diagnostics.csv')
    line_csv_path = utils.write_csv(pd.DataFrame(line_rows), output_dir,
                                    'spectral_line_measurements.csv')
    txt_path = os.path.join(output_dir, 'spectral_diagnostics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return csv_path, txt_path, line_csv_path


def analyze_sed(flux_data):
    """Flag UV/IR excesses and possible two-component SEDs."""
    result = {
        'status': 'no_data',
        'flags': [],
        'uv_excess_flag': False,
        'ir_excess_flag': False,
        'bimodal_flag': False,
        'max_uv_excess_dex': np.nan,
        'max_ir_excess_dex': np.nan,
        'band_residuals': {},
        'interpretation': 'insufficient photometry',
    }
    if not flux_data:
        return result

    rows = []
    for band, vals in flux_data.items():
        if len(vals) < 3:
            continue
        flux, flux_err, wave = vals[:3]
        if np.isfinite(flux) and flux > 0 and np.isfinite(wave) and wave > 0:
            rows.append((band, float(wave), float(flux), float(flux_err or 0)))
    if len(rows) < 3:
        return result

    rows.sort(key=lambda x: x[1])
    bands = [r[0] for r in rows]
    wave = np.array([r[1] for r in rows], dtype=float)
    flux = np.array([r[2] for r in rows], dtype=float)
    x = np.log10(wave)
    y = np.log10(wave * flux)

    optical = (wave >= 3000) & (wave <= 10000)
    if np.sum(optical) >= 2:
        coeff = np.polyfit(x[optical], y[optical], 1)
    else:
        coeff = np.polyfit(x, y, 1)
    baseline = np.polyval(coeff, x)
    residual = y - baseline

    for band, res in zip(bands, residual):
        result['band_residuals'][band] = float(res)

    uv = wave < 3000
    ir = wave > 10000
    if np.any(uv):
        result['max_uv_excess_dex'] = float(np.nanmax(residual[uv]))
    if np.any(ir):
        result['max_ir_excess_dex'] = float(np.nanmax(residual[ir]))

    flags = []
    if np.isfinite(result['max_uv_excess_dex']) and result['max_uv_excess_dex'] > 0.30:
        result['uv_excess_flag'] = True
        flags.append('UV_EXCESS')
    if np.isfinite(result['max_ir_excess_dex']) and result['max_ir_excess_dex'] > 0.30:
        result['ir_excess_flag'] = True
        flags.append('IR_EXCESS')

    if len(rows) >= 5:
        peaks = []
        for i in range(len(y)):
            left = y[i - 1] if i > 0 else -np.inf
            right = y[i + 1] if i < len(y) - 1 else -np.inf
            if y[i] >= left and y[i] >= right:
                peaks.append(i)
        best_pair = None
        for i in peaks:
            for j in peaks:
                if j <= i or abs(x[j] - x[i]) < 0.35:
                    continue
                lo, hi = i, j
                valley = np.nanmin(y[lo:hi + 1])
                prominence = min(y[i], y[j]) - valley
                if prominence > 0.18:
                    best_pair = (i, j, prominence)
        if best_pair or (result['uv_excess_flag'] and result['ir_excess_flag']):
            result['bimodal_flag'] = True
            flags.append('BIMODAL_SED')
            if best_pair:
                i, j, prom = best_pair
                result['blue_peak_band'] = bands[i]
                result['red_peak_band'] = bands[j]
                result['bimodal_prominence_dex'] = float(prom)

    if result['uv_excess_flag'] and result['ir_excess_flag']:
        interp = 'UV and IR excess; possible composite/accreting system'
    elif result['uv_excess_flag']:
        interp = 'UV excess; hot component or activity candidate'
    elif result['ir_excess_flag']:
        interp = 'IR excess; cool companion or dust candidate'
    elif result['bimodal_flag']:
        interp = 'two-component SED candidate'
    else:
        interp = 'single-component SED within simple continuum check'
    result.update({'status': 'ok', 'flags': sorted(set(flags)),
                   'interpretation': interp})
    return result


def save_sed_diagnostics(diag, output_dir):
    import pandas as pd
    if not diag or output_dir is None:
        return None, None
    row = {
        'status': diag.get('status'),
        'uv_excess_flag': diag.get('uv_excess_flag'),
        'ir_excess_flag': diag.get('ir_excess_flag'),
        'bimodal_flag': diag.get('bimodal_flag'),
        'max_uv_excess_dex': diag.get('max_uv_excess_dex'),
        'max_ir_excess_dex': diag.get('max_ir_excess_dex'),
        'flags': ';'.join(diag.get('flags', [])),
        'interpretation': diag.get('interpretation'),
    }
    csv_path = utils.write_csv(pd.DataFrame([row]), output_dir,
                               'sed_diagnostics.csv')
    txt_path = os.path.join(output_dir, 'sed_diagnostics.txt')
    lines = ['# SED Diagnostics', '',
             f"interpretation: {diag.get('interpretation', '')}",
             f"flags: {', '.join(diag.get('flags', [])) or 'none'}",
             f"UV excess dex: {diag.get('max_uv_excess_dex', np.nan):.3f}",
             f"IR excess dex: {diag.get('max_ir_excess_dex', np.nan):.3f}"]
    if diag.get('band_residuals'):
        lines.append('')
        lines.append('Band residuals in dex relative to optical baseline:')
        for band, res in sorted(diag['band_residuals'].items()):
            lines.append(f"  {band}: {res:+.3f}")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return csv_path, txt_path


def evaluate_rv_flags(rv_report):
    """Add quality and science flags to an RV report."""
    if rv_report is None:
        return {'flags': ['NO_RV_REPORT'], 'quality': 'none'}
    flags = []
    best = rv_report.get('best_rv')
    best_err = rv_report.get('best_rv_err')
    if best is None or not np.isfinite(best):
        flags.append('NO_BEST_RV')
    else:
        if abs(best) > 500:
            flags.append('EXTREME_RV')
        elif abs(best) > 250:
            flags.append('HIGH_RV')
    if best_err is None or not np.isfinite(best_err):
        flags.append('NO_RV_ERROR')
    elif best_err > 50:
        flags.append('LOW_PRECISION_RV')
    elif best_err > 20:
        flags.append('MODERATE_PRECISION_RV')
    if rv_report.get('is_sb2'):
        flags.append('SB2_CANDIDATE')

    pipeline = rv_report.get('pipeline_rvs', [])
    if len(pipeline) >= 2:
        vals = np.array([p.get('rv', np.nan) for p in pipeline], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2 and np.nanmax(vals) - np.nanmin(vals) > 50:
            flags.append('PIPELINE_RV_DISAGREEMENT')

    for survey, ccf in rv_report.get('ccf_results', {}).items():
        single = ccf.get('single', {})
        h = single.get('ccf_height', np.nan)
        if np.isfinite(h) and h < 0.10:
            flags.append(f'WEAK_CCF_{survey}')

    if 'NO_BEST_RV' in flags:
        quality = 'none'
    elif any(f in flags for f in ('LOW_PRECISION_RV', 'WEAK_CCF_SDSS',
                                  'WEAK_CCF_DESI', 'WEAK_CCF_LAMOST')):
        quality = 'caution'
    elif flags:
        quality = 'flagged'
    else:
        quality = 'good'
    return {'flags': sorted(set(flags)), 'quality': quality}
