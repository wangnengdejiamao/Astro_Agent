"""
Multi-survey period analysis pipeline
======================================
对所有可用光变曲线运行 MHAOV 周期搜索，
自动对检测到的周期进行多巡天联合折叠绘图。

支持: ZTF (g/r/i), WISE/NEOWISE (W1/W2), TESS, Kepler/K2, Gaia (G/BP/RP)

用法:
    from astro_toolbox.period_analysis import run_period_analysis
    result = run_period_analysis(query_results, output_dir='./output')
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config, utils
from matplotlib.gridspec import GridSpec


MIN_PERIOD_DAY = 5.0 / 1440.0
MIN_PERIOD_MIN = MIN_PERIOD_DAY * 24.0 * 60.0


def fold_phase(time, period):
    """Return phase folded into [0, 1)."""
    return np.mod(time, period) / period % 1.0


# ================================================================
#  数据格式转换
# ================================================================

def flux_to_relative_mag(flux, flux_err=None):
    """
    归一化流量 → 差分星等。
    适用于 TESS/Kepler 的 normalized flux (median ~ 1.0)。
    mag = -2.5 * log10(flux / median_flux)

    Returns:
        (mag, mag_err) numpy arrays. 无效值为 NaN。
    """
    flux = np.asarray(flux, dtype=np.float64)
    median_flux = np.nanmedian(flux)
    if median_flux <= 0:
        return np.full_like(flux, np.nan), None

    ratio = flux / median_flux
    good = np.isfinite(ratio) & (ratio > 0)
    mag = np.full_like(flux, np.nan)
    mag[good] = -2.5 * np.log10(ratio[good])

    mag_err = None
    if flux_err is not None:
        flux_err = np.asarray(flux_err, dtype=np.float64)
        mag_err = np.full_like(flux_err, np.nan)
        mask = good & np.isfinite(flux_err) & (flux_err > 0)
        mag_err[mask] = (2.5 / np.log(10)) * np.abs(flux_err[mask] / flux[mask])

    return mag, mag_err


def _clean_space_flux_time(time, flux, flux_err=None, max_gap_days=1.0):
    """
    Clean TESS/Kepler-like stitched light curves before period search.

    The archive products are often stitched across sectors/campaigns.  A small
    number of nonphysical values can dominate LS/MHAOV and create cadence
    aliases, so each time segment is median-normalized and broad outliers are
    rejected while keeping real eclipse-scale dips.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = None if flux_err is None else np.asarray(flux_err, dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    raw_n = int(len(time))
    finite_n = int(np.sum(np.isfinite(time) & np.isfinite(flux)))
    positive_n = int(np.sum(valid))
    if positive_n < 10:
        return time[valid], flux[valid], err[valid] if err is not None else None, {
            'raw_n': raw_n,
            'finite_n': finite_n,
            'positive_n': positive_n,
            'kept_n': positive_n,
            'removed_n': raw_n - positive_n,
            'note': 'too few points for segment normalization',
        }

    t = time[valid]
    f = flux[valid]
    e = err[valid] if err is not None and err.shape == flux.shape else None
    order = np.argsort(t)
    t = t[order]
    f = f[order]
    if e is not None:
        e = e[order]
        e = np.where(np.isfinite(e) & (e >= 0), e, np.nan)
        if np.sum(np.isfinite(e) & (e > 0)) < 10:
            e = None

    gaps = np.diff(t)
    cuts = np.where(gaps > max_gap_days)[0] + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(t)]

    t_out, f_out, e_out = [], [], []
    for start, stop in zip(starts, stops):
        ts = t[start:stop]
        fs = f[start:stop]
        es = e[start:stop] if e is not None else None
        good_seg = np.isfinite(fs) & (fs > 0)
        if np.sum(good_seg) < 5:
            continue
        med = np.nanmedian(fs[good_seg])
        if not np.isfinite(med) or med <= 0:
            continue
        rel = fs / med
        trend_corr = np.ones_like(rel)
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) > 5:
            dt_med = np.nanmedian(dt)
            if np.isfinite(dt_med) and dt_med > 0:
                from scipy.ndimage import median_filter
                win = int(np.clip(0.5 / dt_med, 51, 2001))
                if win % 2 == 0:
                    win += 1
                if len(rel) > win:
                    trend = median_filter(rel, size=win, mode='nearest')
                    trend = np.where(np.isfinite(trend) & (trend > 0),
                                     trend, np.nanmedian(rel[good_seg]))
                    trend_corr = trend
                    rel = rel / trend
        rel_err = es / med / trend_corr if es is not None else None
        rel_med = np.nanmedian(rel[good_seg])
        mad = np.nanmedian(np.abs(rel[good_seg] - rel_med))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nan
        keep = good_seg & (rel > 0.2) & (rel < 2.0)
        if np.isfinite(sigma) and sigma > 0:
            keep &= np.abs(rel - rel_med) <= max(0.8, 12.0 * sigma)
        t_out.append(ts[keep])
        f_out.append(rel[keep])
        if e is not None:
            e_out.append(rel_err[keep])

    if not t_out:
        return np.array([]), np.array([]), None, {
            'raw_n': raw_n,
            'finite_n': finite_n,
            'positive_n': positive_n,
            'kept_n': 0,
            'removed_n': raw_n,
            'note': 'all points rejected by cleaning',
        }

    t_clean = np.concatenate(t_out)
    f_clean = np.concatenate(f_out)
    e_clean = np.concatenate(e_out) if e_out else None
    if e_clean is not None and np.sum(np.isfinite(e_clean) & (e_clean > 0)) < 10:
        e_clean = None
    stats = {
        'raw_n': raw_n,
        'finite_n': finite_n,
        'positive_n': positive_n,
        'kept_n': int(len(t_clean)),
        'removed_n': int(raw_n - len(t_clean)),
        'segment_count': int(len(starts)),
        'flux_min_kept': float(np.nanmin(f_clean)) if len(f_clean) else np.nan,
        'flux_max_kept': float(np.nanmax(f_clean)) if len(f_clean) else np.nan,
    }
    return t_clean, f_clean, e_clean, stats


# ================================================================
#  从查询结果提取统一格式光变曲线
# ================================================================

def _to_mjd(time, time_system):
    """
    将各巡天的原始时间转换为统一的 MJD。

    时间零点关系 (BJD = MJD + 2400000.5):
        BTJD     = BJD - 2457000.0   → MJD = BTJD  + 56999.5
        BKJD     = BJD - 2454833.0   → MJD = BKJD  + 54832.5
        Gaia_BJD = BJD - 2455197.5   → MJD = Gaia  + 55197.0
        MJD      → 不变
    """
    _OFFSETS = {
        'BTJD':     56999.5,
        'BKJD':     54832.5,
        'Gaia_BJD': 55197.0,
    }
    offset = _OFFSETS.get(time_system, 0.0)
    return np.asarray(time, dtype=float) + offset


def _normalize_mjd_like(time):
    """把 JD/HJD/MJD-like 时间统一到 MJD-like 数值，保留亚日相位。"""
    t = np.asarray(time, dtype=float)
    med = np.nanmedian(t)
    if np.isfinite(med) and med > 2400000:
        return t - 2400000.5
    return t


def _frequency_grid_for_curve(time, survey):
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) < 5:
        return None, None
    span = np.nanmax(t) - np.nanmin(t)
    if not np.isfinite(span) or span <= 0:
        return None, None
    freq_min = max(2.0 / span, 1.0 / 5000.0)
    dt = np.diff(np.sort(t))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return freq_min, None
    dt_ref = np.nanpercentile(dt, 5)
    # User-requested hard floor: do not search periods shorter than 5 minutes.
    # Older output files can still contain sub-5-min aliases, so selection code
    # below also marks any such period as non-physical for the paper products.
    freq_max = min(0.5 / dt_ref, 1.0 / MIN_PERIOD_DAY)
    if survey in ('TESS', 'Kepler', 'K2'):
        # Shorter periods are technically searchable at 2-min cadence, but in
        # stitched survey products they are frequently cadence/systematic
        # aliases.  Ultra-short candidates remain visible in the individual
        # periodogram if users lower this cap.
        freq_min = max(freq_min, 0.2)
        freq_max = min(freq_max, 120.0)
    if survey == 'WISE':
        # WISE 的半年采样窗函数很强，避免把极高频随机别名当成参考周期。
        freq_max = min(freq_max, 80.0)
    if freq_max <= freq_min:
        return None, None
    return freq_min, freq_max


def _period_relation(p1, p2, tol=0.03):
    if not (p1 and p2) or p1 <= 0 or p2 <= 0:
        return None
    ratios = {
        'same': 1.0,
        'half': 0.5,
        'double': 2.0,
        'third': 1.0 / 3.0,
        'triple': 3.0,
    }
    for name, ratio in ratios.items():
        if abs(p1 / p2 - ratio) <= tol:
            return name
    return None


def _select_period_result(label, survey, candidates):
    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None, 'no_result'
    for c in candidates:
        c.setdefault('method', 'unknown')

    # MHAOV handles eclipses/ellipsoidal harmonics better; LS is kept as an
    # independent check and can win when substantially more significant.
    def score(c):
        fap = c.get('fap', np.nan)
        base = -np.log10(max(fap, 1e-300)) if np.isfinite(fap) else 0.0
        if c.get('method') == 'MHAOV':
            base += 0.35
        if survey in ('TESS', 'Kepler'):
            base += 0.45
        if survey == 'WISE':
            base -= 0.55
        return base

    best = max(candidates, key=score)
    other_periods = [c['best_period'] for c in candidates if c is not best]
    relation = None
    if other_periods:
        relation = _period_relation(best['best_period'], other_periods[0])
    quality = 'good' if relation in ('same', 'half', 'double') else 'single_method'
    if survey in ('TESS', 'Kepler', 'K2') and best.get('best_period', np.inf) < 0.01:
        quality = 'cadence_alias'
        best['alias_warning'] = 'space-cadence/systematics alias candidate; not used as reference period'
    if best.get('best_period', np.inf) < MIN_PERIOD_DAY:
        quality = 'below_min_period'
        best['alias_warning'] = (
            f'period shorter than {MIN_PERIOD_MIN:.1f} min; treated as an alias '
            'and not used as a physical period')
    if survey == 'WISE':
        quality = 'fold_only' if quality == 'single_method' else 'caution_alias'
    best['quality'] = quality
    best['agreement'] = relation or ''
    best['candidate_methods'] = '+'.join(c.get('method', '') for c in candidates)
    return best, quality


def _reference_priority(label, pr):
    survey = label.split()[0]
    priority = {
        'TESS': 0,
        'Kepler': 1,
        'K2': 1,
        'ZTF': 2,
        'Gaia': 3,
        'HST': 4,
        'JWST': 4,
        'WISE': 6,
    }.get(survey, 5)
    fap = pr.get('fap', np.inf)
    fap_score = -np.log10(max(fap, 1e-300)) if np.isfinite(fap) else 0.0
    return (priority, -fap_score)


def _extract_lightcurve_data(results, min_pts_ground=20,
                              min_pts_space=20, min_pts_gaia=10,
                              min_pts_wise=5):
    """
    从 AstroQueryAll.results 中提取所有光变曲线。
    ZTF: 用 HJD 时间轴, min-max 归一化星等, 全局 T0 对齐。
    其余巡天时间统一转换为 MJD。

    Returns:
        list of dict, 每个包含:
            time, mag, magerr, label, survey, band
            ZTF 额外含 raw_mag (原始星等) 用于绘图标注
    """
    curves = []

    # --- ZTF: 统一到 MJD-like 时间轴 + min-max 归一化 ---
    ztf = results.get('ZTF_lightcurve')
    if ztf:
        ztf_bands = {}
        for band in ('g', 'r', 'i'):
            if band not in ztf:
                continue
            df = ztf[band]
            if not hasattr(df, '__len__') or len(df) < min_pts_ground:
                continue
            # 优先使用 HJD 但转换为 MJD-like 数值，避免和 TESS/WISE/Kepler
            # 联合折叠时落在不同零点。
            if 'hjd' in df.columns:
                t = _normalize_mjd_like(df['hjd'])
            else:
                t = _normalize_mjd_like(df['mjd'])
            m = np.asarray(df['mag'], dtype=float)
            me = np.asarray(df['magerr'], dtype=float)
            valid = np.isfinite(t) & np.isfinite(m) & np.isfinite(me) & (me > 0)
            t, m, me = t[valid], m[valid], me[valid]
            if len(t) >= min_pts_ground:
                ztf_bands[band] = (t, m, me)

        if ztf_bands:
            for band, (t, m, me) in ztf_bands.items():
                # min-max 归一化 (与参考代码一致)
                m_min, m_max = m.min(), m.max()
                if m_max > m_min:
                    m_norm = (m - m_min) / (m_max - m_min)
                else:
                    m_norm = np.zeros_like(m)
                curves.append({
                    'time': t,
                    'mag': m_norm,
                    'magerr': me,
                    'raw_mag': m,
                    'label': f'ZTF {band}', 'survey': 'ZTF', 'band': band,
                })

    # --- WISE/NEOWISE (已是 MJD) ---
    wise = results.get('WISE_lightcurve')
    if wise:
        for band in ('W1', 'W2'):
            if band in wise and hasattr(wise[band], '__len__') and len(wise[band]) >= min_pts_wise:
                df = wise[band]
                curves.append({
                    'time': np.asarray(df['mjd'], dtype=float),
                    'mag': np.asarray(df['mag'], dtype=float),
                    'magerr': np.asarray(df['magerr'], dtype=float),
                    'label': f'WISE {band}', 'survey': 'WISE', 'band': band,
                })

    # --- TESS (BTJD → MJD) ---
    tess = results.get('TESS')
    if tess and tess.get('n_points', 0) >= min_pts_space:
        tsys = tess.get('time_system', 'BTJD')
        t_clean, f_clean, ferr_clean, clean_stats = _clean_space_flux_time(
            tess['time'], tess['flux'], tess.get('flux_err'))
        if len(t_clean) >= min_pts_space:
            mag, magerr = flux_to_relative_mag(f_clean, ferr_clean)
            curves.append({
                'time': _to_mjd(t_clean, tsys),
                'mag': mag, 'magerr': magerr,
                'label': 'TESS', 'survey': 'TESS', 'band': 'T',
                'cleaning': clean_stats,
            })

    # --- Kepler/K2 (BKJD/BTJD → MJD) ---
    kep = results.get('Kepler/K2')
    if kep and kep.get('n_points', 0) >= min_pts_space:
        tsys = kep.get('time_system', 'BKJD')
        t_clean, f_clean, ferr_clean, clean_stats = _clean_space_flux_time(
            kep['time'], kep['flux'], kep.get('flux_err'))
        if len(t_clean) >= min_pts_space:
            mag, magerr = flux_to_relative_mag(f_clean, ferr_clean)
            curves.append({
                'time': _to_mjd(t_clean, tsys),
                'mag': mag, 'magerr': magerr,
                'label': kep.get('survey', 'Kepler'), 'survey': 'Kepler', 'band': 'Kp',
                'cleaning': clean_stats,
            })

    # --- Gaia (Gaia_BJD → MJD) ---
    gaia = results.get('Gaia_lightcurve')
    if gaia:
        tsys = gaia.get('time_system', 'Gaia_BJD')
        for band in ('G', 'BP', 'RP'):
            if band not in gaia:
                continue
            df = gaia[band]
            if not hasattr(df, 'columns') or len(df) < min_pts_gaia:
                continue
            if 'mag' in df.columns:
                curves.append({
                    'time': _to_mjd(df['time'], tsys),
                    'mag': np.asarray(df['mag'], dtype=float),
                    'magerr': None,
                    'label': f'Gaia {band}', 'survey': 'Gaia', 'band': band,
                })
            elif 'flux' in df.columns:
                flux = np.asarray(df['flux'], dtype=float)
                ferr = np.asarray(df['flux_error'], dtype=float) if 'flux_error' in df.columns else None
                mag, magerr = flux_to_relative_mag(flux, ferr)
                curves.append({
                    'time': _to_mjd(df['time'], tsys),
                    'mag': mag, 'magerr': magerr,
                    'label': f'Gaia {band}', 'survey': 'Gaia', 'band': band,
                })

    # --- HST (已是 MJD) ---
    hst_lc = results.get('HST_lightcurve')
    if hst_lc and hst_lc.get('filters'):
        for filt_name, df in hst_lc['filters'].items():
            if hasattr(df, '__len__') and len(df) >= min_pts_ground:
                mag_arr = np.asarray(df['mag'], dtype=float)
                if np.sum(np.isfinite(mag_arr)) >= min_pts_ground:
                    curves.append({
                        'time': np.asarray(df['mjd'], dtype=float),
                        'mag': mag_arr,
                        'magerr': np.asarray(df['magerr'], dtype=float),
                        'label': f'HST {filt_name}', 'survey': 'HST',
                        'band': filt_name,
                    })

    # --- JWST (已是 MJD) ---
    jwst_lc = results.get('JWST_lightcurve')
    if jwst_lc and jwst_lc.get('filters'):
        for filt_name, df in jwst_lc['filters'].items():
            if hasattr(df, '__len__') and len(df) >= min_pts_ground:
                mag_arr = np.asarray(df['mag'], dtype=float)
                if np.sum(np.isfinite(mag_arr)) >= min_pts_ground:
                    curves.append({
                        'time': np.asarray(df['mjd'], dtype=float),
                        'mag': mag_arr,
                        'magerr': np.asarray(df['magerr'], dtype=float),
                        'label': f'JWST {filt_name}', 'survey': 'JWST',
                        'band': filt_name,
                    })

    return curves


# ================================================================
#  多巡天联合折叠图
# ================================================================

SURVEY_COLORS = {
    'ZTF g': 'green', 'ZTF r': 'red', 'ZTF i': 'goldenrod',
    'WISE W1': 'dodgerblue', 'WISE W2': 'orangered',
    'TESS': 'purple', 'Kepler': 'darkcyan', 'K2': 'teal',
    'Gaia G': 'darkgreen', 'Gaia BP': 'royalblue', 'Gaia RP': 'firebrick',
}


def _phase_bin_median(phase, mag, n_bins=60):
    phase = np.asarray(phase, dtype=float)
    mag = np.asarray(mag, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    x, y, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (phase >= lo) & (phase < hi) & np.isfinite(mag)
        if np.sum(mask) >= 1:
            x.append((lo + hi) / 2.0)
            y.append(np.nanmedian(mag[mask]))
            counts.append(int(np.sum(mask)))
    return np.asarray(x), np.asarray(y), np.asarray(counts)


def analyze_folded_morphology(curve, period, t0=None, n_bins=60):
    """
    Compute simple folded-lightcurve morphology statistics.

    Magnitudes are assumed: larger values are fainter, so a dip/eclipse is a
    high-magnitude excursion.
    """
    if curve is None or period is None or period <= 0:
        return None
    t = np.asarray(curve.get('time'), dtype=float)
    if 'raw_mag' in curve:
        m = np.asarray(curve['raw_mag'], dtype=float)
    else:
        m = np.asarray(curve.get('mag'), dtype=float)
    valid = np.isfinite(t) & np.isfinite(m)
    t, m = t[valid], m[valid]
    if len(t) < 5:
        return None
    if t0 is None:
        t0 = np.nanmin(t)
    phase = ((t - t0) / period) % 1.0
    bx, by, counts = _phase_bin_median(phase, m, n_bins=n_bins)
    if len(by) < 5:
        amp = float(np.nanpercentile(m, 95) - np.nanpercentile(m, 5))
        return {
            'curve': curve.get('label', ''),
            'period_day': period,
            'n_points': int(len(m)),
            'amplitude_mag': amp,
            'asymmetry': np.nan,
            'dip_duty_cycle': np.nan,
            'primary_dip_phase': np.nan,
            'morphology': 'sparse_fold',
        }

    amp = float(np.nanpercentile(by, 95) - np.nanpercentile(by, 5))
    med = float(np.nanmedian(by))
    if amp > 0:
        dip_threshold = med + 0.45 * amp
        dip_bins = by >= dip_threshold
        dip_duty = float(np.sum(dip_bins) / len(by))
        primary_phase = float(bx[np.nanargmax(by)])
        half_a = np.nanmean(by[(bx >= 0.0) & (bx < 0.5)])
        half_b = np.nanmean(by[(bx >= 0.5) & (bx < 1.0)])
        asym = float(abs(half_a - half_b) / amp)
        diffs = np.diff(np.r_[by, by[0]])
        smoothness = float(_safe_std(diffs) / amp)
    else:
        dip_duty = 0.0
        primary_phase = np.nan
        asym = 0.0
        smoothness = np.nan

    if amp < 0.03:
        morphology = 'low_amplitude_or_flat'
    elif dip_duty <= 0.18 and amp >= 0.08:
        morphology = 'eclipsing_or_dipper'
    elif dip_duty <= 0.35 and amp >= 0.05:
        morphology = 'broad_dip_or_ellipsoidal'
    elif np.isfinite(smoothness) and smoothness < 0.45:
        morphology = 'sinusoidal_or_spotted'
    else:
        morphology = 'complex_or_irregular'

    return {
        'curve': curve.get('label', ''),
        'period_day': float(period),
        'period_min': float(period * 24 * 60),
        'n_points': int(len(m)),
        'amplitude_mag': amp,
        'asymmetry': asym,
        'dip_duty_cycle': dip_duty,
        'primary_dip_phase': primary_phase,
        'smoothness': smoothness,
        'morphology': morphology,
    }


def _safe_std(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.nanstd(values)) if len(values) else np.nan


def _fit_harmonic_phase_curve(phase, mag, magerr=None, n_harmonics=4):
    """
    Fit a compact multi-harmonic sinusoid to a folded light curve.

    Model:
        m(phi) = c0 + sum_k [a_k sin(2*pi*k*phi) + b_k cos(2*pi*k*phi)]

    This gives a clean scientific guide curve without drawing many connected
    median-bin segments across noisy folded photometry.
    """
    phase = np.asarray(phase, dtype=float)
    mag = np.asarray(mag, dtype=float)
    valid = np.isfinite(phase) & np.isfinite(mag)
    if magerr is not None:
        err = np.asarray(magerr, dtype=float)
        if err.shape == mag.shape:
            valid &= np.isfinite(err) & (err > 0)
        else:
            err = None
    else:
        err = None

    ph = phase[valid] % 1.0
    y = mag[valid]
    if err is not None:
        e = err[valid]
    else:
        e = None

    if len(y) < max(10, 2 * n_harmonics + 3):
        return None

    n_h = int(max(1, min(n_harmonics, (len(y) - 3) // 2, 6)))
    cols = [np.ones_like(ph)]
    for k in range(1, n_h + 1):
        ang = 2.0 * np.pi * k * ph
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    X = np.column_stack(cols)

    if e is not None and len(e) == len(y):
        floor = np.nanmedian(e[np.isfinite(e) & (e > 0)])
        if not np.isfinite(floor) or floor <= 0:
            floor = np.nanstd(y) if np.nanstd(y) > 0 else 1.0
        sigma = np.clip(e, floor * 0.5, floor * 20.0)
        w = 1.0 / sigma
    else:
        w = np.ones_like(y)

    keep = np.ones_like(y, dtype=bool)
    coeff = None
    for _ in range(2):
        Xw = X[keep] * w[keep, None]
        yw = y[keep] * w[keep]
        try:
            coeff = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        resid = y - X @ coeff
        mad = np.nanmedian(np.abs(resid[keep] - np.nanmedian(resid[keep])))
        sigma_r = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(resid[keep])
        if not np.isfinite(sigma_r) or sigma_r <= 0:
            break
        new_keep = np.abs(resid) < 5.0 * sigma_r
        if new_keep.sum() < max(10, 2 * n_h + 3) or np.all(new_keep == keep):
            break
        keep = new_keep

    if coeff is None:
        return None
    resid = y[keep] - X[keep] @ coeff
    rms = float(np.sqrt(np.nanmean(resid**2))) if len(resid) else np.nan
    return {
        'coeff': coeff,
        'n_harmonics': n_h,
        'n_fit': int(np.sum(keep)),
        'rms_mag': rms,
    }


def _evaluate_harmonic_phase_curve(fit, phase_grid):
    """Evaluate a harmonic fit returned by _fit_harmonic_phase_curve."""
    if not fit:
        return None
    phase_grid = np.asarray(phase_grid, dtype=float)
    coeff = np.asarray(fit.get('coeff'), dtype=float)
    if coeff.size < 3:
        return None
    y = np.full_like(phase_grid, coeff[0], dtype=float)
    idx = 1
    for k in range(1, int(fit.get('n_harmonics', 1)) + 1):
        ang = 2.0 * np.pi * k * phase_grid
        y += coeff[idx] * np.sin(ang) + coeff[idx + 1] * np.cos(ang)
        idx += 2
        if idx + 1 >= coeff.size:
            break
    return y


def _harmonic_count_for_curve(n_points):
    if n_points < 30:
        return 1
    if n_points < 80:
        return 2
    return 4


def plot_combined_fold(curves, period, period_source, output_dir,
                       ra=None, dec=None, title_prefix='', t0=None):
    """
    多面板折叠图: 每个巡天/波段一行, 统一折叠到给定周期。

    Args:
        t0: 全局折叠零点 (MJD)。所有曲线使用同一 T0 确保相位对齐。
            若为 None 则取所有曲线的最小时间。
    """
    n = len(curves)
    if n == 0:
        return None

    # 全局统一 T0: 所有曲线共享同一折叠零点
    if t0 is None:
        all_t_min = []
        for c in curves:
            t_arr = np.asarray(c['time'], dtype=float)
            t_valid = t_arr[np.isfinite(t_arr)]
            if len(t_valid) > 0:
                all_t_min.append(t_valid.min())
        t0 = min(all_t_min) if all_t_min else 0.0

    fig, axes = plt.subplots(n, 1, figsize=(12, max(2.5 * n + 1, 4)),
                              sharex=True, squeeze=False)
    axes = axes.flatten()

    for ax, curve in zip(axes, curves):
        t = np.asarray(curve['time'], dtype=float)
        # 折叠图用原始星等 (ZTF 存了 raw_mag); 其余巡天直接用 mag
        if 'raw_mag' in curve:
            m = np.asarray(curve['raw_mag'], dtype=float)
        else:
            m = np.asarray(curve['mag'], dtype=float)
        me = curve.get('magerr')

        valid = np.isfinite(t) & np.isfinite(m)
        if me is not None:
            me = np.asarray(me, dtype=float)
            valid &= np.isfinite(me)
            me = me[valid]
        t, m = t[valid], m[valid]

        if len(t) == 0:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                    ha='center', va='center')
            ax.set_ylabel(curve['label'], fontsize=10)
            continue

        phase = ((t - t0) / period) % 1.0
        color = SURVEY_COLORS.get(curve['label'], 'black')
        morph = analyze_folded_morphology(curve, period, t0=t0)

        has_errors = (me is not None and len(me) > 0)
        if has_errors:
            if len(m) > 6000:
                ax.scatter(phase, m, s=5.0, c=color, alpha=0.055,
                           edgecolors='none')
                ax.scatter(phase + 1, m, s=5.0, c=color, alpha=0.028,
                           edgecolors='none')
                rng = np.random.default_rng(12345)
                idx = np.sort(rng.choice(len(m), 6000, replace=False))
                ax.errorbar(phase[idx], m[idx], yerr=me[idx], fmt='.',
                            color=color, ms=3.4, elinewidth=0.25,
                            alpha=0.18, capsize=0)
                ax.errorbar(phase[idx] + 1, m[idx], yerr=me[idx], fmt='.',
                            color=color, ms=3.4, elinewidth=0.25,
                            alpha=0.09, capsize=0)
            else:
                ax.errorbar(phase, m, yerr=me, fmt='.', color=color,
                           ms=5.0, elinewidth=0.45, alpha=0.55)
                ax.errorbar(phase + 1, m, yerr=me, fmt='.', color=color,
                           ms=5.0, elinewidth=0.45, alpha=0.24)
        else:
            if len(m) > 5000:
                size, alpha1, alpha2 = 5.0, 0.055, 0.028
            elif len(m) > 1000:
                size, alpha1, alpha2 = 12.0, 0.22, 0.10
            else:
                size, alpha1, alpha2 = 18.0, 0.55, 0.24
            ax.scatter(phase, m, s=size, c=color, alpha=alpha1,
                       edgecolors='none')
            ax.scatter(phase + 1, m, s=size, c=color, alpha=alpha2,
                       edgecolors='none')

        fit = _fit_harmonic_phase_curve(
            phase, m, me, n_harmonics=_harmonic_count_for_curve(len(m)))
        if fit:
            grid = np.linspace(0.0, 2.0, 800)
            model = _evaluate_harmonic_phase_curve(fit, grid)
            if model is not None:
                ax.plot(grid, model, '-', color='black', lw=1.8, alpha=0.9,
                        label=f"{fit['n_harmonics']}-harmonic sinusoid")

        bx, by, counts = _phase_bin_median(phase, m, n_bins=55)
        if len(bx) >= 3:
            ax.scatter(bx, by, s=np.clip(counts * 1.5, 18, 72), c='white',
                       edgecolors='black', linewidths=0.65, zorder=5)
            ax.scatter(bx + 1, by, s=np.clip(counts * 1.5, 18, 72), c='white',
                       edgecolors='black', linewidths=0.65, zorder=5)

        ax.set_ylabel(curve['label'], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 2)
        ax.grid(True, alpha=0.3)
        if len(m) > 3:
            lo, hi = np.nanpercentile(m, [1, 99])
            pad = max((hi - lo) * 0.18, 0.02)
            if np.isfinite(lo + hi) and hi > lo:
                ax.set_ylim(hi + pad, lo - pad)
        if morph:
            label = (f"{morph['morphology']}  "
                     f"A={morph['amplitude_mag']:.3f}  "
                     f"D={morph['dip_duty_cycle']:.2f}  "
                     f"Asym={morph['asymmetry']:.2f}")
            if fit and np.isfinite(fit.get('rms_mag', np.nan)):
                label += f"  fit RMS={fit['rms_mag']:.3f}"
        else:
            label = f'N={len(m)}'
        ax.text(0.99, 0.94, label, transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#333',
                bbox=dict(facecolor='white', alpha=0.72, edgecolor='none',
                          boxstyle='round,pad=0.2'))

    axes[-1].set_xlabel('Phase', fontsize=11)
    coord = f'  RA={ra:.4f} DEC={dec:.4f}' if ra is not None else ''
    period_h = period * 24.0
    title = (f'{title_prefix}Phase-folded at P = {period_h:.4f} h '
             f'({period:.6f} d; from {period_source}){coord}')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    safe_src = period_source.replace(' ', '_')
    path = os.path.join(output_dir, f'combined_fold_P{period_h:.3f}h_{safe_src}.png')
    utils.save_and_close(fig, path)
    return path


def _normalize_for_aov(mag):
    mag = np.asarray(mag, dtype=float)
    lo = np.nanmin(mag)
    hi = np.nanmax(mag)
    if not np.isfinite(lo + hi) or hi <= lo:
        return mag - np.nanmedian(mag)
    return (mag - lo) / (hi - lo)


def _aov_periodogram_reference(time, mag, magerr=None,
                               fmin=1e-3, fmax=145.0,
                               fresolution=1e-3,
                               finetune_resolution=1e-4):
    """
    Run the user-requested P4J MHAOV periodogram when available.

    P4J is optional because it requires compiled extensions on macOS.  The
    fallback uses the toolbox MHAOV implementation on the same frequency range
    and keeps the same output convention, including P_orb = 2 / f_best.
    """
    time = np.asarray(time, dtype=float)
    mag = np.asarray(mag, dtype=float)
    if magerr is None:
        magerr = np.full_like(mag, np.nanmedian(np.abs(mag - np.nanmedian(mag))) or 0.01)
    magerr = np.asarray(magerr, dtype=float)
    valid = np.isfinite(time) & np.isfinite(mag) & np.isfinite(magerr) & (magerr > 0)
    time = time[valid]
    mag = mag[valid]
    magerr = magerr[valid]
    if len(time) < 5:
        return None

    try:
        import P4J  # type: ignore
        per = P4J.periodogram('MHAOV')
        per.set_data(time, mag, magerr)
        per.frequency_grid_evaluation(
            fmin=fmin, fmax=fmax, fresolution=fresolution)
        per.finetune_best_frequencies(
            fresolution=finetune_resolution, n_local_optima=10)
        freq, power = per.get_periodogram()
        fbest, pbest = per.get_best_frequencies()
        best_freq = float(np.ravel(fbest)[0])
        best_power = float(np.ravel(pbest)[0]) if np.size(pbest) else np.nan
        backend = 'P4J_MHAOV'
    except Exception as exc:
        n_freq = int(max(1000, min(120000, (fmax - fmin) / fresolution)))
        pr = utils.mhaov(
            time, mag, magerr,
            freq_min=fmin, freq_max=fmax,
            n_freq=n_freq, n_harmonics=3,
        )
        if pr is None:
            return None
        freq = np.asarray(pr['freqs'], dtype=float)
        power = np.asarray(pr['power'], dtype=float)
        idx = int(np.nanargmax(power))
        best_freq = float(freq[idx])
        best_power = float(power[idx])
        backend = f'toolbox_MHAOV_fallback:{type(exc).__name__}'

    objperiod = 2.0 / best_freq
    return {
        'best_period': float(objperiod),
        'best_freq': best_freq,
        'freqs': np.asarray(freq, dtype=float),
        'power': np.asarray(power, dtype=float),
        'best_power': best_power,
        'backend': backend,
        'n_points': int(len(time)),
        'period_min': float(objperiod * 24.0 * 60.0),
        'period_hour': float(objperiod * 24.0),
    }


def _ztf_band_dataframe(ztf_result, band):
    if ztf_result is None or band not in ztf_result:
        return None
    df = ztf_result[band].copy()
    if len(df) == 0:
        return None
    time_col = 'hjd' if 'hjd' in df.columns else 'mjd'
    cols = {time_col: 'time', 'mjd': 'mjd', 'mag': 'mag', 'magerr': 'magerr'}
    keep = [c for c in cols if c in df.columns]
    out = df[keep].rename(columns=cols)
    if 'time' not in out.columns and 'mjd' in out.columns:
        out['time'] = out['mjd']
    if 'ra' not in out.columns:
        out['ra'] = ztf_result.get('ra', np.nan)
    if 'dec' not in out.columns:
        out['dec'] = ztf_result.get('dec', np.nan)
    out = out.dropna(subset=['time', 'mag', 'magerr']).sort_values('time')
    return out.reset_index(drop=True)


def plot_ztf_aov_reference(ztf_result, output_dir, ra=None, dec=None,
                           target_name='', filename='ztf_aov_reference.png'):
    """
    Plot a ZTF AOV period figure in the user-provided layout.

    The plot prefers g/r when both are available, but will fall back to any
    two usable ZTF bands, or a single available ZTF band, so every ZTF period
    source can still get a consistent AOV reference image.
    """
    if ztf_result is None:
        return None
    utils.ensure_dir(output_dir)
    bands = []
    for band in ('g', 'r', 'i', 'all'):
        df = _ztf_band_dataframe(ztf_result, band)
        if df is not None and len(df) >= 5:
            bands.append((band, df))
    if not bands:
        return None

    if len(bands) >= 2:
        b1_name, b1 = bands[0]
        b2_name, b2 = bands[1]
    else:
        b1_name, b1 = bands[0]
        b2_name, b2 = bands[0]

    tref0 = min(float(b1['time'].min()), float(b2['time'].min()))
    t_1 = np.asarray(b1['time'], dtype=float) - tref0
    t_2 = np.asarray(b2['time'], dtype=float) - tref0
    y_1 = _normalize_for_aov(b1['mag'])
    y_2 = _normalize_for_aov(b2['mag'])
    dy_1 = np.asarray(b1['magerr'], dtype=float)
    dy_2 = np.asarray(b2['magerr'], dtype=float)
    t_a = np.concatenate([t_1, t_2]) if b1_name != b2_name else t_1
    mag_a = np.concatenate([np.asarray(b1['mag'], dtype=float),
                            np.asarray(b2['mag'], dtype=float)]) if b1_name != b2_name else np.asarray(b1['mag'], dtype=float)
    y_a = _normalize_for_aov(mag_a)
    dy_a = np.concatenate([dy_1, dy_2]) if b1_name != b2_name else dy_1

    entropy_1 = _aov_periodogram_reference(t_1, y_1, dy_1)
    entropy_2 = _aov_periodogram_reference(t_2, y_2, dy_2)
    entropy_a = _aov_periodogram_reference(t_a, y_a, dy_a)
    if entropy_1 is None or entropy_2 is None or entropy_a is None:
        return None

    period = entropy_2['best_period']
    phase_1 = fold_phase(t_1, entropy_1['best_period'])
    phase_2 = fold_phase(t_2, entropy_2['best_period'])

    try:
        import astropy.coordinates as coord
        c = coord.SkyCoord(
            float(b1['ra'].iloc[0]) if np.isfinite(b1['ra'].iloc[0]) else ra,
            float(b1['dec'].iloc[0]) if np.isfinite(b1['dec'].iloc[0]) else dec,
            unit='deg',
            frame='icrs')
        ztf_name = c.to_string('hmsdms', sep='', precision=2).replace(' ', '')
    except Exception:
        ztf_name = target_name or 'unknown'

    with plt.rc_context({'font.size': 13, 'font.family': 'serif'}):
        fig = plt.figure(figsize=(14, 16))
        gs = GridSpec(6, 1, height_ratios=[4.0, 1.5, 4.0, 1.0, 2.0, 2.0],
                      hspace=0)
        ax0 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[4])
        ax5 = fig.add_subplot(gs[5])

        band_colors = {'g': 'g', 'r': 'r', 'i': 'goldenrod', 'all': 'k'}
        band_markers = {'g': 'x', 'r': 'o', 'i': '^', 'all': '.'}
        plot_bands = [(b1_name, b1, 0.78), (b2_name, b2, 0.72)]
        if b1_name == b2_name:
            plot_bands = [(b1_name, b1, 0.78)]

        for band_name, data, alpha in plot_bands:
            ax0.errorbar(data['mjd'] if 'mjd' in data else data['time'],
                         data['mag'], yerr=data['magerr'],
                         c=band_colors.get(band_name, 'k'),
                         label=band_name,
                         marker=band_markers.get(band_name, '.'),
                         ms=3.0, elinewidth=0.55, alpha=alpha, ls='none')
        ax0.set_xlabel('MJD', fontsize=16)
        ax0.set_ylabel('Mag', fontsize=16)
        ax0.minorticks_on()
        ax0.tick_params(which='both', axis='both', direction='in',
                        right=True, top=True, labelsize=13)
        ax0.legend(fontsize=12, handletextpad=0.5, numpoints=3,
                   markerscale=0.9)
        ax0.invert_yaxis()
        band_count_text = '{} N = {}'.format(b1_name, len(b1))
        if b2_name != b1_name:
            band_count_text += ',  {} N = {}'.format(b2_name, len(b2))
        title_a = 'RA = {:0.6f}  Dec = {:0.6f},  {}'.format(
            float(ra if ra is not None else b1['ra'].iloc[0]),
            float(dec if dec is not None else b1['dec'].iloc[0]),
            band_count_text)
        ax0.set_title(title_a, fontsize=16, pad=8)

        for i in range(3):
            ax2.errorbar(phase_1 + float(i), b1['mag'],
                         yerr=b1['magerr'], c=band_colors.get(b1_name, 'k'),
                         label=b1_name if i == 0 else None,
                         marker=band_markers.get(b1_name, '.'),
                         ms=3.0, elinewidth=0.5,
                         alpha=0.74, ls='none')
            if b2_name != b1_name:
                ax2.errorbar(phase_2 + float(i), b2['mag'],
                             yerr=b2['magerr'],
                             c=band_colors.get(b2_name, 'k'),
                             label=b2_name if i == 0 else None,
                             marker=band_markers.get(b2_name, '.'),
                             ms=2.8, elinewidth=0.5,
                             alpha=0.66, ls='none')
        ax2.set_xlim(-0.05, 2.05)
        title = 'ZTF J{} Light Curve Folded on Period (AOV) = {:.6f} d ({:.2f} min)'.format(
            ztf_name, period, period * 24.0 * 60.0)
        ax2.set_title(title, fontsize=15, pad=8)
        ax2.legend(fontsize=12, handletextpad=0.5, numpoints=3,
                   markerscale=0.9)
        ax2.set_xlabel('Orbital Phase', fontsize=16)
        ax2.set_ylabel('Apparent Magnitude', fontsize=16)
        ax2.minorticks_on()
        ax2.tick_params(which='both', axis='both', direction='in',
                        right=True, top=True, labelbottom=True, labelsize=13)
        ax2.invert_yaxis()

        ax4.plot(1.0 / entropy_1['freqs'], entropy_1['power'], ls='-',
                 c=band_colors.get(b1_name, 'k'), lw=1.0)
        ax5.plot(1.0 / entropy_2['freqs'], entropy_2['power'], ls='-',
                 c=band_colors.get(b2_name, 'k'), lw=1.0)
        ax4.axvline(entropy_1['best_period'] / 2.0, color='k', ls='--', lw=1.0)
        ax5.axvline(entropy_2['best_period'] / 2.0, color='k', ls='--', lw=1.0)
        ax4.set_xlim(0, 20)
        ax5.set_xlim(0, 20)
        ax5.set_xlabel('Period [day]', fontsize=16)
        ax5.set_ylabel('AOV power', fontsize=14)
        ax4.set_ylabel('AOV power', fontsize=14)
        ax4.minorticks_on()
        ax5.minorticks_on()
        ax4.tick_params(which='both', axis='both', direction='in',
                        right=True, top=True, labelbottom=False, labelsize=13)
        ax5.tick_params(which='both', axis='both', direction='in',
                        right=True, top=True, labelbottom=True, labelsize=13)

        if b2_name != b1_name:
            backend_text = (
                f"backend {b1_name}={entropy_1['backend']}; "
                f"{b2_name}={entropy_2['backend']}; "
                f"combined P_orb={entropy_a['period_min']:.2f} min."
            )
        else:
            backend_text = (
                f"backend {b1_name}={entropy_1['backend']}; "
                f"P_orb={entropy_1['period_min']:.2f} min."
            )
        footer = (
            f"{backend_text} P_orb=2/f_best as in the supplied reference "
            "script; periodogram x-axis is 1/f_best."
        )
        fig.text(0.10, 0.025, footer, fontsize=9.5, color='#333')
        fig.subplots_adjust(left=0.10, right=0.97, top=0.955,
                            bottom=0.075, hspace=0.28)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=240, bbox_inches='tight', pad_inches=0.16)
        plt.close(fig)

    import pandas as pd
    rows = []
    entropy_rows = [(b1_name, entropy_1)]
    if b2_name != b1_name:
        entropy_rows.append((b2_name, entropy_2))
        entropy_rows.append(('combined', entropy_a))
    for band, ent in entropy_rows:
        rows.append({
            'band': band,
            'best_period_day_orbital_2_over_f': ent['best_period'],
            'best_period_min': ent['period_min'],
            'best_freq_day': ent['best_freq'],
            'best_power': ent['best_power'],
            'n_points': ent['n_points'],
            'backend': ent['backend'],
            'target': target_name,
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, filename.replace('.png', '.csv')), index=False)
    return path


# ================================================================
#  主流程
# ================================================================

def run_period_analysis(results, output_dir, ra=None, dec=None,
                        fap_threshold=0.01, title_prefix=''):
    """
    对所有光变曲线运行 MHAOV 周期搜索 + 联合折叠。

    Steps:
      1. 提取所有光变数据 → 统一 (time, mag, magerr) 格式
      2. 对每条曲线独立运行 mhaov()
      3. 选定参考周期 (ZTF 优先, 否则最低 FAP)
      4. 每条曲线画 periodogram + fold 图
      5. 每个检测到的周期画联合多面板折叠图
      6. 即使 FAP 不满足阈值, 也生成最佳候选周期的联合折叠图

    Returns:
        dict: curves, period_results, detections, reference_period, figures
        或 None (无光变数据)
    """
    utils.ensure_dir(output_dir)
    curves = _extract_lightcurve_data(results)
    if not curves:
        print("  周期分析: 无可用光变数据")
        return None

    print(f"  周期分析: 提取到 {len(curves)} 条光变曲线")
    for c in curves:
        n_valid = np.sum(np.isfinite(c['mag']))
        print(f"    {c['label']:12s}: {n_valid} 个有效点")
        if c.get('cleaning'):
            stats = c['cleaning']
            print(f"      cleaned {stats.get('raw_n', 0)} -> "
                  f"{stats.get('kept_n', n_valid)} points "
                  f"({stats.get('segment_count', 0)} segments)")

    tess_lk_period = None
    tess_result = results.get('TESS')
    if isinstance(tess_result, dict):
        tess_lk_period = tess_result.get('lightkurve_period_analysis')
        if tess_lk_period is None:
            try:
                from . import tess as tess_module
                tess_lk_period = tess_module.analyze_period_lightkurve(
                    tess_result, output_dir)
                if tess_lk_period is not None:
                    tess_result['lightkurve_period_analysis'] = tess_lk_period
            except Exception as exc:
                print(f"    TESS Lightkurve 周期分析跳过: {exc}")

    # 对每条曲线运行 MHAOV + Lomb-Scargle，互相检查别名/半周期问题。
    period_results = {}
    all_period_results = {}
    for curve in curves:
        label = curve['label']
        if curve.get('survey') == 'TESS' and tess_lk_period is not None:
            pr = {
                'best_period': float(tess_lk_period['best_period_day']),
                'fap': 0.0,
                'method': 'lightkurve_lombscargle',
                'quality': 'lightkurve_native',
                'agreement': 'TESS native periodogram',
                'n_points': int(tess_lk_period.get('n_points', 0)),
                'power': tess_lk_period.get('power'),
                'two_period_day': tess_lk_period.get('two_period_day'),
                'fap_note': 'Lightkurve periodogram used; FAP not calibrated here',
            }
            period_results[label] = pr
            all_period_results[label] = [pr]
            print(f"    {label}: P={pr['best_period'] * 24.0:.4f} h "
                  f"({pr['best_period']:.6f} d), Lightkurve; "
                  f"2P={pr['two_period_day'] * 24.0:.4f} h")
            continue
        t = curve['time']
        m = curve['mag']
        me = curve['magerr']
        # 清理 NaN
        valid = np.isfinite(t) & np.isfinite(m)
        if me is not None:
            me_arr = np.asarray(me, dtype=float)
            valid &= np.isfinite(me_arr)

        t_clean = t[valid]
        m_clean = m[valid]
        me_clean = me_arr[valid] if me is not None else None
        N = len(t_clean)

        # 对超大数据集 (如 TESS 200k+) 降采样以保证 MHAOV 可计算
        max_pts = 10000
        if N > max_pts:
            idx = np.sort(np.random.default_rng(42).choice(N, max_pts, replace=False))
            t_clean = t_clean[idx]
            m_clean = m_clean[idx]
            me_clean = me_clean[idx] if me_clean is not None else None
            print(f"    {label}: 降采样 {N} -> {max_pts} 用于周期搜索")

        freq_min, freq_max = _frequency_grid_for_curve(
            t_clean, curve.get('survey', ''))
        if freq_min is None:
            period_results[label] = None
            all_period_results[label] = []
            continue

        t_mhaov, m_mhaov, me_mhaov = t_clean, m_clean, me_clean
        if len(t_mhaov) > 1000:
            idx_m = np.sort(np.random.default_rng(43).choice(
                len(t_mhaov), 1000, replace=False))
            t_mhaov = t_mhaov[idx_m]
            m_mhaov = m_mhaov[idx_m]
            me_mhaov = me_mhaov[idx_m] if me_mhaov is not None else None

        pr_mhaov = utils.mhaov(t_mhaov, m_mhaov, me_mhaov, n_harmonics=3,
                               freq_min=freq_min, freq_max=freq_max,
                               n_freq=6000)
        pr_ls = utils.lomb_scargle(t_clean, m_clean, me_clean,
                                   freq_min=freq_min, freq_max=freq_max,
                                   n_freq=100000)
        pr, quality = _select_period_result(
            label, curve.get('survey', ''), [pr_mhaov, pr_ls])
        period_results[label] = pr
        all_period_results[label] = [p for p in (pr_mhaov, pr_ls) if p is not None]
        if pr is not None:
            p_h = pr['best_period'] * 24.0
            p_min = p_h * 60.0
            print(f"    {label}: P={p_h:.4f} h ({p_min:.2f} min; {pr['best_period']:.6f} d), "
                  f"{pr.get('method','?')}, FAP={pr['fap']:.2e}, quality={quality}")

    # 识别可靠检测
    detections = {}
    for label, pr in period_results.items():
        if (pr is not None and pr['fap'] < fap_threshold
                and pr.get('quality') != 'below_min_period'
                and pr.get('best_period', np.inf) >= MIN_PERIOD_DAY):
            detections[label] = pr

    # 选定参考周期: 优先 TESS/Kepler，其次 ZTF，再看其它；WISE 只作折叠验证。
    reference_period = None
    reference_source = None
    usable_detections = {
        label: pr for label, pr in detections.items()
        if pr.get('quality') not in ('fold_only', 'cadence_alias', 'below_min_period')
    }
    if usable_detections:
        for label, pr in usable_detections.items():
            support = 0
            harmonics = []
            for other_label, other_pr in usable_detections.items():
                if other_label == label:
                    continue
                rel = _period_relation(pr.get('best_period'),
                                       other_pr.get('best_period'), tol=0.08)
                if rel in ('same', 'half', 'double'):
                    support += 1
                    harmonics.append(f'{other_label}:{rel}')
            pr['cross_survey_support'] = support
            pr['cross_survey_harmonics'] = ';'.join(harmonics)
        max_support = max(pr.get('cross_survey_support', 0)
                          for pr in usable_detections.values())
        if max_support > 0:
            ref_pool = {
                label: pr for label, pr in usable_detections.items()
                if pr.get('cross_survey_support', 0) == max_support
            }
        else:
            ref_pool = usable_detections
        best_label = min(ref_pool, key=lambda k: _reference_priority(k, ref_pool[k]))
        reference_period = usable_detections[best_label]['best_period']
        reference_source = best_label

    # 个别 periodogram + fold 图
    figures = []
    morphology = {}
    for curve in curves:
        label = curve['label']
        pr = period_results.get(label)
        if pr is not None:
            morphology[label] = analyze_folded_morphology(
                curve, pr['best_period'])
            safe_name = label.replace(' ', '_').replace('/', '_')
            path = os.path.join(output_dir, f'{safe_name}_period.png')
            utils.plot_period_analysis(
                pr, curve['time'], curve['mag'], curve['magerr'],
                save_path=path, title=label)
            figures.append(path)
            print(f"    -> {os.path.basename(path)}")

    # 计算全局统一折叠零点 T0 (所有曲线中最早的有效时间, MJD)
    global_t0 = None
    for c in curves:
        t_arr = np.asarray(c['time'], dtype=float)
        t_valid = t_arr[np.isfinite(t_arr)]
        if len(t_valid) > 0:
            t_min = t_valid.min()
            if global_t0 is None or t_min < global_t0:
                global_t0 = t_min

    # 收集不同的周期 (差异 >10% 视为不同)
    distinct_periods = []
    for label, pr in detections.items():
        if pr.get('quality') in ('fold_only', 'cadence_alias', 'below_min_period'):
            continue
        p = pr['best_period']
        is_new = all(abs(p - dp) / max(p, dp) > 0.10
                     for dp, _ in distinct_periods)
        if is_new:
            distinct_periods.append((p, label))

    # 联合折叠图
    for period, source_label in distinct_periods:
        path = plot_combined_fold(
            curves, period, source_label, output_dir,
            ra=ra, dec=dec, title_prefix=title_prefix, t0=global_t0)
        if path:
            figures.append(path)
            print(f"    -> {os.path.basename(path)}")

    # 参考周期额外保障
    if reference_period is not None:
        already = any(abs(reference_period - dp) / max(reference_period, dp) < 0.10
                      for dp, _ in distinct_periods)
        if not already:
            path = plot_combined_fold(
                curves, reference_period, reference_source,
                output_dir, ra=ra, dec=dec, title_prefix=title_prefix,
                t0=global_t0)
            if path:
                figures.append(path)

    n_det = len(detections)
    orbital_period_candidate = None
    orbital_period_source = None
    if reference_period:
        for curve in curves:
            morphology[f"{curve['label']} @ reference"] = analyze_folded_morphology(
                curve, reference_period, t0=global_t0)
        ref_pr = period_results.get(reference_source, {}) if reference_source else {}
        harmonic_note = str(ref_pr.get('cross_survey_harmonics', ''))
        if reference_period < 0.1 and harmonic_note:
            # Compact WD/DWD light curves often return the strongest
            # photometric harmonic rather than the physical orbital period.
            # Save the 2x fold so the primary/secondary eclipse ambiguity can
            # be checked without rerunning the pipeline.
            orbital_period_candidate = 2.0 * reference_period
            orbital_period_source = f'{reference_source}_2x_candidate'
            path = plot_combined_fold(
                curves, orbital_period_candidate, orbital_period_source,
                output_dir, ra=ra, dec=dec, title_prefix=title_prefix,
                t0=global_t0)
            if path:
                figures.append(path)
                print(f"    -> {os.path.basename(path)} (2x 候选轨道周期)")
            for curve in curves:
                morphology[f"{curve['label']} @ 2x candidate"] = (
                    analyze_folded_morphology(
                        curve, orbital_period_candidate, t0=global_t0))
        print(f"  周期分析完成: {n_det} 个周期检测, "
              f"参考 P={reference_period * 24.0:.4f} h "
              f"({reference_period:.6f} d, {reference_source})")
        if orbital_period_candidate:
            print(f"    2x 候选轨道周期: P={orbital_period_candidate * 24.0:.4f} h "
                  f"({orbital_period_candidate:.6f} d) "
                  f"({orbital_period_source})")
    else:
        # 未达到 FAP 阈值, 但仍然生成最佳候选周期的联合折叠图供目视检查
        # 选择 FAP 最低的周期结果
        best_candidate = None
        best_fap = 1.0
        for label, pr in period_results.items():
            if (pr is not None and pr['fap'] < best_fap
                    and pr.get('quality') not in ('cadence_alias', 'below_min_period')
                    and pr.get('best_period', np.inf) >= MIN_PERIOD_DAY):
                best_fap = pr['fap']
                best_candidate = (pr['best_period'], label)

        if best_candidate is not None:
            cand_period, cand_label = best_candidate
            print(f"  周期分析完成: 未达到 FAP 阈值 ({fap_threshold}), "
                  f"最佳候选 P={cand_period * 24.0:.4f} h "
                  f"({cand_period:.6f} d, {cand_label}, FAP={best_fap:.2e})")
            reference_period = cand_period
            reference_source = cand_label
            path = plot_combined_fold(
                curves, cand_period, f'{cand_label}_candidate',
                output_dir, ra=ra, dec=dec, title_prefix=title_prefix,
                t0=global_t0)
            if path:
                figures.append(path)
                print(f"    -> {os.path.basename(path)} (候选, 需目视确认)")
            for curve in curves:
                morphology[f"{curve['label']} @ candidate"] = analyze_folded_morphology(
                    curve, cand_period, t0=global_t0)
        else:
            print(f"  周期分析完成: 未检测到显著周期 (FAP阈值={fap_threshold})")

    return {
        'curves': curves,
        'cleaning': {c['label']: c.get('cleaning', {}) for c in curves
                     if c.get('cleaning')},
        'period_results': period_results,
        'all_period_results': all_period_results,
        'detections': detections,
        'reference_period': reference_period,
        'reference_source': reference_source,
        'orbital_period_candidate': orbital_period_candidate,
        'orbital_period_source': orbital_period_source,
        'distinct_periods': distinct_periods,
        'figures': figures,
        'morphology': morphology,
    }


def save_csv(pa_result, output_dir):
    """保存周期分析结果为 CSV"""
    import pandas as pd
    if pa_result is None or output_dir is None:
        return None

    rows = []
    for label, pr in pa_result.get('period_results', {}).items():
        if pr is not None:
            rows.append({
                'curve': label,
                'best_period_day': pr['best_period'],
                'best_period_hour': pr['best_period'] * 24,
                'best_period_min': pr['best_period'] * 24 * 60,
                'fap': pr['fap'],
                'n_points': pr.get('n_points'),
                'detected': (
                    pr['fap'] < 0.01
                    and pr.get('best_period', np.inf) >= MIN_PERIOD_DAY
                    and pr.get('quality') != 'below_min_period'
                ),
                'method': pr.get('method'),
                'quality': pr.get('quality'),
                'agreement': pr.get('agreement'),
                'two_period_day': pr.get('two_period_day'),
                'two_period_hour': (
                    pr.get('two_period_day') * 24
                    if pr.get('two_period_day') is not None else np.nan
                ),
                'fap_note': pr.get('fap_note', ''),
                'alias_warning': pr.get('alias_warning', ''),
                'cross_survey_support': pr.get('cross_survey_support'),
                'cross_survey_harmonics': pr.get('cross_survey_harmonics', ''),
                'candidate_methods': pr.get('candidate_methods'),
            })
    for key, morph in pa_result.get('morphology', {}).items():
        if not morph:
            continue
        rows.append({
            'curve': morph.get('curve') or key,
            'best_period_day': morph.get('period_day'),
            'best_period_hour': (
                morph.get('period_day') * 24
                if morph.get('period_day') is not None else np.nan
            ),
            'best_period_min': morph.get('period_min'),
            'fap': np.nan,
            'n_points': morph.get('n_points'),
            'detected': False,
            'morphology': morph.get('morphology'),
            'amplitude_mag': morph.get('amplitude_mag'),
            'asymmetry': morph.get('asymmetry'),
            'dip_duty_cycle': morph.get('dip_duty_cycle'),
            'primary_dip_phase': morph.get('primary_dip_phase'),
            'smoothness': morph.get('smoothness'),
            })
    for label, stats in pa_result.get('cleaning', {}).items():
        row = {'curve': label, 'best_period_day': np.nan,
               'best_period_hour': np.nan,
               'best_period_min': np.nan, 'fap': np.nan,
               'detected': False, 'method': 'cleaning_summary'}
        row.update(stats)
        rows.append(row)
    if pa_result.get('orbital_period_candidate'):
        p = pa_result.get('orbital_period_candidate')
        rows.append({
            'curve': pa_result.get('orbital_period_source', '2x_candidate'),
            'best_period_day': p,
            'best_period_hour': p * 24,
            'best_period_min': p * 24 * 60,
            'fap': np.nan,
            'detected': False,
            'method': '2x_harmonic_candidate',
            'quality': 'inspect_fold',
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'period_analysis.csv')
    df.to_csv(path, index=False)
    return path
