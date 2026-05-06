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
    freq_max = min(0.5 / dt_ref, 720.0)
    if survey == 'WISE':
        # WISE 的半年采样窗函数很强，避免把极高频随机别名当成参考周期。
        freq_max = min(freq_max, 80.0)
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
        mag, magerr = flux_to_relative_mag(tess['flux'], tess.get('flux_err'))
        tsys = tess.get('time_system', 'BTJD')
        curves.append({
            'time': _to_mjd(tess['time'], tsys),
            'mag': mag, 'magerr': magerr,
            'label': 'TESS', 'survey': 'TESS', 'band': 'T',
        })

    # --- Kepler/K2 (BKJD/BTJD → MJD) ---
    kep = results.get('Kepler/K2')
    if kep and kep.get('n_points', 0) >= min_pts_space:
        mag, magerr = flux_to_relative_mag(kep['flux'], kep.get('flux_err'))
        tsys = kep.get('time_system', 'BKJD')
        curves.append({
            'time': _to_mjd(kep['time'], tsys),
            'mag': mag, 'magerr': magerr,
            'label': kep.get('survey', 'Kepler'), 'survey': 'Kepler', 'band': 'Kp',
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

        if me is not None and len(me) > 0:
            ax.errorbar(phase, m, yerr=me, fmt='.', color=color,
                       ms=3, elinewidth=0.35, alpha=0.45)
            ax.errorbar(phase + 1, m, yerr=me, fmt='.', color=color,
                       ms=3, elinewidth=0.35, alpha=0.18)
        else:
            ax.scatter(phase, m, s=8, c=color, alpha=0.45,
                       edgecolors='none')
            ax.scatter(phase + 1, m, s=8, c=color, alpha=0.18,
                       edgecolors='none')

        bx, by, counts = _phase_bin_median(phase, m, n_bins=70)
        if len(bx) >= 3:
            ax.plot(bx, by, '-', color='black', lw=1.8, alpha=0.9,
                    label='binned median')
            ax.plot(bx + 1, by, '-', color='black', lw=1.8, alpha=0.9)
            ax.scatter(bx, by, s=np.clip(counts, 8, 45), c='white',
                       edgecolors='black', linewidths=0.5, zorder=5)
            ax.scatter(bx + 1, by, s=np.clip(counts, 8, 45), c='white',
                       edgecolors='black', linewidths=0.5, zorder=5)
        if morph and np.isfinite(morph.get('primary_dip_phase', np.nan)):
            ph0 = morph['primary_dip_phase']
            ax.axvline(ph0, color='crimson', lw=0.8, ls='--', alpha=0.6)
            ax.axvline(ph0 + 1, color='crimson', lw=0.8, ls='--', alpha=0.6)

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
        else:
            label = f'N={len(m)}'
        ax.text(0.99, 0.94, label, transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='#333',
                bbox=dict(facecolor='white', alpha=0.72, edgecolor='none',
                          boxstyle='round,pad=0.2'))

    axes[-1].set_xlabel('Phase', fontsize=11)
    coord = f'  RA={ra:.4f} DEC={dec:.4f}' if ra is not None else ''
    title = f'{title_prefix}Phase-folded at P = {period:.6f} d  (from {period_source}){coord}'
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    safe_src = period_source.replace(' ', '_')
    path = os.path.join(output_dir, f'combined_fold_P{period:.4f}d_{safe_src}.png')
    utils.save_and_close(fig, path)
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

    # 对每条曲线运行 MHAOV + Lomb-Scargle，互相检查别名/半周期问题。
    period_results = {}
    all_period_results = {}
    for curve in curves:
        label = curve['label']
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
            p_min = pr['best_period'] * 24 * 60
            print(f"    {label}: P={pr['best_period']:.6f} d ({p_min:.2f} min), "
                  f"{pr.get('method','?')}, FAP={pr['fap']:.2e}, quality={quality}")

    # 识别可靠检测
    detections = {}
    for label, pr in period_results.items():
        if pr is not None and pr['fap'] < fap_threshold:
            detections[label] = pr

    # 选定参考周期: 优先 TESS/Kepler，其次 ZTF，再看其它；WISE 只作折叠验证。
    reference_period = None
    reference_source = None
    usable_detections = {
        label: pr for label, pr in detections.items()
        if pr.get('quality') != 'fold_only'
    }
    if usable_detections:
        best_label = min(usable_detections,
                         key=lambda k: _reference_priority(k, usable_detections[k]))
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
    if reference_period:
        for curve in curves:
            morphology[f"{curve['label']} @ reference"] = analyze_folded_morphology(
                curve, reference_period, t0=global_t0)
        print(f"  周期分析完成: {n_det} 个周期检测, "
              f"参考 P={reference_period:.6f} d ({reference_source})")
    else:
        # 未达到 FAP 阈值, 但仍然生成最佳候选周期的联合折叠图供目视检查
        # 选择 FAP 最低的周期结果
        best_candidate = None
        best_fap = 1.0
        for label, pr in period_results.items():
            if pr is not None and pr['fap'] < best_fap:
                best_fap = pr['fap']
                best_candidate = (pr['best_period'], label)

        if best_candidate is not None:
            cand_period, cand_label = best_candidate
            print(f"  周期分析完成: 未达到 FAP 阈值 ({fap_threshold}), "
                  f"最佳候选 P={cand_period:.6f} d ({cand_label}, FAP={best_fap:.2e})")
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
        'period_results': period_results,
        'all_period_results': all_period_results,
        'detections': detections,
        'reference_period': reference_period,
        'reference_source': reference_source,
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
                'best_period_min': pr['best_period'] * 24 * 60,
                'fap': pr['fap'],
                'n_points': pr.get('n_points'),
                'detected': pr['fap'] < 0.01,
                'method': pr.get('method'),
                'quality': pr.get('quality'),
                'agreement': pr.get('agreement'),
                'candidate_methods': pr.get('candidate_methods'),
            })
    for key, morph in pa_result.get('morphology', {}).items():
        if not morph:
            continue
        rows.append({
            'curve': morph.get('curve') or key,
            'best_period_day': morph.get('period_day'),
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
    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'period_analysis.csv')
    df.to_csv(path, index=False)
    return path
