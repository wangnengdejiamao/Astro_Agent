"""
联合绘图模块
=============
将多波段光谱、测光、光变曲线统一绘制在一起。

1. plot_combined_spectra:  所有光谱画在一张图上 (统一 erg/s/cm²/A)
2. plot_combined_fold:     所有测光按周期折叠在一张图上
3. plot_spectra_with_photometry: 光谱 + 宽带测光 + 黑体拟合

用法:
    from astro_toolbox.combined_plots import (
        plot_combined_spectra, plot_combined_fold,
        plot_spectra_with_photometry,
    )
    plot_combined_spectra(results, save_path='combined_spectra.png')
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config, utils


# ================================================================
#  全局颜色表
# ================================================================

SPEC_COLORS = {
    'SDSS':    '#1f77b4',
    'DESI':    '#d62728',
    'HST':     '#9467bd',
    'JWST':    '#8c564b',
    'SPHEREx': '#e377c2',
    'KOA/LRIS': '#111111',
}

PHOT_COLORS = {
    'GALEX': 'purple', 'SDSS': 'blue', 'Gaia': 'green',
    '2MASS': 'orange', 'WISE': 'red', 'SPHEREx': 'darkorange',
}

LC_COLORS = {
    'ZTF g': 'green', 'ZTF r': 'red', 'ZTF i': 'goldenrod',
    'WISE W1': 'dodgerblue', 'WISE W2': 'orangered',
    'TESS': 'purple', 'Kepler': 'darkcyan', 'K2': 'teal',
    'Gaia G': 'darkgreen', 'Gaia BP': 'royalblue', 'Gaia RP': 'firebrick',
    'HST': 'violet', 'JWST': 'brown',
}

LC_MARKERS = {
    'ZTF g': 'o', 'ZTF r': 's', 'ZTF i': '^',
    'WISE W1': 'D', 'WISE W2': 'v',
    'TESS': '.', 'Kepler': '.', 'K2': '.',
    'Gaia G': 'p', 'Gaia BP': 'h', 'Gaia RP': 'H',
}


# ================================================================
#  流量单位统一
# ================================================================

def _extract_spectrum_cgs(results, key, label):
    """
    从 results 中提取光谱, 统一转换为 (wave_A, flux_cgs, err_cgs)。
    flux_cgs: erg/s/cm²/A

    SDSS / DESI 的原始流量单位是 1e-17 erg/s/cm²/A,
    HST / JWST / SPHEREx 已经是 erg/s/cm²/A。
    """
    r = results.get(key)
    if r is None:
        return None

    NEEDS_SCALE = {'SDSS_spectrum', 'LAMOST_spectrum'}

    if key == 'DESI' and isinstance(r, dict) and 'spectrum' in r:
        sp = r['spectrum']
        all_w, all_f, all_e = [], [], []
        for band in ('B', 'R', 'Z'):
            if band not in sp:
                continue
            w = np.asarray(sp[band]['wavelength'], dtype=float)
            f = np.asarray(sp[band]['flux'], dtype=float) * 1e-17
            e = np.asarray(sp[band]['error'], dtype=float) * 1e-17
            all_w.append(w)
            all_f.append(f)
            all_e.append(e)
        if not all_w:
            return None
        wave = np.concatenate(all_w)
        flux = np.concatenate(all_f)
        error = np.concatenate(all_e)
        info = sp.get('obs_mjd', '')
        return wave, flux, error, f'{label} (MJD {info})' if info else label

    if 'wavelength' not in r and 'wavelength' not in (r or {}):
        return None

    wave = np.asarray(r['wavelength'], dtype=float)
    flux = np.asarray(r['flux'], dtype=float)
    error = np.asarray(r.get('error', np.zeros_like(flux)), dtype=float)

    if key in NEEDS_SCALE:
        flux = flux * 1e-17
        error = error * 1e-17

    # 构造标签
    parts = [label]
    if r.get('obs_id'):
        parts.append(str(r['obs_id']))
    elif r.get('instrument'):
        parts.append(str(r['instrument']))
    tag = '  '.join(parts)

    return wave, flux, error, tag


# ================================================================
#  1. 所有光谱画一起
# ================================================================

def plot_combined_spectra(results, save_path=None, ra=None, dec=None):
    """
    将所有可用光谱绘制在同一张图上。

    所有流量统一为 erg/s/cm²/A。
    """
    spectra = []
    spec_keys = [
        ('SDSS_spectrum', 'SDSS'),
        ('DESI',          'DESI'),
        ('KOA_spectrum',  'KOA/LRIS'),
        ('HST_spectrum',  'HST'),
        ('JWST_spectrum', 'JWST'),
        ('SPHEREx',       'SPHEREx'),
    ]

    for key, label in spec_keys:
        extracted = _extract_spectrum_cgs(results, key, label)
        if extracted is not None:
            wave, flux, error, tag = extracted
            # 过滤无效
            mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
            if mask.sum() > 0:
                spectra.append({
                    'wave': wave[mask], 'flux': flux[mask],
                    'error': error[mask], 'label': tag,
                    'color': SPEC_COLORS.get(label, 'gray'),
                })

    if not spectra:
        print("  联合光谱: 无可用光谱数据")
        return None

    fig, ax = plt.subplots(figsize=(16, 7))

    for sp in spectra:
        w, f = sp['wave'], sp['flux']
        # 排序
        idx = np.argsort(w)
        w, f = w[idx], f[idx]
        ax.plot(w, f, '-', color=sp['color'], lw=0.7, alpha=0.85,
                label=sp['label'])

    ax.set_xlabel('Wavelength (A)', fontsize=13)
    ax.set_ylabel(r'$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=13)

    # 自动判断是否用对数纵轴, 并设置紧凑的轴范围
    all_wave = np.concatenate([sp['wave'] for sp in spectra])
    all_flux = np.concatenate([sp['flux'][sp['flux'] > 0] for sp in spectra
                               if np.any(sp['flux'] > 0)])
    if len(all_flux) > 0:
        fmin, fmax = np.percentile(all_flux[all_flux > 0], [1, 99])
        if fmax / max(fmin, 1e-30) > 100:
            ax.set_yscale('log')
            ax.set_ylim(fmin * 0.3, fmax * 3)
        else:
            df = (fmax - fmin) * 0.1
            ax.set_ylim(fmin - df, fmax + df)

    # x 轴按真正的连续光谱范围收紧。SPHEREx 这类只有少数宽带点的
    # 稀疏谱不再把 SDSS/DESI/HST 的可读范围拉到几万埃。
    dense_wave = [
        sp['wave'] for sp in spectra
        if np.isfinite(sp['wave']).sum() >= 20
    ]
    if dense_wave:
        x_wave = np.concatenate(dense_wave)
    else:
        x_wave = all_wave
    w_valid = x_wave[np.isfinite(x_wave) & (x_wave > 0)]
    if len(w_valid) > 0:
        wmin, wmax = w_valid.min(), w_valid.max()
        dw = (wmax - wmin) * 0.02
        ax.set_xlim(wmin - dw, wmax + dw)

    title = 'Combined Spectra'
    if ra is not None:
        title += f'  RA={ra:.4f} DEC={dec:.4f}'
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    print(f"  联合光谱: {len(spectra)} surveys -> {save_path}")
    return fig


# ================================================================
#  2. 所有测光折叠周期画一起
# ================================================================

def plot_combined_fold(results, save_path=None, ra=None, dec=None):
    """
    将所有光变曲线按检测到的周期折叠在同一张图上。

    上面板: 星等型 (ZTF, WISE, Gaia) — 共享 phase 轴, 各自 Y 归一化
    下面板: 流量型 (TESS, Kepler) — 差分星等
    """
    from .period_analysis import (
        _extract_lightcurve_data, _phase_bin_median,
        _fit_harmonic_phase_curve, _evaluate_harmonic_phase_curve,
        _harmonic_count_for_curve,
        analyze_folded_morphology)

    # 获取周期分析结果
    pa = results.get('period_analysis')
    if pa is None or pa.get('reference_period') is None:
        print("  联合折叠: 无周期检测结果")
        return None

    period = pa['reference_period']
    source = pa.get('reference_source', '?')
    curves = pa.get('curves', [])

    if not curves:
        curves = _extract_lightcurve_data(results)
    if not curves:
        print("  联合折叠: 无可用光变数据")
        return None

    # 全局 T0
    t0 = None
    for c in curves:
        t_arr = np.asarray(c['time'], dtype=float)
        t_valid = t_arr[np.isfinite(t_arr)]
        if len(t_valid) > 0:
            tmin = t_valid.min()
            t0 = tmin if t0 is None else min(t0, tmin)
    if t0 is None:
        t0 = 0.0

    fig, ax = plt.subplots(figsize=(14, 7))

    plotted = 0
    for curve in curves:
        t = np.asarray(curve['time'], dtype=float)
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

        if len(t) < 3:
            continue

        # 归一化为差分星等 (减去中位值)
        med = np.median(m)
        dm = m - med

        phase = ((t - t0) / period) % 1.0
        morph = analyze_folded_morphology(curve, period, t0=t0)

        label_str = curve['label']
        color = LC_COLORS.get(label_str, 'black')
        marker = LC_MARKERS.get(label_str, '.')
        ms = 5 if len(t) > 100 else 7

        has_errors = (me is not None and len(me) > 0)
        if has_errors:
            if len(dm) > 6000:
                ax.scatter(phase, dm, s=5.0, c=color, marker=marker,
                           alpha=0.055,
                           label=f'{label_str} (N={len(t)}, med={med:.2f})')
                ax.scatter(phase + 1, dm, s=5.0, c=color, marker=marker,
                           alpha=0.028)
                rng = np.random.default_rng(12345)
                idx = np.sort(rng.choice(len(dm), 6000, replace=False))
                ax.errorbar(phase[idx], dm[idx], yerr=me[idx], fmt=marker,
                            color=color, ms=3.4, elinewidth=0.25,
                            alpha=0.18, capsize=0)
                ax.errorbar(phase[idx] + 1, dm[idx], yerr=me[idx], fmt=marker,
                            color=color, ms=3.4, elinewidth=0.25,
                            alpha=0.09, capsize=0)
            else:
                ax.errorbar(phase, dm, yerr=me, fmt=marker,
                            color=color, ms=ms, elinewidth=0.42,
                            alpha=0.50, label=f'{label_str} (N={len(t)}, med={med:.2f})')
                # 重复相位
                ax.errorbar(phase + 1, dm, yerr=me, fmt=marker,
                            color=color, ms=ms, elinewidth=0.42, alpha=0.22)
        else:
            if len(dm) > 5000:
                size, alpha1, alpha2 = 5.0, 0.055, 0.028
            elif len(dm) > 1000:
                size, alpha1, alpha2 = 14.0, 0.22, 0.10
            else:
                size, alpha1, alpha2 = ms**2, 0.50, 0.22
            ax.scatter(phase, dm, s=size, c=color, marker=marker,
                       alpha=alpha1,
                       label=f'{label_str} (N={len(t)}, med={med:.2f})')
            ax.scatter(phase + 1, dm, s=size, c=color, marker=marker,
                       alpha=alpha2)
        fit = _fit_harmonic_phase_curve(
            phase, dm, me, n_harmonics=_harmonic_count_for_curve(len(dm)))
        if fit:
            grid = np.linspace(0.0, 2.0, 800)
            model = _evaluate_harmonic_phase_curve(fit, grid)
            if model is not None:
                ax.plot(grid, model, '-', color=color, lw=2.0, alpha=0.9)

        bx, by, counts = _phase_bin_median(phase, dm, n_bins=55)
        if len(bx) >= 3:
            ax.scatter(bx, by, s=np.clip(counts * 1.5, 20, 72), c='white',
                       edgecolors=color, linewidths=0.75, zorder=6)
            ax.scatter(bx + 1, by, s=np.clip(counts * 1.5, 20, 72), c='white',
                       edgecolors=color, linewidths=0.75, zorder=6)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("  联合折叠: 无有效数据")
        return None

    ax.set_xlabel('Phase', fontsize=13)
    ax.set_ylabel(r'$\Delta$ mag (from median)', fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 2)
    ax.grid(True, alpha=0.3)

    period_h = period * 24.0
    title = f'Phase-folded at P = {period_h:.4f} h ({period:.6f} d; from {source})'
    if ra is not None:
        title += f'  RA={ra:.4f} DEC={dec:.4f}'
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc='best', ncol=2)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    print(f"  联合折叠: {plotted} bands, P={period_h:.4f}h -> {save_path}")
    return fig


# ================================================================
#  3. 光谱 + 测光 (SED) 画一起
# ================================================================

def plot_spectra_with_photometry(results, save_path=None, ra=None, dec=None):
    """
    将所有光谱和宽带测光点绘制在同一张 log-log 图上。

    光谱: 细线 (各巡天不同颜色)
    测光: 大圆点 + 误差棒
    黑体拟合: 虚线 (如果有)
    所有流量单位: erg/s/cm²/A
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    has_data = False

    # --- 光谱 ---
    spec_keys = [
        ('SDSS_spectrum', 'SDSS'),
        ('DESI',          'DESI'),
        ('KOA_spectrum',  'KOA/LRIS'),
        ('HST_spectrum',  'HST'),
        ('JWST_spectrum', 'JWST'),
        ('SPHEREx',       'SPHEREx'),
    ]

    for key, label in spec_keys:
        extracted = _extract_spectrum_cgs(results, key, label)
        if extracted is None:
            continue
        wave, flux, error, tag = extracted
        mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0) & (flux > 0)
        if mask.sum() < 2:
            continue
        w, f = wave[mask], flux[mask]
        idx = np.argsort(w)
        ax.plot(w[idx], f[idx], '-', color=SPEC_COLORS.get(label, 'gray'),
                lw=0.6, alpha=0.7, label=f'{tag} (spec)')
        has_data = True

    # --- 宽带测光 (from SED fitter) ---
    sed_fitter = results.get('SED')
    if sed_fitter is not None and hasattr(sed_fitter, 'flux_data'):
        for band_name, (flux, flux_err, wave_A) in sed_fitter.flux_data.items():
            if flux <= 0 or not np.isfinite(flux):
                continue
            prefix = band_name.split('_')[0]
            c = PHOT_COLORS.get(prefix, 'gray')
            ax.errorbar(wave_A, flux, yerr=flux_err,
                        fmt='o', color=c, markersize=10, capsize=4,
                        zorder=20, markeredgecolor='white', markeredgewidth=0.5)
            ax.annotate(band_name.replace('_', ' '),
                        (wave_A, flux), fontsize=7,
                        rotation=30, ha='left', va='bottom',
                        xytext=(3, 5), textcoords='offset points')
            has_data = True

        # 黑体拟合曲线 (暂未实现)
        pass

    if not has_data:
        plt.close(fig)
        print("  光谱+测光: 无可用数据")
        return None

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength (A)', fontsize=13)
    ax.set_ylabel(r'$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=13)

    title = 'Spectra + Photometry (SED)'
    if ra is not None:
        title += f'  RA={ra:.4f} DEC={dec:.4f}'
    if sed_fitter and hasattr(sed_fitter, 'ebv') and sed_fitter.ebv is not None:
        title += f'  E(B-V)={sed_fitter.ebv:.3f}'
    ax.set_title(title, fontsize=14)

    # 收集所有数据的波长/流量范围用于设置轴和背景色
    all_w, all_f = [], []
    # 从光谱收集
    for key, label in spec_keys:
        extracted = _extract_spectrum_cgs(results, key, label)
        if extracted is None:
            continue
        wave, flux, error, tag = extracted
        mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0) & (flux > 0)
        if mask.any():
            all_w.append(wave[mask])
            all_f.append(flux[mask])
    # 从测光收集
    if sed_fitter is not None and hasattr(sed_fitter, 'flux_data'):
        for band_name, (flux, flux_err, wave_A) in sed_fitter.flux_data.items():
            if flux > 0 and np.isfinite(flux):
                all_w.append(np.array([wave_A]))
                all_f.append(np.array([flux]))

    if all_w:
        cw = np.concatenate(all_w)
        cf = np.concatenate(all_f)
        w_min, w_max = cw.min(), cw.max()
        f_min, f_max = cf.min(), cf.max()
        ax.set_xlim(w_min * 0.5, w_max * 2)
        ax.set_ylim(f_min * 0.3, f_max * 5)

        # 只绘制与数据范围重叠的波段区域背景色
        plot_xmin, plot_xmax = w_min * 0.5, w_max * 2
        band_regions = [
            (912,    3000,   'purple'),
            (3000,   10000,  'blue'),
            (10000,  50000,  'orange'),
            (50000,  300000, 'red'),
        ]
        for lo, hi, clr in band_regions:
            if hi > plot_xmin and lo < plot_xmax:
                ax.axvspan(max(lo, plot_xmin), min(hi, plot_xmax),
                           alpha=0.03, color=clr)

    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    print(f"  光谱+测光: -> {save_path}")
    return fig
