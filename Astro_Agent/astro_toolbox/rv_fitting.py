"""
径向速度 (RV) 拟合模块
=======================
1. 交叉相关法 (CCF): 观测光谱 × WD 模版 → 径向速度
2. 单星 RV: CCF 峰值 → 单高斯拟合
3. 双线双星 (SB2) RV: CCF → 双高斯拟合 → 两组分分别的 RV
4. Pipeline RV: 从 SDSS z / DESI redrock 提取

用法:
    from astro_toolbox.rv_fitting import measure_rv, measure_rv_binary
    rv = measure_rv(wavelength, flux, err)
    rv_result = measure_rv_binary(wavelength, flux, err)
"""
import numpy as np
from scipy.signal import correlate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config, utils

# 光速 km/s
C_KMS = 2.99792458e5


def _optical_ccf_coverage(wave, require_balmer=False):
    """Return whether a spectrum is appropriate for optical WD-template CCF."""
    try:
        wave = np.asarray(wave, dtype=float)
    except Exception:
        return {'usable': False, 'reason': 'invalid wavelength array'}
    wave = wave[np.isfinite(wave)]
    if wave.size < 50:
        return {'usable': False, 'reason': 'too few finite wavelength pixels'}
    wmin = float(np.nanmin(wave))
    wmax = float(np.nanmax(wave))
    optical_overlap = (wmax >= 3700.0 and wmin <= 9200.0)
    balmer = [6562.8, 4861.3, 4340.5, 4101.7]
    n_balmer = int(sum(wmin <= line <= wmax for line in balmer))
    if not optical_overlap:
        return {
            'usable': False,
            'reason': f'outside optical WD-template range ({wmin:.1f}-{wmax:.1f} A)',
            'wavelength_min_A': wmin,
            'wavelength_max_A': wmax,
            'n_balmer_lines_covered': n_balmer,
        }
    if require_balmer and n_balmer < 2:
        return {
            'usable': False,
            'reason': f'insufficient Balmer coverage for optical RV ({n_balmer} lines)',
            'wavelength_min_A': wmin,
            'wavelength_max_A': wmax,
            'n_balmer_lines_covered': n_balmer,
        }
    return {
        'usable': True,
        'reason': '',
        'wavelength_min_A': wmin,
        'wavelength_max_A': wmax,
        'n_balmer_lines_covered': n_balmer,
    }


# ================================================================
#  核心 CCF 函数
# ================================================================

def _rebin_to_logwave(wave, flux, err=None, dv=30.0):
    """
    将光谱重采样到等速度间距的 log-wavelength 网格。

    Parameters
    ----------
    wave : array, Angstrom
    flux : array
    err  : array or None
    dv   : float, km/s, 速度像素宽度 (默认 30 km/s)

    Returns
    -------
    ln_wave : array, ln(wavelength) 等间距
    flux_rebin : array
    err_rebin : array or None
    """
    mask = np.isfinite(flux)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)
    wave, flux = wave[mask], flux[mask]
    if err is not None:
        err = err[mask]
    if len(wave) < 50:
        return None, None, None

    # ln(wave) 等间距网格
    dlnw = dv / C_KMS
    ln_w_min = np.log(wave.min())
    ln_w_max = np.log(wave.max())
    n_pix = int((ln_w_max - ln_w_min) / dlnw) + 1
    ln_wave = np.linspace(ln_w_min, ln_w_max, n_pix)

    # 插值到新网格
    f_interp = interp1d(np.log(wave), flux, kind='linear',
                        bounds_error=False, fill_value=0.0)
    flux_rebin = f_interp(ln_wave)

    err_rebin = None
    if err is not None:
        e_interp = interp1d(np.log(wave), err, kind='linear',
                            bounds_error=False, fill_value=1.0)
        err_rebin = e_interp(ln_wave)

    return ln_wave, flux_rebin, err_rebin


def _normalize_spectrum(flux):
    """连续谱归一化: 用滑动中位数"""
    from scipy.ndimage import median_filter
    if len(flux) < 100:
        continuum = np.median(flux)
    else:
        # 窗口大小 ~ 5% 的光谱长度
        win = max(51, len(flux) // 20)
        if win % 2 == 0:
            win += 1
        continuum = median_filter(flux, size=win)
        # 避免除零
        continuum = np.maximum(continuum, np.percentile(flux[flux > 0], 5) * 0.1
                               if np.any(flux > 0) else 1e-30)
    return flux / continuum - 1.0  # 去连续谱，变成吸收/发射特征


def compute_ccf(obs_wave, obs_flux, obs_err, tmpl_wave, tmpl_flux,
                v_min=-1000, v_max=1000, dv=30.0):
    """
    计算观测光谱与模版的交叉相关函数 (CCF)。
    使用 FFT 加速。

    Parameters
    ----------
    obs_wave, obs_flux, obs_err : 观测光谱 (Angstrom, flux, error)
    tmpl_wave, tmpl_flux : 模版光谱
    v_min, v_max : 速度搜索范围 (km/s)
    dv : 速度步长 (km/s)

    Returns
    -------
    velocities : array, km/s
    ccf : array, 归一化 CCF 值
    """
    # 重采样到公共 log-wave 网格
    ln_w_obs, f_obs, e_obs = _rebin_to_logwave(obs_wave, obs_flux, obs_err, dv=dv)
    ln_w_tmpl, f_tmpl, _ = _rebin_to_logwave(tmpl_wave, tmpl_flux, dv=dv)

    if ln_w_obs is None or ln_w_tmpl is None:
        return None, None

    # 找重叠的 ln(wave) 范围
    ln_min = max(ln_w_obs[0], ln_w_tmpl[0])
    ln_max = min(ln_w_obs[-1], ln_w_tmpl[-1])

    # 截取重叠区
    obs_mask = (ln_w_obs >= ln_min) & (ln_w_obs <= ln_max)
    tmpl_mask = (ln_w_tmpl >= ln_min) & (ln_w_tmpl <= ln_max)

    if obs_mask.sum() < 50 or tmpl_mask.sum() < 50:
        return None, None

    # 插值模版到观测的 log-wave 网格上
    ln_w = ln_w_obs[obs_mask]
    f_o = f_obs[obs_mask]
    f_t = np.interp(ln_w, ln_w_tmpl, f_tmpl, left=0, right=0)

    # 归一化 (去连续谱)
    f_o = _normalize_spectrum(f_o)
    f_t = _normalize_spectrum(f_t)

    # 去均值
    f_o = f_o - np.mean(f_o)
    f_t = f_t - np.mean(f_t)

    # FFT 交叉相关
    n = len(f_o)
    fft_o = np.fft.rfft(f_o, n=2 * n)
    fft_t = np.fft.rfft(f_t, n=2 * n)
    cc = np.fft.irfft(fft_o * np.conj(fft_t))

    # 归一化
    norm = np.sqrt(np.sum(f_o ** 2) * np.sum(f_t ** 2))
    if norm <= 0:
        return None, None
    cc = cc / norm

    # 转换 lag → velocity
    dlnw = dv / C_KMS
    lags = np.arange(len(cc))
    lags[lags > n] -= 2 * n
    velocities_all = lags * dlnw * C_KMS

    # 截取感兴趣的速度范围
    v_mask = (velocities_all >= v_min) & (velocities_all <= v_max)
    if v_mask.sum() < 10:
        return None, None

    # 按速度排序
    idx = np.argsort(velocities_all[v_mask])
    velocities = velocities_all[v_mask][idx]
    ccf = cc[v_mask][idx]

    return velocities, ccf


# ================================================================
#  高斯拟合 CCF 峰值
# ================================================================

def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    return (_gaussian(x, amp1, mu1, sigma1) +
            _gaussian(x, amp2, mu2, sigma2))


def _fit_ccf_peak(velocities, ccf, v_guess=None):
    """
    对 CCF 峰值做单高斯拟合 → RV, RV_err, CCF_height

    Returns
    -------
    dict: {rv, rv_err, ccf_height, sigma_v, success}
    """
    if velocities is None or ccf is None or len(ccf) < 10:
        return None

    # 找最大值
    if v_guess is None:
        i_peak = np.argmax(ccf)
    else:
        # 在 v_guess 附近找
        near = np.abs(velocities - v_guess) < 200
        if near.sum() > 0:
            sub_idx = np.where(near)[0]
            i_peak = sub_idx[np.argmax(ccf[near])]
        else:
            i_peak = np.argmax(ccf)

    v_peak = velocities[i_peak]
    ccf_peak = ccf[i_peak]

    if ccf_peak < 0.05:
        return None

    # 取峰值附近 ±200 km/s 的数据做高斯拟合
    fit_mask = np.abs(velocities - v_peak) < 200
    v_fit = velocities[fit_mask]
    c_fit = ccf[fit_mask]

    if len(v_fit) < 5:
        return None

    try:
        popt, pcov = curve_fit(_gaussian, v_fit, c_fit,
                               p0=[ccf_peak, v_peak, 50.0],
                               bounds=([0, v_peak - 200, 5],
                                       [1.5, v_peak + 200, 500]),
                               maxfev=3000)
        perr = np.sqrt(np.diag(pcov))
        return {
            'rv': popt[1],
            'rv_err': perr[1],
            'ccf_height': popt[0],
            'sigma_v': abs(popt[2]),
            'success': True,
        }
    except Exception:
        # 退回到直接用峰值
        return {
            'rv': v_peak,
            'rv_err': abs(velocities[1] - velocities[0]) * 2,
            'ccf_height': ccf_peak,
            'sigma_v': 50.0,
            'success': False,
        }


def _fit_ccf_double_peak(velocities, ccf):
    """
    对 CCF 做双高斯拟合 → 两个 RV (SB2 双线双星)

    Returns
    -------
    dict: {rv1, rv1_err, rv2, rv2_err, ccf_height1, ccf_height2,
           sigma1, sigma2, dv, success}
    or None if double peak not significant
    """
    if velocities is None or ccf is None or len(ccf) < 20:
        return None

    # 先找主峰
    i_peak1 = np.argmax(ccf)
    v1 = velocities[i_peak1]
    h1 = ccf[i_peak1]

    if h1 < 0.1:
        return None

    # 遮住主峰 (±100 km/s)，找次峰
    mask2 = np.abs(velocities - v1) > 100
    if mask2.sum() < 10:
        return None
    ccf_masked = ccf.copy()
    ccf_masked[~mask2] = 0

    i_peak2 = np.argmax(ccf_masked)
    v2 = velocities[i_peak2]
    h2 = ccf_masked[i_peak2]

    # 次峰太弱则不算双星 (需要 CCF > 0.1 且 > 主峰的 25%)
    if h2 < 0.25 * h1 or h2 < 0.10:
        return None

    # 双高斯拟合
    try:
        p0 = [h1, v1, 50.0, h2, v2, 50.0]
        bounds_lo = [0, v1 - 200, 5, 0, v2 - 200, 5]
        bounds_hi = [1.5, v1 + 200, 500, 1.5, v2 + 200, 500]
        popt, pcov = curve_fit(_double_gaussian, velocities, ccf,
                               p0=p0, bounds=(bounds_lo, bounds_hi),
                               maxfev=5000)
        perr = np.sqrt(np.diag(pcov))

        rv_a, rv_b = popt[1], popt[4]
        # 保证 rv_a < rv_b (按速度排序)
        if rv_a > rv_b:
            rv_a, rv_b = rv_b, rv_a
            popt = [popt[3], popt[4], popt[5], popt[0], popt[1], popt[2]]
            perr = [perr[3], perr[4], perr[5], perr[0], perr[1], perr[2]]

        return {
            'rv1': popt[1], 'rv1_err': perr[1],
            'rv2': popt[4], 'rv2_err': perr[4],
            'ccf_height1': popt[0], 'ccf_height2': popt[3],
            'sigma1': abs(popt[2]), 'sigma2': abs(popt[5]),
            'dv': abs(popt[4] - popt[1]),
            'success': True,
        }
    except Exception:
        return {
            'rv1': v1, 'rv1_err': 30.0,
            'rv2': v2, 'rv2_err': 30.0,
            'ccf_height1': h1, 'ccf_height2': h2,
            'sigma1': 50.0, 'sigma2': 50.0,
            'dv': abs(v2 - v1),
            'success': False,
        }


# ================================================================
#  主 API: 测量 RV
# ================================================================

def _get_best_wd_templates(n=5):
    """获取覆盖不同温度的 WD 模版用于 CCF"""
    from .sed import _load_koester2_templates
    templates = _load_koester2_templates()
    if not templates:
        return []

    # 选取 logg=8.0 (典型 WD), 覆盖 6000-60000K 的几个温度
    target_teffs = [8000, 12000, 20000, 30000, 50000]
    target_logg = 8.0
    selected = []

    for t_target in target_teffs:
        best_key = min(templates.keys(),
                       key=lambda k: abs(k[0] - t_target) + abs(k[1] - target_logg) * 5000)
        if best_key not in [s[0] for s in selected]:
            t_dict = templates[best_key]
            selected.append((best_key, (t_dict['wavelength'], t_dict['flux'])))

    return selected[:n]


def measure_rv(wave, flux, err=None, v_min=-800, v_max=800, dv=30.0,
               templates=None):
    """
    用 CCF 测量单星径向速度。

    尝试多个 WD 模版温度，取 CCF 峰值最高的结果。

    Parameters
    ----------
    wave : array, Angstrom
    flux : array, flux (any units)
    err  : array or None
    v_min, v_max : 速度搜索范围 (km/s)
    dv : 速度步长 (km/s)
    templates : list of (key, (wave, flux)) or None (auto-load WD templates)

    Returns
    -------
    dict: {rv, rv_err, ccf_height, best_template, velocities, ccf,
           method: 'CCF'}
    or None
    """
    if templates is None:
        templates = _get_best_wd_templates()
    if not templates:
        return None

    best_result = None
    best_ccf_height = -1
    best_velocities = None
    best_ccf = None
    best_key = None

    for key, (tw, tf) in templates:
        velocities, ccf = compute_ccf(wave, flux, err, tw, tf,
                                       v_min=v_min, v_max=v_max, dv=dv)
        if velocities is None:
            continue

        result = _fit_ccf_peak(velocities, ccf)
        if result is None:
            continue

        if result['ccf_height'] > best_ccf_height:
            best_ccf_height = result['ccf_height']
            best_result = result
            best_velocities = velocities
            best_ccf = ccf
            best_key = key

    if best_result is None:
        return None

    best_result['best_template'] = best_key
    best_result['velocities'] = best_velocities
    best_result['ccf'] = best_ccf
    best_result['method'] = 'CCF'
    return best_result


def measure_rv_binary(wave, flux, err=None, v_min=-800, v_max=800, dv=30.0):
    """
    测量双线双星 (SB2) 的双组分径向速度。

    先做单星 CCF，再尝试双高斯拟合。如果双峰显著则报告双星结果。

    Parameters
    ----------
    wave, flux, err : 观测光谱

    Returns
    -------
    dict: {is_sb2: bool, single: {...}, binary: {...} or None,
           best_template, velocities, ccf}
    """
    templates = _get_best_wd_templates()
    if not templates:
        return None

    # 对每个模版做 CCF
    best_single = None
    best_binary = None
    best_ccf_h = -1
    best_v = None
    best_c = None
    best_key = None

    for key, (tw, tf) in templates:
        velocities, ccf = compute_ccf(wave, flux, err, tw, tf,
                                       v_min=v_min, v_max=v_max, dv=dv)
        if velocities is None:
            continue

        single = _fit_ccf_peak(velocities, ccf)
        if single is None:
            continue

        if single['ccf_height'] > best_ccf_h:
            best_ccf_h = single['ccf_height']
            best_single = single
            best_v = velocities
            best_c = ccf
            best_key = key

    if best_single is None:
        return None

    # 尝试双高斯拟合
    best_binary = _fit_ccf_double_peak(best_v, best_c)

    # 判断是否 SB2: 严格条件
    is_sb2 = False
    if best_binary is not None and best_binary.get('success', False):
        dv = best_binary.get('dv', 0)
        h2 = best_binary.get('ccf_height2', 0)
        h1 = best_binary.get('ccf_height1', 1)
        err1 = best_binary.get('rv1_err', 999)
        err2 = best_binary.get('rv2_err', 999)
        # 要求: 双峰间距合理，次峰够强，拟合误差合理
        is_sb2 = (50 < dv < 600 and
                  h2 > 0.10 and h2 > 0.25 * h1 and
                  err1 < 50 and err2 < 50)  # 拟合误差 < 50 km/s

    return {
        'is_sb2': is_sb2,
        'single': best_single,
        'binary': best_binary if is_sb2 else None,
        'best_template': best_key,
        'velocities': best_v,
        'ccf': best_c,
    }


def extract_pipeline_rv(results):
    """
    从 SDSS/DESI/LAMOST pipeline 结果中提取 RV。

    Parameters
    ----------
    results : dict, 来自 AstroQueryAll.results

    Returns
    -------
    list of dict: [{survey, rv, rv_err, source}]
    """
    pipeline_rvs = []

    # SDSS: z → cz (km/s)
    sdss = results.get('SDSS_spectrum')
    if sdss and 'z' in sdss:
        z = sdss['z']
        # 相对论修正: v = c * ((1+z)^2 - 1) / ((1+z)^2 + 1)
        rv_sdss = C_KMS * ((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1)
        pipeline_rvs.append({
            'survey': 'SDSS',
            'rv': rv_sdss,
            'rv_err': 5.0,  # SDSS typical RV error for WDs
            'source': 'pipeline_z',
        })

    # DESI: 从 redrock 提取
    desi = results.get('DESI')
    if desi and isinstance(desi, dict):
        match = desi.get('match', {})
        z = match.get('z')
        if z is not None:
            rv_desi = C_KMS * ((1 + z) ** 2 - 1) / ((1 + z) ** 2 + 1)
            zerr = match.get('zerr', 0.0001)
            pipeline_rvs.append({
                'survey': 'DESI',
                'rv': rv_desi,
                'rv_err': C_KMS * zerr,
                'source': 'pipeline_z',
            })

    # LAMOST: 直接有 RV
    lamost = results.get('LAMOST')
    if lamost and isinstance(lamost, dict) and 'rv' in lamost:
        rv_lamost = lamost['rv']
        if rv_lamost is not None and np.isfinite(rv_lamost):
            pipeline_rvs.append({
                'survey': 'LAMOST',
                'rv': rv_lamost,
                'rv_err': lamost.get('rv_err', 10.0),
                'source': 'pipeline_rv',
            })

    return pipeline_rvs


# ================================================================
#  完整 RV 分析流程
# ================================================================

def run_rv_analysis(results, output_dir=None, ra=None, dec=None):
    """
    对所有可用光谱做 RV 分析 (单星 + 双星 CCF 拟合)。

    Parameters
    ----------
    results : dict, 来自 AstroQueryAll.results
    output_dir : str, 输出目录
    ra, dec : float, 目标坐标

    Returns
    -------
    dict: {
        pipeline_rvs: list,       # pipeline 提取的 RV
        ccf_results: dict,        # 各光谱的 CCF 结果
        best_rv: float,           # 最终采用的 RV (km/s)
        best_rv_err: float,
        best_rv_source: str,
        is_sb2: bool,
        figures: list,            # 保存的图片路径
    }
    """
    import os

    rv_report = {
        'pipeline_rvs': [],
        'ccf_results': {},
        'skipped_spectra': [],
        'best_rv': None,
        'best_rv_err': None,
        'best_rv_source': None,
        'is_sb2': False,
        'figures': [],
    }

    # 1. Pipeline RV
    pipeline_rvs = extract_pipeline_rv(results)
    rv_report['pipeline_rvs'] = pipeline_rvs
    if pipeline_rvs:
        print(f"  Pipeline RV:")
        for p in pipeline_rvs:
            print(f"    {p['survey']}: RV = {p['rv']:.2f} ± {p['rv_err']:.2f} km/s")

    # 2. CCF RV from spectra
    spectra_to_fit = []

    # SDSS
    sdss = results.get('SDSS_spectrum')
    if sdss and 'wavelength' in sdss:
        spectra_to_fit.append(('SDSS', sdss['wavelength'], sdss['flux'],
                               sdss.get('error')))

    # DESI
    desi = results.get('DESI')
    if desi and isinstance(desi, dict) and 'spectrum' in desi:
        sp = desi['spectrum']
        waves, fluxes, errors = [], [], []
        for band in ('B', 'R', 'Z'):
            if band in sp:
                waves.append(np.array(sp[band]['wavelength']))
                fluxes.append(np.array(sp[band]['flux']))
                errors.append(np.array(sp[band]['error']))
        if waves:
            spectra_to_fit.append(('DESI',
                                   np.concatenate(waves),
                                   np.concatenate(fluxes),
                                   np.concatenate(errors)))

    # LAMOST
    lamost = results.get('LAMOST')
    if lamost and isinstance(lamost, dict) and 'wavelength' in lamost:
        spectra_to_fit.append(('LAMOST', lamost['wavelength'],
                               lamost['flux'], lamost.get('error')))

    # KOA / Keck LRIS
    koa = results.get('KOA_spectrum')
    if koa and isinstance(koa, dict) and 'wavelength' in koa:
        spectra_to_fit.append(('KOA/LRIS', koa['wavelength'],
                               koa['flux'], koa.get('error')))

    # HST
    hst = results.get('HST_spectrum')
    if hst and isinstance(hst, dict) and 'wavelength' in hst:
        coverage = _optical_ccf_coverage(hst['wavelength'], require_balmer=True)
        if coverage['usable'] and hst.get('usable_for_optical_rv', True):
            spectra_to_fit.append(('HST', hst['wavelength'],
                                   hst['flux'], hst.get('error')))
        else:
            reason = coverage['reason']
            if not reason and not hst.get('usable_for_optical_rv', True):
                reason = 'HST spectrum diagnostics marked not usable for optical RV'
            skip = {'survey': 'HST', 'reason': reason}
            skip.update({k: v for k, v in coverage.items() if k != 'usable'})
            rv_report['skipped_spectra'].append(skip)
            print(f"  跳过 HST CCF: {reason}")

    if not spectra_to_fit:
        print("  无可用光谱做 CCF RV 拟合")
        # 用 pipeline RV
        if pipeline_rvs:
            best = min(pipeline_rvs, key=lambda p: p['rv_err'])
            rv_report['best_rv'] = best['rv']
            rv_report['best_rv_err'] = best['rv_err']
            rv_report['best_rv_source'] = best['survey'] + '_pipeline'
        try:
            from .diagnostics import evaluate_rv_flags
            rv_flags = evaluate_rv_flags(rv_report)
            rv_report['quality_flags'] = rv_flags.get('flags', [])
            rv_report['rv_quality'] = rv_flags.get('quality', '')
        except Exception:
            rv_report['quality_flags'] = []
            rv_report['rv_quality'] = ''
        if output_dir:
            _save_rv_summary(rv_report, output_dir, ra, dec)
            save_csv(rv_report, output_dir)
        return rv_report

    # 对每个光谱做 CCF (单星 + 双星)
    all_ccf = {}
    for survey, wave, flux, err in spectra_to_fit:
        coverage = _optical_ccf_coverage(wave)
        if not coverage['usable']:
            skip = {'survey': survey, 'reason': coverage['reason']}
            skip.update({k: v for k, v in coverage.items() if k != 'usable'})
            rv_report['skipped_spectra'].append(skip)
            print(f"  跳过 {survey} CCF: {coverage['reason']}")
            continue
        print(f"  CCF RV ({survey})...")
        result = measure_rv_binary(wave, flux, err)
        if result is None:
            print(f"    CCF 失败")
            continue

        all_ccf[survey] = result
        single = result['single']
        print(f"    单星: RV = {single['rv']:.2f} ± {single['rv_err']:.2f} km/s "
              f"(CCF_h={single['ccf_height']:.3f})")

        if result['is_sb2']:
            b = result['binary']
            print(f"    SB2双星: RV1 = {b['rv1']:.2f} ± {b['rv1_err']:.2f}, "
                  f"RV2 = {b['rv2']:.2f} ± {b['rv2_err']:.2f} km/s "
                  f"(dv={b['dv']:.1f} km/s)")

        # 画 CCF 图
        if output_dir:
            fig_path = os.path.join(output_dir, f'rv_ccf_{survey.lower()}.png')
            _plot_ccf(result, survey, fig_path, ra=ra, dec=dec)
            rv_report['figures'].append(fig_path)

    rv_report['ccf_results'] = all_ccf

    # 3. 选择最佳 RV
    # CCF 峰值高 (>0.15) 且误差小时用 CCF; 否则用 pipeline
    best_rv = None
    best_err = None
    best_src = None
    is_sb2 = False

    # 从 CCF 结果选最佳
    best_ccf_result = None
    for survey, r in all_ccf.items():
        s = r['single']
        if s['ccf_height'] > 0.1:
            if best_ccf_result is None or s['ccf_height'] > best_ccf_result[1]['ccf_height']:
                best_ccf_result = (survey, s, r['is_sb2'])

    if best_ccf_result is not None:
        survey, s, sb2 = best_ccf_result
        # 如果 CCF height > 0.15 且误差 < 30 km/s，信任 CCF
        if s['ccf_height'] > 0.15 and s['rv_err'] < 30:
            best_rv = s['rv']
            best_err = s['rv_err']
            best_src = f'{survey}_CCF'
            is_sb2 = sb2
        elif pipeline_rvs:
            # CCF 较弱但 pipeline 可用 → 用 pipeline, 附注 CCF
            best_p = min(pipeline_rvs, key=lambda p: p['rv_err'])
            best_rv = best_p['rv']
            best_err = best_p['rv_err']
            best_src = f"{best_p['survey']}_pipeline"
            print(f"    (CCF 弱 h={s['ccf_height']:.3f} < 0.15, 回退到 pipeline)")
        else:
            # 只有 CCF
            best_rv = s['rv']
            best_err = s['rv_err']
            best_src = f'{survey}_CCF'
            is_sb2 = sb2
    elif pipeline_rvs:
        best_p = min(pipeline_rvs, key=lambda p: p['rv_err'])
        best_rv = best_p['rv']
        best_err = best_p['rv_err']
        best_src = f"{best_p['survey']}_pipeline"

    rv_report['best_rv'] = best_rv
    rv_report['best_rv_err'] = best_err
    rv_report['best_rv_source'] = best_src
    rv_report['is_sb2'] = is_sb2
    try:
        from .diagnostics import evaluate_rv_flags
        rv_flags = evaluate_rv_flags(rv_report)
        rv_report['quality_flags'] = rv_flags.get('flags', [])
        rv_report['rv_quality'] = rv_flags.get('quality', '')
    except Exception:
        rv_report['quality_flags'] = []
        rv_report['rv_quality'] = ''

    if best_rv is not None:
        sb2_flag = " [SB2]" if is_sb2 else ""
        print(f"  => 最佳 RV: {best_rv:.2f} ± {best_err:.2f} km/s "
              f"({best_src}){sb2_flag}")
        if rv_report.get('quality_flags'):
            print("     RV flags: " + ', '.join(rv_report['quality_flags']))

    # 保存 RV 汇总
    if output_dir:
        _save_rv_summary(rv_report, output_dir, ra, dec)
        save_csv(rv_report, output_dir)

    return rv_report


# ================================================================
#  绘图
# ================================================================

def _plot_ccf(result, survey, save_path, ra=None, dec=None):
    """绘制 CCF + 拟合结果"""
    fig, ax = plt.subplots(figsize=(10, 5))

    v = result['velocities']
    c = result['ccf']
    ax.plot(v, c, 'k-', lw=1, alpha=0.7, label='CCF')

    single = result['single']
    if single and single.get('success'):
        v_fit = np.linspace(single['rv'] - 300, single['rv'] + 300, 200)
        g_fit = _gaussian(v_fit, single['ccf_height'],
                          single['rv'], single['sigma_v'])
        ax.plot(v_fit, g_fit, 'b--', lw=1.5,
                label=f"Single: RV={single['rv']:.1f}±{single['rv_err']:.1f} km/s")
        ax.axvline(single['rv'], color='blue', ls=':', alpha=0.5)

    if result['is_sb2'] and result['binary']:
        b = result['binary']
        v_fit = np.linspace(min(b['rv1'], b['rv2']) - 300,
                            max(b['rv1'], b['rv2']) + 300, 300)
        g_fit = _double_gaussian(v_fit, b['ccf_height1'], b['rv1'], b['sigma1'],
                                  b['ccf_height2'], b['rv2'], b['sigma2'])
        ax.plot(v_fit, g_fit, 'r-', lw=1.5, alpha=0.8,
                label=f"SB2: RV1={b['rv1']:.1f}, RV2={b['rv2']:.1f} km/s")
        ax.axvline(b['rv1'], color='red', ls=':', alpha=0.5)
        ax.axvline(b['rv2'], color='red', ls=':', alpha=0.5)

    tmpl = result.get('best_template')
    tmpl_str = f"Teff={tmpl[0]}K logg={tmpl[1]:.1f}" if tmpl else ''

    title = f"CCF RV Fit — {survey}"
    if ra is not None:
        title += f"  RA={ra:.4f} DEC={dec:.4f}"
    if tmpl_str:
        title += f"\nTemplate: {tmpl_str}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Radial Velocity (km/s)', fontsize=11)
    ax.set_ylabel('CCF', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(v.min(), v.max())

    fig.tight_layout()
    utils.save_and_close(fig, save_path)


def _save_rv_summary(rv_report, output_dir, ra, dec):
    """保存 RV 分析结果到文件"""
    import os
    lines = []
    lines.append(f"# RV Analysis Report")
    if ra is not None:
        lines.append(f"# RA={ra:.6f}, DEC={dec:.6f}")
    lines.append("")

    # Pipeline
    lines.append("## Pipeline RV")
    for p in rv_report['pipeline_rvs']:
        lines.append(f"  {p['survey']}: RV = {p['rv']:.2f} ± {p['rv_err']:.2f} km/s "
                      f"({p['source']})")
    if not rv_report['pipeline_rvs']:
        lines.append("  (none)")
    lines.append("")

    # CCF
    lines.append("## CCF RV")
    for survey, r in rv_report['ccf_results'].items():
        s = r['single']
        lines.append(f"  {survey}: RV = {s['rv']:.2f} ± {s['rv_err']:.2f} km/s "
                      f"(CCF_h={s['ccf_height']:.3f}, template={r['best_template']})")
        if r['is_sb2'] and r['binary']:
            b = r['binary']
            lines.append(f"    SB2: RV1={b['rv1']:.2f}±{b['rv1_err']:.2f}, "
                          f"RV2={b['rv2']:.2f}±{b['rv2_err']:.2f} km/s "
                          f"(dv={b['dv']:.1f} km/s)")
    if not rv_report['ccf_results']:
        lines.append("  (none)")
    lines.append("")

    # Skipped spectra
    if rv_report.get('skipped_spectra'):
        lines.append("## Skipped Spectra")
        for item in rv_report['skipped_spectra']:
            wave_range = ''
            if item.get('wavelength_min_A') is not None:
                wave_range = (f" [{item.get('wavelength_min_A'):.1f}-"
                              f"{item.get('wavelength_max_A'):.1f} A]")
            lines.append(f"  {item.get('survey')}: {item.get('reason', '')}{wave_range}")
        lines.append("")

    # Best
    lines.append("## Best RV")
    if rv_report['best_rv'] is not None:
        sb2 = " [SB2]" if rv_report['is_sb2'] else ""
        lines.append(f"  RV = {rv_report['best_rv']:.2f} ± "
                      f"{rv_report['best_rv_err']:.2f} km/s "
                      f"({rv_report['best_rv_source']}){sb2}")
        if rv_report.get('rv_quality'):
            lines.append(f"  Quality: {rv_report['rv_quality']}")
        if rv_report.get('quality_flags'):
            lines.append("  Flags: " + ', '.join(rv_report['quality_flags']))
    else:
        lines.append("  No RV measurement available")

    path = os.path.join(output_dir, 'rv_analysis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  RV 分析报告: {path}")


def save_csv(rv_report, output_dir):
    """保存 RV 分析结果为 CSV"""
    import pandas as pd
    if rv_report is None or output_dir is None:
        return None

    rows = []
    for p in rv_report.get('pipeline_rvs', []):
        rows.append({
            'source': p['survey'] + '_pipeline',
            'rv_kms': p['rv'], 'rv_err_kms': p['rv_err'],
            'method': 'pipeline',
        })
    for survey, r in rv_report.get('ccf_results', {}).items():
        s = r['single']
        rows.append({
            'source': survey + '_CCF',
            'rv_kms': s['rv'], 'rv_err_kms': s['rv_err'],
            'method': 'CCF_single', 'ccf_height': s['ccf_height'],
        })
        if r.get('is_sb2') and r.get('binary'):
            b = r['binary']
            rows.append({
                'source': survey + '_CCF_SB2_comp1',
                'rv_kms': b.get('rv1'), 'rv_err_kms': b.get('rv1_err'),
                'method': 'CCF_SB2',
            })
            rows.append({
                'source': survey + '_CCF_SB2_comp2',
                'rv_kms': b.get('rv2'), 'rv_err_kms': b.get('rv2_err'),
                'method': 'CCF_SB2',
            })
    for item in rv_report.get('skipped_spectra', []):
        rows.append({
            'source': str(item.get('survey', '')) + '_skipped',
            'rv_kms': np.nan,
            'rv_err_kms': np.nan,
            'method': 'skipped',
            'note': item.get('reason', ''),
            'wavelength_min_A': item.get('wavelength_min_A'),
            'wavelength_max_A': item.get('wavelength_max_A'),
            'n_balmer_lines_covered': item.get('n_balmer_lines_covered'),
        })
    rows.append({
        'source': rv_report.get('best_rv_source', ''),
        'rv_kms': rv_report.get('best_rv'),
        'rv_err_kms': rv_report.get('best_rv_err'),
        'method': 'best', 'is_sb2': rv_report.get('is_sb2', False),
        'rv_quality': rv_report.get('rv_quality', ''),
        'quality_flags': ';'.join(rv_report.get('quality_flags', [])),
    })

    df = pd.DataFrame(rows)
    return utils.write_csv(df, output_dir, 'rv_analysis.csv')
