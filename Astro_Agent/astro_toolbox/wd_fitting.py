"""
WD 光谱拟合模块 — 光谱分类 + Teff/logg 网格拟合 + SED 拟合 + DWD 拟合
=======================================================================
Module 1 of the WD analysis pipeline.

功能:
1. WD 光谱分类 (DA / DB / DC / DZ / DQ)
   - DA: Balmer 系列 (H-alpha 6563, H-beta 4861, H-gamma 4340 ...)
   - DB: He I 线 (4471, 5876, 6678 ...)
   - DC: 无明显吸收线 (featureless continuum)
2. Koester2 模版网格拟合 → Teff, log g (chi-squared minimisation)
3. 宽波段 SED 拟合 (测光 + Koester2 model → independent Teff 约束)
4. 双白矮星 (DWD) 组合拟合 → 检查是否未分辨 DWD
5. 集成 cooling_age.py → M, R, 冷却年龄, 总年龄

用法:
    from astro_toolbox.wd_fitting import WDFitter
    fitter = WDFitter(wave, flux, err)
    result = fitter.fit_single()           # → Teff, logg, chi2
    dwd    = fitter.fit_dwd()              # → 两组 (Teff, logg) + 统计检验
    age    = fitter.derive_physical_params(parallax_mas=5.0, bp_rp=0.2, M_G=12.5)
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar, minimize
from scipy.ndimage import median_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config, utils

# 物理常数 (CGS)
C_CMS = 2.99792458e10       # cm/s
G_CGS = 6.67430e-8          # dyne cm^2 / g^2
M_SUN_G = 1.98892e33        # g
R_SUN_CM = 6.9634e10        # cm
C_KMS = 2.99792458e5        # km/s

# Balmer 系列真空波长 (Angstrom)
BALMER_LINES = {
    'H-alpha':  6564.61,
    'H-beta':   4862.68,
    'H-gamma':  4341.68,
    'H-delta':  4102.89,
    'H-epsilon': 3971.20,
    'H-zeta':   3890.16,
}

# He I 特征线 (DA vs DB 分类用)
HE_I_LINES = {
    'HeI_4026':  4026.2,
    'HeI_4471':  4471.5,
    'HeI_4922':  4921.9,
    'HeI_5876':  5875.6,
    'HeI_6678':  6678.2,
    'HeI_7065':  7065.2,
}


# ==================================================================
#  Koester2 模版加载 (复用 sed.py 的缓存)
# ==================================================================

def _load_koester2():
    """返回 Koester2 DA 模版字典 {(teff, logg): {'wavelength': arr, 'flux': arr}}"""
    from .sed import _load_koester2_templates
    return _load_koester2_templates()


def _get_model_grid_params():
    """返回模版覆盖的 (teff_list, logg_list) 唯一值排序列表"""
    templates = _load_koester2()
    if not templates:
        return [], []
    teffs = sorted(set(k[0] for k in templates))
    loggs = sorted(set(k[1] for k in templates))
    return teffs, loggs


# ==================================================================
#  光谱预处理
# ==================================================================

def _prepare_spectrum(wave, flux, err=None, w_min=3700, w_max=9200):
    """
    清洗光谱: 截取波长范围、去 NaN/负 err、连续谱归一化。

    Returns
    -------
    wave, flux, err, continuum  (all masked arrays)
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if err is not None:
        err = np.asarray(err, dtype=np.float64)

    # 波长范围截取
    mask = (wave >= w_min) & (wave <= w_max) & np.isfinite(flux) & np.isfinite(wave)
    if err is not None:
        mask &= np.isfinite(err) & (err > 0)
    wave, flux = wave[mask], flux[mask]
    if err is not None:
        err = err[mask]

    if len(wave) < 100:
        return None, None, None, None

    # 连续谱: 滑动中位数 (窗口 ~5% 光谱长度, 奇数)
    win = max(51, len(flux) // 20)
    if win % 2 == 0:
        win += 1
    continuum = median_filter(flux, size=win)
    continuum = np.maximum(continuum,
                           np.percentile(flux[flux > 0], 5) * 0.1
                           if np.any(flux > 0) else 1e-30)

    return wave, flux, err, continuum


def _normalize(flux, continuum):
    """归一化: flux / continuum"""
    return flux / continuum


# ==================================================================
#  光谱分类: DA / DB / DC
# ==================================================================

def classify_wd_type(wave, flux, err=None):
    """
    通过检测 Balmer / He I 吸收线的等值宽度 (EW) 判断 WD 光谱型。

    Returns
    -------
    dict: {
        spectral_type: 'DA' / 'DB' / 'DC' / 'DAB' / ...,
        balmer_ew: dict of EW per Balmer line (Angstrom),
        he_ew: dict of EW per He I line (Angstrom),
        balmer_total_ew: float,
        he_total_ew: float,
        confidence: float (0-1),
    }
    """
    result = {
        'spectral_type': 'DC',
        'balmer_ew': {},
        'he_ew': {},
        'balmer_total_ew': 0.0,
        'he_total_ew': 0.0,
        'confidence': 0.0,
    }

    w, f, e, cont = _prepare_spectrum(wave, flux, err)
    if w is None:
        return result

    f_norm = _normalize(f, cont)

    # 测量 EW: 对每条线取 ±15 A 的窗口
    def _measure_ew(line_center, half_width=15.0):
        """EW = integral (1 - f_norm) dw, 正值表示吸收线"""
        mask = (w >= line_center - half_width) & (w <= line_center + half_width)
        if mask.sum() < 5:
            return 0.0
        dw = np.median(np.diff(w[mask]))
        ew = np.sum((1.0 - f_norm[mask])) * dw
        return max(ew, 0.0)

    for name, lam in BALMER_LINES.items():
        ew = _measure_ew(lam)
        result['balmer_ew'][name] = round(ew, 2)

    for name, lam in HE_I_LINES.items():
        ew = _measure_ew(lam)
        result['he_ew'][name] = round(ew, 2)

    bew = sum(result['balmer_ew'].values())
    hew = sum(result['he_ew'].values())
    result['balmer_total_ew'] = round(bew, 2)
    result['he_total_ew'] = round(hew, 2)

    # 分类判据
    # DA: 显著 Balmer 吸收 (H-beta 和 H-gamma 是最可靠的 DA 指标)
    h_beta_ew = result['balmer_ew'].get('H-beta', 0)
    h_gamma_ew = result['balmer_ew'].get('H-gamma', 0)

    if bew > 15 and h_beta_ew > 3:
        if hew > 10:
            result['spectral_type'] = 'DAB'
            result['confidence'] = min(0.7, bew / 50)
        else:
            result['spectral_type'] = 'DA'
            result['confidence'] = min(0.95, bew / 30)
    elif hew > 10:
        result['spectral_type'] = 'DB'
        result['confidence'] = min(0.9, hew / 20)
    else:
        result['spectral_type'] = 'DC'
        result['confidence'] = 0.5

    try:
        from .diagnostics import analyze_spectrum
        diag = analyze_spectrum(w, f, e, survey='WD_spectrum')
        result['diagnostics'] = diag
        result['emission_flag'] = diag.get('emission_flag', False)
        result['anomaly_flags'] = diag.get('flags', [])
        result['misclassification_flag'] = diag.get('misclassification_flag', False)
        result['nonstellar_score'] = diag.get('nonstellar_score', 0.0)
        if result['emission_flag'] and result['spectral_type'] in ('DA', 'DB', 'DAB'):
            result['spectral_type'] += 'e'
    except Exception:
        result['emission_flag'] = False
        result['anomaly_flags'] = []
        result['misclassification_flag'] = False
        result['nonstellar_score'] = 0.0

    return result


# ==================================================================
#  核心: 单白矮星 Koester2 网格拟合
# ==================================================================

def _chi2_single(wave_obs, flux_obs, err_obs, wave_model, flux_model):
    """
    计算模型光谱与观测的 reduced chi-squared.
    模版被插值到观测波长网格, 并乘最佳缩放因子.

    Returns
    -------
    chi2_red, scale_factor, n_dof
    """
    # 截取重叠范围
    w_min = max(wave_obs[0], wave_model[0])
    w_max = min(wave_obs[-1], wave_model[-1])
    mask_o = (wave_obs >= w_min) & (wave_obs <= w_max)
    if mask_o.sum() < 50:
        return np.inf, 0.0, 0

    wo = wave_obs[mask_o]
    fo = flux_obs[mask_o]
    eo = err_obs[mask_o] if err_obs is not None else np.ones_like(fo)

    # 插值模版到观测网格
    f_interp = interp1d(wave_model, flux_model, kind='linear',
                        bounds_error=False, fill_value=0.0)
    fm = f_interp(wo)

    # 最佳缩放因子 (加权最小二乘)
    w = 1.0 / eo**2
    w[~np.isfinite(w)] = 0
    denom = np.sum(w * fm**2)
    if denom <= 0:
        return np.inf, 0.0, 0
    scale = np.sum(w * fo * fm) / denom

    residual = fo - scale * fm
    chi2 = np.sum(w * residual**2)
    n_dof = max(mask_o.sum() - 2, 1)  # 2 params: scale + selection
    chi2_red = chi2 / n_dof

    return chi2_red, scale, n_dof


def fit_single_wd(wave, flux, err=None, line_only=False):
    """
    对 WD 光谱进行 Koester2 全网格 chi-squared 拟合.

    Parameters
    ----------
    wave : array, Angstrom
    flux : array, observed flux (any units)
    err  : array or None
    line_only : bool
        If True, 只拟合 Balmer 线核区 (3800-6800 A) 获取更可靠的 logg;
        If False, 拟合全波段连续谱+线.

    Returns
    -------
    dict: {
        teff, logg, chi2_red, scale,
        teff_err, logg_err,  (from chi2 surface)
        chi2_grid,  (full chi2 map for plotting)
        best_model_wave, best_model_flux,  (scaled model spectrum)
    }
    """
    templates = _load_koester2()
    if not templates:
        return None

    teffs, loggs = _get_model_grid_params()
    if not teffs:
        return None

    # 预处理
    if line_only:
        w, f, e, cont = _prepare_spectrum(wave, flux, err, w_min=3800, w_max=6800)
    else:
        w, f, e, cont = _prepare_spectrum(wave, flux, err)
    if w is None:
        return None

    # 遍历所有模版
    chi2_grid = {}
    best_chi2 = np.inf
    best_params = None
    best_scale = 0.0

    for (t, g), tmpl in templates.items():
        chi2_r, sc, ndof = _chi2_single(w, f, e, tmpl['wavelength'], tmpl['flux'])
        chi2_grid[(t, g)] = chi2_r
        if chi2_r < best_chi2:
            best_chi2 = chi2_r
            best_params = (t, g)
            best_scale = sc

    if best_params is None:
        return None

    t_best, g_best = best_params

    # 误差估计: 从 chi2 surface, Delta chi2 < 1 → 1-sigma
    # 沿 Teff 轴
    teff_err = _estimate_1d_error(chi2_grid, teffs, loggs, axis='teff',
                                   best_t=t_best, best_g=g_best)
    logg_err = _estimate_1d_error(chi2_grid, teffs, loggs, axis='logg',
                                   best_t=t_best, best_g=g_best)

    # 最佳模版光谱 (缩放后)
    best_tmpl = templates[best_params]
    best_model_wave = best_tmpl['wavelength']
    best_model_flux = best_tmpl['flux'] * best_scale

    return {
        'teff': t_best,
        'logg': g_best,
        'chi2_red': best_chi2,
        'scale': best_scale,
        'teff_err': teff_err,
        'logg_err': logg_err,
        'chi2_grid': chi2_grid,
        'best_model_wave': best_model_wave,
        'best_model_flux': best_model_flux,
    }


def _estimate_1d_error(chi2_grid, teffs, loggs, axis, best_t, best_g):
    """
    从 chi2 surface 估计 1-sigma 误差.
    沿指定轴 marginalise (取另一轴的 minimum).
    找 Delta chi2_red < 1 / sqrt(n_dof) 的范围.
    """
    best_chi2 = chi2_grid.get((best_t, best_g), np.inf)

    if axis == 'teff':
        values = teffs
        profile = []
        for t in teffs:
            chi2_at_t = [chi2_grid.get((t, g), np.inf) for g in loggs]
            profile.append(min(chi2_at_t))
        best_val = best_t
    else:
        values = loggs
        profile = []
        for g in loggs:
            chi2_at_g = [chi2_grid.get((t, g), np.inf) for t in teffs]
            profile.append(min(chi2_at_g))
        best_val = best_g

    profile = np.array(profile)
    values = np.array(values)
    delta = profile - best_chi2

    # 1-sigma: Delta chi2 < 1 (for chi2_red, scale by ~1)
    within = values[delta < 1.0]
    if len(within) < 2:
        # 网格太稀疏, 用 grid step 作为下限
        step = np.median(np.diff(values)) if len(values) > 1 else 0
        return step

    return max(abs(within.max() - best_val), abs(best_val - within.min()))


# ==================================================================
#  SED 拟合 (测光点 vs Koester2 model)
# ==================================================================

def fit_sed(photometry, parallax_mas):
    """
    宽波段 SED 拟合: 用 Koester2 合成测光与观测测光比较.

    Parameters
    ----------
    photometry : dict {band_name: (mag, err, wave_A)}
        从 SEDFitter.photometry 传入
    parallax_mas : float
        Gaia 视差 (mas), 用于推算固体角 → 半径

    Returns
    -------
    dict: {
        teff_sed, logg_sed, chi2_sed,
        angular_radius_rad, R_Rsun,
        synthetic_mags: {band: mag_model},
    }
    """
    if not photometry or parallax_mas <= 0:
        return None

    templates = _load_koester2()
    if not templates:
        return None

    dist_pc = 1000.0 / parallax_mas
    dist_cm = dist_pc * 3.0857e18  # pc → cm

    # 观测测光 → flux (erg/s/cm^2/A)
    obs_data = []
    for band_name, (mag, err, wave_A) in photometry.items():
        info = config.BAND_INFO.get(band_name, {})
        zero_jy = info.get('zero_Jy', 3631.0)
        f_obs, f_err = utils.mag_to_flux_cgs(mag, wave_A, err, zero_jy)
        if f_obs > 0 and f_err > 0:
            obs_data.append({
                'band': band_name, 'wave': wave_A,
                'flux': f_obs, 'err': f_err,
            })

    if len(obs_data) < 3:
        return None

    obs_waves = np.array([d['wave'] for d in obs_data])
    obs_fluxes = np.array([d['flux'] for d in obs_data])
    obs_errs = np.array([d['err'] for d in obs_data])

    # 遍历模版网格
    best_chi2 = np.inf
    best_params = None
    best_scale = 0.0

    for (t, g), tmpl in templates.items():
        tw = tmpl['wavelength']
        tf = tmpl['flux']  # Eddington flux at stellar surface (erg/s/cm^2/A)

        # 在各测光波段处插值模版 flux
        f_interp = interp1d(tw, tf, kind='linear', bounds_error=False, fill_value=0)
        model_fluxes = f_interp(obs_waves)

        if np.all(model_fluxes <= 0):
            continue

        # 缩放因子 = (R/d)^2, R=stellar radius
        # scale = sum(w * obs * model) / sum(w * model^2)
        w = 1.0 / obs_errs**2
        denom = np.sum(w * model_fluxes**2)
        if denom <= 0:
            continue
        scale = np.sum(w * obs_fluxes * model_fluxes) / denom

        residual = obs_fluxes - scale * model_fluxes
        chi2 = np.sum(w * residual**2)
        ndof = max(len(obs_data) - 2, 1)
        chi2_red = chi2 / ndof

        if chi2_red < best_chi2:
            best_chi2 = chi2_red
            best_params = (t, g)
            best_scale = scale

    if best_params is None:
        return None

    # scale = (R / dist)^2   →   R = dist * sqrt(scale)
    R_cm = dist_cm * np.sqrt(max(best_scale, 0))
    R_Rsun = R_cm / R_SUN_CM
    angular_radius = np.sqrt(max(best_scale, 0))  # radians

    # 合成测光 (mag)
    best_tmpl = templates[best_params]
    f_interp = interp1d(best_tmpl['wavelength'], best_tmpl['flux'],
                        kind='linear', bounds_error=False, fill_value=0)
    syn_mags = {}
    for d in obs_data:
        fm = f_interp(d['wave']) * best_scale
        if fm > 0:
            # flux → mag (AB): m = -2.5 * log10(f_lambda * wave^2 / c) - 48.6
            info = config.BAND_INFO.get(d['band'], {})
            zero_jy = info.get('zero_Jy', 3631.0)
            # reverse of mag_to_flux_cgs
            c_A = 2.99792458e18
            f_hz = fm * d['wave']**2 / c_A
            f_jy = f_hz / 1e-23
            syn_mags[d['band']] = -2.5 * np.log10(f_jy / zero_jy)

    return {
        'teff_sed': best_params[0],
        'logg_sed': best_params[1],
        'chi2_sed': best_chi2,
        'angular_radius_rad': angular_radius,
        'R_Rsun': R_Rsun,
        'scale': best_scale,
        'synthetic_mags': syn_mags,
    }


# ==================================================================
#  DWD 组合拟合
# ==================================================================

def fit_dwd(wave, flux, err=None, single_result=None):
    """
    双白矮星 (DWD) 组合光谱拟合.

    将两个 Koester2 模版 (不同 Teff/logg) 加权组合, 与观测比较.
    用 F-test 判断 DWD 模型是否显著优于单星模型.

    Parameters
    ----------
    wave, flux, err : 观测光谱
    single_result : dict from fit_single_wd() (用于 F-test 对比)

    Returns
    -------
    dict: {
        is_dwd: bool,
        teff_1, logg_1, teff_2, logg_2,
        flux_ratio, chi2_dwd, chi2_single,
        f_statistic, p_value,
    }
    """
    templates = _load_koester2()
    if not templates:
        return None

    w, f, e, cont = _prepare_spectrum(wave, flux, err)
    if w is None:
        return None

    # 为加速: 只取一部分有代表性的模版 (稀疏网格)
    # 取 logg = {7.0, 7.5, 8.0, 8.5, 9.0} 和每隔 1000 K 的 Teff
    sparse_keys = [(t, g) for (t, g) in templates
                   if g in (7.0, 7.5, 8.0, 8.5, 9.0) and t % 1000 == 0]
    if len(sparse_keys) < 5:
        sparse_keys = list(templates.keys())[:50]

    best_chi2 = np.inf
    best_result = None

    for i, k1 in enumerate(sparse_keys):
        tmpl1 = templates[k1]
        f1_interp = interp1d(tmpl1['wavelength'], tmpl1['flux'],
                             kind='linear', bounds_error=False, fill_value=0)
        fm1 = f1_interp(w)

        for k2 in sparse_keys[i+1:]:
            tmpl2 = templates[k2]
            f2_interp = interp1d(tmpl2['wavelength'], tmpl2['flux'],
                                 kind='linear', bounds_error=False, fill_value=0)
            fm2 = f2_interp(w)

            # 两参数线性拟合: flux = a1*fm1 + a2*fm2
            if e is not None:
                wt = 1.0 / e**2
            else:
                wt = np.ones_like(f)
            wt[~np.isfinite(wt)] = 0

            # Normal equations
            A11 = np.sum(wt * fm1 * fm1)
            A12 = np.sum(wt * fm1 * fm2)
            A22 = np.sum(wt * fm2 * fm2)
            b1 = np.sum(wt * f * fm1)
            b2 = np.sum(wt * f * fm2)

            det = A11 * A22 - A12**2
            if det <= 0:
                continue

            a1 = (A22 * b1 - A12 * b2) / det
            a2 = (A11 * b2 - A12 * b1) / det

            # 两个缩放因子都必须为正 (物理: 都有正流量贡献)
            if a1 <= 0 or a2 <= 0:
                continue

            residual = f - (a1 * fm1 + a2 * fm2)
            chi2 = np.sum(wt * residual**2)
            ndof = max(len(w) - 3, 1)  # 3 params: a1, a2, implicit shift
            chi2_red = chi2 / ndof

            if chi2_red < best_chi2:
                best_chi2 = chi2_red
                best_result = {
                    'teff_1': k1[0], 'logg_1': k1[1],
                    'teff_2': k2[0], 'logg_2': k2[1],
                    'scale_1': a1, 'scale_2': a2,
                    'chi2_dwd': chi2_red,
                    'n_dof_dwd': ndof,
                }

    if best_result is None:
        return None

    # F-test: 比较 DWD 模型 vs 单星模型
    is_dwd = False
    f_stat = np.nan
    p_val = 1.0
    chi2_single = np.nan

    if single_result is not None:
        chi2_single = single_result['chi2_red']
        n_dof_single = len(w) - 2  # single: 2 params
        n_dof_dwd = best_result['n_dof_dwd']

        # chi2_red * n_dof = chi2_total
        chi2_s_total = chi2_single * n_dof_single
        chi2_d_total = best_result['chi2_dwd'] * n_dof_dwd

        # F = (chi2_single - chi2_dwd) / (p_dwd - p_single) / (chi2_dwd / n_dof_dwd)
        dp = 1  # extra param: a2 (component 2 scale)
        if chi2_d_total > 0 and chi2_s_total > chi2_d_total:
            f_stat = ((chi2_s_total - chi2_d_total) / dp) / (chi2_d_total / n_dof_dwd)

            from scipy.stats import f as f_dist
            p_val = f_dist.sf(f_stat, dp, n_dof_dwd)
            # DWD 显著 if p < 0.01 且 chi2 改善 > 10%
            is_dwd = (p_val < 0.01 and
                      best_result['chi2_dwd'] < 0.9 * chi2_single)

    flux_ratio = best_result['scale_2'] / best_result['scale_1'] if best_result['scale_1'] > 0 else 0

    best_result.update({
        'is_dwd': is_dwd,
        'flux_ratio': flux_ratio,
        'chi2_single': chi2_single,
        'f_statistic': f_stat,
        'p_value': p_val,
    })

    return best_result


# ==================================================================
#  物理参数推导 (集成 cooling_age.py)
# ==================================================================

def compute_wd_radius(mass_msun, logg):
    """
    从 mass 和 logg 推算 WD 半径.
    logg = log10(G * M / R^2)  →  R = sqrt(G * M / 10^logg)
    R 以 R_sun 为单位返回.
    """
    M_g = mass_msun * M_SUN_G
    g_cgs = 10**logg  # cm/s^2
    R_cm = np.sqrt(G_CGS * M_g / g_cgs)
    return R_cm / R_SUN_CM


def derive_physical_params(teff, logg, parallax_mas=None,
                           bp_rp=None, M_G=None):
    """
    推导 WD 物理参数: Mass, Radius, 冷却年龄, 总年龄.

    优先使用 WD_models (Sihao Cheng) 从 (BP-RP, M_G) 插值.
    如果没有测光, 则用 mass-radius relation 从 logg 估算.

    Parameters
    ----------
    teff : float, K
    logg : float
    parallax_mas : float or None
    bp_rp : float or None (Gaia BP-RP)
    M_G : float or None (Gaia absolute G mag)

    Returns
    -------
    dict: {mass, radius_rsun, teff, logg,
           cooling_age_gyr, total_age_gyr,
           m_progenitor, ms_lifetime_gyr, source}
    """
    result = {
        'teff': teff,
        'logg': logg,
        'mass': np.nan,
        'radius_rsun': np.nan,
        'cooling_age_gyr': np.nan,
        'total_age_gyr': np.nan,
        'm_progenitor': np.nan,
        'ms_lifetime_gyr': np.nan,
        'source': 'none',
    }

    # 方法 A: 用 WD_models 从 (BP-RP, M_G) 插值
    if bp_rp is not None and M_G is not None:
        try:
            from .cooling_age import interpolate_wd_params, compute_progenitor_lifetime
            wd = interpolate_wd_params(bp_rp, M_G)
            if wd is not None:
                result['mass'] = wd['mass']
                result['radius_rsun'] = compute_wd_radius(wd['mass'], wd['logg'])
                result['cooling_age_gyr'] = wd['cooling_age_gyr']
                result['total_age_gyr'] = wd['total_age_gyr']
                result['teff'] = wd['teff']
                result['logg'] = wd['logg']
                result['source'] = 'WD_models_HR'

                prog = compute_progenitor_lifetime(wd['mass'])
                if prog is not None:
                    result['m_progenitor'] = prog['m_progenitor']
                    result['ms_lifetime_gyr'] = prog['ms_lifetime_gyr']

                return result
        except Exception:
            pass

    # 方法 B: 用 logg + 经验 mass-radius relation 估计
    # WD 典型: M ~ 0.6 Msun for logg=8.0; 使用 Hamada-Salpeter 零温 MR
    # 简化: logg = log10(G*M/R^2), 结合 M-R: R ~ 0.013 * (M/0.6)^{-1/3} Rsun
    # 用 Nauenberg (1972) 的解析 MR relation
    mass_est = _logg_to_mass(logg)
    R_est = compute_wd_radius(mass_est, logg)
    result['mass'] = mass_est
    result['radius_rsun'] = R_est
    result['source'] = 'logg_MR_relation'

    # 估算冷却年龄 (Mestel law 简化)
    # t_cool ~ 8.8e6 * (M/0.6)^{5/7} * (Teff/12000)^{-5/2} yr
    t_cool_yr = 8.8e6 * (mass_est / 0.6)**(5.0/7.0) * (teff / 12000)**(-2.5)
    result['cooling_age_gyr'] = t_cool_yr / 1e9

    return result


def _logg_to_mass(logg):
    """
    从 logg 用 WD mass-radius relation 反推质量.
    迭代求解: logg = log10(G * M / R(M)^2)
    使用 Nauenberg (1972) 零温 C/O WD MR relation:
        R = 0.0115 * sqrt( (M_ch/M)^{2/3} - (M/M_ch)^{2/3} ) Rsun
        M_ch = 1.44 Msun
    """
    M_CH = 1.44  # Chandrasekhar mass

    def _radius(m):
        """R in Rsun"""
        if m <= 0 or m >= M_CH:
            return 1e-10
        r = 0.0115 * np.sqrt((M_CH / m)**(2.0/3.0) - (m / M_CH)**(2.0/3.0))
        return max(r, 1e-10)

    def _logg_from_m(m):
        r_cm = _radius(m) * R_SUN_CM
        m_g = m * M_SUN_G
        return np.log10(G_CGS * m_g / r_cm**2)

    # Bisection search
    target = logg
    m_lo, m_hi = 0.15, 1.40
    for _ in range(60):
        m_mid = 0.5 * (m_lo + m_hi)
        lg = _logg_from_m(m_mid)
        if lg < target:
            m_lo = m_mid
        else:
            m_hi = m_mid

    return 0.5 * (m_lo + m_hi)


# ==================================================================
#  主类: WDFitter
# ==================================================================

class WDFitter:
    """
    白矮星光谱拟合综合工具.

    用法:
        fitter = WDFitter(wave, flux, err)
        fitter.classify()                  # → DA/DB/DC
        fitter.fit_single()                # → Teff, logg
        fitter.fit_dwd()                   # → DWD 检验
        fitter.derive_params(plx, bp_rp, M_G)  # → M, R, age
        fitter.plot_all(output_dir)        # → 诊断图
    """

    def __init__(self, wave, flux, err=None):
        self.wave = np.asarray(wave, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64) if err is not None else None

        self.classification = None
        self.single_fit = None
        self.dwd_fit = None
        self.physical_params = None
        self.sed_fit = None

    def classify(self):
        """光谱分类"""
        self.classification = classify_wd_type(self.wave, self.flux, self.err)
        return self.classification

    def fit_single(self, line_only=False):
        """单星 Koester2 网格拟合"""
        self.single_fit = fit_single_wd(self.wave, self.flux, self.err,
                                        line_only=line_only)
        return self.single_fit

    def fit_double(self):
        """DWD 组合拟合"""
        if self.single_fit is None:
            self.fit_single()
        self.dwd_fit = fit_dwd(self.wave, self.flux, self.err,
                               single_result=self.single_fit)
        return self.dwd_fit

    def fit_sed(self, photometry, parallax_mas):
        """宽波段 SED 拟合"""
        self.sed_fit = fit_sed(photometry, parallax_mas)
        return self.sed_fit

    def derive_params(self, parallax_mas=None, bp_rp=None, M_G=None):
        """推导物理参数"""
        if self.single_fit is None:
            self.fit_single()
        if self.single_fit is None:
            return None
        self.physical_params = derive_physical_params(
            self.single_fit['teff'], self.single_fit['logg'],
            parallax_mas=parallax_mas, bp_rp=bp_rp, M_G=M_G)
        return self.physical_params

    def run_all(self, photometry=None, parallax_mas=None,
                bp_rp=None, M_G=None, output_dir=None):
        """
        一键执行全部分析步骤.

        Returns
        -------
        dict: {classification, single_fit, dwd_fit, sed_fit, physical_params}
        """
        print("  [1/5] 光谱分类...")
        self.classify()
        sp_type = self.classification['spectral_type']
        conf = self.classification['confidence']
        print(f"    类型: {sp_type} (confidence={conf:.2f})")

        print("  [2/5] 单星光谱拟合...")
        self.fit_single()
        if self.single_fit:
            print(f"    Teff = {self.single_fit['teff']} K  "
                  f"logg = {self.single_fit['logg']:.2f}  "
                  f"chi2_red = {self.single_fit['chi2_red']:.4f}")
        else:
            print("    拟合失败")

        print("  [3/5] DWD 组合拟合...")
        self.fit_double()
        if self.dwd_fit:
            flag = "YES" if self.dwd_fit['is_dwd'] else "No"
            print(f"    DWD: {flag}  "
                  f"(p={self.dwd_fit['p_value']:.4f}, "
                  f"chi2_dwd={self.dwd_fit['chi2_dwd']:.4f} "
                  f"vs chi2_single={self.dwd_fit['chi2_single']:.4f})")

        print("  [4/5] SED 拟合...")
        if photometry and parallax_mas:
            self.fit_sed(photometry, parallax_mas)
            if self.sed_fit:
                print(f"    Teff_SED = {self.sed_fit['teff_sed']} K  "
                      f"R = {self.sed_fit['R_Rsun']:.4f} Rsun  "
                      f"chi2 = {self.sed_fit['chi2_sed']:.4f}")
        else:
            print("    (跳过: 无测光或视差)")

        print("  [5/5] 物理参数...")
        self.derive_params(parallax_mas=parallax_mas,
                           bp_rp=bp_rp, M_G=M_G)
        if self.physical_params:
            p = self.physical_params
            print(f"    M = {p['mass']:.3f} Msun  R = {p['radius_rsun']:.4f} Rsun  "
                  f"t_cool = {p['cooling_age_gyr']:.4f} Gyr  "
                  f"({p['source']})")

        if output_dir:
            self.plot_all(output_dir)
            self.save_report(output_dir)
            self.save_csv(output_dir)

        return {
            'classification': self.classification,
            'single_fit': self.single_fit,
            'dwd_fit': self.dwd_fit,
            'sed_fit': self.sed_fit,
            'physical_params': self.physical_params,
        }

    # ==============================================================
    #  绘图
    # ==============================================================

    def plot_spectral_fit(self, save_path=None, ra=None, dec=None):
        """绘制光谱拟合结果: 观测 + 最佳模版 + 残差"""
        if self.single_fit is None:
            return None

        sf = self.single_fit
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                  gridspec_kw={'height_ratios': [3, 1]},
                                  sharex=True)

        ax_spec = axes[0]
        ax_res = axes[1]

        # 观测光谱
        ax_spec.plot(self.wave, self.flux, 'k-', lw=0.5, alpha=0.7,
                     label='Observed')

        # 最佳模型
        ax_spec.plot(sf['best_model_wave'], sf['best_model_flux'],
                     'r-', lw=1.0, alpha=0.8,
                     label=f"Model: Teff={sf['teff']}K, logg={sf['logg']:.2f}")

        # Balmer 线标注
        for name, lam in BALMER_LINES.items():
            ax_spec.axvline(lam, color='blue', ls=':', alpha=0.3, lw=0.8)
            ax_spec.text(lam, ax_spec.get_ylim()[1] * 0.95, name,
                        fontsize=7, rotation=90, va='top', ha='right',
                        color='blue', alpha=0.5)

        coord_str = f"  RA={ra:.4f} DEC={dec:.4f}" if ra is not None else ""
        sp_type = self.classification['spectral_type'] if self.classification else '?'
        ax_spec.set_title(
            f"WD Spectral Fit — {sp_type}{coord_str}\n"
            f"Teff={sf['teff']}K  logg={sf['logg']:.2f}  "
            f"$\\chi^2_{{\\rm red}}$={sf['chi2_red']:.4f}",
            fontsize=12)
        ax_spec.set_ylabel('Flux')
        ax_spec.legend(fontsize=10)
        ax_spec.grid(True, alpha=0.3)

        # 设置合理轴范围
        utils.set_spectrum_axes(ax_spec, self.wave, self.flux,
                                model=sf['best_model_flux'])

        # 残差
        f_interp = interp1d(sf['best_model_wave'], sf['best_model_flux'],
                            kind='linear', bounds_error=False, fill_value=0)
        model_at_obs = f_interp(self.wave)
        residual = self.flux - model_at_obs
        ax_res.plot(self.wave, residual, 'k-', lw=0.5, alpha=0.6)
        ax_res.axhline(0, color='red', ls='--', lw=0.8)
        ax_res.set_xlabel('Wavelength (A)')
        ax_res.set_ylabel('Residual')
        ax_res.grid(True, alpha=0.3)

        fig.tight_layout()
        utils.save_and_close(fig, save_path)
        return fig

    def plot_chi2_map(self, save_path=None):
        """绘制 chi2 surface (Teff vs logg)"""
        if self.single_fit is None or 'chi2_grid' not in self.single_fit:
            return None

        chi2_grid = self.single_fit['chi2_grid']
        teffs, loggs = _get_model_grid_params()

        # 构建 2D array
        chi2_arr = np.full((len(loggs), len(teffs)), np.nan)
        for i, g in enumerate(loggs):
            for j, t in enumerate(teffs):
                chi2_arr[i, j] = chi2_grid.get((t, g), np.nan)

        fig, ax = plt.subplots(figsize=(10, 7))
        T_arr = np.array(teffs)
        G_arr = np.array(loggs)

        # 用 chi2_min 附近的等高线
        chi2_min = self.single_fit['chi2_red']
        levels = chi2_min * np.array([1.0, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10.0])

        cs = ax.contourf(T_arr, G_arr, chi2_arr, levels=30, cmap='viridis_r')
        ax.contour(T_arr, G_arr, chi2_arr, levels=levels,
                   colors='white', linewidths=0.5, linestyles='--')

        ax.plot(self.single_fit['teff'], self.single_fit['logg'],
                'r*', ms=15, label=f"Best: {self.single_fit['teff']}K, "
                f"logg={self.single_fit['logg']:.2f}")

        plt.colorbar(cs, ax=ax, label='$\\chi^2_{\\rm red}$')
        ax.set_xlabel('$T_{\\rm eff}$ (K)', fontsize=12)
        ax.set_ylabel('log g', fontsize=12)
        ax.set_title('$\\chi^2$ Map — Koester2 Grid Fit', fontsize=13)
        ax.legend(fontsize=10)
        ax.invert_xaxis()

        fig.tight_layout()
        utils.save_and_close(fig, save_path)
        return fig

    def plot_sed_fit(self, photometry, save_path=None):
        """绘制 SED 拟合: 观测测光点 + 最佳模版"""
        if self.sed_fit is None:
            return None

        sf = self.sed_fit
        templates = _load_koester2()
        best_tmpl = templates.get((sf['teff_sed'], sf['logg_sed']))
        if best_tmpl is None:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # 模型连续谱 (缩放后)
        model_wave = best_tmpl['wavelength']
        model_flux = best_tmpl['flux'] * sf['scale']
        ax.plot(model_wave, model_flux, 'b-', lw=0.8, alpha=0.5,
                label=f"Model: Teff={sf['teff_sed']}K, logg={sf['logg_sed']:.1f}")

        # 观测测光点
        color_map = {
            'GALEX': 'purple', 'SDSS': 'blue', 'Gaia': 'green',
            '2MASS': 'orange', 'WISE': 'red', 'SPHEREx': 'darkorange',
        }
        for band, (mag, mag_err, wave_A) in photometry.items():
            info = config.BAND_INFO.get(band, {})
            zero_jy = info.get('zero_Jy', 3631.0)
            f_obs, f_err = utils.mag_to_flux_cgs(mag, wave_A, mag_err, zero_jy)
            prefix = band.split('_')[0]
            c = color_map.get(prefix, 'gray')
            ax.errorbar(wave_A, f_obs, yerr=f_err,
                        fmt='o', color=c, ms=8, capsize=3, zorder=10)
            ax.annotate(band.replace('_', ' '), (wave_A, f_obs),
                        fontsize=7, rotation=30, ha='left', va='bottom',
                        xytext=(3, 5), textcoords='offset points')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength (A)', fontsize=12)
        ax.set_ylabel(r'$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=12)
        ax.set_title(f'SED Fit  Teff={sf["teff_sed"]}K  '
                     f'R={sf["R_Rsun"]:.4f} R$_\\odot$  '
                     f'$\\chi^2$={sf["chi2_sed"]:.3f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        fig.tight_layout()
        utils.save_and_close(fig, save_path)
        return fig

    def plot_all(self, output_dir, ra=None, dec=None):
        """生成所有诊断图"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        figs = []

        # 光谱拟合图
        path = os.path.join(output_dir, 'wd_spectral_fit.png')
        fig = self.plot_spectral_fit(path, ra=ra, dec=dec)
        if fig:
            figs.append(path)

        # Chi2 map
        path = os.path.join(output_dir, 'wd_chi2_map.png')
        fig = self.plot_chi2_map(path)
        if fig:
            figs.append(path)

        return figs

    def save_report(self, output_dir):
        """保存拟合结果到文本文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'wd_fitting_report.txt')

        lines = []
        lines.append("=" * 60)
        lines.append("WD Spectral Fitting Report")
        lines.append("=" * 60)

        if self.classification:
            c = self.classification
            lines.append(f"\n--- Spectral Classification ---")
            lines.append(f"  Type: {c['spectral_type']} "
                        f"(confidence: {c['confidence']:.2f})")
            lines.append(f"  Balmer total EW: {c['balmer_total_ew']:.1f} A")
            lines.append(f"  He I total EW: {c['he_total_ew']:.1f} A")
            if c.get('emission_flag'):
                lines.append("  Emission flag: YES")
            if c.get('misclassification_flag'):
                lines.append(f"  Misclassification check: non-stellar score "
                             f"{c.get('nonstellar_score', 0):.2f}")
            if c.get('anomaly_flags'):
                lines.append("  Anomaly flags: " + ', '.join(c['anomaly_flags']))
            for name, ew in c['balmer_ew'].items():
                if ew > 0:
                    lines.append(f"    {name}: EW = {ew:.1f} A")

        if self.single_fit:
            sf = self.single_fit
            lines.append(f"\n--- Single WD Fit (Koester2 Grid) ---")
            lines.append(f"  Teff = {sf['teff']} +/- {sf['teff_err']:.0f} K")
            lines.append(f"  logg = {sf['logg']:.2f} +/- {sf['logg_err']:.2f}")
            lines.append(f"  chi2_red = {sf['chi2_red']:.6f}")
            lines.append(f"  scale = {sf['scale']:.6e}")

        if self.dwd_fit:
            d = self.dwd_fit
            lines.append(f"\n--- DWD Composite Fit ---")
            lines.append(f"  Is DWD: {d['is_dwd']}")
            lines.append(f"  Component 1: Teff={d['teff_1']}K, logg={d['logg_1']:.2f}")
            lines.append(f"  Component 2: Teff={d['teff_2']}K, logg={d['logg_2']:.2f}")
            lines.append(f"  Flux ratio (2/1): {d['flux_ratio']:.3f}")
            lines.append(f"  chi2_dwd = {d['chi2_dwd']:.6f}")
            lines.append(f"  chi2_single = {d['chi2_single']:.6f}")
            lines.append(f"  F-statistic = {d['f_statistic']:.2f}")
            lines.append(f"  p-value = {d['p_value']:.6f}")

        if self.sed_fit:
            s = self.sed_fit
            lines.append(f"\n--- SED Fit ---")
            lines.append(f"  Teff_SED = {s['teff_sed']} K")
            lines.append(f"  logg_SED = {s['logg_sed']:.2f}")
            lines.append(f"  R = {s['R_Rsun']:.4f} R_sun")
            lines.append(f"  chi2_SED = {s['chi2_sed']:.4f}")

        if self.physical_params:
            p = self.physical_params
            lines.append(f"\n--- Physical Parameters ---")
            lines.append(f"  Mass = {p['mass']:.4f} M_sun")
            lines.append(f"  Radius = {p['radius_rsun']:.4f} R_sun")
            lines.append(f"  Teff = {p['teff']:.0f} K")
            lines.append(f"  logg = {p['logg']:.3f}")
            lines.append(f"  Cooling age = {p['cooling_age_gyr']:.4f} Gyr "
                        f"({p['cooling_age_gyr']*1e3:.1f} Myr)")
            if not np.isnan(p['total_age_gyr']):
                lines.append(f"  Total age = {p['total_age_gyr']:.4f} Gyr")
            if not np.isnan(p['m_progenitor']):
                lines.append(f"  M_progenitor = {p['m_progenitor']:.3f} M_sun")
                lines.append(f"  MS lifetime = {p['ms_lifetime_gyr']:.4f} Gyr")
            lines.append(f"  Source: {p['source']}")

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"  WD 拟合报告: {path}")

    def save_csv(self, output_dir):
        """保存 WD 拟合结果为 CSV"""
        import pandas as pd
        os.makedirs(output_dir, exist_ok=True)

        row = {}
        if self.classification:
            c = self.classification
            row['spectral_type'] = c['spectral_type']
            row['confidence'] = c['confidence']
            row['balmer_total_ew'] = c['balmer_total_ew']
            row['he_total_ew'] = c['he_total_ew']
            row['emission_flag'] = c.get('emission_flag', False)
            row['misclassification_flag'] = c.get('misclassification_flag', False)
            row['nonstellar_score'] = c.get('nonstellar_score', 0.0)
            row['anomaly_flags'] = ';'.join(c.get('anomaly_flags', []))
        if self.single_fit:
            sf = self.single_fit
            row.update({
                'teff': sf['teff'], 'teff_err': sf['teff_err'],
                'logg': sf['logg'], 'logg_err': sf['logg_err'],
                'chi2_red': sf['chi2_red'],
            })
        if self.dwd_fit:
            d = self.dwd_fit
            row.update({
                'is_dwd': d['is_dwd'],
                'dwd_teff_1': d['teff_1'], 'dwd_logg_1': d['logg_1'],
                'dwd_teff_2': d['teff_2'], 'dwd_logg_2': d['logg_2'],
                'dwd_flux_ratio': d['flux_ratio'],
                'f_statistic': d['f_statistic'], 'p_value': d['p_value'],
            })
        if self.sed_fit:
            s = self.sed_fit
            row.update({
                'teff_sed': s['teff_sed'], 'logg_sed': s['logg_sed'],
                'R_Rsun': s['R_Rsun'], 'chi2_sed': s['chi2_sed'],
            })
        if self.physical_params:
            p = self.physical_params
            row.update({
                'mass': p['mass'], 'radius_rsun': p['radius_rsun'],
                'cooling_age_gyr': p['cooling_age_gyr'],
                'total_age_gyr': p.get('total_age_gyr'),
                'm_progenitor': p.get('m_progenitor'),
                'ms_lifetime_gyr': p.get('ms_lifetime_gyr'),
                'param_source': p.get('source', ''),
            })

        if not row:
            return None
        df = pd.DataFrame([row])
        path = os.path.join(output_dir, 'wd_fitting_results.csv')
        df.to_csv(path, index=False)
        return path
