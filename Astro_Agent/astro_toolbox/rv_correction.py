"""
真运动学径向速度 (RV_true) 提取模块
=====================================
Module 2 of the WD analysis pipeline.

核心物理:
    观测的 WD RV 包含两个组分:
    1. RV_obs — 天体的视线方向运动学多普勒频移
    2. V_grav — 白矮星表面引力红移 (非运动学)

    引力红移公式 (广义相对论):
        V_grav = G * M / (R * c)    [km/s]

    其中 M, R 来自 Module 1 的光谱/SED 拟合.

    最终运动学 RV:
        RV_true = RV_obs - V_grav

    注意: SDSS/LAMOST/DESI 的 1D 光谱已经在日心/质心真空静止系,
          因此 **不需要** 做质心修正.

RV_obs 测量方法:
    1. Balmer 线核拟合 (Lorentzian/Voigt profile) — 精度优先
    2. CCF 交叉相关 (复用 rv_fitting.py) — 鲁棒性优先
    3. 取加权平均

用法:
    from astro_toolbox.rv_correction import measure_true_rv
    result = measure_true_rv(wave, flux, err,
                              mass_msun=0.6, radius_rsun=0.012)
"""

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config, utils

# 物理常数
C_KMS = 2.99792458e5          # km/s
G_CGS = 6.67430e-8            # dyne cm^2 g^-2
M_SUN_G = 1.98892e33          # g
R_SUN_CM = 6.9634e10          # cm

# Balmer 系列真空波长 (Angstrom)
BALMER_LINES = {
    'H-alpha':    6564.61,
    'H-beta':     4862.68,
    'H-gamma':    4341.68,
    'H-delta':    4102.89,
    'H-epsilon':  3971.20,
    'H-zeta':     3890.16,
}

# He I 线 (用于 DB 型 WD)
HE_LINES = {
    'HeI_4471': 4471.5,
    'HeI_5876': 5875.6,
    'HeI_6678': 6678.2,
}


# ==================================================================
#  引力红移计算
# ==================================================================

def gravitational_redshift(mass_msun, radius_rsun):
    """
    计算白矮星表面引力红移.

    V_grav = G * M / (R * c)   [km/s]

    对典型 WD (0.6 Msun, 0.012 Rsun):
        V_grav ~ 29 km/s

    Parameters
    ----------
    mass_msun : float — WD 质量 (太阳质量)
    radius_rsun : float — WD 半径 (太阳半径)

    Returns
    -------
    float — V_grav (km/s), always > 0
    """
    if mass_msun <= 0 or radius_rsun <= 0:
        return 0.0

    M = mass_msun * M_SUN_G       # g
    R = radius_rsun * R_SUN_CM    # cm
    c = C_KMS * 1e5               # cm/s

    v_grav = G_CGS * M / (R * c)  # cm/s → km/s
    v_grav_kms = v_grav / 1e5

    return v_grav_kms


def gravitational_redshift_from_logg(mass_msun, logg):
    """
    从 mass 和 logg 计算引力红移 (不需要独立的 R).
    logg = log10(G * M / R^2)  →  R = sqrt(G*M / 10^logg)
    """
    from .wd_fitting import compute_wd_radius
    R = compute_wd_radius(mass_msun, logg)
    return gravitational_redshift(mass_msun, R)


# ==================================================================
#  Balmer / He 线核拟合
# ==================================================================

def _voigt_approx(x, amp, x0, sigma, gamma):
    """
    伪 Voigt 近似 (Gaussian + Lorentzian 的加权和).
    用于拟合 WD 的宽 Balmer 吸收线 (Stark broadening → Lorentzian wing).
    """
    f_G = np.exp(-0.5 * ((x - x0) / sigma)**2)
    f_L = gamma**2 / ((x - x0)**2 + gamma**2)
    # Lorentzian 权重
    eta = gamma / (sigma + gamma) if (sigma + gamma) > 0 else 0.5
    return amp * ((1 - eta) * f_G + eta * f_L)


def _lorentzian(x, amp, x0, gamma):
    """Lorentzian profile (Stark-broadened WD Balmer line)."""
    return amp * gamma**2 / ((x - x0)**2 + gamma**2)


def _gaussian(x, amp, x0, sigma):
    """Gaussian profile."""
    return amp * np.exp(-0.5 * ((x - x0) / sigma)**2)


def fit_line_core(wave, flux, err=None, line_center=4862.68,
                  half_window=50.0, profile='voigt',
                  dc_mode=False):
    """
    拟合单条吸收线的线核位置 → Doppler 位移 → RV.

    方法:
    1. 取 line_center ± half_window 范围的光谱
    2. 局部连续谱归一化 (线性拟合线翼)
    3. [DC模式] Savitzky-Golay 平滑 + 显著性检验
    4. 拟合 pseudo-Voigt (或 Lorentzian/Gaussian)
    5. 线心位移 → RV = c * (lambda_obs - lambda_lab) / lambda_lab

    Parameters
    ----------
    wave : array (A)
    flux : array
    err : array or None
    line_center : float (A), 实验室波长
    half_window : float (A), 拟合窗口半宽
    profile : str, 'voigt', 'lorentzian', 'gaussian'
    dc_mode : bool, 对 DC 型启用平滑 + 严格显著性检验

    Returns
    -------
    dict: {
        rv_kms, rv_err_kms,
        line_center_obs, line_center_lab,
        amplitude, fwhm,
        fit_success, profile_type,
        wave_fit, flux_fit, model_fit,  (for plotting)
        significance,  (线检测显著性)
    }
    or None
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    # 截取窗口
    mask = ((wave >= line_center - half_window) &
            (wave <= line_center + half_window) &
            np.isfinite(flux))
    if mask.sum() < 15:
        return None

    w = wave[mask]
    f = flux[mask]
    if err is not None:
        e = np.asarray(err, dtype=np.float64)[mask]
        e[e <= 0] = np.median(e[e > 0]) if np.any(e > 0) else 1.0
    else:
        e = np.ones_like(f)

    # 局部连续谱: 用线翼 (距中心 > half_window * 0.7) 拟合直线
    wing = np.abs(w - line_center) > half_window * 0.6
    if wing.sum() < 4:
        wing = np.abs(w - line_center) > half_window * 0.4

    if wing.sum() >= 4:
        p = np.polyfit(w[wing], f[wing], 1)
        continuum = np.polyval(p, w)
    else:
        continuum = np.median(f) * np.ones_like(f)

    # 避免除零
    continuum = np.maximum(continuum, np.percentile(f[f > 0], 5) * 0.1
                           if np.any(f > 0) else 1e-30)

    f_norm = f / continuum
    e_norm = e / continuum

    # === DC 模式: 平滑 + 显著性检验 ===
    if dc_mode:
        # SNR 自适应 Savitzky-Golay 平滑
        local_snr = np.median(f / e) if np.median(e) > 0 else 1.0
        if local_snr > 15:
            sg_win = 5
        elif local_snr > 8:
            sg_win = 9
        else:
            sg_win = 15
        # 窗口必须 <= 数据点数且为奇数
        sg_win = min(sg_win, len(f_norm) - 1)
        if sg_win % 2 == 0:
            sg_win -= 1
        if sg_win >= 5:
            f_smooth = savgol_filter(f_norm, sg_win, 2)
        else:
            f_smooth = f_norm.copy()

        # 显著性检验: 线核区域的吸收深度 vs 连续谱噪声
        core_mask = np.abs(w - line_center) < 15
        wing_noise = np.std(f_smooth[wing]) if wing.sum() > 3 else 1.0
        if core_mask.sum() > 0 and wing_noise > 0:
            core_depth = 1.0 - np.median(f_smooth[core_mask])
            n_core = core_mask.sum()
            significance = core_depth / wing_noise * np.sqrt(n_core)
        else:
            core_depth = 0
            significance = 0

        # DC 模式门槛 (严格):
        # 1. 必须是真吸收 (depth > 0)
        # 2. 显著性 > 2 sigma
        # 3. 吸收深度必须物理合理: DC 的浅线 < 30%
        #    (噪声"吸收"可达 50-80%, 真 DC 浅线 < 15%)
        # 4. 局部 SNR > 3 (太嘈杂的区域完全不可信)
        local_snr_core = np.median(f[core_mask] / e[core_mask]) if core_mask.sum() > 0 else 0
        if (core_depth < 0.02 or core_depth > 0.30 or
                significance < 2.0 or local_snr_core < 3.0):
            return None

        # 用平滑后的光谱做拟合 (减少噪声对线心定位的干扰)
        depth = 1.0 - f_smooth
    else:
        significance = np.nan
        # 吸收线深度: depth = 1 - f_norm (正值=吸收)
        depth = 1.0 - f_norm

    # 初始猜测
    i_min = np.argmin(f_norm)  # 最深吸收位置
    x0_guess = w[i_min]
    amp_guess = depth[i_min] if depth[i_min] > 0 else 1.0 - f_norm[i_min]
    width_guess = 5.0  # A

    if amp_guess < 0.02:
        # 线太浅
        return None

    try:
        if profile == 'voigt':
            popt, pcov = curve_fit(
                _voigt_approx, w, depth,
                p0=[amp_guess, x0_guess, width_guess, width_guess],
                sigma=e_norm,
                bounds=([0, line_center - 30, 0.5, 0.5],
                        [2, line_center + 30, 100, 100]),
                maxfev=5000)
            x0_fit = popt[1]
            perr = np.sqrt(np.diag(pcov))
            x0_err = perr[1]
            fwhm = 2 * np.sqrt(popt[2]**2 + popt[3]**2)
            model = _voigt_approx(w, *popt)

        elif profile == 'lorentzian':
            popt, pcov = curve_fit(
                _lorentzian, w, depth,
                p0=[amp_guess, x0_guess, width_guess],
                sigma=e_norm,
                bounds=([0, line_center - 30, 0.5],
                        [2, line_center + 30, 100]),
                maxfev=5000)
            x0_fit = popt[1]
            perr = np.sqrt(np.diag(pcov))
            x0_err = perr[1]
            fwhm = 2 * abs(popt[2])
            model = _lorentzian(w, *popt)

        else:  # gaussian
            popt, pcov = curve_fit(
                _gaussian, w, depth,
                p0=[amp_guess, x0_guess, width_guess],
                sigma=e_norm,
                bounds=([0, line_center - 30, 0.5],
                        [2, line_center + 30, 100]),
                maxfev=5000)
            x0_fit = popt[1]
            perr = np.sqrt(np.diag(pcov))
            x0_err = perr[1]
            fwhm = 2.355 * abs(popt[2])
            model = _gaussian(w, *popt)

        # 线心位移 → RV
        delta_lambda = x0_fit - line_center
        rv_kms = C_KMS * delta_lambda / line_center
        rv_err_kms = C_KMS * x0_err / line_center

        return {
            'rv_kms': rv_kms,
            'rv_err_kms': rv_err_kms,
            'line_center_obs': x0_fit,
            'line_center_lab': line_center,
            'delta_lambda': delta_lambda,
            'amplitude': popt[0],
            'fwhm': fwhm,
            'fit_success': True,
            'profile_type': profile,
            'wave_fit': w,
            'flux_norm': f_norm,
            'depth_data': depth,
            'model_fit': model,
            'continuum': continuum,
            'significance': significance,
        }

    except Exception:
        return None


def measure_rv_from_lines(wave, flux, err=None, wd_type='DA',
                          dc_mode=False):
    """
    从多条 Balmer / He I 线核拟合测量 RV, 取加权平均.

    Parameters
    ----------
    wave, flux, err : 观测光谱
    wd_type : str, 'DA' or 'DB'
    dc_mode : bool, DC 型启用平滑 + 显著性检验 + sigma-clipping

    Returns
    -------
    dict: {
        rv_obs, rv_obs_err,
        line_results: {line_name: fit_result},
        n_lines_used, method, rv_quality,
    }
    """
    lines = BALMER_LINES if wd_type == 'DA' else HE_LINES

    line_results = {}
    rvs = []
    weights = []
    names_used = []

    for name, lam in lines.items():
        # 检查光谱是否覆盖该线
        if lam < wave.min() + 50 or lam > wave.max() - 50:
            continue

        # 对较宽的 H-alpha 用更大窗口
        hw = 60 if 'alpha' in name else 40
        result = fit_line_core(wave, flux, err, line_center=lam,
                               half_window=hw, profile='voigt',
                               dc_mode=dc_mode)
        if result is not None and result['fit_success']:
            # DC 模式下额外过滤: 单线 |RV| > 500 km/s 几乎不可能是真实的
            if dc_mode and abs(result['rv_kms']) > 500:
                continue
            line_results[name] = result
            rvs.append(result['rv_kms'])
            weights.append(1.0 / max(result['rv_err_kms'], 1.0)**2)
            names_used.append(name)

    if not rvs:
        return None

    rvs = np.array(rvs)
    weights = np.array(weights)

    # === DC 模式: sigma-clipping 剔除异常线 ===
    if dc_mode and len(rvs) > 2:
        rv_median = np.median(rvs)
        # 先用 MAD 估计 robust scatter
        mad = np.median(np.abs(rvs - rv_median))
        sigma_est = 1.4826 * mad if mad > 0 else 100.0
        # 剔除 > 2.5 sigma 的异常线
        keep = np.abs(rvs - rv_median) < 2.5 * sigma_est
        if keep.sum() >= 2:
            rejected = [names_used[i] for i in range(len(rvs)) if not keep[i]]
            rvs = rvs[keep]
            weights = weights[keep]
            names_used = [names_used[i] for i in range(len(keep)) if keep[i]]
            line_results = {k: v for k, v in line_results.items()
                           if k in names_used}

    if len(rvs) < 2 and dc_mode:
        # DC 模式下至少需要 2 条一致的线
        return None

    # 加权平均
    rv_mean = np.sum(weights * rvs) / np.sum(weights)
    rv_err = 1.0 / np.sqrt(np.sum(weights))

    # 如果多于 2 条线, 用线间 scatter 与形式误差取较大者
    if len(rvs) > 2:
        scatter = np.sqrt(np.sum(weights * (rvs - rv_mean)**2) / np.sum(weights))
        rv_err = max(rv_err, scatter)

    # RV 质量评估
    if dc_mode:
        scatter_val = np.std(rvs) if len(rvs) > 1 else 999
        if len(rvs) >= 3 and scatter_val < 50:
            rv_quality = 'good'
        elif len(rvs) >= 2 and scatter_val < 100:
            rv_quality = 'marginal'
        else:
            rv_quality = 'unreliable'
    else:
        rv_quality = 'good'

    return {
        'rv_obs': rv_mean,
        'rv_obs_err': rv_err,
        'line_results': line_results,
        'n_lines_used': len(rvs),
        'method': 'line_core_fit',
        'rv_quality': rv_quality,
    }


# ==================================================================
#  综合 RV 测量: 线核 + CCF 融合
# ==================================================================

def measure_rv_obs(wave, flux, err=None, wd_type='DA', dc_mode=False):
    """
    综合测量 RV_obs: 线核拟合 + CCF, 取最优.

    优先级:
    1. 如果 >=3 条 Balmer 线拟合成功 → 线核结果 (精度 ~3-10 km/s)
    2. 否则 → CCF 结果 (鲁棒, ~10-30 km/s)
    3. 两者都有时, 如果一致 (差 < 2 sigma) 则取线核; 否则取 CCF

    Returns
    -------
    dict: {
        rv_obs, rv_obs_err, source,
        line_fit: {...} or None,
        ccf_fit: {...} or None,
        rv_quality: str,
    }
    """
    # 线核拟合
    line_result = measure_rv_from_lines(wave, flux, err, wd_type=wd_type,
                                         dc_mode=dc_mode)

    # CCF (复用 rv_fitting.py)
    from .rv_fitting import measure_rv as ccf_measure_rv
    ccf_result = ccf_measure_rv(wave, flux, err)

    # 决策
    rv_obs = None
    rv_err = None
    source = None

    has_line = (line_result is not None and line_result['n_lines_used'] >= 2)
    has_ccf = (ccf_result is not None and ccf_result.get('ccf_height', 0) > 0.05)

    if has_line and has_ccf:
        rv_line = line_result['rv_obs']
        rv_ccf = ccf_result['rv']
        diff = abs(rv_line - rv_ccf)
        combined_err = np.sqrt(line_result['rv_obs_err']**2 +
                               ccf_result['rv_err']**2)

        if diff < 2.0 * combined_err:
            # 一致 → 取线核 (更精确)
            rv_obs = rv_line
            rv_err = line_result['rv_obs_err']
            source = f'line_core ({line_result["n_lines_used"]} lines)'
        else:
            # 不一致 → 如果线核线数多则信线核, 否则信 CCF
            if line_result['n_lines_used'] >= 3:
                rv_obs = rv_line
                rv_err = line_result['rv_obs_err']
                source = f'line_core ({line_result["n_lines_used"]} lines, CCF disagrees)'
            else:
                rv_obs = ccf_result['rv']
                rv_err = ccf_result['rv_err']
                source = 'CCF (line_core disagrees)'
    elif has_line:
        rv_obs = line_result['rv_obs']
        rv_err = line_result['rv_obs_err']
        source = f'line_core ({line_result["n_lines_used"]} lines)'
    elif has_ccf:
        rv_obs = ccf_result['rv']
        rv_err = ccf_result['rv_err']
        source = 'CCF'
    else:
        return None

    # rv_quality 从线核拟合传递, CCF 默认 'good'
    rv_quality = 'good'
    if has_line and line_result.get('rv_quality'):
        rv_quality = line_result['rv_quality']

    return {
        'rv_obs': rv_obs,
        'rv_obs_err': rv_err,
        'source': source,
        'line_fit': line_result,
        'ccf_fit': ccf_result,
        'rv_quality': rv_quality,
    }


# ==================================================================
#  主 API: 测量 RV_true
# ==================================================================

def measure_true_rv(wave, flux, err=None,
                    mass_msun=None, radius_rsun=None, logg=None,
                    wd_type='DA', spectral_type=None):
    """
    测量白矮星的真运动学 RV, 包含引力红移修正.

    RV_true = RV_obs - V_grav

    IMPORTANT:
    - 输入光谱已在日心/质心真空静止系 (SDSS/LAMOST/DESI 1D spectra)
    - 因此 **不做** 质心修正
    - 只需: (1) 测 RV_obs, (2) 计算 V_grav, (3) 做差

    Parameters
    ----------
    wave : array, Angstrom
    flux : array, flux (any units)
    err  : array or None
    mass_msun : float, WD 质量 (Msun) — from Module 1
    radius_rsun : float, WD 半径 (Rsun) — from Module 1
    logg : float, 如果没有 radius_rsun 则用 logg + mass 推 R
    wd_type : str, 'DA' or 'DB'
    spectral_type : str, 光谱类型 (DA/DB/DC/DAB), DC 时启用增强模式

    Returns
    -------
    dict: {
        rv_true, rv_true_err,
        rv_obs, rv_obs_err,
        v_grav,
        rv_obs_details: {...},
        mass_msun, radius_rsun,
    }
    """
    # Step 1: 测量 RV_obs
    dc_mode = (spectral_type is not None and
               spectral_type.upper().startswith('DC'))
    obs = measure_rv_obs(wave, flux, err, wd_type=wd_type, dc_mode=dc_mode)
    if obs is None:
        return None

    rv_obs = obs['rv_obs']
    rv_obs_err = obs['rv_obs_err']

    # Step 2: 计算 V_grav
    v_grav = 0.0
    R_used = radius_rsun
    M_used = mass_msun

    if mass_msun is not None and mass_msun > 0:
        if radius_rsun is not None and radius_rsun > 0:
            v_grav = gravitational_redshift(mass_msun, radius_rsun)
        elif logg is not None:
            v_grav = gravitational_redshift_from_logg(mass_msun, logg)
            from .wd_fitting import compute_wd_radius
            R_used = compute_wd_radius(mass_msun, logg)
    else:
        # 如果没有 M/R, 用典型 WD 值 (M=0.6, logg=8.0)
        M_used = 0.6
        R_used = 0.0126  # Rsun
        v_grav = gravitational_redshift(M_used, R_used)

    # Step 3: RV_true = RV_obs - V_grav
    rv_true = rv_obs - v_grav
    rv_true_err = rv_obs_err  # V_grav 的系统误差不影响随机误差

    return {
        'rv_true': rv_true,
        'rv_true_err': rv_true_err,
        'rv_obs': rv_obs,
        'rv_obs_err': rv_obs_err,
        'v_grav': v_grav,
        'rv_obs_source': obs['source'],
        'rv_obs_details': obs,
        'rv_quality': obs.get('rv_quality', 'good'),
        'mass_msun': M_used,
        'radius_rsun': R_used,
    }


# ==================================================================
#  批量处理: 对 merged_all.csv 中的所有 WD 计算 RV_true
# ==================================================================

def run_rv_correction(wave, flux, err=None,
                      physical_params=None,
                      survey_name='',
                      output_dir=None, ra=None, dec=None):
    """
    完整 RV_true 提取流程, 含诊断图.

    Parameters
    ----------
    wave, flux, err : 观测光谱
    physical_params : dict from wd_fitting.derive_physical_params()
        必须包含 'mass', 'radius_rsun' 或 'logg'
    survey_name : str, 巡天名 (SDSS/DESI/LAMOST)
    output_dir : str
    ra, dec : float

    Returns
    -------
    dict — measure_true_rv() 的结果 + 图片路径
    """
    # 推断 WD 类型
    wd_type = 'DA'  # 默认
    spectral_type = None
    if physical_params:
        sp = physical_params.get('spectral_type', 'DA')
        spectral_type = sp
        if sp.startswith('DB'):
            wd_type = 'DB'

    # M, R 从物理参数
    mass = None
    radius = None
    logg = None
    if physical_params:
        mass = physical_params.get('mass')
        radius = physical_params.get('radius_rsun')
        logg = physical_params.get('logg')

    result = measure_true_rv(wave, flux, err,
                              mass_msun=mass, radius_rsun=radius,
                              logg=logg, wd_type=wd_type,
                              spectral_type=spectral_type)

    if result is None:
        print(f"  RV 测量失败 ({survey_name})")
        return None

    print(f"  RV_{survey_name}: obs={result['rv_obs']:.2f} "
          f"- V_grav={result['v_grav']:.2f} "
          f"= true={result['rv_true']:.2f} +/- {result['rv_true_err']:.2f} km/s "
          f"({result['rv_obs_source']})")

    # 诊断图
    if output_dir:
        result['figures'] = []

        # 线核拟合图
        fig_path = os.path.join(output_dir,
                                f'rv_line_fit_{survey_name.lower()}.png')
        fig = plot_line_fits(result, survey_name, fig_path, ra=ra, dec=dec)
        if fig:
            result['figures'].append(fig_path)

        # 保存文本报告
        _save_rv_correction_report(result, survey_name, output_dir, ra, dec)
        save_csv(result, survey_name, output_dir)

    return result


# ==================================================================
#  诊断图
# ==================================================================

def plot_line_fits(rv_result, survey_name, save_path=None,
                   ra=None, dec=None):
    """
    绘制各 Balmer / He 线的核拟合诊断图.

    每条线一个子图: 归一化光谱 + 拟合 profile + 标注 RV.
    底部: 综合结果 (RV_obs, V_grav, RV_true).
    """
    obs = rv_result.get('rv_obs_details')
    if obs is None:
        return None

    line_fit = obs.get('line_fit')
    if line_fit is None or not line_fit.get('line_results'):
        return None

    line_results = line_fit['line_results']
    n_lines = len(line_results)
    if n_lines == 0:
        return None

    n_cols = min(n_lines, 3)
    n_rows = (n_lines + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_lines == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (name, lr) in enumerate(line_results.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        w = lr['wave_fit']
        f_norm = lr['flux_norm']
        depth = lr['depth_data']
        model = lr['model_fit']

        # 归一化光谱
        ax.plot(w, f_norm, 'k-', lw=0.8, alpha=0.7, label='Data')
        # 拟合 profile (反转: 1 - model)
        ax.plot(w, 1 - model, 'r-', lw=1.5, alpha=0.8, label='Fit')

        # 实验室波长
        ax.axvline(lr['line_center_lab'], color='blue', ls=':', alpha=0.5,
                   label=f'Lab: {lr["line_center_lab"]:.1f} A')
        # 拟合线心
        ax.axvline(lr['line_center_obs'], color='red', ls='--', alpha=0.5,
                   label=f'Obs: {lr["line_center_obs"]:.1f} A')

        ax.set_title(f'{name}  RV={lr["rv_kms"]:.1f} +/- {lr["rv_err_kms"]:.1f} km/s',
                     fontsize=10)
        ax.set_xlabel('Wavelength (A)')
        ax.set_ylabel('Normalized flux')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(n_lines, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    # 总标题
    coord_str = f"  RA={ra:.4f} DEC={dec:.4f}" if ra is not None else ""
    fig.suptitle(
        f"RV Line Core Fits — {survey_name}{coord_str}\n"
        f"RV_obs = {rv_result['rv_obs']:.2f} km/s,  "
        f"V_grav = {rv_result['v_grav']:.2f} km/s,  "
        f"RV_true = {rv_result['rv_true']:.2f} +/- {rv_result['rv_true_err']:.2f} km/s",
        fontsize=12, y=1.02)

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def _save_rv_correction_report(rv_result, survey_name, output_dir, ra, dec):
    """保存 RV_true 报告"""
    path = os.path.join(output_dir, f'rv_true_{survey_name.lower()}.txt')

    lines = []
    lines.append("=" * 60)
    lines.append(f"True Kinematic RV Report  ({survey_name})")
    lines.append("=" * 60)
    if ra is not None:
        lines.append(f"RA = {ra:.6f},  DEC = {dec:.6f}")
    lines.append("")

    lines.append("--- Observed RV ---")
    lines.append(f"  RV_obs = {rv_result['rv_obs']:.4f} +/- "
                 f"{rv_result['rv_obs_err']:.4f} km/s")
    lines.append(f"  Source: {rv_result['rv_obs_source']}")
    lines.append("")

    lines.append("--- Gravitational Redshift ---")
    lines.append(f"  V_grav = G*M / (R*c) = {rv_result['v_grav']:.4f} km/s")
    lines.append(f"  M_WD = {rv_result['mass_msun']:.4f} M_sun")
    lines.append(f"  R_WD = {rv_result['radius_rsun']:.6f} R_sun")
    lines.append("")

    lines.append("--- True Kinematic RV ---")
    lines.append(f"  RV_true = RV_obs - V_grav")
    lines.append(f"  RV_true = {rv_result['rv_obs']:.4f} - "
                 f"{rv_result['v_grav']:.4f}")
    lines.append(f"  RV_true = {rv_result['rv_true']:.4f} +/- "
                 f"{rv_result['rv_true_err']:.4f} km/s")
    lines.append("")

    # 逐线结果
    obs = rv_result.get('rv_obs_details', {})
    line_fit = obs.get('line_fit')
    if line_fit and line_fit.get('line_results'):
        lines.append("--- Individual Line Fits ---")
        for name, lr in line_fit['line_results'].items():
            lines.append(f"  {name:12s}  "
                        f"lab={lr['line_center_lab']:.2f} A  "
                        f"obs={lr['line_center_obs']:.2f} A  "
                        f"dL={lr['delta_lambda']:+.2f} A  "
                        f"RV={lr['rv_kms']:+.2f} +/- {lr['rv_err_kms']:.2f} km/s  "
                        f"FWHM={lr['fwhm']:.1f} A")
        lines.append(f"\n  Weighted mean: {line_fit['rv_obs']:.2f} "
                    f"+/- {line_fit['rv_obs_err']:.2f} km/s "
                    f"({line_fit['n_lines_used']} lines)")

    ccf_fit = obs.get('ccf_fit')
    if ccf_fit:
        lines.append(f"\n--- CCF Result ---")
        lines.append(f"  RV_CCF = {ccf_fit['rv']:.2f} +/- "
                    f"{ccf_fit['rv_err']:.2f} km/s  "
                    f"(CCF_h={ccf_fit['ccf_height']:.3f})")

    lines.append("")
    lines.append("NOTE: Input spectrum is in heliocentric/barycentric vacuum frame.")
    lines.append("      No barycentric correction applied.")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  RV 修正报告: {path}")


def save_csv(rv_result, survey_name, output_dir):
    """保存 RV 改正结果为 CSV"""
    import pandas as pd
    if rv_result is None or output_dir is None:
        return None
    row = {
        'survey': survey_name,
        'rv_obs_kms': rv_result.get('rv_obs'),
        'rv_obs_err_kms': rv_result.get('rv_obs_err'),
        'rv_obs_source': rv_result.get('rv_obs_source', ''),
        'v_grav_kms': rv_result.get('v_grav'),
        'rv_true_kms': rv_result.get('rv_true'),
        'rv_true_err_kms': rv_result.get('rv_true_err'),
        'mass_msun': rv_result.get('mass_msun'),
        'radius_rsun': rv_result.get('radius_rsun'),
        'rv_quality': rv_result.get('rv_quality', ''),
    }
    df = pd.DataFrame([row])
    path = os.path.join(output_dir, f'rv_correction_{survey_name.lower()}.csv')
    df.to_csv(path, index=False)
    return path
