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

# Conservative floors used when low/medium-resolution survey spectra are
# combined line-by-line.  The formal covariance of a single line fit does not
# include wavelength-calibration and WD line-profile systematics.
LINE_RV_SYSTEMATIC_FLOOR_KMS = 5.0
LINE_RV_MAX_FORMAL_ERR_KMS = 80.0
LINE_RV_MAX_ABS_RV_KMS = 800.0
LINE_RV_MIN_FWHM_A = 0.8
LINE_RV_MAX_FWHM_A = 220.0
GRAV_REDSHIFT_MIN_ERR_KMS = 2.0
GRAV_REDSHIFT_FRACTIONAL_FLOOR = 0.05
GRAV_REDSHIFT_DEFAULT_ERR_KMS = 10.0
GRAV_REDSHIFT_6D_FLOOR_KMS = 10.0
GRAV_REDSHIFT_6D_FRACTIONAL_FLOOR = 0.20

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

# Emission RV defaults are intentionally limited to stellar/CV-like lines.
# Forbidden lines often trace nebular or galaxy contamination rather than the WD.
EMISSION_LINES = {
    **BALMER_LINES,
    'HeI_4471': 4471.5,
    'HeII_4686': 4685.7,
    'HeI_5876': 5875.6,
    'HeI_6678': 6678.2,
    'CaII_K': 3933.7,
    'CaII_H': 3968.5,
    'Na_D': 5892.0,
}

FORBIDDEN_EMISSION_LINES = {
    '[OIII]_4959': 4958.9,
    '[OIII]_5007': 5006.8,
    '[NII]_6548': 6548.1,
    '[NII]_6583': 6583.5,
    '[SII]_6716': 6716.4,
    '[SII]_6731': 6730.8,
}


def _canonical_line_name(name):
    key = str(name or '').lower()
    repl = {
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'ζ': 'zeta',
    }
    for old, new in repl.items():
        key = key.replace(old, new)
    for ch in ' []_-':
        key = key.replace(ch, '')
    aliases = {
        'ha': 'halpha', 'h1': 'halpha',
        'hb': 'hbeta', 'h2': 'hbeta',
        'hg': 'hgamma', 'h3': 'hgamma',
        'hd': 'hdelta', 'h4': 'hdelta',
        'he': 'hepsilon',
        'hz': 'hzeta',
    }
    return aliases.get(key, key)


def _filter_line_dict(line_dict, preferred_lines=None):
    if not preferred_lines:
        return dict(line_dict)
    wanted = {_canonical_line_name(x) for x in preferred_lines}
    return {
        name: lam for name, lam in line_dict.items()
        if _canonical_line_name(name) in wanted
    }


def rvopt_uncertainty_layer(rv_result, cluster_rv=None, cluster_rv_err=None,
                            cluster_intrinsic_rv_dispersion=2.0,
                            wd_physical_params=None,
                            rv_physical_variants=None):
    """
    Add literature-calibrated sandbox RV uncertainty columns without changing
    production RV central values.

    The optimized layer keeps RV_obs, V_grav and RV_true central values fixed,
    then reports *_opt columns using conservative survey-spectrum floors and a
    gravitational-redshift floor of max(10 km/s, 0.2 |V_grav|).  If the formal
    V_grav uncertainty is dominated by a low-resolution Teff/logg/M-R
    degeneracy, and a Gaia/SED M/R branch gives a consistent V_grav, the
    sandbox error replaces the degenerate formal value by the physical floor.
    """
    if rv_result is None:
        return {}
    obs = rv_result.get('rv_obs_details') or {}
    line_fit = obs.get('line_fit') or {}
    rv_quality = str(rv_result.get('rv_quality', '') or '').lower()
    method = str(obs.get('method', rv_result.get('rv_obs_source', '')) or '').lower()
    n_used = line_fit.get('n_lines_used')
    scatter = line_fit.get('cluster_scatter_kms')
    rejected = line_fit.get('rejected_lines', []) or []

    try:
        n_used = int(n_used)
    except Exception:
        n_used = 0
    try:
        scatter = float(scatter)
    except Exception:
        scatter = np.nan

    severe_rv_flag = False
    if 'line_core' in method and rv_quality == 'good' and n_used >= 3:
        rv_floor = 10.0
        rv_floor_reason = 'high_quality_line_core'
    elif 'ccf' in method or rv_quality == 'good':
        rv_floor = 15.0
        rv_floor_reason = 'normal_corv_or_ccf'
    elif rv_quality == 'marginal' or rejected or (np.isfinite(scatter) and scatter >= 25):
        rv_floor = 25.0
        rv_floor_reason = 'marginal_or_line_ccf_disagreement'
    else:
        rv_floor = 40.0
        rv_floor_reason = 'severe_or_unreliable_rv'
        severe_rv_flag = True

    rv_obs = _finite_or_nan(rv_result.get('rv_obs'))
    rv_obs_err_formal = _finite_or_nan(rv_result.get('rv_obs_err'))
    rv_obs_err_opt = (
        float(np.hypot(rv_obs_err_formal, rv_floor))
        if np.isfinite(rv_obs_err_formal) else rv_floor
    )
    v_grav = _finite_or_nan(rv_result.get('v_grav'))
    v_grav_err_formal = _finite_or_nan(rv_result.get('v_grav_err'))
    v_grav_err_source = str(rv_result.get('v_grav_err_source', '') or '')
    vgrav_gaia_sed = _estimate_gaia_sed_vgrav(
        rv_result, wd_physical_params, rv_physical_variants)
    vgrav_balmer_gaia_delta = (
        abs(v_grav - vgrav_gaia_sed)
        if np.isfinite(v_grav + vgrav_gaia_sed) else np.nan
    )
    vgrav_model_discrepant = (
        np.isfinite(vgrav_balmer_gaia_delta)
        and vgrav_balmer_gaia_delta > 25.0
    )
    rel_m_err = _relative_error(
        rv_result.get('mass_msun'), rv_result.get('mass_msun_err'))
    rel_r_err = _relative_error(
        rv_result.get('radius_rsun'), rv_result.get('radius_rsun_err'))
    logg_err = _finite_or_nan(rv_result.get('logg_err'))
    lowres_degenerate = (
        (np.isfinite(rel_m_err) and rel_m_err > 0.30)
        or (np.isfinite(rel_r_err) and rel_r_err > 0.30)
        or (np.isfinite(logg_err) and logg_err > 0.50)
        or 'propagated' in v_grav_err_source.lower()
    )
    huge_formal = (
        np.isfinite(v_grav_err_formal)
        and v_grav_err_formal > max(30.0, 0.50 * abs(v_grav))
    )
    gaia_sed_consistent = (
        np.isfinite(vgrav_balmer_gaia_delta)
        and vgrav_balmer_gaia_delta <= 25.0
    )
    base_vgrav_floor = max(10.0, 0.20 * abs(v_grav)) if np.isfinite(v_grav) else 10.0
    if not np.isfinite(v_grav_err_formal):
        v_grav_err_opt = base_vgrav_floor
        v_grav_err_mode = 'missing_formal_replaced_by_floor'
    elif vgrav_model_discrepant:
        v_grav_err_opt = max(v_grav_err_formal, base_vgrav_floor,
                             vgrav_balmer_gaia_delta)
        v_grav_err_mode = 'formal_with_model_discrepancy_floor'
    elif huge_formal and lowres_degenerate and gaia_sed_consistent:
        v_grav_err_opt = max(base_vgrav_floor, vgrav_balmer_gaia_delta)
        v_grav_err_mode = 'degenerate_formal_replaced_by_floor'
    elif v_grav_err_formal > 100.0 and not _truthy_local(rv_result.get('vgrav_model_discrepant', False)):
        v_grav_err_opt = base_vgrav_floor
        v_grav_err_mode = 'degenerate_formal_replaced_by_floor_no_gaia_sed'
    else:
        v_grav_err_opt = max(v_grav_err_formal, base_vgrav_floor)
        v_grav_err_mode = 'formal_with_floor'

    rv_true_opt = rv_obs - v_grav if np.isfinite(rv_obs + v_grav) else np.nan
    rv_true_err_opt = (
        float(np.hypot(rv_obs_err_opt, v_grav_err_opt))
        if np.isfinite(rv_obs_err_opt + v_grav_err_opt) else np.nan
    )
    delta_rv_opt = (
        rv_true_opt - cluster_rv
        if np.isfinite(_finite_or_nan(cluster_rv) + rv_true_opt) else np.nan
    )
    denom = np.nan
    cl_err = _finite_or_nan(cluster_rv_err)
    cl_disp = _finite_or_nan(cluster_intrinsic_rv_dispersion)
    if np.isfinite(rv_true_err_opt):
        terms = [rv_true_err_opt]
        if np.isfinite(cl_err):
            terms.append(cl_err)
        if np.isfinite(cl_disp):
            terms.append(cl_disp)
        denom = float(np.sqrt(np.sum(np.square(terms))))
    rv_sigma_opt = (
        abs(delta_rv_opt) / denom
        if np.isfinite(delta_rv_opt + denom) and denom > 0 else np.nan
    )

    rv_true_gaia_sed = (
        rv_obs - vgrav_gaia_sed
        if np.isfinite(rv_obs + vgrav_gaia_sed) else np.nan
    )

    max_allowed = 'Gold'
    flags = []
    if severe_rv_flag:
        max_allowed = 'Likely'
        flags.append('severe_rv_flag')
    if vgrav_model_discrepant:
        max_allowed = 'Likely'
        flags.append('vgrav_model_discrepant')
    if rv_quality == 'marginal':
        flags.append('marginal_rv_fit_no_gold')
    if v_grav_err_mode.startswith('degenerate_formal_replaced'):
        flags.append('degenerate_vgrav_formal_replaced')
    if huge_formal and lowres_degenerate and not gaia_sed_consistent:
        flags.append('degenerate_vgrav_needs_manual_review')

    tier = 'Not6D'
    reason = 'missing_or_failed_rv'
    abs_delta = abs(delta_rv_opt) if np.isfinite(delta_rv_opt) else np.inf
    if (np.isfinite(rv_true_err_opt + rv_sigma_opt)
            and rv_true_err_opt <= 20 and abs_delta <= 20 and rv_sigma_opt < 2
            and not vgrav_model_discrepant
            and not severe_rv_flag and rv_quality != 'marginal'):
        tier, reason = 'Gold', 'tight RV agreement and small optimized error'
    elif (np.isfinite(rv_true_err_opt + rv_sigma_opt)
          and rv_true_err_opt <= 30 and abs_delta <= 30 and rv_sigma_opt < 2
          and not vgrav_model_discrepant
          and not severe_rv_flag):
        tier, reason = 'Secure', 'secure RV agreement with optimized error'
    elif (np.isfinite(rv_true_err_opt + rv_sigma_opt)
          and rv_true_err_opt <= 50 and abs_delta <= 50 and rv_sigma_opt < 3):
        tier, reason = 'Likely', 'relaxed RV agreement'
    elif (np.isfinite(rv_true_err_opt + rv_sigma_opt)
          and rv_true_err_opt <= 100 and abs_delta <= 100 and rv_sigma_opt < 3):
        tier, reason = 'Compatible', 'broad-error RV compatibility'

    rank = {'Not6D': 0, 'Compatible': 1, 'Likely': 2, 'Secure': 3, 'Gold': 4}
    max_rank = rank.get(max_allowed, 4)
    if rank.get(tier, 0) > max_rank:
        tier = max_allowed
        reason += '; downgraded_by_quality_flags'

    return {
        'rv_obs_err_formal': rv_obs_err_formal,
        'rv_obs_floor_used': rv_floor,
        'rv_obs_err_opt': rv_obs_err_opt,
        'rv_obs_rescorr': 0.0,
        'rv_obs_rescorr_applied': False,
        'rv_obs_err_mode': rv_floor_reason,
        'vgrav_preferred': v_grav,
        'vgrav_err_formal': v_grav_err_formal,
        'vgrav_err_opt': v_grav_err_opt,
        'vgrav_err_mode': v_grav_err_mode,
        'vgrav_balmer': v_grav,
        'vgrav_gaia_sed': vgrav_gaia_sed,
        'vgrav_balmer_gaia_delta': vgrav_balmer_gaia_delta,
        'vgrav_model_discrepant': bool(vgrav_model_discrepant),
        'rv_true_opt': rv_true_opt,
        'rv_true_err_opt': rv_true_err_opt,
        'rv_true_gaia_sed': rv_true_gaia_sed,
        'delta_rv_opt': delta_rv_opt,
        'rv_sigma_opt': rv_sigma_opt,
        'rv_tier_opt': tier,
        'rv_tier_reason': reason,
        'max_allowed_rv_tier': max_allowed,
        'rv_quality_flags': ';'.join(flags),
    }


def _finite_or_nan(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _relative_error(value, err):
    value = _finite_or_nan(value)
    err = _finite_or_nan(err)
    if not np.isfinite(value + err) or value == 0:
        return np.nan
    return abs(err / value)


def _radius_from_mass_logg(mass_msun, logg):
    mass_msun = _finite_or_nan(mass_msun)
    logg = _finite_or_nan(logg)
    if not np.isfinite(mass_msun + logg) or mass_msun <= 0:
        return np.nan
    return np.sqrt(G_CGS * mass_msun * M_SUN_G / (10.0 ** logg)) / R_SUN_CM


def _estimate_gaia_sed_vgrav(rv_result, wd_physical_params=None,
                             rv_physical_variants=None):
    """Return the Gaia/SED M/R gravitational redshift branch when available."""
    direct = _finite_or_nan((rv_result or {}).get('vgrav_gaia_sed'))
    if np.isfinite(direct):
        return direct

    phys = wd_physical_params or {}
    mass = _finite_or_nan(
        phys.get('gaia_hr_mass', phys.get('gaia_sed_mass', np.nan)))
    radius = _finite_or_nan(phys.get(
        'gaia_hr_radius_rsun',
        phys.get('gaia_sed_radius_rsun', np.nan)))
    if not np.isfinite(radius):
        radius = _radius_from_mass_logg(
            mass,
            phys.get('gaia_hr_logg', phys.get('gaia_sed_logg', np.nan)))
    if np.isfinite(mass + radius) and mass > 0 and radius > 0:
        return gravitational_redshift(mass, radius)

    for row in rv_physical_variants or []:
        try:
            source = str(row.get('physical_source', '') or '').lower()
        except AttributeError:
            continue
        if 'gaia' not in source and 'sed' not in source and 'phot' not in source:
            continue
        val = _finite_or_nan(row.get('v_grav_kms'))
        if np.isfinite(val):
            return val
        mass = _finite_or_nan(row.get('mass_msun'))
        radius = _finite_or_nan(row.get('radius_rsun'))
        if np.isfinite(mass + radius) and mass > 0 and radius > 0:
            return gravitational_redshift(mass, radius)
    return np.nan


def _truthy_local(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if isinstance(value, float) and np.isnan(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {'true', '1', 'yes', 'y'}


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


def _finite_positive(value):
    try:
        value = float(value)
    except Exception:
        return None
    if np.isfinite(value) and value > 0:
        return value
    return None


def _first_finite(mapping, *keys):
    if not mapping:
        return None
    for key in keys:
        value = _finite_positive(mapping.get(key))
        if value is not None:
            return value
    return None


def estimate_gravitational_redshift_error(
    v_grav,
    mass_msun=None,
    radius_rsun=None,
    logg=None,
    mass_msun_err=None,
    radius_rsun_err=None,
    logg_err=None,
    used_default_mr=False,
):
    """
    Estimate the uncertainty of the WD gravitational-redshift correction.

    For v_grav = GM/(Rc), independent mass/radius errors give
    sigma_v/v = sqrt((sigma_M/M)^2 + (sigma_R/R)^2).  If the radius was inferred
    from logg, v_grav = sqrt(G M g)/c and logg contributes
    0.5 ln(10) sigma_logg in fractional units.
    """
    v_grav = float(v_grav) if np.isfinite(v_grav) else 0.0
    if v_grav <= 0:
        return 0.0, 'none'

    mass = _finite_positive(mass_msun)
    radius = _finite_positive(radius_rsun)
    mass_err = _finite_positive(mass_msun_err)
    radius_err = _finite_positive(radius_rsun_err)
    logg_err_val = _finite_positive(logg_err)

    terms = []
    pieces = []
    if mass is not None and mass_err is not None:
        terms.append((mass_err / mass) ** 2)
        pieces.append('mass')
    if radius is not None and radius_err is not None:
        terms.append((radius_err / radius) ** 2)
        pieces.append('radius')
    elif logg_err_val is not None:
        terms.append((0.5 * np.log(10.0) * logg_err_val) ** 2)
        pieces.append('logg')

    propagated = abs(v_grav) * np.sqrt(np.sum(terms)) if terms else 0.0
    floor = max(
        GRAV_REDSHIFT_MIN_ERR_KMS,
        GRAV_REDSHIFT_FRACTIONAL_FLOOR * abs(v_grav),
    )
    if used_default_mr:
        floor = max(floor, GRAV_REDSHIFT_DEFAULT_ERR_KMS)

    err = float(max(propagated, floor))
    source = 'propagated_' + '+'.join(pieces) if pieces else 'systematic_floor'
    if used_default_mr:
        source += '+default_MR'
    return err, source


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
        resid = depth - model
        fit_rms = float(np.sqrt(np.nanmean(resid**2))) if resid.size else np.nan
        good_err = np.isfinite(e_norm) & (e_norm > 0)
        if int(np.sum(good_err)) > len(popt):
            fit_chi2_red = float(
                np.nansum((resid[good_err] / e_norm[good_err])**2)
                / max(int(np.sum(good_err)) - len(popt), 1)
            )
        else:
            fit_chi2_red = np.nan

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
            'fit_rms': fit_rms,
            'fit_chi2_red': fit_chi2_red,
        }

    except Exception:
        return None


def fit_emission_line_core(wave, flux, err=None, line_center=6564.61,
                           half_window=25.0, min_snr=4.0):
    """
    Fit a local emission peak and convert its centroid to RV.

    This is separate from fit_line_core(), which is absorption-line oriented.
    The returned dictionary intentionally mirrors the absorption-line fit
    structure so the same plotting/reporting code can display both.
    """
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    mask = ((wave >= line_center - half_window) &
            (wave <= line_center + half_window) &
            np.isfinite(flux))
    if mask.sum() < 8:
        return None

    w = wave[mask]
    f = flux[mask]
    if err is not None:
        e = np.asarray(err, dtype=np.float64)[mask]
        e[e <= 0] = np.nanmedian(e[e > 0]) if np.any(e > 0) else 1.0
    else:
        e = np.ones_like(f)

    wing = np.abs(w - line_center) > half_window * 0.55
    if wing.sum() >= 4:
        p = np.polyfit(w[wing], f[wing], 1)
        continuum = np.polyval(p, w)
    else:
        continuum = np.nanmedian(f) * np.ones_like(f)
    continuum = np.maximum(
        continuum,
        np.nanpercentile(f[f > 0], 5) * 0.1 if np.any(f > 0) else 1e-30)

    f_norm = f / continuum
    e_norm = e / continuum
    excess = f_norm - 1.0

    if wing.sum() >= 4:
        noise = np.nanstd(excess[wing])
    else:
        noise = np.nanmedian(e_norm)
    if not np.isfinite(noise) or noise <= 0:
        noise = 0.05

    i_peak = np.nanargmax(excess)
    amp_guess = excess[i_peak]
    if not np.isfinite(amp_guess) or amp_guess <= 0:
        return None
    significance = amp_guess / noise
    if significance < min_snr:
        return None

    x0_guess = w[i_peak]
    dw = np.nanmedian(np.diff(w))
    if not np.isfinite(dw) or dw <= 0:
        dw = max(half_window * 2.0 / max(len(w), 1), 0.1)
    sigma_guess = max(1.5 * dw, min(3.0, half_window / 3.0))
    center_span = min(half_window * 0.75, 30.0)

    try:
        popt, pcov = curve_fit(
            _gaussian, w, excess,
            p0=[amp_guess, x0_guess, sigma_guess],
            sigma=e_norm,
            bounds=([0, line_center - center_span, 0.3 * dw],
                    [max(amp_guess * 5.0, 1.0),
                     line_center + center_span, half_window]),
            maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        x0_fit = popt[1]
        x0_err = perr[1] if len(perr) > 1 and np.isfinite(perr[1]) else np.nan
        if not np.isfinite(x0_err) or x0_err <= 0:
            x0_err = max(0.1 * dw, abs(popt[2]) / max(significance, 1.0))
        else:
            x0_err = max(x0_err, 0.1 * dw)
        model = _gaussian(w, *popt)
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
            'fwhm': 2.355 * abs(popt[2]),
            'fit_success': True,
            'profile_type': 'gaussian_emission',
            'line_kind': 'emission',
            'wave_fit': w,
            'flux_norm': f_norm,
            'depth_data': excess,
            'model_fit': model,
            'continuum': continuum,
            'significance': significance,
        }
    except Exception:
        return None


def _weighted_mean_and_scatter(rvs, weights, systematic_floor=0.0):
    """Weighted RV mean and uncertainty from formal errors plus line scatter."""
    rvs = np.asarray(rvs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(rvs) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return np.nan, np.nan, np.nan
    rvs = rvs[good]
    weights = weights[good]
    rv_mean = float(np.sum(weights * rvs) / np.sum(weights))
    formal_err = float(1.0 / np.sqrt(np.sum(weights)))
    scatter = 0.0
    scatter_err = 0.0
    if len(rvs) > 1:
        scatter = float(np.sqrt(np.sum(weights * (rvs - rv_mean)**2)
                                / np.sum(weights)))
        n_eff = float(np.sum(weights) ** 2 / np.sum(weights ** 2))
        scatter_err = scatter / np.sqrt(max(n_eff, 1.0))
    rv_err = max(formal_err, scatter_err, float(systematic_floor or 0.0))
    return rv_mean, rv_err, scatter


def _line_fit_reject_reason(result):
    """Return a short reason if a line-core fit is not reliable enough for RV."""
    if result is None or not result.get('fit_success'):
        return 'fit_failed_or_too_shallow'
    rv = result.get('rv_kms')
    rv_err = result.get('rv_err_kms')
    fwhm = result.get('fwhm')
    amp = result.get('amplitude')
    vals = [rv, rv_err, fwhm, amp]
    if not all(np.isfinite(v) for v in vals):
        return 'nonfinite_line_fit'
    if amp < 0.02:
        return 'line_too_shallow'
    if rv_err <= 0 or rv_err > LINE_RV_MAX_FORMAL_ERR_KMS:
        return 'formal_rv_error_too_large'
    if abs(rv) > LINE_RV_MAX_ABS_RV_KMS:
        return 'rv_centroid_unphysical'
    if fwhm < LINE_RV_MIN_FWHM_A or fwhm > LINE_RV_MAX_FWHM_A:
        return 'line_width_unphysical'
    fit_rms = result.get('fit_rms')
    if np.isfinite(fit_rms) and fit_rms > max(0.35, 1.5 * amp):
        return 'line_profile_residual_too_large'
    return ''


def _select_dominant_rv_cluster(rvs, weights, names, cluster_width=45.0):
    """
    Pick the most common nearby velocity group before averaging.

    A sliding velocity window is used instead of a plain global mean.  Ties are
    broken by smaller weighted scatter, then by larger total weight.
    """
    rvs = np.asarray(rvs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    names = list(names)
    n = len(rvs)
    if n == 0:
        return np.array([], dtype=bool), {}
    if n <= 2:
        keep = np.ones(n, dtype=bool)
        rv_mean, rv_err, scatter = _weighted_mean_and_scatter(rvs, weights)
        return keep, {
            'rv_selection': 'all_lines',
            'cluster_width_kms': float(cluster_width),
            'cluster_center_kms': rv_mean,
            'cluster_scatter_kms': scatter,
            'selected_lines': names,
            'rejected_lines': [],
            'n_lines_total': int(n),
        }

    order = np.argsort(rvs)
    sorted_rv = rvs[order]
    best = None
    for i in range(n):
        j = i
        while j + 1 < n and sorted_rv[j + 1] - sorted_rv[i] <= cluster_width:
            j += 1
        idx = order[i:j + 1]
        rv_mean, rv_err, scatter = _weighted_mean_and_scatter(rvs[idx], weights[idx])
        total_weight = float(np.sum(weights[idx]))
        candidate = {
            'idx': idx,
            'count': len(idx),
            'scatter': scatter if np.isfinite(scatter) else np.inf,
            'weight': total_weight,
            'center': rv_mean,
            'err': rv_err,
        }
        if best is None:
            best = candidate
            continue
        if candidate['count'] > best['count']:
            best = candidate
        elif candidate['count'] == best['count']:
            if candidate['scatter'] < best['scatter'] - 1e-9:
                best = candidate
            elif abs(candidate['scatter'] - best['scatter']) <= 1e-9:
                if candidate['weight'] > best['weight']:
                    best = candidate

    keep = np.zeros(n, dtype=bool)
    keep[best['idx']] = True
    selected = [names[i] for i in range(n) if keep[i]]
    rejected = [names[i] for i in range(n) if not keep[i]]
    return keep, {
        'rv_selection': 'dominant_velocity_cluster',
        'cluster_width_kms': float(cluster_width),
        'cluster_center_kms': float(best['center']),
        'cluster_scatter_kms': float(best['scatter']),
        'selected_lines': selected,
        'rejected_lines': rejected,
        'n_lines_total': int(n),
    }


def measure_rv_from_lines(wave, flux, err=None, wd_type='DA',
                          dc_mode=False, preferred_lines=None,
                          strict_consistency=False,
                          min_consistent_lines=None):
    """
    从多条 Balmer / He I 线核拟合测量 RV.

    多条线不再直接全局平均；先找速度相近且出现次数最多的主簇，再对
    主簇内的线做加权平均，避免单条偏离线把 RV_obs 拉偏。

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
    lines = _filter_line_dict(lines, preferred_lines)
    if not lines:
        return None

    if min_consistent_lines is None:
        min_consistent_lines = 2 if strict_consistency else 1

    line_results = {}
    all_line_results = {}
    line_rejection_reasons = {}
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
        reason = _line_fit_reject_reason(result)
        if result is None:
            line_rejection_reasons[name] = reason
            continue
        all_line_results[name] = result
        if dc_mode and abs(result['rv_kms']) > 500:
            reason = 'dc_mode_single_line_rv_too_large'
        if reason:
            result['used_for_rv'] = False
            result['rv_reject_reason'] = reason
            line_rejection_reasons[name] = reason
            continue
        line_results[name] = result
        rvs.append(result['rv_kms'])
        weights.append(1.0 / max(result['rv_err_kms'], 1.0)**2)
        names_used.append(name)

    if not rvs:
        return None
    if len(rvs) < min_consistent_lines:
        return None

    rvs = np.array(rvs)
    weights = np.array(weights)
    all_line_results.update(line_results)

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

    cluster_keep, cluster_info = _select_dominant_rv_cluster(
        rvs, weights, names_used, cluster_width=45.0)
    if np.sum(cluster_keep) >= min_consistent_lines:
        rvs_selected = rvs[cluster_keep]
        weights_selected = weights[cluster_keep]
        selected_names = [names_used[i] for i in range(len(names_used))
                          if cluster_keep[i]]
    elif strict_consistency:
        return None
    else:
        rvs_selected = rvs
        weights_selected = weights
        selected_names = list(names_used)
        cluster_info = {
            'rv_selection': 'all_lines',
            'cluster_width_kms': 45.0,
            'cluster_center_kms': np.nan,
            'cluster_scatter_kms': np.nan,
            'selected_lines': selected_names,
            'rejected_lines': [],
            'n_lines_total': int(len(rvs)),
        }

    for name, lr in all_line_results.items():
        lr['used_for_rv'] = name in selected_names
        if name not in selected_names and not lr.get('rv_reject_reason'):
            lr['rv_reject_reason'] = 'velocity_outlier_from_dominant_cluster'
            line_rejection_reasons[name] = lr['rv_reject_reason']
    line_results = {k: all_line_results[k] for k in selected_names
                    if k in all_line_results}

    # 主速度簇加权平均。线间散布按有效线数降低，但保留系统底噪，避免
    # 多条相似低分辨率线把误差压得不真实。
    rv_mean, rv_err, scatter = _weighted_mean_and_scatter(
        rvs_selected, weights_selected,
        systematic_floor=LINE_RV_SYSTEMATIC_FLOOR_KMS)
    cluster_info['cluster_center_kms'] = rv_mean
    cluster_info['cluster_scatter_kms'] = scatter
    rejected_names = list(cluster_info.get('rejected_lines', []) or [])
    for name in line_rejection_reasons:
        if name not in rejected_names and name not in selected_names:
            rejected_names.append(name)
    cluster_info['rejected_lines'] = rejected_names

    # RV 质量评估
    if dc_mode:
        scatter_val = np.std(rvs_selected) if len(rvs_selected) > 1 else 999
        if len(rvs_selected) >= 3 and scatter_val < 50:
            rv_quality = 'good'
        elif len(rvs_selected) >= 2 and scatter_val < 100:
            rv_quality = 'marginal'
        else:
            rv_quality = 'unreliable'
    else:
        if len(rvs_selected) >= 3 and scatter < 25 and rv_err < 20:
            rv_quality = 'good'
        elif len(rvs_selected) >= 2 and scatter < 50 and rv_err < 50:
            rv_quality = 'marginal'
        else:
            rv_quality = 'unreliable'

    return {
        'rv_obs': rv_mean,
        'rv_obs_err': rv_err,
        'line_results': line_results,
        'all_line_results': all_line_results,
        'n_lines_used': len(rvs_selected),
        'n_lines_total': len(all_line_results),
        'method': 'line_core_fit_dominant_cluster',
        'rv_quality': rv_quality,
        'line_rejection_reasons': line_rejection_reasons,
        **cluster_info,
    }


def measure_rv_from_emission_lines(wave, flux, err=None, preferred_lines=None,
                                   include_forbidden=False, min_snr=4.0,
                                   min_lines=2, cluster_width=45.0):
    """
    Measure RV from emission lines using the dominant repeated velocity group.

    Multiple lines are fitted first, then the most populated nearby velocity
    window is selected. Lines outside that velocity group are marked rejected
    and do not enter the final RV.
    """
    lines = dict(EMISSION_LINES)
    if include_forbidden:
        lines.update(FORBIDDEN_EMISSION_LINES)
    lines = _filter_line_dict(lines, preferred_lines)
    if not lines:
        return None

    wave = np.asarray(wave, dtype=float)
    line_results = {}
    rvs = []
    weights = []
    names_used = []

    for name, lam in lines.items():
        if lam < np.nanmin(wave) + 30 or lam > np.nanmax(wave) - 30:
            continue
        hw = 35 if ('alpha' in name.lower() or 'nii' in name.lower()
                    or 'sii' in name.lower()) else 25
        result = fit_emission_line_core(
            wave, flux, err, line_center=lam, half_window=hw,
            min_snr=min_snr)
        if result is None or not result.get('fit_success'):
            continue
        if abs(result['rv_kms']) > 800:
            continue
        line_results[name] = result
        rvs.append(result['rv_kms'])
        weights.append(1.0 / max(result['rv_err_kms'], 1.0)**2)
        names_used.append(name)

    if not rvs:
        return None

    rvs = np.asarray(rvs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    all_line_results = dict(line_results)
    cluster_keep, cluster_info = _select_dominant_rv_cluster(
        rvs, weights, names_used, cluster_width=cluster_width)

    if np.sum(cluster_keep) >= min_lines:
        rvs_selected = rvs[cluster_keep]
        weights_selected = weights[cluster_keep]
        selected_names = [names_used[i] for i in range(len(names_used))
                          if cluster_keep[i]]
    elif len(rvs) == 1 and min_lines <= 1:
        rvs_selected = rvs
        weights_selected = weights
        selected_names = list(names_used)
        cluster_info = {
            'rv_selection': 'single_requested_line',
            'cluster_width_kms': float(cluster_width),
            'cluster_center_kms': float(rvs[0]),
            'cluster_scatter_kms': 0.0,
            'selected_lines': selected_names,
            'rejected_lines': [],
            'n_lines_total': int(len(rvs)),
        }
    else:
        return None

    for name, lr in all_line_results.items():
        lr['used_for_rv'] = name in selected_names
    line_results = {k: all_line_results[k] for k in selected_names
                    if k in all_line_results}

    rv_mean, rv_err, scatter = _weighted_mean_and_scatter(
        rvs_selected, weights_selected,
        systematic_floor=LINE_RV_SYSTEMATIC_FLOOR_KMS)
    cluster_info['cluster_center_kms'] = rv_mean
    cluster_info['cluster_scatter_kms'] = scatter

    if len(rvs_selected) >= 3 and scatter < 25 and rv_err < 20:
        rv_quality = 'good'
    elif len(rvs_selected) >= 2 and scatter < 50 and rv_err < 40:
        rv_quality = 'marginal'
    else:
        rv_quality = 'unreliable'

    return {
        'rv_obs': rv_mean,
        'rv_obs_err': rv_err,
        'line_results': line_results,
        'all_line_results': all_line_results,
        'n_lines_used': len(rvs_selected),
        'n_lines_total': len(all_line_results),
        'method': 'emission_line_dominant_cluster',
        'rv_quality': rv_quality,
        **cluster_info,
    }


# ==================================================================
#  综合 RV 测量: 线核 + CCF 融合
# ==================================================================

def measure_rv_obs(wave, flux, err=None, wd_type='DA', dc_mode=False,
                   rv_mode='absorption', preferred_lines=None):
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
    mode = str(rv_mode or 'absorption').lower()
    if mode in {'line', 'line_core', 'photospheric'}:
        mode = 'absorption'
    line_only_modes = {
        'line_core_only',
        'balmer_line_core_only',
        'absorption_line_core_only',
        'absorption_lines_only',
    }
    line_only = mode in line_only_modes

    line_result = None
    emission_result = None
    ccf_result = None

    if mode in {'absorption', 'auto'} or line_only:
        line_result = measure_rv_from_lines(
            wave, flux, err, wd_type=wd_type, dc_mode=dc_mode,
            preferred_lines=preferred_lines,
            strict_consistency=line_only,
            min_consistent_lines=2 if line_only else None)

    if mode in {'emission', 'auto'}:
        min_lines = 1 if preferred_lines and len(preferred_lines) == 1 else 2
        emission_result = measure_rv_from_emission_lines(
            wave, flux, err, preferred_lines=preferred_lines,
            min_lines=min_lines)

    if mode in {'absorption', 'auto', 'ccf'}:
        # CCF (复用 rv_fitting.py)
        from .rv_fitting import measure_rv as ccf_measure_rv
        ccf_result = ccf_measure_rv(wave, flux, err)

    # 决策
    rv_obs = None
    rv_err = None
    source = None

    has_line = (line_result is not None and line_result['n_lines_used'] >= 2)
    has_emission = (
        emission_result is not None
        and emission_result['n_lines_used'] >= (
            1 if preferred_lines and len(preferred_lines) == 1 else 2
        )
    )
    has_ccf = (ccf_result is not None and ccf_result.get('ccf_height', 0) > 0.05)

    chosen_method = ''
    chosen_line_fit = line_result

    if line_only:
        if not has_line:
            return None
        rv_obs = line_result['rv_obs']
        rv_err = line_result['rv_obs_err']
        chosen_method = line_result.get('method', 'line_core')
        source = (f'line_core only '
                  f'({line_result["n_lines_used"]}/'
                  f'{line_result.get("n_lines_total", line_result["n_lines_used"])} lines)')
    elif mode == 'emission':
        if not has_emission:
            return None
        rv_obs = emission_result['rv_obs']
        rv_err = emission_result['rv_obs_err']
        chosen_line_fit = emission_result
        chosen_method = emission_result.get('method', 'emission_line')
        source = (f'emission_line dominant cluster '
                  f'({emission_result["n_lines_used"]}/'
                  f'{emission_result.get("n_lines_total", emission_result["n_lines_used"])} lines)')
    elif mode == 'auto' and has_emission and not has_line:
        rv_obs = emission_result['rv_obs']
        rv_err = emission_result['rv_obs_err']
        chosen_line_fit = emission_result
        chosen_method = emission_result.get('method', 'emission_line')
        source = (f'emission_line dominant cluster '
                  f'({emission_result["n_lines_used"]}/'
                  f'{emission_result.get("n_lines_total", emission_result["n_lines_used"])} lines)')
    elif mode == 'auto' and has_emission and has_line:
        # Prefer the line family with smaller error only if both are internally
        # coherent; the source label preserves which family was chosen.
        if emission_result['rv_obs_err'] < line_result['rv_obs_err']:
            rv_obs = emission_result['rv_obs']
            rv_err = emission_result['rv_obs_err']
            chosen_line_fit = emission_result
            chosen_method = emission_result.get('method', 'emission_line')
            source = (f'emission_line dominant cluster '
                      f'({emission_result["n_lines_used"]}/'
                      f'{emission_result.get("n_lines_total", emission_result["n_lines_used"])} lines, '
                      f'absorption available)')
        else:
            rv_obs = line_result['rv_obs']
            rv_err = line_result['rv_obs_err']
            chosen_line_fit = line_result
            chosen_method = line_result.get('method', 'line_core')
            source = (f'line_core dominant cluster '
                      f'({line_result["n_lines_used"]}/'
                      f'{line_result.get("n_lines_total", line_result["n_lines_used"])} lines, '
                      f'emission available)')
    elif has_line and has_ccf:
        rv_line = line_result['rv_obs']
        rv_ccf = ccf_result['rv']
        diff = abs(rv_line - rv_ccf)
        combined_err = np.sqrt(line_result['rv_obs_err']**2 +
                               ccf_result['rv_err']**2)

        if diff < 2.0 * combined_err:
            # 一致 → 取线核 (更精确)
            rv_obs = rv_line
            rv_err = line_result['rv_obs_err']
            chosen_method = line_result.get('method', 'line_core')
            source = (f'line_core dominant cluster '
                      f'({line_result["n_lines_used"]}/'
                      f'{line_result.get("n_lines_total", line_result["n_lines_used"])} lines)')
        else:
            # 不一致 → 如果线核线数多则信线核, 否则信 CCF
            if line_result['n_lines_used'] >= 3:
                rv_obs = rv_line
                rv_err = line_result['rv_obs_err']
                chosen_method = line_result.get('method', 'line_core')
                source = (f'line_core dominant cluster '
                          f'({line_result["n_lines_used"]}/'
                          f'{line_result.get("n_lines_total", line_result["n_lines_used"])} lines, '
                          f'CCF disagrees)')
            else:
                rv_obs = ccf_result['rv']
                rv_err = ccf_result['rv_err']
                chosen_method = 'CCF'
                chosen_line_fit = None
                source = 'CCF (line_core disagrees)'
    elif has_line:
        rv_obs = line_result['rv_obs']
        rv_err = line_result['rv_obs_err']
        chosen_method = line_result.get('method', 'line_core')
        source = (f'line_core dominant cluster '
                  f'({line_result["n_lines_used"]}/'
                  f'{line_result.get("n_lines_total", line_result["n_lines_used"])} lines)')
    elif has_ccf:
        rv_obs = ccf_result['rv']
        rv_err = ccf_result['rv_err']
        chosen_method = 'CCF'
        chosen_line_fit = None
        source = 'CCF'
    else:
        return None

    # rv_quality 从线核拟合传递, CCF 默认 'good'
    rv_quality = 'good'
    if chosen_line_fit and chosen_line_fit.get('rv_quality'):
        rv_quality = chosen_line_fit['rv_quality']

    return {
        'rv_obs': rv_obs,
        'rv_obs_err': rv_err,
        'source': source,
        'method': chosen_method,
        'line_fit': chosen_line_fit,
        'emission_fit': emission_result,
        'ccf_fit': ccf_result,
        'rv_quality': rv_quality,
    }


# ==================================================================
#  主 API: 测量 RV_true
# ==================================================================

def measure_true_rv(wave, flux, err=None,
                    mass_msun=None, radius_rsun=None, logg=None,
                    mass_msun_err=None, radius_rsun_err=None, logg_err=None,
                    wd_type='DA', spectral_type=None,
                    rv_mode='absorption', preferred_lines=None,
                    apply_grav_redshift=None):
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
        rv_true_random_err, rv_true_grav_err,
        rv_obs, rv_obs_err,
        v_grav, v_grav_err,
        rv_obs_details: {...},
        mass_msun, radius_rsun,
    }
    """
    # Step 1: 测量 RV_obs
    dc_mode = (spectral_type is not None and
               spectral_type.upper().startswith('DC'))
    obs = measure_rv_obs(
        wave, flux, err, wd_type=wd_type, dc_mode=dc_mode,
        rv_mode=rv_mode, preferred_lines=preferred_lines)
    if obs is None:
        return None

    rv_obs = obs['rv_obs']
    rv_obs_err = obs['rv_obs_err']

    # Step 2: 计算 V_grav
    v_grav = 0.0
    R_used = radius_rsun
    M_used = mass_msun
    used_default_mr = False

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
        used_default_mr = True

    # Step 3: RV_true.  Emission lines usually trace gas/accretion/companion
    # rather than WD photosphere, so do not subtract WD gravitational redshift
    # unless the caller explicitly requests it.
    is_emission_rv = str(obs.get('method', '')).startswith('emission')
    if apply_grav_redshift is None:
        apply_grav_redshift = not is_emission_rv
    rv_true = rv_obs - v_grav if apply_grav_redshift else rv_obs
    v_grav_err, v_grav_err_source = estimate_gravitational_redshift_error(
        v_grav,
        mass_msun=M_used,
        radius_rsun=R_used,
        logg=logg,
        mass_msun_err=mass_msun_err,
        radius_rsun_err=radius_rsun_err,
        logg_err=logg_err,
        used_default_mr=used_default_mr,
    )
    rv_true_random_err = rv_obs_err
    rv_true_grav_err = v_grav_err if apply_grav_redshift else 0.0
    rv_true_err = float(np.sqrt(rv_true_random_err**2 + rv_true_grav_err**2))
    rv_true_grav_err_conservative = 0.0
    if apply_grav_redshift:
        rv_true_grav_err_conservative = max(
            rv_true_grav_err,
            GRAV_REDSHIFT_6D_FLOOR_KMS,
            GRAV_REDSHIFT_6D_FRACTIONAL_FLOOR * abs(v_grav),
        )
    rv_true_err_conservative = float(
        np.sqrt(rv_true_random_err**2 + rv_true_grav_err_conservative**2)
    )

    return {
        'rv_true': rv_true,
        'rv_true_err': rv_true_err,
        'rv_true_err_conservative_6d': rv_true_err_conservative,
        'rv_true_random_err': rv_true_random_err,
        'rv_true_grav_err': rv_true_grav_err,
        'rv_true_grav_err_conservative_6d': rv_true_grav_err_conservative,
        'rv_obs': rv_obs,
        'rv_obs_err': rv_obs_err,
        'v_grav': v_grav,
        'v_grav_err': v_grav_err,
        'v_grav_err_source': v_grav_err_source,
        'rv_obs_source': obs['source'],
        'rv_obs_details': obs,
        'rv_quality': obs.get('rv_quality', 'good'),
        'gravitational_redshift_applied': bool(apply_grav_redshift),
        'rv_mode': rv_mode,
        'preferred_lines': ';'.join(preferred_lines or []),
        'mass_msun': M_used,
        'mass_msun_err': mass_msun_err,
        'radius_rsun': R_used,
        'radius_rsun_err': radius_rsun_err,
        'logg': logg,
        'logg_err': logg_err,
    }


# ==================================================================
#  批量处理: 对 merged_all.csv 中的所有 WD 计算 RV_true
# ==================================================================

def run_rv_correction(wave, flux, err=None,
                      physical_params=None,
                      survey_name='',
                      output_dir=None, ra=None, dec=None,
                      rv_mode='absorption', preferred_lines=None,
                      apply_grav_redshift=None):
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
        mass_err = _first_finite(
            physical_params,
            'mass_err', 'mass_msun_err', 'mass_msun_mr_err',
            'wd_mass_msun_err')
        radius_err = _first_finite(
            physical_params,
            'radius_rsun_err', 'radius_err', 'radius_mr_err',
            'wd_radius_rsun_err')
        logg_err = _first_finite(physical_params, 'logg_err', 'wd_logg_err')
    else:
        mass_err = radius_err = logg_err = None

    result = measure_true_rv(wave, flux, err,
                              mass_msun=mass, radius_rsun=radius,
                              logg=logg,
                              mass_msun_err=mass_err,
                              radius_rsun_err=radius_err,
                              logg_err=logg_err,
                              wd_type=wd_type,
                              spectral_type=spectral_type,
                              rv_mode=rv_mode,
                              preferred_lines=preferred_lines,
                              apply_grav_redshift=apply_grav_redshift)

    if result is None:
        print(f"  RV 测量失败 ({survey_name})")
        return None

    print(f"  RV_{survey_name}: obs={result['rv_obs']:.2f} "
          f"- V_grav={result['v_grav']:.2f}±{result['v_grav_err']:.2f} "
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

    line_results = line_fit.get('all_line_results') or line_fit['line_results']
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
        line_kind = lr.get('line_kind', 'absorption')

        used = lr.get('used_for_rv', True)
        data_color = 'black' if used else '0.55'
        fit_color = 'red' if used else '0.45'
        fit_alpha = 0.85 if used else 0.45

        # 归一化光谱
        ax.plot(w, f_norm, color=data_color, lw=0.8, alpha=0.7, label='Data')
        # 拟合 profile: absorption stores positive depth, emission stores excess.
        model_flux = 1 + model if line_kind == 'emission' else 1 - model
        ax.plot(w, model_flux, color=fit_color, lw=1.5, alpha=fit_alpha, label='Fit')

        # 实验室波长
        ax.axvline(lr['line_center_lab'], color='blue', ls=':', alpha=0.5,
                   label=f'Lab: {lr["line_center_lab"]:.1f} A')
        # 拟合线心
        ax.axvline(lr['line_center_obs'], color='red', ls='--', alpha=0.5,
                   label=f'Obs: {lr["line_center_obs"]:.1f} A')

        tag = 'used' if used else 'rejected'
        ax.set_title(
            f'{name}  RV={lr["rv_kms"]:.1f} +/- {lr["rv_err_kms"]:.1f} km/s ({tag})',
            fontsize=10)
        ax.set_xlabel('Wavelength (A)')
        ax.set_ylabel('Normalized flux')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for idx in range(n_lines, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    selected = []
    rejected = []
    if line_fit:
        selected = line_fit.get('selected_lines', []) or []
        rejected = line_fit.get('rejected_lines', []) or []
    rv_true_random = rv_result.get('rv_true_random_err', np.nan)
    rv_true_grav = rv_result.get('rv_true_grav_err', np.nan)
    err_note = (
        f"RV_obs = {rv_result['rv_obs']:.2f} +/- {rv_result['rv_obs_err']:.2f} km/s;  "
        f"V_grav = {rv_result['v_grav']:.2f} +/- {rv_result.get('v_grav_err', 0.0):.2f} km/s;  "
        f"RV_true = {rv_result['rv_true']:.2f} +/- {rv_result['rv_true_err']:.2f} km/s"
    )
    comp_note = (
        f"sigma_true = sqrt(sigma_RVobs^2 + sigma_Vgrav^2) = "
        f"sqrt({rv_true_random:.2f}^2 + {rv_true_grav:.2f}^2) km/s"
        if np.isfinite(rv_true_random) and np.isfinite(rv_true_grav)
        else "sigma_true = sqrt(sigma_RVobs^2 + sigma_Vgrav^2)"
    )
    conservative_err = rv_result.get('rv_true_err_conservative_6d', np.nan)
    conservative_grav = rv_result.get('rv_true_grav_err_conservative_6d', np.nan)
    if np.isfinite(conservative_err) and np.isfinite(conservative_grav):
        comp_note += (
            f";  6D conservative sigma = {conservative_err:.2f} km/s "
            f"(grav term {conservative_grav:.2f})"
        )
    line_note = (
        f"Used: {', '.join(selected) if selected else 'none'}"
        f"{'; rejected: ' + ', '.join(rejected) if rejected else ''}"
    )

    fig.text(
        0.5, 0.015,
        err_note + "\n" + comp_note + "\n" + line_note,
        ha='center', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor='0.55', alpha=0.92),
    )

    # 总标题
    coord_str = f"  RA={ra:.4f} DEC={dec:.4f}" if ra is not None else ""
    fig.suptitle(
        f"RV Line Core Fits — {survey_name}{coord_str}\n"
        f"RV_obs = {rv_result['rv_obs']:.2f} +/- {rv_result['rv_obs_err']:.2f} km/s,  "
        f"V_grav = {rv_result['v_grav']:.2f} +/- {rv_result.get('v_grav_err', 0.0):.2f} km/s,  "
        f"RV_true = {rv_result['rv_true']:.2f} +/- {rv_result['rv_true_err']:.2f} km/s",
        fontsize=12, y=1.03)

    fig.tight_layout(rect=[0, 0.15, 1, 0.94])
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
    lines.append(f"  V_grav = G*M / (R*c) = {rv_result['v_grav']:.4f} +/- "
                 f"{rv_result.get('v_grav_err', 0.0):.4f} km/s")
    lines.append(f"  V_grav_err source = {rv_result.get('v_grav_err_source', 'none')}")
    lines.append(f"  M_WD = {rv_result['mass_msun']:.4f} M_sun")
    if rv_result.get('mass_msun_err') is not None:
        lines.append(f"  sigma_M_WD = {rv_result['mass_msun_err']:.4f} M_sun")
    lines.append(f"  R_WD = {rv_result['radius_rsun']:.6f} R_sun")
    if rv_result.get('radius_rsun_err') is not None:
        lines.append(f"  sigma_R_WD = {rv_result['radius_rsun_err']:.6f} R_sun")
    lines.append("")

    lines.append("--- True Kinematic RV ---")
    if rv_result.get('gravitational_redshift_applied', True):
        lines.append(f"  RV_true = RV_obs - V_grav")
        lines.append(f"  RV_true = {rv_result['rv_obs']:.4f} - "
                     f"{rv_result['v_grav']:.4f}")
    else:
        lines.append("  RV_true = RV_obs")
        lines.append("  WD photospheric gravitational redshift was not subtracted")
        lines.append("  because the adopted RV is from emission lines.")
    lines.append(f"  RV_true = {rv_result['rv_true']:.4f} +/- "
                 f"{rv_result['rv_true_err']:.4f} km/s")
    lines.append(f"  sigma_random(line RV) = {rv_result.get('rv_true_random_err', np.nan):.4f} km/s")
    lines.append(f"  sigma_grav(redshift) = {rv_result.get('rv_true_grav_err', np.nan):.4f} km/s")
    lines.append("  sigma_total used for 6D = sqrt(sigma_random^2 + sigma_grav^2)")
    if rv_result.get('rv_true_err_conservative_6d') is not None:
        lines.append(
            f"  conservative 6D sigma_grav = "
            f"{rv_result.get('rv_true_grav_err_conservative_6d', np.nan):.4f} km/s"
        )
        lines.append(
            f"  conservative 6D sigma_total = "
            f"{rv_result.get('rv_true_err_conservative_6d', np.nan):.4f} km/s"
        )
    lines.append("")

    # 逐线结果
    obs = rv_result.get('rv_obs_details', {})
    line_fit = obs.get('line_fit')
    if line_fit and (line_fit.get('line_results') or line_fit.get('all_line_results')):
        lines.append("--- Individual Line Fits ---")
        selected_list = line_fit.get('selected_lines', []) or []
        rejected_list = line_fit.get('rejected_lines', []) or []
        selected = set(selected_list)
        display_results = line_fit.get('all_line_results') or line_fit['line_results']
        for name, lr in display_results.items():
            status = 'USED' if lr.get('used_for_rv', name in selected) else 'REJECT'
            reason = lr.get('rv_reject_reason', '')
            lines.append(f"  {name:12s}  {status:6s}  "
                        f"lab={lr['line_center_lab']:.2f} A  "
                        f"obs={lr['line_center_obs']:.2f} A  "
                        f"dL={lr['delta_lambda']:+.2f} A  "
                        f"RV={lr['rv_kms']:+.2f} +/- {lr['rv_err_kms']:.2f} km/s  "
                        f"FWHM={lr['fwhm']:.1f} A"
                        f"{'  reason=' + reason if reason else ''}")
        lines.append(f"\n  RV selection: {line_fit.get('rv_selection', 'weighted_mean')}")
        lines.append(f"  Dominant velocity window: {line_fit.get('cluster_width_kms', np.nan):.1f} km/s")
        lines.append(f"  Selected lines: {', '.join(selected_list) if selected_list else 'none'}")
        if rejected_list:
            lines.append(f"  Rejected lines: {', '.join(rejected_list)}")
        lines.append(f"  Cluster scatter: {line_fit.get('cluster_scatter_kms', np.nan):.2f} km/s")
        lines.append(f"  Cluster weighted mean: {line_fit['rv_obs']:.2f} "
                    f"+/- {line_fit['rv_obs_err']:.2f} km/s "
                    f"({line_fit['n_lines_used']}/"
                    f"{line_fit.get('n_lines_total', line_fit['n_lines_used'])} lines)")

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
    obs = rv_result.get('rv_obs_details') or {}
    line_fit = obs.get('line_fit') or {}
    row = {
        'survey': survey_name,
        'rv_obs_kms': rv_result.get('rv_obs'),
        'rv_obs_err_kms': rv_result.get('rv_obs_err'),
        'rv_obs_source': rv_result.get('rv_obs_source', ''),
        'rv_selection': line_fit.get('rv_selection'),
        'rv_cluster_width_kms': line_fit.get('cluster_width_kms'),
        'rv_cluster_scatter_kms': line_fit.get('cluster_scatter_kms'),
        'rv_selected_lines': ';'.join(line_fit.get('selected_lines', []) or []),
        'rv_rejected_lines': ';'.join(line_fit.get('rejected_lines', []) or []),
        'rv_line_rejection_reasons': ';'.join(
            f"{k}:{v}" for k, v in (line_fit.get('line_rejection_reasons') or {}).items()
        ),
        'n_lines_used': line_fit.get('n_lines_used'),
        'n_lines_total': line_fit.get('n_lines_total'),
        'v_grav_kms': rv_result.get('v_grav'),
        'v_grav_err_kms': rv_result.get('v_grav_err'),
        'v_grav_err_source': rv_result.get('v_grav_err_source'),
        'v_grav_applied': rv_result.get('gravitational_redshift_applied'),
        'rv_true_kms': rv_result.get('rv_true'),
        'rv_true_err_kms': rv_result.get('rv_true_err'),
        'rv_true_err_conservative_6d_kms': rv_result.get('rv_true_err_conservative_6d'),
        'rv_true_random_err_kms': rv_result.get('rv_true_random_err'),
        'rv_true_grav_err_kms': rv_result.get('rv_true_grav_err'),
        'rv_true_grav_err_conservative_6d_kms': rv_result.get('rv_true_grav_err_conservative_6d'),
        'rv_mode': rv_result.get('rv_mode', ''),
        'preferred_lines': rv_result.get('preferred_lines', ''),
        'mass_msun': rv_result.get('mass_msun'),
        'mass_msun_err': rv_result.get('mass_msun_err'),
        'radius_rsun': rv_result.get('radius_rsun'),
        'radius_rsun_err': rv_result.get('radius_rsun_err'),
        'logg': rv_result.get('logg'),
        'logg_err': rv_result.get('logg_err'),
        'rv_quality': rv_result.get('rv_quality', ''),
    }
    df = pd.DataFrame([row])
    path = os.path.join(output_dir, f'rv_correction_{survey_name.lower()}.csv')
    df.to_csv(path, index=False)
    return path
