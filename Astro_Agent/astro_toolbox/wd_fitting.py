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

# Literature WD atmospheric fits usually use continuum-normalized Balmer line
# profiles, not the absolute flux-calibrated spectrum.  For J1529+2928 the
# published fits are H-beta through H8, so keep that line set explicit.
BALMER_PROFILE_LINES = {
    'H-beta':   4862.68,
    'H-gamma':  4341.68,
    'H-delta':  4102.89,
    'H-epsilon': 3971.20,
    'H8':       3890.16,
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


_NN_WD_CACHE = {}


def _nn_model_dir(kind):
    """Return the local DA/DB neural-network WD model directory."""
    kind = str(kind).upper()
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default = os.path.join(parent, 'data', f'{kind}WDmodel')
    env_name = f'{kind}WD_NN_MODEL_DIR'
    return os.environ.get(env_name, default)


def _coerce_wavelength_angstrom(wavelength):
    """
    Convert a model wavelength array to Angstrom when the unit is obvious.

    The supplied DA/DB grids have ranges 899-29992 and 3199-9997, i.e. Angstrom.
    This helper keeps a guardrail for future grids saved in nm or micron.
    """
    wave = np.asarray(wavelength, dtype=np.float64)
    finite = wave[np.isfinite(wave)]
    if finite.size == 0:
        return wave, 'unknown'
    wmax = float(np.nanmax(finite))
    wmed = float(np.nanmedian(finite))
    if 100.0 <= wmed <= 300000.0 and wmax > 1000.0:
        return wave, 'Angstrom'
    if 100.0 <= wmed <= 3000.0:
        return wave * 10.0, 'nm->Angstrom'
    if 0.05 <= wmed <= 30.0:
        return wave * 1.0e4, 'micron->Angstrom'
    return wave, 'unknown'


def load_nn_wd_templates(specclass='DA', model_dir=None):
    """
    Load local neural-network WD spectral grids from ``*_x.npy``, ``*_y.npy``,
    and ``*_wl.npy``.

    Returns
    -------
    dict
        ``{(teff, logg): {'wavelength', 'flux', 'spectral_type',
        'model_source', 'wavelength_unit', 'flux_unit'}}``.

    Notes
    -----
    The checked local grids are saved on an Angstrom wavelength scale.  The flux
    arrays are surface-like ``f_lambda`` spectra.  Observed SDSS/DESI spectra in
    this toolbox are usually stored in survey units of ``1e-17 erg s-1 cm-2 A-1``;
    radius checks therefore convert the fitted scale before using
    ``R = d sqrt(scale)``.
    """
    kind = str(specclass or 'DA').upper()
    if kind.startswith('DA'):
        kind = 'DA'
    elif kind.startswith('DB'):
        kind = 'DB'
    else:
        return {}

    if model_dir is None:
        model_dir = _nn_model_dir(kind)
    cache_key = (kind, os.path.abspath(model_dir))
    if cache_key in _NN_WD_CACHE:
        return _NN_WD_CACHE[cache_key]

    x_path = os.path.join(model_dir, f'{kind}_x.npy')
    y_path = os.path.join(model_dir, f'{kind}_y.npy')
    wl_path = os.path.join(model_dir, f'{kind}_wl.npy')
    if not (os.path.exists(x_path) and os.path.exists(y_path)
            and os.path.exists(wl_path)):
        _NN_WD_CACHE[cache_key] = {}
        return {}

    labels = np.load(x_path)
    flux_grid = np.load(y_path)
    wavelength, wl_unit = _coerce_wavelength_angstrom(np.load(wl_path))

    labels = np.asarray(labels, dtype=np.float64)
    flux_grid = np.asarray(flux_grid, dtype=np.float64)
    if labels.ndim != 2 or labels.shape[1] < 2:
        _NN_WD_CACHE[cache_key] = {}
        return {}
    if flux_grid.ndim != 2 or flux_grid.shape[0] != labels.shape[0]:
        _NN_WD_CACHE[cache_key] = {}
        return {}
    if flux_grid.shape[1] != wavelength.size:
        _NN_WD_CACHE[cache_key] = {}
        return {}

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    templates = {}
    for row, flux in zip(labels, flux_grid):
        teff = float(row[0])
        logg = float(row[1])
        fl = np.asarray(flux, dtype=np.float64)[order]
        good = np.isfinite(wavelength) & np.isfinite(fl)
        if np.sum(good) < 100:
            continue
        templates[(teff, logg)] = {
            'wavelength': wavelength[good],
            'flux': fl[good],
            'spectral_type': kind,
            'model_source': f'NN_{kind}',
            'model_dir': model_dir,
            'wavelength_unit': wl_unit,
            'flux_unit': 'surface_f_lambda_cgs_like',
        }

    _NN_WD_CACHE[cache_key] = templates
    return templates


def _grid_params_from_templates(templates):
    if not templates:
        return [], []
    teffs = sorted(set(float(k[0]) for k in templates))
    loggs = sorted(set(float(k[1]) for k in templates))
    return teffs, loggs


def _resolve_template_grid(model_grid='auto', spectral_type='DA'):
    """
    Resolve the WD template source.

    ``auto`` prefers the local NN DA/DB grids and falls back to Koester2.
    """
    label = str(model_grid or 'auto').lower()
    stype = str(spectral_type or 'DA').upper()
    kind = 'DB' if stype.startswith('DB') else 'DA'

    if label in ('auto', 'nn', 'nn_da', 'nn_db', 'neural', 'neural_network'):
        nn_kind = 'DB' if label == 'nn_db' else ('DA' if label == 'nn_da' else kind)
        templates = load_nn_wd_templates(nn_kind)
        if templates:
            teffs, loggs = _grid_params_from_templates(templates)
            return templates, teffs, loggs, f'NN_{nn_kind}'
        if label not in ('auto', 'nn', 'neural', 'neural_network'):
            return {}, [], [], f'NN_{nn_kind}'

    templates = _load_koester2()
    teffs, loggs = _grid_params_from_templates(templates)
    return templates, teffs, loggs, 'Koester2'


def _get_model_grid_params(templates=None, model_grid='auto',
                           spectral_type='DA'):
    """返回模版覆盖的 (teff_list, logg_list) 唯一值排序列表"""
    if templates is None:
        templates, _, _, _ = _resolve_template_grid(model_grid, spectral_type)
    return _grid_params_from_templates(templates)


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


def _infer_observed_flux_unit(flux):
    """
    Infer the multiplier that converts stored spectral flux to cgs f_lambda.

    SDSS/DESI/LAMOST style spectra commonly store flux in units of
    1e-17 erg s-1 cm-2 A-1, while HST and synthetic products may already be
    cgs.  This is only used for physical radius validation, never for the
    chi-squared shape fit.
    """
    arr = np.asarray(flux, dtype=float)
    arr = np.abs(arr[np.isfinite(arr)])
    arr = arr[arr > 0]
    if arr.size == 0:
        return 1.0, 'unknown'
    med = float(np.nanmedian(arr))
    if med > 1e-8:
        return 1.0e-17, 'survey_1e-17_flam'
    return 1.0, 'cgs_flam'


def _distance_scale_radius(scale, parallax_mas, observed_flux_unit=1.0):
    """Convert a fitted surface-flux scale into an implied radius."""
    try:
        scale = float(scale)
        parallax_mas = float(parallax_mas)
        observed_flux_unit = float(observed_flux_unit)
    except (TypeError, ValueError):
        return None
    if (not np.isfinite(scale) or scale <= 0
            or not np.isfinite(parallax_mas) or parallax_mas <= 0
            or not np.isfinite(observed_flux_unit) or observed_flux_unit <= 0):
        return None
    dist_pc = 1000.0 / parallax_mas
    dist_cm = dist_pc * 3.085677581491367e18
    physical_scale = scale * observed_flux_unit
    if physical_scale <= 0:
        return None
    radius_rsun = dist_cm * np.sqrt(physical_scale) / R_SUN_CM
    return {
        'distance_pc': dist_pc,
        'physical_scale': physical_scale,
        'radius_rsun': radius_rsun,
    }


def attach_distance_scale_check(fit_result, parallax_mas, observed_flux,
                                expected_radius_rsun=None):
    """
    Attach a Gaia-distance sanity check to a WD spectral fit result.

    For NN/Koester surface-flux templates, the fitted scale should be roughly
    ``(R/d)^2`` after converting stored survey fluxes to cgs units.
    """
    if not fit_result or parallax_mas is None:
        return fit_result
    flux_unit, flux_unit_label = _infer_observed_flux_unit(observed_flux)
    radius = _distance_scale_radius(
        fit_result.get('scale'), parallax_mas,
        observed_flux_unit=flux_unit)
    if radius is None:
        return fit_result
    fit_result['scale_observed_flux_unit'] = flux_unit
    fit_result['scale_observed_flux_unit_label'] = flux_unit_label
    fit_result['scale_physical_factor'] = radius['physical_scale']
    fit_result['scale_distance_pc'] = radius['distance_pc']
    fit_result['scale_radius_rsun'] = radius['radius_rsun']
    note = 'scale converted with stored flux unit before R=d*sqrt(scale)'
    if expected_radius_rsun is not None and np.isfinite(expected_radius_rsun):
        ratio = radius['radius_rsun'] / expected_radius_rsun
        fit_result['scale_radius_ratio_to_expected'] = ratio
        fit_result['scale_radius_ok'] = bool(0.35 <= ratio <= 2.8)
        note += f'; ratio_to_expected={ratio:.3f}'
    else:
        fit_result['scale_radius_ok'] = bool(0.003 <= radius['radius_rsun'] <= 0.05)
    fit_result['scale_unit_note'] = note
    return fit_result


def _normalize_line_segment(wave, flux, err, center, half_width=80.0):
    """Extract one Balmer line and divide by a local linear continuum."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float) if err is not None else None
    min_points = 8 if err is None else 20
    mask = ((wave >= center - half_width) & (wave <= center + half_width)
            & np.isfinite(wave) & np.isfinite(flux))
    if err is not None and len(err) == len(wave):
        mask &= np.isfinite(err) & (err > 0)
    if np.sum(mask) < min_points:
        return None

    x = wave[mask] - center
    y = flux[mask]
    e = err[mask] if err is not None and len(err) == len(wave) else None
    side = np.abs(x) >= 0.55 * half_width
    if np.sum(side) >= max(4, min_points // 2):
        try:
            if e is not None:
                wt = 1.0 / np.maximum(e[side], np.nanmedian(e[side]))**2
                coeff = np.polyfit(x[side], y[side], 1, w=np.sqrt(wt))
            else:
                coeff = np.polyfit(x[side], y[side], 1)
            cont = np.polyval(coeff, x)
        except Exception:
            cont = np.full_like(y, np.nanmedian(y[side]))
    else:
        cont = np.full_like(y, np.nanmedian(y))
    fallback = np.nanmedian(y[np.isfinite(y)])
    cont = np.where(np.isfinite(cont) & (np.abs(cont) > 0), cont, fallback)
    if not np.isfinite(fallback) or fallback == 0:
        return None

    norm = y / cont
    norm_err = None
    if e is not None:
        norm_err = np.abs(e / cont)
    good = np.isfinite(x) & np.isfinite(norm)
    if norm_err is not None:
        good &= np.isfinite(norm_err) & (norm_err > 0)
    if np.sum(good) < min_points:
        return None
    return x[good], norm[good], norm_err[good] if norm_err is not None else None


def fit_balmer_line_profiles(wave, flux, err=None, rv_grid=None,
                             half_width=50.0, lines=None,
                             teff_prior=None, teff_prior_sigma=3000.0,
                             model_grid='auto', spectral_type='DA'):
    """
    Fit continuum-normalized Balmer profiles H-beta through H8.

    This mirrors the usual WD atmospheric fitting workflow better than a
    full-spectrum flux fit: each Balmer line is locally continuum-normalized,
    the model grid is normalized the same way, and a coarse photospheric
    velocity grid is searched so line centers do not bias log g.
    """
    templates, teffs, loggs, model_source = _resolve_template_grid(
        model_grid, spectral_type)
    if not templates:
        return None
    if not teffs:
        return None

    lines = lines or BALMER_PROFILE_LINES
    if rv_grid is None:
        rv_grid = np.arange(-350.0, 351.0, 25.0)

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float) if err is not None else None
    finite = np.isfinite(wave) & np.isfinite(flux)
    if err is not None and len(err) == len(wave):
        finite &= np.isfinite(err) & (err > 0)
    wave = wave[finite]
    flux = flux[finite]
    err = err[finite] if err is not None and len(err) == len(finite) else None
    if len(wave) < 100:
        return None

    chi2_grid = {}
    best = None
    best_score = np.inf
    best_chi2 = np.inf
    best_scale = 0.0
    best_model_wave = None
    best_model_flux = None

    # Pre-normalize the model profiles once per template/line.
    model_profiles = {}
    for key, tmpl in templates.items():
        per_line = {}
        for line_name, center in lines.items():
            prof = _normalize_line_segment(
                tmpl['wavelength'], tmpl['flux'], None, center,
                half_width=half_width)
            if prof is not None:
                per_line[line_name] = prof[:2]
        if per_line:
            model_profiles[key] = per_line

    for rv in rv_grid:
        rest_wave = wave / (1.0 + rv / C_KMS)
        obs_profiles = {}
        for line_name, center in lines.items():
            prof = _normalize_line_segment(
                rest_wave, flux, err, center, half_width=half_width)
            if prof is not None:
                obs_profiles[line_name] = prof
        if len(obs_profiles) < 3:
            continue

        for key, per_line in model_profiles.items():
            chi2 = 0.0
            n_pix = 0
            used = []
            for line_name, (x_obs, f_obs, e_obs) in obs_profiles.items():
                if line_name not in per_line:
                    continue
                x_mod, f_mod = per_line[line_name]
                model_at_obs = np.interp(x_obs, x_mod, f_mod,
                                         left=np.nan, right=np.nan)
                good = np.isfinite(model_at_obs) & np.isfinite(f_obs)
                if np.sum(good) < 15:
                    continue
                if e_obs is not None:
                    e = np.asarray(e_obs, dtype=float)
                    e = np.where(np.isfinite(e) & (e > 0), e, np.nanmedian(e[good]))
                else:
                    e = np.full_like(f_obs, 0.03)
                e = np.maximum(e, 0.02)
                resid = f_obs[good] - model_at_obs[good]
                chi2 += float(np.sum((resid / e[good]) ** 2))
                n_pix += int(np.sum(good))
                used.append(line_name)
            if n_pix < 80 or len(set(used)) < 3:
                continue
            chi2_red = chi2 / max(n_pix - 3, 1)
            prev = chi2_grid.get(key, np.inf)
            if chi2_red < prev:
                chi2_grid[key] = chi2_red
            score = chi2_red
            if teff_prior is not None and teff_prior_sigma:
                try:
                    sigma = max(float(teff_prior_sigma), 500.0)
                    score += ((float(key[0]) - float(teff_prior)) / sigma) ** 2
                except (TypeError, ValueError):
                    pass
            if score < best_score:
                tmpl = templates[key]
                # Give the plotting/reporting code a scaled full model too.
                chi2_full, scale, _ = _chi2_single(
                    wave, flux, err, tmpl['wavelength'], tmpl['flux'])
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                best_score = score
                best_chi2 = chi2_red
                best = (key, rv, sorted(set(used)), n_pix)
                best_scale = scale
                best_model_wave = tmpl['wavelength'] * (1.0 + rv / C_KMS)
                best_model_flux = tmpl['flux'] * scale

    if best is None:
        return None

    (t_best, g_best), rv_best, lines_used, n_pix = best
    teff_err = _estimate_1d_error(chi2_grid, teffs, loggs, axis='teff',
                                  best_t=t_best, best_g=g_best)
    logg_err = _estimate_1d_error(chi2_grid, teffs, loggs, axis='logg',
                                  best_t=t_best, best_g=g_best)
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
        'rv_kms': float(rv_best),
        'n_profile_pixels': int(n_pix),
        'lines_used': lines_used,
        'method': 'Balmer_profile_Hbeta_to_H8',
        'model_grid': model_source,
        'spectral_type': spectral_type,
        'teff_grid': teffs,
        'logg_grid': loggs,
        'fit_score': best_score,
        'teff_prior': teff_prior,
        'teff_prior_sigma': teff_prior_sigma if teff_prior is not None else None,
    }


def fit_single_wd(wave, flux, err=None, line_only=False,
                  teff_prior=None, teff_prior_sigma=3000.0,
                  model_grid='auto', spectral_type='DA'):
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
    templates, teffs, loggs, model_source = _resolve_template_grid(
        model_grid, spectral_type)
    if not templates:
        return None

    if not teffs:
        return None

    if line_only:
        return fit_balmer_line_profiles(
            wave, flux, err, teff_prior=teff_prior,
            teff_prior_sigma=teff_prior_sigma,
            model_grid=model_grid, spectral_type=spectral_type)

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
        'model_grid': model_source,
        'spectral_type': spectral_type,
        'teff_grid': teffs,
        'logg_grid': loggs,
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

def fit_sed(photometry, parallax_mas, model_grid='auto',
            spectral_type='DA'):
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

    templates, _, _, model_source = _resolve_template_grid(
        model_grid, spectral_type)
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

    def _model_flux_at_bands(tw, tf, waves):
        """Interpolate inside the grid and use an RJ tail for redder bands."""
        tw = np.asarray(tw, dtype=np.float64)
        tf = np.asarray(tf, dtype=np.float64)
        waves = np.asarray(waves, dtype=np.float64)
        good = np.isfinite(tw) & np.isfinite(tf) & (tf > 0)
        if np.sum(good) < 5:
            return (np.full_like(waves, np.nan, dtype=np.float64),
                    np.zeros_like(waves, dtype=bool),
                    np.zeros_like(waves, dtype=bool))

        tw = tw[good]
        tf = tf[good]
        order = np.argsort(tw)
        tw = tw[order]
        tf = tf[order]

        model = np.full_like(waves, np.nan, dtype=np.float64)
        in_grid = (waves >= tw[0]) & (waves <= tw[-1])
        if np.any(in_grid):
            model[in_grid] = np.interp(waves[in_grid], tw, tf)

        red_tail = waves > tw[-1]
        if np.any(red_tail):
            tail_mask = tw >= max(tw[-1] * 0.75, tw[-1] - 5000.0)
            if np.sum(tail_mask) < 5:
                tail_mask = np.arange(len(tw)) >= max(len(tw) - 50, 0)
            tail_norm = np.nanmedian(tf[tail_mask] * tw[tail_mask] ** 4)
            if np.isfinite(tail_norm) and tail_norm > 0:
                model[red_tail] = tail_norm / waves[red_tail] ** 4

        return model, in_grid, red_tail

    # 遍历模版网格
    best_chi2 = np.inf
    best_params = None
    best_scale = 0.0
    best_chi2_all = np.nan
    best_model_fluxes = None
    best_fit_mask = None
    best_in_grid = None
    best_red_tail = None

    for (t, g), tmpl in templates.items():
        tw = tmpl['wavelength']
        tf = tmpl['flux']  # Eddington flux at stellar surface (erg/s/cm^2/A)

        model_fluxes, in_grid, red_tail = _model_flux_at_bands(tw, tf, obs_waves)

        fit_mask = in_grid & np.isfinite(model_fluxes) & (model_fluxes > 0)
        if np.sum(fit_mask) < 3:
            # Fallback for sparse SEDs: allow the RJ tail, but never a zero
            # fill-value outside the template grid.
            fit_mask = np.isfinite(model_fluxes) & (model_fluxes > 0)

        if np.sum(fit_mask) < 3:
            continue

        # 缩放因子 = (R/d)^2, R=stellar radius
        # scale = sum(w * obs * model) / sum(w * model^2)
        w = 1.0 / obs_errs[fit_mask]**2
        denom = np.sum(w * model_fluxes[fit_mask]**2)
        if denom <= 0:
            continue
        scale = np.sum(w * obs_fluxes[fit_mask] * model_fluxes[fit_mask]) / denom

        residual = obs_fluxes[fit_mask] - scale * model_fluxes[fit_mask]
        chi2 = np.sum(w * residual**2)
        ndof = max(np.sum(fit_mask) - 2, 1)
        chi2_red = chi2 / ndof
        all_mask = np.isfinite(model_fluxes) & (model_fluxes > 0)
        if np.any(all_mask):
            all_resid = obs_fluxes[all_mask] - scale * model_fluxes[all_mask]
            all_chi2 = np.sum((all_resid / obs_errs[all_mask]) ** 2)
            chi2_all = all_chi2 / max(np.sum(all_mask) - 2, 1)
        else:
            chi2_all = np.nan

        if chi2_red < best_chi2:
            best_chi2 = chi2_red
            best_params = (t, g)
            best_scale = scale
            best_chi2_all = chi2_all
            best_model_fluxes = model_fluxes
            best_fit_mask = fit_mask
            best_in_grid = in_grid
            best_red_tail = red_tail

    if best_params is None:
        return None

    # scale = (R / dist)^2   →   R = dist * sqrt(scale)
    R_cm = dist_cm * np.sqrt(max(best_scale, 0))
    R_Rsun = R_cm / R_SUN_CM
    angular_radius = np.sqrt(max(best_scale, 0))  # radians

    # 合成测光 (mag) + IR excess diagnostics.
    best_tmpl = templates[best_params]
    syn_mags = {}
    band_residuals = {}
    ir_excess_bands = []
    max_ir_excess_dex = np.nan
    max_ir_excess_sigma = np.nan

    for i, d in enumerate(obs_data):
        if best_model_fluxes is None:
            continue
        fm = best_model_fluxes[i] * best_scale
        if fm > 0:
            # flux → mag (AB): m = -2.5 * log10(f_lambda * wave^2 / c) - 48.6
            info = config.BAND_INFO.get(d['band'], {})
            zero_jy = info.get('zero_Jy', 3631.0)
            # reverse of mag_to_flux_cgs
            c_A = 2.99792458e18
            f_hz = fm * d['wave']**2 / c_A
            f_jy = f_hz / 1e-23
            syn_mags[d['band']] = -2.5 * np.log10(f_jy / zero_jy)
            residual_dex = np.log10(d['flux'] / fm) if d['flux'] > 0 else np.nan
            residual_sigma = (d['flux'] - fm) / d['err'] if d['err'] > 0 else np.nan
            band_residuals[d['band']] = {
                'wave_A': d['wave'],
                'observed_flux': d['flux'],
                'model_flux': fm,
                'residual_dex': float(residual_dex),
                'residual_sigma': float(residual_sigma),
                'in_template_grid': bool(best_in_grid[i]) if best_in_grid is not None else False,
                'red_tail_model': bool(best_red_tail[i]) if best_red_tail is not None else False,
            }
            is_ir = d['wave'] >= 25000.0 or d['band'].upper().startswith('WISE')
            if is_ir and np.isfinite(residual_dex) and np.isfinite(residual_sigma):
                if not np.isfinite(max_ir_excess_dex):
                    max_ir_excess_dex = residual_dex
                    max_ir_excess_sigma = residual_sigma
                elif residual_dex > max_ir_excess_dex:
                    max_ir_excess_dex = residual_dex
                    max_ir_excess_sigma = residual_sigma
                if residual_dex >= 0.30 and residual_sigma >= 3.0:
                    ir_excess_bands.append(d['band'])

    tail_wave = np.array([])
    tail_flux = np.array([])
    if np.nanmax(obs_waves) > best_tmpl['wavelength'][-1]:
        tail_wave = np.geomspace(best_tmpl['wavelength'][-1],
                                 np.nanmax(obs_waves) * 1.05, 120)
        tail_model, _, tail_red = _model_flux_at_bands(
            best_tmpl['wavelength'], best_tmpl['flux'], tail_wave)
        ok = tail_red & np.isfinite(tail_model) & (tail_model > 0)
        tail_wave = tail_wave[ok]
        tail_flux = tail_model[ok] * best_scale

    return {
        'teff_sed': best_params[0],
        'logg_sed': best_params[1],
        'chi2_sed': best_chi2,
        'chi2_sed_photospheric': best_chi2,
        'chi2_sed_all': best_chi2_all,
        'angular_radius_rad': angular_radius,
        'R_Rsun': R_Rsun,
        'scale': best_scale,
        'synthetic_mags': syn_mags,
        'sed_fit_bands': [obs_data[i]['band'] for i in range(len(obs_data))
                          if best_fit_mask is not None and best_fit_mask[i]],
        'sed_red_tail_bands': [obs_data[i]['band'] for i in range(len(obs_data))
                               if best_red_tail is not None and best_red_tail[i]],
        'band_residuals': band_residuals,
        'ir_excess_flag': bool(ir_excess_bands),
        'ir_excess_bands': ir_excess_bands,
        'max_ir_excess_dex': float(max_ir_excess_dex)
        if np.isfinite(max_ir_excess_dex) else np.nan,
        'max_ir_excess_sigma': float(max_ir_excess_sigma)
        if np.isfinite(max_ir_excess_sigma) else np.nan,
        'model_grid': model_source,
        'spectral_type': spectral_type,
        'best_model_wave': best_tmpl['wavelength'],
        'best_model_flux': best_tmpl['flux'] * best_scale,
        'best_model_tail_wave': tail_wave,
        'best_model_tail_flux': tail_flux,
    }


# ==================================================================
#  NN 模板 MCMC 拟合
# ==================================================================

def _nn_template_arrays(specclass='DA'):
    templates = load_nn_wd_templates(specclass)
    if not templates:
        return None
    keys = sorted(templates)
    wave = templates[keys[0]]['wavelength']
    labels = np.array(keys, dtype=float)
    flux = np.vstack([templates[k]['flux'] for k in keys])
    return wave, labels, flux, templates[keys[0]].get('model_source', 'NN')


def _template_sampler_arrays(specclass='DA', model_grid='auto'):
    """Return a common-wavelength template array for posterior sampling."""
    templates, _, _, model_source = _resolve_template_grid(
        model_grid=model_grid, spectral_type=specclass)
    if not templates:
        return None
    keys = sorted(templates)
    base_wave = np.asarray(templates[keys[0]]['wavelength'], dtype=float)
    order = np.argsort(base_wave)
    base_wave = base_wave[order]
    labels = []
    spectra = []
    for key in keys:
        wave_i = np.asarray(templates[key]['wavelength'], dtype=float)
        flux_i = np.asarray(templates[key]['flux'], dtype=float)
        good = np.isfinite(wave_i) & np.isfinite(flux_i)
        if np.sum(good) < 100:
            continue
        sort_i = np.argsort(wave_i[good])
        wave_i = wave_i[good][sort_i]
        flux_i = flux_i[good][sort_i]
        if len(wave_i) == len(base_wave) and np.allclose(wave_i, base_wave):
            flux_common = flux_i
        else:
            flux_common = np.interp(base_wave, wave_i, flux_i,
                                    left=np.nan, right=np.nan)
        if np.sum(np.isfinite(flux_common)) < 100:
            continue
        labels.append(key)
        spectra.append(np.nan_to_num(flux_common, nan=0.0, posinf=0.0,
                                     neginf=0.0))
    if not spectra:
        return None
    return (base_wave, np.asarray(labels, dtype=float),
            np.vstack(spectra), model_source)


def _weighted_grid_spectrum(labels, flux_grid, teff, logg, k=8):
    """Fast local inverse-distance interpolation over the NN output grid."""
    labels = np.asarray(labels, dtype=float)
    flux_grid = np.asarray(flux_grid, dtype=float)
    t_span = max(np.nanmax(labels[:, 0]) - np.nanmin(labels[:, 0]), 1.0)
    g_span = max(np.nanmax(labels[:, 1]) - np.nanmin(labels[:, 1]), 0.1)
    dt = (labels[:, 0] - teff) / max(t_span / 20.0, 250.0)
    dg = (labels[:, 1] - logg) / max(g_span / 8.0, 0.125)
    dist2 = dt * dt + dg * dg
    i0 = int(np.nanargmin(dist2))
    if dist2[i0] < 1e-12:
        return flux_grid[i0]
    k = min(k, len(labels))
    idx = np.argpartition(dist2, k - 1)[:k]
    weights = 1.0 / np.maximum(dist2[idx], 1e-12)
    weights /= np.sum(weights)
    return np.sum(flux_grid[idx] * weights[:, None], axis=0)


def _vgrav_from_mass_radius(mass_msun, radius_rsun):
    mass_msun = np.asarray(mass_msun, dtype=float)
    radius_rsun = np.asarray(radius_rsun, dtype=float)
    out = np.full(np.broadcast_shapes(mass_msun.shape, radius_rsun.shape),
                  np.nan, dtype=float)
    mass_b = np.broadcast_to(mass_msun, out.shape)
    radius_b = np.broadcast_to(radius_rsun, out.shape)
    good = np.isfinite(mass_b) & np.isfinite(radius_b) & (mass_b > 0) & (radius_b > 0)
    out[good] = (
        G_CGS * mass_b[good] * M_SUN_G
        / (radius_b[good] * R_SUN_CM)
        / (C_KMS * 1.0e5)
        / 1.0e5
    )
    return out


def fit_wd_mcmc_nn(wave, flux, err=None, specclass='DA', initial=None,
                   parallax_mas=None, teff_prior=None,
                   teff_prior_sigma=None, nwalkers=32, nsteps=800,
                   burn=200, thin=5, random_seed=42,
                   output_dir=None, max_pixels=900,
                   model_grid='auto', sampler='auto', n_importance=None,
                   parallax_err_mas=None,
                   gaia_teff_prior=None, gaia_teff_prior_sigma=None,
                   gaia_logg_prior=None, gaia_logg_prior_sigma=None,
                   gaia_mass_prior=None, gaia_mass_prior_sigma=None,
                   gaia_radius_prior=None, gaia_radius_prior_sigma=None):
    """
    MCMC fit using the local neural-network WD spectral grid.

    Parameters
    ----------
    wave, flux, err : array
        Observed spectrum.  Wavelength must be Angstrom.  SDSS/DESI style flux
        stored in 1e-17 cgs units is detected for the radius sanity check.
    specclass : {'DA', 'DB'}
        Which local NN grid to use.
    parallax_mas : float, optional
        If supplied, the sampler adds a weak radius prior using
        ``R = d sqrt(scale)`` and the logg mass-radius relation.

    Returns
    -------
    dict
        Posterior medians/errors and paths to generated plots/tables.
    """
    try:
        import emcee
    except Exception:
        emcee = None

    arrays = _template_sampler_arrays(specclass, model_grid=model_grid)
    if arrays is None:
        return {
            'status': 'skipped',
            'error': f'{model_grid} {specclass} templates not found',
            'model_grid': str(model_grid),
        }
    model_wave, labels, flux_grid, model_source = arrays
    t_min, t_max = float(np.min(labels[:, 0])), float(np.max(labels[:, 0]))
    g_min, g_max = float(np.min(labels[:, 1])), float(np.max(labels[:, 1]))

    w, f, e, _ = _prepare_spectrum(wave, flux, err, w_min=3700, w_max=9200)
    if w is None:
        return {'status': 'failed', 'error': 'not enough optical spectrum'}
    order = np.argsort(w)
    w, f = w[order], f[order]
    if e is None:
        e = np.full_like(f, np.nan)
    else:
        e = e[order]
    if len(w) > max_pixels:
        idx = np.linspace(0, len(w) - 1, int(max_pixels)).astype(int)
        w, f, e = w[idx], f[idx], e[idx]
    med_flux = np.nanmedian(np.abs(f[np.isfinite(f)]))
    if not np.isfinite(med_flux) or med_flux <= 0:
        med_flux = 1.0
    e = np.asarray(e, dtype=float)
    e_floor = np.maximum(0.03 * np.abs(f), 0.02 * med_flux)
    e = np.where(np.isfinite(e) & (e > 0), e, e_floor)
    e = np.maximum(e, e_floor)

    if initial is None:
        cont = fit_single_wd(
            wave, flux, err, line_only=False,
            model_grid=f'nn_{str(specclass).lower()}',
            spectral_type=specclass)
        balmer = fit_balmer_line_profiles(
            wave, flux, err, teff_prior=(cont or {}).get('teff'),
            model_grid=f'nn_{str(specclass).lower()}',
            spectral_type=specclass)
        initial = balmer or cont or {}
        if cont and 'scale' in cont:
            initial = dict(initial)
            initial.setdefault('scale', cont.get('scale'))

    t0 = float(initial.get('teff', np.nanmedian(labels[:, 0])))
    g0 = float(initial.get('logg', np.nanmedian(labels[:, 1])))
    rv0 = float(initial.get('rv_kms', 0.0) or 0.0)
    scale0 = float(initial.get('scale', 0.0) or 0.0)
    if not np.isfinite(scale0) or scale0 <= 0:
        guess_surface = _weighted_grid_spectrum(labels, flux_grid, t0, g0)
        guess_model = np.interp(w, model_wave, guess_surface, left=0, right=0)
        denom = np.sum((guess_model / e) ** 2)
        scale0 = np.sum(f * guess_model / e**2) / denom if denom > 0 else 1e-20
    scale0 = max(scale0, 1e-40)

    t0 = float(np.clip(t0, t_min + 1.0, t_max - 1.0))
    g0 = float(np.clip(g0, g_min + 0.01, g_max - 0.01))
    p0_center = np.array([t0, g0, rv0, np.log(scale0)], dtype=float)
    rng = np.random.default_rng(random_seed)
    nwalkers = max(int(nwalkers), 2 * len(p0_center) + 2)
    p0 = np.repeat(p0_center[None, :], nwalkers, axis=0)
    p0[:, 0] += rng.normal(0.0, 250.0, nwalkers)
    p0[:, 1] += rng.normal(0.0, 0.06, nwalkers)
    p0[:, 2] += rng.normal(0.0, 20.0, nwalkers)
    p0[:, 3] += rng.normal(0.0, 0.15, nwalkers)
    p0[:, 0] = np.clip(p0[:, 0], t_min + 1.0, t_max - 1.0)
    p0[:, 1] = np.clip(p0[:, 1], g_min + 0.01, g_max - 0.01)

    flux_unit, flux_unit_label = _infer_observed_flux_unit(f)
    parallax_val = None
    try:
        parallax_val = float(parallax_mas)
    except (TypeError, ValueError):
        parallax_val = None
    parallax_err_val = None
    try:
        parallax_err_val = float(parallax_err_mas)
    except (TypeError, ValueError):
        parallax_err_val = None

    def _finite_prior(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    gaia_teff_prior = _finite_prior(gaia_teff_prior)
    gaia_logg_prior = _finite_prior(gaia_logg_prior)
    gaia_mass_prior = _finite_prior(gaia_mass_prior)
    gaia_radius_prior = _finite_prior(gaia_radius_prior)

    def _best_ln_scale(teff, logg, rv):
        surface = _weighted_grid_spectrum(labels, flux_grid, teff, logg)
        rest_wave = w / (1.0 + rv / C_KMS)
        model = np.interp(rest_wave, model_wave, surface, left=0.0, right=0.0)
        denom = np.sum((model / e) ** 2)
        scale = np.sum(f * model / e**2) / denom if denom > 0 else scale0
        if not np.isfinite(scale) or scale <= 0:
            scale = scale0
        return float(np.log(max(scale, 1.0e-60)))

    def _log_prob(theta):
        teff, logg, rv, ln_scale = theta
        if not (t_min <= teff <= t_max and g_min <= logg <= g_max
                and -650.0 <= rv <= 650.0 and -120.0 <= ln_scale <= 60.0):
            return -np.inf
        surface = _weighted_grid_spectrum(labels, flux_grid, teff, logg)
        rest_wave = w / (1.0 + rv / C_KMS)
        model = np.interp(rest_wave, model_wave, surface, left=0.0, right=0.0)
        model = model * np.exp(ln_scale)
        good = np.isfinite(model) & (model > 0)
        if np.sum(good) < max(50, len(w) // 3):
            return -np.inf
        resid = f[good] - model[good]
        logp = -0.5 * np.sum((resid / e[good]) ** 2 + np.log(2.0 * np.pi * e[good] ** 2))
        if teff_prior is not None and teff_prior_sigma:
            sig = max(float(teff_prior_sigma), 100.0)
            logp += -0.5 * ((teff - float(teff_prior)) / sig) ** 2
        if gaia_teff_prior is not None:
            sig = max(float(gaia_teff_prior_sigma or 3000.0), 500.0)
            logp += -0.5 * ((teff - gaia_teff_prior) / sig) ** 2
        if gaia_logg_prior is not None:
            sig = max(float(gaia_logg_prior_sigma or 0.25), 0.08)
            logp += -0.5 * ((logg - gaia_logg_prior) / sig) ** 2
        mass_theta = _logg_to_mass(logg)
        radius_theta = compute_wd_radius(mass_theta, logg)
        if gaia_mass_prior is not None:
            sig = max(float(gaia_mass_prior_sigma or 0.12), 0.04)
            logp += -0.5 * ((mass_theta - gaia_mass_prior) / sig) ** 2
        if gaia_radius_prior is not None:
            sig = max(float(gaia_radius_prior_sigma or 0.003), 0.0008)
            logp += -0.5 * ((radius_theta - gaia_radius_prior) / sig) ** 2
        if parallax_val is not None and np.isfinite(parallax_val) and parallax_val > 0:
            r_info = _distance_scale_radius(
                np.exp(ln_scale), parallax_val,
                observed_flux_unit=flux_unit)
            if r_info is not None:
                try:
                    mr_radius = radius_theta
                    par_frac = (
                        abs(parallax_err_val / parallax_val)
                        if (parallax_err_val is not None
                            and np.isfinite(parallax_err_val)
                            and parallax_err_val > 0) else 0.0
                    )
                    sigma_r = np.hypot(max(0.18 * mr_radius, 0.0012),
                                       par_frac * r_info['radius_rsun'])
                    logp += -0.5 * ((r_info['radius_rsun'] - mr_radius) / sigma_r) ** 2
                except Exception:
                    pass
        return float(logp)

    use_emcee = emcee is not None and str(sampler).lower() not in {
        'importance', 'importance_sampling', 'fallback'
    }
    sampler_backend = 'emcee' if use_emcee else 'importance_sampling_fallback'
    mcmc_acceptance_fraction = np.nan
    mcmc_acceptance_fraction_min = np.nan
    mcmc_acceptance_fraction_max = np.nan
    mcmc_autocorr_time_max = np.nan
    mcmc_converged = False
    mcmc_warning = ''
    if use_emcee:
        np.random.seed(int(random_seed))
        sampler_obj = emcee.EnsembleSampler(nwalkers, 4, _log_prob)
        sampler_obj.run_mcmc(p0, int(nsteps), progress=False)
        try:
            acc = np.asarray(sampler_obj.acceptance_fraction, dtype=float)
            acc = acc[np.isfinite(acc)]
            if acc.size:
                mcmc_acceptance_fraction = float(np.nanmedian(acc))
                mcmc_acceptance_fraction_min = float(np.nanmin(acc))
                mcmc_acceptance_fraction_max = float(np.nanmax(acc))
                if mcmc_acceptance_fraction < 0.15 or mcmc_acceptance_fraction > 0.70:
                    mcmc_warning = 'acceptance_fraction_outside_nominal_range'
        except Exception:
            pass
        try:
            tau = np.asarray(sampler_obj.get_autocorr_time(tol=0), dtype=float)
            tau = tau[np.isfinite(tau) & (tau > 0)]
            if tau.size:
                mcmc_autocorr_time_max = float(np.nanmax(tau))
                mcmc_converged = bool(int(nsteps) >= 50.0 * mcmc_autocorr_time_max)
                if not mcmc_converged:
                    extra = 'chain_shorter_than_50_autocorr_times'
                    mcmc_warning = f'{mcmc_warning};{extra}' if mcmc_warning else extra
        except Exception as exc:
            mcmc_warning = (
                f'{mcmc_warning};autocorr_unavailable:{exc}'
                if mcmc_warning else f'autocorr_unavailable:{exc}'
            )
        burn = min(max(int(burn), 0), max(int(nsteps) - 1, 0))
        thin = max(int(thin), 1)
        flat = sampler_obj.get_chain(discard=burn, thin=thin, flat=True)
        logp = sampler_obj.get_log_prob(discard=burn, thin=thin, flat=True)
        if flat.size == 0:
            flat = sampler_obj.get_chain(flat=True)
            logp = sampler_obj.get_log_prob(flat=True)
    else:
        n_prop = int(n_importance or min(max(int(nwalkers) * int(nsteps) // 2, 2500), 6000))
        t_center = gaia_teff_prior if gaia_teff_prior is not None else t0
        g_center = gaia_logg_prior if gaia_logg_prior is not None else g0
        t_sigma = max(float(gaia_teff_prior_sigma or teff_prior_sigma or 3000.0), 900.0)
        g_sigma = max(float(gaia_logg_prior_sigma or 0.28), 0.12)
        candidates = np.empty((n_prop, 4), dtype=float)
        broad = rng.random(n_prop) < 0.25
        candidates[:, 0] = rng.normal(t_center, t_sigma, n_prop)
        candidates[:, 1] = rng.normal(g_center, g_sigma, n_prop)
        candidates[:, 2] = rng.normal(rv0, 90.0, n_prop)
        candidates[broad, 0] = rng.uniform(t_min, t_max, int(np.sum(broad)))
        candidates[broad, 1] = rng.uniform(g_min, g_max, int(np.sum(broad)))
        candidates[broad, 2] = rng.uniform(-350.0, 350.0, int(np.sum(broad)))
        candidates[:, 0] = np.clip(candidates[:, 0], t_min + 1.0, t_max - 1.0)
        candidates[:, 1] = np.clip(candidates[:, 1], g_min + 0.01, g_max - 0.01)
        for i in range(n_prop):
            candidates[i, 3] = (
                _best_ln_scale(candidates[i, 0], candidates[i, 1], candidates[i, 2])
                + rng.normal(0.0, 0.10)
            )
        logp_all = np.array([_log_prob(theta) for theta in candidates], dtype=float)
        finite = np.isfinite(logp_all)
        if not np.any(finite):
            candidates = p0.copy()
            logp_all = np.array([_log_prob(theta) for theta in candidates], dtype=float)
            finite = np.isfinite(logp_all)
        cand_f = candidates[finite]
        logp_f = logp_all[finite]
        if len(cand_f) == 0:
            cand_f = p0.copy()
            logp_f = np.zeros(len(cand_f), dtype=float)
        # Low/medium-resolution survey spectra have imperfect flux calibration
        # and template systematics.  A literal pixel likelihood can collapse the
        # fallback sampler onto one point, so temper it until the posterior has
        # a useful effective sample size.  emcee users still get the untempered
        # likelihood above.
        span = np.nanpercentile(logp_f, 95) - np.nanpercentile(logp_f, 5)
        temperature = max(1.0, span / 25.0) if np.isfinite(span) else 1.0
        target_ess = min(max(120.0, 0.05 * len(logp_f)), 700.0)
        for _ in range(8):
            weights = np.exp((logp_f - np.nanmax(logp_f)) / temperature)
            sw = np.sum(weights)
            if not np.isfinite(sw) or sw <= 0:
                weights = np.ones_like(logp_f, dtype=float)
                sw = np.sum(weights)
            weights = weights / sw
            ess = 1.0 / np.sum(weights**2)
            if ess >= target_ess:
                break
            temperature *= 1.8
        sampler_temperature = float(temperature)
        sampler_effective_n = float(1.0 / np.sum(weights**2))
        n_post = min(max(1000, len(cand_f)), 10000)
        idx = rng.choice(len(cand_f), size=n_post, replace=True, p=weights)
        flat = cand_f[idx]
        logp = logp_f[idx]
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    best_idx = int(np.nanargmax(logp)) if len(logp) else 0
    best = flat[best_idx]
    med = q50
    med_surface = _weighted_grid_spectrum(labels, flux_grid, med[0], med[1])
    med_model_wave = model_wave * (1.0 + med[2] / C_KMS)
    med_model_flux = med_surface * np.exp(med[3])

    def _finite_percentiles(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        return np.percentile(values, [16, 50, 84])

    scale = float(np.exp(med[3]))
    r_info = _distance_scale_radius(scale, parallax_val, flux_unit)
    scale_samples = np.exp(flat[:, 3])
    mass_samples = np.array([_logg_to_mass(float(g)) for g in flat[:, 1]], dtype=float)
    radius_samples = compute_wd_radius(mass_samples, flat[:, 1])
    mass_q = _finite_percentiles(mass_samples)
    radius_q = _finite_percentiles(radius_samples)
    vgrav_mr_q = _finite_percentiles(
        _vgrav_from_mass_radius(mass_samples, radius_samples))
    mass = float(mass_q[1]) if np.isfinite(mass_q[1]) else _logg_to_mass(float(med[1]))
    radius_mr = (
        float(radius_q[1]) if np.isfinite(radius_q[1])
        else compute_wd_radius(mass, float(med[1]))
    )
    cool_samples = _cooling_from_teff_mass_samples(flat[:, 0], mass_samples)
    cool_q = _finite_percentiles(cool_samples.get('cooling_age_gyr'))
    total_q = _finite_percentiles(cool_samples.get('total_age_gyr'))
    result = {
        'status': 'ok',
        'model_grid': model_source,
        'spectral_type': specclass,
        'teff': float(med[0]),
        'teff_err_minus': float(med[0] - q16[0]),
        'teff_err_plus': float(q84[0] - med[0]),
        'teff_err': float(0.5 * ((med[0] - q16[0]) + (q84[0] - med[0]))),
        'logg': float(med[1]),
        'logg_err_minus': float(med[1] - q16[1]),
        'logg_err_plus': float(q84[1] - med[1]),
        'logg_err': float(0.5 * ((med[1] - q16[1]) + (q84[1] - med[1]))),
        'rv_kms': float(med[2]),
        'rv_err_minus': float(med[2] - q16[2]),
        'rv_err_plus': float(q84[2] - med[2]),
        'rv_err': float(0.5 * ((med[2] - q16[2]) + (q84[2] - med[2]))),
        'scale': scale,
        'scale_err_minus': float(scale - np.exp(q16[3])),
        'scale_err_plus': float(np.exp(q84[3]) - scale),
        'scale_err': float(0.5 * ((scale - np.exp(q16[3])) + (np.exp(q84[3]) - scale))),
        'best_teff': float(best[0]),
        'best_logg': float(best[1]),
        'best_rv_kms': float(best[2]),
        'best_scale': float(np.exp(best[3])),
        'mass_msun_mr': float(mass),
        'mass_msun_mr_err_minus': float(mass_q[1] - mass_q[0]),
        'mass_msun_mr_err_plus': float(mass_q[2] - mass_q[1]),
        'mass_msun_mr_err': float(0.5 * ((mass_q[1] - mass_q[0]) + (mass_q[2] - mass_q[1]))),
        'radius_rsun_mr': float(radius_mr),
        'radius_rsun_mr_err_minus': float(radius_q[1] - radius_q[0]),
        'radius_rsun_mr_err_plus': float(radius_q[2] - radius_q[1]),
        'radius_rsun_mr_err': float(0.5 * ((radius_q[1] - radius_q[0]) + (radius_q[2] - radius_q[1]))),
        'v_grav_mr_kms': float(vgrav_mr_q[1]),
        'v_grav_mr_err_minus': float(vgrav_mr_q[1] - vgrav_mr_q[0]),
        'v_grav_mr_err_plus': float(vgrav_mr_q[2] - vgrav_mr_q[1]),
        'v_grav_mr_err': float(0.5 * ((vgrav_mr_q[1] - vgrav_mr_q[0]) + (vgrav_mr_q[2] - vgrav_mr_q[1]))),
        'mass_msun_preferred': float(mass),
        'mass_msun_preferred_err': float(0.5 * ((mass_q[1] - mass_q[0]) + (mass_q[2] - mass_q[1]))),
        'radius_rsun_preferred': float(radius_mr),
        'radius_rsun_preferred_err': float(0.5 * ((radius_q[1] - radius_q[0]) + (radius_q[2] - radius_q[1]))),
        'v_grav_preferred_kms': float(vgrav_mr_q[1]),
        'v_grav_preferred_err': float(0.5 * ((vgrav_mr_q[1] - vgrav_mr_q[0]) + (vgrav_mr_q[2] - vgrav_mr_q[1]))),
        'preferred_physical_source': 'mcmc_logg_mass_radius_relation',
        'observed_flux_unit': flux_unit,
        'observed_flux_unit_label': flux_unit_label,
        'sampler_backend': sampler_backend,
        'sampler_temperature': locals().get('sampler_temperature', 1.0),
        'sampler_effective_n': locals().get('sampler_effective_n', float(len(flat))),
        'mcmc_acceptance_fraction': mcmc_acceptance_fraction,
        'mcmc_acceptance_fraction_min': mcmc_acceptance_fraction_min,
        'mcmc_acceptance_fraction_max': mcmc_acceptance_fraction_max,
        'mcmc_autocorr_time_max': mcmc_autocorr_time_max,
        'mcmc_converged': bool(mcmc_converged),
        'mcmc_warning': mcmc_warning,
        'nwalkers': int(nwalkers),
        'nsteps': int(nsteps),
        'burn': int(burn),
        'thin': int(thin),
        'n_posterior_samples': int(len(flat)),
        'best_model_wave': med_model_wave,
        'best_model_flux': med_model_flux,
    }
    if r_info is not None:
        dist_cm = r_info['distance_pc'] * 3.085677581491367e18
        scale_radius_samples = (
            dist_cm * np.sqrt(np.maximum(scale_samples * flux_unit, 0.0))
            / R_SUN_CM)
        scale_radius_q = _finite_percentiles(scale_radius_samples)
        scale_mass_samples = (
            (10.0 ** flat[:, 1])
            * (scale_radius_samples * R_SUN_CM) ** 2
            / G_CGS / M_SUN_G)
        scale_mass_q = _finite_percentiles(scale_mass_samples)
        vgrav_scale_q = _finite_percentiles(
            _vgrav_from_mass_radius(scale_mass_samples, scale_radius_samples))
        result.update({
            'distance_pc': r_info['distance_pc'],
            'scale_physical_factor': r_info['physical_scale'],
            'scale_radius_rsun': r_info['radius_rsun'],
            'scale_radius_rsun_err_minus': float(scale_radius_q[1] - scale_radius_q[0]),
            'scale_radius_rsun_err_plus': float(scale_radius_q[2] - scale_radius_q[1]),
            'scale_radius_rsun_err': float(0.5 * ((scale_radius_q[1] - scale_radius_q[0])
                                                 + (scale_radius_q[2] - scale_radius_q[1]))),
            'scale_radius_ratio_to_mr': r_info['radius_rsun'] / radius_mr,
        })
        result['mass_msun_from_scale_logg'] = float(scale_mass_q[1])
        result['mass_msun_from_scale_logg_err_minus'] = float(scale_mass_q[1] - scale_mass_q[0])
        result['mass_msun_from_scale_logg_err_plus'] = float(scale_mass_q[2] - scale_mass_q[1])
        result['mass_msun_from_scale_logg_err'] = float(0.5 * ((scale_mass_q[1] - scale_mass_q[0])
                                                              + (scale_mass_q[2] - scale_mass_q[1])))
        result['v_grav_scale_logg_kms'] = float(vgrav_scale_q[1])
        result['v_grav_scale_logg_err_minus'] = float(vgrav_scale_q[1] - vgrav_scale_q[0])
        result['v_grav_scale_logg_err_plus'] = float(vgrav_scale_q[2] - vgrav_scale_q[1])
        result['v_grav_scale_logg_err'] = float(0.5 * ((vgrav_scale_q[1] - vgrav_scale_q[0])
                                                       + (vgrav_scale_q[2] - vgrav_scale_q[1])))
        if (np.isfinite(scale_radius_q[1]) and 0.003 <= scale_radius_q[1] <= 0.05
                and np.isfinite(scale_mass_q[1]) and scale_mass_q[1] > 0):
            result['mass_msun_preferred'] = float(scale_mass_q[1])
            result['mass_msun_preferred_err'] = result['mass_msun_from_scale_logg_err']
            result['radius_rsun_preferred'] = float(scale_radius_q[1])
            result['radius_rsun_preferred_err'] = result['scale_radius_rsun_err']
            result['v_grav_preferred_kms'] = float(vgrav_scale_q[1])
            result['v_grav_preferred_err'] = result['v_grav_scale_logg_err']
            result['preferred_physical_source'] = 'mcmc_gaia_parallax_scale_logg'

    if np.isfinite(cool_q[1]):
        result['cooling_age_gyr'] = float(cool_q[1])
        result['cooling_age_gyr_err_minus'] = float(cool_q[1] - cool_q[0])
        result['cooling_age_gyr_err_plus'] = float(cool_q[2] - cool_q[1])
        result['cooling_age_gyr_err'] = float(0.5 * ((cool_q[1] - cool_q[0]) + (cool_q[2] - cool_q[1])))
        result['total_age_gyr'] = float(total_q[1])
        result['total_age_gyr_err_minus'] = float(total_q[1] - total_q[0])
        result['total_age_gyr_err_plus'] = float(total_q[2] - total_q[1])
        result['total_age_gyr_err'] = float(0.5 * ((total_q[1] - total_q[0]) + (total_q[2] - total_q[1])))
        result['age_source'] = 'NN_MCMC_Teff_logg+MR_mass+WD_cooling_tracks'
    else:
        cool_fallback = (
            8.8e6 * (mass_samples / 0.6)**(5.0/7.0)
            * (flat[:, 0] / 12000)**(-2.5) / 1e9)
        cool_q = _finite_percentiles(cool_fallback)
        result['cooling_age_gyr'] = float(cool_q[1])
        result['cooling_age_gyr_err_minus'] = float(cool_q[1] - cool_q[0])
        result['cooling_age_gyr_err_plus'] = float(cool_q[2] - cool_q[1])
        result['cooling_age_gyr_err'] = float(0.5 * ((cool_q[1] - cool_q[0]) + (cool_q[2] - cool_q[1])))
        result['total_age_gyr'] = np.nan
        result['age_source'] = 'NN_MCMC_Teff_logg+MR_mass+rough_Mestel_age'
    try:
        from .cooling_age import compute_progenitor_lifetime
        if len(mass_samples) > 1500:
            prog_idx = np.linspace(0, len(mass_samples) - 1, 1500).astype(int)
        else:
            prog_idx = np.arange(len(mass_samples))
        progenitor = []
        ms_life = []
        cool_for_ms = []
        for sample_i, m in zip(prog_idx, mass_samples[prog_idx]):
            prog = compute_progenitor_lifetime(float(m))
            if prog is not None:
                progenitor.append(prog.get('m_progenitor', np.nan))
                ms_life.append(prog.get('ms_lifetime_gyr', np.nan))
                cool_for_ms.append(np.asarray(cool_samples.get('cooling_age_gyr'), dtype=float)[sample_i])
        prog_q = _finite_percentiles(progenitor)
        ms_q = _finite_percentiles(ms_life)
        if np.isfinite(prog_q[1]):
            result['m_progenitor_msun'] = float(prog_q[1])
            result['m_progenitor_msun_err_minus'] = float(prog_q[1] - prog_q[0])
            result['m_progenitor_msun_err_plus'] = float(prog_q[2] - prog_q[1])
            result['m_progenitor_msun_err'] = float(0.5 * ((prog_q[1] - prog_q[0])
                                                          + (prog_q[2] - prog_q[1])))
        if np.isfinite(ms_q[1]):
            result['ms_lifetime_gyr'] = float(ms_q[1])
            result['ms_lifetime_gyr_err_minus'] = float(ms_q[1] - ms_q[0])
            result['ms_lifetime_gyr_err_plus'] = float(ms_q[2] - ms_q[1])
            result['ms_lifetime_gyr_err'] = float(0.5 * ((ms_q[1] - ms_q[0])
                                                        + (ms_q[2] - ms_q[1])))
            if np.isfinite(result.get('cooling_age_gyr', np.nan)):
                total_ms = np.asarray(cool_for_ms, dtype=float) + np.asarray(ms_life, dtype=float)
                total_ms_q = _finite_percentiles(total_ms)
                result['total_age_with_ms_gyr'] = float(total_ms_q[1])
                result['total_age_with_ms_gyr_err_minus'] = float(total_ms_q[1] - total_ms_q[0])
                result['total_age_with_ms_gyr_err_plus'] = float(total_ms_q[2] - total_ms_q[1])
                result['total_age_with_ms_gyr_err'] = float(0.5 * ((total_ms_q[1] - total_ms_q[0])
                                                                  + (total_ms_q[2] - total_ms_q[1])))
    except Exception:
        pass

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        import pandas as pd
        summary_path = os.path.join(output_dir, 'wd_nn_mcmc_summary.csv')
        summary = {k: v for k, v in result.items()
                   if not isinstance(v, np.ndarray)}
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        result['summary_path'] = summary_path

        sample_path = os.path.join(output_dir, 'wd_nn_mcmc_samples.csv')
        sample_df = pd.DataFrame(
            flat, columns=['teff', 'logg', 'rv_kms', 'ln_scale'])
        sample_df['scale'] = np.exp(sample_df['ln_scale'])
        if len(sample_df) > 10000:
            sample_df = sample_df.sample(10000, random_state=random_seed)
        sample_df.to_csv(sample_path, index=False)
        result['samples_path'] = sample_path

        model_band = None
        try:
            n_band = min(160, len(flat))
            band_idx = rng.choice(len(flat), size=n_band, replace=False)
            model_samples = []
            for theta in flat[band_idx]:
                surface = _weighted_grid_spectrum(
                    labels, flux_grid, theta[0], theta[1])
                rest_wave = w / (1.0 + theta[2] / C_KMS)
                sample_model = np.interp(
                    rest_wave, model_wave, surface, left=np.nan, right=np.nan)
                sample_model = sample_model * np.exp(theta[3])
                model_samples.append(sample_model)
            model_samples = np.asarray(model_samples, dtype=float)
            model_band = np.nanpercentile(model_samples, [16, 50, 84], axis=0)
        except Exception:
            model_band = None

        model_csv_path = os.path.join(output_dir, 'wd_nn_mcmc_model_band.csv')
        if model_band is not None:
            pd.DataFrame({
                'wavelength_A': w,
                'flux_obs': f,
                'flux_err': e,
                'model_p16': model_band[0],
                'model_p50': model_band[1],
                'model_p84': model_band[2],
                'residual': f - model_band[1],
            }).to_csv(model_csv_path, index=False)
        else:
            interp_med = np.interp(w, med_model_wave, med_model_flux,
                                   left=np.nan, right=np.nan)
            pd.DataFrame({
                'wavelength_A': w,
                'flux_obs': f,
                'flux_err': e,
                'model_p50': interp_med,
                'residual': f - interp_med,
            }).to_csv(model_csv_path, index=False)
        result['model_band_path'] = model_csv_path

        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 sharex=True)
        axes[0].fill_between(
            w, f - e, f + e, color='0.70', alpha=0.28, lw=0,
            label='Observed 1 sigma')
        axes[0].plot(w, f, color='black', lw=0.55, alpha=0.72,
                     label='Observed')
        if model_band is not None:
            axes[0].fill_between(
                w, model_band[0], model_band[2],
                color='lightcoral', alpha=0.30, lw=0,
                label='Model 68% posterior')
            axes[0].plot(
                w, model_band[1], color='crimson', lw=1.1, alpha=0.95,
                label=f"NN MCMC: Teff={med[0]:.0f} K, logg={med[1]:.2f}")
        else:
            axes[0].plot(med_model_wave, med_model_flux, color='crimson',
                         lw=1.0, alpha=0.9,
                         label=f"NN MCMC: Teff={med[0]:.0f} K, logg={med[1]:.2f}")
        for name, lam in BALMER_PROFILE_LINES.items():
            axes[0].axvline(lam, color='steelblue', ls=':', lw=0.7, alpha=0.35)
            axes[0].text(lam, 0.97, name, transform=axes[0].get_xaxis_transform(),
                         rotation=90, va='top', ha='right', fontsize=7,
                         color='steelblue', alpha=0.7)
        axes[0].set_ylabel('Flux')
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.25)
        utils.set_spectrum_axes(axes[0], wave, flux, model=med_model_flux)
        if model_band is not None:
            interp_model = model_band[1]
        else:
            interp_model = np.interp(w, med_model_wave, med_model_flux,
                                     left=np.nan, right=np.nan)
        axes[1].fill_between(
            w, -e, e, color='0.70', alpha=0.28, lw=0,
            label='Observed 1 sigma')
        if model_band is not None:
            axes[1].fill_between(
                w, f - model_band[2], f - model_band[0],
                color='lightcoral', alpha=0.25, lw=0,
                label='Model 68% posterior')
        axes[1].plot(w, f - interp_model, color='black', lw=0.5, alpha=0.65)
        axes[1].axhline(0, color='crimson', ls='--', lw=0.8)
        axes[1].set_xlabel('Wavelength (A)')
        axes[1].set_ylabel('Residual')
        axes[1].legend(fontsize=8, loc='best')
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        fit_path = os.path.join(output_dir, 'wd_nn_mcmc_fit.png')
        utils.save_and_close(fig, fit_path)
        result['fit_plot_path'] = fit_path

        try:
            import corner
            corner_fig = corner.corner(
                flat, labels=['Teff', 'logg', 'RV', 'ln scale'],
                truths=med, show_titles=True)
            corner_path = os.path.join(output_dir, 'wd_nn_mcmc_corner.png')
            utils.save_and_close(corner_fig, corner_path)
            result['corner_plot_path'] = corner_path
        except Exception as exc:
            result['corner_error'] = str(exc)

    return result


# ==================================================================
#  DWD 组合拟合
# ==================================================================

def fit_dwd(wave, flux, err=None, single_result=None,
            model_grid='auto', spectral_type='DA'):
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
    templates, _, _, model_source = _resolve_template_grid(
        model_grid, spectral_type)
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
        'model_grid': model_source,
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

    # 方法 A: 光谱 log g + WD mass-radius relation。This preserves the
    # atmospheric fit as the primary Teff/logg measurement; Gaia/SED checks are
    # attached below instead of silently replacing the spectroscopic solution.
    mass_est = _logg_to_mass(logg)
    R_est = compute_wd_radius(mass_est, logg)
    result['mass'] = mass_est
    result['radius_rsun'] = R_est
    result['source'] = 'spectroscopic_logg_MR_relation'
    try:
        cool = _cooling_from_teff_mass(teff, mass_est)
        if cool is not None:
            result['cooling_age_gyr'] = cool.get('cooling_age_gyr', np.nan)
            result['total_age_gyr'] = cool.get('total_age_gyr', np.nan)
            result['source'] = 'spectroscopic_logg_MR_relation+WD_models'
        else:
            raise ValueError('outside cooling grid')
    except Exception:
        # Last-resort rough scaling.  Reports keep the source string so this is
        # not confused with a proper WD cooling-track age.
        t_cool_yr = 8.8e6 * (mass_est / 0.6)**(5.0/7.0) * (teff / 12000)**(-2.5)
        result['cooling_age_gyr'] = t_cool_yr / 1e9
        result['source'] = 'spectroscopic_logg_MR_relation+rough_Mestel_age'

    if parallax_mas is not None and parallax_mas > 0:
        result['distance_pc'] = 1000.0 / parallax_mas
    try:
        from .cooling_age import compute_progenitor_lifetime
        prog = compute_progenitor_lifetime(mass_est)
        if prog is not None:
            result['m_progenitor'] = prog['m_progenitor']
            result['ms_lifetime_gyr'] = prog['ms_lifetime_gyr']
    except Exception:
        pass

    # 方法 B: 用 WD_models 从 (BP-RP, M_G) 插值，作为距离/测光一致性检查。
    if bp_rp is not None and M_G is not None:
        try:
            from .cooling_age import interpolate_wd_params, compute_progenitor_lifetime
            wd = interpolate_wd_params(bp_rp, M_G)
            if wd is not None:
                result['gaia_hr_mass'] = wd['mass']
                result['gaia_hr_radius_rsun'] = compute_wd_radius(wd['mass'], wd['logg'])
                result['gaia_hr_cooling_age_gyr'] = wd['cooling_age_gyr']
                result['gaia_hr_total_age_gyr'] = wd['total_age_gyr']
                result['gaia_hr_teff'] = wd['teff']
                result['gaia_hr_logg'] = wd['logg']
                result['gaia_hr_source'] = 'WD_models_HR'
                result['delta_teff_gaia_minus_spec'] = wd['teff'] - teff
                result['delta_logg_gaia_minus_spec'] = wd['logg'] - logg

                prog = compute_progenitor_lifetime(wd['mass'])
                if prog is not None:
                    result['gaia_hr_m_progenitor'] = prog['m_progenitor']
                    result['gaia_hr_ms_lifetime_gyr'] = prog['ms_lifetime_gyr']
        except Exception:
            pass

    return result


def _cooling_from_teff_mass(teff, mass_msun):
    """Interpolate WD cooling tracks in (mass, logTeff)."""
    if teff is None or mass_msun is None or teff <= 0 or mass_msun <= 0:
        return None
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        from .cooling_age import _load_wd_model
        model = _load_wd_model()
        mass = np.asarray(model['mass_array'], dtype=float)
        logteff = np.asarray(model['logteff'], dtype=float)
        age_cool = np.asarray(model['age_cool'], dtype=float)
        age_total = np.asarray(model['age'], dtype=float)
        good = (np.isfinite(mass) & np.isfinite(logteff)
                & np.isfinite(age_cool) & np.isfinite(age_total))
        pts = np.column_stack([mass[good], logteff[good]])
        target = np.array([[float(mass_msun), np.log10(float(teff))]])
        cool_interp = LinearNDInterpolator(pts, age_cool[good])
        total_interp = LinearNDInterpolator(pts, age_total[good])
        cool = float(cool_interp(target)[0])
        total = float(total_interp(target)[0])
        if not np.isfinite(cool):
            cool = float(NearestNDInterpolator(pts, age_cool[good])(target)[0])
        if not np.isfinite(total):
            total = float(NearestNDInterpolator(pts, age_total[good])(target)[0])
        return {
            'cooling_age_gyr': cool,
            'total_age_gyr': total,
        }
    except Exception:
        return None


def _cooling_from_teff_mass_samples(teff, mass_msun):
    """Vectorized cooling-track interpolation for MCMC posterior samples."""
    teff = np.asarray(teff, dtype=float)
    mass_msun = np.asarray(mass_msun, dtype=float)
    out_shape = np.broadcast(teff, mass_msun).shape
    teff = np.broadcast_to(teff, out_shape).ravel()
    mass_msun = np.broadcast_to(mass_msun, out_shape).ravel()
    cool_out = np.full_like(teff, np.nan, dtype=float)
    total_out = np.full_like(teff, np.nan, dtype=float)
    valid_target = np.isfinite(teff) & np.isfinite(mass_msun) & (teff > 0) & (mass_msun > 0)
    if not np.any(valid_target):
        return {
            'cooling_age_gyr': cool_out.reshape(out_shape),
            'total_age_gyr': total_out.reshape(out_shape),
        }
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        from .cooling_age import _load_wd_model
        model = _load_wd_model()
        mass = np.asarray(model['mass_array'], dtype=float)
        logteff = np.asarray(model['logteff'], dtype=float)
        age_cool = np.asarray(model['age_cool'], dtype=float)
        age_total = np.asarray(model['age'], dtype=float)
        good = (np.isfinite(mass) & np.isfinite(logteff)
                & np.isfinite(age_cool) & np.isfinite(age_total))
        pts = np.column_stack([mass[good], logteff[good]])
        target = np.column_stack([
            mass_msun[valid_target],
            np.log10(teff[valid_target]),
        ])
        cool_interp = LinearNDInterpolator(pts, age_cool[good])
        total_interp = LinearNDInterpolator(pts, age_total[good])
        cool = np.asarray(cool_interp(target), dtype=float)
        total = np.asarray(total_interp(target), dtype=float)
        miss = ~np.isfinite(cool)
        if np.any(miss):
            cool[miss] = np.asarray(
                NearestNDInterpolator(pts, age_cool[good])(target[miss]),
                dtype=float)
        miss = ~np.isfinite(total)
        if np.any(miss):
            total[miss] = np.asarray(
                NearestNDInterpolator(pts, age_total[good])(target[miss]),
                dtype=float)
        cool_out[valid_target] = cool
        total_out[valid_target] = total
    except Exception:
        pass
    return {
        'cooling_age_gyr': cool_out.reshape(out_shape),
        'total_age_gyr': total_out.reshape(out_shape),
    }


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

    def __init__(self, wave, flux, err=None, model_grid='auto'):
        self.wave = np.asarray(wave, dtype=np.float64)
        self.flux = np.asarray(flux, dtype=np.float64)
        self.err = np.asarray(err, dtype=np.float64) if err is not None else None
        self.model_grid = model_grid

        self.classification = None
        self.single_fit = None
        self.continuum_fit = None
        self.balmer_fit = None
        self.dwd_fit = None
        self.physical_params = None
        self.sed_fit = None

    def classify(self):
        """光谱分类"""
        self.classification = classify_wd_type(self.wave, self.flux, self.err)
        return self.classification

    def _fit_spectral_type(self):
        if self.classification and self.classification.get('spectral_type'):
            st = str(self.classification.get('spectral_type')).upper()
            if st.startswith('DB'):
                return 'DB'
        return 'DA'

    def fit_single(self, line_only=False, teff_prior=None,
                   teff_prior_sigma=3000.0, model_grid=None):
        """单星 WD 网格拟合，默认优先使用本地 NN DA/DB 模板。"""
        result = fit_single_wd(self.wave, self.flux, self.err,
                               line_only=line_only,
                               teff_prior=teff_prior,
                               teff_prior_sigma=teff_prior_sigma,
                               model_grid=model_grid or self.model_grid,
                               spectral_type=self._fit_spectral_type())
        if line_only:
            self.balmer_fit = result
        else:
            self.continuum_fit = result
        self.single_fit = self.balmer_fit or self.continuum_fit
        return self.single_fit

    def fit_balmer(self, teff_prior=None, teff_prior_sigma=3000.0,
                   model_grid=None):
        """H-beta through H8 normalized Balmer-line profile fit."""
        self.balmer_fit = fit_balmer_line_profiles(
            self.wave, self.flux, self.err,
            teff_prior=teff_prior,
            teff_prior_sigma=teff_prior_sigma,
            model_grid=model_grid or self.model_grid,
            spectral_type=self._fit_spectral_type())
        self.single_fit = self.balmer_fit or self.continuum_fit
        return self.balmer_fit

    def fit_double(self, model_grid=None):
        """DWD 组合拟合"""
        if self.continuum_fit is None:
            self.fit_single(line_only=False)
        self.dwd_fit = fit_dwd(self.wave, self.flux, self.err,
                               single_result=self.continuum_fit,
                               model_grid=model_grid or self.model_grid,
                               spectral_type=self._fit_spectral_type())
        return self.dwd_fit

    def fit_sed(self, photometry, parallax_mas, model_grid=None):
        """宽波段 SED 拟合"""
        self.sed_fit = fit_sed(
            photometry, parallax_mas,
            model_grid=model_grid or self.model_grid,
            spectral_type=self._fit_spectral_type())
        return self.sed_fit

    def derive_params(self, parallax_mas=None, bp_rp=None, M_G=None):
        """推导物理参数"""
        if self.single_fit is None:
            if self.continuum_fit is None:
                self.fit_single(line_only=False)
            prior = self.continuum_fit.get('teff') if self.continuum_fit else None
            self.fit_balmer(teff_prior=prior)
        preferred = self.balmer_fit or self.single_fit
        if preferred is None:
            self.fit_single(line_only=False)
            preferred = self.single_fit
        if self.single_fit is None:
            return None
        self.physical_params = derive_physical_params(
            preferred['teff'], preferred['logg'],
            parallax_mas=parallax_mas, bp_rp=bp_rp, M_G=M_G)
        if self.physical_params:
            for key in ('teff_err', 'logg_err'):
                if preferred.get(key) is not None:
                    self.physical_params[key] = preferred.get(key)
            logg = self.physical_params.get('logg')
            logg_err = self.physical_params.get('logg_err')
            try:
                logg = float(logg)
                logg_err = float(logg_err)
            except Exception:
                logg = logg_err = np.nan
            if np.isfinite(logg) and np.isfinite(logg_err) and logg_err > 0:
                lo = max(logg - logg_err, 6.0)
                hi = min(logg + logg_err, 10.0)
                m_lo = _logg_to_mass(lo)
                m_hi = _logg_to_mass(hi)
                r_lo = compute_wd_radius(m_lo, lo)
                r_hi = compute_wd_radius(m_hi, hi)
                self.physical_params['mass_err'] = float(abs(m_hi - m_lo) / 2.0)
                self.physical_params['radius_rsun_err'] = float(abs(r_hi - r_lo) / 2.0)
        expected_radius = None
        if self.physical_params:
            expected_radius = self.physical_params.get('radius_rsun')
        attach_distance_scale_check(
            self.continuum_fit, parallax_mas, self.flux,
            expected_radius_rsun=expected_radius)
        attach_distance_scale_check(
            self.balmer_fit, parallax_mas, self.flux,
            expected_radius_rsun=expected_radius)
        attach_distance_scale_check(
            self.single_fit, parallax_mas, self.flux,
            expected_radius_rsun=expected_radius)
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

        print("  [2/5] Balmer 线轮廓拟合 + 全谱对照...")
        self.fit_single(line_only=False)
        prior = self.continuum_fit.get('teff') if self.continuum_fit else None
        self.fit_balmer(teff_prior=prior)
        if self.balmer_fit:
            print(f"    Balmer: Teff = {self.balmer_fit['teff']} K  "
                  f"logg = {self.balmer_fit['logg']:.2f}  "
                  f"RV = {self.balmer_fit.get('rv_kms', np.nan):.1f} km/s  "
                  f"chi2_red = {self.balmer_fit['chi2_red']:.4f}  "
                  f"prior_T={self.balmer_fit.get('teff_prior')}  "
                  f"grid={self.balmer_fit.get('model_grid')}")
        else:
            print("    Balmer 拟合失败")
        if self.continuum_fit:
            print(f"    Full-spectrum check: Teff = {self.continuum_fit['teff']} K  "
                  f"logg = {self.continuum_fit['logg']:.2f}  "
                  f"chi2_red = {self.continuum_fit['chi2_red']:.4f}  "
                  f"grid={self.continuum_fit.get('model_grid')}")

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
            'balmer_fit': self.balmer_fit,
            'continuum_fit': self.continuum_fit,
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

        # 设置合理轴范围，再用轴坐标标注 Balmer 线；这样强发射线/坏点
        # 不会把文字推到图外，导致整张拟合图被压扁。
        utils.set_spectrum_axes(ax_spec, self.wave, self.flux,
                                model=sf['best_model_flux'])

        # Balmer 线标注
        for name, lam in BALMER_LINES.items():
            if not (np.nanmin(self.wave) <= lam <= np.nanmax(self.wave)):
                continue
            ax_spec.axvline(lam, color='blue', ls=':', alpha=0.3, lw=0.8)
            ax_spec.text(lam, 0.96, name, transform=ax_spec.get_xaxis_transform(),
                         fontsize=7, rotation=90, va='top', ha='right',
                         color='blue', alpha=0.55, clip_on=True)

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

        # 残差
        f_interp = interp1d(sf['best_model_wave'], sf['best_model_flux'],
                            kind='linear', bounds_error=False, fill_value=0)
        model_at_obs = f_interp(self.wave)
        residual = self.flux - model_at_obs
        if save_path:
            try:
                import os
                import pandas as pd
                out_dir = os.path.dirname(os.path.abspath(save_path))
                err = self.err if self.err is not None else np.full_like(self.wave, np.nan)
                pd.DataFrame({
                    'wavelength_A': self.wave,
                    'flux_obs': self.flux,
                    'flux_err': err,
                    'flux_model': model_at_obs,
                    'residual': residual,
                }).to_csv(os.path.join(out_dir, 'wd_spectral_fit_model.csv'),
                          index=False)
                pd.DataFrame({
                    'wavelength_A': sf['best_model_wave'],
                    'flux_model_full': sf['best_model_flux'],
                }).to_csv(os.path.join(out_dir, 'wd_spectral_fit_model_full.csv'),
                          index=False)
            except Exception:
                pass
        ax_res.plot(self.wave, residual, 'k-', lw=0.5, alpha=0.6)
        ax_res.axhline(0, color='red', ls='--', lw=0.8)
        good_res = residual[np.isfinite(residual)]
        if len(good_res) > 5:
            rlo, rhi = np.nanpercentile(good_res, [2, 98])
            rpad = max((rhi - rlo) * 0.20, np.nanstd(good_res) * 0.05, 1e-6)
            if np.isfinite(rlo + rhi + rpad) and rhi > rlo:
                ax_res.set_ylim(rlo - rpad, rhi + rpad)
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
        teffs = self.single_fit.get('teff_grid')
        loggs = self.single_fit.get('logg_grid')
        if not teffs or not loggs:
            teffs, loggs = _get_model_grid_params(
                model_grid=self.single_fit.get('model_grid', self.model_grid),
                spectral_type=self.single_fit.get('spectral_type', self._fit_spectral_type()))

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
        ax.set_title(
            f"$\\chi^2$ Map — {self.single_fit.get('model_grid', 'WD')} Grid Fit",
            fontsize=13)
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
        templates, _, _, _ = _resolve_template_grid(
            sf.get('model_grid', self.model_grid),
            sf.get('spectral_type', self._fit_spectral_type()))
        best_tmpl = templates.get((sf['teff_sed'], sf['logg_sed']))
        if best_tmpl is None:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # 模型连续谱 (缩放后)
        model_wave = best_tmpl['wavelength']
        model_flux = best_tmpl['flux'] * sf['scale']
        ax.plot(model_wave, model_flux, 'b-', lw=0.8, alpha=0.5,
                label=f"Model: Teff={sf['teff_sed']}K, logg={sf['logg_sed']:.1f}")
        tail_wave = np.asarray(sf.get('best_model_tail_wave', []), dtype=float)
        tail_flux = np.asarray(sf.get('best_model_tail_flux', []), dtype=float)
        if tail_wave.size > 1 and tail_flux.size == tail_wave.size:
            ax.plot(tail_wave, tail_flux, 'b--', lw=0.8, alpha=0.45,
                    label='WD Rayleigh-Jeans tail')

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
        excess_note = ''
        if sf.get('ir_excess_flag'):
            bands = ','.join(sf.get('ir_excess_bands', []))
            excess_note = f'  IR excess: {bands}'
        ax.set_title(f'SED Fit  Teff={sf["teff_sed"]}K  '
                     f'R={sf["R_Rsun"]:.4f} R$_\\odot$  '
                     f'$\\chi^2_{{phot}}$={sf["chi2_sed"]:.3f}'
                     f'{excess_note}', fontsize=12)
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

        if self.balmer_fit:
            sf = self.balmer_fit
            lines.append(f"\n--- Preferred Balmer Profile Fit (H-beta through H8) ---")
            lines.append(f"  Template grid = {sf.get('model_grid', 'unknown')}")
            lines.append(f"  Teff = {sf['teff']} +/- {sf['teff_err']:.0f} K")
            lines.append(f"  logg = {sf['logg']:.2f} +/- {sf['logg_err']:.2f}")
            lines.append(f"  chi2_red = {sf['chi2_red']:.6f}")
            if sf.get('teff_prior') is not None:
                lines.append(f"  Teff branch prior = {sf.get('teff_prior'):.0f} K "
                             f"(sigma={sf.get('teff_prior_sigma'):.0f} K)")
                lines.append(f"  fit score = {sf.get('fit_score', np.nan):.6f}")
            lines.append(f"  RV shift = {sf.get('rv_kms', np.nan):.1f} km/s")
            lines.append("  Lines used: " + ', '.join(sf.get('lines_used', [])))
            lines.append(f"  scale = {sf['scale']:.6e}")
            if sf.get('scale_radius_rsun') is not None:
                lines.append(f"  Distance-scale radius = {sf['scale_radius_rsun']:.5f} R_sun "
                             f"({sf.get('scale_observed_flux_unit_label', 'unknown')})")
                lines.append(f"  Distance-scale check OK = {sf.get('scale_radius_ok')}")

        if self.continuum_fit:
            sf = self.continuum_fit
            lines.append(f"\n--- Full-Spectrum Flux Fit (continuum check) ---")
            lines.append(f"  Template grid = {sf.get('model_grid', 'unknown')}")
            lines.append(f"  Teff = {sf['teff']} +/- {sf['teff_err']:.0f} K")
            lines.append(f"  logg = {sf['logg']:.2f} +/- {sf['logg_err']:.2f}")
            lines.append(f"  chi2_red = {sf['chi2_red']:.6f}")
            lines.append(f"  scale = {sf['scale']:.6e}")
            if sf.get('scale_radius_rsun') is not None:
                lines.append(f"  Distance-scale radius = {sf['scale_radius_rsun']:.5f} R_sun "
                             f"({sf.get('scale_observed_flux_unit_label', 'unknown')})")
                lines.append(f"  Distance-scale check OK = {sf.get('scale_radius_ok')}")

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
            lines.append(f"  chi2_SED_photospheric = {s['chi2_sed']:.4f}")
            if np.isfinite(s.get('chi2_sed_all', np.nan)):
                lines.append(f"  chi2_SED_all_bands = {s['chi2_sed_all']:.4f}")
            if s.get('sed_red_tail_bands'):
                lines.append("  Red-tail WD model bands = "
                             + ", ".join(s.get('sed_red_tail_bands', [])))
            lines.append(f"  IR excess flag = {s.get('ir_excess_flag', False)}")
            if s.get('ir_excess_bands'):
                lines.append("  IR excess bands = "
                             + ", ".join(s.get('ir_excess_bands', [])))
            if np.isfinite(s.get('max_ir_excess_dex', np.nan)):
                lines.append(f"  Max IR excess = {s['max_ir_excess_dex']:.3f} dex "
                             f"({s.get('max_ir_excess_sigma', np.nan):.1f} sigma)")

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
            if 'distance_pc' in p:
                lines.append(f"  Distance = {p['distance_pc']:.2f} pc")
            if p.get('gaia_hr_source'):
                lines.append(f"  Gaia HR check: M={p.get('gaia_hr_mass', np.nan):.3f} "
                             f"M_sun, Teff={p.get('gaia_hr_teff', np.nan):.0f} K, "
                             f"logg={p.get('gaia_hr_logg', np.nan):.3f}, "
                             f"t_cool={p.get('gaia_hr_cooling_age_gyr', np.nan):.3f} Gyr")
                lines.append(f"  Gaia - spec: dTeff={p.get('delta_teff_gaia_minus_spec', np.nan):+.0f} K, "
                             f"dlogg={p.get('delta_logg_gaia_minus_spec', np.nan):+.3f}")

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
        if self.balmer_fit:
            sf = self.balmer_fit
            row.update({
                'teff': sf['teff'], 'teff_err': sf['teff_err'],
                'logg': sf['logg'], 'logg_err': sf['logg_err'],
                'chi2_red': sf['chi2_red'],
                'fit_method': sf.get('method', 'Balmer_profile_Hbeta_to_H8'),
                'model_grid': sf.get('model_grid'),
                'fit_score': sf.get('fit_score'),
                'teff_prior': sf.get('teff_prior'),
                'teff_prior_sigma': sf.get('teff_prior_sigma'),
                'balmer_rv_kms': sf.get('rv_kms'),
                'balmer_lines_used': ';'.join(sf.get('lines_used', [])),
                'balmer_scale_radius_rsun': sf.get('scale_radius_rsun'),
                'balmer_scale_radius_ok': sf.get('scale_radius_ok'),
                'balmer_scale_flux_unit': sf.get('scale_observed_flux_unit_label'),
            })
        if self.continuum_fit:
            sf = self.continuum_fit
            row.update({
                'continuum_teff': sf['teff'],
                'continuum_teff_err': sf['teff_err'],
                'continuum_logg': sf['logg'],
                'continuum_logg_err': sf['logg_err'],
                'continuum_chi2_red': sf['chi2_red'],
                'continuum_model_grid': sf.get('model_grid'),
                'continuum_scale_radius_rsun': sf.get('scale_radius_rsun'),
                'continuum_scale_radius_ok': sf.get('scale_radius_ok'),
                'continuum_scale_flux_unit': sf.get('scale_observed_flux_unit_label'),
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
                'chi2_sed_all': s.get('chi2_sed_all'),
                'sed_fit_bands': ';'.join(s.get('sed_fit_bands', [])),
                'sed_red_tail_bands': ';'.join(s.get('sed_red_tail_bands', [])),
                'ir_excess_flag': s.get('ir_excess_flag'),
                'ir_excess_bands': ';'.join(s.get('ir_excess_bands', [])),
                'max_ir_excess_dex': s.get('max_ir_excess_dex'),
                'max_ir_excess_sigma': s.get('max_ir_excess_sigma'),
                'sed_model_grid': s.get('model_grid'),
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
                'distance_pc': p.get('distance_pc'),
                'gaia_hr_mass': p.get('gaia_hr_mass'),
                'gaia_hr_teff': p.get('gaia_hr_teff'),
                'gaia_hr_logg': p.get('gaia_hr_logg'),
                'gaia_hr_cooling_age_gyr': p.get('gaia_hr_cooling_age_gyr'),
                'delta_teff_gaia_minus_spec': p.get('delta_teff_gaia_minus_spec'),
                'delta_logg_gaia_minus_spec': p.get('delta_logg_gaia_minus_spec'),
            })

        if not row:
            return None
        df = pd.DataFrame([row])
        path = os.path.join(output_dir, 'wd_fitting_results.csv')
        df.to_csv(path, index=False)
        return path
