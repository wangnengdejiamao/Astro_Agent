"""
astro_toolbox.six_dim — 6D 运动学认证工具
==========================================
功能:
  1. check_6d_match()     — 对单颗白矮星做 6D (5D 运动学 + RV) 匹配判断
  2. recalc_6d()          — 批量重算整个 DataFrame 的 6D 列
  3. plot_spectrum()      — 绘制光谱 + Koester 模型叠加
  4. plot_ztf()           — 绘制 ZTF g/r/i 光变曲线
  5. plot_hrd()           — 绘制 HR 图（Gaia 背景 + 星团成员高亮）
  6. plot_rv_info()       — 绘制 RV / 6D 认证摘要信息板
  7. plot_total_hrd()     — 所有 6D 认证源的总 HRD，按星团着色
  8. make_6d_plots()      — 对 DataFrame 中所有 6D 认证源批量生成全套图集

典型用法
--------
from astro_toolbox import six_dim

# --- 单颗判断 ---
from astro_toolbox.orbit_traceback import load_hunt2023_clusters
clusters = load_hunt2023_clusters()
result = six_dim.check_6d_match(row, rv_true=25.3, rv_err=8.0,
                                 cluster_cache=clusters)

# --- 批量重算 ---
import pandas as pd
df = pd.read_csv('wd_analysis_final.csv')
df = six_dim.recalc_6d(df)

# --- 生成全套图集 ---
six_dim.make_6d_plots(df, merged_df,
                      output_dir='6d_confirmed_plots',
                      gaia_bg_csv='/path/to/TAP_xxx.csv')
"""

from __future__ import annotations

import os
import traceback
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

__all__ = [
    'check_6d_match',
    'evaluate_dynamics_flags',
    'apply_dynamics_flags',
    'recalc_membership',
    'recalc_6d',
    'plot_spectrum',
    'plot_ztf',
    'plot_hrd',
    'plot_5d_astrometry',
    'plot_rv_info',
    'evaluate_final_5d_rv_gates',
    'plot_final_5d_validation',
    'plot_final_rv_validation',
    'plot_final_6d_validation',
    'plot_final_6d_validation_set',
    'plot_sed',
    'plot_total_hrd',
    'make_6d_plots',
]


# ──────────────────────────────────────────────────────────────────
#  6D 匹配逻辑
# ──────────────────────────────────────────────────────────────────

_HUNT_CLUSTER_CACHE = None
_CLUSTER_RV_MEMBER_CACHE = {}
DEFAULT_CLUSTER_RV_DISPERSION_KMS = 2.0


def _as_finite_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _truthy(value) -> bool:
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


def _rv_match_quality(
    rv_diff: float,
    rv_sigma: float,
    rv_err: float | None = None,
    *,
    strict_rv_diff: float = 20.0,
    strict_rv_nsigma: float = 2.0,
    strict_rv_err: float = 20.0,
    loose_rv_nsigma: float = 3.0,
) -> str:
    """Classify RV agreement so uncertainty alone cannot create a secure match."""
    if not np.isfinite(rv_diff) or not np.isfinite(rv_sigma):
        return 'unknown'
    if rv_err is None:
        rv_err = np.nan
    rv_err_ok = np.isfinite(rv_err) and rv_err <= strict_rv_err
    if rv_diff <= strict_rv_diff and rv_sigma < strict_rv_nsigma:
        return 'strict' if rv_err_ok else 'borderline'
    if rv_sigma < loose_rv_nsigma:
        return 'borderline'
    return 'mismatch'


def _cluster_match_key(name: str) -> str:
    return str(name or '').replace('_', ' ').strip().lower()


def _load_default_cluster_cache():
    """Load Hunt+2023 clusters once for callers that did not pass a cache."""
    global _HUNT_CLUSTER_CACHE
    if _HUNT_CLUSTER_CACHE is not None:
        return _HUNT_CLUSTER_CACHE
    try:
        try:
            from .orbit_traceback import load_hunt2023_clusters
        except Exception:
            from astro_toolbox.orbit_traceback import load_hunt2023_clusters
        _HUNT_CLUSTER_CACHE = load_hunt2023_clusters()
    except Exception:
        _HUNT_CLUSTER_CACHE = []
    return _HUNT_CLUSTER_CACHE


def _find_cluster_record(cluster_name: str, cluster_cache):
    """Find a cluster record by exact normalized name, then by loose contains."""
    target = _cluster_match_key(cluster_name)
    if not target or not cluster_cache:
        return None
    for cl in cluster_cache:
        if _cluster_match_key(cl.get('Name', '')) == target:
            return cl
    for cl in cluster_cache:
        key = _cluster_match_key(cl.get('Name', ''))
        if target in key or key in target:
            return cl
    return None


def _estimate_cluster_rv_from_members(cluster_name: str, prob_min: float = 0.5):
    """
    Conservative fallback cluster RV from Hunt+2023 individual-oc members.

    Some Hunt cluster-table entries have an empty RV even though one or a few
    high-probability member stars have Gaia RVS velocities in individual-oc.
    Use a median RV and a deliberately conservative error floor so one member
    cannot make the 6D test over-confident.
    """
    key = _cluster_match_key(cluster_name)
    if key in _CLUSTER_RV_MEMBER_CACHE:
        return _CLUSTER_RV_MEMBER_CACHE[key]

    members = _load_hunt2023_members(cluster_name, prob_min=prob_min)
    if members is None or len(members) == 0:
        out = (np.nan, np.nan, 0)
        _CLUSTER_RV_MEMBER_CACHE[key] = out
        return out

    cols = {str(c).lower(): c for c in members.columns}
    rv_col = cols.get('rv') or cols.get('radial_velocity') or cols.get('gaia_rv')
    err_col = cols.get('e_rv') or cols.get('rv_error') or cols.get('gaia_rv_err')
    if rv_col is None:
        out = (np.nan, np.nan, 0)
        _CLUSTER_RV_MEMBER_CACHE[key] = out
        return out

    rv = pd.to_numeric(members[rv_col], errors='coerce').to_numpy(float)
    good = np.isfinite(rv)
    if not np.any(good):
        out = (np.nan, np.nan, 0)
        _CLUSTER_RV_MEMBER_CACHE[key] = out
        return out

    rv = rv[good]
    n_rv = int(len(rv))
    cl_rv = float(np.nanmedian(rv))

    med_meas_err = np.nan
    if err_col is not None:
        err = pd.to_numeric(members[err_col], errors='coerce').to_numpy(float)[good]
        if np.any(np.isfinite(err) & (err > 0)):
            med_meas_err = float(np.nanmedian(err[np.isfinite(err) & (err > 0)]))

    if n_rv >= 2:
        mad = float(np.nanmedian(np.abs(rv - cl_rv)))
        robust_scatter = 1.4826 * mad if np.isfinite(mad) else np.nan
        terms = []
        if np.isfinite(robust_scatter) and robust_scatter > 0:
            terms.append((robust_scatter / np.sqrt(n_rv)) ** 2)
        if np.isfinite(med_meas_err) and med_meas_err > 0:
            terms.append((med_meas_err / np.sqrt(n_rv)) ** 2)
        cl_rv_err = float(np.sqrt(np.sum(terms))) if terms else np.nan
    else:
        cl_rv_err = med_meas_err

    if not np.isfinite(cl_rv_err) or cl_rv_err <= 0:
        cl_rv_err = 10.0
    cl_rv_err = max(float(cl_rv_err), 5.0)

    out = (cl_rv, cl_rv_err, n_rv)
    _CLUSTER_RV_MEMBER_CACHE[key] = out
    return out


def _resolve_cluster_rv(cluster_name: str, cluster_cache=None):
    """
    Return (rv, rv_err, source, n_rv) for a cluster.

    The primary source is Hunt+2023 clusters.dat. If that entry has no finite
    RV, fall back to high-probability individual-oc members when available.
    """
    if cluster_cache is None:
        cluster_cache = _load_default_cluster_cache()

    record = _find_cluster_record(cluster_name, cluster_cache)
    if record is not None:
        cl_rv = _as_finite_float(record.get('RV'))
        cl_rv_err = _as_finite_float(record.get('e_RV'))
        n_rv = int(_as_finite_float(record.get('n_RV'), 0))
        if np.isfinite(cl_rv):
            return cl_rv, cl_rv_err, 'Hunt+2023 cluster table', n_rv
        member_name = record.get('Name', cluster_name)
    else:
        member_name = cluster_name

    cl_rv, cl_rv_err, n_rv = _estimate_cluster_rv_from_members(member_name)
    if np.isfinite(cl_rv):
        return cl_rv, cl_rv_err, 'Hunt+2023 individual-oc members', n_rv
    return np.nan, np.nan, '', 0


def _row_get(row, key, default=np.nan):
    try:
        return row.get(key, default)
    except AttributeError:
        try:
            return row[key]
        except Exception:
            return default


def _row_num(row, *keys, default=np.nan) -> float:
    for key in keys:
        value = _row_get(row, key, np.nan)
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            return value
    return default


def _row_text(row, *keys, default='') -> str:
    for key in keys:
        value = _row_get(row, key, '')
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text and text.lower() not in {'nan', 'none', 'null'}:
            return text
    return default


def _fmt_num(value, fmt='.2f', unit='') -> str:
    try:
        value = float(value)
    except Exception:
        return 'N/A'
    if not np.isfinite(value):
        return 'N/A'
    text = f'{value:{fmt}}'
    return f'{text} {unit}'.strip()


def _fmt_pm(value, err=np.nan, fmt='.2f', unit='') -> str:
    text = _fmt_num(value, fmt, unit)
    if text == 'N/A':
        return text
    try:
        err = float(err)
    except Exception:
        err = np.nan
    if np.isfinite(err):
        return text + ' +/- ' + _fmt_num(err, fmt, unit)
    return text


def _signed_fmt(value, fmt='.2f', unit='') -> str:
    try:
        value = float(value)
    except Exception:
        return 'N/A'
    if not np.isfinite(value):
        return 'N/A'
    text = f'{value:+{fmt}}'
    return f'{text} {unit}'.strip()


def _cluster_type_from_cache(cluster_name: str) -> str:
    if not cluster_name:
        return ''
    record = _find_cluster_record(cluster_name, _load_default_cluster_cache())
    return str((record or {}).get('Type', '') or '')


def _cluster_age_from_cache(cluster_name: str) -> float:
    if not cluster_name:
        return np.nan
    record = _find_cluster_record(cluster_name, _load_default_cluster_cache())
    log_age = _as_finite_float((record or {}).get('logAge50'))
    return 10 ** log_age / 1e9 if np.isfinite(log_age) else np.nan


def evaluate_final_5d_rv_gates(row: dict) -> dict:
    """
    Literature-calibrated, plot-facing gate evaluation used by the sandbox
    WD 6D products.  It separates three 5D panels from the RV gate so a source
    cannot pass 6D just because RV is broad.
    """
    cluster = _row_text(row, 'sixdim_plot_cluster', 'cluster', 'Cluster')
    cl_info = _load_kinematic_cluster_info(cluster) if cluster else None
    out = {
        'cluster': cluster,
        'cluster_type': _row_text(row, 'sixdim_cluster_type', 'cluster_type')
                        or _cluster_type_from_cache(cluster),
        'cluster_age_gyr': _row_num(row, 'cluster_age_gyr',
                                    default=_cluster_age_from_cache(cluster)),
        'panel_sigma_limit': 3.0,
        'rv_diff_limit_kms': 50.0,
        'rv_sigma_limit': 3.0,
        'rv_true_err_limit_kms': 200.0,
        'cluster_intrinsic_rv_dispersion_kms': DEFAULT_CLUSTER_RV_DISPERSION_KMS,
    }

    ra = _row_num(row, 'ra', 'RA', 'RAdeg')
    dec = _row_num(row, 'dec', 'DEC', 'DEdeg')
    plx = _row_num(row, 'parallax', 'Plx', 'plx')
    e_plx = _row_num(row, 'e_parallax', 'parallax_error', 'e_Plx', 'e_plx',
                     default=np.nan)
    pmra = _row_num(row, 'pmRA', 'pmra', 'pmra_corr')
    pmde = _row_num(row, 'pmDE', 'pmdec', 'pmDE_corr', 'pmdec_corr')
    e_pmra = _row_num(row, 'e_pmRA', 'pmra_error', 'pmRA_error', 'e_pmra',
                      default=np.nan)
    e_pmde = _row_num(row, 'e_pmDE', 'pmdec_error', 'pmDE_error', 'e_pmdec',
                      default=np.nan)

    def _cl_num(*names):
        if cl_info is None:
            return np.nan
        for name in names:
            value = _as_finite_float(cl_info.get(name))
            if np.isfinite(value):
                return value
        return np.nan

    cl_ra = _row_num(row, 'cluster_ra', 'cluster_RA', 'cl_ra',
                     default=_cl_num('RA', 'ra'))
    cl_dec = _row_num(row, 'cluster_dec', 'cluster_DEC', 'cl_dec',
                      default=_cl_num('DEC', 'Dec', 'dec'))
    cl_pmra = _row_num(row, 'cluster_pmRA', 'cluster_pmra', 'cl_pmRA',
                       default=_cl_num('pmRA', 'pmra'))
    cl_pmde = _row_num(row, 'cluster_pmDE', 'cluster_pmdec', 'cl_pmDE',
                       default=_cl_num('pmDE', 'pmdec'))
    cl_plx = _row_num(row, 'cluster_parallax', 'cluster_Plx', 'cl_plx',
                      default=_cl_num('Plx', 'parallax'))
    cl_dist = _row_num(row, 'cluster_dist_pc', 'cluster_dist50', 'cl_dist_pc',
                       'dist50', default=_cl_num('dist50', 'distance_pc'))
    rtpc = _row_num(row, 'cluster_rtpc', 'tidal_radius_pc', 'rtpc',
                    'cluster_tidal_radius_pc', default=_cl_num('rtpc'))
    if not np.isfinite(cl_dist) and np.isfinite(cl_plx) and cl_plx > 0:
        cl_dist = 1000.0 / cl_plx
    if not np.isfinite(cl_plx) and np.isfinite(cl_dist) and cl_dist > 0:
        cl_plx = 1000.0 / cl_dist
    rt_deg = _row_num(row, 'spatial_rt_deg', 'rt_deg',
                      default=_cl_num('rt_deg'))
    if not np.isfinite(rt_deg) and np.isfinite(rtpc + cl_dist) and cl_dist > 0:
        rt_deg = float(np.degrees(np.arctan2(rtpc, cl_dist)))

    s_pmra = _row_num(row, 'sigma_pmra', 'cluster_s_pmRA', 's_pmRA',
                      default=_cl_num('s_pmRA', 'pmRA_sigma'))
    s_pmde = _row_num(row, 'sigma_pmdec', 'cluster_s_pmDE', 's_pmDE',
                      default=_cl_num('s_pmDE', 'pmDE_sigma', 'pmdec_sigma'))
    s_plx = _row_num(row, 'sigma_parallax', 'cluster_s_Plx', 's_Plx',
                     default=_cl_num('s_Plx', 'Plx_sigma', 'parallax_sigma'))
    if not np.isfinite(s_pmra) or s_pmra <= 0:
        s_pmra = 0.15
    if not np.isfinite(s_pmde) or s_pmde <= 0:
        s_pmde = 0.15
    if not np.isfinite(s_plx) or s_plx <= 0:
        s_plx = 0.05

    if np.isfinite(ra + dec + cl_ra + cl_dec):
        spatial_sep_deg = float(np.hypot((ra - cl_ra) * np.cos(np.radians(cl_dec)),
                                         dec - cl_dec))
    else:
        spatial_sep_deg = _row_num(row, 'spatial_sep_deg')
    spatial_ratio = (
        spatial_sep_deg / rt_deg
        if np.isfinite(spatial_sep_deg + rt_deg) and rt_deg > 0 else np.nan
    )

    dpmra = _row_num(row, 'dpmra', default=pmra - cl_pmra)
    dpmde = _row_num(row, 'dpmdec', default=pmde - cl_pmde)
    dplx = _row_num(row, 'dparallax', default=plx - cl_plx)
    dpmra_sigma = _row_num(row, 'dpmra_sigma_vs_sixdim_cluster',
                           'dpmra_sigma',
                           default=dpmra / s_pmra if s_pmra > 0 else np.nan)
    dpmde_sigma = _row_num(row, 'dpmdec_sigma_vs_sixdim_cluster',
                           'dpmdec_sigma',
                           default=dpmde / s_pmde if s_pmde > 0 else np.nan)
    dplx_sigma = _row_num(row, 'dparallax_sigma_vs_sixdim_cluster',
                          'dparallax_sigma',
                          default=dplx / s_plx if s_plx > 0 else np.nan)
    pm_chi2_2d = _row_num(row, 'pm_chi2_2d',
                          default=dpmra_sigma**2 + dpmde_sigma**2
                          if np.isfinite(dpmra_sigma + dpmde_sigma) else np.nan)
    pm_sigma = _row_num(row, 'pm_sigma_equiv',
                        default=np.sqrt(pm_chi2_2d)
                        if np.isfinite(pm_chi2_2d) else np.nan)
    chi2_5d = _row_num(row, 'chi2_5d_recomputed', 'chi2_5d', 'chi2_kin',
                       default=(pm_chi2_2d + dplx_sigma**2)
                       if np.isfinite(pm_chi2_2d + dplx_sigma) else np.nan)

    spatial_ok = (
        bool(_truthy(_row_get(row, 'spatial_panel_ok', False)))
        if str(_row_get(row, 'spatial_panel_ok', '')).strip().lower()
        not in {'', 'nan', 'none'}
        else np.isfinite(spatial_ratio) and spatial_ratio <= 1.0
    )
    pm_ok = (
        bool(_truthy(_row_get(row, 'proper_motion_panel_ok', False)))
        if str(_row_get(row, 'proper_motion_panel_ok', '')).strip().lower()
        not in {'', 'nan', 'none'}
        else np.isfinite(pm_sigma) and pm_sigma <= out['panel_sigma_limit']
    )
    plx_ok = (
        bool(_truthy(_row_get(row, 'parallax_panel_ok', False)))
        if str(_row_get(row, 'parallax_panel_ok', '')).strip().lower()
        not in {'', 'nan', 'none'}
        else np.isfinite(dplx_sigma) and abs(dplx_sigma) <= out['panel_sigma_limit']
    )
    all_5d = bool(spatial_ok and pm_ok and plx_ok)

    rv_obs = _row_num(row, 'rv_obs_adopted', 'hahbhg_rv_obs', 'rv_obs')
    rv_obs_err = _row_num(row, 'rv_obs_err_adopted', 'hahbhg_rv_obs_err',
                          'rv_obs_err', 'rv_true_random_err')
    vgrav = _row_num(row, 'v_grav_adopted', 'wdopt_v_grav_kms',
                     'hahbhg_v_grav', 'v_grav', 'vgrav_used')
    vgrav_err = _row_num(row, 'v_grav_err_adopted', 'wdopt_v_grav_err_kms',
                         'hahbhg_v_grav_err', 'v_grav_err',
                         'rv_true_grav_err', 'vgrav_err_reported')
    grav_floor = (
        max(10.0, 0.20 * abs(vgrav)) if np.isfinite(vgrav) else 10.0
    )
    grav_term = max(vgrav_err if np.isfinite(vgrav_err) else 0.0, grav_floor)
    rv_true = _row_num(row, 'rv_true_adopted', 'wdopt_rv_true_kms',
                       'hahbhg_rv_true', 'rv_true')
    if not np.isfinite(rv_true) and np.isfinite(rv_obs + vgrav):
        rv_true = rv_obs - vgrav
    rv_true_err = _row_num(row, 'rv_true_err_adopted',
                           'wdopt_rv_true_err_opt',
                           'wdopt_rv_true_err_kms',
                           'hahbhg_rv_true_err_with_floor',
                           'rv_true_err_with_grav_floor',
                           'rv_true_err_conservative_6d',
                           'rv_true_err')
    if np.isfinite(rv_obs_err):
        rv_true_err = float(np.hypot(rv_obs_err, grav_term))
    cl_rv = _row_num(row, 'hahbhg_cluster_rv', 'sixdim_cluster_rv',
                     'cluster_rv')
    cl_rv_err = _row_num(row, 'hahbhg_cluster_rv_err', 'cluster_rv_err')
    if not np.isfinite(cl_rv):
        cl_rv, cl_rv_err_resolved, _, _ = _resolve_cluster_rv(cluster)
        if not np.isfinite(cl_rv_err):
            cl_rv_err = cl_rv_err_resolved
    if not np.isfinite(cl_rv_err) or cl_rv_err <= 0:
        cl_rv_err = 5.0
    cluster_disp = _row_num(row, 'cluster_rv_dispersion', 'cluster_rv_sigma',
                            default=DEFAULT_CLUSTER_RV_DISPERSION_KMS)
    if not np.isfinite(cluster_disp) or cluster_disp < 0:
        cluster_disp = DEFAULT_CLUSTER_RV_DISPERSION_KMS
    delta_rv = _row_num(row, 'hahbhg_delta_rv_kms', 'sixdim_delta_rv_kms',
                        'rv_diff_kms')
    if np.isfinite(rv_true + cl_rv):
        delta_rv = rv_true - cl_rv
    denom = (
        np.sqrt(rv_true_err**2 + cl_rv_err**2 + cluster_disp**2)
        if np.isfinite(rv_true_err + cl_rv_err + cluster_disp) else np.nan
    )
    rv_sigma = (
        abs(delta_rv) / denom if np.isfinite(delta_rv + denom) and denom > 0
        else np.nan
    )
    rv_ok = (
        np.isfinite(delta_rv + rv_sigma + rv_true_err)
        and abs(delta_rv) <= out['rv_diff_limit_kms']
        and rv_sigma < out['rv_sigma_limit']
        and rv_true_err <= out['rv_true_err_limit_kms']
    )
    full_6d = bool(all_5d and rv_ok)

    out.update({
        'ra': ra,
        'dec': dec,
        'cluster_ra': cl_ra,
        'cluster_dec': cl_dec,
        'cluster_pmra': cl_pmra,
        'cluster_pmdec': cl_pmde,
        'cluster_parallax': cl_plx,
        'spatial_sep_deg': spatial_sep_deg,
        'spatial_rt_deg': rt_deg,
        'spatial_ratio': spatial_ratio,
        'spatial_panel_ok': bool(spatial_ok),
        'pmra': pmra,
        'pmdec': pmde,
        'dpmra': dpmra,
        'dpmdec': dpmde,
        'sigma_pmra': s_pmra,
        'sigma_pmdec': s_pmde,
        'dpmra_sigma': dpmra_sigma,
        'dpmdec_sigma': dpmde_sigma,
        'pm_chi2_2d': pm_chi2_2d,
        'pm_sigma_equiv': pm_sigma,
        'proper_motion_panel_ok': bool(pm_ok),
        'parallax': plx,
        'dparallax': dplx,
        'sigma_parallax': s_plx,
        'dparallax_sigma': dplx_sigma,
        'parallax_panel_ok': bool(plx_ok),
        'chi2_5d': chi2_5d,
        'all_three_5d_panels_ok': all_5d,
        'rv_obs': rv_obs,
        'rv_obs_err': rv_obs_err,
        'v_grav': vgrav,
        'v_grav_err': vgrav_err,
        'v_grav_err_floor_used': grav_term,
        'rv_true': rv_true,
        'rv_true_err': rv_true_err,
        'cluster_rv': cl_rv,
        'cluster_rv_err': cl_rv_err,
        'cluster_rv_dispersion': cluster_disp,
        'delta_rv_kms': delta_rv,
        'rv_sigma': rv_sigma,
        'rv_ok': bool(rv_ok),
        'full_6d_astrometry_rv_ok': full_6d,
        'final_category': (
            '01_full_6d_astrometry_rv' if full_6d
            else ('02_5d_without_rv' if all_5d else '03_not_5d')
        ),
    })
    return out


def _validation_barh(ax, names, values, limits, oks, labels):
    import numpy as _np
    y = _np.arange(len(names))
    finite_values = [abs(v) if _np.isfinite(v) else 0.0 for v in values]
    max_x = max([1.0] + finite_values + [l for l in limits if _np.isfinite(l)])
    colors = ['#2e7d32' if ok else '#c62828' for ok in oks]
    ax.barh(y, finite_values, color=colors, alpha=0.82, height=0.46)
    for idx, lim in enumerate(limits):
        if _np.isfinite(lim):
            ax.axvline(lim, color='#222', lw=1.0, ls='--', alpha=0.75)
            ax.text(lim, idx + 0.27, f'limit {lim:g}', fontsize=8,
                    ha='center', va='bottom')
    ax.set_yticks(y, names)
    ax.set_xlim(0, max(max_x * 1.25, 1.0))
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2)
    for idx, label in enumerate(labels):
        ax.text(finite_values[idx] + max_x * 0.03, idx, label,
                va='center', fontsize=8.5)


def _validation_summary_table(ax, rows, title):
    ax.axis('off')
    ax.text(0.0, 1.0, title, ha='left', va='top',
            fontsize=13, fontweight='bold')
    y = 0.88
    for key, value, ok in rows:
        if ok is None:
            color = '#222'
            state = ''
        else:
            color = '#1b7f43' if ok else '#b42318'
            state = 'PASS' if ok else 'FAIL'
        ax.text(0.0, y, key, ha='left', va='center',
                fontsize=9.2, color='#333')
        ax.text(0.42, y, str(value), ha='left', va='center',
                fontsize=9.2, color='#111')
        if state:
            ax.text(0.96, y, state, ha='right', va='center',
                    fontsize=9.2, color=color, fontweight='bold')
        y -= 0.085


def plot_final_5d_validation(row, out_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    g = evaluate_final_5d_rv_gates(row)
    fig = plt.figure(figsize=(12, 7.5), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08],
                          hspace=0.55, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    source_id = _row_text(row, 'source_id', 'source')
    title = (
        f"Final 5D Astrometry Check | Gaia DR3 {source_id} | "
        f"{g['cluster'] or 'no cluster'}"
    )
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    rows = [
        ('Cluster used', g['cluster'] or 'N/A', None),
        ('Cluster type', g['cluster_type'] or 'N/A', None),
        ('Spatial',
         f"{_fmt_num(g['spatial_sep_deg'], '.3f', 'deg')} / rt={_fmt_num(g['spatial_rt_deg'], '.3f', 'deg')}",
         g['spatial_panel_ok']),
        ('Proper motion',
         f"chi2^0.5={_fmt_num(g['pm_sigma_equiv'], '.2f')} <= {g['panel_sigma_limit']:g}",
         g['proper_motion_panel_ok']),
        ('Parallax',
         f"|Delta pi|/sigma={_fmt_num(abs(g['dparallax_sigma']), '.2f')} <= {g['panel_sigma_limit']:g}",
         g['parallax_panel_ok']),
        ('All 5D panels', str(g['all_three_5d_panels_ok']),
         g['all_three_5d_panels_ok']),
        ('Moving group type m', str(g['cluster_type']).lower() == 'm', None),
    ]
    _validation_summary_table(ax0, rows, 'Gate Summary')

    _validation_barh(
        ax1,
        ['spatial sep/rt', 'PM sigma', 'parallax sigma'],
        [g['spatial_ratio'], g['pm_sigma_equiv'], abs(g['dparallax_sigma'])],
        [1.0, g['panel_sigma_limit'], g['panel_sigma_limit']],
        [g['spatial_panel_ok'], g['proper_motion_panel_ok'],
         g['parallax_panel_ok']],
        [_fmt_num(g['spatial_ratio'], '.2f'),
         _fmt_num(g['pm_sigma_equiv'], '.2f'),
         _fmt_num(abs(g['dparallax_sigma']), '.2f')],
    )
    ax1.set_title('5D Gate Metrics')

    ax2.axhline(0, color='#444', lw=1)
    ax2.axvline(0, color='#444', lw=1)
    if np.isfinite(g['dpmra_sigma'] + g['dpmdec_sigma']):
        ax2.plot([g['dpmra_sigma']], [g['dpmdec_sigma']], 'o',
                 color='#1565c0', label='star - cluster')
    lim = max(g['panel_sigma_limit'] * 1.25,
              abs(g['dpmra_sigma']) * 1.25 if np.isfinite(g['dpmra_sigma']) else 0,
              abs(g['dpmdec_sigma']) * 1.25 if np.isfinite(g['dpmdec_sigma']) else 0)
    ax2.add_patch(plt.Circle((0, 0), g['panel_sigma_limit'],
                             transform=ax2.transData, fill=False,
                             color='#111', lw=1.2))
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('Delta pmRA / cluster sigma')
    ax2.set_ylabel('Delta pmDE / cluster sigma')
    ax2.set_title(
        'Proper Motion Offset\n'
        f"Delta pmRA/sigma={_signed_fmt(g['dpmra_sigma'], '.2f')}, "
        f"Delta pmDE/sigma={_signed_fmt(g['dpmdec_sigma'], '.2f')}\n"
        f"raw Delta=({_fmt_num(g['dpmra'], '.2f')}, {_fmt_num(g['dpmdec'], '.2f')}) mas/yr; "
        f"sigma=({_fmt_num(g['sigma_pmra'], '.2f')}, {_fmt_num(g['sigma_pmdec'], '.2f')}) mas/yr",
        fontsize=10,
        pad=8,
    )
    ax2.grid(alpha=0.25)
    ax2.legend(loc='best', fontsize=8)

    if np.isfinite(g['parallax']):
        yerr = g['sigma_parallax'] if np.isfinite(g['sigma_parallax']) else None
        ax3.errorbar([0], [g['parallax']], yerr=yerr, fmt='o',
                     color='#6a1b9a', ecolor='#ce93d8', capsize=4,
                     label='source')
    if np.isfinite(g['cluster_parallax']):
        ax3.axhline(g['cluster_parallax'], color='#333', lw=1.4,
                    label='cluster')
    if np.isfinite(g['sigma_parallax'] + g['cluster_parallax']):
        lim3 = g['panel_sigma_limit'] * g['sigma_parallax']
        ax3.axhspan(g['cluster_parallax'] - lim3,
                    g['cluster_parallax'] + lim3,
                    color='#6a1b9a', alpha=0.10, label='+/-3 sigma cluster')
    ax3.set_xticks([])
    ax3.set_ylabel('Parallax (mas)')
    ax3.set_title(
        f"Parallax: source={_fmt_num(g['parallax'], '.3f', 'mas')}, "
        f"cluster={_fmt_num(g['cluster_parallax'], '.3f', 'mas')}, "
        f"sigma={_fmt_num(g['sigma_parallax'], '.3f', 'mas')}",
        fontsize=10,
        pad=8,
    )
    ax3.grid(axis='y', alpha=0.25)
    ax3.legend(loc='best', fontsize=8)

    fig.text(
        0.02, 0.015,
        '5D rule: spatial sep <= Hunt tidal radius; PM chi2^0.5 <= 3; '
        '|Delta parallax|/cluster_sigma <= 3.  Type=m moving groups are '
        'shown explicitly and are not automatically promoted by the plot.',
        fontsize=8.5, color='#444',
    )
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_final_rv_validation(row, out_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    g = evaluate_final_5d_rv_gates(row)
    fig = plt.figure(figsize=(12, 7.5), dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1],
                          hspace=0.38, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    source_id = _row_text(row, 'source_id', 'source')
    fig.suptitle(
        f"Final 6D RV Check | Gaia DR3 {source_id} | {g['cluster'] or 'no cluster'}",
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    selected = _row_text(row, 'hahbhg_rv_selected_lines', 'preferred_lines',
                         'rv_selected_lines', default='H-alpha;H-beta;H-gamma')
    rejected = _row_text(row, 'hahbhg_rv_rejected_lines',
                         'rv_rejected_lines', default='')
    rows = [
        ('RV source',
         f"{_row_text(row, 'hahbhg_survey', 'rv_true_source')} lines={selected}",
         None),
        ('Rejected lines', rejected or 'none', None),
        ('RV_obs', _fmt_pm(g['rv_obs'], g['rv_obs_err'], '.2f', 'km/s'), None),
        ('V_grav formal', _fmt_pm(g['v_grav'], g['v_grav_err'], '.2f', 'km/s'), None),
        ('V_grav err used',
         _fmt_num(g['v_grav_err_floor_used'], '.2f', 'km/s'), None),
        ('RV_true', _fmt_pm(g['rv_true'], g['rv_true_err'], '.2f', 'km/s'), None),
        ('Cluster RV', _fmt_pm(g['cluster_rv'], g['cluster_rv_err'], '.2f', 'km/s'), None),
        ('Delta RV',
         f"{_signed_fmt(g['delta_rv_kms'], '.2f', 'km/s')} <= {g['rv_diff_limit_kms']:g}",
         np.isfinite(g['delta_rv_kms']) and abs(g['delta_rv_kms']) <= g['rv_diff_limit_kms']),
        ('RV sigma',
         f"{_fmt_num(g['rv_sigma'], '.2f')} < {g['rv_sigma_limit']:g}",
         np.isfinite(g['rv_sigma']) and g['rv_sigma'] < g['rv_sigma_limit']),
        ('RV_true err',
         f"{_fmt_num(g['rv_true_err'], '.2f', 'km/s')} <= {g['rv_true_err_limit_kms']:g}",
         np.isfinite(g['rv_true_err']) and g['rv_true_err'] <= g['rv_true_err_limit_kms']),
    ]
    _validation_summary_table(ax0, rows, 'RV Gate Summary')

    _validation_barh(
        ax1,
        ['|Delta RV|', 'RV sigma', 'RV_true err'],
        [abs(g['delta_rv_kms']), g['rv_sigma'], g['rv_true_err']],
        [g['rv_diff_limit_kms'], g['rv_sigma_limit'],
         g['rv_true_err_limit_kms']],
        [np.isfinite(g['delta_rv_kms']) and abs(g['delta_rv_kms']) <= g['rv_diff_limit_kms'],
         np.isfinite(g['rv_sigma']) and g['rv_sigma'] < g['rv_sigma_limit'],
         np.isfinite(g['rv_true_err']) and g['rv_true_err'] <= g['rv_true_err_limit_kms']],
        [_fmt_num(abs(g['delta_rv_kms']), '.1f', 'km/s'),
         _fmt_num(g['rv_sigma'], '.2f'),
         _fmt_num(g['rv_true_err'], '.1f', 'km/s')],
    )
    ax1.set_title('RV Gate Metrics')

    y = [2, 1, 0]
    vals = [g['rv_obs'], g['rv_true'], g['cluster_rv']]
    errs = [g['rv_obs_err'], g['rv_true_err'], g['cluster_rv_err']]
    names = ['RV_obs', 'RV_true', 'Cluster RV']
    colors = ['#1565c0', '#2e7d32', '#333333']
    finite_vals = []
    for yy, val, err, name, color in zip(y, vals, errs, names, colors):
        if not np.isfinite(val):
            continue
        finite_vals.append(val)
        ax2.errorbar(val, yy, xerr=err if np.isfinite(err) else None,
                     fmt='o', color=color, ecolor=color, alpha=0.9,
                     capsize=4)
        ax2.text(val, yy + 0.16, f'{name}: {_fmt_pm(val, err, ".1f", "km/s")}',
                 ha='center', fontsize=9)
    ax2.set_yticks(y, names)
    ax2.set_xlabel('Radial velocity (km/s)')
    ax2.set_title('Observed RV, Gravitational-Redshift Correction, and Cluster RV')
    ax2.grid(axis='x', alpha=0.25)
    if not finite_vals:
        ax2.text(0.5, 0.5, 'No reliable RV values',
                 transform=ax2.transAxes, ha='center', va='center')

    fig.text(
        0.02, 0.015,
        'RV formula: RV_true = RV_obs - V_grav.  In HaHbHg mode RV_obs is '
        'fitted only from H-alpha/H-beta/H-gamma line cores; discrepant or bad '
        'lines are rejected.  RV_true_err = sqrt(RV_obs_err^2 + '
        'max(V_grav_err, 10 km/s, 0.2|V_grav|)^2).  RV sigma denominator also '
        f"includes cluster_RV_err and {DEFAULT_CLUSTER_RV_DISPERSION_KMS:g} km/s intrinsic dispersion.",
        fontsize=8.4, color='#444',
    )
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_final_6d_validation(row, out_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    g = evaluate_final_5d_rv_gates(row)
    fig = plt.figure(figsize=(13.2, 9.0), dpi=180)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.85, 1.0, 1.0],
                          hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    source_id = _row_text(row, 'source_id', 'source')
    ok = g['full_6d_astrometry_rv_ok']
    title_color = '#1b7f43' if ok else '#b42318'
    ax0.axis('off')
    ax0.text(0.0, 0.98,
             f"Final 6D Certification: {'PASS' if ok else 'FAIL'}",
             fontsize=15, fontweight='bold', color=title_color, va='top')
    ax0.text(
        0.0, 0.70,
        f"Gaia DR3 {source_id}  |  cluster={g['cluster']}  "
        f"| type={g['cluster_type'] or 'N/A'}  |  Hunt age={_fmt_num(g['cluster_age_gyr'], '.3f', 'Gyr')}",
        fontsize=10.5, color='#222')
    ax0.text(
        0.0, 0.46,
        f"WD cooling age={_fmt_num(_row_num(row, 'final_wd_cooling_age_gyr', 'cooling_age_gyr'), '.3f', 'Gyr')}  "
        f"| mass={_fmt_num(_row_num(row, 'hahbhg_mass_msun', 'mass'), '.3f', 'Msun')}  "
        f"| selected lines={_row_text(row, 'hahbhg_rv_selected_lines', 'preferred_lines', 'rv_selected_lines')}",
        fontsize=10.5, color='#222')
    ax0.text(
        0.0, 0.20,
        'Final rule: spatial PASS + PM PASS + parallax PASS + '
        'Halpha/Hbeta/Hgamma RV PASS. Moving groups are flagged but not '
        'automatically promoted.',
        fontsize=9.5, color='#444')

    _validation_barh(
        ax1,
        ['spatial sep/rt', 'PM sigma', 'parallax sigma'],
        [g['spatial_ratio'], g['pm_sigma_equiv'], abs(g['dparallax_sigma'])],
        [1.0, g['panel_sigma_limit'], g['panel_sigma_limit']],
        [g['spatial_panel_ok'], g['proper_motion_panel_ok'],
         g['parallax_panel_ok']],
        [_fmt_num(g['spatial_ratio'], '.2f'),
         _fmt_num(g['pm_sigma_equiv'], '.2f'),
         _fmt_num(abs(g['dparallax_sigma']), '.2f')],
    )
    ax1.set_title('5D Astrometry Gates')

    _validation_barh(
        ax2,
        ['|Delta RV|', 'RV sigma', 'RV_true err'],
        [abs(g['delta_rv_kms']), g['rv_sigma'], g['rv_true_err']],
        [g['rv_diff_limit_kms'], g['rv_sigma_limit'],
         g['rv_true_err_limit_kms']],
        [np.isfinite(g['delta_rv_kms']) and abs(g['delta_rv_kms']) <= g['rv_diff_limit_kms'],
         np.isfinite(g['rv_sigma']) and g['rv_sigma'] < g['rv_sigma_limit'],
         np.isfinite(g['rv_true_err']) and g['rv_true_err'] <= g['rv_true_err_limit_kms']],
        [_fmt_num(abs(g['delta_rv_kms']), '.1f', 'km/s'),
         _fmt_num(g['rv_sigma'], '.2f'),
         _fmt_num(g['rv_true_err'], '.1f', 'km/s')],
    )
    ax2.set_title('6D RV Gates')

    ax3.axis('off')
    ax3.text(0, 1, 'RV Values With Errors',
             fontsize=12, fontweight='bold', va='top')
    text_rows = [
        ('RV_obs', _fmt_pm(g['rv_obs'], g['rv_obs_err'], '.2f', 'km/s')),
        ('V_grav', _fmt_pm(g['v_grav'], g['v_grav_err'], '.2f', 'km/s')),
        ('V_grav err used', _fmt_num(g['v_grav_err_floor_used'], '.2f', 'km/s')),
        ('RV_true', _fmt_pm(g['rv_true'], g['rv_true_err'], '.2f', 'km/s')),
        ('Cluster RV', _fmt_pm(g['cluster_rv'], g['cluster_rv_err'], '.2f', 'km/s')),
        ('Delta RV', _signed_fmt(g['delta_rv_kms'], '.2f', 'km/s')),
        ('RV sigma', _fmt_num(g['rv_sigma'], '.2f')),
    ]
    y = 0.84
    for key, value in text_rows:
        ax3.text(0.0, y, key, fontsize=10, color='#333')
        ax3.text(0.36, y, value, fontsize=10, color='#111')
        y -= 0.105

    ax4.axis('off')
    method = [
        '5D: sep <= rt; PM chi2^0.5 <= 3; |Delta parallax|/sigma <= 3.',
        f"RV: |Delta RV| <= {g['rv_diff_limit_kms']:g} km/s; RV sigma < {g['rv_sigma_limit']:g}; RV_true_err <= {g['rv_true_err_limit_kms']:g} km/s.",
        'RV_obs: Halpha/Hbeta/Hgamma line cores in the optimized sandbox mode.',
        'RV_obs_err: formal line fit, accepted-line scatter, and systematic floor.',
        'V_grav = G M / (R c), subtracted from photospheric WD RV.',
        'V_grav error used: max(formal V_grav_err, 10 km/s, 0.2|V_grav|).',
        'RV_true_err includes the gravitational-redshift term; it can be larger than RV_obs_err.',
    ]
    ax4.text(0, 1, 'How This Plot Was Computed',
             fontsize=12, fontweight='bold', va='top')
    y = 0.84
    for line in method:
        ax4.text(0.0, y, f'- {line}', fontsize=9.0,
                 color='#333', va='top', wrap=True)
        y -= 0.12

    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def write_final_6d_validation_summary(row, out_path: str) -> None:
    g = evaluate_final_5d_rv_gates(row)
    lines = [
        f"source_id: {_row_text(row, 'source_id', 'source')}",
        f"cluster: {g['cluster']}",
        f"cluster_type: {g['cluster_type']}",
        f"final_6d_pass: {g['full_6d_astrometry_rv_ok']}",
        "",
        "5D screening:",
        f"  spatial: sep={_fmt_num(g['spatial_sep_deg'], '.6f', 'deg')}, rt={_fmt_num(g['spatial_rt_deg'], '.6f', 'deg')}, ratio={_fmt_num(g['spatial_ratio'], '.6f')}, pass={g['spatial_panel_ok']}",
        f"  PM: sqrt(chi2)={_fmt_num(g['pm_sigma_equiv'], '.4f')}, limit={g['panel_sigma_limit']:g}, dpmra/sigma={_signed_fmt(g['dpmra_sigma'], '.4f')}, dpmdec/sigma={_signed_fmt(g['dpmdec_sigma'], '.4f')}, pass={g['proper_motion_panel_ok']}",
        f"  parallax: dparallax/sigma={_signed_fmt(g['dparallax_sigma'], '.4f')}, limit={g['panel_sigma_limit']:g}, pass={g['parallax_panel_ok']}",
        f"  all_three_5d_panels_ok={g['all_three_5d_panels_ok']}",
        "",
        "6D RV screening:",
        f"  selected_lines={_row_text(row, 'hahbhg_rv_selected_lines', 'preferred_lines', 'rv_selected_lines')}",
        f"  rejected_lines={_row_text(row, 'hahbhg_rv_rejected_lines', 'rv_rejected_lines') or 'none'}",
        f"  RV_obs={_fmt_pm(g['rv_obs'], g['rv_obs_err'], '.6f', 'km/s')}",
        f"  V_grav={_fmt_pm(g['v_grav'], g['v_grav_err'], '.6f', 'km/s')}",
        f"  V_grav_err_used=max(formal, 10, 0.2*abs(V_grav))={_fmt_num(g['v_grav_err_floor_used'], '.6f', 'km/s')}",
        f"  RV_true=RV_obs-V_grav={_fmt_pm(g['rv_true'], g['rv_true_err'], '.6f', 'km/s')}",
        f"  cluster_RV={_fmt_pm(g['cluster_rv'], g['cluster_rv_err'], '.6f', 'km/s')}",
        f"  cluster_intrinsic_dispersion={g['cluster_rv_dispersion']:g} km/s",
        f"  delta_RV={_signed_fmt(g['delta_rv_kms'], '.6f', 'km/s')}",
        f"  rv_sigma={_fmt_num(g['rv_sigma'], '.6f')}",
        f"  rv_ok={g['rv_ok']}",
    ]
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


def plot_final_6d_validation_set(row, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = [
        os.path.join(output_dir, 'final_5d_validation.png'),
        os.path.join(output_dir, 'final_rv_validation.png'),
        os.path.join(output_dir, 'final_6d_validation.png'),
    ]
    plot_final_5d_validation(row, paths[0])
    plot_final_rv_validation(row, paths[1])
    plot_final_6d_validation(row, paths[2])
    summary = os.path.join(output_dir, 'final_6d_validation_summary.txt')
    write_final_6d_validation_summary(row, summary)
    return paths + [summary]


def check_6d_match(
    row: dict,
    rv_true: float,
    rv_err: float | None = None,
    *,
    max_rv_err: float = 50.0,
    max_rv_diff: float = 100.0,
    chi2_kin_limit: float = 20.0,
    rv_nsigma: float = 3.0,
    cluster_rv_dispersion_floor: float = DEFAULT_CLUSTER_RV_DISPERSION_KMS,
    cluster_cache=None,
) -> dict:
    """
    对单颗白矮星做 6D 匹配判断（两条路径）。

    路径A: 5D运动学通过 (chi2_kin < limit) + RV匹配
    路径B: 轨道回溯在潮汐半径内 (orbit_within_tidal=True) + RV匹配

    Parameters
    ----------
    row : dict-like
        包含 'cluster', 'chi2_kin', 'orbit_within_tidal' 字段。
    rv_true : float
        白矮星的真实视向速度 (km/s，已改正引力红移和大气位移)。
    rv_err : float, optional
        rv_true 的误差 (km/s)。
    max_rv_err : float
        rv_err 超过该值时直接排除，默认 50 km/s。
    max_rv_diff : float
        |RV_WD − RV_cluster| 超过该值时排除，默认 100 km/s。
    chi2_kin_limit : float
        chi2_kin 阈值，低于此值视为 5D 运动学匹配，默认 20。
    rv_nsigma : float
        |ΔRV| / σ_total 阈值，低于此值视为 RV 匹配，默认 3。
    cluster_rv_dispersion_floor : float
        星团成员本征 RV 弥散/未解析双星等造成的保守速度宽度项，默认
        2 km/s。它进入 σ_total，而不是作为独立通过条件。
    cluster_cache : list of dict, optional
        由 orbit_traceback.load_hunt2023_clusters() 返回的星团列表。
        每个元素包含 'Name', 'RV', 'e_RV' 键。

    Returns
    -------
    dict
        键: is_6d_matched, match_path, rv_diff_kms, rv_sigma, cluster_rv, match_note
    """
    result = {
        'is_6d_matched': False,
        'match_path': '',
        'rv_diff_kms': np.nan,
        'rv_sigma': np.nan,
        'cluster_rv': np.nan,
        'cluster_rv_err': np.nan,
        'cluster_rv_source': '',
        'cluster_rv_n': np.nan,
        'cluster_rv_dispersion': np.nan,
        'rv_match_quality': 'unknown',
        'is_6d_strict': False,
        'is_6d_borderline': False,
        'match_note': '',
    }

    rv_err_input = rv_err

    if rv_true is None or not np.isfinite(rv_true):
        result['match_note'] = 'no_rv_true'
        return result

    if rv_err is not None and np.isfinite(rv_err) and rv_err > max_rv_err:
        result['match_note'] = f'rv_err_too_large ({rv_err:.0f} km/s > {max_rv_err:.0f})'
        return result

    cluster_name = str(row.get('cluster', ''))
    if not cluster_name or cluster_name == 'nan':
        result['match_note'] = 'no_cluster'
        return result

    # 查星团 RV。旧入口常常没有显式传 cluster_cache，这里自动加载；
    # cluster table 无 RV 时，用 individual-oc 成员 RV 做保守兜底。
    cl_rv, cl_rv_err, cl_rv_source, cl_rv_n = _resolve_cluster_rv(
        cluster_name, cluster_cache=cluster_cache)

    if not np.isfinite(cl_rv):
        result['match_note'] = 'cluster_has_no_rv'
        return result

    result['cluster_rv'] = cl_rv
    result['cluster_rv_err'] = cl_rv_err
    result['cluster_rv_source'] = cl_rv_source
    result['cluster_rv_n'] = cl_rv_n
    cl_rv_disp = _as_finite_float(
        row.get('cluster_rv_dispersion',
                row.get('cluster_rv_sigma',
                        row.get('rv_cluster_sigma',
                                cluster_rv_dispersion_floor))),
        cluster_rv_dispersion_floor)
    if not np.isfinite(cl_rv_disp) or cl_rv_disp < 0:
        cl_rv_disp = cluster_rv_dispersion_floor
    result['cluster_rv_dispersion'] = cl_rv_disp

    dv = abs(rv_true - cl_rv)
    result['rv_diff_kms'] = dv

    if dv > max_rv_diff:
        result['match_note'] = f'rv_diff_too_large (dv={dv:.1f} km/s > {max_rv_diff:.0f})'
        return result

    if not np.isfinite(cl_rv_err) or cl_rv_err <= 0:
        cl_rv_err = 5.0
    if rv_err is None or not np.isfinite(rv_err) or rv_err <= 0:
        rv_err = 20.0
    sigma_total = np.sqrt(rv_err**2 + cl_rv_err**2 + cl_rv_disp**2)
    result['rv_sigma'] = dv / sigma_total if sigma_total > 0 else np.inf
    result['rv_match_quality'] = _rv_match_quality(
        dv, result['rv_sigma'], rv_err_input, loose_rv_nsigma=rv_nsigma)

    chi2_kin = row.get('chi2_kin', np.nan)
    if isinstance(chi2_kin, str):
        try:
            chi2_kin = float(chi2_kin)
        except ValueError:
            chi2_kin = np.nan

    tier = str(row.get('tier', '') or '').strip()
    membership = str(row.get('membership', '') or '').strip()
    has_5d_from_chi2 = np.isfinite(chi2_kin) and chi2_kin < chi2_kin_limit
    has_5d_from_legacy = (
        not np.isfinite(chi2_kin)
        and (
            tier.lower() == 'tier1'
            or membership.lower() in {'tier1_5d', '5d_matched'}
        )
    )
    has_5d = has_5d_from_chi2 or has_5d_from_legacy
    kin_desc = (
        f'chi2={chi2_kin:.1f}'
        if np.isfinite(chi2_kin)
        else f'legacy_5D={tier or membership or "unknown"}'
    )
    has_backtrack = (
        _truthy(row.get('orbit_within_tidal', False))
        or _truthy(row.get('orbit_confirmed', False))
        or membership.lower() in {'backtrack_matched', 'backtrack_only'}
    )
    rv_ok = result['rv_sigma'] < rv_nsigma
    rv_strict = result['rv_match_quality'] == 'strict'

    if has_5d and rv_ok:
        result['is_6d_matched'] = True
        result['is_6d_strict'] = rv_strict
        result['is_6d_borderline'] = not rv_strict
        result['match_path'] = '5D+RV'
        label = '6D matched' if rv_strict else '6D candidate'
        result['match_note'] = (
            f'{label} via 5D+RV ({kin_desc}, '
            f'dv={dv:.1f} km/s, {result["rv_sigma"]:.1f}σ)'
        )
    elif has_backtrack and rv_ok:
        result['is_6d_matched'] = True
        result['is_6d_strict'] = rv_strict
        result['is_6d_borderline'] = not rv_strict
        result['match_path'] = 'backtrack+RV'
        label = '6D matched' if rv_strict else '6D candidate'
        result['match_note'] = (
            f'{label} via backtrack+RV '
            f'(dv={dv:.1f} km/s, {result["rv_sigma"]:.1f}σ)'
        )
    elif has_5d and not rv_ok:
        result['match_note'] = (
            f'5D ok but RV mismatch (dv={dv:.1f} km/s, {result["rv_sigma"]:.1f}σ)'
        )
    elif has_backtrack and not rv_ok:
        result['match_note'] = (
            f'backtrack ok but RV mismatch (dv={dv:.1f} km/s, {result["rv_sigma"]:.1f}σ)'
        )
    else:
        chi2_str = f'{chi2_kin:.1f}' if np.isfinite(chi2_kin) else 'NaN'
        if rv_ok:
            result['match_note'] = (
                f'RV ok but 5D poor (chi2_kin={chi2_str}, tier={tier or "NA"})'
            )
        else:
            result['match_note'] = (
                f'both 5D and RV poor (chi2_kin={chi2_str}, tier={tier or "NA"})'
            )

    return result


def recalc_membership(row: dict) -> str:
    """根据 is_6d_matched / match_path / orbit_within_tidal / tier 重算 membership。"""
    if row.get('is_6d_strict'):
        return '6D_matched'
    if row.get('is_6d_matched') or row.get('is_6d_borderline'):
        return '6D_candidate'
    owt = row.get('orbit_within_tidal', False)
    if owt is True or owt == 'True':
        return 'backtrack_only'
    elif row.get('tier') == 'Tier1':
        return 'Tier1_5D'
    else:
        return 'unconfirmed'


def recalc_6d(
    df: pd.DataFrame,
    cluster_cache=None,
    *,
    max_rv_err: float = 100.0,
    max_rv_diff: float = 100.0,
    chi2_kin_limit: float = 20.0,
    rv_nsigma: float = 3.0,
    update_membership: bool = True,
) -> pd.DataFrame:
    """
    批量重算 DataFrame 中所有行的 6D 列。

    会在 df 上原地更新以下列并返回:
      is_6d_matched, rv_diff_kms, rv_sigma, cluster_rv, match_note
    若 update_membership=True 还会更新 membership 列。

    Parameters
    ----------
    df : pd.DataFrame
        需含 rv_true, rv_true_err, chi2_kin, cluster 列。
    cluster_cache : list of dict, optional
        Hunt+2023 星团列表，None 时自动尝试从 orbit_traceback 加载。
    """
    if cluster_cache is None:
        try:
            from astro_toolbox.orbit_traceback import load_hunt2023_clusters
            cluster_cache = load_hunt2023_clusters()
            print(f"[six_dim] Loaded {len(cluster_cache)} clusters from Hunt+2023")
        except Exception as e:
            raise RuntimeError(f"Cannot load cluster catalog: {e}") from e

    records = []
    for _, row in df.iterrows():
        rv_true = row.get('rv_true', np.nan)
        rv_err  = row.get('rv_true_err', np.nan)
        res = check_6d_match(
            row, rv_true, rv_err,
            max_rv_err=max_rv_err,
            max_rv_diff=max_rv_diff,
            chi2_kin_limit=chi2_kin_limit,
            rv_nsigma=rv_nsigma,
            cluster_cache=cluster_cache,
        )
        records.append(res)

    res_df = pd.DataFrame(records)
    df = df.copy()
    df['is_6d_matched'] = res_df['is_6d_matched'].values
    df['match_path']    = res_df['match_path'].values
    df['rv_diff_kms']   = res_df['rv_diff_kms'].values
    df['rv_sigma']      = res_df['rv_sigma'].values
    df['cluster_rv']    = res_df['cluster_rv'].values
    if 'cluster_rv_err' in res_df:
        df['cluster_rv_err'] = res_df['cluster_rv_err'].values
    if 'cluster_rv_source' in res_df:
        df['cluster_rv_source'] = res_df['cluster_rv_source'].values
    if 'cluster_rv_n' in res_df:
        df['cluster_rv_n'] = res_df['cluster_rv_n'].values
    if 'cluster_rv_dispersion' in res_df:
        df['cluster_rv_dispersion'] = res_df['cluster_rv_dispersion'].values
    if 'rv_match_quality' in res_df:
        df['rv_match_quality'] = res_df['rv_match_quality'].values
    if 'is_6d_strict' in res_df:
        df['is_6d_strict'] = res_df['is_6d_strict'].values
    if 'is_6d_borderline' in res_df:
        df['is_6d_borderline'] = res_df['is_6d_borderline'].values
    df['match_note']    = res_df['match_note'].values

    if update_membership:
        df['membership'] = df.apply(recalc_membership, axis=1)

    df = apply_dynamics_flags(df)

    # 统计输出
    n6d = df['is_6d_matched'].sum()
    n_5d_rv = (df['match_path'] == '5D+RV').sum()
    n_bt_rv = (df['match_path'] == 'backtrack+RV').sum()
    print(f"[six_dim] 6D matched: {n6d}  (5D+RV: {n_5d_rv}, backtrack+RV: {n_bt_rv})")

    # DWD 候选: cooling_age > cluster_age
    if 'cooling_age_gyr' in df.columns and 'cluster_age_gyr' in df.columns:
        mask_6d = df['is_6d_matched'] == True
        mask_age = (df['cooling_age_gyr'].notna() & df['cluster_age_gyr'].notna()
                    & (df['cooling_age_gyr'] > df['cluster_age_gyr']))
        dwd_cands = df[mask_6d & mask_age].drop_duplicates(subset=['source_id'] if 'source_id' in df.columns else ['ra', 'dec'])
        print(f"[six_dim] DWD candidates (cooling_age > cluster_age among 6D): {len(dwd_cands)}")
        if len(dwd_cands) > 0:
            for _, r in dwd_cands.iterrows():
                print(f"  {r.get('cluster','?'):20s}  "
                      f"RA={r['ra']:.4f} Dec={r['dec']:.4f}  "
                      f"cool={r['cooling_age_gyr']:.3f} Gyr > "
                      f"cl={r['cluster_age_gyr']:.3f} Gyr  "
                      f"path={r.get('match_path','')}")

    return df


def evaluate_dynamics_flags(row) -> dict:
    """Evaluate RV/6D dynamical consistency and age-warning flags."""
    flags = []
    severity = 'ok'

    def _val(name):
        try:
            v = row.get(name, np.nan)
        except AttributeError:
            v = row[name] if name in row else np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    def _first_val(*names):
        for name in names:
            value = _val(name)
            if np.isfinite(value):
                return value
        return np.nan

    rv_true = _first_val('rv_true_adopted', 'rv_true')
    rv_err = _first_val('rv_true_err_with_grav_floor', 'rv_true_err', 'rv_true_err_adopted')
    rv_diff = _first_val('gravfloor_rv_diff_kms', 'rv_diff_kms')
    rv_sigma = _first_val('gravfloor_rv_sigma', 'rv_sigma')
    chi2_kin = _val('chi2_kin')
    cooling_age = _first_val('cooling_age_gyr', 'wd_cooling_age_gyr')
    ms_lifetime = _first_val('ms_lifetime_gyr', 'wd_ms_lifetime_gyr',
                             'single_star_ms_lifetime_gyr')
    total_age = _first_val('total_age_with_ms_gyr', 'wd_total_age_with_ms_gyr',
                           'single_star_total_age_gyr', 'wd_total_age_gyr',
                           'total_age_gyr')
    if not np.isfinite(total_age) and np.isfinite(cooling_age) and np.isfinite(ms_lifetime):
        total_age = cooling_age + ms_lifetime
    cluster_age = _val('cluster_age_gyr')
    is_6d = bool(row.get('is_6d_matched', False)) if hasattr(row, 'get') else False

    if not np.isfinite(rv_true):
        flags.append('NO_RV')
    if np.isfinite(rv_err) and rv_err > 50:
        flags.append('RV_ERROR_LARGE')
    if np.isfinite(rv_diff) and rv_diff > 100:
        flags.append('RV_DIFF_LARGE')
    if np.isfinite(rv_sigma) and rv_sigma >= 3:
        flags.append('RV_SIGMA_MISMATCH')
    if np.isfinite(chi2_kin) and chi2_kin > 20:
        flags.append('KINEMATIC_CHI2_HIGH')
    if np.isfinite(cooling_age) and np.isfinite(cluster_age):
        if cooling_age > cluster_age:
            flags.append('COOLING_AGE_GT_CLUSTER_AGE')
        elif cooling_age > 0.8 * cluster_age:
            flags.append('COOLING_AGE_CLOSE_TO_CLUSTER_AGE')
    if np.isfinite(total_age) and np.isfinite(cluster_age):
        if total_age > cluster_age:
            flags.append('TOTAL_AGE_GT_CLUSTER_AGE')
        elif total_age > 0.8 * cluster_age:
            flags.append('TOTAL_AGE_CLOSE_TO_CLUSTER_AGE')
    if not is_6d:
        flags.append('NOT_6D_CONFIRMED')

    if any(f in flags for f in ('RV_SIGMA_MISMATCH', 'RV_DIFF_LARGE',
                                'COOLING_AGE_GT_CLUSTER_AGE',
                                'TOTAL_AGE_GT_CLUSTER_AGE')):
        severity = 'high'
    elif any(f in flags for f in ('RV_ERROR_LARGE', 'KINEMATIC_CHI2_HIGH',
                                  'COOLING_AGE_CLOSE_TO_CLUSTER_AGE',
                                  'TOTAL_AGE_CLOSE_TO_CLUSTER_AGE')):
        severity = 'caution'
    elif flags:
        severity = 'info'

    return {
        'dynamics_flags': flags,
        'dynamics_severity': severity,
        'dynamics_note': ', '.join(flags) if flags else 'dynamically consistent',
    }


def apply_dynamics_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Append dynamics flag columns to a DataFrame."""
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    records = [evaluate_dynamics_flags(row) for _, row in df.iterrows()]
    flag_df = pd.DataFrame(records)
    df['dynamics_flags'] = flag_df['dynamics_flags'].apply(lambda x: ';'.join(x))
    df['dynamics_severity'] = flag_df['dynamics_severity'].values
    df['dynamics_note'] = flag_df['dynamics_note'].values
    return df


# ──────────────────────────────────────────────────────────────────
#  内部辅助
# ──────────────────────────────────────────────────────────────────

_templates_cache = None


def _get_koester_model(teff: float, logg: float):
    """返回 (wavelength_array, flux_array) 或 (None, None)。"""
    global _templates_cache
    if _templates_cache is None:
        try:
            from astro_toolbox.sed import _load_koester2_templates
            _templates_cache = _load_koester2_templates()
        except Exception:
            _templates_cache = {}
    if not _templates_cache:
        return None, None
    best_key, best_d = None, np.inf
    for (t_teff, t_logg) in _templates_cache:
        d = abs(t_teff - teff) / 1000.0 + abs(t_logg - logg)
        if d < best_d:
            best_d, best_key = d, (t_teff, t_logg)
    if best_key is None:
        return None, None
    v = _templates_cache[best_key]
    return np.asarray(v['wavelength']), np.asarray(v['flux'])


def _load_spectrum(result_dir: str | None):
    """从 result_dir 读取第一个有效光谱，返回 (wave, flux, err, survey)。"""
    for fname, survey in [
        ('sdss_spectrum.csv', 'SDSS'),
        ('desi_spectrum.csv', 'DESI'),
        ('lamost_lrs_spectrum.csv', 'LAMOST'),
    ]:
        if result_dir is None:
            break
        p = os.path.join(result_dir, fname)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if len(df) < 50:
            continue
        wave = df['wavelength_A'].values.astype(float)
        flux = df['flux'].values.astype(float)
        err  = (df['error'].values.astype(float)
                if 'error' in df.columns else np.ones_like(flux))
        mask = np.isfinite(wave) & np.isfinite(flux)
        if mask.sum() < 50:
            continue
        return wave[mask], flux[mask], err[mask], survey
    return None, None, None, None


def _file_prefix(cluster: str, ra: float, dec: float) -> str:
    cl = cluster.replace(' ', '_').replace('/', '_') if cluster else 'unknown'
    return f"{cl}_ra={ra:.4f}_dec={dec:.4f}"


# Hunt+2023 成员星目录
_HUNT_MEMBERS_DIR = os.environ.get(
    'ASTRO_TOOLBOX_HUNT2023_MEMBERS_DIR',
    '/Users/a1/Desktop/星团/Hunt+2023/individual-oc',
)
_CLUSTER_COV_CACHE_PKL = os.environ.get(
    'ASTRO_TOOLBOX_CLUSTER_COV_CACHE',
    '/Users/a1/Desktop/desi匹配/data/cluster_covariance_cache_strict.pkl',
)
_CLUSTER_COV_CACHE = None


def _cluster_key(name: str) -> str:
    return str(name or '').replace('_', ' ').strip().lower()


def _load_cluster_covariance_cache(path: str = _CLUSTER_COV_CACHE_PKL):
    """Load the strict cluster covariance cache used by plot_6d_separate.py."""
    global _CLUSTER_COV_CACHE
    if _CLUSTER_COV_CACHE is not None:
        return _CLUSTER_COV_CACHE
    if not os.path.exists(path):
        _CLUSTER_COV_CACHE = []
        return _CLUSTER_COV_CACHE
    try:
        import pickle
        with open(path, 'rb') as fh:
            cache = pickle.load(fh)
        _CLUSTER_COV_CACHE = cache if isinstance(cache, list) else []
    except Exception:
        _CLUSTER_COV_CACHE = []
    return _CLUSTER_COV_CACHE


def _load_kinematic_cluster_info(cluster_name: str):
    """
    Match the reference 6D plotting workflow:
    Hunt+2023 provides the cluster center, while the strict covariance cache
    provides s_pmRA/s_pmDE/s_Plx, rt/rtot and the PM covariance matrix.
    """
    if not cluster_name:
        return None
    info = {}
    target = _cluster_key(cluster_name)

    try:
        from .orbit_traceback import load_hunt2023_clusters
    except Exception:
        try:
            from astro_toolbox.orbit_traceback import load_hunt2023_clusters
        except Exception:
            load_hunt2023_clusters = None

    if load_hunt2023_clusters is not None:
        try:
            for c in load_hunt2023_clusters():
                if _cluster_key(c.get('Name', '')) == target:
                    info.update({
                        'Name': c.get('Name', cluster_name),
                        'RA': c.get('RA', np.nan),
                        'DEC': c.get('DEC', np.nan),
                        'pmRA': c.get('pmRA', np.nan),
                        'pmDE': c.get('pmDE', np.nan),
                        'Plx': c.get('Plx', np.nan),
                        'dist50': c.get('dist50', np.nan),
                        'RV': c.get('RV', np.nan),
                        'e_RV': c.get('e_RV', np.nan),
                        'rtpc': c.get('rtpc', np.nan),
                        'logAge50': c.get('logAge50', np.nan),
                    })
                    break
        except Exception:
            pass

    for c in _load_cluster_covariance_cache():
        if _cluster_key(c.get('Name', '')) != target:
            continue
        info.setdefault('Name', c.get('Name', cluster_name))
        info.setdefault('RA', c.get('RAdeg', c.get('ra_c', np.nan)))
        info.setdefault('DEC', c.get('DEdeg', c.get('de_c', np.nan)))
        info.setdefault('pmRA', c.get('pmRA', np.nan))
        info.setdefault('pmDE', c.get('pmDE', np.nan))
        info.setdefault('Plx', c.get('Plx', c.get('plx_c', np.nan)))
        info.update({
            's_pmRA': c.get('s_pmRA', np.nan),
            's_pmDE': c.get('s_pmDE', np.nan),
            's_Plx': c.get('s_Plx', np.nan),
            'rt_deg': c.get('rt', np.nan),
            'rtot_deg': c.get('rtot', np.nan),
        })
        params = c.get('cluster_params') or {}
        if params.get('cov_kin') is not None:
            info['cov_kin'] = np.asarray(params.get('cov_kin'), dtype=float)
        if params.get('mu_kin') is not None:
            info['mu_kin'] = np.asarray(params.get('mu_kin'), dtype=float)
        break

    return info if info else None

def _load_hunt2023_members(cluster_name: str, prob_min: float = 0.5):
    """从 Hunt+2023 individual-oc/ 读取星团成员星，返回 DataFrame。"""
    fname = cluster_name.replace(' ', '_') + '.csv'
    path = os.path.join(_HUNT_MEMBERS_DIR, fname)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        members = pd.read_csv(path)
        if 'Prob' in members.columns:
            members = members[members['Prob'] >= prob_min]
        return members
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────
#  绘图函数
# ──────────────────────────────────────────────────────────────────

def plot_spectrum(
    row,
    result_dir: str | None,
    out_path: str,
) -> None:
    """
    绘制光谱 + Koester 模型叠加图，保存到 out_path。

    Parameters
    ----------
    row : dict-like
        需含 ra, dec, teff, logg, spectral_type, cluster 字段。
    result_dir : str or None
        包含 *_spectrum.csv 的目录，None 时图中显示 "No spectrum"。
    out_path : str
        输出 PNG 路径。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ra      = float(row['ra'])
    dec     = float(row['dec'])
    teff    = float(row['teff'])  if pd.notna(row.get('teff'))  else np.nan
    logg    = float(row['logg'])  if pd.notna(row.get('logg'))  else np.nan
    sp_type = str(row.get('spectral_type', 'DA'))
    cluster = str(row.get('cluster', ''))

    wave, flux, err, survey = _load_spectrum(result_dir)

    fig, ax = plt.subplots(figsize=(12, 5))

    if wave is not None:
        med = np.nanmedian(flux)
        if med <= 0:
            med = 1.0
        f_n = flux / med
        e_n = err  / med
        ax.plot(wave, f_n, '-', color='#1f77b4', lw=0.7, alpha=0.85,
                label=f'{survey} spectrum')
        ax.fill_between(wave, f_n - e_n, f_n + e_n, color='#1f77b4', alpha=0.15)

        if np.isfinite(teff) and np.isfinite(logg):
            mw, mf = _get_koester_model(teff, logg)
            if mw is not None:
                mask_m = (mw >= wave.min()) & (mw <= wave.max())
                if mask_m.sum() > 10:
                    mi = np.interp(wave, mw[mask_m], mf[mask_m])
                    pos = (mi > 0) & (flux > 0)
                    if pos.sum() > 10:
                        scale = np.median(
                            (flux[pos] / med) / (mi[pos] / np.median(mi[pos]))
                        )
                        ax.plot(
                            wave,
                            mi / np.median(mi[pos]) * scale * np.median(f_n[pos]),
                            '-', color='red', lw=1.3, alpha=0.85,
                            label=f'Koester T={teff:.0f} K  log g={logg:.1f}',
                        )

        for lbl, wl in [
            (r'H$\alpha$', 6563), (r'H$\beta$', 4861),
            (r'H$\gamma$', 4341), (r'H$\delta$', 4102),
            (r'H$\epsilon$', 3970), (r'H$\zeta$', 3889),
        ]:
            if wave.min() < wl < wave.max():
                ax.axvline(wl, color='gray', lw=0.6, alpha=0.4, ls='--')
                ax.text(wl, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                        lbl, fontsize=7, ha='center', va='top',
                        color='gray', rotation=90, clip_on=True)

        p1, p99 = np.nanpercentile(f_n, [1, 99])
        ax.set_ylim(max(-0.2, p1 - 0.3), p99 + 0.6)
        ax.set_xlim(3400, min(9500, wave.max()))
    else:
        ax.text(0.5, 0.5, 'No spectrum available',
                ha='center', va='center', fontsize=13,
                transform=ax.transAxes, color='gray')

    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Normalized flux', fontsize=12)
    ax.set_title(
        f'Spectrum  |  RA={ra:.4f}  Dec={dec:.4f}  |  {cluster}  |  '
        f'{sp_type}  Teff={teff:.0f} K  log g={logg:.2f}',
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ztf(row, ztf_lc, out_path: str) -> None:
    """
    绘制 ZTF g/r/i 光变曲线图。

    Parameters
    ----------
    row : dict-like
        需含 ra, dec, cluster 字段。
    ztf_lc : dict or None
        由 astro_toolbox.ztf.query_lightcurve() 返回的字典，
        键为波段名 ('g','r','i')，值为含 mjd/mag/magerr 列的 DataFrame。
    out_path : str
        输出 PNG 路径。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ra      = float(row['ra'])
    dec     = float(row['dec'])
    cluster = str(row.get('cluster', ''))

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {'g': '#2ca02c', 'r': '#d62728', 'i': '#ff7f0e'}
    has = False
    if ztf_lc is not None:
        for band in ('g', 'r', 'i'):
            if band not in ztf_lc:
                continue
            df_b = ztf_lc[band]
            if len(df_b) < 3:
                continue
            ax.errorbar(df_b['mjd'], df_b['mag'], yerr=df_b['magerr'],
                        fmt='o', color=colors[band], ms=2, elinewidth=0.4,
                        alpha=0.6, label=f'ZTF {band} (N={len(df_b)})')
            has = True
    if not has:
        ax.text(0.5, 0.5, 'No ZTF data',
                ha='center', va='center', fontsize=13,
                transform=ax.transAxes, color='gray')
    else:
        ax.invert_yaxis()
        ax.legend(fontsize=9)

    ax.set_xlabel('MJD', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title(
        f'ZTF Light Curve  |  RA={ra:.4f}  Dec={dec:.4f}  |  {cluster}',
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_hrd(
    row,
    bg_bp: np.ndarray,
    bg_mg: np.ndarray,
    out_path: str,
    hunt_members_dir: str | None = None,
) -> None:
    """
    绘制单源的 HR 图（Gaia 背景密度图 + Hunt+2023 成员星蓝点 + 目标红星）。

    Parameters
    ----------
    row : dict-like
        需含 ra, dec, phot_g_mean_mag, bp_rp, parallax, teff,
        cluster, membership 字段。
    bg_bp : np.ndarray
        背景星的 BP-RP 颜色数组。
    bg_mg : np.ndarray
        背景星的绝对星等 M_G 数组。
    out_path : str
        输出 PNG 路径。
    hunt_members_dir : str, optional
        Hunt+2023 individual-oc/ 目录路径，None 时使用默认路径。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ra        = float(row['ra'])
    dec       = float(row['dec'])
    g_mag     = float(row['phot_g_mean_mag']) if pd.notna(row.get('phot_g_mean_mag')) else np.nan
    bp_rp     = float(row['bp_rp'])           if pd.notna(row.get('bp_rp'))           else np.nan
    plx       = float(row['parallax'])        if pd.notna(row.get('parallax'))        else np.nan
    teff      = float(row['teff'])            if pd.notna(row.get('teff'))            else np.nan
    cluster   = str(row.get('cluster', ''))
    membership = str(row.get('membership', ''))

    M_G = np.nan
    if plx > 0 and np.isfinite(g_mag):
        M_G = g_mag + 5 * np.log10(plx / 1000.0) + 5

    fig, ax = plt.subplots(figsize=(8, 10))

    if len(bg_bp) > 0:
        ax.hist2d(bg_bp, bg_mg, bins=300,
                  cmap='Greys', norm=matplotlib.colors.LogNorm(),
                  alpha=0.85, zorder=1)

    # Hunt+2023 成员星（蓝点）
    if cluster:
        members = _load_hunt2023_members(cluster)
        if len(members) > 0 and 'Plx' in members.columns and 'Gmag' in members.columns:
            m_plx = members['Plx'].values.astype(float)
            m_gmag = members['Gmag'].values.astype(float)
            m_bprp = members['BP-RP'].values.astype(float) if 'BP-RP' in members.columns else np.full(len(members), np.nan)
            valid = (m_plx > 0) & np.isfinite(m_gmag) & np.isfinite(m_bprp)
            if valid.sum() > 0:
                m_MG = m_gmag[valid] + 5 * np.log10(m_plx[valid] / 1000.0) + 5
                ax.scatter(m_bprp[valid], m_MG, c='#1f77b4', s=8, alpha=0.5,
                           zorder=3, edgecolors='none',
                           label=f'{cluster} Hunt+2023 (N={valid.sum()})')

    if np.isfinite(bp_rp) and np.isfinite(M_G):
        ax.scatter(
            [bp_rp], [M_G], c='red', s=150,
            edgecolors='black', linewidths=0.8, zorder=10, marker='*',
            label=f'This WD  T={teff:.0f} K' if np.isfinite(teff) else 'This WD',
        )

    ax.set_xlabel('BP − RP (mag)', fontsize=13)
    ax.set_ylabel(r'$M_G$ (mag)', fontsize=13)
    ax.set_title(
        f'HR Diagram  |  RA={ra:.4f}  Dec={dec:.4f}\n{cluster}  |  {membership}',
        fontsize=12,
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(16, -4)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')
    ax.text(0.3,  -2, 'Main Sequence', fontsize=9, color='gray', rotation=55, ha='center')
    ax.text(2.5,  -1, 'Giants',        fontsize=9, color='gray')
    ax.text(-0.2, 10, 'White\nDwarfs', fontsize=9, color='gray')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_5d_astrometry_legacy(row, out_path: str) -> None:
    """
    绘制 Gaia 5D 认证预检图: 位置、视差/距离、自行、RUWE 和 CMD 位置。

    这张图不需要 RV，因此应在 6D/RV 信息板之前生成，用于判断 Gaia
    天体测量本身是否可靠。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _num(key):
        try:
            value = row.get(key, np.nan)
            value = float(value)
            return value if np.isfinite(value) else np.nan
        except Exception:
            return np.nan

    def _fmt(value, fmt='.3f', unit=''):
        if np.isfinite(value):
            return f'{value:{fmt}} {unit}'.strip()
        return 'N/A'

    ra = _num('ra')
    dec = _num('dec')
    source_id = str(row.get('source_id', '') or '')
    plx = _num('parallax')
    e_plx = _num('e_parallax')
    pmra = _num('pmRA')
    epmra = _num('e_pmRA')
    pmde = _num('pmDE')
    epmde = _num('e_pmDE')
    ruwe = _num('RUWE')
    gmag = _num('phot_g_mean_mag')
    bp_rp = _num('bp_rp')
    mg = _num('M_G')
    if not np.isfinite(mg) and np.isfinite(gmag) and np.isfinite(plx) and plx > 0:
        mg = gmag + 5 * np.log10(plx / 1000.0) + 5
    dist_pc = 1000.0 / plx if np.isfinite(plx) and plx > 0 else np.nan
    vt = (4.74047 * np.hypot(pmra, pmde) / plx
          if np.isfinite(pmra + pmde + plx) and plx > 0 else np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor('white')

    ax = axes[0, 0]
    ax.axis('off')
    rows = [
        ('Gaia source_id', source_id if source_id else 'N/A'),
        ('RA, Dec', f'{_fmt(ra, ".6f")} , {_fmt(dec, ".6f")}'),
        ('Parallax', f'{_fmt(plx, ".3f", "mas")} ± {_fmt(e_plx, ".3f", "mas")}'),
        ('Distance', _fmt(dist_pc, '.2f', 'pc')),
        ('RUWE', _fmt(ruwe, '.3f')),
        ('G, BP-RP', f'{_fmt(gmag, ".3f", "mag")} , {_fmt(bp_rp, ".3f", "mag")}'),
        ('M_G', _fmt(mg, '.3f', 'mag')),
        ('v_tan', _fmt(vt, '.1f', 'km/s')),
    ]
    y = 0.95
    for key, val in rows:
        ax.text(0.03, y, key + ':', fontsize=10, fontweight='bold',
                color='#444', ha='left', va='top', transform=ax.transAxes)
        ax.text(0.36, y, val, fontsize=10, color='#111',
                ha='left', va='top', transform=ax.transAxes)
        y -= 0.105
    ax.set_title('Gaia 5D Astrometry', fontsize=12)

    ax = axes[0, 1]
    if np.isfinite(pmra) and np.isfinite(pmde):
        ax.errorbar(pmra, pmde,
                    xerr=epmra if np.isfinite(epmra) else None,
                    yerr=epmde if np.isfinite(epmde) else None,
                    fmt='o', ms=8, color='#1f77b4', ecolor='#1f77b4',
                    capsize=3, label='Gaia DR3')
        ax.arrow(0, 0, pmra, pmde, color='#d62728', alpha=0.65,
                 width=0.0, head_width=max(np.hypot(pmra, pmde) * 0.035, 0.2),
                 length_includes_head=True)
        span = max(abs(pmra), abs(pmde), 1.0) * 1.35
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.axhline(0, color='0.6', lw=0.8)
        ax.axvline(0, color='0.6', lw=0.8)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No Gaia proper motion',
                ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_xlabel(r'$\mu_{\alpha*}$ (mas yr$^{-1}$)')
    ax.set_ylabel(r'$\mu_{\delta}$ (mas yr$^{-1}$)')
    ax.set_title('Proper Motion Vector', fontsize=12)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    bp_grid = np.linspace(-0.5, 1.8, 250)
    try:
        ms = np.array([estimate_main_sequence_mg(x) for x in bp_grid])
        wd = np.array([estimate_white_dwarf_mg(x) for x in bp_grid])
        ax.plot(bp_grid, ms, color='0.55', lw=1.2, label='MS guide')
        ax.plot(bp_grid, wd, color='#1f77b4', lw=1.4, label='WD guide')
    except Exception:
        pass
    if np.isfinite(bp_rp) and np.isfinite(mg):
        ax.scatter([bp_rp], [mg], marker='*', s=180, c='crimson',
                   edgecolors='black', linewidths=0.8, zorder=5,
                   label='Target')
    else:
        ax.text(0.5, 0.5, 'No Gaia CMD point',
                ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_xlabel('BP - RP (mag)')
    ax.set_ylabel(r'$M_G$ (mag)')
    ax.set_xlim(-0.5, 1.8)
    ax.set_ylim(16, 6)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title('5D Photometric Consistency', fontsize=12)

    ax = axes[1, 1]
    ax.axis('off')
    quality = []
    if np.isfinite(plx) and plx > 0:
        snr = plx / e_plx if np.isfinite(e_plx) and e_plx > 0 else np.nan
        quality.append(f'Parallax S/N = {_fmt(snr, ".1f")}')
    if np.isfinite(ruwe):
        quality.append('RUWE OK' if ruwe < 1.4 else 'RUWE high: check astrometric excess')
    if np.isfinite(pmra + pmde):
        quality.append(f'Proper motion = {_fmt(np.hypot(pmra, pmde), ".2f", "mas/yr")}')
    if np.isfinite(vt):
        quality.append(f'Tangential velocity = {_fmt(vt, ".1f", "km/s")}')
    for extra in ('membership', 'match_note', 'dynamics_flags'):
        value = str(row.get(extra, '') or '')
        if value and value != 'nan':
            quality.append(f'{extra}: {value}')
    if not quality:
        quality.append('Insufficient Gaia 5D information')

    y = 0.92
    for item in quality[:9]:
        ax.text(0.06, y, item, fontsize=10, color='#222',
                ha='left', va='top', transform=ax.transAxes)
        y -= 0.095
    ax.set_title('5D Quality Notes', fontsize=12)

    fig.suptitle(f'5D Astrometric Check  |  RA={ra:.4f}  Dec={dec:.4f}',
                 fontsize=13, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_5d_astrometry(row, out_path: str) -> None:
    """
    绘制 Gaia 5D 认证预检图。

    版式参考 sdss_lamost_desi/plot_6d_separate.py 的 6D kinematic 图:
    (a) 空间分布, (b) 自行平面, (c) 视差对比。若 row 提供 cluster 且
    本地 Hunt+2023 成员星可用，则叠加星团中心、成员星、潮汐半径和
    5D 偏离量；若没有星团信息，也保持同样三联图风格输出单源 Gaia
    5D 质量图。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    def _get(key, default=np.nan):
        try:
            return row.get(key, default)
        except AttributeError:
            try:
                return row[key]
            except Exception:
                return default

    def _num(*keys, default=np.nan):
        for key in keys:
            try:
                value = _get(key, np.nan)
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                value = float(value)
                if np.isfinite(value):
                    return value
            except Exception:
                continue
        return default

    def _text(*keys, default=''):
        for key in keys:
            value = _get(key, '')
            try:
                if value is None or pd.isna(value):
                    continue
            except Exception:
                if value is None:
                    continue
            value = str(value).strip()
            if value and value.lower() not in ('nan', 'none', 'null'):
                return value
        return default

    def _dict_num(obj, *keys, default=np.nan):
        if not obj:
            return default
        for key in keys:
            try:
                value = obj.get(key, np.nan)
                if value is None:
                    continue
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass
                value = float(value)
                if np.isfinite(value):
                    return value
            except Exception:
                continue
        return default

    def _fmt(value, fmt='.3f', unit=''):
        if np.isfinite(value):
            return f'{value:{fmt}} {unit}'.strip()
        return 'N/A'

    def _safe_err(value, fallback):
        return value if np.isfinite(value) and value > 0 else fallback

    def _member_array(members, *names):
        if members is None or len(members) == 0:
            return np.array([], dtype=float)
        for name in names:
            if name in members.columns:
                return pd.to_numeric(members[name], errors='coerce').to_numpy(dtype=float)
        return np.array([], dtype=float)

    def _finite(arr):
        arr = np.asarray(arr, dtype=float)
        return arr[np.isfinite(arr)]

    def _dedupe_legend(ax, **kwargs):
        handles, labels = ax.get_legend_handles_labels()
        seen, keep_h, keep_l = set(), [], []
        for handle, label in zip(handles, labels):
            if not label or label in seen or label.startswith('_'):
                continue
            seen.add(label)
            keep_h.append(handle)
            keep_l.append(label)
        if keep_h:
            ax.legend(keep_h, keep_l, **kwargs)

    def _find_cluster(cluster_name):
        return _load_kinematic_cluster_info(cluster_name)

    def _set_limits_from_points(ax, xs, ys, min_pad=0.5):
        xs = _finite(xs)
        ys = _finite(ys)
        if len(xs) == 0 or len(ys) == 0:
            return
        # The reference 6D plot must always include the WD and cluster center,
        # even when the WD is a strong kinematic outlier.  Use full finite
        # extrema rather than percentile clipping.
        xlo, xhi = xs.min(), xs.max()
        ylo, yhi = ys.min(), ys.max()
        xpad = max((xhi - xlo) * 0.18, min_pad)
        ypad = max((yhi - ylo) * 0.18, min_pad)
        ax.set_xlim(xlo - xpad, xhi + xpad)
        ax.set_ylim(ylo - ypad, yhi + ypad)

    ra = _num('ra', 'RA', 'RAdeg')
    dec = _num('dec', 'DEC', 'DEdeg')
    source_id = _text('source_id', 'source', 'source_id_dr3')
    cluster = _text('cluster', 'Cluster', 'cluster_name')
    membership = _text('membership')
    match_path = _text('match_path', 'match_path_new')

    plx = _num('parallax', 'Plx', 'plx')
    e_plx = _num('e_parallax', 'parallax_error', 'e_Plx', 'e_plx', 'plx_error')
    pmra = _num('pmRA', 'pmra', 'pmra_corr')
    pmde = _num('pmDE', 'pmdec', 'pmDE_corr', 'pmdec_corr')
    e_pmra = _num('e_pmRA', 'pmra_error', 'pmRA_error', 'e_pmra')
    e_pmde = _num('e_pmDE', 'pmdec_error', 'pmDE_error', 'e_pmdec')
    ruwe = _num('RUWE', 'ruwe')
    gmag = _num('phot_g_mean_mag', 'Gmag', 'gmag')
    bp_rp = _num('bp_rp', 'BP_RP', 'bp-rp')
    mg = _num('M_G', 'MG', 'absmag')
    chi2 = _num('chi2_kin', 'chi2_5d')
    rv_sigma = _num('rv_sigma', 'rv_sigma_new')
    rv_diff = _num('rv_diff_kms', 'rv_diff_kms_new')
    match_quality = _text('rv_match_quality', 'rv_match_quality_new')
    if not match_quality:
        match_quality = _rv_match_quality(rv_diff, rv_sigma, _num('rv_true_err', 'rv_true_err_adopted'))
    else:
        match_quality = match_quality.lower()
    is_6d_display = (
        '6d' in membership.lower()
        or _truthy(_get('is_6d_matched', False))
        or _truthy(_get('is_6d_matched_new', False))
    )
    if is_6d_display and (
        match_quality in {'strict', 'borderline'}
        or np.isfinite(rv_sigma)
    ):
        membership = '6D_matched' if match_quality == 'strict' else '6D_candidate'

    if not np.isfinite(mg) and np.isfinite(gmag) and np.isfinite(plx) and plx > 0:
        mg = gmag + 5 * np.log10(plx / 1000.0) + 5
    dist_pc = 1000.0 / plx if np.isfinite(plx) and plx > 0 else np.nan
    vt = (4.74047 * np.hypot(pmra, pmde) / plx
          if np.isfinite(pmra + pmde + plx) and plx > 0 else np.nan)

    cl_info = _find_cluster(cluster)
    row_cluster_fields = {
        'RA': _num('cluster_ra', 'cluster_RA', 'cl_ra', 'cl_RA'),
        'DEC': _num('cluster_dec', 'cluster_DEC', 'cl_dec', 'cl_DEC'),
        'pmRA': _num('cluster_pmRA', 'cluster_pmra', 'cl_pmRA', 'cl_pmra'),
        'pmDE': _num('cluster_pmDE', 'cluster_pmdec', 'cl_pmDE', 'cl_pmdec'),
        'Plx': _num('cluster_parallax', 'cluster_Plx', 'cl_plx', 'cl_Plx'),
        'dist50': _num('cluster_dist_pc', 'cluster_dist50', 'cl_dist_pc', 'dist50'),
        'rtpc': _num('cluster_rtpc', 'tidal_radius_pc', 'rtpc'),
        's_pmRA': _num('cluster_s_pmRA', 's_pmRA', 'cluster_pmra_sigma'),
        's_pmDE': _num('cluster_s_pmDE', 's_pmDE', 'cluster_pmdec_sigma'),
        's_Plx': _num('cluster_s_Plx', 's_Plx', 'cluster_plx_sigma'),
        'rt_deg': _num('rt_deg'),
        'rtot_deg': _num('rtot_deg'),
    }
    if any(np.isfinite(v) for v in row_cluster_fields.values()):
        if cl_info is None:
            cl_info = {'Name': cluster}
        cl_info.update({k: v for k, v in row_cluster_fields.items() if np.isfinite(v)})

    members = _load_hunt2023_members(cluster) if cluster else pd.DataFrame()
    if len(members) > 0:
        kin_cols = ['RAdeg', 'DEdeg', 'pmRA', 'pmDE', 'Plx']
        if all(c in members.columns for c in kin_cols):
            members = members[kin_cols].dropna()
            members = members[pd.to_numeric(members['Plx'], errors='coerce') > 0.05]
    mem_ra = _member_array(members, 'RAdeg', 'RA', 'ra')
    mem_dec = _member_array(members, 'DEdeg', 'DEC', 'dec')
    mem_pmra = _member_array(members, 'pmRA', 'pmra')
    mem_pmde = _member_array(members, 'pmDE', 'pmdec')
    mem_plx = _member_array(members, 'Plx', 'parallax', 'plx')

    ra_cl = _dict_num(cl_info, 'RA', 'ra')
    dec_cl = _dict_num(cl_info, 'DEC', 'Dec', 'dec')
    pmra_cl = _dict_num(cl_info, 'pmRA', 'pmra')
    pmde_cl = _dict_num(cl_info, 'pmDE', 'pmdec')
    plx_cl = _dict_num(cl_info, 'Plx', 'parallax', 'plx')
    dist_cl = _dict_num(cl_info, 'dist50', 'distance_pc')
    rtpc = _dict_num(cl_info, 'rtpc', 'tidal_radius_pc')

    if not np.isfinite(plx_cl) and np.isfinite(dist_cl) and dist_cl > 0:
        plx_cl = 1000.0 / dist_cl
    if not np.isfinite(dist_cl) and np.isfinite(plx_cl) and plx_cl > 0:
        dist_cl = 1000.0 / plx_cl

    s_pmra = _dict_num(cl_info, 's_pmRA', 'pmRA_sigma', 'pmra_sigma')
    s_pmde = _dict_num(cl_info, 's_pmDE', 'pmDE_sigma', 'pmdec_sigma')
    s_plx = _dict_num(cl_info, 's_Plx', 'Plx_sigma', 'parallax_sigma')
    mem_pmra_f = _finite(mem_pmra)
    mem_pmde_f = _finite(mem_pmde)
    mem_plx_f = _finite(mem_plx)
    if not np.isfinite(s_pmra) and len(mem_pmra_f) >= 5:
        s_pmra = np.nanstd(mem_pmra_f)
    if not np.isfinite(s_pmde) and len(mem_pmde_f) >= 5:
        s_pmde = np.nanstd(mem_pmde_f)
    if not np.isfinite(s_plx) and len(mem_plx_f) >= 5:
        s_plx = np.nanstd(mem_plx_f)
    s_pmra = _safe_err(s_pmra, 0.15)
    s_pmde = _safe_err(s_pmde, 0.15)
    s_plx = _safe_err(s_plx, 0.05)

    e_pmra_plot = _safe_err(e_pmra, 0.20)
    e_pmde_plot = _safe_err(e_pmde, 0.20)
    e_plx_plot = _safe_err(e_plx, max(abs(plx) * 0.02, 0.05) if np.isfinite(plx) else 0.10)

    rt_deg = _dict_num(cl_info, 'rt_deg')
    if not np.isfinite(rt_deg) and np.isfinite(rtpc + dist_cl) and dist_cl > 0:
        rt_deg = np.degrees(np.arctan2(rtpc, dist_cl))
    rtot_deg = _dict_num(cl_info, 'rtot_deg')
    if not np.isfinite(rtot_deg) and np.isfinite(rt_deg):
        rtot_deg = max(3.0 * rt_deg, rt_deg + 0.05)

    has_cluster = (
        cl_info is not None
        and np.isfinite(ra_cl + dec_cl + pmra_cl + pmde_cl + plx_cl)
    )

    cov_kin = None
    pm_mask = np.isfinite(mem_pmra) & np.isfinite(mem_pmde)
    if cl_info is not None and np.asarray(cl_info.get('cov_kin', [])).shape == (2, 2):
        cov_kin = np.asarray(cl_info.get('cov_kin'), dtype=float)
        if not np.all(np.isfinite(cov_kin)):
            cov_kin = None
    if pm_mask.sum() >= 8:
        try:
            if cov_kin is None:
                cov_kin = np.cov(np.vstack([mem_pmra[pm_mask], mem_pmde[pm_mask]]))
                if not np.all(np.isfinite(cov_kin)):
                    cov_kin = None
        except Exception:
            cov_kin = None
    if cov_kin is None and has_cluster:
        cov_kin = np.diag([s_pmra**2, s_pmde**2])

    if not np.isfinite(chi2) and has_cluster:
        terms = []
        if np.isfinite(pmra + pmde):
            delta_pm = np.array([pmra - pmra_cl, pmde - pmde_cl], dtype=float)
            try:
                cov_total = np.asarray(cov_kin, dtype=float) + np.diag([e_pmra_plot**2, e_pmde_plot**2])
                terms.append(float(delta_pm @ np.linalg.pinv(cov_total) @ delta_pm))
            except Exception:
                denom = s_pmra**2 + e_pmra_plot**2
                if np.isfinite(pmra) and denom > 0:
                    terms.append((pmra - pmra_cl)**2 / denom)
                denom = s_pmde**2 + e_pmde_plot**2
                if np.isfinite(pmde) and denom > 0:
                    terms.append((pmde - pmde_cl)**2 / denom)
        denom = s_plx**2 + e_plx_plot**2
        if np.isfinite(plx) and denom > 0:
            terms.append((plx - plx_cl)**2 / denom)
        if terms:
            chi2 = float(np.sum(terms))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    fig.patch.set_facecolor('white')

    # ======== (a) 空间分布 ========
    ax = axes[0]
    mem_pos_mask = np.isfinite(mem_ra) & np.isfinite(mem_dec)
    if mem_pos_mask.sum() > 0:
        ax.scatter(mem_ra[mem_pos_mask], mem_dec[mem_pos_mask],
                   c='#aaaaaa', s=8, alpha=0.45, zorder=2, edgecolors='none',
                   label=f'Members (N={mem_pos_mask.sum()}, Prob>=0.5)')
    if has_cluster:
        ax.plot(ra_cl, dec_cl, '*', ms=18, c='#e74c3c', zorder=10,
                label=f'{cluster} center')
        if np.isfinite(rt_deg):
            ax.add_patch(Circle((ra_cl, dec_cl), rt_deg, fill=False,
                                ec='#e74c3c', ls='--', lw=1.5,
                                label=fr'$r_t$ = {rt_deg:.3f}$^\circ$'))
        if np.isfinite(rtot_deg):
            ax.add_patch(Circle((ra_cl, dec_cl), rtot_deg, fill=False,
                                ec='#e74c3c', ls=':', lw=1.0, alpha=0.6,
                                label=fr'$r_{{tot}}$ = {rtot_deg:.3f}$^\circ$'))
    if np.isfinite(ra + dec):
        ax.plot(ra, dec, 'D', ms=10, c='#2980b9', zorder=11,
                markeredgecolor='k', markeredgewidth=0.5, label='WD candidate')
    else:
        ax.text(0.5, 0.5, 'No Gaia sky position',
                ha='center', va='center', transform=ax.transAxes, color='gray')

    if has_cluster and np.isfinite(ra + dec):
        ax.plot([ra_cl, ra], [dec_cl, dec], 'k--', alpha=0.35, lw=0.9)
        ang_sep = np.sqrt(((ra - ra_cl) * np.cos(np.radians(dec_cl)))**2 + (dec - dec_cl)**2)
        mid_ra = (ra_cl + ra) / 2
        mid_dec = (dec_cl + dec) / 2
        ax.text(mid_ra, mid_dec, fr'{ang_sep:.3f}$^\circ$',
                fontsize=8, ha='center', va='bottom', color='#555',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#ccc', alpha=0.85))
        d_pmra = pmra - pmra_cl
        d_pmde = pmde - pmde_cl
        pm_len = np.hypot(d_pmra, d_pmde)
        if np.isfinite(pm_len) and pm_len > 0:
            arrow_scale = max(rt_deg if np.isfinite(rt_deg) else 0.1, 0.05) * 0.35
            dx = d_pmra / pm_len * arrow_scale
            dy = d_pmde / pm_len * arrow_scale
            v_rel = (4.74047 * pm_len * dist_cl / 1000.0
                     if np.isfinite(dist_cl) else np.nan)
            ax.annotate('', xy=(ra + dx, dec + dy), xytext=(ra, dec),
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.2))
            if np.isfinite(v_rel):
                ax.text(ra + dx * 1.25, dec + dy * 1.25,
                        fr'$\Delta v_t$ = {v_rel:.1f} km/s',
                        fontsize=8, color='#27ae60', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec='#27ae60', alpha=0.85))
        pad = max((rtot_deg if np.isfinite(rtot_deg) else 0.0) * 1.35,
                  ang_sep * 1.35, 0.05)
        center_ra = (ra_cl + ra) / 2
        center_dec = (dec_cl + dec) / 2
        ax.set_xlim(center_ra + pad, center_ra - pad)
        ax.set_ylim(center_dec - pad, center_dec + pad)
    elif np.isfinite(ra + dec):
        pad = 0.05
        ax.set_xlim(ra + pad, ra - pad)
        ax.set_ylim(dec - pad, dec + pad)
        note = [
            f'Gaia source_id: {source_id or "N/A"}',
            f'RUWE: {_fmt(ruwe, ".3f")}',
            f'G/BP-RP: {_fmt(gmag, ".3f")} / {_fmt(bp_rp, ".3f")}',
            f'M_G: {_fmt(mg, ".3f")}',
        ]
        ax.text(0.03, 0.97, '\n'.join(note), transform=ax.transAxes,
                fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cfd8dc', alpha=0.90))
    elif mem_pos_mask.sum() > 0:
        _set_limits_from_points(ax, mem_ra[mem_pos_mask], mem_dec[mem_pos_mask], min_pad=0.05)
        ax.invert_xaxis()
    ax.set_xlabel('RA (deg)')
    ax.set_ylabel('Dec (deg)')
    ax.set_title('(a) Spatial Distribution')
    ax.grid(alpha=0.20)
    ax.set_aspect('equal', adjustable='box')
    _dedupe_legend(ax, fontsize=7, loc='upper right', framealpha=0.88)

    # ======== (b) 自行 ========
    ax = axes[1]
    mem_pm_mask = np.isfinite(mem_pmra) & np.isfinite(mem_pmde)
    if mem_pm_mask.sum() > 0:
        ax.scatter(mem_pmra[mem_pm_mask], mem_pmde[mem_pm_mask],
                   c='#aaaaaa', s=8, alpha=0.45, zorder=2, edgecolors='none',
                   label=f'Members (N={mem_pm_mask.sum()})')
    if has_cluster:
        ax.errorbar(pmra_cl, pmde_cl, xerr=s_pmra, yerr=s_pmde,
                    fmt='*', ms=18, c='#e74c3c', capsize=4, capthick=1.4,
                    elinewidth=1.4, label=f'{cluster}', zorder=5)
        if cov_kin is not None and np.all(np.isfinite(cov_kin)):
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_kin)
                eigvals = np.clip(eigvals, 1e-8, None)
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                for ns, alpha_v in [(1, 0.35), (2, 0.18), (3, 0.09)]:
                    width = 2 * ns * np.sqrt(eigvals[0])
                    height = 2 * ns * np.sqrt(eigvals[1])
                    ell = Ellipse((pmra_cl, pmde_cl), width, height,
                                  angle=angle, fill=False, ec='#e74c3c',
                                  lw=0.9, alpha=alpha_v)
                    ax.add_patch(ell)
                    if ns == 3:
                        ax.text(pmra_cl + width / 2 * 0.8,
                                pmde_cl + height / 2 * 0.8,
                                f'{ns}$\\sigma$', fontsize=7,
                                color='#e74c3c', alpha=0.6)
            except Exception:
                pass
    if np.isfinite(pmra + pmde):
        ax.errorbar(pmra, pmde, xerr=e_pmra_plot, yerr=e_pmde_plot,
                    fmt='D', ms=10, c='#2980b9', capsize=4, capthick=1.4,
                    elinewidth=1.4, markeredgecolor='k', markeredgewidth=0.5,
                    label='WD', zorder=6)
    else:
        ax.text(0.5, 0.5, 'No Gaia proper motion',
                ha='center', va='center', transform=ax.transAxes, color='gray')
    if has_cluster and np.isfinite(pmra + pmde):
        ax.plot([pmra_cl, pmra], [pmde_cl, pmde], 'k-', lw=1.5, alpha=0.30)
    if np.isfinite(chi2):
        ax.text(0.03, 0.97, fr'$\chi^2_{{\rm 5D}}$ = {chi2:.2f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#ecf0f1', ec='#bdc3c7'))
    elif not has_cluster:
        ax.text(0.03, 0.97,
                f'Proper motion = {_fmt(np.hypot(pmra, pmde), ".2f", "mas/yr")}\n'
                f'v_tan = {_fmt(vt, ".1f", "km/s")}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#ecf0f1', ec='#bdc3c7'))
    ax.axhline(0, color='0.65', lw=0.8, zorder=0)
    ax.axvline(0, color='0.65', lw=0.8, zorder=0)
    xs = list(mem_pmra[mem_pm_mask]) + [pmra, pmra_cl]
    ys = list(mem_pmde[mem_pm_mask]) + [pmde, pmde_cl]
    _set_limits_from_points(ax, xs, ys, min_pad=0.8)
    ax.set_xlabel(r'$\mu_\alpha \cos\delta$ (mas yr$^{-1}$)')
    ax.set_ylabel(r'$\mu_\delta$ (mas yr$^{-1}$)')
    ax.set_title('(b) Proper Motion')
    ax.grid(alpha=0.20)
    _dedupe_legend(ax, fontsize=8, loc='best', framealpha=0.88)

    # ======== (c) 视差 ========
    ax = axes[2]
    positive_mem_plx = mem_plx_f[mem_plx_f > 0] if len(mem_plx_f) else np.array([])
    if len(positive_mem_plx) >= 5:
        ax.hist(positive_mem_plx, bins=22, density=False,
                color='#9e9e9e', alpha=0.42, edgecolor='white', lw=0.35,
                label=f'Members (N={len(positive_mem_plx)})', zorder=1)

    if has_cluster and np.isfinite(plx_cl):
        band_specs = [
            (3, '#f6b6aa', 0.18),
            (2, '#ef8a7e', 0.20),
            (1, '#e74c3c', 0.24),
        ]
        for ns, color, alpha_v in band_specs:
            ax.axvspan(plx_cl - ns * s_plx, plx_cl + ns * s_plx,
                       color=color, alpha=alpha_v, zorder=0,
                       label=fr'Cluster $\pm {ns}\sigma_\varpi$' if ns == 3 else None)
        ax.axvline(plx_cl, color='#c0392b', lw=2.0, ls='--',
                   label=fr'{cluster}: $\varpi_c$={plx_cl:.3f} mas', zorder=3)
    if np.isfinite(plx):
        ax.axvline(plx, color='#1565c0', lw=2.4,
                   label=fr'WD: $\varpi$={plx:.3f}$\pm${e_plx_plot:.3f} mas',
                   zorder=4)
        ax.axvspan(plx - e_plx_plot, plx + e_plx_plot,
                   color='#1565c0', alpha=0.10, zorder=2)

    plx_edges = []
    if has_cluster and np.isfinite(plx_cl):
        plx_edges += [plx_cl - 3.5 * s_plx, plx_cl + 3.5 * s_plx]
    if np.isfinite(plx):
        plx_edges += [plx - 3.5 * e_plx_plot, plx + 3.5 * e_plx_plot]
    if len(mem_plx_f) >= 5:
        plx_edges += list(np.nanpercentile(mem_plx_f, [1, 99]))
    if not plx_edges:
        plx_edges = [0.0, 1.0]
    plx_lo = min(plx_edges) - 0.05
    plx_hi = max(plx_edges) + 0.05
    if not np.isfinite(plx_lo + plx_hi) or plx_hi <= plx_lo:
        center = plx if np.isfinite(plx) else 0.5
        plx_lo, plx_hi = center - 0.5, center + 0.5

    if has_cluster and np.isfinite(plx):
        sigma_total_plx = np.sqrt(s_plx**2 + e_plx_plot**2)
        n_sig = abs(plx - plx_cl) / sigma_total_plx if sigma_total_plx > 0 else np.inf
        par_text = (
            fr'$|\Delta\varpi|/\sigma$ = {n_sig:.2f}' + '\n'
            + fr'$\Delta\varpi$ = {plx - plx_cl:+.3f} mas'
        )
    else:
        snr = plx / e_plx if np.isfinite(plx + e_plx) and e_plx > 0 else np.nan
        par_text = f'Parallax S/N = {_fmt(snr, ".1f")}\nDistance = {_fmt(dist_pc, ".1f", "pc")}'
    ax.text(0.03, 0.97, par_text, transform=ax.transAxes, fontsize=10,
            va='top', bbox=dict(boxstyle='round,pad=0.3', fc='#ecf0f1', ec='#bdc3c7'))
    ax.set_xlabel('Parallax (mas)')
    ax.set_ylabel('Member count')
    ax.set_title('(c) Parallax Comparison')
    ax.grid(alpha=0.20)
    ax.set_xlim(plx_lo, plx_hi)
    _dedupe_legend(ax, fontsize=7, loc='best', framealpha=0.88)

    title_parts = [cluster if cluster else 'Gaia 5D Astrometric Check']
    if np.isfinite(ra + dec):
        title_parts.append(f'RA={ra:.4f}  Dec={dec:.4f}')
    if source_id:
        title_parts.append(f'Gaia DR3 {source_id}')
    if match_path:
        title_parts.append(f'Path: {match_path}')
    if np.isfinite(rv_sigma):
        title_parts.append(fr'$\Delta v_r/\sigma$ = {rv_sigma:.1f}')
    if membership:
        title_parts.append(membership)
    title_color = '#1b5e20' if (membership == '6D_matched' or '5D' in membership) else (
        '#e65100' if cluster else '#263238'
    )
    fig.suptitle('  |  '.join(title_parts), fontsize=12,
                 fontweight='bold', color=title_color, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_rv_info(row, out_path: str) -> None:
    """
    绘制 RV / 6D 认证摘要信息板（左: 运动学参数；右: WD 物理参数）。

    Parameters
    ----------
    row : dict-like
        需含 ra, dec, cluster, tier, teff, logg, mass, radius_rsun,
        cooling_age_gyr, cluster_age_gyr, rv_true, rv_true_err, v_grav,
        rv_diff_kms, rv_sigma, cluster_rv, chi2_kin, membership,
        match_note, spectral_type, is_dwd, rv_true_source,
        has_DESI, has_SDSS, has_LAMOST 字段。
    out_path : str
        输出 PNG 路径。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _get(name, default=np.nan):
        try:
            value = row.get(name, default)
        except AttributeError:
            value = row[name] if name in row else default
        return value

    def _num(*names, default=np.nan) -> float:
        for name in names:
            value = _get(name, np.nan)
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            try:
                value = float(value)
            except Exception:
                continue
            if np.isfinite(value):
                return value
        return default

    def _text(*names, default='') -> str:
        for name in names:
            value = _get(name, '')
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            text = str(value).strip()
            if text and text.lower() not in {'nan', 'none'}:
                return text
        return default

    def _f(v, fmt='.1f', unit=''):
        try:
            fv = float(v)
        except Exception:
            return 'N/A'
        return f'{fv:{fmt}} {unit}'.strip() if np.isfinite(fv) else 'N/A'

    def _f_pm(v, err=np.nan, fmt='.3f', unit=''):
        text = _f(v, fmt, unit)
        if text == 'N/A':
            return text
        if np.isfinite(err):
            return text + ' ± ' + _f(err, fmt, unit)
        return text

    ra          = _num('ra', 'RA', '_ra_float')
    dec         = _num('dec', 'DEC', '_dec_float')
    cluster     = _text('cluster', 'Cluster', 'cluster_name')
    teff        = _num('wdopt_teff', 'teff', 'wd_teff')
    logg        = _num('wdopt_logg', 'logg', 'wd_logg')
    mass        = _num('wdopt_mass_msun', 'mass', 'wd_mass_msun', 'm_final_msun')
    radius      = _num('wdopt_radius_rsun', 'radius_rsun', 'wd_radius_rsun', 'radius')
    cooling_age = _num('wdopt_cooling_age_gyr', 'cooling_age_gyr',
                       'wd_cooling_age_gyr', 'gaia_hr_cooling_age_gyr')
    cooling_age_err = _num('wdopt_cooling_age_gyr_err',
                           'cooling_age_gyr_err', 'wd_cooling_age_gyr_err',
                           'gaia_hr_cooling_age_gyr_err')
    cluster_age = _num('cluster_age_gyr')
    if not np.isfinite(cluster_age):
        cluster_age_myr = _num('cluster_age_myr')
        cluster_age = cluster_age_myr / 1000.0 if np.isfinite(cluster_age_myr) else np.nan
    m_init = _num('wdopt_m_progenitor_msun',
                  'm_progenitor', 'm_progenitor_msun', 'm_initial_msun',
                  'progenitor_mass_msun', 'wd_progenitor_mass_msun',
                  'single_star_m_initial_msun', 'gaia_hr_m_progenitor')
    ms_lifetime = _num('wdopt_ms_lifetime_gyr',
                       'ms_lifetime_gyr', 'wd_ms_lifetime_gyr',
                       'single_star_ms_lifetime_gyr',
                       'gaia_hr_ms_lifetime_gyr')
    ms_lifetime_err = _num('wdopt_ms_lifetime_gyr_err',
                           'ms_lifetime_gyr_err', 'wd_ms_lifetime_gyr_err',
                           'single_star_ms_lifetime_gyr_err')
    total_age = _num('wdopt_total_age_with_ms_gyr',
                     'total_age_with_ms_gyr', 'wd_total_age_with_ms_gyr',
                     'single_star_total_age_gyr', 'wd_total_age_gyr',
                     'wdopt_total_age_gyr', 'total_age_gyr',
                     'gaia_hr_total_age_gyr')
    total_age_err = _num('wdopt_total_age_with_ms_gyr_err',
                         'total_age_with_ms_gyr_err', 'wd_total_age_gyr_err',
                         'single_star_total_age_gyr_err')
    if (not np.isfinite(ms_lifetime) or not np.isfinite(m_init)) and np.isfinite(mass):
        try:
            try:
                from .wd_age_methods import single_star_ifmr_age
            except Exception:
                from wd_age_methods import single_star_ifmr_age
            age_info = single_star_ifmr_age(mass, cooling_age)
            if not np.isfinite(m_init):
                m_init = _num(default=age_info.get('m_initial_msun', np.nan))
            if not np.isfinite(ms_lifetime):
                ms_lifetime = _num(default=age_info.get('ms_lifetime_gyr', np.nan))
            if not np.isfinite(total_age):
                total_age = _num(default=age_info.get('total_age_gyr', np.nan))
        except Exception:
            pass
    if not np.isfinite(total_age) and np.isfinite(cooling_age) and np.isfinite(ms_lifetime):
        total_age = cooling_age + ms_lifetime
    cluster_minus_cooling = (
        cluster_age - cooling_age
        if np.isfinite(cluster_age) and np.isfinite(cooling_age) else np.nan
    )
    rv_true     = _num('rv_true_adopted', 'wdopt_rv_true_kms',
                       'rv_true_opt', 'rv_true', 'rv_true_kms')
    rv_err_obs  = _num('rv_obs_err_adopted', 'rv_obs_err_opt',
                       'rv_true_err_adopted', 'rv_true_random_err',
                       'best_rv_err_kms', 'rv_true_err')
    rv_err      = _num('rv_true_err_adopted', 'wdopt_rv_true_err_opt',
                       'wdopt_rv_true_err_kms',
                       'rv_true_err_opt', 'rv_true_err_with_grav_floor', 'rv_true_err',
                       'rv_true_err_adopted')
    v_grav      = _num('v_grav_adopted', 'wdopt_v_grav_kms',
                       'vgrav_preferred', 'v_grav', 'approx_v_grav_kms')
    v_grav_err  = _num('v_grav_err_adopted', 'wdopt_v_grav_err_kms',
                       'vgrav_err_opt', 'v_grav_err', 'approx_v_grav_err_kms',
                       'rv_true_grav_err')
    rv_diff     = _num('delta_rv_adopted', 'delta_rv_opt', 'gravfloor_rv_diff_kms',
                       'rv_diff_kms', 'rv_diff_kms_new')
    rv_sigma    = _num('rv_sigma_adopted', 'rv_sigma_opt', 'gravfloor_rv_sigma',
                       'rv_sigma', 'rv_sigma_new')
    cluster_rv  = _num('gravfloor_cluster_rv', 'cluster_rv', 'cluster_rv_new')
    cluster_rv_err = _num('gravfloor_cluster_rv_err', 'cluster_rv_err',
                          'cluster_rv_err_new')
    cluster_rv_disp = _num('gravfloor_cluster_rv_dispersion',
                           'cluster_rv_dispersion', 'cluster_rv_sigma',
                           default=DEFAULT_CLUSTER_RV_DISPERSION_KMS)
    sigma_terms = []
    if np.isfinite(rv_err):
        sigma_terms.append(rv_err)
        if np.isfinite(cluster_rv_err):
            sigma_terms.append(cluster_rv_err)
        if np.isfinite(cluster_rv_disp):
            sigma_terms.append(cluster_rv_disp)
    sigma_total = (
        np.sqrt(np.sum(np.square(sigma_terms))) if sigma_terms else np.nan
    )
    if np.isfinite(rv_diff) and np.isfinite(sigma_total) and sigma_total > 0:
        rv_sigma = rv_diff / sigma_total
    chi2_kin    = _num('chi2_kin')
    membership  = _text('membership')
    match_note  = _text('match_note', 'match_note_new')
    sp_type     = _text('spectral_type', 'wd_spectral_type', default='DA')
    is_dwd      = _truthy(_get('is_dwd', False))
    tier        = _text('tier')
    rv_source   = _text('rv_true_source', 'best_rv_source', 'cluster_rv_source',
                        'cluster_rv_source_new')
    match_path  = _text('match_path', 'match_path_new')
    is_6d = (
        _truthy(row.get('is_6d_matched', False))
        or _truthy(row.get('is_6d_matched_new', False))
    )
    match_quality = str(
        row.get('rv_match_quality', row.get('rv_match_quality_new', ''))
        or ''
    ).strip().lower()
    if not match_quality or match_quality in {'nan', 'none'}:
        match_quality = _rv_match_quality(rv_diff, rv_sigma, rv_err)
    is_6d_strict = (
        is_6d
        and (
            _truthy(row.get('is_6d_strict', False))
            or _truthy(row.get('is_6d_strict_new', False))
            or match_quality == 'strict'
        )
    )
    is_6d_borderline = (
        is_6d
        and not is_6d_strict
        and (
            _truthy(row.get('is_6d_borderline', False))
            or _truthy(row.get('is_6d_borderline_new', False))
            or match_quality in {'borderline', 'unknown'}
        )
    )
    if is_6d:
        membership = '6D_matched' if is_6d_strict else '6D_candidate'
    if is_6d and not match_path:
        match_path = str(row.get('match_path_new', '') or '')
    if is_6d_borderline and match_note.startswith('6D matched'):
        match_note = match_note.replace('6D matched', '6D candidate', 1)
    if np.isfinite(mass) and np.isfinite(radius) and radius > 0:
        expected_v_grav = 0.635 * mass / radius
        if (not np.isfinite(v_grav)
                or v_grav < 0.2 * expected_v_grav
                or v_grav > 5.0 * expected_v_grav
                or abs(v_grav - expected_v_grav) > max(5.0, 0.5 * expected_v_grav)):
            v_grav = expected_v_grav
    dyn_flags   = str(row.get('dynamics_flags', ''))
    if not dyn_flags or dyn_flags == 'nan':
        try:
            dyn_flags = ';'.join(evaluate_dynamics_flags(row)['dynamics_flags'])
        except Exception:
            dyn_flags = ''

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    fig.patch.set_facecolor('#f5f5f5')

    # ---- 左: RV & 6D ----
    ax = axes[0]
    ax.axis('off')
    color_bg = '#e8f5e9' if membership == '6D_matched' else (
        '#fff8e1' if membership == '6D_candidate' else '#fff3e0')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                facecolor=color_bg, edgecolor='#aaa', linewidth=1.5))
    rv_val = _f_pm(rv_true, rv_err, '.1f', 'km/s')
    rv_obs_val = _f_pm(rv_true, rv_err_obs, '.1f', 'km/s')
    cluster_rv_val = _f_pm(cluster_rv, cluster_rv_err, '.1f', 'km/s')

    rows_rv = [
        ('Membership',  membership,
         '#1b5e20' if membership == '6D_matched' else '#e65100'),
        ('Match path',  match_path if match_path else 'N/A',
         '#1b5e20' if match_path else '#999'),
        ('Cluster',     cluster,           '#333'),
        ('Tier',        tier,              '#333'),
        ('chi2_kin',    _f(chi2_kin, '.3f'), '#333'),
        ('',            '',                ''),
        ('RV_true total err', rv_val,      '#1a237e'),
        ('RV_true obs err', rv_obs_val,    '#333'),
        ('V_grav',      _f_pm(v_grav, v_grav_err, '.2f', 'km/s'), '#333'),
        ('Cluster RV',  cluster_rv_val,    '#333'),
        ('Cluster sigma_int', _f(cluster_rv_disp, '.1f', 'km/s'), '#333'),
        ('sigma_total', _f(sigma_total, '.1f', 'km/s'), '#1a237e'),
        ('|ΔRV|',       _f(rv_diff,  '.1f', 'km/s'), '#333'),
        ('ΔRV / sigma_total', _f(rv_sigma, '.2f') + ' σ',
         '#1b5e20' if match_quality == 'strict'
         else ('#e65100' if match_quality == 'borderline' else '#b71c1c')),
        ('RV quality',  match_quality if match_quality else 'N/A',
         '#1b5e20' if match_quality == 'strict'
         else ('#e65100' if match_quality == 'borderline' else '#666')),
        ('Strong rule', 'S<2, |ΔRV|<20, RVerr<20', '#555'),
        ('',            '',                ''),
        ('Note',        match_note[:55] if match_note else '', '#666'),
        ('RV source',   rv_source[:40]  if rv_source  else '', '#666'),
        ('Dyn flags',   dyn_flags[:55] if dyn_flags else 'none',
         '#b71c1c' if dyn_flags else '#2e7d32'),
    ]

    y = 0.95
    for key, val, color in rows_rv:
        if not key:
            y -= 0.025
            continue
        ax.text(0.04, y, key + ':', fontsize=9.2, ha='left', va='top',
                transform=ax.transAxes, color='#555', fontweight='bold')
        ax.text(0.46, y, val,        fontsize=9.2, ha='left', va='top',
                transform=ax.transAxes, color=color)
        y -= 0.050
    ax.set_title('6D Kinematics & RV', fontsize=12, pad=8)

    # ---- 右: WD 参数 ----
    ax = axes[1]
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                facecolor='#e3f2fd', edgecolor='#aaa', linewidth=1.5))

    cooling_age_gt_cluster = (
        np.isfinite(cooling_age) and np.isfinite(cluster_age)
        and cooling_age > cluster_age
    )
    total_age_gt_cluster = (
        np.isfinite(total_age) and np.isfinite(cluster_age)
        and total_age > cluster_age
    )
    cooling_gt_text = (
        'Yes' if cooling_age_gt_cluster else
        ('No' if np.isfinite(cooling_age) and np.isfinite(cluster_age) else 'N/A')
    )
    total_gt_text = (
        'Yes' if total_age_gt_cluster else
        ('No' if np.isfinite(total_age) and np.isfinite(cluster_age) else 'N/A')
    )
    cooling_gt_color = '#b71c1c' if cooling_age_gt_cluster else (
        '#1b5e20' if cooling_gt_text == 'No' else '#666'
    )
    total_gt_color = '#b71c1c' if total_age_gt_cluster else (
        '#1b5e20' if total_gt_text == 'No' else '#666'
    )
    age_clock_note = 'single-star IFMR clock'
    if np.isfinite(mass) and mass < 0.55 and not np.isfinite(ms_lifetime):
        age_clock_note = 'MS lifetime N/A: low-mass/binary WD'

    rows_wd = [
        ('Spectral Type', sp_type,                    '#333'),
        ('Teff',          _f(teff, '.0f', 'K'),        '#333'),
        ('log g',         _f(logg, '.2f'),              '#333'),
        ('Mass',          _f(mass, '.3f', 'M⊙'),       '#333'),
        ('Radius',        _f(radius, '.4f', 'R⊙'),     '#333'),
        ('',              '',                          ''),
        ('Cooling age',   _f_pm(cooling_age, cooling_age_err, '.3f', 'Gyr'), '#333'),
        ('M_init (IFMR)', _f(m_init, '.2f', 'M⊙'), '#333'),
        ('MS lifetime',   _f_pm(ms_lifetime, ms_lifetime_err, '.3f', 'Gyr'), '#333'),
        ('WD total age',  _f_pm(total_age, total_age_err, '.3f', 'Gyr'),
         '#b71c1c' if total_age_gt_cluster else '#333'),
        ('Cluster age',   _f(cluster_age, '.3f', 'Gyr'), '#333'),
        ('Cluster - t_cool', _f(cluster_minus_cooling, '.3f', 'Gyr'),
         '#b71c1c' if np.isfinite(cluster_minus_cooling) and cluster_minus_cooling <= 0 else '#333'),
        ('t_cool > Cluster?', cooling_gt_text, cooling_gt_color),
        ('t_total > Cluster?', total_gt_text, total_gt_color),
        ('Age method',    age_clock_note[:38], '#666'),
        ('',              '',                          ''),
        ('Is DWD',        'Yes' if is_dwd else 'No',
         '#b71c1c' if is_dwd else '#2e7d32'),
        ('Has DESI',      'Yes' if row.get('has_DESI')   else 'No', '#333'),
        ('Has SDSS',      'Yes' if row.get('has_SDSS')   else 'No', '#333'),
        ('Has LAMOST',    'Yes' if row.get('has_LAMOST') else 'No', '#333'),
    ]

    y = 0.95
    for key, val, color in rows_wd:
        if not key:
            y -= 0.025
            continue
        ax.text(0.04, y, key + ':', fontsize=9.2, ha='left', va='top',
                transform=ax.transAxes, color='#555', fontweight='bold')
        ax.text(0.54, y, val,        fontsize=9.2, ha='left', va='top',
                transform=ax.transAxes, color=color)
        y -= 0.050
    ax.set_title('WD Parameters', fontsize=12, pad=8)

    title_coord = (
        f'RA={ra:.4f}  Dec={dec:.4f}'
        if np.isfinite(ra) and np.isfinite(dec) else 'RA/Dec=N/A'
    )
    fig.suptitle(f'{title_coord}  |  {cluster}',
                 fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_total_hrd(
    df_6d: pd.DataFrame,
    bg_bp: np.ndarray,
    bg_mg: np.ndarray,
    out_path: str,
) -> None:
    """
    所有 6D 认证源的总 HRD，每个星团用不同颜色标记（★ 形状）。

    Parameters
    ----------
    df_6d : pd.DataFrame
        is_6d_matched=True 的子集，需含 parallax, phot_g_mean_mag, bp_rp, cluster 列。
    bg_bp, bg_mg : np.ndarray
        Gaia 背景星的 BP-RP 和 M_G 数组。
    out_path : str
        输出 PNG 路径。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 12))

    if len(bg_bp) > 0:
        ax.hist2d(bg_bp, bg_mg, bins=300,
                  cmap='Greys', norm=matplotlib.colors.LogNorm(),
                  alpha=0.85, zorder=1)

    clusters = [c for c in df_6d['cluster'].unique() if pd.notna(c)]
    cmap_obj = plt.cm.get_cmap('tab20', max(len(clusters), 1))
    cluster_color = {c: cmap_obj(i) for i, c in enumerate(clusters)}
    seen: set = set()

    for _, r in df_6d.iterrows():
        try:
            plx = float(r['parallax'])
            g   = float(r['phot_g_mean_mag'])
            bp  = float(r['bp_rp'])
            if plx <= 0 or not np.isfinite(g) or not np.isfinite(bp):
                continue
            M_G = g + 5 * np.log10(plx / 1000.0) + 5
            cl  = str(r['cluster'])
            color = cluster_color.get(cl, 'red')
            lbl = cl if cl not in seen else None
            seen.add(cl)
            ax.scatter([bp], [M_G], c=[color], s=80,
                       edgecolors='black', linewidths=0.5,
                       zorder=10, marker='*', label=lbl, alpha=0.9)
        except (ValueError, TypeError):
            continue

    ax.set_xlabel('BP − RP (mag)', fontsize=13)
    ax.set_ylabel(r'$M_G$ (mag)', fontsize=13)
    ax.set_title(
        'HR Diagram — All 6D Confirmed WDs\n(colored by cluster, ★ = 6D confirmed)',
        fontsize=13,
    )
    ax.invert_yaxis()
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(16, -4)
    ax.grid(True, alpha=0.3)
    ax.text(0.3,  -2, 'Main Sequence', fontsize=9, color='gray', rotation=55, ha='center')
    ax.text(2.5,  -1, 'Giants',        fontsize=9, color='gray')
    ax.text(-0.2, 10, 'White\nDwarfs', fontsize=9, color='gray')
    ax.legend(fontsize=7, loc='lower left', ncol=2, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  total HRD -> {out_path}")


def plot_sed(ra: float, dec: float, out_path: str) -> bool:
    """
    调用 astro_toolbox.sed.quick_sed 生成 SED 图。

    Returns True if plot was generated, False otherwise.
    """
    try:
        from astro_toolbox.sed import quick_sed
        result = quick_sed(ra, dec, save_path=out_path)
        return result is not None
    except Exception as e:
        print(f"    SED failed: {e}")
        return False


def make_6d_plots(
    df: pd.DataFrame,
    output_dir: str,
    gaia_bg_csv: str,
    results_base: str | None = None,
    merged_df: pd.DataFrame | None = None,
) -> None:
    """
    对 df 中所有 is_6d_matched=True 的源批量生成五张图 + 一个 CSV。

    生成文件（均平铺在 output_dir/）:
      {cluster}_ra=XXX_dec=YYY_spectrum.png   光谱 + Koester 模型
      {cluster}_ra=XXX_dec=YYY_ztf.png        ZTF 光变曲线
      {cluster}_ra=XXX_dec=YYY_hrd.png        HR 图 (Hunt+2023 成员星)
      {cluster}_ra=XXX_dec=YYY_rv.png         RV / 6D 信息板
      {cluster}_ra=XXX_dec=YYY_sed.png        SED (如有数据)
      {cluster}_ra=XXX_dec=YYY_data.csv       单源关键参数
      all_6d_hrd.png                          所有源总 HRD
    """
    os.makedirs(output_dir, exist_ok=True)

    df_6d = df[df['is_6d_matched'] == True].drop_duplicates(subset=['source_id']).copy()
    print(f"6D confirmed unique sources: {len(df_6d)}")

    # 背景星
    bg = pd.read_csv(gaia_bg_csv)
    bp_arr = bg['bp_rp'].values.astype(float)
    mg_arr = bg['absmag'].values.astype(float)
    mask_bg = np.isfinite(bp_arr) & np.isfinite(mg_arr)
    bg_bp, bg_mg = bp_arr[mask_bg], mg_arr[mask_bg]
    print(f"Background: {len(bg_bp)} stars")

    # 预加载 Koester 模板
    global _templates_cache
    try:
        from astro_toolbox.sed import _load_koester2_templates
        _templates_cache = _load_koester2_templates()
        print(f"Koester templates: {len(_templates_cache)} loaded")
    except Exception as e:
        print(f"Koester template load failed: {e}")

    # 总 HRD
    print("\n[1] Total HRD...")
    plot_total_hrd(df_6d, bg_bp, bg_mg,
                   os.path.join(output_dir, 'all_6d_hrd.png'))

    # 单源列集合
    csv_cols = [
        'source_id', 'ra', 'dec', 'cluster', 'tier', 'parallax',
        'phot_g_mean_mag', 'bp_rp', 'spectral_type', 'teff', 'teff_err',
        'logg', 'logg_err', 'mass', 'radius_rsun', 'cooling_age_gyr',
        'cluster_age_gyr', 'wd_age_gt_cluster',
        'is_dwd', 'rv_true', 'rv_true_err', 'v_grav', 'rv_true_source',
        'is_6d_matched', 'rv_diff_kms', 'rv_sigma', 'cluster_rv',
        'chi2_kin', 'membership', 'match_note',
        'has_DESI', 'has_SDSS', 'has_LAMOST',
    ]
    csv_cols = [c for c in csv_cols if c in df_6d.columns]

    print(f"\n[2] Individual plots + CSV ({len(df_6d)} sources)...")
    ok, fail = 0, 0

    for i, (_, r) in enumerate(df_6d.iterrows()):
        ra  = float(r['ra'])
        dec = float(r['dec'])
        cluster = str(r['cluster']) if pd.notna(r.get('cluster')) else ''
        pre = _file_prefix(cluster, ra, dec)
        print(f"\n  [{i+1}/{len(df_6d)}] {pre}  cluster={r['cluster']}")

        # 找 result_dir
        result_dir = None
        if results_base is not None and merged_df is not None:
            cluster_clean = cluster.replace(' ', '_').replace('/', '_')
            dists = np.sqrt(
                (merged_df['ra'] - ra)**2 + (merged_df['dec'] - dec)**2
            )
            idx = dists.idxmin()
            if dists[idx] <= 0.01 and 'source_id' in merged_df.columns:
                sid = int(merged_df.loc[idx, 'source_id'])
                dn  = (f'{cluster_clean}_Gaia_{sid}'
                       if cluster_clean else f'Gaia_{sid}')
                path = os.path.join(results_base, dn)
                if os.path.exists(path):
                    result_dir = path

        try:
            plot_spectrum(r, result_dir,
                          os.path.join(output_dir, pre + '_spectrum.png'))
            print("    spectrum OK")
        except Exception:
            traceback.print_exc()
            fail += 1

        try:
            from astro_toolbox.ztf import query_lightcurve
            ztf_lc = query_lightcurve(ra, dec)
        except Exception as e:
            print(f"    ZTF query failed: {e}")
            ztf_lc = None

        try:
            plot_ztf(r, ztf_lc,
                     os.path.join(output_dir, pre + '_ztf.png'))
            if ztf_lc is not None:
                dfs = []
                for band in ('g', 'r', 'i'):
                    if band in ztf_lc:
                        tmp = ztf_lc[band].copy()
                        tmp['band'] = band
                        dfs.append(tmp)
                if dfs:
                    pd.concat(dfs, ignore_index=True).to_csv(
                        os.path.join(output_dir, pre + '_ztf.csv'), index=False
                    )
            print("    ZTF OK")
        except Exception:
            traceback.print_exc()
            fail += 1

        try:
            plot_hrd(r, bg_bp, bg_mg,
                     os.path.join(output_dir, pre + '_hrd.png'))
            print("    HRD OK")
        except Exception:
            traceback.print_exc()
            fail += 1

        try:
            plot_rv_info(r,
                         os.path.join(output_dir, pre + '_rv.png'))
            print("    RV info OK")
        except Exception:
            traceback.print_exc()
            fail += 1

        try:
            sed_path = os.path.join(output_dir, pre + '_sed.png')
            if plot_sed(ra, dec, sed_path):
                print("    SED OK")
            else:
                print("    SED skipped (no data)")
        except Exception:
            traceback.print_exc()
            fail += 1

        try:
            df_6d[df_6d.index == r.name][csv_cols].to_csv(
                os.path.join(output_dir, pre + '_data.csv'), index=False
            )
            print("    data CSV OK")
            ok += 1
        except Exception:
            traceback.print_exc()
            fail += 1

    print(f"\n{'='*60}")
    print(f"Done!  success={ok}  errors={fail}")
    print(f"Output dir: {output_dir}")
