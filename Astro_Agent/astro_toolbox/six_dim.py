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
    'plot_rv_info',
    'plot_sed',
    'plot_total_hrd',
    'make_6d_plots',
]


# ──────────────────────────────────────────────────────────────────
#  6D 匹配逻辑
# ──────────────────────────────────────────────────────────────────

def check_6d_match(
    row: dict,
    rv_true: float,
    rv_err: float | None = None,
    *,
    max_rv_err: float = 50.0,
    max_rv_diff: float = 100.0,
    chi2_kin_limit: float = 20.0,
    rv_nsigma: float = 3.0,
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
        'match_note': '',
    }

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

    # 查星团 RV
    cl_rv, cl_rv_err = np.nan, np.nan
    if cluster_cache is not None:
        for cl in cluster_cache:
            if cl['Name'].strip() == cluster_name.strip():
                cl_rv = cl['RV']
                cl_rv_err = cl['e_RV']
                break
        if not np.isfinite(cl_rv):
            for cl in cluster_cache:
                if cluster_name.strip().lower() in cl['Name'].strip().lower():
                    cl_rv = cl['RV']
                    cl_rv_err = cl['e_RV']
                    break

    if not np.isfinite(cl_rv):
        result['match_note'] = 'cluster_has_no_rv'
        return result

    result['cluster_rv'] = cl_rv

    dv = abs(rv_true - cl_rv)
    result['rv_diff_kms'] = dv

    if dv > max_rv_diff:
        result['match_note'] = f'rv_diff_too_large (dv={dv:.1f} km/s > {max_rv_diff:.0f})'
        return result

    if not np.isfinite(cl_rv_err) or cl_rv_err <= 0:
        cl_rv_err = 5.0
    if rv_err is None or not np.isfinite(rv_err) or rv_err <= 0:
        rv_err = 20.0
    sigma_total = np.sqrt(rv_err**2 + cl_rv_err**2)
    result['rv_sigma'] = dv / sigma_total if sigma_total > 0 else np.inf

    chi2_kin = row.get('chi2_kin', np.nan)
    if isinstance(chi2_kin, str):
        try:
            chi2_kin = float(chi2_kin)
        except ValueError:
            chi2_kin = np.nan

    has_5d = np.isfinite(chi2_kin) and chi2_kin < chi2_kin_limit
    has_backtrack = bool(row.get('orbit_within_tidal', False))
    rv_ok = result['rv_sigma'] < rv_nsigma

    if has_5d and rv_ok:
        result['is_6d_matched'] = True
        result['match_path'] = '5D+RV'
        result['match_note'] = (
            f'6D matched via 5D+RV (chi2={chi2_kin:.1f}, '
            f'dv={dv:.1f} km/s, {result["rv_sigma"]:.1f}σ)'
        )
    elif has_backtrack and rv_ok:
        result['is_6d_matched'] = True
        result['match_path'] = 'backtrack+RV'
        result['match_note'] = (
            f'6D matched via backtrack+RV '
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
            result['match_note'] = f'RV ok but 5D poor (chi2_kin={chi2_str})'
        else:
            result['match_note'] = f'both 5D and RV poor (chi2_kin={chi2_str})'

    return result


def recalc_membership(row: dict) -> str:
    """根据 is_6d_matched / match_path / orbit_within_tidal / tier 重算 membership。"""
    if row.get('is_6d_matched'):
        return '6D_matched'
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

    rv_true = _val('rv_true')
    rv_err = _val('rv_true_err')
    rv_diff = _val('rv_diff_kms')
    rv_sigma = _val('rv_sigma')
    chi2_kin = _val('chi2_kin')
    cooling_age = _val('cooling_age_gyr')
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
    if not is_6d:
        flags.append('NOT_6D_CONFIRMED')

    if any(f in flags for f in ('RV_SIGMA_MISMATCH', 'RV_DIFF_LARGE',
                                'COOLING_AGE_GT_CLUSTER_AGE')):
        severity = 'high'
    elif any(f in flags for f in ('RV_ERROR_LARGE', 'KINEMATIC_CHI2_HIGH',
                                  'COOLING_AGE_CLOSE_TO_CLUSTER_AGE')):
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
_HUNT_MEMBERS_DIR = '/Users/ljm/Desktop/csst/星团/Hunt+2023/individual-oc'

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

    def _f(v, fmt='.1f', unit=''):
        return f'{v:{fmt}} {unit}'.strip() if (v is not None and np.isfinite(float(v))) else 'N/A'

    ra          = float(row['ra'])
    dec         = float(row['dec'])
    cluster     = str(row.get('cluster', ''))
    teff        = float(row['teff'])         if pd.notna(row.get('teff'))         else np.nan
    logg        = float(row['logg'])         if pd.notna(row.get('logg'))         else np.nan
    mass        = float(row['mass'])         if pd.notna(row.get('mass'))         else np.nan
    radius      = float(row['radius_rsun']) if pd.notna(row.get('radius_rsun'))  else np.nan
    cooling_age = float(row['cooling_age_gyr']) if pd.notna(row.get('cooling_age_gyr')) else np.nan
    cluster_age = float(row['cluster_age_gyr'])  if pd.notna(row.get('cluster_age_gyr'))  else np.nan
    rv_true     = float(row['rv_true'])      if pd.notna(row.get('rv_true'))      else np.nan
    rv_err      = float(row['rv_true_err'])  if pd.notna(row.get('rv_true_err'))  else np.nan
    v_grav      = float(row['v_grav'])       if pd.notna(row.get('v_grav'))       else np.nan
    rv_diff     = float(row['rv_diff_kms'])  if pd.notna(row.get('rv_diff_kms'))  else np.nan
    rv_sigma    = float(row['rv_sigma'])     if pd.notna(row.get('rv_sigma'))     else np.nan
    cluster_rv  = float(row['cluster_rv'])   if pd.notna(row.get('cluster_rv'))   else np.nan
    chi2_kin    = float(row['chi2_kin'])     if pd.notna(row.get('chi2_kin'))     else np.nan
    membership  = str(row.get('membership', ''))
    match_note  = str(row.get('match_note', ''))
    sp_type     = str(row.get('spectral_type', 'DA'))
    is_dwd      = bool(row.get('is_dwd', False))
    tier        = str(row.get('tier', ''))
    rv_source   = str(row.get('rv_true_source', ''))
    match_path  = str(row.get('match_path', ''))
    dyn_flags   = str(row.get('dynamics_flags', ''))
    if not dyn_flags or dyn_flags == 'nan':
        try:
            dyn_flags = ';'.join(evaluate_dynamics_flags(row)['dynamics_flags'])
        except Exception:
            dyn_flags = ''

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor('#f5f5f5')

    # ---- 左: RV & 6D ----
    ax = axes[0]
    ax.axis('off')
    color_bg = '#e8f5e9' if '6D' in membership else '#fff3e0'
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                facecolor=color_bg, edgecolor='#aaa', linewidth=1.5))
    rv_val = _f(rv_true, '.1f', 'km/s')
    if np.isfinite(rv_err):
        rv_val += ' ± ' + _f(rv_err, '.1f', 'km/s')

    rows_rv = [
        ('Membership',  membership,
         '#1b5e20' if '6D' in membership else '#e65100'),
        ('Match path',  match_path if match_path else 'N/A',
         '#1b5e20' if match_path else '#999'),
        ('Cluster',     cluster,           '#333'),
        ('Tier',        tier,              '#333'),
        ('chi2_kin',    _f(chi2_kin, '.3f'), '#333'),
        ('',            '',                ''),
        ('RV_true',     rv_val,            '#1a237e'),
        ('V_grav',      _f(v_grav,  '.2f', 'km/s'), '#333'),
        ('Cluster RV',  _f(cluster_rv, '.1f', 'km/s'), '#333'),
        ('|ΔRV|',       _f(rv_diff,  '.1f', 'km/s'), '#333'),
        ('ΔRV / σ',     _f(rv_sigma, '.2f') + ' σ',
         '#1a237e' if np.isfinite(rv_sigma) and rv_sigma < 3 else '#b71c1c'),
        ('',            '',                ''),
        ('Note',        match_note[:55] if match_note else '', '#666'),
        ('RV source',   rv_source[:40]  if rv_source  else '', '#666'),
        ('Dyn flags',   dyn_flags[:55] if dyn_flags else 'none',
         '#b71c1c' if dyn_flags else '#2e7d32'),
    ]

    y = 0.95
    for key, val, color in rows_rv:
        if not key:
            y -= 0.04
            continue
        ax.text(0.04, y, key + ':', fontsize=10, ha='left', va='top',
                transform=ax.transAxes, color='#555', fontweight='bold')
        ax.text(0.42, y, val,        fontsize=10, ha='left', va='top',
                transform=ax.transAxes, color=color)
        y -= 0.068
    ax.set_title('6D Kinematics & RV', fontsize=12, pad=8)

    # ---- 右: WD 参数 ----
    ax = axes[1]
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                facecolor='#e3f2fd', edgecolor='#aaa', linewidth=1.5))

    age_ok = (np.isfinite(cooling_age) and np.isfinite(cluster_age)
              and cooling_age < cluster_age)

    rows_wd = [
        ('Spectral Type', sp_type,                    '#333'),
        ('Teff',          _f(teff, '.0f', 'K'),        '#333'),
        ('log g',         _f(logg, '.2f'),              '#333'),
        ('Mass',          _f(mass, '.3f', 'M⊙'),       '#333'),
        ('Radius',        _f(radius, '.4f', 'R⊙'),     '#333'),
        ('',              '',                          ''),
        ('Cooling age',   _f(cooling_age, '.3f', 'Gyr'), '#333'),
        ('Cluster age',   _f(cluster_age, '.3f', 'Gyr'), '#333'),
        ('Age < Cluster?', 'Yes ✓' if age_ok else 'No ✗',
         '#1b5e20' if age_ok else '#b71c1c'),
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
            y -= 0.04
            continue
        ax.text(0.04, y, key + ':', fontsize=10, ha='left', va='top',
                transform=ax.transAxes, color='#555', fontweight='bold')
        ax.text(0.52, y, val,        fontsize=10, ha='left', va='top',
                transform=ax.transAxes, color=color)
        y -= 0.063
    ax.set_title('WD Parameters', fontsize=12, pad=8)

    fig.suptitle(f'RA={ra:.4f}  Dec={dec:.4f}  |  {cluster}',
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
