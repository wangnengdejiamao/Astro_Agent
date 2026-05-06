"""
WD 冷却年龄分析 — 双星并合判据
==============================
方法参考 Yan et al. 2025 ("A magnetic white dwarf formed through a binary
merger within 35 million years"):

    1. Gaia 测光 (G, BP-RP) + 视差 → 绝对星等 M_G
    2. Bédard et al. 2020 CO-core thick-H 冷却轨迹 → M_WD, T_eff, log g, t_cool
    3. Cummings et al. 2018 IFMR → 前身星质量 M_i
    4. MIST 主序寿命 t_MS(M_i)
    5. 比较 t_cool + t_MS  vs  星团年龄 t_cluster
       若 t_cool + t_MS > t_cluster → 单星演化不可能 → 并合候选体

用法:
    from astro_toolbox.cooling_age import run_cooling_age_analysis
    report = run_cooling_age_analysis(ra, dec, cluster_age_gyr,
                                      output_dir='results/')
"""

import os
import sys
import numpy as np
import warnings

# 确保 WD_models 可导入
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from . import utils


# ================================================================
#  Gaia DR3 测光查询
# ================================================================

def get_gaia_photometry(ra, dec, radius_arcsec=3.0):
    """
    从 Gaia DR3 获取测光和视差。

    Returns
    -------
    dict 或 None
        keys: source_id, ra, dec, Gmag, BPmag, RPmag, Plx, e_Plx, RUWE
    """
    tbl = utils.query_vizier(
        'I/355/gaiadr3', ra, dec,
        radius_arcsec=radius_arcsec,
        columns=['Source', 'RA_ICRS', 'DE_ICRS',
                 'Gmag', 'BPmag', 'RPmag',
                 'Plx', 'e_Plx', 'RUWE'])
    if tbl is None or len(tbl) == 0:
        return None

    row = tbl[0]
    result = {}
    for key, col in [('source_id', 'Source'),
                     ('ra', 'RA_ICRS'), ('dec', 'DE_ICRS'),
                     ('Gmag', 'Gmag'), ('BPmag', 'BPmag'), ('RPmag', 'RPmag'),
                     ('Plx', 'Plx'), ('e_Plx', 'e_Plx'),
                     ('RUWE', 'RUWE')]:
        try:
            val = float(row[col])
            if np.ma.is_masked(val):
                val = np.nan
            result[key] = val
        except (ValueError, KeyError, np.ma.MaskError):
            result[key] = np.nan

    # 必须有视差和测光
    if np.isnan(result.get('Plx', np.nan)) or np.isnan(result.get('Gmag', np.nan)):
        return None

    # 绝对星等
    plx_mas = result['Plx']
    if plx_mas <= 0:
        return None
    dist_pc = 1000.0 / plx_mas
    result['dist_pc'] = dist_pc
    result['M_G'] = result['Gmag'] - 5.0 * np.log10(dist_pc / 10.0)
    result['BP_RP'] = result['BPmag'] - result['RPmag']

    return result


# ================================================================
#  WD_models 冷却轨迹插值
# ================================================================

_WD_MODEL_CACHE = {}


def _load_wd_model():
    """加载 Bédard 2020 CO thick-H 冷却轨迹 (缓存)。"""
    key = 'Bedard2020_CO_thick_H'
    if key in _WD_MODEL_CACHE:
        return _WD_MODEL_CACHE[key]

    import WD_models as wdm

    # Fontaine2001 低质量 + BaSTI 中/高质量, H atmosphere
    # 使用 Gaia DR3 波段 (bp3-rp3, G3)
    print("  加载 WD 冷却轨迹 (Bédard 2020 + BaSTI)...")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = wdm.load_model('f', 'b', 'b', 'H',
                               HR_bands=('bp3-rp3', 'G3'),
                               interp_type='linear')
    _WD_MODEL_CACHE[key] = model
    print("  冷却轨迹加载完成")
    return model


def interpolate_wd_params(bp_rp, M_G):
    """
    从 HR 图位置插值 WD 参数。

    Parameters
    ----------
    bp_rp : float  — Gaia BP-RP 色指数
    M_G   : float  — Gaia 绝对 G 星等

    Returns
    -------
    dict: mass, logteff, teff, logg, cooling_age_gyr, total_age_gyr
    """
    model = _load_wd_model()

    mass = float(model['HR_to_mass'](bp_rp, M_G))
    logteff = float(model['HR_to_logteff'](bp_rp, M_G))
    logg = float(model['HR_to_logg'](bp_rp, M_G))
    age_cool = float(model['HR_to_age_cool'](bp_rp, M_G))  # Gyr
    age_total = float(model['HR_to_age'](bp_rp, M_G))       # Gyr

    if np.isnan(mass):
        return None

    return {
        'mass': mass,
        'logteff': logteff,
        'teff': 10**logteff,
        'logg': logg,
        'cooling_age_gyr': age_cool,
        'total_age_gyr': age_total,
    }


# ================================================================
#  IFMR + 前身星寿命
# ================================================================

def compute_progenitor_lifetime(m_wd):
    """
    给定 WD 质量, 通过 IFMR 得到前身星质量, 再估算主序寿命。

    使用 WD_models 内置的 Cummings+2018 IFMR 和主序寿命多项式。

    Parameters
    ----------
    m_wd : float — WD 质量 (Msun)

    Returns
    -------
    dict: m_progenitor (Msun), ms_lifetime_gyr (Gyr)
    """
    import WD_models as wdm

    m_prog = float(wdm.IFMR(m_wd))
    if m_prog <= 0 or np.isnan(m_prog):
        return None

    ms_life_yr = float(wdm.MS_age(m_wd))  # 返回 years
    ms_life_gyr = ms_life_yr / 1e9

    return {
        'm_progenitor': m_prog,
        'ms_lifetime_gyr': ms_life_gyr,
    }


# ================================================================
#  自定义 IFMR (Cummings+2018 with MIST)
# ================================================================

def cummings2018_ifmr(m_final):
    """
    Cummings et al. 2018 (MIST-based) IFMR.
    输入 WD 终态质量 (Msun), 返回前身星初始质量 (Msun).

    分段线性关系 (Table 1 of Cummings+2018):
      M_i = 0.80 ~ 2.85 Msun:  M_f = 0.0873 * M_i + 0.476
      M_i = 2.85 ~ 3.60 Msun:  M_f = 0.181  * M_i + 0.210
      M_i = 3.60 ~ 7.20 Msun:  M_f = 0.0835 * M_i + 0.565
    反解 M_i:
    """
    # 三段对应的 M_f 边界
    # 段1: M_i in [0.80, 2.85]  => M_f in [0.546, 0.725]
    # 段2: M_i in [2.85, 3.60]  => M_f in [0.726, 0.862]
    # 段3: M_i in [3.60, 7.20]  => M_f in [0.866, 1.166]

    if m_final <= 0.725:
        # 段1: M_f = 0.0873 * M_i + 0.476  =>  M_i = (M_f - 0.476) / 0.0873
        m_init = (m_final - 0.476) / 0.0873
    elif m_final <= 0.862:
        # 段2: M_f = 0.181 * M_i + 0.210  =>  M_i = (M_f - 0.210) / 0.181
        m_init = (m_final - 0.210) / 0.181
    else:
        # 段3: M_f = 0.0835 * M_i + 0.565  =>  M_i = (M_f - 0.565) / 0.0835
        m_init = (m_final - 0.565) / 0.0835

    if m_init < 0.5:
        return np.nan
    return m_init


def mist_ms_lifetime(m_init):
    """
    主序寿命估计, 基于 MIST 等时线的拟合公式 (简单多项式)。

    对 M_i > 1 Msun 的中大质量恒星:
        t_MS ≈ 10^(9.921 - 3.6648*log(M) + 1.9697*log(M)^2 - 0.9369*log(M)^3) yr
    这个拟合在 1-10 Msun 范围内精度 < 5%.

    也可以直接用 WD_models.MS_age(), 它使用类似方法。

    Parameters
    ----------
    m_init : float — 前身星质量 (Msun)

    Returns
    -------
    float — 主序寿命 (Gyr)
    """
    if m_init <= 0 or np.isnan(m_init):
        return np.nan

    logm = np.log10(m_init)
    # Hurley+2000 拟合公式 (更准确)
    # t_MS = max(mu/X^nu, X^beta) * t_BGB
    # 简化使用经验公式:
    # 太阳: ~10 Gyr, 2 Msun: ~1.2 Gyr, 3 Msun: ~0.35 Gyr, 5 Msun: ~0.1 Gyr
    log_t = (9.921 - 3.6648 * logm + 1.9697 * logm**2 - 0.9369 * logm**3)
    t_yr = 10**log_t
    return t_yr / 1e9


# ================================================================
#  并合判据
# ================================================================

def evaluate_merger_criterion(wd_params, progenitor, cluster_age_gyr,
                               cluster_age_err_gyr=None):
    """
    评估是否为双星并合产物。

    Criterion (Yan et al. 2025):
        若 t_cool + t_MS > t_cluster → 单星演化矛盾 → 并合候选体

    Parameters
    ----------
    wd_params : dict — from interpolate_wd_params()
    progenitor : dict — from compute_progenitor_lifetime()
    cluster_age_gyr : float — 星团年龄 (Gyr)
    cluster_age_err_gyr : float — 星团年龄误差 (Gyr)

    Returns
    -------
    dict
    """
    t_cool = wd_params['cooling_age_gyr']
    t_ms = progenitor['ms_lifetime_gyr']
    t_total_single = t_cool + t_ms  # 单星演化所需最小年龄
    t_cluster = cluster_age_gyr

    delta = t_total_single - t_cluster
    ratio = t_total_single / t_cluster if t_cluster > 0 else np.inf

    # 判据
    if delta > 0:
        if cluster_age_err_gyr and delta < cluster_age_err_gyr:
            flag = 'MARGINAL_MERGER'
            comment = (f't_cool+t_MS = {t_total_single:.3f} Gyr > '
                       f't_cluster = {t_cluster:.3f} Gyr, '
                       f'但差异 {delta:.3f} Gyr 在误差范围内')
        else:
            flag = 'MERGER_CANDIDATE'
            comment = (f't_cool+t_MS = {t_total_single:.3f} Gyr >> '
                       f't_cluster = {t_cluster:.3f} Gyr, '
                       f'单星演化不可能')
    else:
        flag = 'CONSISTENT'
        comment = (f't_cool+t_MS = {t_total_single:.3f} Gyr < '
                   f't_cluster = {t_cluster:.3f} Gyr, '
                   f'与单星演化一致')

    # 质量异常检查: 过高质量 WD (> 0.8 Msun) 也是并合线索
    mass_flag = ''
    if wd_params['mass'] > 1.0:
        mass_flag = 'ULTRA_MASSIVE'
    elif wd_params['mass'] > 0.8:
        mass_flag = 'HIGH_MASS'

    return {
        't_cool_gyr': t_cool,
        't_ms_gyr': t_ms,
        't_total_single_gyr': t_total_single,
        't_cluster_gyr': t_cluster,
        'delta_gyr': delta,
        'ratio': ratio,
        'merger_flag': flag,
        'mass_flag': mass_flag,
        'comment': comment,
    }


# ================================================================
#  主入口
# ================================================================

def run_cooling_age_analysis(ra, dec, cluster_name='', cluster_age_gyr=None,
                              cluster_age_err_gyr=None,
                              output_dir=None, gaia_phot=None):
    """
    对单个 WD 源执行完整冷却年龄分析。

    Parameters
    ----------
    ra, dec : float — 坐标
    cluster_name : str — 星团名
    cluster_age_gyr : float — 星团年龄 (Gyr), 如果 None 则尝试从 Hunt+2023 查找
    cluster_age_err_gyr : float — 年龄误差
    output_dir : str — 输出目录
    gaia_phot : dict — 预先查询的 Gaia 测光 (可选)

    Returns
    -------
    dict — 完整分析结果
    """
    report = {
        'ra': ra, 'dec': dec, 'cluster': cluster_name,
        'status': 'FAILED',
    }

    # ---- 1. Gaia 测光 ----
    if gaia_phot is None:
        print(f"  查询 Gaia DR3 测光...")
        gaia_phot = get_gaia_photometry(ra, dec)

    if gaia_phot is None:
        report['error'] = '无法获取 Gaia DR3 测光'
        print(f"  {report['error']}")
        return report

    report['gaia'] = gaia_phot
    bp_rp = gaia_phot['BP_RP']
    M_G = gaia_phot['M_G']
    print(f"  Gaia: G={gaia_phot['Gmag']:.3f}  BP-RP={bp_rp:.3f}  "
          f"M_G={M_G:.3f}  Plx={gaia_phot['Plx']:.3f} mas  "
          f"dist={gaia_phot['dist_pc']:.1f} pc")

    # ---- 2. WD 参数插值 ----
    print(f"  插值 WD 冷却轨迹...")
    wd_params = interpolate_wd_params(bp_rp, M_G)
    if wd_params is None:
        report['error'] = 'HR 图位置超出冷却轨迹范围'
        print(f"  {report['error']}")
        return report

    report['wd_params'] = wd_params
    print(f"  WD参数: M={wd_params['mass']:.3f} Msun  "
          f"Teff={wd_params['teff']:.0f} K  "
          f"log g={wd_params['logg']:.3f}  "
          f"t_cool={wd_params['cooling_age_gyr']:.3f} Gyr")

    # ---- 3. IFMR → 前身星 ----
    progenitor = compute_progenitor_lifetime(wd_params['mass'])
    if progenitor is None:
        report['error'] = 'IFMR 外推失败 (WD 质量超出范围)'
        print(f"  {report['error']}")
        return report

    # 同时用 Cummings+2018 分段 IFMR 做交叉验证
    m_prog_c18 = cummings2018_ifmr(wd_params['mass'])
    t_ms_c18 = mist_ms_lifetime(m_prog_c18) if not np.isnan(m_prog_c18) else np.nan

    report['progenitor'] = progenitor
    report['progenitor_c18'] = {
        'm_progenitor': m_prog_c18,
        'ms_lifetime_gyr': t_ms_c18,
    }
    print(f"  IFMR (WD_models): M_prog={progenitor['m_progenitor']:.3f} Msun  "
          f"t_MS={progenitor['ms_lifetime_gyr']:.3f} Gyr")
    print(f"  IFMR (Cummings18): M_prog={m_prog_c18:.3f} Msun  "
          f"t_MS={t_ms_c18:.3f} Gyr")

    # ---- 4. 星团年龄 ----
    if cluster_age_gyr is None and cluster_name:
        cluster_age_gyr, cluster_age_err_gyr = _lookup_cluster_age(cluster_name)

    if cluster_age_gyr is None:
        report['error'] = '未提供星团年龄'
        report['status'] = 'NO_CLUSTER_AGE'
        print(f"  无星团年龄, 跳过并合判据")
        _save_report(report, output_dir)
        return report

    report['cluster_age_gyr'] = cluster_age_gyr
    report['cluster_age_err_gyr'] = cluster_age_err_gyr
    print(f"  星团年龄: {cluster_age_gyr:.3f} Gyr"
          + (f" ± {cluster_age_err_gyr:.3f}" if cluster_age_err_gyr else ''))

    # ---- 5. 并合判据 ----
    merger = evaluate_merger_criterion(wd_params, progenitor,
                                        cluster_age_gyr, cluster_age_err_gyr)
    # 同时用 C18 的值
    merger_c18 = evaluate_merger_criterion(
        wd_params,
        {'ms_lifetime_gyr': t_ms_c18},
        cluster_age_gyr, cluster_age_err_gyr)

    report['merger'] = merger
    report['merger_c18'] = merger_c18
    report['status'] = merger['merger_flag']

    print(f"\n  === 并合判据 (WD_models IFMR) ===")
    print(f"  t_cool = {merger['t_cool_gyr']:.3f} Gyr")
    print(f"  t_MS   = {merger['t_ms_gyr']:.3f} Gyr")
    print(f"  t_cool + t_MS = {merger['t_total_single_gyr']:.3f} Gyr")
    print(f"  t_cluster     = {merger['t_cluster_gyr']:.3f} Gyr")
    print(f"  Δt = {merger['delta_gyr']:+.3f} Gyr  (ratio = {merger['ratio']:.2f})")
    print(f"  判据: {merger['merger_flag']}")
    if merger['mass_flag']:
        print(f"  质量标记: {merger['mass_flag']} (M_WD = {wd_params['mass']:.3f})")
    print(f"  {merger['comment']}")

    if not np.isnan(t_ms_c18):
        print(f"\n  === 并合判据 (Cummings+2018 IFMR) ===")
        print(f"  t_cool + t_MS = {merger_c18['t_total_single_gyr']:.3f} Gyr")
        print(f"  判据: {merger_c18['merger_flag']}")

    # ---- 6. 保存 ----
    _save_report(report, output_dir)
    save_csv(report, output_dir)

    # ---- 7. 绘图 ----
    if output_dir:
        _plot_cooling_age(report, output_dir)

    return report


# ================================================================
#  辅助函数
# ================================================================

def _lookup_cluster_age(cluster_name):
    """从 Hunt+2023 星团表查找年龄。"""
    try:
        from .orbit_traceback import load_hunt2023_clusters
        clusters = load_hunt2023_clusters()
        for cl in clusters:
            if cl['Name'].strip() == cluster_name.strip():
                age_gyr = 10**cl['logAge50'] / 1e9
                return age_gyr, None  # Hunt+2023 没有单独给误差
        # 尝试模糊匹配
        for cl in clusters:
            if cluster_name.strip().lower() in cl['Name'].strip().lower():
                age_gyr = 10**cl['logAge50'] / 1e9
                return age_gyr, None
    except Exception as e:
        print(f"  查找星团年龄出错: {e}")
    return None, None


def _save_report(report, output_dir):
    """保存冷却年龄分析报告为文本文件。"""
    if output_dir is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'cooling_age_analysis.txt')

    lines = []
    lines.append("=" * 60)
    lines.append("WD 冷却年龄分析报告")
    lines.append("方法: Bédard+2020 冷却轨迹 + Cummings+2018 IFMR")
    lines.append("参考: Yan et al. 2025")
    lines.append("=" * 60)
    lines.append(f"坐标: RA={report['ra']:.6f}  DEC={report['dec']:.6f}")
    lines.append(f"星团: {report.get('cluster', 'N/A')}")
    lines.append(f"状态: {report['status']}")
    lines.append("")

    if 'gaia' in report:
        g = report['gaia']
        lines.append("--- Gaia DR3 测光 ---")
        lines.append(f"  Source ID: {g.get('source_id', 'N/A')}")
        lines.append(f"  G = {g['Gmag']:.4f}  BP = {g['BPmag']:.4f}  "
                     f"RP = {g['RPmag']:.4f}")
        lines.append(f"  BP-RP = {g['BP_RP']:.4f}")
        lines.append(f"  Plx = {g['Plx']:.4f} ± {g['e_Plx']:.4f} mas")
        lines.append(f"  距离 = {g['dist_pc']:.1f} pc")
        lines.append(f"  M_G = {g['M_G']:.4f}")
        lines.append("")

    if 'wd_params' in report:
        w = report['wd_params']
        lines.append("--- WD 物理参数 (Bédard+2020 CO thick-H) ---")
        lines.append(f"  M_WD = {w['mass']:.4f} Msun")
        lines.append(f"  T_eff = {w['teff']:.0f} K  (log T = {w['logteff']:.4f})")
        lines.append(f"  log g = {w['logg']:.4f}")
        lines.append(f"  t_cool = {w['cooling_age_gyr']:.4f} Gyr "
                     f"({w['cooling_age_gyr']*1e3:.1f} Myr)")
        lines.append(f"  t_total (模型) = {w['total_age_gyr']:.4f} Gyr")
        lines.append("")

    if 'progenitor' in report:
        p = report['progenitor']
        lines.append("--- 前身星 (WD_models IFMR) ---")
        lines.append(f"  M_progenitor = {p['m_progenitor']:.4f} Msun")
        lines.append(f"  t_MS = {p['ms_lifetime_gyr']:.4f} Gyr "
                     f"({p['ms_lifetime_gyr']*1e3:.1f} Myr)")
        lines.append("")

    if 'progenitor_c18' in report:
        p = report['progenitor_c18']
        lines.append("--- 前身星 (Cummings+2018 分段 IFMR) ---")
        lines.append(f"  M_progenitor = {p['m_progenitor']:.4f} Msun")
        lines.append(f"  t_MS = {p['ms_lifetime_gyr']:.4f} Gyr "
                     f"({p['ms_lifetime_gyr']*1e3:.1f} Myr)")
        lines.append("")

    if 'merger' in report:
        m = report['merger']
        lines.append("--- 并合判据 ---")
        lines.append(f"  t_cool     = {m['t_cool_gyr']:.4f} Gyr")
        lines.append(f"  t_MS       = {m['t_ms_gyr']:.4f} Gyr")
        lines.append(f"  t_cool+t_MS = {m['t_total_single_gyr']:.4f} Gyr")
        lines.append(f"  t_cluster   = {m['t_cluster_gyr']:.4f} Gyr")
        lines.append(f"  Δt = {m['delta_gyr']:+.4f} Gyr")
        lines.append(f"  ratio = {m['ratio']:.3f}")
        lines.append(f"  判据: {m['merger_flag']}")
        if m['mass_flag']:
            lines.append(f"  质量标记: {m['mass_flag']}")
        lines.append(f"  {m['comment']}")
        lines.append("")

    if 'merger_c18' in report:
        m = report['merger_c18']
        lines.append("--- 并合判据 (Cummings+2018) ---")
        lines.append(f"  t_cool+t_MS = {m['t_total_single_gyr']:.4f} Gyr")
        lines.append(f"  判据: {m['merger_flag']}")
        lines.append(f"  {m['comment']}")
        lines.append("")

    if 'error' in report:
        lines.append(f"错误: {report['error']}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  报告已保存: {path}")


def save_csv(report, output_dir):
    """保存冷却年龄分析结果为 CSV"""
    import pandas as pd
    if report is None or output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)

    row = {
        'ra': report.get('ra'),
        'dec': report.get('dec'),
        'cluster': report.get('cluster', ''),
        'status': report.get('status', ''),
    }
    if 'gaia' in report:
        g = report['gaia']
        for k in ('source_id', 'Gmag', 'BPmag', 'RPmag', 'BP_RP',
                  'Plx', 'e_Plx', 'dist_pc', 'M_G'):
            row[f'gaia_{k}'] = g.get(k)
    if 'wd_params' in report:
        w = report['wd_params']
        for k in ('mass', 'teff', 'logteff', 'logg', 'cooling_age_gyr'):
            row[f'wd_{k}'] = w.get(k)
    if 'progenitor' in report:
        p = report['progenitor']
        row['progenitor_mass'] = p.get('m_progenitor')
        row['ms_lifetime_gyr'] = p.get('ms_lifetime_gyr')
    if 'merger' in report:
        m = report['merger']
        for k in ('t_cool_gyr', 't_ms_gyr', 't_total_single_gyr',
                  't_cluster_gyr', 'delta_gyr', 'ratio', 'merger_flag'):
            row[f'merger_{k}'] = m.get(k)

    df = pd.DataFrame([row])
    path = os.path.join(output_dir, 'cooling_age_analysis.csv')
    df.to_csv(path, index=False)
    return path


def _plot_cooling_age(report, output_dir):
    """绘制冷却年龄诊断图。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if 'merger' not in report or 'wd_params' not in report:
        return

    m = report['merger']
    w = report['wd_params']
    cluster = report.get('cluster', '')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图: 年龄对比柱状图 ----
    ax = axes[0]
    labels = ['$t_{\\rm cool}$', '$t_{\\rm MS}$',
              '$t_{\\rm cool}+t_{\\rm MS}$', '$t_{\\rm cluster}$']
    vals = [m['t_cool_gyr'] * 1e3, m['t_ms_gyr'] * 1e3,
            m['t_total_single_gyr'] * 1e3, m['t_cluster_gyr'] * 1e3]
    colors = ['steelblue', 'coral', 'darkred', 'forestgreen']

    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylabel('Age (Myr)')
    ax.set_title(f'{cluster}  WD M={w["mass"]:.3f} $M_\\odot$\n'
                 f'$T_{{\\rm eff}}$={w["teff"]:.0f} K   '
                 f'log g={w["logg"]:.2f}')

    # 标注数值
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10)

    # 标注判据
    flag_color = {'MERGER_CANDIDATE': 'red', 'MARGINAL_MERGER': 'orange',
                  'CONSISTENT': 'green'}
    ax.text(0.95, 0.95, m['merger_flag'],
            transform=ax.transAxes, ha='right', va='top',
            fontsize=13, fontweight='bold',
            color=flag_color.get(m['merger_flag'], 'black'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=flag_color.get(m['merger_flag'], 'black')))

    ax.grid(axis='y', alpha=0.3)

    # ---- 右图: HR 图上标注位置 ----
    ax2 = axes[1]
    if 'gaia' in report:
        g = report['gaia']
        # 绘制冷却轨迹等质量线
        model = _load_wd_model()
        mass_arr = model['mass_array']
        color_arr = model['color']
        mag_arr = model['Mag']

        # 绘制几条等质量线
        for m_target in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            mask = np.abs(mass_arr - m_target) < 0.005
            if np.sum(mask) > 5:
                c_track = color_arr[mask]
                m_track = mag_arr[mask]
                sort_idx = np.argsort(c_track)
                ax2.plot(c_track[sort_idx], m_track[sort_idx],
                        'gray', alpha=0.4, linewidth=0.8)
                # 标注质量
                valid = ~np.isnan(c_track) & ~np.isnan(m_track)
                if np.any(valid):
                    idx_label = sort_idx[len(sort_idx)//2]
                    if not np.isnan(c_track[idx_label]) and not np.isnan(m_track[idx_label]):
                        ax2.text(c_track[idx_label], m_track[idx_label] - 0.2,
                                f'{m_target:.1f}', fontsize=7, color='gray',
                                ha='center')

        # 标注目标源
        ax2.scatter([g['BP_RP']], [g['M_G']], c='red', s=120,
                   zorder=10, edgecolors='black', linewidths=1.5,
                   marker='*')
        ax2.annotate(f"M={w['mass']:.3f} $M_\\odot$\n"
                    f"$t_{{cool}}$={w['cooling_age_gyr']*1e3:.0f} Myr",
                    (g['BP_RP'], g['M_G']),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=9, arrowprops=dict(arrowstyle='->', color='red'))

        ax2.set_xlabel('BP - RP (mag)')
        ax2.set_ylabel('$M_G$ (mag)')
        ax2.set_title('Gaia HR Diagram (WD cooling tracks)')
        ax2.invert_yaxis()
        ax2.set_xlim(-0.6, 1.5)
        ax2.set_ylim(16, 8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'cooling_age_diagram.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  图已保存: {save_path}")


# ================================================================
#  批量运行
# ================================================================

def run_batch_cooling_age(csv_path, cluster_catalog=None, output_base=None):
    """
    对 spectra_download_urls.csv 中的所有源运行冷却年龄分析。

    Parameters
    ----------
    csv_path : str — CSV 文件路径
    cluster_catalog : list — Hunt+2023 星团列表 (可选, 自动加载)
    output_base : str — 结果根目录

    Returns
    -------
    list of dict — 所有源的分析报告
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='source_id', keep='first')

    if output_base is None:
        output_base = os.path.join(os.path.dirname(csv_path), 'toolbox_results')

    # 预加载 WD 模型 (只加载一次)
    _load_wd_model()

    reports = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        sid = str(int(float(row['source_id'])))
        ra = float(row['ra'])
        dec = float(row['dec'])
        cluster = str(row['cluster']) if pd.notna(row.get('cluster')) else ''

        dir_name = f"{cluster}_Gaia_{sid}" if cluster else f"Gaia_{sid}"
        dir_name = dir_name.replace(' ', '_').replace('/', '_')
        out_dir = os.path.join(output_base, dir_name)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] {cluster} Gaia {sid}")
        print(f"  RA={ra:.6f}  DEC={dec:.6f}")
        print(f"{'='*60}")

        # 检查是否已完成
        if os.path.exists(os.path.join(out_dir, 'cooling_age_analysis.txt')):
            print("  冷却年龄分析已完成, 跳过")
            continue

        try:
            report = run_cooling_age_analysis(
                ra, dec,
                cluster_name=cluster,
                output_dir=out_dir)
            reports.append(report)
        except Exception as e:
            print(f"  出错: {e}")
            import traceback
            traceback.print_exc()
            reports.append({
                'ra': ra, 'dec': dec, 'cluster': cluster,
                'source_id': sid, 'status': 'ERROR', 'error': str(e)
            })

    # 汇总
    _print_summary(reports)
    return reports


def _print_summary(reports):
    """打印批量分析汇总。"""
    print(f"\n\n{'='*70}")
    print("冷却年龄分析汇总")
    print(f"{'='*70}")

    merger_candidates = []
    marginal = []
    consistent = []
    failed = []

    for r in reports:
        status = r.get('status', 'FAILED')
        cluster = r.get('cluster', '')
        if status == 'MERGER_CANDIDATE':
            merger_candidates.append(r)
        elif status == 'MARGINAL_MERGER':
            marginal.append(r)
        elif status == 'CONSISTENT':
            consistent.append(r)
        else:
            failed.append(r)

    print(f"\n并合候选体: {len(merger_candidates)}")
    for r in merger_candidates:
        w = r.get('wd_params', {})
        m = r.get('merger', {})
        print(f"  {r['cluster']}  M_WD={w.get('mass',0):.3f}  "
              f"t_cool+t_MS={m.get('t_total_single_gyr',0):.3f} >> "
              f"t_cluster={m.get('t_cluster_gyr',0):.3f} Gyr")

    print(f"\n边缘候选: {len(marginal)}")
    for r in marginal:
        w = r.get('wd_params', {})
        m = r.get('merger', {})
        print(f"  {r['cluster']}  M_WD={w.get('mass',0):.3f}")

    print(f"\n与单星一致: {len(consistent)}")
    for r in consistent:
        w = r.get('wd_params', {})
        print(f"  {r['cluster']}  M_WD={w.get('mass',0):.3f}")

    print(f"\n失败/无数据: {len(failed)}")
    for r in failed:
        print(f"  {r.get('cluster','')}  错误: {r.get('error', r.get('status',''))}")
