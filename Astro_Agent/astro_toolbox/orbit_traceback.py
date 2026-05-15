"""
轨道回溯模块 — 白矮星逃逸起源追踪
====================================
用 galpy 在银河势场中回溯白矮星轨道，检查是否经过 Hunt+2023 星团。

流程:
1. 加载 Hunt+2023 星团表 (位置、距离、自行、RV、年龄、潮汐半径)
2. 获取白矮星 Gaia DR3 天体测量参数 (位置、视差、自行)
3. 结合测量的 RV → 6D 相空间
4. 在 MWPotential2014 中向后积分轨道
5. 对星团也做轨道积分 (有 RV 的星团)
6. 找最小间距 < 潮汐半径的星团 → 可能的母星团

用法:
    from astro_toolbox.orbit_traceback import trace_back_to_clusters
    result = trace_back_to_clusters(ra, dec, rv, rv_err)
"""
import os
import numpy as np
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import config, utils

# Hunt+2023 星团表路径
# Allow override via env var; otherwise try a list of plausible locations
# (different machines, different users).
import os as _os
_HUNT_CANDIDATES = [
    _os.environ.get('ASTRO_TOOLBOX_HUNT_DIR') or '',
    '/Users/a1/Desktop/星团/Hunt+2023',
    '/Users/ljm/Desktop/csst/星团/Hunt+2023',
    _os.path.expanduser('~/Desktop/星团/Hunt+2023'),
]
_HUNT_DIR = next((p for p in _HUNT_CANDIDATES if p and _os.path.isdir(p)), _HUNT_CANDIDATES[1])
_CLUSTER_CACHE = None


# ================================================================
#  加载 Hunt+2023 星团表
# ================================================================

def load_hunt2023_clusters(hunt_dir=None):
    """
    解析 Hunt+2023 clusters.dat (固定宽度格式)。

    Returns
    -------
    list of dict, 每个包含:
        Name, RA, DEC, pmRA, pmDE, Plx, dist50, RV, e_RV, n_RV,
        logAge50, rtpc (潮汐半径 pc), X, Y, Z, Type
    """
    global _CLUSTER_CACHE
    if _CLUSTER_CACHE is not None:
        return _CLUSTER_CACHE

    hunt_dir = hunt_dir or _HUNT_DIR
    fpath = os.path.join(hunt_dir, 'clusters.dat')
    if not os.path.isfile(fpath):
        print(f"  警告: 未找到 Hunt+2023 星团表: {fpath}")
        return []

    clusters = []
    with open(fpath, 'r') as f:
        for line in f:
            if len(line.strip()) < 100:
                continue
            try:
                name = line[0:20].strip()
                obj_type = line[273:274].strip()
                ra = float(line[312:324].strip())
                dec = float(line[325:337].strip())
                pmra = float(line[467:480].strip())
                pmde = float(line[504:516].strip())
                plx = float(line[540:552].strip())
                dist50 = float(line[592:607].strip())
                rtpc = float(line[439:452].strip())

                # RV 可能为空
                rv_str = line[684:697].strip()
                rv = float(rv_str) if rv_str else np.nan
                srv_str = line[698:711].strip()
                s_rv = float(srv_str) if srv_str else np.nan
                erv_str = line[712:725].strip()
                e_rv = float(erv_str) if erv_str else np.nan
                nrv_str = line[726:730].strip()
                n_rv = int(nrv_str) if nrv_str else 0

                age_str = line[799:809].strip()
                logAge50 = float(age_str) if age_str else np.nan

                # X, Y, Z 坐标
                x = float(line[633:649].strip())
                y = float(line[650:666].strip())
                z = float(line[667:683].strip())

                clusters.append({
                    'Name': name, 'Type': obj_type,
                    'RA': ra, 'DEC': dec,
                    'pmRA': pmra, 'pmDE': pmde, 'Plx': plx,
                    'dist50': dist50, 'rtpc': rtpc,
                    'RV': rv, 's_RV': s_rv, 'e_RV': e_rv, 'n_RV': n_rv,
                    'logAge50': logAge50,
                    'X': x, 'Y': y, 'Z': z,
                })
            except (ValueError, IndexError):
                continue

    _CLUSTER_CACHE = clusters
    print(f"  Hunt+2023 星团表加载完成: {len(clusters)} 个星团")
    return clusters


# ================================================================
#  获取白矮星 Gaia 天体测量参数
# ================================================================

def get_gaia_astrometry(ra, dec, radius_arcsec=3.0):
    """
    从 Gaia DR3 获取天体测量参数。

    Returns
    -------
    dict: {ra, dec, pmRA, pmDE, Plx, e_Plx, e_pmRA, e_pmDE, source_id}
    or None
    """
    tbl = utils.query_vizier('I/355/gaiadr3', ra, dec,
                              radius_arcsec=radius_arcsec,
                              columns=['Source', 'RA_ICRS', 'DE_ICRS',
                                       'pmRA', 'e_pmRA', 'pmDE', 'e_pmDE',
                                       'Plx', 'e_Plx', 'RUWE',
                                       'RV', 'e_RV'])
    if tbl is None or len(tbl) == 0:
        return None

    row = tbl[0]
    result = {}
    for key, col in [('source_id', 'Source'),
                     ('ra', 'RA_ICRS'), ('dec', 'DE_ICRS'),
                     ('pmRA', 'pmRA'), ('e_pmRA', 'e_pmRA'),
                     ('pmDE', 'pmDE'), ('e_pmDE', 'e_pmDE'),
                     ('Plx', 'Plx'), ('e_Plx', 'e_Plx'),
                     ('RUWE', 'RUWE'),
                     ('gaia_rv', 'RV'), ('gaia_rv_err', 'e_RV')]:
        try:
            val = float(row[col])
            if np.ma.is_masked(val):
                val = np.nan
            result[key] = val
        except (ValueError, KeyError, np.ma.MaskError):
            result[key] = np.nan

    if np.isnan(result.get('Plx', np.nan)):
        return None

    return result


# ================================================================
#  轨道积分
# ================================================================

def integrate_orbit_back(ra, dec, dist_kpc, pmra, pmde, rv,
                         t_back_myr=500, dt_myr=0.5):
    """
    在 MWPotential2014 中向后积分轨道。

    Parameters
    ----------
    ra, dec : 度
    dist_kpc : 距离 kpc
    pmra, pmde : 自行 mas/yr (pmRA*cos(dec))
    rv : 径向速度 km/s
    t_back_myr : 回溯时间 Myr
    dt_myr : 时间步长 Myr

    Returns
    -------
    dict: {ts_myr, ra_arr, dec_arr, dist_arr, X_arr, Y_arr, Z_arr,
           R_arr (galactocentric cylindrical radius)}
    or None
    """
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from galpy.orbit import Orbit
        from galpy.potential import MWPotential2014
    except ImportError as e:
        print(f"  轨道积分需要 galpy: {e}")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        try:
            o = Orbit(vxvv=[ra * u.deg, dec * u.deg, dist_kpc * u.kpc,
                            pmra * u.mas / u.yr, pmde * u.mas / u.yr,
                            rv * u.km / u.s],
                      radec=True)

            # 向后积分 (负时间)
            n_steps = int(t_back_myr / dt_myr) + 1
            ts = np.linspace(0, -t_back_myr / 1e3, n_steps) * u.Gyr

            o.integrate(ts, MWPotential2014)

            # 提取轨道参数
            ts_myr = np.linspace(0, t_back_myr, n_steps)

            ra_arr = o.ra(ts)
            dec_arr = o.dec(ts)
            dist_arr = o.dist(ts, use_physical=True)  # kpc

            # galpy with astropy units: x/y/z 已经是 kpc
            X_arr = o.x(ts, use_physical=True)
            Y_arr = o.y(ts, use_physical=True)
            Z_arr = o.z(ts, use_physical=True)
            R_arr = np.sqrt(X_arr ** 2 + Y_arr ** 2)

            return {
                'ts_myr': ts_myr,
                'ra': ra_arr, 'dec': dec_arr, 'dist': dist_arr,
                'X': X_arr, 'Y': Y_arr, 'Z': Z_arr, 'R': R_arr,
            }
        except Exception as e:
            print(f"  轨道积分失败: {e}")
            return None


# ================================================================
#  星团-白矮星最小间距计算
# ================================================================

def find_closest_approach(wd_orbit, cluster, t_back_myr=500, dt_myr=0.5):
    """
    计算白矮星轨道与星团的最小 3D 间距。

    如果星团有 RV，也积分星团轨道；否则假设星团位置不变(用当前位置)。

    Returns
    -------
    dict: {min_sep_pc, min_sep_time_myr, within_tidal, rtpc,
           cluster_name, cluster_age_myr}
    or None
    """
    if wd_orbit is None:
        return None

    import astropy.units as u

    cl_name = cluster['Name']
    cl_dist = cluster['dist50'] / 1000.0  # pc → kpc
    cl_ra = cluster['RA']
    cl_dec = cluster['DEC']
    cl_pmra = cluster['pmRA']
    cl_pmde = cluster['pmDE']
    cl_rv = cluster['RV']
    rtpc = cluster['rtpc']
    logAge = cluster['logAge50']
    age_myr = 10 ** logAge / 1e6 if np.isfinite(logAge) else np.nan

    if cl_dist <= 0 or cl_dist > 30:
        return None

    # 积分星团轨道 (如果有 RV)
    if np.isfinite(cl_rv) and cluster['n_RV'] > 0:
        cl_orbit = integrate_orbit_back(cl_ra, cl_dec, cl_dist,
                                         cl_pmra, cl_pmde, cl_rv,
                                         t_back_myr=t_back_myr, dt_myr=dt_myr)
    else:
        # 无 RV: 只用自行外推 (近似)
        # 简化: 只考虑位置不变
        cl_orbit = None

    # 计算 3D 间距
    n = len(wd_orbit['ts_myr'])
    seps = np.full(n, np.inf)

    # 获取星团的银心坐标 (kpc) — 用 astropy 统一转换
    from astropy.coordinates import SkyCoord

    if cl_orbit is not None and len(cl_orbit['X']) == n:
        # 两个轨道都有，直接算 3D 间距 (已经是 kpc)
        dx = wd_orbit['X'] - cl_orbit['X']
        dy = wd_orbit['Y'] - cl_orbit['Y']
        dz = wd_orbit['Z'] - cl_orbit['Z']
        seps = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * 1000  # kpc → pc
    else:
        # 星团位置: 用 astropy 从 RA/DEC/dist 转到银心坐标
        cl_coord = SkyCoord(ra=cl_ra, dec=cl_dec, distance=cl_dist,
                            unit=('deg', 'deg', 'kpc'), frame='icrs')
        cl_gc = cl_coord.galactocentric
        cl_x = cl_gc.x.to(u.kpc).value
        cl_y = cl_gc.y.to(u.kpc).value
        cl_z = cl_gc.z.to(u.kpc).value

        dx = wd_orbit['X'] - cl_x
        dy = wd_orbit['Y'] - cl_y
        dz = wd_orbit['Z'] - cl_z
        seps = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) * 1000  # kpc → pc

    i_min = np.argmin(seps)
    min_sep = seps[i_min]
    min_time = wd_orbit['ts_myr'][i_min]

    # Two independent physical filters:
    #   within_tidal: spatial — was the source inside the cluster's tidal radius?
    #   time_consistent: temporal — was the cluster already gravitationally bound
    #       at the closest-approach epoch?  Approach time > cluster age means the
    #       cluster had not yet formed, so escape from it is impossible.
    within_tidal = bool(min_sep < rtpc)
    time_consistent = bool(np.isfinite(age_myr) and min_time < age_myr)
    return {
        'cluster_name': cl_name,
        'min_sep_pc': float(min_sep),
        'min_sep_time_myr': float(min_time),
        'within_tidal': within_tidal,
        'time_consistent': time_consistent,
        'physically_plausible_host': within_tidal and time_consistent,
        'rtpc': rtpc,
        'cluster_age_myr': age_myr,
        'cluster_rv': cl_rv,
        'cluster_dist_kpc': cl_dist,
    }


# ================================================================
#  主 API: 轨道回溯到星团
# ================================================================

def trace_back_to_clusters(ra, dec, rv, rv_err=None,
                           gaia_params=None,
                           t_back_myr=None,
                           max_sep_pc=200,
                           output_dir=None):
    """
    白矮星轨道回溯: 找可能的母星团。

    Parameters
    ----------
    ra, dec : float, 度
    rv : float, 径向速度 km/s
    rv_err : float, RV 误差 km/s
    gaia_params : dict or None, Gaia 天体测量 (如不提供则自动查询)
    t_back_myr : float or None, 回溯时间 Myr (默认: 自动根据附近星团年龄)
    max_sep_pc : float, 初筛最大间距 pc
    output_dir : str or None

    Returns
    -------
    dict: {
        gaia: {...},
        orbit: {...},
        candidates: [{cluster_name, min_sep_pc, min_sep_time_myr, ...}],
        best_match: {...} or None,
        figures: [...]
    }
    """
    result = {
        'gaia': None,
        'orbit': None,
        'candidates': [],
        'best_match': None,
        'figures': [],
    }

    # 1. Gaia 天体测量
    if gaia_params is None:
        print("  查询 Gaia DR3 天体测量...")
        gaia_params = get_gaia_astrometry(ra, dec)
    if gaia_params is None:
        print("  无法获取 Gaia 天体测量参数")
        return result
    result['gaia'] = gaia_params

    plx = gaia_params['Plx']
    if plx <= 0.1:
        print(f"  视差太小 (Plx={plx:.3f} mas), 距离不可靠")
        return result

    dist_kpc = 1.0 / plx  # 简单距离估计
    pmra = gaia_params['pmRA']
    pmde = gaia_params['pmDE']

    # Gaia 自身的 RV (如果有)
    gaia_rv = gaia_params.get('gaia_rv', np.nan)
    if not np.isfinite(rv) and np.isfinite(gaia_rv):
        rv = gaia_rv
        rv_err = gaia_params.get('gaia_rv_err', 10.0)
        print(f"  使用 Gaia RV: {rv:.2f} km/s")

    if not np.isfinite(rv):
        print("  无 RV, 无法回溯轨道")
        return result

    print(f"  WD: d={dist_kpc:.3f} kpc, pmRA={pmra:.2f}, pmDE={pmde:.2f}, "
          f"RV={rv:.2f} km/s")

    # 2. 加载星团表
    clusters = load_hunt2023_clusters()
    if not clusters:
        return result

    # 3. 先用天球距离初筛 (考虑自行 → 搜索范围大一些)
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    wd_coord = SkyCoord(ra=ra, dec=dec, unit='deg')

    # 初筛: 只保留距离在 2 倍以内、天球距离 < 30 度的星团
    pre_candidates = []
    for cl in clusters:
        cl_dist = cl['dist50'] / 1000.0  # kpc
        if cl_dist <= 0:
            continue
        dist_ratio = max(dist_kpc / cl_dist, cl_dist / dist_kpc)
        if dist_ratio > 3:
            continue
        cl_coord = SkyCoord(ra=cl['RA'], dec=cl['DEC'], unit='deg')
        ang_sep = wd_coord.separation(cl_coord).deg
        if ang_sep < 60:
            pre_candidates.append(cl)

    print(f"  初筛候选星团: {len(pre_candidates)} 个")

    # 4. 轨道积分
    if t_back_myr is None:
        # 自动: WD 冷却年龄估计 + 星团年龄范围
        t_back_myr = 500  # 默认 500 Myr

    print(f"  轨道回溯 {t_back_myr:.0f} Myr...")
    wd_orbit = integrate_orbit_back(ra, dec, dist_kpc, pmra, pmde, rv,
                                     t_back_myr=t_back_myr, dt_myr=0.5)
    if wd_orbit is None:
        return result
    result['orbit'] = wd_orbit

    # 5. 计算与每个候选星团的最小间距
    approaches = []
    for cl in pre_candidates:
        approach = find_closest_approach(wd_orbit, cl,
                                          t_back_myr=t_back_myr, dt_myr=0.5)
        if approach and approach['min_sep_pc'] < max_sep_pc:
            approaches.append(approach)

    # Sort by (plausibility desc, min_sep asc): physically plausible hosts
    # (inside tidal radius AND approach time before cluster age) come first.
    approaches.sort(key=lambda x: (not x.get('physically_plausible_host', False),
                                   not x.get('within_tidal', False),
                                   x['min_sep_pc']))
    result['candidates'] = approaches

    plausible = [a for a in approaches if a.get('physically_plausible_host')]
    result['plausible_hosts'] = plausible
    best = None
    if plausible:
        best = plausible[0]
        result['best_match'] = best
        result['best_match_basis'] = 'within_tidal_and_time_consistent'
    elif approaches:
        best = approaches[0]
        result['best_match'] = best
        result['best_match_basis'] = 'closest_only_NOT_time_consistent'

    if best is not None:
        tidal_flag = " (< 潮汐半径)" if best['within_tidal'] else ""
        time_flag = " (时间一致)" if best.get('time_consistent') else " (时间不一致!)"
        basis = result.get('best_match_basis', '?')
        print(f"  最佳匹配: {best['cluster_name']}  "
              f"最小间距={best['min_sep_pc']:.1f} pc (rt={best['rtpc']:.1f} pc) "
              f"@ {best['min_sep_time_myr']:.1f} Myr ago{tidal_flag}{time_flag} "
              f"[basis={basis}]")
        # 打印前 5
        print(f"\n  Top-5 候选星团 (S=空间通过 T=时间通过 ***=两者都通过):")
        for i, a in enumerate(approaches[:5]):
            s = "S" if a['within_tidal'] else "."
            t = "T" if a.get('time_consistent') else "."
            flag = f" [{s}{t}]"
            if a.get('physically_plausible_host'):
                flag += " ***"
            age_str = f"{a['cluster_age_myr']:.0f}" if np.isfinite(a['cluster_age_myr']) else "?"
            print(f"    {i+1}. {a['cluster_name']:20s}  "
                  f"sep={a['min_sep_pc']:7.1f} pc  "
                  f"rt={a['rtpc']:6.1f} pc  "
                  f"t={a['min_sep_time_myr']:6.1f} Myr  "
                  f"age={age_str} Myr{flag}")
    else:
        print(f"  未找到 {max_sep_pc} pc 内的候选星团")

    # 6. 绘图
    if output_dir:
        figs = _plot_traceback(result, output_dir, ra, dec, rv)
        result['figures'] = figs
        _save_traceback_report(result, output_dir, ra, dec, rv, rv_err)
        save_csv(result, output_dir)

    return result


# ================================================================
#  绘图
# ================================================================

def _plot_traceback(result, output_dir, ra, dec, rv):
    """绘制轨道回溯图"""
    import os
    figs = []
    orbit = result.get('orbit')
    if orbit is None:
        return figs

    candidates = result.get('candidates', [])

    # 图1: X-Y 投影 (银面)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # X-Y
    ax = axes[0]
    ax.plot(orbit['X'], orbit['Y'], 'b-', lw=1, alpha=0.6, label='WD orbit')
    ax.plot(orbit['X'][0], orbit['Y'][0], 'b*', ms=12, label='Now')

    from astropy.coordinates import SkyCoord as _SC
    import astropy.units as _u

    for a in candidates[:10]:
        cl = None
        for c in load_hunt2023_clusters():
            if c['Name'] == a['cluster_name']:
                cl = c
                break
        if cl:
            cl_dist_kpc = cl['dist50'] / 1000.0
            cl_c = _SC(ra=cl['RA'], dec=cl['DEC'], distance=cl_dist_kpc,
                        unit=('deg', 'deg', 'kpc'), frame='icrs')
            cl_gc = cl_c.galactocentric
            cl_x = cl_gc.x.to(_u.kpc).value
            cl_y = cl_gc.y.to(_u.kpc).value
            color = 'red' if a['within_tidal'] else 'orange'
            marker = '*' if a['within_tidal'] else 'o'
            ax.plot(cl_x, cl_y, marker=marker, color=color, ms=10,
                    label=f"{a['cluster_name']} ({a['min_sep_pc']:.0f}pc)")
            rt_kpc = a['rtpc'] / 1000.0
            circle = plt.Circle((cl_x, cl_y), rt_kpc, fill=False,
                                 color=color, ls='--', alpha=0.5)
            ax.add_patch(circle)

    ax.set_xlabel('X (kpc)', fontsize=11)
    ax.set_ylabel('Y (kpc)', fontsize=11)
    ax.set_title('Galactic X-Y plane', fontsize=12)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # X-Z
    ax = axes[1]
    ax.plot(orbit['X'], orbit['Z'], 'b-', lw=1, alpha=0.6)
    ax.plot(orbit['X'][0], orbit['Z'][0], 'b*', ms=12)
    for a in candidates[:10]:
        cl = None
        for c in load_hunt2023_clusters():
            if c['Name'] == a['cluster_name']:
                cl = c
                break
        if cl:
            cl_dist_kpc = cl['dist50'] / 1000.0
            cl_c = _SC(ra=cl['RA'], dec=cl['DEC'], distance=cl_dist_kpc,
                        unit=('deg', 'deg', 'kpc'), frame='icrs')
            cl_gc = cl_c.galactocentric
            cl_x = cl_gc.x.to(_u.kpc).value
            cl_z = cl_gc.z.to(_u.kpc).value
            color = 'red' if a['within_tidal'] else 'orange'
            ax.plot(cl_x, cl_z, 'o', color=color, ms=8)
    ax.set_xlabel('X (kpc)', fontsize=11)
    ax.set_ylabel('Z (kpc)', fontsize=11)
    ax.set_title('Galactic X-Z plane', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 间距 vs 时间
    ax = axes[2]
    for a in candidates[:5]:
        cl = None
        for c in load_hunt2023_clusters():
            if c['Name'] == a['cluster_name']:
                cl = c
                break
        if cl:
            # 重新计算间距时间序列
            cl_dist = cl['dist50'] / 1000.0
            if np.isfinite(cl['RV']) and cl['n_RV'] > 0:
                cl_orbit = integrate_orbit_back(
                    cl['RA'], cl['DEC'], cl_dist,
                    cl['pmRA'], cl['pmDE'], cl['RV'],
                    t_back_myr=orbit['ts_myr'][-1], dt_myr=0.5)
            else:
                cl_orbit = None

            if cl_orbit is not None and len(cl_orbit['X']) == len(orbit['X']):
                dx = orbit['X'] - cl_orbit['X']
                dy = orbit['Y'] - cl_orbit['Y']
                dz = orbit['Z'] - cl_orbit['Z']
            else:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                cl_coord = SkyCoord(ra=cl['RA'], dec=cl['DEC'],
                                    distance=cl_dist,
                                    unit=('deg', 'deg', 'kpc'), frame='icrs')
                cl_gc = cl_coord.galactocentric
                dx = orbit['X'] - cl_gc.x.to(u.kpc).value
                dy = orbit['Y'] - cl_gc.y.to(u.kpc).value
                dz = orbit['Z'] - cl_gc.z.to(u.kpc).value

            seps = np.sqrt(dx**2 + dy**2 + dz**2) * 1000  # pc
            color = 'red' if a['within_tidal'] else 'C0'
            ax.plot(orbit['ts_myr'], seps, '-', color=color, lw=1.5,
                    label=f"{a['cluster_name']} (rt={a['rtpc']:.0f}pc)")
            ax.axhline(a['rtpc'], color=color, ls=':', alpha=0.3)

    ax.set_xlabel('Time ago (Myr)', fontsize=11)
    ax.set_ylabel('3D separation (pc)', fontsize=11)
    ax.set_title('Distance to clusters', fontsize=12)
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Orbit Traceback  RA={ra:.4f} DEC={dec:.4f}  RV={rv:.1f} km/s',
                 fontsize=13, y=1.02)
    fig.tight_layout()

    fig_path = os.path.join(output_dir, 'orbit_traceback.png')
    utils.save_and_close(fig, fig_path)
    figs.append(fig_path)
    print(f"  轨道回溯图: {fig_path}")

    return figs


def _save_traceback_report(result, output_dir, ra, dec, rv, rv_err):
    """保存轨道回溯报告"""
    import os
    lines = []
    lines.append("# Orbit Traceback Report")
    lines.append(f"# RA={ra:.6f}, DEC={dec:.6f}, RV={rv:.2f} km/s")
    if rv_err:
        lines.append(f"# RV_err={rv_err:.2f} km/s")
    lines.append("")

    gaia = result.get('gaia', {})
    if gaia:
        lines.append("## Gaia DR3 Astrometry")
        plx = gaia.get('Plx', np.nan)
        lines.append(f"  Source ID: {int(gaia.get('source_id', 0))}")
        lines.append(f"  Parallax: {plx:.4f} ± {gaia.get('e_Plx', 0):.4f} mas")
        lines.append(f"  Distance: {1/plx*1000:.1f} pc" if plx > 0 else "  Distance: N/A")
        lines.append(f"  pmRA: {gaia.get('pmRA', 0):.3f} ± {gaia.get('e_pmRA', 0):.3f} mas/yr")
        lines.append(f"  pmDE: {gaia.get('pmDE', 0):.3f} ± {gaia.get('e_pmDE', 0):.3f} mas/yr")
        grv = gaia.get('gaia_rv', np.nan)
        if np.isfinite(grv):
            lines.append(f"  Gaia RV: {grv:.2f} ± {gaia.get('gaia_rv_err', 0):.2f} km/s")
        lines.append("")

    lines.append("## Candidate Parent Clusters")
    lines.append(f"  (within {200} pc minimum approach)")
    lines.append("")

    candidates = result.get('candidates', [])
    if candidates:
        lines.append(
            f"{'Rank':>4s}  {'Cluster':20s}  {'MinSep(pc)':>10s}  "
            f"{'TidalR(pc)':>10s}  {'TimeMyr':>8s}  {'Age(Myr)':>10s}  "
            f"{'InTidal':>8s}  {'TimeOK':>7s}  {'Plausible':>10s}"
        )
        lines.append("-" * 110)
        for i, a in enumerate(candidates, 1):
            age_str = f"{a['cluster_age_myr']:.0f}" if np.isfinite(a['cluster_age_myr']) else "?"
            tidal_flag = "YES" if a['within_tidal'] else "no"
            time_flag = "YES" if a.get('time_consistent') else "no"
            plaus = "YES ***" if a.get('physically_plausible_host') else "no"
            lines.append(
                f"{i:4d}  {a['cluster_name']:20s}  "
                f"{a['min_sep_pc']:10.1f}  {a['rtpc']:10.1f}  "
                f"{a['min_sep_time_myr']:8.1f}  {age_str:>10s}  "
                f"{tidal_flag:>8s}  {time_flag:>7s}  {plaus:>10s}"
            )
    else:
        lines.append("  No candidates found.")

    lines.append("")
    best = result.get('best_match')
    if best:
        basis = result.get('best_match_basis', '?')
        lines.append("## Best Match")
        lines.append(f"  Cluster: {best['cluster_name']}")
        lines.append(f"  Min separation: {best['min_sep_pc']:.1f} pc "
                      f"(tidal radius: {best['rtpc']:.1f} pc)")
        lines.append(f"  Time of closest approach: {best['min_sep_time_myr']:.1f} Myr ago")
        lines.append(f"  Cluster age: {best['cluster_age_myr']:.1f} Myr"
                      if np.isfinite(best['cluster_age_myr']) else "  Cluster age: unknown")
        lines.append(f"  Within tidal radius: {'YES' if best['within_tidal'] else 'no'}")
        lines.append(f"  Time consistent (approach < age): {'YES' if best.get('time_consistent') else 'NO'}")
        if best.get('physically_plausible_host'):
            lines.append("  ⇒ Physically plausible escape host (both spatial AND temporal filters pass).")
        else:
            lines.append("  ⇒ NOT a plausible escape host — closest-only match; cluster not yet formed at approach time OR outside tidal radius.")
        lines.append(f"  Match basis: {basis}")
        plausible = result.get('plausible_hosts', [])
        lines.append(f"  Plausible-host count in 200 pc window: {len(plausible)}")

    path = os.path.join(output_dir, 'orbit_traceback.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  轨道回溯报告: {path}")


def save_csv(result, output_dir):
    """保存轨道回溯候选星团为 CSV"""
    import pandas as pd
    if result is None or output_dir is None:
        return None
    candidates = result.get('candidates', [])
    if not candidates:
        return None
    rows = []
    for c in candidates:
        rows.append({
            'cluster_name': c.get('cluster_name', ''),
            'min_sep_pc': c.get('min_sep_pc'),
            'min_sep_time_myr': c.get('min_sep_time_myr'),
            'within_tidal': c.get('within_tidal', False),
            'tidal_radius_pc': c.get('rtpc'),
            'cluster_age_myr': c.get('cluster_age_myr'),
            'cluster_rv': c.get('cluster_rv'),
            'cluster_dist_kpc': c.get('cluster_dist_kpc'),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, 'orbit_traceback_candidates.csv')
    df.to_csv(path, index=False)
    return path


# ================================================================
#  完整分析入口
# ================================================================

def run_traceback_analysis(results, rv_report, output_dir=None,
                           ra=None, dec=None):
    """
    综合 RV + 轨道回溯分析。

    Parameters
    ----------
    results : dict, 来自 AstroQueryAll.results
    rv_report : dict, 来自 rv_fitting.run_rv_analysis
    output_dir : str
    ra, dec : float

    Returns
    -------
    dict: traceback result
    """
    if rv_report is None:
        print("  无 RV 数据，无法做轨道回溯")
        return None

    rv = rv_report.get('best_rv')
    rv_err = rv_report.get('best_rv_err')

    if rv is None or not np.isfinite(rv):
        print("  无 RV 测量，无法做轨道回溯")
        return None

    print(f"\n  轨道回溯分析 (RV={rv:.2f} ± {rv_err:.2f} km/s)...")

    tb = trace_back_to_clusters(
        ra, dec, rv, rv_err,
        output_dir=output_dir,
    )

    return tb
