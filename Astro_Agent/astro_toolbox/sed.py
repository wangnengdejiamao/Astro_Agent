"""
SED 模块
=============
1. 收集多波段测光: GALEX(FUV,NUV) + SDSS(ugriz) + Gaia(G,BP,RP) + 2MASS(JHK) + WISE(W1-W4)
2. 消光改正: 优先用本地 Bayestar2019 三维尘埃图 + Gaia 距离,
   回退到 IRSA/SFD, 用 Fitzpatrick+1999 消光律改正
3. 绘图: 多波段 SED 图 (按实际数据范围)

用法:
    # 方式1: 直接传入已有测光 (推荐, 避免重复查询)
    from astro_toolbox.sed import SEDFitter
    fitter = SEDFitter(190.305, 2.596)
    fitter.load_photometry(galex_phot, sdss_phot, twomass_phot, wise_phot, ...)
    fitter.apply_extinction()   # 自动查询 E(B-V) 并改正
    fitter.plot('sed_result.png')

    # 方式2: 自动查询 (兼容旧接口)
    fitter = SEDFitter(190.305, 2.596)
    fitter.collect_photometry()
    fitter.plot('sed_result.png')
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config, utils


# ================================================================
#  消光改正常量
# ================================================================

# Fitzpatrick+1999 R_V=3.1 消光系数 A_lambda/E(B-V)
# 来源: Schlafly & Finkbeiner 2011, Table 6 (Landolt RV=3.1)
EXTINCTION_COEFFS = {
    'GALEX_FUV':  8.06,
    'GALEX_NUV':  8.74,
    'SDSS_u':     4.239,
    'SDSS_g':     3.303,
    'SDSS_r':     2.285,
    'SDSS_i':     1.698,
    'SDSS_z':     1.263,
    'Gaia_G':     2.740,
    'Gaia_BP':    3.374,
    'Gaia_RP':    2.035,
    '2MASS_J':    0.723,
    '2MASS_H':    0.460,
    '2MASS_Ks':   0.310,
    'WISE_W1':    0.189,
    'WISE_W2':    0.146,
    'WISE_W3':    0.0,
    'WISE_W4':    0.0,
    'SPHEREx_1.0': 0.56,
    'SPHEREx_1.5': 0.36,
    'SPHEREx_2.0': 0.21,
    'SPHEREx_3.0': 0.10,
    'SPHEREx_4.5': 0.05,
}


_BAYESTAR_QUERY = None


def _distance_pc_from_parallax(parallax_mas, parallax_err_mas=None,
                               max_frac_err=0.3):
    try:
        plx = float(parallax_mas)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(plx) or plx <= 0:
        return None
    if parallax_err_mas is not None:
        try:
            e_plx = float(parallax_err_mas)
            if np.isfinite(e_plx) and e_plx > 0 and e_plx / plx > max_frac_err:
                return None
        except (TypeError, ValueError):
            pass
    return 1000.0 / plx


def _get_bayestar_query():
    """Load and cache local Bayestar2019 map."""
    global _BAYESTAR_QUERY
    if _BAYESTAR_QUERY is not None:
        return _BAYESTAR_QUERY
    import os
    if not os.path.exists(config.BAYESTAR2019_PATH):
        return None
    from dustmaps.bayestar import BayestarQuery
    _BAYESTAR_QUERY = BayestarQuery(
        map_fname=config.BAYESTAR2019_PATH,
        max_samples=config.BAYESTAR_MAX_SAMPLES,
        version='bayestar2019')
    return _BAYESTAR_QUERY


def query_gaia_distance(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    从 Gaia DR3 获取 SED 消光需要的距离。

    Returns
    -------
    dict 或 None
        source_id, Plx, e_Plx, dist_pc, G/BP/RP photometry.
    """
    tbl = utils.query_vizier(
        'I/355/gaiadr3', ra, dec, radius_arcsec,
        columns=['Source', 'RA_ICRS', 'DE_ICRS', 'Gmag', 'BPmag', 'RPmag',
                 'Plx', 'e_Plx', 'RUWE'])
    if tbl is None or len(tbl) == 0:
        return None
    row = tbl[0]
    result = {}
    for key, col in [('source_id', 'Source'), ('gaia_ra', 'RA_ICRS'),
                     ('gaia_dec', 'DE_ICRS'), ('Gmag', 'Gmag'),
                     ('BPmag', 'BPmag'), ('RPmag', 'RPmag'),
                     ('Plx', 'Plx'), ('e_Plx', 'e_Plx'), ('RUWE', 'RUWE')]:
        try:
            val = row[col]
            if np.ma.is_masked(val):
                continue
            if key == 'source_id':
                result[key] = str(val).strip()
            else:
                result[key] = float(val)
        except (KeyError, ValueError, TypeError, np.ma.MaskError):
            continue
    dist_pc = _distance_pc_from_parallax(
        result.get('Plx'), result.get('e_Plx'), max_frac_err=0.3)
    if dist_pc is not None:
        result['dist_pc'] = dist_pc
    return result if result else None


def query_bayestar_ebv(ra, dec, distance_pc=None, parallax_mas=None,
                       parallax_err_mas=None, mode=None,
                       return_details=False):
    """
    查询本地 Bayestar2019 三维消光。

    如果提供 Gaia 距离/视差，则查询该距离处的累计 reddening；否则使用
    config.BAYESTAR_FALLBACK_DISTANCE_PC 作为远端近似。
    """
    distance_source = 'input_distance'
    if distance_pc is None:
        distance_pc = _distance_pc_from_parallax(parallax_mas, parallax_err_mas)
        distance_source = 'gaia_parallax'
    if distance_pc is None:
        distance_pc = config.BAYESTAR_FALLBACK_DISTANCE_PC
        distance_source = 'fallback_distance'
    if mode is None:
        mode = config.BAYESTAR_QUERY_MODE

    try:
        bayestar = _get_bayestar_query()
        if bayestar is None:
            return None if not return_details else {
                'ebv': None,
                'source': 'bayestar2019_missing',
                'distance_pc': distance_pc,
                'distance_source': distance_source,
            }
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                         distance=distance_pc*u.pc, frame='icrs')
        value = bayestar(coord, mode=mode)
        ebv = float(np.asarray(value).squeeze()) * config.BAYESTAR_TO_EBV
        if not np.isfinite(ebv) or ebv < 0:
            ebv = None
        details = {
            'ebv': ebv,
            'source': 'bayestar2019',
            'map_path': config.BAYESTAR2019_PATH,
            'distance_pc': float(distance_pc),
            'distance_source': distance_source,
            'mode': mode,
        }
        return details if return_details else ebv
    except Exception as e:
        details = {
            'ebv': None,
            'source': 'bayestar2019_failed',
            'error': str(e),
            'distance_pc': float(distance_pc) if distance_pc is not None else None,
            'distance_source': distance_source,
        }
        return details if return_details else None


def query_ebv(ra, dec):
    """
    通过 IRSA Dust Tool 查询银河系消光 E(B-V)。
    使用 Schlegel, Finkbeiner & Davis (1998) + Schlafly & Finkbeiner (2011) 修正。

    Returns:
        float: E(B-V) [mag], 或 None
    """
    try:
        url = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust"
        params = {'locstr': f'{ra:.6f} {dec:.6f} Equ J2000'}
        session = utils.get_session(url)
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        text = resp.text

        # 解析 XML 结果, 提取 Schlafly & Finkbeiner 修正值
        # 格式: <meanValueSandF>0.0123</meanValueSandF>
        import re
        m = re.search(r'<meanValueSandF>\s*([\d.]+)\s*</meanValueSandF>', text)
        if m:
            ebv = float(m.group(1))
            return ebv

        # 回退: 使用 SFD 原始值
        m = re.search(r'<meanValueSFD>\s*([\d.]+)\s*</meanValueSFD>', text)
        if m:
            ebv = float(m.group(1)) * 0.86  # Schlafly+2011 修正因子
            return ebv
    except Exception as e:
        print(f"  E(B-V) 查询失败: {e}")

    # 回退: 尝试 dustmaps
    try:
        from dustmaps.sfd import SFDQuery
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        sfd = SFDQuery()
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        ebv = float(sfd(coord)) * 0.86
        return ebv
    except Exception:
        pass

    return None


def query_extinction(ra, dec, distance_pc=None, parallax_mas=None,
                     parallax_err_mas=None):
    """
    获取 SED 消光信息。优先 Bayestar2019 + Gaia 距离，失败后回退旧方法。
    """
    details = query_bayestar_ebv(
        ra, dec, distance_pc=distance_pc, parallax_mas=parallax_mas,
        parallax_err_mas=parallax_err_mas, return_details=True)
    if details and details.get('ebv') is not None:
        return details

    ebv = query_ebv(ra, dec)
    return {
        'ebv': ebv,
        'source': 'irsa_or_sfd_fallback' if ebv is not None else 'none',
        'bayestar_status': details.get('source') if details else 'not_run',
        'bayestar_error': details.get('error') if details else '',
        'distance_pc': distance_pc,
        'distance_source': 'input_distance' if distance_pc is not None else '',
    }


class SEDFitter:
    """多波段 SED 绘图器"""

    def __init__(self, ra, dec):
        self.ra = ra
        self.dec = dec
        self.photometry = {}   # {band_name: (mag, err, wave_A)}
        self.flux_data = {}    # {band_name: (flux, flux_err, wave_A)}
        self.ebv = None        # E(B-V) 消光值
        self.extinction_info = {}
        self.extinction_per_band = {}
        self.gaia_distance_info = None
        self.sed_diagnostics = None

    def load_photometry(self, *phot_dicts):
        """
        从已有测光结果加载数据 (避免重复查询)。

        Args:
            *phot_dicts: 多个测光 dict, 格式 {band_name: (mag, err, wave_A)}
                         可以是 galex.get_photometry(), sdss.get_photometry() 等的返回值
                         None 值会被跳过
        """
        for phot in phot_dicts:
            if phot and isinstance(phot, dict):
                for k, v in phot.items():
                    if isinstance(v, (tuple, list)) and len(v) == 3:
                        self.photometry[k] = v
        if self.photometry:
            self._convert_to_flux()
        return self.photometry

    def collect_photometry(self, include_galex=True, include_sdss=True,
                           include_gaia=True, include_2mass=True,
                           include_wise=True, include_spherex=True):
        """从各巡天收集测光数据 (并行查询加速)。如果已有数据则跳过。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 检查哪些巡天已有数据
        has_galex = any(k.startswith('GALEX') for k in self.photometry)
        has_sdss = any(k.startswith('SDSS') for k in self.photometry)
        has_gaia = any(k.startswith('Gaia') for k in self.photometry)
        has_2mass = any(k.startswith('2MASS') for k in self.photometry)
        has_wise = any(k.startswith('WISE') for k in self.photometry)
        has_spherex = any(k.startswith('SPHEREx') for k in self.photometry)

        tasks = {}
        if include_galex and not has_galex:
            tasks['GALEX'] = lambda: __import__('astro_toolbox.galex', fromlist=['galex']).get_photometry(self.ra, self.dec)
        if include_sdss and not has_sdss:
            tasks['SDSS'] = lambda: __import__('astro_toolbox.sdss', fromlist=['sdss']).get_photometry(self.ra, self.dec)
        if include_gaia and not has_gaia:
            tasks['Gaia'] = self._get_gaia_photometry
        if include_2mass and not has_2mass:
            tasks['2MASS'] = lambda: __import__('astro_toolbox.twomass', fromlist=['twomass']).get_photometry(self.ra, self.dec)
        if include_wise and not has_wise:
            tasks['WISE'] = lambda: __import__('astro_toolbox.wise', fromlist=['wise']).get_photometry(self.ra, self.dec)
        if include_spherex and not has_spherex:
            tasks['SPHEREx'] = lambda: __import__('astro_toolbox.spherex', fromlist=['spherex']).get_photometry(self.ra, self.dec)

        if not tasks:
            print(f"  SED: 已有 {len(self.photometry)} 个波段, 无需查询")
            self._convert_to_flux()
            return self.photometry

        print(f"收集多波段测光: RA={self.ra:.4f}, DEC={self.dec:.4f}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(func): name for name, func in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    phot = future.result()
                    if phot:
                        self.photometry.update(phot)
                    print(f"  {name:6s}: {len(phot) if phot else 0} 波段")
                except Exception as e:
                    print(f"  {name:6s} 失败: {e}")

        print(f"  总计: {len(self.photometry)} 个波段")
        self._convert_to_flux()
        return self.photometry

    def _get_gaia_photometry(self):
        """通过 Vizier 获取 Gaia DR3 测光"""
        tbl = utils.query_vizier('I/355/gaiadr3', self.ra, self.dec,
                                 columns=['Gmag', 'e_Gmag', 'BPmag', 'e_BPmag',
                                          'RPmag', 'e_RPmag', 'Plx', 'e_Plx',
                                          'RUWE'])
        if tbl is None:
            return {}
        row = tbl[0]
        phot = {}
        gaia_info = {}
        for key, col in [('source_id', 'Source'), ('Plx', 'Plx'),
                         ('e_Plx', 'e_Plx'), ('RUWE', 'RUWE')]:
            try:
                val = row[col]
                if np.ma.is_masked(val):
                    continue
                if key == 'source_id':
                    gaia_info[key] = str(val).strip()
                else:
                    gaia_info[key] = float(val)
            except (ValueError, KeyError, TypeError, np.ma.MaskError):
                continue
        for band, mag_col, err_col in [('Gaia_G', 'Gmag', 'e_Gmag'),
                                         ('Gaia_BP', 'BPmag', 'e_BPmag'),
                                         ('Gaia_RP', 'RPmag', 'e_RPmag')]:
            try:
                mag = float(row[mag_col])
                err = float(row[err_col])
                if 0 < mag < 30 and err > 0:
                    wave = config.BAND_INFO[band]['wave_A']
                    phot[band] = (mag, err, wave)
            except (ValueError, KeyError, np.ma.MaskError):
                continue
        if gaia_info:
            self.set_gaia_distance(gaia_info)
        return phot

    def set_gaia_distance(self, gaia_params=None, distance_pc=None,
                          parallax_mas=None, parallax_err_mas=None):
        """Set Gaia distance used by Bayestar 3D extinction."""
        info = {}
        if gaia_params:
            info.update(gaia_params)
            if distance_pc is None:
                distance_pc = gaia_params.get('dist_pc')
            if parallax_mas is None:
                parallax_mas = gaia_params.get('Plx')
            if parallax_err_mas is None:
                parallax_err_mas = gaia_params.get('e_Plx')
        if distance_pc is None:
            distance_pc = _distance_pc_from_parallax(
                parallax_mas, parallax_err_mas)
        if distance_pc is not None:
            info['dist_pc'] = distance_pc
        if parallax_mas is not None:
            info['Plx'] = parallax_mas
        if parallax_err_mas is not None:
            info['e_Plx'] = parallax_err_mas
        self.gaia_distance_info = info if info else None
        return self.gaia_distance_info

    def ensure_gaia_distance(self):
        """Query Gaia distance if not already supplied by HR/Gaia modules."""
        if self.gaia_distance_info and self.gaia_distance_info.get('dist_pc'):
            return self.gaia_distance_info
        info = query_gaia_distance(self.ra, self.dec)
        if info:
            self.set_gaia_distance(info)
        return self.gaia_distance_info

    def apply_extinction(self, ebv=None, distance_pc=None, gaia_params=None):
        """
        应用银河系消光改正。

        Args:
            ebv: E(B-V) 值。如果为 None, 优先 Bayestar2019 + Gaia 距离。
            distance_pc: 目标距离 (pc)，优先来自 Gaia 视差。
            gaia_params: Gaia/HR 查询结果，可含 dist_pc, Plx, e_Plx。

        消光律: Fitzpatrick+1999, R_V=3.1
        系数来源: Schlafly & Finkbeiner 2011
        """
        if gaia_params is not None or distance_pc is not None:
            self.set_gaia_distance(gaia_params=gaia_params,
                                   distance_pc=distance_pc)
        else:
            self.ensure_gaia_distance()

        if ebv is None:
            g = self.gaia_distance_info or {}
            info = query_extinction(
                self.ra, self.dec,
                distance_pc=g.get('dist_pc'),
                parallax_mas=g.get('Plx'),
                parallax_err_mas=g.get('e_Plx'))
            self.extinction_info = info or {}
            ebv = self.extinction_info.get('ebv')
        else:
            self.extinction_info = {
                'ebv': ebv,
                'source': 'manual',
                'distance_pc': distance_pc,
                'distance_source': 'manual',
            }
        if ebv is None or ebv <= 0:
            print(f"  消光改正: E(B-V) 无法获取或为零, 跳过")
            return

        self.ebv = ebv
        src = self.extinction_info.get('source', 'unknown')
        dist = self.extinction_info.get('distance_pc')
        dist_str = f", d={dist:.1f} pc" if dist is not None else ""
        print(f"  消光改正: E(B-V) = {ebv:.4f} mag ({src}{dist_str})")

        corrected = {}
        self.extinction_per_band = {}
        for band_name, (mag, err, wave_A) in self.photometry.items():
            coeff = EXTINCTION_COEFFS.get(band_name)
            if coeff is None:
                # 未知波段: 按波长用幂律近似 A_lambda/E(B-V) ~ R_V * (lambda/5500)^-1.3
                coeff = 3.1 * (wave_A / 5500.0) ** (-1.3)
            A_lambda = coeff * ebv
            self.extinction_per_band[band_name] = A_lambda
            corrected[band_name] = (mag - A_lambda, err, wave_A)

        self.photometry = corrected
        self._convert_to_flux()
        n = len(corrected)
        print(f"  消光改正完成: {n} 个波段已改正")

    def _convert_to_flux(self):
        """星等转 f_lambda (erg/s/cm^2/A)"""
        self.flux_data = {}
        for band_name, (mag, err, wave_A) in self.photometry.items():
            info = config.BAND_INFO.get(band_name, {})
            zero_jy = info.get('zero_Jy', 3631.0)
            flux, flux_err = utils.mag_to_flux_cgs(mag, wave_A, err, zero_jy)
            self.flux_data[band_name] = (flux, flux_err, wave_A)
        self.sed_diagnostics = None

    def analyze_excesses(self):
        """Flag UV/IR excess and two-component SED candidates."""
        from .diagnostics import analyze_sed
        self.sed_diagnostics = analyze_sed(self.flux_data)
        return self.sed_diagnostics

    def save_diagnostics(self, output_dir):
        """Save SED excess/bimodality diagnostics."""
        from .diagnostics import save_sed_diagnostics
        if self.sed_diagnostics is None:
            self.analyze_excesses()
        paths = save_sed_diagnostics(self.sed_diagnostics, output_dir)
        self.save_extinction_report(output_dir)
        return paths

    def save_extinction_report(self, output_dir):
        """Save extinction source, Gaia distance, and per-band A_lambda."""
        import os
        import pandas as pd
        if output_dir is None or not self.extinction_info:
            return None, None
        os.makedirs(output_dir, exist_ok=True)
        row = dict(self.extinction_info)
        if self.gaia_distance_info:
            for key in ('source_id', 'Plx', 'e_Plx', 'dist_pc', 'RUWE'):
                if key in self.gaia_distance_info:
                    row[f'gaia_{key}'] = self.gaia_distance_info[key]
        csv_path = os.path.join(output_dir, 'sed_extinction.csv')
        pd.DataFrame([row]).to_csv(csv_path, index=False)

        txt_path = os.path.join(output_dir, 'sed_extinction.txt')
        lines = ['# SED Extinction', '']
        lines.append(f"E(B-V) = {self.extinction_info.get('ebv')}")
        lines.append(f"source = {self.extinction_info.get('source', '')}")
        lines.append(f"map_path = {self.extinction_info.get('map_path', '')}")
        lines.append(f"distance_pc = {self.extinction_info.get('distance_pc', '')}")
        lines.append(f"distance_source = {self.extinction_info.get('distance_source', '')}")
        if self.gaia_distance_info:
            lines.append('')
            lines.append('Gaia distance input:')
            for key in ('source_id', 'Plx', 'e_Plx', 'dist_pc', 'RUWE'):
                if key in self.gaia_distance_info:
                    lines.append(f"  {key}: {self.gaia_distance_info[key]}")
        if self.extinction_per_band:
            lines.append('')
            lines.append('Band extinction A_lambda:')
            for band, a_lam in sorted(self.extinction_per_band.items()):
                lines.append(f"  {band}: {a_lam:.5f} mag")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        return csv_path, txt_path

    def plot(self, save_path=None):
        """绘制 SED 图 (多波段测光点，按实际数据范围)"""
        if len(self.flux_data) == 0:
            print("  无数据可绘")
            return None

        if self.sed_diagnostics is None:
            self.analyze_excesses()

        fig, ax = plt.subplots(figsize=(12, 6))

        # 按波长排序绘点
        band_names = list(self.flux_data.keys())
        waves = np.array([self.flux_data[b][2] for b in band_names])
        fluxes = np.array([self.flux_data[b][0] for b in band_names])
        errors = np.array([self.flux_data[b][1] for b in band_names])

        color_map = {
            'GALEX': 'purple', 'SDSS': 'blue', 'Gaia': 'green',
            '2MASS': 'orange', 'WISE': 'red', 'ROSAT': 'violet',
            'SPHEREx': 'darkorange',
        }
        for i, band in enumerate(band_names):
            prefix = band.split('_')[0]
            c = color_map.get(prefix, 'gray')
            ax.errorbar(waves[i], fluxes[i], yerr=errors[i],
                        fmt='o', color=c, markersize=8, capsize=3,
                        zorder=10)
            if self.sed_diagnostics:
                residual = self.sed_diagnostics.get('band_residuals', {}).get(band)
                if residual is not None and residual > 0.30:
                    ax.scatter([waves[i]], [fluxes[i]], s=140,
                               facecolors='none', edgecolors='crimson',
                               linewidths=1.4, zorder=12)
            fs = 9 if prefix == 'WISE' else 7
            ax.annotate(band.replace('_', ' '), (waves[i], fluxes[i]),
                        fontsize=fs, rotation=30, ha='left', va='bottom',
                        xytext=(3, 5), textcoords='offset points')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavelength (Å)', fontsize=12)
        ax.set_ylabel(r'$f_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=12)

        title = f'SED  RA={self.ra:.4f} DEC={self.dec:.4f}'
        if self.ebv is not None:
            src = self.extinction_info.get('source', 'ext')
            title += f'  E(B-V)={self.ebv:.4f} {src} (dereddened)'
        ax.set_title(title, fontsize=13)
        ax.grid(True, alpha=0.3, which='both')

        if self.sed_diagnostics and self.sed_diagnostics.get('flags'):
            txt = ', '.join(self.sed_diagnostics['flags'])
            ax.text(0.02, 0.96, txt, transform=ax.transAxes,
                    ha='left', va='top', fontsize=10, color='crimson',
                    bbox=dict(facecolor='white', edgecolor='crimson',
                              alpha=0.85, boxstyle='round,pad=0.25'))

        # 轴范围基于实际数据
        valid = (fluxes > 0) & np.isfinite(fluxes)
        if valid.any():
            w_min, w_max = waves[valid].min(), waves[valid].max()
            f_min, f_max = fluxes[valid].min(), fluxes[valid].max()
            ax.set_xlim(w_min * 0.5, w_max * 2)
            ax.set_ylim(f_min * 0.3, f_max * 5)

            # 波段区域背景色 (只画与数据重叠的部分)
            band_regions = [
                (1,      100,    'violet'),  # X-ray
                (912,    3000,   'purple'),  # UV
                (3000,   10000,  'blue'),    # Optical
                (10000,  50000,  'orange'),  # NIR
                (50000,  300000, 'red'),     # MIR
            ]
            plot_xmin, plot_xmax = w_min * 0.5, w_max * 2
            for lo, hi, clr in band_regions:
                if hi > plot_xmin and lo < plot_xmax:
                    ax.axvspan(max(lo, plot_xmin), min(hi, plot_xmax),
                               alpha=0.05, color=clr)

        fig.tight_layout()
        utils.save_and_close(fig, save_path)
        return fig

    def save_csv(self, output_dir):
        """保存 SED 测光数据为 CSV"""
        import pandas as pd
        if not self.flux_data:
            return None
        rows = []
        for band_name, (flux, flux_err, wave_A) in self.flux_data.items():
            row = {'band': band_name, 'wave_A': wave_A,
                   'flux_cgs': flux, 'flux_err_cgs': flux_err}
            if band_name in self.photometry:
                mag, err, _ = self.photometry[band_name]
                row['mag'] = mag
                row['mag_err'] = err
            if self.ebv is not None:
                row['ebv'] = self.ebv
                row['extinction_source'] = self.extinction_info.get('source', '')
                row['extinction_distance_pc'] = self.extinction_info.get('distance_pc')
                row['A_lambda_mag'] = self.extinction_per_band.get(band_name)
            if self.sed_diagnostics:
                res = self.sed_diagnostics.get('band_residuals', {}).get(band_name)
                if res is not None:
                    row['sed_residual_dex'] = res
            rows.append(row)
        df = pd.DataFrame(rows)
        return utils.write_csv(df, output_dir, 'sed_photometry.csv')


def quick_sed(ra, dec, save_path=None):
    """一键 SED 绘图"""
    fitter = SEDFitter(ra, dec)
    fitter.collect_photometry()
    fitter.apply_extinction()
    fitter.plot(save_path)
    return fitter


# ------------------------------------------------------------------
# Koester2 WD 模板加载 (供 rv_fitting.py CCF 使用)
# ------------------------------------------------------------------

_KOESTER2_CACHE = None


def _load_koester2_templates():
    """加载 Koester2 WD 光谱模板，返回 {(teff, logg): {'wavelength': arr, 'flux': arr}}"""
    global _KOESTER2_CACHE
    if _KOESTER2_CACHE is not None:
        return _KOESTER2_CACHE

    import os, glob, re
    import numpy as np

    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'koester2')
    files = sorted(glob.glob(os.path.join(base_dir, 'da*.dk.dat.txt')))
    if not files:
        return {}

    templates = {}
    for path in files:
        fname = os.path.basename(path)
        m = re.match(r'da(\d{5})_(\d{3})\.dk\.dat\.txt', fname)
        if not m:
            continue
        teff = int(m.group(1))
        logg = int(m.group(2)) / 100.0
        try:
            data = np.loadtxt(path, skiprows=5)
            if data.ndim == 2 and data.shape[1] >= 2:
                templates[(teff, logg)] = {
                    'wavelength': np.asarray(data[:, 0], dtype=np.float64),
                    'flux': np.asarray(data[:, 1], dtype=np.float64),
                }
        except Exception:
            continue

    _KOESTER2_CACHE = templates
    return templates
