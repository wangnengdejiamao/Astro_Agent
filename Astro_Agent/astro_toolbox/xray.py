"""
X 射线模块: ROSAT + XMM-Newton + Chandra + eROSITA + HEASARC
============================================================
通过 Vizier/HEASARC 查询 X 射线源，并把不同目录的流量统一成
可用于距离、光学/SED、白矮星或双星解释的诊断量。

用法:
    from astro_toolbox.xray import query_xray, analyze_xray
    result = query_xray(190.305, 2.596)
    analysis = analyze_xray(result, ra=190.305, dec=2.596)
"""
import os
import numpy as np
from . import config, utils


PC_CM = 3.0856775814913673e18
R_SUN_CM = 6.957e10
SIGMA_SB = 5.670374419e-5  # erg/s/cm^2/K^4

# ROSAT PSPC 常用粗略换算: 1 count/s ~ 1e-11 erg/s/cm^2 (0.1-2.4 keV)。
# 真正的 ECF 依赖谱型和吸收柱密度，所以所有由 count-rate 得到的流量都标记为 estimated。
ROSAT_PSPC_ECF = float(os.environ.get('ASTRO_XRAY_ROSAT_ECF', '1e-11'))


def _is_masked(value):
    try:
        return bool(np.ma.is_masked(value))
    except Exception:
        return False


def _clean_value(value):
    """Return a Python scalar/string, or None for masked/invalid values."""
    if value is None or _is_masked(value):
        return None
    try:
        if hasattr(value, 'item'):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    try:
        if not np.isfinite(float(value)):
            return None
    except Exception:
        pass
    return value


def _row_value(row, key, default=None):
    try:
        if hasattr(row, 'colnames') and key not in row.colnames:
            return default
        value = row[key]
    except Exception:
        try:
            value = row.get(key, default)
        except Exception:
            return default
    value = _clean_value(value)
    return default if value is None else value


def _first_value(row_or_dict, keys, default=None):
    for key in keys:
        value = _row_value(row_or_dict, key, None)
        if value is not None:
            return value
    return default


def _first_float(row_or_dict, keys, default=np.nan):
    value = _first_value(row_or_dict, keys, None)
    if value is None:
        return default
    try:
        value = float(value)
    except Exception:
        return default
    return value if np.isfinite(value) else default


def _first_string(row_or_dict, keys, default=''):
    value = _first_value(row_or_dict, keys, None)
    if value is None:
        return default
    return str(value).strip()


def _safe_log10(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(value) or value <= 0:
        return np.nan
    return float(np.log10(value))


def _angular_sep_arcsec(ra1, dec1, ra2, dec2):
    try:
        c1 = utils.coord(float(ra1), float(dec1))
        c2 = utils.coord(float(ra2), float(dec2))
        return float(c1.separation(c2).arcsec)
    except Exception:
        return np.nan


def _add_match_geometry(info, row, ra, dec):
    sep = _first_float(row, ['_r', 'angDist', 'ANG_DIST', 'offset',
                             'OFFSET', 'separation', 'Separation'])
    xra = _first_float(row, ['RAJ2000', 'RA_ICRS', 'RA', 'ra', 'RAdeg'])
    xdec = _first_float(row, ['DEJ2000', 'DE_ICRS', 'DEC', 'Dec', 'dec',
                              'DEdeg'])
    if not np.isfinite(sep) and np.isfinite(xra) and np.isfinite(xdec):
        sep = _angular_sep_arcsec(ra, dec, xra, xdec)
    if np.isfinite(sep):
        info['separation_arcsec'] = sep
    if np.isfinite(xra):
        info['xray_ra'] = xra
    if np.isfinite(xdec):
        info['xray_dec'] = xdec

    pos_err = _first_float(row, ['PosErr', 'ePos', 'e_Pos', 'R98',
                                 'Err', 'errPos', 'r0', 'r90'])
    if np.isfinite(pos_err):
        info['pos_err_arcsec'] = pos_err
    return info


def _assign_flux(info, flux=None, flux_err=None, band='', source='',
                 estimated=False):
    try:
        flux = float(flux)
    except Exception:
        return info
    if not np.isfinite(flux) or flux <= 0:
        return info
    info['flux_erg_cm2_s'] = flux
    info['flux_band'] = band or info.get('flux_band', '')
    info['flux_source'] = source or info.get('flux_source', '')
    info['flux_is_estimated'] = bool(estimated)
    try:
        flux_err = float(flux_err)
        if np.isfinite(flux_err) and flux_err > 0:
            info['flux_err_erg_cm2_s'] = flux_err
    except Exception:
        pass
    return info


def _normalize_from_row(row, survey, ra, dec, id_keys=None, flux_keys=None,
                        flux_err_keys=None, rate_keys=None, band='',
                        estimated_flux=False):
    id_keys = id_keys or ['Name', 'NAME', 'IAUName', 'SrcID', 'SRCID',
                          'Source', '1RXS', '2RXS']
    flux_keys = flux_keys or ['Flux', 'FLUX', 'F_X', 'FX', 'Flux8',
                              'flux', 'flux_broad', 'Flux_Broad',
                              'energy_flux']
    flux_err_keys = flux_err_keys or ['e_Flux', 'E_Flux', 'e_Flux8',
                                      'flux_err', 'FLUX_ERR', 'eFX']
    rate_keys = rate_keys or ['Rate', 'RATE', 'CountRate', 'COUNT_RATE',
                              'CR', 'C_RATE']

    info = {
        'survey': survey,
        'name': _first_string(row, id_keys, ''),
    }
    _add_match_geometry(info, row, ra, dec)

    rate = _first_float(row, rate_keys)
    rate_err = _first_float(row, ['e_Rate', 'E_Rate', 'RateErr',
                                  'e_COUNT_RATE'])
    if np.isfinite(rate):
        info['count_rate'] = rate
        if np.isfinite(rate_err):
            info['count_rate_err'] = rate_err

    for key in ['HR1', 'HR2', 'HR3', 'HR4', 'DET_LIKE', 'ExpTime',
                'EXPOSURE', 'exposure', 'ObsID', 'OBSID', 'Det']:
        value = _row_value(row, key, None)
        if value is not None:
            clean_key = key.lower() if key.isupper() else key
            info[clean_key] = value

    flux = _first_float(row, flux_keys)
    flux_err = _first_float(row, flux_err_keys)
    if np.isfinite(flux):
        _assign_flux(info, flux, flux_err, band=band,
                     source='catalog_flux', estimated=estimated_flux)
    elif survey.lower().startswith('rosat') and np.isfinite(rate):
        _assign_flux(info, rate * ROSAT_PSPC_ECF,
                     rate_err * ROSAT_PSPC_ECF if np.isfinite(rate_err) else None,
                     band='0.1-2.4 keV',
                     source=f'count_rate_x_{ROSAT_PSPC_ECF:.1e}',
                     estimated=True)
    return info


def query_rosat(ra, dec, radius_arcsec=30.0):
    """查询 ROSAT All-Sky Survey (2RXS) 源 (IX/10A)"""
    tbl = utils.query_vizier('IX/10A/2rxs', ra, dec, radius_arcsec,
                             columns=['1RXS', 'Rate', 'e_Rate', 'HR1', 'HR2',
                                      'ExpTime'])
    if tbl is None:
        return None
    row = tbl[0]
    try:
        return _normalize_from_row(
            row, 'ROSAT_2RXS', ra, dec,
            id_keys=['1RXS', '2RXS', 'Name'],
            rate_keys=['Rate'], band='0.1-2.4 keV')
    except (ValueError, KeyError):
        return None


def query_xmm(ra, dec, radius_arcsec=30.0):
    """查询 XMM-Newton 源目录 (4XMM-DR14)"""
    tbl = utils.query_vizier('IX/68/xmm4d14s', ra, dec, radius_arcsec,
                             columns=['Name', 'Flux8', 'e_Flux8', 'HR1', 'HR2',
                                      'ExpTime', 'Det'])
    if tbl is None:
        return None
    row = tbl[0]
    try:
        info = _normalize_from_row(
            row, 'XMM_4XMM-DR14', ra, dec,
            id_keys=['Name', 'SRCID'],
            flux_keys=['Flux8', 'Flux'],
            flux_err_keys=['e_Flux8', 'e_Flux'],
            band='0.2-12 keV')
        if 'flux_erg_cm2_s' in info:
            info['flux_0p2_12keV'] = info['flux_erg_cm2_s']
        return info
    except (ValueError, KeyError, np.ma.MaskError):
        return None


def query_chandra(ra, dec, radius_arcsec=30.0):
    """查询 Chandra Source Catalog 2.1 via HEASARC"""
    # 先尝试源目录；如果 astroquery/HEASARC 表名不可用，再退到观测 master。
    csc = _query_heasarc_table(ra, dec, 'csc2master',
                               'Chandra CSC 2.0', radius_arcsec)
    if csc:
        csc['survey'] = 'Chandra_CSC2'
        return csc
    try:
        from astroquery.heasarc import Heasarc
        import astropy.units as u
        h = Heasarc()
        c = utils.coord(ra, dec)
        tbl = h.query_region(c, mission='chanmaster',
                             radius=radius_arcsec * u.arcsec)
        if tbl is None or len(tbl) == 0:
            return None
        row = tbl[0]
        info = _normalize_from_row(
            row, 'Chandra_Master', ra, dec,
            id_keys=['OBSID', 'OBS_ID', 'NAME', 'TARGET_NAME'],
            flux_keys=['FLUX_AP', 'FLUX', 'flux'],
            band='0.5-7 keV')
        if not info.get('obsid'):
            obsid = _row_value(row, 'OBSID', None)
            if obsid is not None:
                info['obsid'] = str(obsid)
        exp = _first_float(row, ['EXPOSURE', 'Exposure', 'exposure'])
        if np.isfinite(exp):
            info['exposure'] = exp
        return info
    except Exception:
        return None


def query_erosita(ra, dec, radius_arcsec=30.0):
    """查询 eROSITA eRASS1 源 (IX/73)"""
    tbl = utils.query_vizier('IX/73/erasscat', ra, dec, radius_arcsec,
                             columns=['Name', 'Flux', 'e_Flux', 'HR1', 'HR2',
                                      'DET_LIKE'])
    if tbl is None:
        return None
    row = tbl[0]
    try:
        info = _normalize_from_row(
            row, 'eROSITA_eRASS1', ra, dec,
            id_keys=['Name', 'IAUName'],
            flux_keys=['Flux'],
            flux_err_keys=['e_Flux'],
            band='0.2-2.3 keV')
        if 'flux_erg_cm2_s' in info:
            info['flux_0p2_2p3keV'] = info['flux_erg_cm2_s']
        return info
    except (ValueError, KeyError, np.ma.MaskError):
        return None


def query_xray(ra, dec, radius_arcsec=30.0, include_heasarc=False,
               heasarc_radius_arcsec=60.0):
    """
    查询所有 X 射线巡天。

    Returns:
        dict: {'rosat': ..., 'xmm': ..., 'chandra': ..., 'erosita': ...}
    """
    result = {}
    for name, func in [('rosat', query_rosat), ('xmm', query_xmm),
                        ('chandra', query_chandra), ('erosita', query_erosita)]:
        try:
            r = func(ra, dec, radius_arcsec)
            if r is not None:
                result[name] = r
        except Exception as e:
            print(f"  {name} 查询失败: {e}")
    if include_heasarc:
        heasarc = query_heasarc_browse(ra, dec, heasarc_radius_arcsec)
        for key, value in heasarc.items():
            result[f'heasarc_{key}'] = value
    return result


# ===================== HEASARC Browse =====================

HEASARC_CATALOGS = {
    'chandra_csc2':   {'table': 'csc2master',  'label': 'Chandra CSC 2.0'},
    'chandra_master': {'table': 'chanmaster',  'label': 'Chandra Master'},
    'xmm_master':     {'table': 'xmmmaster',   'label': 'XMM-Newton Master'},
    'xmm_slew':       {'table': 'xmmslewcln',  'label': 'XMM Slew Clean'},
    'swift_master':  {'table': 'swiftmastr',  'label': 'Swift Master'},
    'swift_2sxps':   {'table': 'swift2sxps',  'label': 'Swift-XRT 2SXPS'},
    'nustar':        {'table': 'numaster',    'label': 'NuSTAR Master'},
    'nicer':         {'table': 'nicermastr',  'label': 'NICER Master'},
    'maxi':          {'table': 'maximaster',  'label': 'MAXI Master'},
    'suzaku':        {'table': 'suzamaster',  'label': 'Suzaku Master'},
    'rxte':          {'table': 'xtemaster',   'label': 'RXTE Master'},
}


def _query_heasarc_table(ra, dec, table_name, label, radius_arcsec=60.0):
    """查询单个 HEASARC 目录"""
    try:
        from astroquery.heasarc import Heasarc
        import astropy.units as u
        h = Heasarc()
        c = utils.coord(ra, dec)
        tbl = h.query_region(c, mission=table_name,
                             radius=radius_arcsec * u.arcsec)
        if tbl is None or len(tbl) == 0:
            return None
        row = tbl[0]
        info = _normalize_from_row(row, label, ra, dec,
                                   id_keys=['NAME', 'Name', 'OBJECT',
                                            'TARGET_NAME', 'OBSID', 'OBS_ID',
                                            'SOURCE_ID', 'SRCID'],
                                   band='')
        info.update({'catalog': label, 'table': table_name,
                     'n_matches': len(tbl)})
        # 提取常见字段
        for col in tbl.colnames:
            cl = col.upper()
            if any(k in cl for k in ['NAME', 'OBSID', 'OBS_ID', 'EXPOSURE',
                                      'TIME', 'FLUX', 'RATE', 'STATUS',
                                      'RA', 'DEC', 'TARGET']):
                try:
                    val = row[col]
                    val = _clean_value(val)
                    if val is not None:
                        info[col.lower()] = val
                except Exception:
                    pass
        return info
    except Exception:
        return None


def query_heasarc_browse(ra, dec, radius_arcsec=60.0):
    """
    通用 HEASARC 搜索: 查询 Swift / NuSTAR / NICER / MAXI / Suzaku / RXTE。

    Returns:
        dict: {'swift_master': {...}, 'nustar': {...}, ...} 仅含有匹配的目录
    """
    result = {}
    for key, cfg in HEASARC_CATALOGS.items():
        try:
            r = _query_heasarc_table(ra, dec, cfg['table'], cfg['label'],
                                     radius_arcsec)
            if r is not None:
                result[key] = r
        except Exception as e:
            print(f"  HEASARC {cfg['label']} 查询失败: {e}")
    return result


def flatten_xray_results(xray_result=None, heasarc_result=None):
    """Return a list of detection dictionaries from both X-ray result blocks."""
    rows = []
    for source_block, result in [('xray', xray_result),
                                 ('heasarc', heasarc_result)]:
        if not isinstance(result, dict):
            continue
        for key, data in result.items():
            if str(key).startswith('_') or not isinstance(data, dict):
                continue
            row = {'source_key': key, 'source_block': source_block}
            row.update(data)
            rows.append(row)
    return rows


def _extract_distance_from_results(results=None, gaia_params=None,
                                   distance_pc=None, ra=None, dec=None):
    if distance_pc is not None:
        try:
            distance_pc = float(distance_pc)
            if np.isfinite(distance_pc) and distance_pc > 0:
                return {'dist_pc': distance_pc, 'source': 'input_distance'}
        except Exception:
            pass

    if gaia_params:
        dist = _first_float(gaia_params, ['dist_pc', 'distance_pc'])
        if np.isfinite(dist) and dist > 0:
            return {'dist_pc': dist, 'source': 'input_gaia_params',
                    'gaia': gaia_params}
        plx = _first_float(gaia_params, ['Plx', 'parallax'])
        if np.isfinite(plx) and plx > 0:
            return {'dist_pc': 1000.0 / plx, 'source': 'input_parallax',
                    'gaia': gaia_params}

    if isinstance(results, dict):
        hr = results.get('HR_diagram')
        params = hr.get('params') if isinstance(hr, dict) else None
        if params:
            dist = _first_float(params, ['dist_pc', 'distance_pc'])
            if np.isfinite(dist) and dist > 0:
                return {'dist_pc': dist, 'source': 'HR_diagram/Gaia',
                        'gaia': params}
            plx = _first_float(params, ['Plx', 'parallax'])
            if np.isfinite(plx) and plx > 0:
                return {'dist_pc': 1000.0 / plx,
                        'source': 'HR_diagram/Gaia_parallax',
                        'gaia': params}

        sed = results.get('SED')
        ginfo = getattr(sed, 'gaia_distance_info', None)
        if ginfo:
            dist = _first_float(ginfo, ['dist_pc', 'distance_pc'])
            if np.isfinite(dist) and dist > 0:
                return {'dist_pc': dist, 'source': 'SED/Gaia',
                        'gaia': ginfo}

    if ra is not None and dec is not None:
        try:
            from .sed import query_gaia_distance
            info = query_gaia_distance(ra, dec)
            if info and info.get('dist_pc'):
                return {'dist_pc': float(info['dist_pc']),
                        'source': 'Gaia_query', 'gaia': info}
        except Exception:
            pass
    return {'dist_pc': np.nan, 'source': 'none'}


def _estimate_g_band_flux(gaia_params):
    """Approximate integrated optical G-band flux from Gaia G magnitude."""
    if not gaia_params:
        return np.nan
    gmag = _first_float(gaia_params, ['Gmag', 'phot_g_mean_mag'])
    if not np.isfinite(gmag):
        return np.nan
    # Maccacaro-style approximation with V ~ G:
    # log(f_X/f_V)=log(f_X)+0.4 V+5.37, so f_V=10^(-0.4V-5.37).
    return 10 ** (-0.4 * gmag - 5.37)


def _estimate_bolometric_flux(results=None, gaia_params=None, distance_pc=None):
    """
    Estimate observed bolometric flux from existing WD/SED fit products.
    Returns (fbol, source, stellar_params).
    """
    teff = np.nan
    radius_rsun = np.nan
    source = ''
    params = {}

    if isinstance(results, dict):
        hr = results.get('HR_diagram')
        if isinstance(hr, dict):
            analysis = hr.get('analysis')
            if analysis is None and hr.get('params'):
                analysis = hr['params'].get('hr_analysis')
            wd = (analysis or {}).get('wd_model') or {}
            if wd.get('status') == 'ok':
                teff = _first_float(wd, ['teff_k', 'teff'])
                mass = _first_float(wd, ['mass_msun', 'mass'])
                logg = _first_float(wd, ['logg'])
                if np.isfinite(mass) and np.isfinite(logg):
                    try:
                        from .wd_fitting import compute_wd_radius
                        radius_rsun = compute_wd_radius(mass, logg)
                    except Exception:
                        radius_rsun = np.nan
                source = 'HRD_WD_model'
                params.update({'teff_k': teff, 'mass_msun': mass,
                               'logg': logg, 'radius_rsun': radius_rsun})

        if not np.isfinite(radius_rsun):
            wd_fit = results.get('WD_fitting') or results.get('wd_fitting')
            if isinstance(wd_fit, dict):
                phys = wd_fit.get('physical_params') or {}
                radius_rsun = _first_float(phys, ['radius_rsun', 'R_Rsun'])
                teff = _first_float(phys, ['teff', 'teff_k'])
                source = 'WD_fitting'
                params.update({'teff_k': teff, 'radius_rsun': radius_rsun})

        if not np.isfinite(radius_rsun):
            sed_fit = results.get('sed_fit') or results.get('SED_fit')
            if isinstance(sed_fit, dict):
                radius_rsun = _first_float(sed_fit, ['R_Rsun', 'radius_rsun'])
                teff = _first_float(sed_fit, ['teff_sed', 'teff'])
                source = 'SED_fit'
                params.update({'teff_k': teff, 'radius_rsun': radius_rsun})

    if distance_pc is None and gaia_params:
        distance_pc = _first_float(gaia_params, ['dist_pc', 'distance_pc'])
    if (np.isfinite(teff) and np.isfinite(radius_rsun)
            and distance_pc is not None and np.isfinite(distance_pc)
            and teff > 0 and radius_rsun > 0 and distance_pc > 0):
        ratio = (radius_rsun * R_SUN_CM) / (distance_pc * PC_CM)
        fbol = SIGMA_SB * teff ** 4 * ratio ** 2
        params['fbol_erg_cm2_s'] = fbol
        return fbol, source, params

    return np.nan, source or 'none', params


def _collect_context_flags(results):
    flags = []
    if not isinstance(results, dict):
        return flags

    spec_diag = results.get('spectral_diagnostics') or {}
    for survey, diag in spec_diag.items():
        dflags = diag.get('flags', []) if isinstance(diag, dict) else []
        if any('EMISSION' in f for f in dflags):
            flags.append(f'{survey}:emission_lines')
        if any('NONSTELLAR' in f or 'AGN' in f for f in dflags):
            flags.append(f'{survey}:nonstellar_spectrum')

    sed = results.get('SED')
    sed_diag = getattr(sed, 'sed_diagnostics', None)
    if sed_diag and sed_diag.get('flags'):
        for flag in sed_diag['flags']:
            if flag in ('IR_EXCESS', 'UV_EXCESS', 'BIMODAL_SED'):
                flags.append(f'SED:{flag.lower()}')

    rv = results.get('rv_analysis')
    if isinstance(rv, dict):
        for flag in rv.get('quality_flags', []):
            if flag in ('SB2_CANDIDATE', 'PIPELINE_RV_DISAGREEMENT'):
                flags.append(f'RV:{flag.lower()}')

    pa = results.get('period_analysis')
    if isinstance(pa, dict):
        morphs = pa.get('morphology') or {}
        for label, morph in morphs.items():
            if isinstance(morph, dict) and morph.get('morphology'):
                m = morph['morphology']
                if m in ('eclipsing_or_dipper', 'sinusoidal_or_ellipsoidal',
                         'strongly_variable'):
                    flags.append(f'LC:{label}:{m}')

    hr = results.get('HR_diagram')
    if isinstance(hr, dict) and hr.get('analysis'):
        region = hr['analysis'].get('region')
        if region:
            flags.append(f'HRD:{region}')
    return flags


def _classify_xray_origin(row, log_lx, log_fx_fopt, log_fx_fbol,
                          context_flags):
    flags = []
    suggestions = []
    interpretation = []

    sep = _first_float(row, ['separation_arcsec'])
    survey = str(row.get('survey') or row.get('catalog') or '')
    if np.isfinite(sep):
        loose_limit = 25.0 if 'ROSAT' in survey.upper() else 8.0
        if sep > loose_limit:
            flags.append('OFFSET_LARGE')
            suggestions.append('检查 X-ray 位置误差和邻近源，必要时用 finding chart 排除混淆')

    if row.get('flux_is_estimated'):
        flags.append('FLUX_ESTIMATED_FROM_COUNT_RATE')

    if np.isfinite(log_lx):
        if log_lx >= 32.0:
            flags.append('VERY_HIGH_LX')
            interpretation.append('高 X-ray 光度，优先考虑强吸积源、AGN 或错配')
        elif log_lx >= 30.0:
            flags.append('HIGH_LX')
            interpretation.append('X-ray 光度达到 CV/磁活动双星常见范围')
        elif log_lx >= 28.0:
            interpretation.append('X-ray 光度中等，可能来自活动伴星或弱吸积')
        else:
            interpretation.append('X-ray 光度较弱')

    if np.isfinite(log_fx_fopt):
        if log_fx_fopt > -1.0:
            flags.append('XOPT_HIGH')
            interpretation.append('F_X/F_opt 偏高，需排查 CV/AGN/错配')
        elif log_fx_fopt > -3.0:
            flags.append('XOPT_ACTIVE_OR_ACCRETING')
            interpretation.append('F_X/F_opt 支持磁活动或吸积贡献')
        else:
            interpretation.append('F_X/F_opt 接近普通恒星活动范围')

    if np.isfinite(log_fx_fbol):
        if log_fx_fbol > -2.5:
            flags.append('XBOL_HIGH')
            suggestions.append('若是白矮星，应重点检查吸积、磁场或非白矮星污染')
        elif log_fx_fbol > -4.0:
            flags.append('XBOL_NOTICEABLE')

    cflags = set(context_flags or [])
    if any('HRD:white_dwarf_sequence' in f for f in cflags):
        if 'HIGH_LX' in flags or 'XOPT_ACTIVE_OR_ACCRETING' in flags:
            suggestions.append('白矮星带 + X-ray: 优先检查 CV、极向星/IP 或 WD+活动伴星模型')
    if any('emission_lines' in f for f in cflags):
        flags.append('SPECTRAL_EMISSION_SUPPORT')
        suggestions.append('发射线支持吸积/活动解释，可与 Hα/He 线强度一起判断')
    if any('SED:ir_excess' in f for f in cflags):
        flags.append('IR_EXCESS_SUPPORT')
        suggestions.append('IR excess 支持低温伴星/尘埃/双峰 SED，应与 X-ray 源关联检查')
    if any('SED:uv_excess' in f for f in cflags):
        flags.append('UV_EXCESS_SUPPORT')
    if any('eclipsing_or_dipper' in f for f in cflags):
        flags.append('LC_DIP_SUPPORT')
        suggestions.append('光变低谷/食特征支持近双星几何，可结合 X-ray 相位信息')
    if any('nonstellar_spectrum' in f for f in cflags):
        flags.append('NONSTELLAR_CONTEXT')
        suggestions.append('光谱非恒星特征存在时，优先排查 AGN/星系错分')

    if not interpretation:
        interpretation.append('有 X-ray 匹配，但缺少可靠流量或距离，暂不能定量判断')
    if not suggestions:
        suggestions.append('保留为候选 X-ray 对应体；建议检查误差圈内光学/IR 邻源')

    return {
        'flags': sorted(set(flags)),
        'interpretation': '; '.join(dict.fromkeys(interpretation)),
        'recommended_followup': '; '.join(dict.fromkeys(suggestions)),
    }


def analyze_xray(xray_result=None, heasarc_result=None, results=None,
                 ra=None, dec=None, distance_pc=None, gaia_params=None):
    """
    Combine X-ray detections with Gaia distance and existing toolbox diagnostics.

    Returns a dict with per-detection luminosities, X-ray/optical ratios and
    suggested physical interpretations.
    """
    detections = flatten_xray_results(xray_result, heasarc_result)
    dist_info = _extract_distance_from_results(
        results=results, gaia_params=gaia_params, distance_pc=distance_pc,
        ra=ra, dec=dec)
    dist_pc = dist_info.get('dist_pc', np.nan)
    gaia = gaia_params or dist_info.get('gaia') or {}
    fopt_g = _estimate_g_band_flux(gaia)
    fbol, fbol_source, stellar_params = _estimate_bolometric_flux(
        results=results, gaia_params=gaia, distance_pc=dist_pc)
    context_flags = _collect_context_flags(results)

    rows = []
    best = None
    for det in detections:
        row = dict(det)
        fx = _first_float(row, ['flux_erg_cm2_s', 'flux_0p2_12keV',
                                'flux_0p2_2p3keV', 'flux'])
        lx = np.nan
        if np.isfinite(fx) and np.isfinite(dist_pc) and dist_pc > 0:
            lx = 4.0 * np.pi * (dist_pc * PC_CM) ** 2 * fx
            row['lx_erg_s'] = lx
            row['log_lx'] = _safe_log10(lx)
        else:
            row['log_lx'] = np.nan

        if np.isfinite(fopt_g) and fopt_g > 0 and np.isfinite(fx):
            row['fx_over_fopt_g'] = fx / fopt_g
            row['log_fx_over_fopt_g'] = _safe_log10(row['fx_over_fopt_g'])
        else:
            row['fx_over_fopt_g'] = np.nan
            row['log_fx_over_fopt_g'] = np.nan

        if np.isfinite(fbol) and fbol > 0 and np.isfinite(fx):
            row['fx_over_fbol'] = fx / fbol
            row['log_fx_over_fbol'] = _safe_log10(row['fx_over_fbol'])
        else:
            row['fx_over_fbol'] = np.nan
            row['log_fx_over_fbol'] = np.nan

        cls = _classify_xray_origin(
            row, row.get('log_lx', np.nan),
            row.get('log_fx_over_fopt_g', np.nan),
            row.get('log_fx_over_fbol', np.nan),
            context_flags)
        row.update(cls)
        rows.append(row)

        if np.isfinite(fx):
            if best is None or fx > _first_float(best, ['flux_erg_cm2_s']):
                best = row

    return {
        'detections': rows,
        'n_detections': len(rows),
        'distance_pc': float(dist_pc) if np.isfinite(dist_pc) else np.nan,
        'distance_source': dist_info.get('source', 'none'),
        'gaia': gaia,
        'fopt_g_erg_cm2_s': fopt_g,
        'fbol_erg_cm2_s': fbol,
        'fbol_source': fbol_source,
        'stellar_params': stellar_params,
        'context_flags': context_flags,
        'best_detection': best,
    }


def save_analysis(analysis, output_dir):
    """Save X-ray luminosity/contribution analysis as CSV and text."""
    if not analysis:
        return None, None
    import pandas as pd
    utils.ensure_dir(output_dir)
    detections = analysis.get('detections') or []
    csv_path = None
    if detections:
        rows = []
        for det in detections:
            row = {}
            for key, value in det.items():
                if isinstance(value, (list, tuple, set)):
                    row[key] = ';'.join(str(v) for v in value)
                elif isinstance(value, (int, float, str, bool, np.integer,
                                        np.floating)) or value is None:
                    row[key] = value
            rows.append(row)
        csv_path = utils.write_csv(pd.DataFrame(rows), output_dir,
                                   'xray_analysis.csv')

    lines = []
    lines.append('# X-ray Analysis')
    lines.append(f"n_detections = {analysis.get('n_detections', 0)}")
    lines.append(f"distance_pc = {analysis.get('distance_pc', np.nan)}")
    lines.append(f"distance_source = {analysis.get('distance_source', '')}")
    lines.append(f"fopt_g_erg_cm2_s = {analysis.get('fopt_g_erg_cm2_s', np.nan)}")
    lines.append(f"fbol_erg_cm2_s = {analysis.get('fbol_erg_cm2_s', np.nan)}")
    lines.append(f"fbol_source = {analysis.get('fbol_source', '')}")
    if analysis.get('context_flags'):
        lines.append('context_flags = ' + ', '.join(analysis['context_flags']))
    lines.append('')

    if not detections:
        lines.append('No X-ray counterpart found in the enabled catalog queries.')
    for i, det in enumerate(detections, 1):
        lines.append(f"[{i}] {det.get('survey', det.get('catalog', det.get('source_key', '')))}")
        name = det.get('name') or det.get('object') or det.get('target_name')
        if name:
            lines.append(f"  name = {name}")
        if np.isfinite(_first_float(det, ['separation_arcsec'])):
            lines.append(f"  separation = {det['separation_arcsec']:.2f} arcsec")
        fx = _first_float(det, ['flux_erg_cm2_s'])
        if np.isfinite(fx):
            suffix = ' estimated' if det.get('flux_is_estimated') else ''
            lines.append(f"  F_X = {fx:.3e} erg/s/cm2 ({det.get('flux_band', '')}{suffix})")
        if np.isfinite(_first_float(det, ['log_lx'])):
            lines.append(f"  log L_X = {det['log_lx']:.3f} erg/s")
        if np.isfinite(_first_float(det, ['log_fx_over_fopt_g'])):
            lines.append(f"  log(F_X/F_Gopt) = {det['log_fx_over_fopt_g']:.3f}")
        if np.isfinite(_first_float(det, ['log_fx_over_fbol'])):
            lines.append(f"  log(F_X/F_bol) = {det['log_fx_over_fbol']:.3f}")
        lines.append('  flags = ' + (', '.join(det.get('flags', [])) or 'none'))
        lines.append('  interpretation = ' + det.get('interpretation', ''))
        lines.append('  followup = ' + det.get('recommended_followup', ''))
        lines.append('')

    txt_path = os.path.join(output_dir, 'xray_analysis.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return csv_path, txt_path


def save_csv(result, output_dir):
    """保存 X 射线查询结果为 CSV"""
    import pandas as pd
    if not result:
        return None
    rows = []
    for survey_key, data in result.items():
        if str(survey_key).startswith('_'):
            continue
        if isinstance(data, dict):
            row = {'source': survey_key}
            row.update({k: v for k, v in data.items()
                        if isinstance(v, (int, float, str, bool,
                                          np.integer, np.floating))})
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return utils.write_csv(df, output_dir, 'xray_catalog.csv')


def save_heasarc_csv(result, output_dir):
    """保存 HEASARC Browse 结果为 CSV"""
    import pandas as pd
    if not result:
        return None
    rows = []
    for cat_key, data in result.items():
        if str(cat_key).startswith('_'):
            continue
        if isinstance(data, dict):
            row = {'catalog_key': cat_key}
            row.update({k: v for k, v in data.items()
                        if isinstance(v, (int, float, str, bool,
                                          np.integer, np.floating))})
            rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return utils.write_csv(df, output_dir, 'heasarc_xray.csv')
