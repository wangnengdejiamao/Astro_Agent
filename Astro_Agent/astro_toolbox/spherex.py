"""
SPHEREx 近红外谱光度
====================

IRSA 的 SPHEREx Spectrophotometry Tool 对已知坐标做 Tractor forced
photometry，并返回逐波长/逐曝光的校准谱光度表。科学输出应当保留
``mjd``、``flags``、``fit_ql`` 等质量列；不能把 SPHEREx 低分辨率谱再
合成为几个伪宽带点后参与 SED chi2。

本模块默认尝试调用 IRSA Firefly spectrophotometry processor；若在线任务
未返回表格，会从 SIA Level-2 cutout 逐通道抽取 raw-channel 表。这个
fallback 不做 detector 宽波段去重/合 bin，也不冒充官方 Tractor forced
photometry，``method`` 会明确标出来源。

用法:
    from astro_toolbox.spherex import query_spectrum, save_spectrum_csv
    spec = query_spectrum(232.3955, 29.4672)
    save_spectrum_csv(spec, "./out")
"""
import json
import numpy as np
import io
from . import config, utils

# IRSA SIA v2 端点
SIA_V2_URL = "https://irsa.ipac.caltech.edu/SIA/v2"
FIREFLY_SYNC_URL = "https://irsa.ipac.caltech.edu/applications/spherex/CmdSrv/sync"

# SPHEREx SIA v2 collection 名称 (按优先级: 最新的在前)
SPHEREX_COLLECTIONS = ['spherex_qr2', 'spherex_qr2_deep', 'SPHEREx']

# 物理常数
_MJY_SR_TO_CGS_PER_A = None  # 延迟计算


def _mjy_sr_to_cgs(flux_mjy_sr, wave_um, pixel_sr):
    """
    MJy/sr → erg/s/cm^2/A (f_lambda)

    1 MJy/sr = 1e-17 erg/s/cm^2/Hz/sr
    f_nu (per pixel) = flux_mjy_sr * pixel_sr * 1e-17  [erg/s/cm^2/Hz]
    f_lambda = f_nu * c / lambda^2
    """
    wave_cm = wave_um * 1e-4
    wave_A = wave_um * 1e4
    c_cgs = 2.99792458e10  # cm/s
    # f_nu per pixel
    f_nu = flux_mjy_sr * pixel_sr * 1e-17  # erg/s/cm^2/Hz
    # f_lambda = f_nu * c / lambda^2  (with lambda in cm, result in erg/s/cm^2/cm)
    f_lambda_per_cm = f_nu * c_cgs / wave_cm ** 2
    # 转为 per Angstrom
    f_lambda = f_lambda_per_cm * 1e-8
    return f_lambda


def query_spectrum(ra, dec, radius_arcsec=None,
                   collection=None, cutout_size=5,
                   prefer_spectrophotometry=True,
                   background_region_px=15,
                   timeout=None,
                   allow_cutout_fallback=True):
    """
    查询 SPHEREx 谱光度。

    默认优先使用 IRSA SPHEREx Spectrophotometry Tool 的校准 forced
    photometry 表，输出结构包含 ``lambda_um, lambda_Flambda,
    lambda_Flambda_err, flux_jy, flux_err_jy`` 以及 MJD/flags/fit_ql 等
    质量列。在线表格不可用时，默认回退到 SIA Level-2 raw-channel
    cutout 抽取：保留每个 obs_publisher_did/LVF 通道，不合成 6 个宽波段。

    Args:
        ra, dec: 目标坐标 (度)
        collection: SIA v2 collection 名称 (默认自动尝试)
        cutout_size: cutout 边长 (像素, 默认 5)

    Returns:
        dict: {
            'wavelength': array (Angstrom),
            'flux': array (erg/s/cm^2/A),
            'error': array,
            'flux_mjy_sr': array (原始 MJy/sr),
            'survey': 'SPHEREx',
            'n_channels': int,
        }
        或 None
    """
    if prefer_spectrophotometry:
        spec = _query_firefly_spectrophotometry(
            ra, dec, background_region_px=background_region_px,
            timeout=timeout)
        if spec is not None:
            return spec
        if not allow_cutout_fallback:
            print("  SPHEREx: 未取得 IRSA 定标谱光度表；不使用 SIA raw-channel fallback")
            return None

    return _query_cutout_pixel_spectrum(
        ra, dec, collection=collection, cutout_size=cutout_size)


def _query_cutout_pixel_spectrum(ra, dec, collection=None, cutout_size=5):
    """SIA Level-2 raw-channel cutout extraction.

    这是 raw-image 兜底路径，不是 IRSA 官方 Tractor forced photometry。
    它逐个 SIA row 下载 cutout，保留所有通道和负 flux，不再压缩成 D1-D6
    这类宽波段点。
    """
    import pandas as pd

    # 1. SIA v2 搜索: 找到覆盖目标的所有 spectral image 切片
    image_list = _query_sia_v2(ra, dec, collection)
    if image_list is None or len(image_list) == 0:
        print("  SPHEREx: 该坐标处无数据覆盖")
        return None

    print(f"  SPHEREx: 找到 {len(image_list)} 个 SIA raw 通道, 正在逐通道抽取...")

    session = utils.get_session(SIA_V2_URL)
    pixel_arcsec = 6.2
    pixel_sr = (pixel_arcsec / 206265.0) ** 2
    c_A_s = 2.99792458e18

    rows = []
    n_failed = 0
    for img_info in image_list:
        result = _extract_pixel_from_cutout(
            img_info, ra, dec, cutout_size, session)
        if result is not None:
            rows.append(_cutout_channel_row(
                img_info, result, ra, dec, pixel_sr, c_A_s))
        else:
            n_failed += 1

    if not rows:
        print("  SPHEREx: 无法从 raw-channel cutout 中提取有效数据")
        return None

    table = pd.DataFrame(rows)
    table = table.sort_values(
        by=['lambda_um', 'mjd', 'obs_publisher_did'],
        kind='mergesort',
    ).reset_index(drop=True)

    wave_A = table['wavelength_A'].to_numpy(dtype=float)
    flux_cgs = table['flux_flam'].to_numpy(dtype=float)
    err_cgs = table['fluxerr_flam'].to_numpy(dtype=float)
    wave_um = table['lambda_um'].to_numpy(dtype=float)

    n_nonpos = int(np.sum(table['flux_jy'].to_numpy(dtype=float) <= 0))
    msg = (f"  SPHEREx: 提取到 {len(table)} 个 raw 通道"
           f" ({n_nonpos} 个 flux<=0, 已保留; 未合bin)")
    if n_failed:
        msg += f"; {n_failed} 个 cutout 失败"
    print(msg)

    return {
        'survey': 'SPHEREx',
        'method': 'sia_cutout_center_pixel_raw_channels',
        'is_official_spectrophotometry': False,
        'ra': ra, 'dec': dec,
        'wavelength': wave_A,
        'flux': flux_cgs,
        'error': err_cgs,
        'lambda_um': wave_um,
        'lambda_Flambda': table['lambda_Flambda'].to_numpy(dtype=float),
        'lambda_Flambda_err': table['lambda_Flambda_err'].to_numpy(dtype=float),
        'flux_jy': table['flux_jy'].to_numpy(dtype=float),
        'flux_err_jy': table['flux_err_jy'].to_numpy(dtype=float),
        'flux_mjy_sr': table['flux_mjy_sr'].to_numpy(dtype=float),
        'table': table,
        'n_channels': len(wave_A),
    }


def _cutout_channel_row(img_info, result, ra, dec, pixel_sr, c_A_s):
    """Build one raw-channel row in official-like microJy plus physical units."""
    wave_um = float(result['wave_um'])
    wave_A = wave_um * 1e4
    flux_mjy = float(result['flux_mjy_sr'])
    err_mjy = abs(float(result.get('error_mjy_sr', np.nan)))

    flux_jy = flux_mjy * pixel_sr * 1e6
    flux_err_jy = err_mjy * pixel_sr * 1e6 if np.isfinite(err_mjy) else np.nan
    lambda_Flambda = flux_jy * 1e-23 * c_A_s / wave_A
    lambda_Flambda_err = flux_err_jy * 1e-23 * c_A_s / wave_A
    flux_flam = lambda_Flambda / wave_A
    fluxerr_flam = lambda_Flambda_err / wave_A

    row = {
        'ra': float(ra),
        'dec': float(dec),
        'x_image': result.get('x_image', np.nan),
        'y_image': result.get('y_image', np.nan),
        'mjd': result.get('mjd', img_info.get('t_avg', np.nan)),
        'mjd_beg': result.get('mjd_beg', img_info.get('t_min', np.nan)),
        'mjd_end': result.get('mjd_end', img_info.get('t_max', np.nan)),
        'date_obs': result.get('date_obs'),
        'flux_bkg': result.get('background_mjy_sr', np.nan),
        'raw_flux_mjy_sr': result.get('raw_flux_mjy_sr', np.nan),
        'local_bkg_flg': True,
        'flags': result.get('flags', np.nan),
        'fit_ql': np.nan,
        'deep_flg': bool(result.get('deep_flg', False)),
        'det_id': result.get('det_id', _det_id_from_band(img_info.get('band'))),
        'lvf_id': result.get('lvf_id', np.nan),
        'obs_id': result.get('obs_id', img_info.get('obs_id')),
        'obs_publisher_did': img_info.get('obs_publisher_did'),
        'band': img_info.get('band'),
        'lambda': wave_um,
        'lambda_um': wave_um,
        'lambda_width': float(result.get('bandwidth_um', np.nan)),
        'lambda_width_um': float(result.get('bandwidth_um', np.nan)),
        'flux': flux_jy * 1e6,
        'flux_err': flux_err_jy * 1e6,
        'flux_uJy': flux_jy * 1e6,
        'flux_err_uJy': flux_err_jy * 1e6,
        'flux_jy': flux_jy,
        'flux_err_jy': flux_err_jy,
        'lambda_Flambda': lambda_Flambda,
        'lambda_Flambda_err': lambda_Flambda_err,
        'wavelength_A': wave_A,
        'flux_flam': flux_flam,
        'fluxerr_flam': fluxerr_flam,
        'flam': flux_flam,
        'flux_cgs': flux_flam,
        'fluxerr_cgs': fluxerr_flam,
        'flux_mjy_sr': flux_mjy,
        'error_mjy_sr': err_mjy,
        'extraction_method': 'center_pixel_local_background',
        'cutout_x_center': result.get('cutout_x_center', np.nan),
        'cutout_y_center': result.get('cutout_y_center', np.nan),
        'access_url': img_info.get('access_url'),
        'cloud_access': img_info.get('cloud_access'),
        'dist_to_point_deg': img_info.get('dist_to_point', np.nan),
    }
    return row


def _det_id_from_band(band):
    if band is None:
        return np.nan
    import re
    match = re.search(r'D(\d+)', str(band))
    if match:
        return int(match.group(1))
    return np.nan


def _interpolate_wcs_wave(wcs_data, x_image, y_image,
                          fallback_wave=0.0, fallback_width=0.03):
    """Interpolate SPHEREx WCS-WAVE VALUES at full-frame detector pixels."""
    try:
        if not np.isfinite(x_image) or not np.isfinite(y_image):
            return fallback_wave, fallback_width
        x_grid = np.asarray(wcs_data['X'], dtype=float).ravel()
        y_grid = np.asarray(wcs_data['Y'], dtype=float).ravel()
        values = np.asarray(wcs_data['VALUES'], dtype=float)
        if values.ndim != 3 or values.shape[-1] != 2:
            return fallback_wave, fallback_width
        wave_by_y = np.array([
            np.interp(float(x_image), x_grid, row[:, 0])
            for row in values
        ])
        width_by_y = np.array([
            np.interp(float(x_image), x_grid, row[:, 1])
            for row in values
        ])
        wave_um = float(np.interp(float(y_image), y_grid, wave_by_y))
        width_um = float(np.interp(float(y_image), y_grid, width_by_y))
        return wave_um, width_um
    except Exception:
        return fallback_wave, fallback_width


def _firefly_request(ra, dec, background_region_px=15):
    tbl_id = f"Spec-photo-tbl-{int(abs(ra * 1000000))}-{int(abs(dec * 1000000))}"
    return {
        'startIdx': 0,
        'pageSize': 2147483647,
        'tbl_id': tbl_id,
        'id': 'SpectrophotometryProcessor',
        'UserTargetWorldPt': f'{ra};{dec};EQ_J2000',
        'shapeFit': 'false',
        'bgEstimationRegion': str(int(background_region_px)),
        'META_INFO': {
            'title': 'Spectrophotometry Targets',
            'tbl_id': tbl_id,
            'dataServiceOptions': {
                'DataProductFactoryOptions': {'datalinkDisableMoreDrop': True},
                'generateDownloadFileName': True,
                'obsCoreDownloadProps': {'downloadType': 'package'},
            },
        },
    }


def _query_firefly_spectrophotometry(ra, dec, background_region_px=15,
                                     timeout=None):
    """
    Submit a SPHEREx SpectrophotometryProcessor table request.

    Firefly may background long jobs.  When the HTTP response does not contain
    a table model, this function returns None so the caller can use the
    quick-look fallback while the user can still run/download the IRSA job from
    the web UI.
    """
    req = _firefly_request(ra, dec, background_region_px)
    session = utils.get_session(FIREFLY_SYNC_URL)
    timeout = timeout or (config.CONNECT_TIMEOUT, config.TIMEOUT)
    try:
        resp = session.post(
            FIREFLY_SYNC_URL,
            params={'cmd': 'tableSearch'},
            data={
                'cmd': 'tableSearch',
                'request': json.dumps(req, separators=(',', ':')),
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"  SPHEREx spectrophotometry 查询未返回表格: {exc}")
        return None

    data = payload
    if isinstance(payload, list) and payload:
        item = payload[0]
        if isinstance(item, dict):
            if item.get('success') is False:
                print(f"  SPHEREx spectrophotometry 失败: {item.get('error', '')}")
                return None
            data = item.get('data', item)
    elif isinstance(payload, dict) and payload.get('success') is True:
        data = payload.get('data', payload)

    df = _table_payload_to_dataframe(data)
    if df is None or len(df) == 0:
        print("  SPHEREx spectrophotometry: 在线任务没有立即返回谱表")
        return None

    return _spectrophotometry_dataframe_to_result(
        df, ra=ra, dec=dec, method='irsa_firefly_spectrophotometry')


def _table_payload_to_dataframe(payload):
    """Convert Firefly table-model JSON or a list/dict payload to DataFrame."""
    import pandas as pd
    if payload is None:
        return None
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if 'tableData' in payload:
            table = payload.get('tableData') or {}
            cols = table.get('columns') or payload.get('tableMeta', {}).get('columns')
            data = table.get('data') or []
            names = []
            if cols:
                for col in cols:
                    if isinstance(col, dict):
                        names.append(col.get('name') or col.get('label'))
                    else:
                        names.append(str(col))
            if names and data:
                return pd.DataFrame(data, columns=names[:len(data[0])])
            if data:
                return pd.DataFrame(data)
        for key in ('data', 'rows', 'table'):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
            if isinstance(value, dict):
                nested = _table_payload_to_dataframe(value)
                if nested is not None:
                    return nested
    return None


def _find_col(df, candidates):
    norm = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(' ', '_').replace('-', '_')
        norm[key] = col
    for cand in candidates:
        key = cand.strip().lower().replace(' ', '_').replace('-', '_')
        if key in norm:
            return norm[key]
    return None


def _spectrophotometry_dataframe_to_result(df, ra=None, dec=None,
                                           method='irsa_spectrophotometry'):
    """Standardize IRSA/sample SPHEREx tables into toolbox spectrum arrays."""
    import pandas as pd
    df = df.copy()
    c_wave_um = _find_col(df, ['lambda_um', 'wavelength', 'lambda', 'wave_um'])
    c_lfl = _find_col(df, ['lambda_Flambda', 'lambda_flambda', 'lambda_f_lambda'])
    c_lfl_err = _find_col(df, ['lambda_Flambda_err', 'lambda_flambda_err'])
    c_fjy = _find_col(df, ['flux_jy', 'fnu_jy'])
    c_fjy_err = _find_col(df, ['flux_err_jy', 'flux_jy_err', 'fnu_err_jy'])
    c_flux = _find_col(df, ['flux', 'flux_ujy', 'fnu_ujy'])
    c_flux_err = _find_col(df, ['flux_err', 'fluxerr', 'flux_error', 'flux_err_ujy'])

    if c_wave_um is None:
        return None
    wave_um = pd.to_numeric(df[c_wave_um], errors='coerce').to_numpy(dtype=float)
    wave_A = wave_um * 1e4
    c_A_s = 2.99792458e18

    if c_lfl is not None:
        lambda_Flambda = pd.to_numeric(df[c_lfl], errors='coerce').to_numpy(dtype=float)
        lambda_Flambda_err = (
            pd.to_numeric(df[c_lfl_err], errors='coerce').to_numpy(dtype=float)
            if c_lfl_err is not None else np.full_like(lambda_Flambda, np.nan)
        )
        flux_jy = lambda_Flambda * wave_A / c_A_s * 1e23
        flux_err_jy = lambda_Flambda_err * wave_A / c_A_s * 1e23
    else:
        if c_fjy is not None:
            flux_jy = pd.to_numeric(df[c_fjy], errors='coerce').to_numpy(dtype=float)
            flux_err_jy = (
                pd.to_numeric(df[c_fjy_err], errors='coerce').to_numpy(dtype=float)
                if c_fjy_err is not None else np.full_like(flux_jy, np.nan)
            )
        elif c_flux is not None:
            flux_uJy = pd.to_numeric(df[c_flux], errors='coerce').to_numpy(dtype=float)
            flux_err_uJy = (
                pd.to_numeric(df[c_flux_err], errors='coerce').to_numpy(dtype=float)
                if c_flux_err is not None else np.full_like(flux_uJy, np.nan)
            )
            flux_jy = flux_uJy * 1e-6
            flux_err_jy = flux_err_uJy * 1e-6
        else:
            return None
        lambda_Flambda = flux_jy * 1e-23 * c_A_s / wave_A
        lambda_Flambda_err = flux_err_jy * 1e-23 * c_A_s / wave_A

    flux_flam = lambda_Flambda / wave_A
    fluxerr_flam = lambda_Flambda_err / wave_A
    valid = np.isfinite(wave_A) & np.isfinite(flux_flam) & (wave_A > 0)
    if np.sum(valid) == 0:
        return None
    order = np.argsort(wave_A[valid])
    idx = np.where(valid)[0][order]

    std = pd.DataFrame({
        'lambda_um': wave_um[idx],
        'lambda_Flambda': lambda_Flambda[idx],
        'lambda_Flambda_err': lambda_Flambda_err[idx],
        'flux_jy': flux_jy[idx],
        'flux_err_jy': flux_err_jy[idx],
        'wavelength_A': wave_A[idx],
        'flux_flam': flux_flam[idx],
        'fluxerr_flam': fluxerr_flam[idx],
        'flam': flux_flam[idx],
        'flux_cgs': flux_flam[idx],
        'fluxerr_cgs': fluxerr_flam[idx],
    })
    for name in ['lambda_width', 'flags', 'fit_ql', 'flux_bkg',
                 'lvf_id', 'det_id', 'deep_flg', 'mjd',
                 'x_image', 'y_image', 'ra', 'dec',
                 'obs_id', 'obs_publisher_did', 'band']:
        col = _find_col(df, [name])
        if col is not None:
            std[name] = np.asarray(df[col])[idx]

    return {
        'survey': 'SPHEREx',
        'method': method,
        'ra': ra, 'dec': dec,
        'wavelength': std['wavelength_A'].to_numpy(dtype=float),
        'flux': std['flux_flam'].to_numpy(dtype=float),
        'error': std['fluxerr_flam'].to_numpy(dtype=float),
        'lambda_um': std['lambda_um'].to_numpy(dtype=float),
        'lambda_Flambda': std['lambda_Flambda'].to_numpy(dtype=float),
        'lambda_Flambda_err': std['lambda_Flambda_err'].to_numpy(dtype=float),
        'flux_jy': std['flux_jy'].to_numpy(dtype=float),
        'flux_err_jy': std['flux_err_jy'].to_numpy(dtype=float),
        'n_channels': int(len(std)),
        'table': std,
        'raw_columns': list(df.columns),
        'is_official_spectrophotometry': True,
    }


def _query_sia_v2(ra, dec, collection=None):
    """
    通过 IRSA SIA v2 查询覆盖目标坐标的 SPHEREx spectral image 列表。

    Returns:
        list of dict, 每个包含 access_url, em_min, em_max, band_name
    """
    collections = [collection] if collection else SPHEREX_COLLECTIONS

    for coll in collections:
        try:
            params = {
                'COLLECTION': coll,
                'POS': f'CIRCLE {ra} {dec} 0.01',
                'RESPONSEFORMAT': 'votable',
            }
            session = utils.get_session(SIA_V2_URL)
            resp = session.get(SIA_V2_URL, params=params,
                               timeout=utils.get_timeout())
            if resp.status_code != 200:
                continue

            from astropy.io.votable import parse
            vot = parse(io.BytesIO(resp.content))

            images = []
            for resource in vot.resources:
                if resource.type != 'results':
                    continue
                for table in resource.tables:
                    t = table.to_table()
                    if len(t) == 0:
                        continue

                    col_names = t.colnames
                    name_to_col = {
                        f.name: col_names[j]
                        for j, f in enumerate(table.fields)
                        if f.name is not None
                    }

                    # 关键列索引
                    idx_map = {}
                    for fname, ucd_target in [
                        ('access_url', 'meta.ref.url'),
                        ('em_min', 'em.wl;stat.min'),
                        ('em_max', 'em.wl;stat.max'),
                    ]:
                        for j, f in enumerate(table.fields):
                            if f.name == fname or (f.ucd and ucd_target in f.ucd):
                                idx_map[fname] = col_names[j]
                                break

                    if 'access_url' not in idx_map:
                        continue

                    url_col = idx_map['access_url']
                    em_min_col = idx_map.get('em_min')
                    em_max_col = idx_map.get('em_max')
                    # band name column
                    band_col = None
                    for j, f in enumerate(table.fields):
                        if f.name == 'energy_bandpassname':
                            band_col = col_names[j]
                            break

                    for row in t:
                        url = str(row[url_col])
                        if not url or url == '--':
                            continue
                        info = {'access_url': url}
                        for name in [
                            'obs_id', 'obs_publisher_did', 'cloud_access',
                            't_min', 't_max', 't_resolution',
                            's_ra', 's_dec', 's_pixel_scale',
                            'dist_to_point', 'access_format', 'access_estsize',
                            'obs_collection',
                        ]:
                            col = name_to_col.get(name)
                            if col is not None:
                                info[name] = _scalar(row[col])
                        if em_min_col:
                            info['em_min'] = float(row[em_min_col])
                            info['em_max'] = float(row[em_max_col])
                            # 中心波长 (um)
                            info['wave_um'] = (info['em_min'] +
                                               info['em_max']) / 2.0 * 1e6
                        if band_col:
                            info['band'] = str(row[band_col])
                        images.append(info)

            if images:
                return images

        except Exception as e:
            print(f"  SPHEREx SIA v2 查询失败 ({coll}): {e}")
            continue

    return None


def _scalar(value):
    """Convert astropy/numpy scalar-ish values to plain Python values."""
    try:
        if hasattr(value, 'mask') and bool(value.mask):
            return np.nan
    except Exception:
        pass
    try:
        value = value.item()
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    return value


def _deduplicate_by_wavelength(image_list, tol_um=0.01):
    """
    按中心波长去重: 同一波长多次观测只保留一个。
    """
    if not image_list:
        return []

    # 按波长排序
    sorted_imgs = sorted(image_list, key=lambda x: x.get('wave_um', 0))

    unique = [sorted_imgs[0]]
    for img in sorted_imgs[1:]:
        w = img.get('wave_um', 0)
        w_last = unique[-1].get('wave_um', 0)
        if abs(w - w_last) > tol_um:
            unique.append(img)
    return unique


def _extract_pixel_from_cutout(img_info, ra, dec, cutout_size, session):
    """
    下载 SPHEREx FITS cutout, 提取中心像素的流量和波长。
    **带背景扣除**: 用 cutout 外环像素中位数估计天空背景, 减去后得到纯源流量。

    SPHEREx 像素为 6.2", 对于暗源 (>18 mag) 中心像素被背景主导,
    不做背景扣除会导致流量严重偏高。

    SPHEREx Level 2 FITS 结构:
      [1] IMAGE: 流量 (MJy/sr)
      [3] VARIANCE: 方差
      [6] WCS-WAVE: 波长信息, VALUES shape (ny, nx, 2) = [wave_um, bandwidth_um]

    Returns:
        dict: {wave_um, bandwidth_um, flux_mjy_sr, error_mjy_sr} 或 None
    """
    url = img_info['access_url']
    cutout_url = f"{url}?center={ra},{dec}&size={cutout_size}pix"

    try:
        resp = session.get(cutout_url, timeout=(30, 120))
        if resp.status_code != 200:
            return None

        from astropy.io import fits as pyfits
        hdul = pyfits.open(io.BytesIO(resp.content))

        # IMAGE HDU (index 1)
        if len(hdul) < 2 or hdul[1].data is None:
            hdul.close()
            return None

        img = hdul[1].data
        hdr = hdul[1].header
        ny, nx = img.shape
        cy, cx = ny // 2, nx // 2
        cutout_x, cutout_y = float(cx + 1), float(cy + 1)
        try:
            from astropy.wcs import WCS
            cutout_x, cutout_y = WCS(hdr).all_world2pix([[ra, dec]], 1)[0]
            cutout_x, cutout_y = float(cutout_x), float(cutout_y)
        except Exception:
            pass
        x_offset = hdr.get('CRPIX1A', hdr.get('CRPIX1W'))
        y_offset = hdr.get('CRPIX2A', hdr.get('CRPIX2W'))
        x_image = (
            cutout_x - float(x_offset)
            if x_offset is not None else np.nan
        )
        y_image = (
            cutout_y - float(y_offset)
            if y_offset is not None else np.nan
        )

        # 中心像素原始流量 (MJy/sr)
        raw_flux = float(img[cy, cx])
        if not np.isfinite(raw_flux):
            hdul.close()
            return None

        # 背景估计: 用外环像素 (排除中心 1 像素) 的中位数
        # 构建掩膜: 中心像素=False, 其余=True
        mask = np.ones((ny, nx), dtype=bool)
        mask[cy, cx] = False
        # 也排除无效值
        valid_bg = mask & np.isfinite(img) & (img != 0)
        if np.sum(valid_bg) >= 4:
            bg = np.median(img[valid_bg])
        else:
            bg = 0.0

        flux = raw_flux - bg
        flags = np.nan
        if len(hdul) > 2 and hdul[2].data is not None:
            try:
                flags = int(hdul[2].data[cy, cx])
            except Exception:
                flags = np.nan

        # 方差 (VARIANCE HDU, index 3)
        error = 0.0
        if len(hdul) > 3 and hdul[3].data is not None:
            var_center = float(hdul[3].data[cy, cx])
            if np.isfinite(var_center) and var_center > 0:
                # 背景方差: 外环像素方差的中位数 / N_bg_pixels (背景估计的误差)
                var_img = hdul[3].data
                valid_bg_var = mask & np.isfinite(var_img) & (var_img > 0)
                if np.sum(valid_bg_var) >= 4:
                    bg_var = np.median(var_img[valid_bg_var]) / np.sum(valid_bg_var)
                else:
                    bg_var = 0.0
                error = np.sqrt(var_center + bg_var)

        # 波长 (WCS-WAVE HDU, index 6)
        # X/Y 是 full-frame 像素网格，VALUES[..., 0/1] 是 wavelength/width。
        # IBE cutout header 的 CRPIX1A/2A 给出 cutout → full-frame 偏移；
        # 必须在源的 full-frame 像素位置插值，不能取 WCS-WAVE 网格中心。
        wave_um = img_info.get('wave_um', 0)
        bandwidth_um = 0.03  # default
        if len(hdul) > 6 and hdul[6].data is not None:
            try:
                wcs_data = hdul[6].data[0]
                wave_um, bandwidth_um = _interpolate_wcs_wave(
                    wcs_data, x_image, y_image,
                    fallback_wave=wave_um,
                    fallback_width=bandwidth_um,
                )
            except Exception:
                pass

        hdul.close()

        if wave_um <= 0.1:
            return None

        return {
            'wave_um': wave_um,
            'bandwidth_um': bandwidth_um,
            'flux_mjy_sr': flux,
            'error_mjy_sr': error,
            'raw_flux_mjy_sr': raw_flux,
            'background_mjy_sr': float(bg),
            'flags': flags,
            'mjd': hdr.get('MJD-AVG', hdr.get('MJD-OBS', np.nan)),
            'mjd_beg': hdr.get('MJD-BEG', hdr.get('MJD-OBS', np.nan)),
            'mjd_end': hdr.get('MJD-END', hdr.get('MJD-AVG', np.nan)),
            'date_obs': hdr.get('DATE-OBS'),
            'lvf_id': hdr.get('EXPIDN'),
            'obs_id': hdr.get('OBSID', img_info.get('obs_id')),
            'det_id': hdr.get('DETECTOR', _det_id_from_band(img_info.get('band'))),
            'deep_flg': bool(hdr.get('OBS_IN_SBAND', False)),
            'x_image': x_image,
            'y_image': y_image,
            'cutout_x_center': cutout_x,
            'cutout_y_center': cutout_y,
        }

    except Exception:
        return None


def get_photometry(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    兼容旧接口。

    SPHEREx spectrophotometry 不是宽带测光。为了避免把低分辨谱合成为
    伪宽带点并错误惩罚 SED chi2，这里不再返回 SPHEREx_* 测光点。
    请用 ``query_spectrum()`` 获取带 MJD/flags/fit_ql 的校准谱光度表。
    """
    return {}


def plot_spectrum(spec, save_path=None):
    """绘制 SPHEREx 低分辨率光谱"""
    if spec is None:
        return None

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 4.0))

    def _optional_array(value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        return arr if arr.ndim > 0 else None

    if 'lambda_Flambda' in spec:
        flux = np.asarray(spec['lambda_Flambda'], dtype=float)
        error = _optional_array(spec.get('lambda_Flambda_err'))
        ylabel = r'$\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)'
    elif 'flux_mjy_sr' in spec:
        flux = np.asarray(spec['flux_mjy_sr'], dtype=float)
        error = _optional_array(spec.get('error'))
        ylabel = 'Flux (MJy sr$^{-1}$; quick-look)'
    else:
        flux = np.asarray(spec['flux'], dtype=float)
        error = _optional_array(spec.get('error'))
        ylabel = r'$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)'

    wave = np.asarray(spec.get('lambda_um', np.asarray(spec['wavelength']) / 10000.0),
                      dtype=float)
    if wave.shape != flux.shape:
        fallback_wave = np.asarray(spec.get('wavelength', []), dtype=float)
        if fallback_wave.shape == flux.shape and len(fallback_wave) > 0:
            wave = fallback_wave / 10000.0
        else:
            n = min(len(wave), len(flux))
            wave = wave[:n]
            flux = flux[:n]
            if error is not None:
                error = error[:n]

    valid = np.isfinite(flux) & (flux != 0)
    ax.plot(wave[valid], flux[valid], color='black', marker='o', lw=0.7,
            ms=2.5, alpha=0.85, label='SPHEREx spectrophotometry')
    if error is not None and len(error) == len(flux) and np.any(error > 0):
        err_valid = error[valid] if len(error) == len(flux) else None
        if err_valid is not None:
            ax.fill_between(wave[valid], flux[valid] - err_valid,
                            flux[valid] + err_valid,
                            color='0.55', alpha=0.18, lw=0)

    ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    method = spec.get('method', 'spectrophotometry')
    ax.set_title(f"SPHEREx {method}  RA={spec['ra']:.4f} DEC={spec['dec']:.4f}  "
                 f"({spec.get('n_channels', 0)} points)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # 轴范围紧凑到光谱数据 (单位是 um, 直接设置)
    w_valid = wave[valid]
    f_valid = flux[valid]
    if len(w_valid) > 1:
        wmin, wmax = w_valid.min(), w_valid.max()
        dw = max((wmax - wmin) * 0.02, 0.01)
        ax.set_xlim(wmin - dw, wmax + dw)
        flo, fhi = np.percentile(f_valid[np.isfinite(f_valid)], [1, 99])
        df = max((fhi - flo) * 0.1, abs(fhi) * 0.01)
        ax.set_ylim(flo - df, fhi + df)

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_spectrum_csv(result, output_dir):
    """保存 SPHEREx 光谱为 CSV"""
    import pandas as pd
    if result is None or 'wavelength' not in result:
        return None
    method = result.get('method', '')
    if 'table' in result:
        table = result['table'].copy()
        is_official = bool(result.get('is_official_spectrophotometry', False))
        full_name = (
            'spherex_spectrophotometry_full.csv'
            if is_official else 'spherex_raw_channels_full.csv'
        )
        table_path = utils.write_csv(
            table, output_dir, full_name)
        core_cols = [
            'lambda_um', 'wavelength_A', 'lambda_width',
            'lambda_Flambda', 'lambda_Flambda_err',
            'flam', 'flux_cgs', 'fluxerr_cgs',
            'flux_jy', 'flux_err_jy',
            'flux', 'flux_err', 'flux_uJy', 'flux_err_uJy',
            'mjd', 'x_image', 'y_image',
            'det_id', 'lvf_id', 'obs_id', 'obs_publisher_did',
            'flags', 'fit_ql', 'deep_flg', 'extraction_method',
        ]
        cols = [c for c in core_cols if c in table.columns]
        df = table[cols].copy() if cols else table
    else:
        err = result.get('error')
        if err is None:
            err = np.full_like(np.asarray(result['wavelength'], dtype=float), np.nan)
        wave_A = np.asarray(result['wavelength'], dtype=float)
        flux = np.asarray(result['flux'], dtype=float)
        df = pd.DataFrame({
            'lambda_um': result.get('lambda_um', wave_A / 1e4),
            'wavelength_A': wave_A,
            'lambda_Flambda': result.get(
                'lambda_Flambda',
                wave_A * flux),
            'lambda_Flambda_err': result.get(
                'lambda_Flambda_err',
                wave_A * np.asarray(err)),
            'flam': flux,
            'flux_cgs': flux,
            'fluxerr_cgs': err,
            'flux_jy': result.get('flux_jy'),
            'flux_err_jy': result.get('flux_err_jy'),
        })
        table_path = None
    path = utils.write_csv(df, output_dir, 'spherex_spectrum.csv')
    return {'spectrum_csv': path, 'full_table_csv': table_path, 'method': method}


def save_photometry_csv(result, output_dir):
    """保存 SPHEREx 合成测光为 CSV"""
    df = utils.photometry_to_dataframe(result)
    return utils.write_csv(df, output_dir, 'spherex_photometry.csv')
