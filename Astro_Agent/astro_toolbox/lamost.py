"""
LAMOST 光谱查询 (DR12 token/FTP + Vizier 回退)
=============================================
低分辨率 (R~1800): 3700-9100 A
中分辨率 (R~7500): 4950-5350 + 6300-6800 A

优先使用用户在 LAMOST 官网 My Data Disk Requests 申请到的 DR12
FTP/pylamost 访问信息；如果本地没有 token 或远端不可用，再回退到 Vizier
LAMOST DR8 星表 (V/164)。

用法:
    from astro_toolbox.lamost import query_spectrum, plot_spectrum
    result = query_spectrum(190.305, 2.596)
"""
import ftplib
import io
import os
import tempfile

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from . import config, utils

# LAMOST FITS 下载 (通过 obsid)
LAMOST_FITS_URL = "https://dr8.lamost.org/spectrum/fits"
LAMOST_DR12_WEB_FITS_URLS = (
    "https://www.lamost.org/dr12/v1.0/spectrum/fits/{obsid}",
    "https://dr12.lamost.org/v1.0/spectrum/fits/{obsid}",
)
FITS_SUFFIXES = ('.fits', '.fit', '.fits.gz', '.fit.gz')


def _safe_float(val):
    try:
        v = float(val)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError, np.ma.MaskError):
        return None


def _safe_str(val):
    try:
        if np.ma.is_masked(val):
            return ''
        return str(val)
    except (ValueError, TypeError):
        return ''


def _read_spectrum_from_hdul(hdul):
    """从 LAMOST 常见 1D FITS 格式读取 wavelength/flux/error。"""
    header = hdul[0].header
    data = hdul[0].data
    wave = flux = err = None

    if data is not None:
        arr = np.asarray(data)
        if arr.ndim == 1:
            flux = arr.astype(float)
        elif arr.ndim >= 2:
            flux = np.asarray(arr[0], dtype=float)
            if arr.shape[0] > 1:
                ivar = np.asarray(arr[1], dtype=float)
                good = np.isfinite(ivar) & (ivar > 0)
                err = np.full_like(ivar, np.nan, dtype=float)
                err[good] = 1.0 / np.sqrt(ivar[good])
            if arr.shape[0] > 2:
                maybe_wave = np.asarray(arr[2], dtype=float)
                if np.nanmedian(maybe_wave) > 100:
                    wave = maybe_wave

    # 部分 LAMOST/SDSS-like 产品把光谱存在二进制表。
    if flux is None:
        for hdu in hdul[1:]:
            tab = getattr(hdu, 'data', None)
            names = list(getattr(tab, 'names', []) or [])
            lower = {n.lower(): n for n in names}
            fcol = lower.get('flux')
            wcol = (lower.get('wavelength') or lower.get('wave')
                    or lower.get('lambda') or lower.get('loglam'))
            if fcol and wcol:
                flux = np.asarray(tab[fcol], dtype=float)
                wave = np.asarray(tab[wcol], dtype=float)
                if wcol.lower() == 'loglam' or np.nanmedian(wave) < 100:
                    wave = 10 ** wave
                ecol = lower.get('ivar') or lower.get('inverse_variance')
                if ecol:
                    ivar = np.asarray(tab[ecol], dtype=float)
                    err = np.full_like(ivar, np.nan, dtype=float)
                    good = np.isfinite(ivar) & (ivar > 0)
                    err[good] = 1.0 / np.sqrt(ivar[good])
                else:
                    ecol = lower.get('error') or lower.get('sigma')
                    if ecol:
                        err = np.asarray(tab[ecol], dtype=float)
                break

    if flux is None:
        return None, None, None

    if wave is None:
        n = len(flux)
        crval1 = header.get('CRVAL1', 3700)
        cdelt1 = header.get('CD1_1', header.get('CDELT1', 1.0))
        crpix1 = header.get('CRPIX1', 1)
        wave = crval1 + (np.arange(n) - (crpix1 - 1)) * cdelt1
        if header.get('DC-FLAG', 0) == 1 or crval1 < 10:
            wave = 10 ** wave

    return np.asarray(wave, dtype=float), np.asarray(flux, dtype=float), err


def _result_from_hdul(hdul, ra, dec, source='', remote_path='', obsid=''):
    header = hdul[0].header
    wave, flux, err = _read_spectrum_from_hdul(hdul)
    if wave is None or flux is None:
        return None

    result = {
        'survey': 'LAMOST',
        'data_release': config.LAMOST_DR.upper(),
        'access': source,
        'remote_path': remote_path,
        'ra': ra,
        'dec': dec,
        'obsid': str(obsid or header.get('OBSID', header.get('OBS_ID', ''))),
        'rv': _safe_float(header.get('HELIO_RV', header.get('RV'))),
        'teff': _safe_float(header.get('TEFF')),
        'logg': _safe_float(header.get('LOGG')),
        'feh': _safe_float(header.get('FEH', header.get('FE_H'))),
        'snr_g': _safe_float(header.get('SNRG', header.get('SNR_G'))),
        'class': _safe_str(header.get('CLASS', header.get('OBJTYPE', ''))),
        'subclass': _safe_str(header.get('SUBCLASS', '')),
        'obs_date': _safe_str(header.get('DATE-OBS', header.get('DATE', ''))),
        'obs_mjd': _safe_float(header.get('MJD', header.get('LMJD'))),
        'wavelength': wave,
        'flux': flux,
    }
    if err is not None and len(err) == len(flux):
        result['error'] = err
    return result


def _candidate_header_coord(header):
    ra = _safe_float(header.get('RA', header.get('RAJ2000')))
    dec = _safe_float(header.get('DEC', header.get('DECJ2000')))
    if ra is None:
        ra = _safe_float(header.get('OBJRA'))
    if dec is None:
        dec = _safe_float(header.get('OBJDEC'))
    if ra is None or dec is None:
        return None
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)


def _angular_sep_arcsec(header, ra, dec):
    coord = _candidate_header_coord(header)
    if coord is None:
        return None
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    return float(target.separation(coord).arcsec)


def _download_dr12_web(obsid):
    if not obsid:
        return None
    token = getattr(config, 'LAMOST_TOKEN', '') or ''
    headers = {'Authorization': f'token {token}'} if token else {}
    params = {'token': token} if token else None
    for tmpl in LAMOST_DR12_WEB_FITS_URLS:
        url = tmpl.format(obsid=obsid)
        try:
            resp = utils.get_session(url).get(
                url, params=params, headers=headers, timeout=utils.get_timeout())
            resp.raise_for_status()
            if resp.content[:6].startswith(b'SIMPLE') or b'SIMPLE' in resp.content[:80]:
                return fits.open(io.BytesIO(resp.content))
        except Exception:
            continue
    return None


def _query_vizier_row(ra, dec, radius_arcsec):
    tbl = utils.query_vizier(
        'V/164/dr8lrs', ra, dec, radius_arcsec,
        columns=['ObsID', 'RAJ2000', 'DEJ2000', 'RV', 'Teff', 'logg',
                 'FeH', 'snrg', 'Class', 'SubClass', 'ObsDate',
                 'lmjd', 'planid', 'spid', 'fiberid'])
    if tbl is None:
        return None
    return tbl[0]


def _ftp_credentials_available():
    return bool(getattr(config, 'LAMOST_FTP_SERVER', '')
                and getattr(config, 'LAMOST_FTP_USER', '')
                and getattr(config, 'LAMOST_FTP_PASSWORD', ''))


def _is_fits_name(name):
    low = name.lower()
    return low.endswith(FITS_SUFFIXES)


def _ftp_recursive_list(ftp, path='.', depth=0, max_depth=None, rows=None):
    rows = rows if rows is not None else []
    max_depth = config.LAMOST_FTP_MAX_DEPTH if max_depth is None else max_depth
    if depth > max_depth:
        return rows
    try:
        entries = list(ftp.mlsd(path))
    except Exception:
        try:
            names = ftp.nlst(path)
        except Exception:
            return rows
        entries = [(name, {'type': 'file'}) for name in names]

    for name, facts in entries:
        if name in ('.', '..'):
            continue
        full = name if name.startswith('/') else f"{path.rstrip('/')}/{name}"
        kind = (facts or {}).get('type', '')
        if kind == 'dir' or (not _is_fits_name(name) and '.' not in os.path.basename(name)):
            _ftp_recursive_list(ftp, full, depth + 1, max_depth, rows)
        elif _is_fits_name(name):
            rows.append(full)
    return rows


def _load_or_build_ftp_manifest(force_refresh=False):
    import pandas as pd
    manifest_path = config.LAMOST_FTP_MANIFEST
    if (not force_refresh) and os.path.exists(manifest_path):
        try:
            return pd.read_csv(manifest_path)
        except Exception:
            pass
    if not _ftp_credentials_available():
        return None

    utils.ensure_dir(os.path.dirname(manifest_path))
    with ftplib.FTP(config.LAMOST_FTP_SERVER, timeout=config.CONNECT_TIMEOUT) as ftp:
        ftp.login(config.LAMOST_FTP_USER, config.LAMOST_FTP_PASSWORD)
        paths = _ftp_recursive_list(ftp)

    rows = [{'remote_path': p, 'filename': os.path.basename(p)} for p in paths]
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(manifest_path, index=False)
    return df


def _open_ftp_fits(remote_path):
    suffix = '.fits.gz' if remote_path.lower().endswith('.gz') else '.fits'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    try:
        with ftplib.FTP(config.LAMOST_FTP_SERVER, timeout=config.CONNECT_TIMEOUT) as ftp:
            ftp.login(config.LAMOST_FTP_USER, config.LAMOST_FTP_PASSWORD)
            with open(tmp.name, 'wb') as fh:
                ftp.retrbinary(f"RETR {remote_path}", fh.write)
        return fits.open(tmp.name)
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def _query_spectrum_ftp(ra, dec, radius_arcsec, force_manifest_refresh=False):
    if not _ftp_credentials_available():
        return None
    try:
        manifest = _load_or_build_ftp_manifest(force_refresh=force_manifest_refresh)
    except Exception as exc:
        print(f"LAMOST DR12 FTP 清单读取失败: {exc}")
        return None
    if manifest is None or len(manifest) == 0 or 'remote_path' not in manifest:
        return None

    candidates = []
    # 先抽样/扫描 FITS 头，只下载 header 即可时 astropy 会懒加载；FTP 仍需取文件。
    # My Data Disk Request 通常是目标小批量，完整扫描可接受且会缓存路径。
    for remote_path in manifest['remote_path'].dropna().astype(str):
        try:
            hdul = _open_ftp_fits(remote_path)
            sep = _angular_sep_arcsec(hdul[0].header, ra, dec)
            if sep is not None and sep <= radius_arcsec:
                snr = (_safe_float(hdul[0].header.get('SNRG'))
                       or _safe_float(hdul[0].header.get('SNRR')) or 0.0)
                candidates.append((sep, -snr, remote_path, hdul))
            else:
                hdul.close()
        except Exception:
            continue

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    sep, _, remote_path, hdul = candidates[0]
    try:
        result = _result_from_hdul(
            hdul, ra, dec, source='LAMOST_DR12_FTP',
            remote_path=remote_path)
        if result:
            result['match_sep_arcsec'] = float(sep)
        return result
    finally:
        hdul.close()


def _query_spectrum_pylamost(ra, dec, radius_arcsec):
    """
    Best-effort pylamost entry point.

    pylamost has changed API names across user installations, so this wrapper
    tries common callable names and normalizes either a FITS path, HDUList, or
    dict-like spectrum result.
    """
    try:
        import pylamost  # type: ignore
    except Exception:
        return None

    token = getattr(config, 'LAMOST_TOKEN', '') or ''
    call_specs = (
        ('query_spectrum', {'ra': ra, 'dec': dec, 'radius': radius_arcsec, 'token': token}),
        ('get_spectrum', {'ra': ra, 'dec': dec, 'radius': radius_arcsec, 'token': token}),
        ('cone_search', {'ra': ra, 'dec': dec, 'radius': radius_arcsec, 'token': token}),
    )
    for name, kwargs in call_specs:
        func = getattr(pylamost, name, None)
        if not callable(func):
            continue
        try:
            obj = func(**kwargs)
        except TypeError:
            kwargs.pop('token', None)
            try:
                obj = func(**kwargs)
            except Exception:
                continue
        except Exception:
            continue

        try:
            if isinstance(obj, fits.HDUList):
                return _result_from_hdul(obj, ra, dec, source='pylamost')
            if isinstance(obj, str) and os.path.exists(obj):
                with fits.open(obj) as hdul:
                    return _result_from_hdul(hdul, ra, dec, source='pylamost',
                                             remote_path=obj)
            if isinstance(obj, dict):
                if 'wavelength' in obj and 'flux' in obj:
                    res = dict(obj)
                    res.setdefault('survey', 'LAMOST')
                    res.setdefault('data_release', config.LAMOST_DR.upper())
                    res.setdefault('access', 'pylamost')
                    return res
                for key in ('fits_path', 'file', 'filename', 'path'):
                    p = obj.get(key)
                    if isinstance(p, str) and os.path.exists(p):
                        with fits.open(p) as hdul:
                            return _result_from_hdul(
                                hdul, ra, dec, source='pylamost',
                                remote_path=p)
        except Exception:
            continue
    return None


def query_spectrum(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 LAMOST 光谱。优先 DR12 token/FTP/pylamost，回退 Vizier DR8。

    Returns:
        dict: {'obsid', 'ra', 'dec', 'rv', 'teff', 'logg', 'feh',
               'snr_g', 'class', 'subclass', 'obs_date', 'wavelength', 'flux'}
        或 None
    """
    # 1) pylamost / DR12 本地安装入口
    res = _query_spectrum_pylamost(ra, dec, radius_arcsec)
    if res is not None and 'wavelength' in res:
        return res

    # 2) My Data Disk Requests 的 DR12 FTP 文件
    res = _query_spectrum_ftp(ra, dec, radius_arcsec)
    if res is not None and 'wavelength' in res:
        return res

    # 3) 旧 DR8 VizieR 星表 + 尝试 DR12/DR8 FITS URL 下载
    row = _query_vizier_row(ra, dec, radius_arcsec)
    if row is None:
        return None

    try:
        obsid = str(row['ObsID'])
    except (KeyError, ValueError):
        obsid = ''

    obs_date = _safe_str(row.get('ObsDate', ''))
    lmjd = _safe_float(row.get('lmjd', None))

    result = {
        'survey': 'LAMOST',
        'data_release': 'DR8_catalog_DR12_url_attempted',
        'access': 'vizier_fallback',
        'ra': ra, 'dec': dec,
        'obsid': obsid,
        'rv': _safe_float(row.get('RV')),
        'teff': _safe_float(row.get('Teff')),
        'logg': _safe_float(row.get('logg')),
        'feh': _safe_float(row.get('FeH')),
        'snr_g': _safe_float(row.get('snrg')),
        'class': _safe_str(row.get('Class', '')),
        'subclass': _safe_str(row.get('SubClass', '')),
        'obs_date': obs_date,
        'obs_mjd': lmjd,
    }

    # 下载光谱 FITS
    if obsid:
        try:
            hdul = _download_dr12_web(obsid)
            if hdul is None:
                spec_url = f"{LAMOST_FITS_URL}/{obsid}"
                resp = utils.get_session(spec_url).get(spec_url, timeout=utils.get_timeout())
                resp.raise_for_status()
                hdul = fits.open(io.BytesIO(resp.content))
            wave, flux, err = _read_spectrum_from_hdul(hdul)
            result['wavelength'] = wave
            result['flux'] = flux
            if err is not None:
                result['error'] = err
            hdul.close()
        except Exception as e:
            print(f"LAMOST 光谱下载失败: {e}")

    return result


def plot_spectrum(result, save_path=None):
    """绘制 LAMOST 光谱"""
    if result is None or 'wavelength' not in result:
        return None
    fig, ax = utils.setup_spectrum_plot()
    ax.plot(result['wavelength'], result['flux'], 'k-', lw=0.6)
    dr = result.get('data_release', '')
    title = f"LAMOST {dr} {result.get('obsid','')}  class={result.get('class','')}"
    if result.get('teff'):
        title += f"  Teff={result['teff']:.0f}K"
    if result.get('obs_date'):
        title += f"  ({result['obs_date']})"
    ax.set_title(title)
    ax.set_ylabel('Flux')

    # 轴范围紧凑到光谱数据
    utils.set_spectrum_axes(ax, result['wavelength'], result['flux'])

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_csv(result, output_dir):
    """保存 LAMOST 光谱和参数为 CSV"""
    if result is None:
        return None
    paths = []
    # 光谱数据
    if 'wavelength' in result:
        spec_df = utils.spectrum_to_dataframe(result)
        p = utils.write_csv(spec_df, output_dir, 'lamost_spectrum.csv')
        if p:
            paths.append(p)
    # 参数 (单行)
    param_keys = ['survey', 'data_release', 'access', 'remote_path',
                  'match_sep_arcsec', 'ra', 'dec', 'obsid', 'rv', 'teff',
                  'logg', 'feh', 'snr_g', 'class', 'subclass', 'obs_date',
                  'obs_mjd']
    param_df = utils.keyvalue_to_dataframe(result, keys=param_keys)
    p = utils.write_csv(param_df, output_dir, 'lamost_params.csv')
    if p:
        paths.append(p)
    return paths[0] if paths else None
