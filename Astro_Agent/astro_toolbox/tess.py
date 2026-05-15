"""
TESS 光变曲线
=============
使用 lightkurve 包查询和下载 TESS 光变曲线

用法:
    from astro_toolbox.tess import query_lightcurve, plot_lightcurve
    lc = query_lightcurve(190.305, 2.596)
    plot_lightcurve(lc, save_path='tess_lc.png')
"""
import numpy as np
from . import config, utils


def _row_get(row, name, default=None):
    try:
        return row[name]
    except Exception:
        return default


def _cache_path_for_product(row):
    import os

    filename = str(_row_get(row, 'productFilename', '') or '').strip()
    obs_id = str(_row_get(row, 'obs_id', '') or '').strip()
    if not filename:
        return None
    cache_dir = os.path.expanduser(os.path.join('~', '.lightkurve', 'cache',
                                                'mastDownload', 'TESS', obs_id))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


def _valid_tess_fits(path):
    from astropy.io import fits

    if not path:
        return False
    try:
        with fits.open(path, memmap=False) as hdul:
            if len(hdul) <= 1 or hdul[1].data is None or len(hdul[1].data) == 0:
                return False
            _ = np.asarray(hdul[1].data['TIME'], dtype=float)
            return True
    except Exception:
        return False


def _download_product(row):
    import os
    from urllib.parse import quote

    path = _cache_path_for_product(row)
    if path is None:
        return None
    if os.path.exists(path) and _valid_tess_fits(path):
        return path
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

    uri = str(_row_get(row, 'dataURI', '') or '').strip()
    if not uri:
        return None
    url = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=' + quote(uri, safe=':/')
    part = path + '.partial'
    try:
        # Direct MAST downloads are more reliable here than tunneling the large
        # FITS stream through the generic proxy session.
        session = utils.get_session_no_proxy()
        resp = session.get(url, stream=True, timeout=(60, 600))
        resp.raise_for_status()
        expected = int(resp.headers.get('content-length') or 0)
        with open(part, 'wb') as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 512):
                if chunk:
                    fh.write(chunk)
        if expected and os.path.getsize(part) < expected:
            raise IOError(f"incomplete TESS download: {os.path.getsize(part)} < {expected} bytes")
        os.replace(part, path)
    except Exception:
        try:
            os.remove(part)
        except OSError:
            pass
        return None
    if not _valid_tess_fits(path):
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    return path


def _read_tess_fits(path):
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        data = hdul[1].data
        header = hdul[1].header
        primary_header = hdul[0].header
        time = np.asarray(data['TIME'], dtype=float)
        if 'PDCSAP_FLUX' in data.names:
            flux = np.asarray(data['PDCSAP_FLUX'], dtype=float)
            err_name = 'PDCSAP_FLUX_ERR'
        else:
            flux = np.asarray(data['SAP_FLUX'], dtype=float)
            err_name = 'SAP_FLUX_ERR'
        flux_err = np.asarray(data[err_name], dtype=float) if err_name in data.names else np.zeros_like(flux)
        quality = np.asarray(data['QUALITY'], dtype=int) if 'QUALITY' in data.names else np.zeros_like(time, dtype=int)
        sector = header.get('SECTOR', primary_header.get('SECTOR'))

    good = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
    if np.any(np.isfinite(flux_err)):
        good &= np.isfinite(flux_err)
    time = time[good]
    flux = flux[good]
    flux_err = flux_err[good] if len(flux_err) == len(good) else np.zeros_like(flux)
    if len(time) == 0:
        return None
    med = np.nanmedian(flux)
    if np.isfinite(med) and med != 0:
        flux = flux / med
        flux_err = flux_err / abs(med)
    return time, flux, flux_err, sector


def query_lightcurve(ra, dec, author='SPOC'):
    """
    查询 TESS 光变曲线。

    Args:
        author: 'SPOC' (2-min cadence) 或 'TESS-SPOC' 或 'QLP'

    Returns:
        dict: {'time': array, 'flux': array, 'flux_err': array,
               'sector': list, 'author': str}
        或 None
    """
    import lightkurve as lk
    c = f"{ra} {dec}"
    authors = []
    for candidate in (author, 'SPOC', 'TESS-SPOC', 'QLP'):
        if candidate and candidate not in authors:
            authors.append(candidate)

    lc_collection = None
    search = None
    used_author = None
    for candidate in authors:
        try:
            search = lk.search_lightcurve(c, mission='TESS', author=candidate)
        except Exception:
            search = None
        if search is None or len(search) == 0:
            continue

        max_products = int(__import__('os').environ.get('ASTRO_TOOLBOX_TESS_MAX_PRODUCTS', '4'))
        rows = list(search.table)
        rows.sort(key=lambda row: float(_row_get(row, 'exptime', 0) or 0) < 60)
        products = []
        for row in rows[:max_products]:
            path = _download_product(row)
            if not path:
                continue
            product = _read_tess_fits(path)
            if product is not None:
                products.append(product)
        if products:
            time = np.concatenate([p[0] for p in products])
            flux = np.concatenate([p[1] for p in products])
            flux_err = np.concatenate([p[2] for p in products])
            sectors = sorted({p[3] for p in products if p[3] is not None})
            used_author = candidate
            lc_collection = (time, flux, flux_err, sectors)
            break

    if lc_collection is None:
        return None

    time, flux, flux_err, sectors = lc_collection
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    flux_err = flux_err[order]

    return {
        'survey': 'TESS',
        'ra': ra, 'dec': dec,
        'time': time,      # BTJD
        'flux': flux,       # 归一化流量
        'flux_err': flux_err,
        'sectors': sectors,
        'author': used_author or author,
        'n_points': len(time),
        'obs_time_min': float(np.nanmin(time)),
        'obs_time_max': float(np.nanmax(time)),
        'time_system': 'BTJD',
    }


def plot_lightcurve(result, save_path=None):
    """绘制 TESS 光变曲线"""
    if result is None:
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(result['time'], result['flux'], s=0.5, c='black', alpha=0.5)
    ax.set_xlabel('Time (BTJD)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f"TESS Light Curve  RA={result['ra']:.4f} DEC={result['dec']:.4f}  "
                 f"Sectors={result['sectors']}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_csv(result, output_dir):
    """保存 TESS 光变曲线为 CSV"""
    import pandas as pd
    if result is None:
        return None
    df = pd.DataFrame({
        'time_BTJD': result['time'],
        'flux': result['flux'],
        'flux_err': result['flux_err'],
    })
    return utils.write_csv(df, output_dir, 'tess_lightcurve.csv')
