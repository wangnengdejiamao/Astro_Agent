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
    search = lk.search_lightcurve(c, mission='TESS', author=author)
    if search is None or len(search) == 0:
        return None

    # 下载并拼接所有 sector
    lc_collection = search.download_all()
    if lc_collection is None or len(lc_collection) == 0:
        return None

    lc = lc_collection.stitch()

    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value if hasattr(lc.flux_err, 'value') else np.zeros_like(flux)

    sectors = list(set([r.meta.get('SECTOR', '?') for r in lc_collection]))

    return {
        'survey': 'TESS',
        'ra': ra, 'dec': dec,
        'time': time,      # BTJD
        'flux': flux,       # 归一化流量
        'flux_err': flux_err,
        'sectors': sectors,
        'author': author,
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
