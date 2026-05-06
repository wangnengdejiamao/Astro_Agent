"""
Kepler / K2 光变曲线
====================
使用 lightkurve 包

用法:
    from astro_toolbox.kepler import query_lightcurve
    lc = query_lightcurve(290.98, 44.27)  # Kepler field
"""
import numpy as np
from . import config, utils


def query_lightcurve(ra, dec, mission='Kepler'):
    """
    查询 Kepler 或 K2 光变曲线。

    Args:
        mission: 'Kepler' 或 'K2'

    Returns:
        dict 或 None
    """
    import lightkurve as lk
    c = f"{ra} {dec}"
    search = lk.search_lightcurve(c, mission=mission)
    if search is None or len(search) == 0:
        return None

    # 过滤掉不支持的产品 (如 K2SC)
    supported_mask = []
    for i in range(len(search)):
        author = str(search.table['author'][i]).upper() if 'author' in search.table.colnames else ''
        if 'K2SC' in author:
            continue
        supported_mask.append(i)

    if not supported_mask:
        return None

    # 逐个下载, 跳过失败的
    lc_list = []
    for idx in supported_mask:
        try:
            lc_i = search[idx].download()
            if lc_i is not None:
                lc_list.append(lc_i)
        except Exception:
            continue

    if not lc_list:
        return None

    from lightkurve import LightCurveCollection
    lc_collection = LightCurveCollection(lc_list)
    lc = lc_collection.stitch()
    time = lc.time.value
    flux = lc.flux.value
    flux_err = lc.flux_err.value if hasattr(lc.flux_err, 'value') else np.zeros_like(flux)

    return {
        'survey': mission,
        'ra': ra, 'dec': dec,
        'time': time,
        'flux': flux,
        'flux_err': flux_err,
        'n_quarters': len(lc_collection),
        'n_points': len(time),
        'obs_time_min': float(np.nanmin(time)),
        'obs_time_max': float(np.nanmax(time)),
        'time_system': 'BKJD' if mission == 'Kepler' else 'BTJD',
    }


def plot_lightcurve(result, save_path=None):
    """绘制 Kepler/K2 光变曲线"""
    if result is None:
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(result['time'], result['flux'], s=0.3, c='black', alpha=0.5)
    ax.set_xlabel('Time (BKJD)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f"{result['survey']} Light Curve  RA={result['ra']:.4f} "
                 f"DEC={result['dec']:.4f}  N={result['n_points']}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_csv(result, output_dir):
    """保存 Kepler/K2 光变曲线为 CSV"""
    import pandas as pd
    if result is None:
        return None
    time_col = f"time_{result.get('time_system', 'BKJD')}"
    df = pd.DataFrame({
        time_col: result['time'],
        'flux': result['flux'],
        'flux_err': result['flux_err'],
    })
    return utils.write_csv(df, output_dir, 'kepler_lightcurve.csv')
