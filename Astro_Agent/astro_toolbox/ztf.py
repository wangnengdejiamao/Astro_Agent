"""
ZTF DR23 光变曲线
=================
使用 IRSA Light Curve API，通过 COLLECTION 参数固定 DR23。
波段: g, r, i

自动代理回退: 优先走代理，代理不通则直连。

用法:
    from astro_toolbox.ztf import query_lightcurve, plot_lightcurve
    lc = query_lightcurve(190.305, 2.596)
    plot_lightcurve(lc, save_path='ztf_lc.png')
    # 获取网页下载链接
    print(get_web_url(190.305, 2.596))
"""
import io
import numpy as np
import pandas as pd
import requests
from . import config, utils

# ZTF IRSA light curve service
ZTF_LC_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

# IRSA 网页搜索界面 (用于手动下载)
ZTF_WEB_URL = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"


def get_web_url(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC,
                collection='ztf_dr23'):
    """
    生成 ZTF IRSA 网页查询链接，可在浏览器中打开手动下载。

    Returns:
        str: 网页 URL
    """
    radius_deg = radius_arcsec / 3600.0
    return (f"{ZTF_LC_URL}?POS=CIRCLE+{ra}+{dec}+{radius_deg}"
            f"&BANDNAME=g,r,i&NOBS_MIN=3&FORMAT=CSV"
            f"&BAD_CATFLAGS_MASK=32768&COLLECTION={collection}")


def query_lightcurve(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC,
                     bands='g,r,i', collection='ztf_dr23'):
    """
    查询 ZTF DR23 光变曲线。

    Args:
        ra, dec: 坐标 (度)
        radius_arcsec: 搜索半径 (角秒)
        bands: 波段 (默认 'g,r,i')
        collection: 数据集版本 (默认 'ztf_dr23')

    Returns:
        dict: {'g': DataFrame, 'r': DataFrame, 'i': DataFrame, 'ra', 'dec'}
              每个 DataFrame 含 mjd, mag, magerr 列
        或 None
    """
    radius_deg = radius_arcsec / 3600.0
    params = {
        'POS': f'CIRCLE {ra} {dec} {radius_deg}',
        'BANDNAME': bands,
        'NOBS_MIN': '3',
        'FORMAT': 'CSV',
        'BAD_CATFLAGS_MASK': '32768',
        'COLLECTION': collection,
    }

    # 优先走代理，失败后自动回退直连
    resp = None
    for attempt_proxy in (True, False):
        try:
            if attempt_proxy:
                session = utils.get_session(ZTF_LC_URL)
            else:
                session = utils.get_session_no_proxy()
                print("  ZTF: 代理不可用，尝试直连...")
            resp = session.get(ZTF_LC_URL, params=params,
                               timeout=utils.get_timeout())
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt_proxy and config.PROXY_URL:
                continue  # 尝试直连
            print(f"ZTF 查询失败: {e}")
            web = get_web_url(ra, dec, radius_arcsec, collection)
            print(f"  可在浏览器中手动下载: {web}")
            return None

    if resp is None:
        return None

    text = resp.text.strip()
    if not text or 'mjd' not in text.lower():
        return None

    df = pd.read_csv(io.StringIO(text))
    if len(df) == 0:
        return None

    # 统一列名 (ZTF API 返回的列名可能有变化)
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == 'mjd':
            col_map[col] = 'mjd'
        elif cl == 'hjd':
            col_map[col] = 'hjd'
        elif cl == 'mag':
            col_map[col] = 'mag'
        elif cl == 'magerr':
            col_map[col] = 'magerr'
        elif cl in ('filtercode', 'filter', 'bandname', 'filterid'):
            col_map[col] = 'band'
        elif cl == 'catflags':
            col_map[col] = 'catflags'
    df = df.rename(columns=col_map)

    # 确保 mjd 存在 (由 hjd 推算)
    if 'mjd' not in df.columns and 'hjd' in df.columns:
        df['mjd'] = df['hjd'] - 2400000.5

    # 客户端严格过滤: catflags == 0 (比服务端 BAD_CATFLAGS_MASK 更严格)
    if 'catflags' in df.columns:
        df = df[df['catflags'] == 0].copy()

    result = {'ra': ra, 'dec': dec, 'survey': collection.upper(),
              'web_url': get_web_url(ra, dec, radius_arcsec, collection)}
    band_map = {'zg': 'g', 'zr': 'r', 'zi': 'i', 'g': 'g', 'r': 'r', 'i': 'i',
                '1': 'g', '2': 'r', '3': 'i'}

    # 保留 hjd 列供周期分析使用
    keep_cols_base = ['mjd', 'mag', 'magerr']
    if 'hjd' in df.columns:
        keep_cols_base = ['mjd', 'hjd', 'mag', 'magerr']

    if 'band' in df.columns:
        for raw_band in df['band'].unique():
            b = band_map.get(str(raw_band).strip(), str(raw_band).strip())
            keep_cols = [c for c in keep_cols_base if c in df.columns]
            sub = df[df['band'] == raw_band][keep_cols].copy()
            sub = sub.dropna(subset=['mjd', 'mag', 'magerr']).sort_values('mjd').reset_index(drop=True)
            if len(sub) > 0:
                result[b] = sub
    else:
        # 所有数据混在一起
        keep_cols = [c for c in keep_cols_base if c in df.columns]
        result['all'] = df[keep_cols].dropna(subset=['mjd', 'mag', 'magerr']).sort_values('mjd')

    # 观测时间范围摘要
    all_mjd = df['mjd'].dropna()
    if len(all_mjd) > 0:
        result['obs_mjd_min'] = float(all_mjd.min())
        result['obs_mjd_max'] = float(all_mjd.max())
        result['n_epochs'] = int(len(all_mjd))

    return result


def plot_lightcurve(result, save_path=None):
    """绘制 ZTF 光变曲线 (g/r/i 多波段)"""
    if result is None:
        return None

    fig, ax = utils.setup_lightcurve_plot()
    colors = {'g': 'green', 'r': 'red', 'i': 'gold', 'all': 'black'}
    markers = {'g': 'o', 'r': 's', 'i': '^', 'all': '.'}

    for band in ('g', 'r', 'i', 'all'):
        if band not in result:
            continue
        df = result[band]
        ax.errorbar(df['mjd'], df['mag'], yerr=df['magerr'],
                    fmt=markers.get(band, '.'), color=colors.get(band, 'k'),
                    markersize=2, elinewidth=0.5, alpha=0.7,
                    label=f'ZTF {band} ({len(df)} pts)')

    ax.set_title(f"{result.get('survey', 'ZTF')} Light Curve  RA={result['ra']:.4f} DEC={result['dec']:.4f}")
    ax.legend()
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_csv(result, output_dir):
    """保存 ZTF 光变曲线为 CSV"""
    df = utils.lightcurve_to_dataframe(result, ['g', 'r', 'i', 'all'])
    return utils.write_csv(df, output_dir, 'ztf_lightcurve.csv')
