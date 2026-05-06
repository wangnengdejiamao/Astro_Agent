"""
JWST (James Webb Space Telescope) 光谱与光变曲线
=================================================
通过 MAST (Mikulski Archive for Space Telescopes) 查询
光谱仪器: NIRSpec, MIRI (MRS/LRS), NIRISS (WFSS/SOSS)
测光: 多历元成像观测

用法:
    from astro_toolbox.jwst import query_spectrum, query_lightcurve
    spec = query_spectrum(190.305, 2.596)
    lc = query_lightcurve(190.305, 2.596)
"""
import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from . import config, utils


# JWST 光谱仪器关键字 (用于本地过滤; MAST 仪器名格式不统一)
JWST_SPEC_INSTRUMENT_KEYWORDS = [
    'NIRSPEC', 'NIRISS', 'MIRI',
]

# JWST 成像仪器关键字
JWST_IMAGE_INSTRUMENT_KEYWORDS = [
    'NIRCAM', 'MIRI',
]

# 光谱产品优先级
_SPEC_PRODUCT_PRIORITY = ['X1DINTS', 'X1D', 'S2D', 'S3D']

# MAST 中 JWST 光谱数据的 dataproduct_type 可能是 'spectrum' 或 'timeseries'
# (NIRSpec/NIRISS SOSS 的时间序列光谱常被标记为 timeseries)
_SPEC_DATAPRODUCT_TYPES = ['spectrum', 'timeseries']


def _setup_mast_proxy():
    """为 astroquery.mast 配置代理 (仅在代理可用时)"""
    if utils._should_use_proxy('https://mast.stsci.edu'):
        import os as _os
        proxy = config.PROXY_URL
        if proxy:
            _os.environ.setdefault('HTTP_PROXY', proxy)
            _os.environ.setdefault('HTTPS_PROXY', proxy)


def _is_spec_instrument(instrument_name):
    """判断仪器名是否为 JWST 光谱仪 (NIRSpec/NIRISS SOSS/MIRI spectroscopy)
    排除纯成像模式 (NIRCAM/IMAGE, MIRI/IMAGE)"""
    name = str(instrument_name).upper()
    # 纯成像模式排除
    if 'IMAGE' in name and 'NIRSPEC' not in name:
        return False
    # NIRCAM 仅做成像
    if 'NIRCAM' in name:
        return False
    return any(kw in name for kw in JWST_SPEC_INSTRUMENT_KEYWORDS)


def _is_image_instrument(instrument_name):
    """判断仪器名是否为 JWST 成像仪 (NIRCam/MIRI)"""
    name = str(instrument_name).upper()
    return any(kw in name for kw in JWST_IMAGE_INSTRUMENT_KEYWORDS)


def query_spectrum(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 JWST 光谱 (NIRSpec / MIRI / NIRISS)。

    通过 astroquery.mast 搜索 MAST，下载 x1d FITS 产品，
    提取波长/流量/误差。

    Returns:
        dict 或 None
    """
    from astroquery.mast import Observations

    _setup_mast_proxy()
    coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

    # 搜索 JWST 光谱观测 (含 'spectrum' 和 'timeseries' 两种类型,
    # 不在服务端过滤 instrument_name, 改用本地过滤)
    try:
        obs = Observations.query_criteria(
            coordinates=coord,
            radius=radius_arcsec * u.arcsec,
            obs_collection='JWST',
            dataproduct_type=_SPEC_DATAPRODUCT_TYPES,
            intentType='science',
            calib_level=[2, 3],
        )
    except Exception as e:
        print(f"JWST spectrum MAST 查询失败: {e}")
        return None

    if obs is None or len(obs) == 0:
        return None

    # 本地过滤: 只保留光谱仪器
    mask = [_is_spec_instrument(row['instrument_name']) for row in obs]
    obs = obs[mask]

    if len(obs) == 0:
        return None

    # 优先高 calibration level, 再优先 NIRSPEC > NIRISS > MIRI
    _INSTR_PRIORITY = {'NIRSPEC': 3, 'NIRISS': 2, 'MIRI': 1}

    def _sort_key(row):
        name = str(row['instrument_name']).upper()
        instr_score = 0
        for kw, score in _INSTR_PRIORITY.items():
            if kw in name:
                instr_score = score
                break
        return (int(row['calib_level']), instr_score)

    sorted_indices = sorted(range(len(obs)), key=lambda i: _sort_key(obs[i]),
                            reverse=True)

    cache_dir = os.path.join(config.CACHE_DIR, 'jwst')
    utils.ensure_dir(cache_dir)

    for idx in sorted_indices:
        result = _try_download_spectrum(obs[idx], cache_dir, ra, dec)
        if result is not None:
            return result

    # 全部下载失败: 把找到的观测元数据打印出来 (常见于 proprietary / 服务器错误)
    print(f"  [JWST] 找到 {len(obs)} 条光谱观测但下载/解析全部失败:")
    for row in obs[:5]:
        pid = row['proposal_id'] if 'proposal_id' in obs.colnames else '?'
        pi = row['proposal_pi'] if 'proposal_pi' in obs.colnames else ''
        instr = row['instrument_name'] if 'instrument_name' in obs.colnames else ''
        oid = row['obs_id'] if 'obs_id' in obs.colnames else ''
        title = row['obs_title'] if 'obs_title' in obs.colnames else ''
        print(f"    PID {pid}  {instr}  obs_id={oid}  PI={pi}")
        if title:
            print(f"      {title}")
    if len(obs) > 5:
        print(f"    ... 共 {len(obs)} 条")
    print("  提示: ERROR 通常意味着数据仍在 proprietary period 或 MAST 暂时不可用。"
          "可访问 https://mast.stsci.edu/ 查询 proposal 详情和释放日期。")
    return None


def _try_download_spectrum(obs_row, cache_dir, ra, dec):
    """尝试从单条观测记录下载并解析 JWST 光谱"""
    from astroquery.mast import Observations

    obs_id = str(obs_row.get('obs_id', '?'))
    try:
        products = Observations.get_product_list(obs_row)
        if products is None or len(products) == 0:
            return None

        # 按优先级筛选
        filtered = None
        for ptype in _SPEC_PRODUCT_PRIORITY:
            filtered = Observations.filter_products(
                products,
                productSubGroupDescription=ptype,
                extension='fits',
            )
            if filtered is not None and len(filtered) > 0:
                break

        if filtered is None or len(filtered) == 0:
            # 尝试更宽泛的筛选
            filtered = Observations.filter_products(
                products,
                extension='fits',
                productType='SCIENCE',
            )
            if filtered is None or len(filtered) == 0:
                return None

        manifest = Observations.download_products(
            filtered[:1],
            download_dir=cache_dir,
            cache=True,
        )
        if manifest is None or len(manifest) == 0:
            return None

        filepath = str(manifest['Local Path'][0])
        # 检查下载状态
        if 'Status' in manifest.colnames:
            status = str(manifest['Status'][0]).upper()
            if 'ERROR' in status:
                print(f"JWST 下载错误 ({obs_id}): {status}")
                return None

        if not os.path.exists(filepath):
            return None

        return _parse_spectrum(filepath, obs_row, ra, dec)

    except Exception as e:
        print(f"JWST spectrum 下载/解析失败 ({obs_id}): {e}")
        return None


def _parse_spectrum(filepath, obs_row, ra, dec):
    """解析 JWST x1d FITS 光谱文件"""
    with fits.open(filepath) as hdul:
        header = hdul[0].header

        # 查找包含光谱数据的 extension
        data = None
        for ext in hdul[1:]:
            if ext.data is not None and hasattr(ext, 'columns'):
                names = [n.upper() for n in ext.columns.names]
                if 'WAVELENGTH' in names and 'FLUX' in names:
                    data = ext.data
                    break

        if data is None:
            return None

        all_wave = []
        all_flux = []
        all_err = []

        # 可能是单行或多行 (多 integration)
        if data.ndim == 0 or (hasattr(data, 'shape') and len(data.shape) == 0):
            return None

        rows = data if len(data.shape) > 0 and data.shape[0] > 0 else [data]

        for row in rows:
            w = np.asarray(row['WAVELENGTH'], dtype=float)
            f = np.asarray(row['FLUX'], dtype=float)

            # JWST 波长单位: micron → 转换为 Angstrom
            if len(w[w > 0]) > 0 and np.nanmedian(w[w > 0]) < 100:
                w = w * 1e4  # micron → Angstrom

            # 误差
            err_names = [n for n in data.names if n.upper() in
                         ('FLUX_ERROR', 'ERROR', 'ERR', 'FLUX_ERR')]
            if err_names:
                e = np.asarray(row[err_names[0]], dtype=float)
            else:
                e = np.zeros_like(f)

            mask = (w > 0) & np.isfinite(f) & np.isfinite(e)
            all_wave.append(w[mask])
            all_flux.append(f[mask])
            all_err.append(e[mask])

        if not all_wave:
            return None

        wavelength = np.concatenate(all_wave)
        flux = np.concatenate(all_flux)
        error = np.concatenate(all_err)

        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        flux = flux[sort_idx]
        error = error[sort_idx]

        if len(wavelength) == 0:
            return None

        prov = utils.build_provenance('JWST', obs_row=obs_row, header=header,
                                      ra=ra, dec=dec)

        return {
            'survey': 'JWST',
            'ra': ra, 'dec': dec,
            'instrument': prov['instrument'],
            'detector': prov['detector'],
            'grating': prov['grating'] or prov['filter'],
            'wavelength': wavelength,
            'flux': flux,
            'error': error,
            'obs_mjd': prov['obs_mjd'] if prov['obs_mjd'] is not None else 0.0,
            'obs_id': prov['obs_id'],
            'proposal_id': prov['proposal_id'],
            'proposal_pi': prov['proposal_pi'],
            'title': prov['title'],
            'obs_date_utc': prov['obs_date_utc'],
            'exptime_s': prov['exptime_s'],
            'provenance': prov,
        }


# ================================================================
#  JWST 多历元测光 (光变曲线)
# ================================================================

def query_lightcurve(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 JWST 多历元测光数据。

    从 MAST 观测元数据中提取同一目标的多次成像观测,
    按滤光片分组。同时搜索 image 和 timeseries 类型。

    Returns:
        dict 或 None
    """
    from astroquery.mast import Observations

    _setup_mast_proxy()
    coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

    try:
        obs = Observations.query_criteria(
            coordinates=coord,
            radius=radius_arcsec * u.arcsec,
            obs_collection='JWST',
            dataproduct_type=['image', 'timeseries'],
            intentType='science',
            calib_level=[2, 3],
        )
    except Exception as e:
        print(f"JWST lightcurve MAST 查询失败: {e}")
        return None

    if obs is None or len(obs) < 2:
        return None

    # 本地过滤: 只保留成像仪器
    mask = [_is_image_instrument(row['instrument_name']) for row in obs]
    obs = obs[mask]

    if len(obs) < 2:
        return None

    df = obs.to_pandas()

    if 't_min' not in df.columns or 'filters' not in df.columns:
        return None

    df['t_min'] = pd.to_numeric(df['t_min'], errors='coerce')
    df = df.dropna(subset=['t_min', 'filters'])

    if len(df) < 2:
        return None

    # 按滤光片分组 (并保留 per-row proposal_id, proposal_pi, obs_id, instrument)
    filters = {}
    for filt_name, group in df.groupby('filters'):
        filt_name = str(filt_name).strip().split(';')[0]
        if len(group) < 2:
            continue

        band_data = {
            'mjd': group['t_min'].values,
            'mag': np.full(len(group), np.nan),
            'magerr': np.full(len(group), np.nan),
        }
        for col, key in [('proposal_id', 'proposal_id'),
                         ('proposal_pi', 'proposal_pi'),
                         ('obs_id', 'obs_id'),
                         ('instrument_name', 'instrument')]:
            if col in group.columns:
                band_data[key] = group[col].astype(str).values

        # 检查是否有 TSO (Time Series Observation) 数据
        if 'obs_title' in group.columns:
            is_tso = group['obs_title'].str.contains('TSO|time.series',
                                                      case=False, na=False)
            if is_tso.any():
                band_data['is_tso'] = is_tso.values

        filters[filt_name] = pd.DataFrame(band_data).sort_values('mjd').reset_index(drop=True)

    if not filters:
        return None

    all_mjds = df['t_min'].dropna()
    n_total = sum(len(f) for f in filters.values())

    rep_row = df.iloc[0]
    prov = utils.build_provenance('JWST', obs_row=rep_row,
                                  ra=ra, dec=dec,
                                  override={'obs_mjd': float(all_mjds.min())})
    if 'proposal_id' in df.columns:
        unique_pids = sorted({str(p).strip() for p in df['proposal_id']
                              if str(p).strip()
                              and str(p).strip().lower() != 'nan'})
        prov['n_proposals'] = len(unique_pids)
        prov['proposal_ids'] = unique_pids

    return {
        'survey': 'JWST',
        'ra': ra, 'dec': dec,
        'filters': filters,
        'n_epochs': n_total,
        'obs_mjd_min': float(all_mjds.min()),
        'obs_mjd_max': float(all_mjds.max()),
        'time_system': 'MJD',
        'provenance': prov,
        'source': 'MAST observations',
    }


# ================================================================
#  绘图
# ================================================================

def plot_spectrum(result, save_path=None):
    """绘制 JWST 光谱"""
    if result is None:
        return None
    import matplotlib.pyplot as plt

    fig, ax = utils.setup_spectrum_plot()
    ax.plot(result['wavelength'], result['flux'], 'k-', lw=0.6, label='Flux')
    ax.fill_between(result['wavelength'],
                    result['flux'] - result['error'],
                    result['flux'] + result['error'],
                    color='gray', alpha=0.2)

    grating = result.get('grating', '')
    grating_str = f"  {grating}" if grating else ''
    prov = result.get('provenance', {})
    title_lines = [f"JWST {result['instrument']}{grating_str} Spectrum"]
    if prov.get('proposal_id'):
        pid_line = f"PID {prov['proposal_id']}"
        if prov.get('proposal_pi'):
            pid_line += f"  PI: {prov['proposal_pi']}"
        title_lines.append(pid_line)
    obs_meta = []
    if prov.get('obs_date_utc'):
        obs_meta.append(prov['obs_date_utc'][:19])
    obs_meta.append(f"obs_id={result['obs_id']}")
    obs_meta.append(f"MJD={result.get('obs_mjd', 0):.1f}")
    title_lines.append('  '.join(obs_meta))
    ax.set_title('\n'.join(title_lines), fontsize=10)

    # 根据波长范围调整标签
    wmin, wmax = result['wavelength'].min(), result['wavelength'].max()
    if wmax > 30000:
        ax.set_xlabel('Wavelength (A)  [MIR]')
    elif wmin > 8000:
        ax.set_xlabel('Wavelength (A)  [NIR]')
    else:
        ax.set_xlabel('Wavelength (A)')

    ax.set_ylabel('Flux')
    ax.legend()

    # 轴范围紧凑到光谱数据
    utils.set_spectrum_axes(ax, result['wavelength'], result['flux'])

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def plot_lightcurve(result, save_path=None):
    """绘制 JWST 多滤光片光变曲线"""
    if result is None or not result.get('filters'):
        return None
    import matplotlib.pyplot as plt

    filters = result['filters']
    n = len(filters)
    if n == 0:
        return None

    fig, axes = plt.subplots(max(n, 1), 1,
                              figsize=(12, max(2.5 * n, 4)),
                              sharex=True, squeeze=False)
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    for ax, (filt_name, df), color in zip(axes, filters.items(), colors):
        valid = np.isfinite(df['mag'])
        if valid.any():
            ax.errorbar(df['mjd'][valid], df['mag'][valid],
                        yerr=df['magerr'][valid] if np.isfinite(df['magerr']).any() else None,
                        fmt='.', color=color, ms=4, elinewidth=0.5, alpha=0.7)
            ax.invert_yaxis()
        else:
            for mjd in df['mjd']:
                ax.axvline(mjd, color=color, alpha=0.3, lw=0.5)
        ax.set_ylabel(filt_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.95, f'N={len(df)}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='gray')

    axes[-1].set_xlabel('MJD')
    fig.suptitle(f"JWST Light Curve  RA={result['ra']:.4f} DEC={result['dec']:.4f}  "
                 f"{n} filters, {result.get('n_epochs', 0)} epochs",
                 fontsize=12)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_spectrum_csv(result, output_dir):
    """保存 JWST 光谱为 CSV (附带 provenance 列 + 边车 JSON)。"""
    df = utils.spectrum_to_dataframe(result)
    prov = (result or {}).get('provenance')
    df = utils.add_provenance_columns(df, prov)
    csv_path = utils.write_csv(df, output_dir, 'jwst_spectrum.csv')
    if prov:
        utils.write_provenance_json(prov, output_dir, 'jwst_spectrum_provenance.json')
    return csv_path


def save_lightcurve_csv(result, output_dir):
    """保存 JWST 光变曲线为 CSV (附带 provenance 列 + 边车 JSON)。"""
    import pandas as pd
    if result is None or not result.get('filters'):
        return None
    all_dfs = []
    for filt_name, df in result['filters'].items():
        df_out = df.copy()
        df_out['filter'] = filt_name
        all_dfs.append(df_out)
    if not all_dfs:
        return None
    combined = pd.concat(all_dfs, ignore_index=True)
    prov = result.get('provenance')
    fill_cols = ['mission']
    if 'proposal_id' not in combined.columns and prov and prov.get('proposal_id'):
        fill_cols.append('proposal_id')
    combined = utils.add_provenance_columns(combined, prov, columns=fill_cols)
    csv_path = utils.write_csv(combined, output_dir, 'jwst_lightcurve.csv')
    if prov:
        utils.write_provenance_json(prov, output_dir, 'jwst_lightcurve_provenance.json')
    return csv_path
