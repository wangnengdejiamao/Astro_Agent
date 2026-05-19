"""
HST (Hubble Space Telescope) 光谱与光变曲线
=============================================
通过 MAST (Mikulski Archive for Space Telescopes) 查询
光谱仪器: COS, STIS, FOS
测光: HST Source Catalog (HSC) 多历元测光

用法:
    from astro_toolbox.hst import query_spectrum, query_lightcurve
    spec = query_spectrum(190.305, 2.596)
    lc = query_lightcurve(190.305, 2.596)
"""
import os
import json
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from . import config, utils


# HST 光谱仪器关键字 (用于本地过滤; MAST 仪器名格式不统一,
# 可能是 'COS/FUV', 'COS', 'STIS/CCD', 'STIS', 'COS-STIS' 等)
HST_SPEC_INSTRUMENT_KEYWORDS = [
    'COS', 'STIS', 'FOS',
]

# 光谱产品优先级 (高→低)
_SPEC_PRODUCT_PRIORITY = ['X1DSUM', 'X1D', 'SX1']


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _setup_mast_proxy():
    """为 astroquery.mast 配置代理 (如果 mast.stsci.edu 需要代理且代理可用)"""
    if utils._should_use_proxy('https://mast.stsci.edu'):
        import os as _os
        proxy = config.PROXY_URL
        if proxy:
            _os.environ.setdefault('HTTP_PROXY', proxy)
            _os.environ.setdefault('HTTPS_PROXY', proxy)


def _is_spec_instrument(instrument_name):
    """判断仪器名是否为 HST 光谱仪 (COS/STIS/FOS)"""
    name = str(instrument_name).upper()
    return any(kw in name for kw in HST_SPEC_INSTRUMENT_KEYWORDS)


def query_spectrum(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 HST 光谱 (COS / STIS / FOS)。

    通过 astroquery.mast 搜索 MAST，下载最佳 x1d FITS 产品，
    提取波长/流量/误差。

    Returns:
        dict 或 None
    """
    from astroquery.mast import Observations

    _setup_mast_proxy()
    coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

    # 搜索 HST 光谱观测 (不在服务端过滤 instrument_name,
    # 因为 MAST 仪器名格式不统一, 改用本地过滤)
    try:
        obs = Observations.query_criteria(
            coordinates=coord,
            radius=radius_arcsec * u.arcsec,
            obs_collection='HST',
            dataproduct_type='spectrum',
            intentType='science',
        )
    except Exception as e:
        print(f"HST spectrum MAST 查询失败: {e}")
        return None

    if obs is None or len(obs) == 0:
        return None

    # 本地过滤: 只保留光谱仪器 (COS/STIS/FOS)
    mask = [_is_spec_instrument(row['instrument_name']) for row in obs]
    obs = obs[mask]

    if len(obs) == 0:
        return None

    # 按 calibration level 降序, 优先选高级别产品
    obs.sort('calib_level')
    obs.reverse()

    # 尝试下载光谱
    cache_dir = os.path.join(config.CACHE_DIR, 'hst')
    utils.ensure_dir(cache_dir)

    for row in obs:
        result = _try_download_spectrum(row, cache_dir, ra, dec)
        if result is not None:
            return result

    # 全部下载失败: 打印观测元数据以便用户排查
    print(f"  [HST] 找到 {len(obs)} 条光谱观测但下载/解析全部失败:")
    for row in obs[:5]:
        pid = row['proposal_id'] if 'proposal_id' in obs.colnames else '?'
        pi = row['proposal_pi'] if 'proposal_pi' in obs.colnames else ''
        instr = row['instrument_name'] if 'instrument_name' in obs.colnames else ''
        oid = row['obs_id'] if 'obs_id' in obs.colnames else ''
        print(f"    PID {pid}  {instr}  obs_id={oid}  PI={pi}")
    if len(obs) > 5:
        print(f"    ... 共 {len(obs)} 条")
    return None


def _try_download_spectrum(obs_row, cache_dir, ra, dec):
    """尝试从单条观测记录下载并解析光谱"""
    from astroquery.mast import Observations

    obs_id = str(utils._row_get(obs_row, 'obs_id', '?'))
    try:
        products = Observations.get_product_list(obs_row)
        if products is None or len(products) == 0:
            return None

        # 按优先级筛选光谱产品
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
            return None

        # 下载第一个匹配的产品
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
                print(f"HST 下载错误 ({obs_id}): {status}")
                return None

        if not os.path.exists(filepath):
            return None

        return _parse_x1d(filepath, obs_row, ra, dec)

    except Exception as e:
        print(f"HST spectrum 下载/解析失败 ({obs_id}): {e}")
        return None


def _parse_x1d(filepath, obs_row, ra, dec):
    """解析 HST x1d/x1dsum FITS 文件"""
    with fits.open(filepath) as hdul:
        if len(hdul) < 2 or hdul[1].data is None:
            return None

        data = hdul[1].data
        header = hdul[0].header

        if len(data) == 0:
            return None

        all_wave = []
        all_flux = []
        all_err = []

        for row in data:
            w = np.asarray(row['WAVELENGTH'], dtype=float)
            f = np.asarray(row['FLUX'], dtype=float)
            # 误差列名可能是 ERROR 或 ERROR_LOWER
            if 'ERROR' in data.names:
                e = np.asarray(row['ERROR'], dtype=float)
            elif 'ERROR_LOWER' in data.names:
                e = np.asarray(row['ERROR_LOWER'], dtype=float)
            else:
                e = np.zeros_like(f)

            # 过滤有效数据 (波长 > 0, 流量有限, 排除 flux=0 且 error=0 的零填充)
            mask = (w > 0) & np.isfinite(f) & np.isfinite(e)
            mask &= ~((f == 0) & (e == 0))  # 排除零填充行
            all_wave.append(w[mask])
            all_flux.append(f[mask])
            all_err.append(e[mask])

        if not all_wave:
            return None

        wavelength = np.concatenate(all_wave)
        flux = np.concatenate(all_flux)
        error = np.concatenate(all_err)

        # 按波长排序
        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        flux = flux[sort_idx]
        error = error[sort_idx]

        if len(wavelength) == 0:
            return None

        # 对低 SNR 区域进行平滑过滤: 滑动窗口 SNR < 1 的连续区域标记为噪声
        # 但保留原始数据, 在 'snr_mask' 中标记高质量区域供绘图参考
        if len(error) > 0 and np.any(error > 0):
            with np.errstate(divide='ignore', invalid='ignore'):
                snr = np.where(error > 0, np.abs(flux) / error, 0)
            # 用 50 像素滑动窗口计算平均 SNR
            kernel = min(50, len(snr) // 4)
            if kernel > 1:
                snr_smooth = np.convolve(snr, np.ones(kernel)/kernel,
                                         mode='same')
            else:
                snr_smooth = snr

        # 统一构造 provenance (使用 MAST 行 + FITS 主头)
        prov = utils.build_provenance('HST', obs_row=obs_row, header=header,
                                      ra=ra, dec=dec)

        result = {
            'survey': 'HST',
            'ra': ra, 'dec': dec,
            'instrument': prov['instrument'],
            'detector': prov['detector'],
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
        try:
            from .diagnostics import analyze_spectrum
            diag = analyze_spectrum(
                wavelength,
                flux,
                error,
                survey='HST',
                metadata=result,
            )
            result['diagnostics'] = diag
            result['spectral_region'] = diag.get('spectral_region')
            result['usable_for_optical_rv'] = diag.get('usable_for_optical_rv')
            result['usable_for_wd_balmer_fit'] = diag.get('usable_for_wd_balmer_fit')
            result['analyzable_lines'] = diag.get('analyzable_lines', [])
            result['strong_emission_lines'] = diag.get('strong_emission_lines', [])
            result['strong_absorption_lines'] = diag.get('strong_absorption_lines', [])
        except Exception:
            result['spectral_region'] = ''
            result['usable_for_optical_rv'] = False
            result['usable_for_wd_balmer_fit'] = False
        return result


# ================================================================
#  HST 多历元测光 (光变曲线)
# ================================================================

def query_lightcurve(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 HST 多历元测光数据。

    方法1: HST Source Catalog (HSC) — 提供跨 visit 的测光
    方法2: 从 MAST 观测元数据提取多历元信息

    Returns:
        dict: {'filters': {filter_name: DataFrame(mjd, mag, magerr)}, ...}
        或 None
    """
    _setup_mast_proxy()

    # 方法1: HSC (首选, 有标准测光)
    result = _query_hsc(ra, dec, radius_arcsec)
    if result is not None:
        return result

    # 方法2: 从观测元数据构建光变曲线
    result = _query_obs_lightcurve(ra, dec, radius_arcsec)
    return result


def _query_hsc(ra, dec, radius_arcsec):
    """通过 HST Source Catalog v3 查询多历元测光"""
    try:
        from astroquery.mast import Catalogs

        coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

        # HSC v3 详细测光 (per-visit measurements)
        hsc = Catalogs.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog='HSC',
            version=3,
            magtype='magaper2',
        )

        if hsc is None or len(hsc) == 0:
            return None

        df = hsc.to_pandas()

        # HSC 表关键列: MagAper2, MagErrAper2, StartMJD, Filter, Detector,
        # ProposalID (per-visit). 列名可能因版本不同而异, 做兼容处理
        mjd_col = _find_col(df, ['StartMJD', 'startmjd', 'MJD', 'mjd'])
        mag_col = _find_col(df, ['MagAper2', 'magaper2', 'Mag', 'mag'])
        err_col = _find_col(df, ['MagErrAper2', 'magerraper2', 'MagErr', 'magerr'])
        filt_col = _find_col(df, ['Filter', 'filter', 'FILTER'])
        prop_col = _find_col(df, ['ProposalID', 'proposalid', 'ProposalId',
                                  'PROPOSAL', 'proposal_id'])
        det_col = _find_col(df, ['Detector', 'detector'])
        instr_col = _find_col(df, ['Instrument', 'instrument'])

        if mjd_col is None or mag_col is None or filt_col is None:
            return None

        # 过滤有效数据
        df = df.dropna(subset=[mjd_col, mag_col, filt_col])
        df[mjd_col] = pd.to_numeric(df[mjd_col], errors='coerce')
        df[mag_col] = pd.to_numeric(df[mag_col], errors='coerce')
        if err_col:
            df[err_col] = pd.to_numeric(df[err_col], errors='coerce')
        df = df.dropna(subset=[mjd_col, mag_col])

        if len(df) == 0:
            return None

        # 按滤光片分组 (并保留 per-visit proposal_id, instrument, detector)
        filters = {}
        for filt_name, group in df.groupby(filt_col):
            filt_name = str(filt_name).strip()
            if len(group) < 2:
                continue
            band_data = {
                'mjd': group[mjd_col].values,
                'mag': group[mag_col].values,
                'magerr': group[err_col].values if err_col else 0.01,
            }
            if prop_col:
                band_data['proposal_id'] = group[prop_col].astype(str).values
            if instr_col:
                band_data['instrument'] = group[instr_col].astype(str).values
            if det_col:
                band_data['detector'] = group[det_col].astype(str).values
            band_df = pd.DataFrame(band_data).sort_values('mjd').reset_index(drop=True)
            filters[filt_name] = band_df

        if not filters:
            return None

        all_mjds = df[mjd_col].dropna()
        n_total = sum(len(f) for f in filters.values())

        # 顶层 provenance: 用最近一次观测的代表行 (作为光变曲线的总体出处)
        # 关键字段 (proposal_pi, title) HSC 表里没有, 这里只填 mission/instrument
        # 多 proposal 信息体现在 per-row 列上
        rep_instr = (df[instr_col].iloc[0] if instr_col else 'HST') if len(df) else 'HST'
        prov = utils.build_provenance(
            'HST', ra=ra, dec=dec,
            override={'instrument': rep_instr,
                      'obs_mjd': float(all_mjds.min())})
        # 如果整段 lightcurve 来自同一 proposal, 把它写进顶层 provenance
        if prop_col:
            unique_pids = sorted({str(p).strip() for p in df[prop_col]
                                  if str(p).strip()
                                  and str(p).strip().lower() != 'nan'})
            if len(unique_pids) == 1:
                try:
                    prov['proposal_id'] = int(unique_pids[0])
                except (ValueError, TypeError):
                    pass
            prov['n_proposals'] = len(unique_pids)
            prov['proposal_ids'] = unique_pids

        return {
            'survey': 'HST',
            'ra': ra, 'dec': dec,
            'filters': filters,
            'n_epochs': n_total,
            'obs_mjd_min': float(all_mjds.min()),
            'obs_mjd_max': float(all_mjds.max()),
            'time_system': 'MJD',
            'provenance': prov,
            'source': 'HSC v3',
        }

    except Exception as e:
        print(f"HSC 查询失败: {e}")
        return None


def _query_obs_lightcurve(ra, dec, radius_arcsec):
    """
    备用方案: 从 MAST 观测元数据构建 HST 光变曲线。
    利用 t_min (MJD) 和 proposal 级别元数据。
    """
    try:
        from astroquery.mast import Observations

        coord = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')

        obs = Observations.query_criteria(
            coordinates=coord,
            radius=radius_arcsec * u.arcsec,
            obs_collection='HST',
            dataproduct_type='image',
            intentType='science',
        )

        if obs is None or len(obs) < 2:
            return None

        df = obs.to_pandas()

        # 需要的列: t_min (MJD), filters, instrument_name
        if 't_min' not in df.columns or 'filters' not in df.columns:
            return None

        df['t_min'] = pd.to_numeric(df['t_min'], errors='coerce')
        df = df.dropna(subset=['t_min', 'filters'])

        if len(df) < 2:
            return None

        # 这种方法没有直接的星等, 只能返回观测时间点
        # 对于没有 HSC 的目标, 仅用于标记哪些历元有观测
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
            filters[filt_name] = pd.DataFrame(band_data).sort_values('mjd').reset_index(drop=True)

        if not filters:
            return None

        all_mjds = df['t_min'].dropna()
        n_total = sum(len(f) for f in filters.values())

        # 顶层 provenance — 多 proposal 时用第一行作为代表; 仪器从最常见取
        rep_row = df.iloc[0]
        prov = utils.build_provenance('HST', obs_row=rep_row,
                                      ra=ra, dec=dec,
                                      override={'obs_mjd': float(all_mjds.min())})
        if 'proposal_id' in df.columns:
            unique_pids = sorted({str(p).strip() for p in df['proposal_id']
                                  if str(p).strip()
                                  and str(p).strip().lower() != 'nan'})
            prov['n_proposals'] = len(unique_pids)
            prov['proposal_ids'] = unique_pids

        return {
            'survey': 'HST',
            'ra': ra, 'dec': dec,
            'filters': filters,
            'n_epochs': n_total,
            'obs_mjd_min': float(all_mjds.min()),
            'obs_mjd_max': float(all_mjds.max()),
            'time_system': 'MJD',
            'note': 'obs_metadata_only',
            'provenance': prov,
            'source': 'MAST observations',
        }

    except Exception as e:
        print(f"HST obs lightcurve 查询失败: {e}")
        return None


def _find_col(df, candidates):
    """在 DataFrame 中查找第一个匹配的列名"""
    for col in candidates:
        if col in df.columns:
            return col
        # 大小写不敏感
        for c in df.columns:
            if c.lower() == col.lower():
                return c
    return None


# ================================================================
#  绘图
# ================================================================

def _median_bin_spectrum(wave, flux, err=None, bin_size=7):
    """Median-bin a 1D spectrum for display without changing the saved data."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float) if err is not None else None
    good = np.isfinite(wave) & np.isfinite(flux)
    if err is not None and len(err) == len(wave):
        good &= np.isfinite(err)
    wave = wave[good]
    flux = flux[good]
    err = err[good] if err is not None and len(err) == len(good) else None
    if len(wave) < bin_size:
        return wave, flux, err
    n = len(wave) // bin_size
    trim = n * bin_size
    wb = np.nanmedian(wave[:trim].reshape(n, bin_size), axis=1)
    fb = np.nanmedian(flux[:trim].reshape(n, bin_size), axis=1)
    eb = None
    if err is not None:
        eb = np.nanmedian(err[:trim].reshape(n, bin_size), axis=1)
    return wb, fb, eb


def _robust_spectrum_ylim(flux, err=None, scale=1e-15, lower=1.0, upper=99.0):
    """Percentile y limits so isolated bad pixels do not dominate the plot."""
    f = np.asarray(flux, dtype=float) / scale
    good = np.isfinite(f)
    if err is not None:
        e = np.asarray(err, dtype=float) / scale
        if len(e) == len(f):
            e_good = e[np.isfinite(e) & (e > 0)]
            if len(e_good) >= 10:
                err_cut = max(np.nanmedian(e_good) * 5.0,
                              np.nanpercentile(e_good, 75))
                good &= np.isfinite(e) & (e > 0) & (e <= err_cut)
    f = f[good]
    if len(f) < 10:
        return None
    lo, hi = np.nanpercentile(f, [lower, upper])
    if not np.isfinite(lo + hi) or hi <= lo:
        return None
    pad = max((hi - lo) * 0.18, 3.0)
    return lo - pad, hi + pad


def _display_quality_mask(flux, err=None, scale=1e-15):
    """Mask very noisy HST bins for plotting while keeping them in CSV files."""
    f = np.asarray(flux, dtype=float)
    good = np.isfinite(f)
    if err is None:
        return good
    e = np.asarray(err, dtype=float) / scale
    if len(e) != len(f):
        return good
    e_good = e[np.isfinite(e) & (e > 0)]
    if len(e_good) < 10:
        return good
    err_cut = max(np.nanmedian(e_good) * 5.0, np.nanpercentile(e_good, 75))
    good &= np.isfinite(e) & (e > 0) & (e <= err_cut)
    return good


def _annotate_uv_lines(ax, wave, diagnostics=None, scale=1e-15):
    """Mark common UV transitions covered by an HST/COS spectrum."""
    try:
        from .diagnostics import UV_LINES
    except Exception:
        return
    wmin, wmax = np.nanmin(wave), np.nanmax(wave)
    line_meas = (diagnostics or {}).get('line_measurements', {})
    y0, y1 = ax.get_ylim()
    label_top = max(y0, y1)
    label_bottom = min(y0, y1)
    span = label_top - label_bottom
    if not np.isfinite(span) or span <= 0:
        return
    for idx, (name, rest, _family) in enumerate(UV_LINES):
        if rest < wmin or rest > wmax:
            continue
        meas = line_meas.get(name, {})
        em_snr = meas.get('emission_snr', np.nan)
        abs_snr = meas.get('absorption_snr', np.nan)
        candidate = ((np.isfinite(em_snr) and em_snr >= 3.0)
                     or (np.isfinite(abs_snr) and abs_snr >= 3.0))
        color = '#b2182b' if candidate else '#2166ac'
        alpha = 0.72 if candidate else 0.42
        ax.axvline(rest, color=color, lw=0.8, ls='--', alpha=alpha, zorder=1)
        y_text = label_top - span * (0.05 + 0.09 * (idx % 3))
        ax.text(rest, y_text, name.replace(' ', '\n', 1),
                rotation=90, ha='center', va='top', fontsize=7,
                color=color, alpha=0.9 if candidate else 0.65)


def plot_spectrum(result, save_path=None, log_flux=True,
                  robust_ylim=True, annotate_lines=True,
                  bin_size=7, figsize=(18, 6)):
    """绘制 HST 光谱: 宽幅、鲁棒 y 轴、UV 谱线标注。"""
    if result is None:
        return None
    import matplotlib.pyplot as plt

    wave = np.asarray(result['wavelength'], dtype=float)
    flux = np.asarray(result['flux'], dtype=float)
    err = np.asarray(result.get('error', np.zeros_like(flux)), dtype=float)
    scale = 1e-15
    fig, ax = plt.subplots(figsize=figsize)
    wb, fb, eb = _median_bin_spectrum(wave, flux, err, bin_size=bin_size)
    native_good = _display_quality_mask(flux, err=err, scale=scale)
    bin_good = _display_quality_mask(fb, err=eb, scale=scale)
    ax.plot(wave[native_good], flux[native_good] / scale,
            color='0.70', lw=0.35, alpha=0.50,
            label='native pixels (display-quality)')
    ax.plot(wb[bin_good], fb[bin_good] / scale, color='black', lw=1.35,
            label=f'{bin_size}-pixel median')
    if eb is not None and len(eb) == len(wb):
        ax.fill_between(wb[bin_good],
                        (fb[bin_good] - eb[bin_good]) / scale,
                        (fb[bin_good] + eb[bin_good]) / scale,
                        color='0.45', alpha=0.14, lw=0,
                        label='median error')
    if robust_ylim:
        ylim = _robust_spectrum_ylim(
            fb if len(fb) else flux,
            err=eb if eb is not None else err,
            scale=scale)
        if ylim:
            ax.set_ylim(*ylim)
    if log_flux:
        linthresh = 0.8
        finite_err = np.asarray(eb if eb is not None else err, dtype=float) / scale
        finite_err = finite_err[np.isfinite(finite_err) & (finite_err > 0)]
        if len(finite_err):
            linthresh = float(np.clip(np.nanmedian(finite_err), 0.2, 8.0))
        ax.set_yscale('symlog', linthresh=linthresh, linscale=0.7)

    if annotate_lines:
        _annotate_uv_lines(ax, wave, result.get('diagnostics'), scale=scale)

    prov = result.get('provenance', {})
    title_lines = [f"HST {result['instrument']} Spectrum"]
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
    if result.get('spectral_region'):
        obs_meta.append(f"region={result.get('spectral_region')}")
    if result.get('diagnostics', {}).get('median_snr') is not None:
        obs_meta.append(f"S/N~{result['diagnostics'].get('median_snr'):.1f}")
    title_lines.append('  '.join(obs_meta))
    ax.set_title('\n'.join(title_lines), fontsize=11)
    ax.set_xlabel('Observed wavelength (A)')
    ax.set_ylabel(r'$F_\lambda$  ($10^{-15}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)')
    ax.set_xlim(float(np.nanmin(wave)), float(np.nanmax(wave)))
    ax.grid(True, which='both', alpha=0.22)
    ax.legend(loc='upper right', fontsize=9, frameon=True)

    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def plot_lightcurve(result, save_path=None):
    """绘制 HST 多滤光片光变曲线"""
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
            # 仅有观测时间, 无星等
            for mjd in df['mjd']:
                ax.axvline(mjd, color=color, alpha=0.3, lw=0.5)
        ax.set_ylabel(filt_name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.text(0.98, 0.95, f'N={len(df)}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='gray')

    axes[-1].set_xlabel('MJD')
    fig.suptitle(f"HST Light Curve  RA={result['ra']:.4f} DEC={result['dec']:.4f}  "
                 f"{n} filters, {result.get('n_epochs', 0)} epochs",
                 fontsize=12)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_spectrum_csv(result, output_dir):
    """保存 HST 光谱为 CSV (附带 provenance 列 + 边车 JSON)。"""
    df = utils.spectrum_to_dataframe(result)
    prov = (result or {}).get('provenance')
    df = utils.add_provenance_columns(df, prov)
    csv_path = utils.write_csv(df, output_dir, 'hst_spectrum.csv')
    if prov:
        utils.write_provenance_json(prov, output_dir, 'hst_spectrum_provenance.json')
    diag = (result or {}).get('diagnostics')
    if diag:
        diag_path = os.path.join(output_dir, 'hst_spectrum_diagnostics.json')
        with open(diag_path, 'w', encoding='utf-8') as fh:
            json.dump(_jsonable(diag), fh, indent=2, ensure_ascii=False)
        line_rows = []
        for name, meas in diag.get('line_measurements', {}).items():
            row = {'line': name}
            row.update(meas)
            row['is_strong_emission'] = name in diag.get('strong_emission_lines', [])
            row['is_strong_absorption'] = name in diag.get('strong_absorption_lines', [])
            line_rows.append(row)
        if line_rows:
            utils.write_csv(pd.DataFrame(line_rows), output_dir,
                            'hst_line_measurements.csv')
    return csv_path


def save_lightcurve_csv(result, output_dir):
    """保存 HST 光变曲线为 CSV (附带 provenance 列 + 边车 JSON)。"""
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
    # 顶层信息只在 per-visit 没有覆盖时补充 (避免覆盖每行的 proposal_id)
    fill_cols = ['mission']
    if 'proposal_id' not in combined.columns and prov and prov.get('proposal_id'):
        fill_cols.append('proposal_id')
    combined = utils.add_provenance_columns(combined, prov, columns=fill_cols)
    csv_path = utils.write_csv(combined, output_dir, 'hst_lightcurve.csv')
    if prov:
        utils.write_provenance_json(prov, output_dir, 'hst_lightcurve_provenance.json')
    return csv_path
