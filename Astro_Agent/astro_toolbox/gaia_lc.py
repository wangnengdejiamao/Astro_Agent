"""
Gaia DR3 epoch photometry (光变曲线)
=====================================
G / BP / RP 三波段多历元测光

Gaia epoch photometry 不在标准 TAP 表中，需通过 DataLink 接口获取。
参考: https://gea.esac.esa.int/archive/documentation/GDR3/

用法:
    from astro_toolbox.gaia_lc import query_lightcurve
    lc = query_lightcurve(190.305, 2.596)
"""
import numpy as np
from . import config, utils


def _get_source_id(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """通过坐标获取 Gaia source_id"""
    tbl = utils.query_vizier('I/355/gaiadr3', ra, dec, radius_arcsec,
                             columns=['Source'])
    if tbl is None:
        return None
    return int(tbl[0]['Source'])


def query_lightcurve(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 Gaia DR3 epoch photometry。
    通过 Gaia DataLink 接口获取 (epoch_photometry 不是标准 TAP 表)。

    Returns:
        dict: {'G': DataFrame, 'BP': DataFrame, 'RP': DataFrame}
        或 None
    """
    source_id = _get_source_id(ra, dec, radius_arcsec)
    if source_id is None:
        return None

    # 方法1: 尝试 astroquery DataLink
    result = _query_via_datalink(source_id, ra, dec)
    if result is not None:
        return result

    # 方法2: 尝试 Vizier epoch photometry 表
    result = _query_via_vizier(source_id, ra, dec)
    if result is not None:
        return result

    return None


def _query_via_datalink(source_id, ra, dec):
    """通过 Gaia DataLink 获取 epoch photometry"""
    try:
        from astroquery.gaia import Gaia
        # 使用 token 认证 (可访问更多数据)
        if config.GAIA_TOKEN:
            try:
                Gaia.login(token=config.GAIA_TOKEN)
            except Exception:
                pass  # 未登录也可查询公开数据

        # DataLink 是获取 epoch photometry 的正确方式
        # Gaia archive 2025-12 升级后不再支持 votable_gzip,
        # 需要显式指定 format='votable' 或 'csv'
        datalink = None
        for fmt in ('votable', 'csv'):
            try:
                datalink = Gaia.load_data(
                    ids=[source_id],
                    data_release='Gaia DR3',
                    retrieval_type='EPOCH_PHOTOMETRY',
                    data_structure='INDIVIDUAL',
                    verbose=False,
                    format=fmt,
                )
                if datalink:
                    break
            except Exception:
                continue

        if not datalink:
            return None

        import pandas as pd
        result = {'survey': 'Gaia_DR3', 'ra': ra, 'dec': dec,
                  'source_id': source_id}

        # DataLink 返回一个 dict, key 是文件名字符串
        for key, table_list in datalink.items():
            # table_list 可能是: list[TableElement], 单个 Table, 或 DataFrame
            df = None
            if isinstance(table_list, list) and len(table_list) > 0:
                item = table_list[0]
                if hasattr(item, 'to_table'):
                    # astropy TableElement -> Table -> pandas
                    df = item.to_table().to_pandas()
                elif hasattr(item, 'to_pandas'):
                    df = item.to_pandas()
            elif hasattr(table_list, 'to_table'):
                df = table_list.to_table().to_pandas()
            elif hasattr(table_list, 'to_pandas'):
                df = table_list.to_pandas()

            if df is None or len(df) == 0:
                continue

            # Gaia DataLink epoch photometry 有两种列格式:
            # (A) 旧格式: band, time, mag, flux 等通用列, 每行一个观测
            # (B) 新格式: g_transit_time, g_transit_mag, bp_obs_time, bp_mag 等, 每行一个 transit

            # 检测格式
            if 'g_transit_time' in df.columns:
                # 新格式 (B): 按 transit 组织, 每行包含 G/BP/RP 数据
                band_map = {
                    'G':  {'time': 'g_transit_time', 'mag': 'g_transit_mag',
                           'flux': 'g_transit_flux', 'flux_error': 'g_transit_flux_error'},
                    'BP': {'time': 'bp_obs_time', 'mag': 'bp_mag',
                           'flux': 'bp_flux', 'flux_error': 'bp_flux_error'},
                    'RP': {'time': 'rp_obs_time', 'mag': 'rp_mag',
                           'flux': 'rp_flux', 'flux_error': 'rp_flux_error'},
                }
                for band, cols in band_map.items():
                    if cols['time'] not in df.columns:
                        continue
                    sub = df[[c for c in cols.values() if c in df.columns]].copy()
                    sub = sub.dropna(subset=[cols['time']])
                    if len(sub) == 0:
                        continue
                    band_df = pd.DataFrame({'time': sub[cols['time']]})
                    if cols['mag'] in sub.columns:
                        band_df['mag'] = sub[cols['mag']]
                    if cols['flux'] in sub.columns:
                        band_df['flux'] = sub[cols['flux']]
                    if cols['flux_error'] in sub.columns:
                        band_df['flux_error'] = sub[cols['flux_error']]
                    band_df = band_df.sort_values('time').reset_index(drop=True)
                    result[band] = band_df

                # 观测时间范围
                all_times = []
                for band in ('G', 'BP', 'RP'):
                    if band in result and 'time' in result[band].columns:
                        all_times.extend(result[band]['time'].dropna().tolist())
                if all_times:
                    result['obs_time_min'] = float(min(all_times))
                    result['obs_time_max'] = float(max(all_times))
                    result['time_system'] = 'Gaia_BJD'

                return result if any(b in result for b in ('G', 'BP', 'RP')) else None

            # 旧格式 (A): 通用列名
            band_col = None
            for col in ('band', 'BAND'):
                if col in df.columns:
                    band_col = col
                    break

            time_col = None
            for col in ('time', 'transit_time', 'TIME'):
                if col in df.columns:
                    time_col = col
                    break

            mag_col = None
            for col in ('mag', 'MAG'):
                if col in df.columns:
                    mag_col = col
                    break

            flux_col = None
            for col in ('flux', 'FLUX'):
                if col in df.columns:
                    flux_col = col
                    break

            flux_err_col = None
            for col in ('flux_error', 'FLUX_ERROR', 'flux_err'):
                if col in df.columns:
                    flux_err_col = col
                    break

            if band_col is None or time_col is None:
                continue

            for band in ('G', 'BP', 'RP'):
                sub = df[df[band_col] == band].copy()
                if len(sub) > 0:
                    sub = sub.sort_values(time_col).reset_index(drop=True)
                    cols_to_keep = [time_col]
                    rename_map = {time_col: 'time'}
                    if mag_col:
                        cols_to_keep.append(mag_col)
                        rename_map[mag_col] = 'mag'
                    if flux_col:
                        cols_to_keep.append(flux_col)
                        rename_map[flux_col] = 'flux'
                    if flux_err_col:
                        cols_to_keep.append(flux_err_col)
                        rename_map[flux_err_col] = 'flux_error'
                    result[band] = sub[cols_to_keep].rename(columns=rename_map)

            # 观测时间范围
            if time_col in df.columns:
                all_times = df[time_col].dropna()
                if len(all_times) > 0:
                    result['obs_time_min'] = float(all_times.min())
                    result['obs_time_max'] = float(all_times.max())
                    result['time_system'] = 'Gaia_BJD'

            return result if any(b in result for b in ('G', 'BP', 'RP')) else None

        return None

    except Exception as e:
        print(f"Gaia DataLink 查询失败: {e}")
        return None


def _query_via_vizier(source_id, ra, dec):
    """通过 Vizier 获取 Gaia epoch photometry (备用方案)"""
    try:
        # Gaia DR3 epoch photometry 在 Vizier: I/355/epphot
        tbl = utils.query_vizier('I/355/epphot', ra, dec,
                                 radius_arcsec=config.SEARCH_RADIUS_ARCSEC,
                                 columns=['Source', 'Band', 'TimeG', 'Gmag',
                                          'FG', 'e_FG'])
        if tbl is None:
            return None

        import pandas as pd
        df = tbl.to_pandas()
        # 只保留匹配的 source_id
        if 'Source' in df.columns:
            df = df[df['Source'] == source_id]
        if len(df) == 0:
            return None

        result = {'survey': 'Gaia_DR3', 'ra': ra, 'dec': dec,
                  'source_id': source_id}

        for band in ('G', 'BP', 'RP'):
            sub = df[df['Band'] == band].copy() if 'Band' in df.columns else pd.DataFrame()
            if len(sub) > 0:
                sub = sub.sort_values('TimeG').reset_index(drop=True)
                band_df = pd.DataFrame({'time': sub['TimeG']})
                if 'Gmag' in sub.columns:
                    band_df['mag'] = sub['Gmag']
                if 'FG' in sub.columns:
                    band_df['flux'] = sub['FG']
                if 'e_FG' in sub.columns:
                    band_df['flux_error'] = sub['e_FG']
                result[band] = band_df

        return result if any(b in result for b in ('G', 'BP', 'RP')) else None

    except Exception as e:
        # Vizier 可能没有 epoch photometry 表
        return None


def plot_lightcurve(result, save_path=None):
    """绘制 Gaia 三波段光变曲线"""
    if result is None:
        return None
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    colors = {'G': 'green', 'BP': 'blue', 'RP': 'red'}

    for ax, band in zip(axes, ('G', 'BP', 'RP')):
        if band in result:
            df = result[band]
            if 'mag' in df.columns:
                ax.scatter(df['time'], df['mag'], s=3, c=colors[band], alpha=0.7)
                ax.set_ylabel(f'{band} (mag)')
                ax.invert_yaxis()
            elif 'flux' in df.columns:
                ax.scatter(df['time'], df['flux'], s=3, c=colors[band], alpha=0.7)
                ax.set_ylabel(f'{band} (flux)')
        ax.set_title(f'Gaia {band}')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (Gaia BJD)')
    fig.suptitle(f"Gaia DR3 Epoch Photometry  source_id={result['source_id']}",
                 fontsize=13)
    fig.tight_layout()
    utils.save_and_close(fig, save_path)
    return fig


def save_csv(result, output_dir):
    """保存 Gaia 光变曲线为 CSV"""
    df = utils.lightcurve_to_dataframe(result, ['G', 'BP', 'RP'])
    return utils.write_csv(df, output_dir, 'gaia_lightcurve.csv')
