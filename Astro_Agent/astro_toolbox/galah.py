"""
GALAH DR4 光谱查询
===================
波长覆盖: 4 CCD (4713-4903, 5648-5873, 6478-6737, 7585-7887 A)
R ~ 28000 (高分辨率)

用法:
    from astro_toolbox.galah import query_spectrum
    result = query_spectrum(190.305, 2.596)
"""
import numpy as np
from . import config, utils


def query_spectrum(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 GALAH DR4 源信息 (通过 Vizier)。

    GALAH 光谱下载需要通过 Data Central (datacentral.org.au)，
    这里返回目录信息用于后续手动下载或交叉匹配。

    Returns:
        dict 或 None
    """
    tbl = utils.query_vizier('J/MNRAS/528/3232', ra, dec, radius_arcsec,
                             columns=['sobject_id', 'star_id', 'teff', 'e_teff',
                                      'logg', 'e_logg', 'fe_h', 'e_fe_h',
                                      'alpha_fe', 'vbroad', 'rv_galah',
                                      'snr_c1_iraf', 'snr_c2_iraf',
                                      'snr_c3_iraf', 'snr_c4_iraf',
                                      'flag_sp', 'flag_fe_h',
                                      'mjd_obs'])
    if tbl is None:
        # 尝试 DR3 (III/283)
        tbl = utils.query_vizier('III/283/galah', ra, dec, radius_arcsec)
        if tbl is None:
            return None

    row = tbl[0]
    result = {
        'survey': 'GALAH_DR4',
        'ra': ra, 'dec': dec,
    }
    for col in row.colnames:
        try:
            val = row[col]
            if np.ma.is_masked(val):
                continue
            result[col] = float(val) if isinstance(val, (int, float, np.number)) else str(val)
        except (ValueError, TypeError):
            result[col] = str(val)

    # 提取观测时间
    if 'mjd_obs' in result:
        result['obs_mjd'] = result['mjd_obs']

    # 构建 Data Central 下载链接
    if 'sobject_id' in result:
        result['download_url'] = (
            f"https://datacentral.org.au/services/sov/"
            f"?sobject_id={result['sobject_id']}")

    return result


def save_csv(result, output_dir):
    """保存 GALAH 参数为 CSV"""
    df = utils.keyvalue_to_dataframe(result)
    return utils.write_csv(df, output_dir, 'galah_params.csv')
