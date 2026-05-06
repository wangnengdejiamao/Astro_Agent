"""
GALEX 紫外测光
===============
FUV (1528 A) 和 NUV (2271 A)
通过 Vizier 查询 GALEX AIS GR6+7 (II/335)

用法:
    from astro_toolbox.galex import get_photometry
    phot = get_photometry(190.305, 2.596)
"""
import numpy as np
from . import config, utils


def get_photometry(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 GALEX FUV/NUV 测光。

    Returns:
        dict: {band_name: (mag, mag_err, wave_A)}
    """
    tbl = utils.query_vizier('II/335/galex_ais', ra, dec, radius_arcsec,
                             columns=['FUVmag', 'e_FUVmag', 'NUVmag', 'e_NUVmag'])
    if tbl is None:
        return {}

    row = tbl[0]
    bands = {
        'GALEX_FUV': ('FUVmag', 'e_FUVmag'),
        'GALEX_NUV': ('NUVmag', 'e_NUVmag'),
    }
    phot = {}
    for band_name, (mag_col, err_col) in bands.items():
        try:
            mag = float(row[mag_col])
            err = float(row[err_col])
            if 0 < mag < 30 and err > 0:
                wave = config.BAND_INFO[band_name]['wave_A']
                phot[band_name] = (mag, err, wave)
        except (ValueError, KeyError, np.ma.MaskError):
            continue
    return phot


def save_csv(result, output_dir):
    """保存 GALEX 测光数据为 CSV"""
    df = utils.photometry_to_dataframe(result)
    return utils.write_csv(df, output_dir, 'galex_photometry.csv')
