"""
2MASS 近红外测光
================
J (1.25um), H (1.65um), Ks (2.17um)
通过 Vizier 查询 2MASS Point Source Catalog (II/246)

用法:
    from astro_toolbox.twomass import get_photometry
    phot = get_photometry(190.305, 2.596)
"""
import numpy as np
from . import config, utils


def get_photometry(ra, dec, radius_arcsec=config.SEARCH_RADIUS_ARCSEC):
    """
    查询 2MASS JHKs 测光。

    Returns:
        dict: {band_name: (mag, mag_err, wave_A)}
    """
    tbl = utils.query_vizier('II/246/out', ra, dec, radius_arcsec,
                             columns=['Jmag', 'e_Jmag', 'Hmag', 'e_Hmag',
                                      'Kmag', 'e_Kmag', 'Qflg'])
    if tbl is None:
        # 尝试更大搜索半径
        if radius_arcsec < 10:
            tbl = utils.query_vizier('II/246/out', ra, dec, 10.0,
                                     columns=['Jmag', 'e_Jmag', 'Hmag', 'e_Hmag',
                                              'Kmag', 'e_Kmag', 'Qflg'])
    if tbl is None:
        return {}

    row = tbl[0]
    bands = {
        '2MASS_J':  ('Jmag', 'e_Jmag'),
        '2MASS_H':  ('Hmag', 'e_Hmag'),
        '2MASS_Ks': ('Kmag', 'e_Kmag'),
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
            # 如果误差被 masked, 用 0.05 mag 默认误差
            try:
                mag = float(row[mag_col])
                if 0 < mag < 30:
                    wave = config.BAND_INFO[band_name]['wave_A']
                    phot[band_name] = (mag, 0.05, wave)
            except (ValueError, KeyError, np.ma.MaskError):
                continue
    return phot


def save_csv(result, output_dir):
    """保存 2MASS 测光数据为 CSV"""
    df = utils.photometry_to_dataframe(result)
    return utils.write_csv(df, output_dir, 'twomass_photometry.csv')
